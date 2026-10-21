#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash routed-expert iVQ: one MoE layer -> one DQVL v3 blob that drops into
the YoungAi `vq8sh14-q4k-mtpnative` base GGUF in place of the weights-only blob.

The container, K and bit layout are the base's (one E4M3 codebook per layer shared by all
384 experts and w1/w3/w2, K=8192 on layers 0-13 and 4096 on 14-39, one F16 gain per row);
only how the codebook, the indices and the gains are chosen changes:

- the 384 experts x w1/w3/w2 (FP4 e2m1 x E8M0 per 32, exact in FP32) stay on the device;
- a stratified training pool (TRAIN_ROWS per expert and matrix) is weighted by the
  calibration statistics (axk2_weighting, `--mode imatrix`: g_r^2 x the expert's column
  importance, tempered by `--col-power`, rare experts shrunk, columns floored);
- one shared codebook: equidistant init, weighted FP64 Lloyd (axk2_vq_gpu), F32 -> F16 ->
  E4M3FN with YoungAi's rounding (ds41_vq_format.codebook_to_e4m3_like_youngai);
- every vector is assigned against the stored codebook under the expert's column metric
  and each row gain is refit by importance-weighted least squares (`--gain-refit ls`);
- the blob is packed by ds41_vq_format.pack_vq_layer and decoded back for a check.

Weight-space errors of the new blob and of the published one are recorded side by side,
unweighted and under the calibration importance. The receipt records the statistics files'
SHA-256 and the calibration run they come from (reduce_stats.calibration_run). A finished
layer (blob-LL.bin + receipt-LL.json) is reused only if the blob matches its receipt and the
receipt was made for the requested variant and statistics; otherwise the converter raises.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from axk2_vq_gpu import GpuVq
from axk2_weighting import Weighting, load_stats
from convert_layer import stats_fingerprint
import ds41_vq_format as fmt
from ds41_ref import Source, fp4_dequant
from ds41_vq_student import decode_blob
from reduce_stats import calibration_run

ROOT = '/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'
PROJ = {'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}       # ds41 name -> weighting name
SHAPES = {'w1': (2304, 5120), 'w3': (2304, 5120), 'w2': (5120, 2304)}
TRAIN_ROWS = {'w1': 32, 'w3': 32, 'w2': 71}                          # ~20.5K vectors each per expert
HELD_ROWS = {'w1': 2, 'w3': 2, 'w2': 4}
SEED = 41017
TOOLS = Path(__file__).resolve().parent


def emit(v):
    print(json.dumps(v), flush=True)


def source_hashes():
    files = [Path(__file__), TOOLS / 'axk2_vq_gpu.py', TOOLS / 'axk2_weighting.py', TOOLS / 'cuda/axk2_vq_convert.cu',
             TOOLS / 'ds41_vq_format.py', TOOLS / 'ds41_ref.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class LayerExperts:
    """Packed FP4 experts of one layer on the device; FP32 dequant on demand."""

    def __init__(self, src, L, dev='cuda'):
        names = [f'layers.{L}.ffn.experts.{e}.{p}.{k}' for e in range(384) for p in SHAPES for k in ('weight', 'scale')]
        t0 = time.monotonic()
        host = src.read(names)
        self.t = {k: v.to(dev) for k, v in host.items()}
        self.L = L
        self.read_seconds = time.monotonic() - t0

    def get(self, e, p, rows=None):
        w = self.t[f'layers.{self.L}.ffn.experts.{e}.{p}.weight']
        s = self.t[f'layers.{self.L}.ffn.experts.{e}.{p}.scale']
        if rows is not None:
            w, s = w[rows], s[rows]
        return fp4_dequant(w, s, out=torch.float32).contiguous()


def build_pool(gpu, lx, wt, L):
    xs, uws, aws, types, held = [], [], [], [], []
    for e in range(384):
        for pi, p in enumerate(SHAPES):
            rows, cols = SHAPES[p]
            ntr, nte = TRAIN_ROWS[p], HELD_ROWS[p]
            gen = torch.Generator().manual_seed(SEED + L * 1_000_000 + e * 3 + pi)
            chosen = torch.randperm(rows, generator=gen)[:ntr + nte].cuda()
            w = lx.get(e, p, chosen)
            norm, gains = gpu.normalize(w)
            norm = norm.reshape(ntr + nte, cols)
            rf = wt.row_factor(e, PROJ[p], gains[:ntr], chosen[:ntr])
            col = wt.col(e, PROJ[p])
            colv = col if col is not None else torch.ones(cols, device='cuda')
            uw = (rf[:, None] * wt.tensor_scale(e, PROJ[p]) * colv.double()[None, :]).float()
            xs.append(norm[:ntr].reshape(-1, 8))
            uws.append(uw.reshape(-1, 8))
            if wt.weighted_assignment:
                aws.append(colv[None, :].expand(ntr, cols).reshape(-1, 8))
            types.append(torch.full((ntr * cols // 8,), pi, dtype=torch.int8, device='cuda'))
            held.append((e, p, w[ntr:].clone(), col))
    pool = torch.cat(xs).contiguous()
    uw = torch.cat(uws).contiguous()
    aw = torch.cat(aws).contiguous() if wt.weighted_assignment else None
    typ = torch.cat(types)
    totals = {}
    if wt.mode == 'none':
        uw = None
    elif wt.mode in ('row', 'imatrix', 'imatrix_freq'):
        sums = [float(uw[typ == pi].double().sum()) for pi in range(3)]
        target = sum(sums) / 3
        for pi in range(3):
            if sums[pi] > 0:
                uw[typ == pi] *= target / sums[pi]
        totals['type_totals_before_equalize'] = sums
    if uw is not None:
        uw /= uw.double().mean().float()
    return pool, uw, aw, held, totals


class Meter:
    """Weight-space error sums: unweighted, and under the raw calibration importance
    (count-normalized E[x^2] per gate/up column, E[a^2] per down column, times the expert's
    routed token count, so frequently routed experts count more)."""

    def __init__(self, stats):
        self.s = stats
        self.v = {k: 0.0 for k in ('sse', 'energy', 'dot', 'rec', 'isse', 'ienergy')}

    def imp(self, e, p):
        c = float(self.s['count'][e])
        if c <= 0:
            return None
        key = 'a2' if p == 'w2' else 'x2'
        return self.s[key][e].to('cuda').double() / c * c                 # = sum over routed tokens

    def add(self, e, p, w, dec):
        w64, d64 = w.double(), dec.double()
        diff2 = (d64 - w64) ** 2
        self.v['sse'] += float(diff2.sum())
        self.v['energy'] += float((w64 ** 2).sum())
        self.v['dot'] += float((w64 * d64).sum())
        self.v['rec'] += float((d64 ** 2).sum())
        h = self.imp(e, p)
        if h is not None:
            self.v['isse'] += float((diff2 * h[None, :]).sum())
            self.v['ienergy'] += float((w64 ** 2 * h[None, :]).sum())

    def summary(self):
        v = self.v
        return {'rel_mse': v['sse'] / v['energy'], 'cos': v['dot'] / (v['energy'] * v['rec']) ** 0.5,
                'importance_rel_err': v['isse'] / v['ienergy'] if v['ienergy'] else None}


def statistics(stats_dir, L, mode):
    """-> (statistics or None, their files, statistics_sha256, (calibration run id, calibration set SHA-256))."""
    stats, files = load_stats(stats_dir, L) if mode not in ('none', 'row') else (None, [])
    return stats, files, stats_fingerprint(files), calibration_run(stats_dir, files)


def reuse(L, out_dir, want):
    """True when blob-LL.bin and receipt-LL.json exist and the blob is the receipt's and the receipt was made for
    `want` (the variant, statistics_sha256 and calibration_run_id); raises listing any difference. A receipt that
    records no statistics hash or calibration run (written before them) is reported as unverified for it."""
    path, rpath = out_dir / f'blob-{L:02d}.bin', out_dir / f'receipt-{L:02d}.json'
    if not (path.exists() and rpath.exists()):
        return False
    r = json.loads(rpath.read_text())
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    hist = (r.get('training') or {}).get('history')
    w = r.get('weighting') or {}
    got = {'sha256': r.get('sha256'), 'tensor': r.get('tensor'), 'mode': r.get('mode'), 'gain_refit': r.get('gain_refit'),
           'lloyd_iterations': len(hist) if isinstance(hist, list) else None, 'codebook_scope': r.get('codebook_scope'),
           **{k: w.get(k) for k in ('col_power', 'shrink_tokens', 'col_floor')},
           **{k: r[k] for k in ('statistics_sha256', 'calibration_run_id') if r.get(k) is not None}}
    need = {'sha256': checksum, 'tensor': f'blk.{L}.ffn_exps_vq.blob', 'codebook_scope': 'shared', **want}
    diff = {k: (got[k], v) for k, v in need.items() if k in got and got[k] != v}
    unverified = [k for k in need if k not in got]
    if diff:
        raise ValueError(f'existing layer {L} ({path.name}) was made for another request: {diff}')
    emit({'layer': L, 'phase': 'reuse-verified', 'sha256': checksum,
          **({'unverified': unverified} if unverified else {})})
    return True


def convert(L, out_dir, st, mode, refit, iterations, col_power, shrink, col_floor, base, compare):
    gpu = GpuVq()
    src = Source(ROOT)
    K = 8192 if L < 14 else 4096
    t0 = time.monotonic()
    stats, stat_files, stats_sha, (cal_run, cal_set) = st
    wt = Weighting(mode, stats, shrink_tokens=shrink, col_floor=col_floor, col_power=col_power)
    lx = LayerExperts(src, L)
    emit({'layer': L, 'phase': 'read', 'seconds': round(lx.read_seconds, 1), 'K': K})

    pool, uw, aw, held, totals = build_pool(gpu, lx, wt, L)
    emit({'layer': L, 'phase': 'pool', 'vectors': len(pool), 'per_codeword': len(pool) / K})
    hist_log = []
    e4m3, f16, f32, history = gpu.train(pool, K, iterations, weights=uw, assign_weights=aw, log=hist_log.append)
    del pool, uw, aw
    cb_bytes = fmt.codebook_to_e4m3_like_youngai(f32.cpu().numpy())        # [K, 8] uint8
    cbf = torch.from_numpy(fmt.e4m3_decode(cb_bytes)).cuda().contiguous()  # the stored values
    emit({'layer': L, 'phase': 'lloyd', 'seconds': round(sum(h['seconds'] for h in history), 1),
          'distortion_first': history[0]['weighted_distortion'], 'distortion_last': history[-1]['weighted_distortion'],
          'empty_centroids': history[-1]['empty_centroids']})

    old = None
    if compare:
        g = fmt.open_base(base)
        t = g.tensor(f'blk.{L}.ffn_exps_vq.blob')
        old = decode_blob(torch.from_numpy(g.vf.pread(t.abs_offset, t.nbytes)).cuda())   # (cb, idx, gains)
    m_new, m_old = Meter(stats), (Meter(stats) if old is not None else None)
    gains = {p: np.empty((384, SHAPES[p][0]), np.float16) for p in SHAPES}
    idx = {p: np.empty((384, SHAPES[p][0], SHAPES[p][1] // 8), np.int16) for p in SHAPES}
    t1 = time.monotonic()
    for e in range(384):
        for p in SHAPES:
            rows, cols = SHAPES[p]
            w = lx.get(e, p)
            col = wt.col(e, PROJ[p])
            norm, g = gpu.normalize(w)
            ids = gpu.assign(norm, cbf, colw=col)
            if refit == 'ls':
                gpu.refit_gain(w, ids, cbf, g, colw=col)
                norm = gpu.normalize_gain(w, g)
                ids = gpu.assign(norm, cbf, colw=col)
                gpu.refit_gain(w, ids, cbf, g, colw=col)
            dec = gpu.restore_ids(ids, g, cbf, cols)
            m_new.add(e, p, w, dec)
            if old is not None:
                od = old[0][old[1][p][e].long()].flatten(1) * old[2][p][e][:, None]
                m_old.add(e, p, w, od)
            idx[p][e] = ids.view(rows, cols // 8).to(torch.int16).cpu().numpy()
            gains[p][e] = g.cpu().numpy()
        if e % 64 == 63:
            emit({'layer': L, 'phase': 'assign', 'experts': e + 1, 'seconds': round(time.monotonic() - t1, 1),
                  **{k: round(v, 5) for k, v in m_new.summary().items() if v is not None}})
    blob = fmt.pack_vq_layer(L=L, cb_bytes=cb_bytes, gains=gains, idx=idx)
    # decode check of a few payloads
    chk = fmt.decode_vq_blob(blob, f'blk.{L}.ffn_exps_vq.blob')
    for e in (0, 191, 383):
        for p in SHAPES:
            assert np.array_equal(chk.idx[p][e], idx[p][e].astype(np.int32)), (L, e, p)
            assert np.array_equal(chk.gains[p][e].view(np.uint16), gains[p][e].view(np.uint16)), (L, e, p)
    assert np.array_equal(chk.cb_bytes, cb_bytes)
    name = f'blk.{L}.ffn_exps_vq.blob'
    if base is not None:
        g = fmt.open_base(base)
        assert blob.size == g.tensor(name).nbytes, (blob.size, g.tensor(name).nbytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'blob-{L:02d}.bin'
    tmp = path.with_suffix('.tmp')
    blob.tofile(tmp)
    tmp.rename(path)
    receipt = {
        'layer': L, 'tensor': name, 'K': K, 'bits': 13 if K == 8192 else 12, 'bytes': int(blob.size),
        'sha256': hashlib.sha256(blob.tobytes()).hexdigest(),
        'mode': mode, 'gain_refit': refit, 'lloyd_iterations': iterations, 'codebook_scope': 'shared',
        'weighting': wt.describe(), 'statistics': stat_files, 'statistics_sha256': stats_sha,
        'calibration_run_id': cal_run, 'calibration_set_sha256': cal_set,
        'pool': {'train_rows': TRAIN_ROWS, 'heldout_rows': HELD_ROWS, 'seed': SEED, **totals},
        'training': {'init': 'equidistant pool[floor(i*N/K)]', 'accumulation': 'FP64 atomics, weighted',
                     'codebook_rounding': 'F32 -> F16 (RNE) -> E4M3FN (YoungAi v41_vq_rate)',
                     'history': history},
        'weight_error': {'ivq': m_new.summary(), 'published_vq': m_old.summary() if m_old else None},
        'seconds': time.monotonic() - t0, 'tools': source_hashes(),
    }
    rtmp = out_dir / f'receipt-{L:02d}.json.partial'
    rtmp.write_text(json.dumps(receipt, indent=1) + '\n')
    rtmp.rename(out_dir / f'receipt-{L:02d}.json')
    emit({'layer': L, 'phase': 'done', 'seconds': round(time.monotonic() - t0, 1),
          'ivq': m_new.summary(), 'published': m_old.summary() if m_old else None})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers', required=True, help='comma list or a-b range')
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', default='imatrix')
    p.add_argument('--gain-refit', default='ls', choices=['none', 'ls'])
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--col-power', type=float, default=0.75)
    p.add_argument('--shrink-tokens', type=float, default=256.0)
    p.add_argument('--col-floor', type=float, default=0.02)
    p.add_argument('--base', default=None, help='base GGUF (default: the published parts)')
    p.add_argument('--no-compare', action='store_true')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if '-' in a.layers:
        lo, hi = a.layers.split('-')
        layers = list(range(int(lo), int(hi) + 1))
    else:
        layers = [int(x) for x in a.layers.split(',')]
    want = {'mode': a.mode, 'gain_refit': a.gain_refit, 'lloyd_iterations': a.iterations, 'col_power': a.col_power,
            'shrink_tokens': a.shrink_tokens, 'col_floor': a.col_floor}
    for L in layers:
        st = statistics(a.stats, L, a.mode)
        if reuse(L, a.out, {**want, 'statistics_sha256': st[2], 'calibration_run_id': st[3][0]}):
            continue
        convert(L, a.out, st, a.mode, a.gain_refit, a.iterations, a.col_power, a.shrink_tokens, a.col_floor,
                a.base, not a.no_compare)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
