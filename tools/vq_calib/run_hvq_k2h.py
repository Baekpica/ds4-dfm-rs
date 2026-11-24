#!/usr/bin/env python3
"""K2-Horizon-375B-A23B routed-expert hVQ (13-bit GVQ1): calibration statistics and conversion, one MoE
layer at a time on N GPUs (torchrun, one rank per GPU). The driver of run_hvq_inkling2.py for K2-Horizon.

Per MoE layer l (3..60):
  1. every rank streams its share of the calibration sequences through layer l (k2h_ref, the original
     BF16 arithmetic; the residual stream stays on the device) and sums, per routed expert, the diagonal
     statistics (count, sum x^2, sum a^2, token-normalized and route-weighted variants) and the
     token-normalized second moments (sum x x^T / mean(x^2) over the gate/up input, sum a a^T / mean(a^2)
     over the down input);
  2. the diagonal statistics are all-reduced; the second moments are reduce-scattered, so rank r holds
     experts [r E/N, (r+1) E/N);
  3. rank 0 trains the layer's codebook (K = 8192, E4M3, shared by the 192 experts and gate/up/down) by
     importance-weighted Lloyd (axk2_weighting `imatrix`, column power 0.75: the iVQ stage) and
     broadcasts it;
  4. every rank assigns its experts' codes and F16 row gains by hVQ (ds41_hvq: 8-column blocks in order,
     error feedback through H_t = D^1/2 ((1-t) I + t R) D^1/2, least-squares gains under H_t), packs them
     as GVQ1 v1 13-bit payloads (12-bit pair stream + 1-bit plane), cross-checks sampled payloads with the
     independent numpy encoder and decoder, and writes them; the moments are dropped.

Differences from the Inkling-2 driver, both forced by the host (RTX PRO 6000 Blackwell: FP64 GEMMs run at
1.5 TFLOPS against 73 for FP32): the chunk Gram matrices are FP32 GEMMs accumulated in FP32 (the DS41
probe measured all-FP32 correlations within 2.0e-6 of FP64), and the off-diagonal products of the gain
refit are FP32 GEMMs with FP64 row sums. Rank 0 also carries held-out sequences forward without
statistics (the original's final states and routes, for the end-to-end check of the release bytes).

    torchrun --nproc-per-node 3 run_hvq_k2h.py --source $SRC --calib tokens.npz --out $W/convert --stats $W/stats
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file

import k2h_ref as R
import k2h_vq_format as fmt

TOOLS = Path(__file__).resolve().parent
TRAIN_ROWS = {'gate_proj': 35, 'up_proj': 35, 'down_proj': 120}     # 26,880 vectors per projection type
HELD_ROWS = {'gate_proj': 2, 'up_proj': 2, 'down_proj': 7}
SEED = 20261124
GROUPS = {'x': ('gate_proj', 'up_proj'), 'a': ('down_proj',)}
BATCH = {'x': 8, 'a': 16}
SHAPES = fmt.SHAPES
CKPT_LAYERS = (14, 26, 38, 50)


def emit(rank, value):
    if rank == 0 or value.get('all_ranks'):
        print(json.dumps({'rank': rank, 't': round(time.time(), 1), **value}), flush=True)


def source_hashes():
    files = [Path(__file__), TOOLS / 'k2h_ref.py', TOOLS / 'k2h_vq_format.py', TOOLS / 'ds41_hvq.py',
             TOOLS / 'cuda/ds41_hvq.cu', TOOLS / 'ds41_hess.py', TOOLS / 'axk2_vq_gpu.py',
             TOOLS / 'axk2_weighting.py', TOOLS / 'cuda/axk2_vq_convert.cu']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def parse_layers(spec):
    out = set()
    for part in (spec or '').split(','):
        if part:
            lo, _, hi = part.partition('-')
            out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


# --------------------------------------------------------------------------------- statistics
class Collector:
    """Per-expert diagonal statistics (FP64 sums of FP32 chunk sums) and the hVQ second moments (FP32 GEMMs,
    FP32 sums) of one layer, on one rank."""

    def __init__(self, dev):
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device=dev)
        self.d = {'count': torch.zeros(R.E, dtype=torch.int64, device=dev), 'w2': z(R.E),
                  'x2': z(R.E, R.H), 'a2': z(R.E, R.FF), 'x2n': z(R.E, R.H), 'a2n': z(R.E, R.FF),
                  'x2w': z(R.E, R.H), 'a2w': z(R.E, R.FF)}
        from ds41_hess import LayerHess
        self.hess = LayerHess(R.E, R.H, R.FF, dev)

    def expert(self, e, xe, act, we):
        d = self.d
        w2 = we.float().square()
        x = xe.float()
        x2 = x.square()
        mx = x2.mean(-1, keepdim=True).clamp_min(1e-30)
        a = act.float()
        a2 = a.square()
        ma = a2.mean(-1, keepdim=True).clamp_min(1e-30)
        d['count'][e] += xe.shape[0]
        d['w2'][e] += w2.double().sum()
        d['x2'][e] += x2.sum(0).double()
        d['a2'][e] += a2.sum(0).double()
        d['x2n'][e] += (x2 / mx).sum(0).double()
        d['a2n'][e] += (a2 / ma).sum(0).double()
        d['x2w'][e] += (x2 * w2[:, None]).sum(0).double()
        d['a2w'][e] += (a2 * w2[:, None]).sum(0).double()
        u = x * mx.rsqrt()
        self.hess.xx[e].addmm_(u.T, u)
        u = a * ma.rsqrt()
        self.hess.aa[e].addmm_(u.T, u)
        self.hess.count[e] += xe.shape[0]


# --------------------------------------------------------------------------------- codebook
def train_codebook(gpu, lw, wt, layer, k, iterations, log):
    """Pool of stratified rows of all experts (TRAIN_ROWS), RMS gains, imatrix weights (row factor g^2
    times the column profile, equal total per projection type, mean 1), weighted FP64 Lloyd -> E4M3 book.
    Also the held-out rows' plain and column-weighted relative error under the stored book."""
    xs, uws, aws, types, held = [], [], [], [], []
    for e in range(R.E):
        gu, dn = lw.expert(e)
        mats = {'gate_proj': gu[:R.FF], 'up_proj': gu[R.FF:], 'down_proj': dn}
        for pi, p in enumerate(fmt.PROJECTIONS):
            rows, cols = SHAPES[p]
            ntr, nte = TRAIN_ROWS[p], HELD_ROWS[p]
            g = torch.Generator().manual_seed(SEED + layer * 1_000_000 + e * 3 + pi)
            chosen = torch.randperm(rows, generator=g)[:ntr + nte]
            w = mats[p][chosen.to(lw.dev)].float().contiguous()
            norm, gains = gpu.normalize(w)
            norm = norm.reshape(ntr + nte, cols)
            col = wt.col(e, p)
            colv = col if col is not None else torch.ones(cols, device=lw.dev)
            rf = gains[:ntr].double() ** 2
            uw = (rf[:, None] * colv.double()[None, :]).float()
            xs.append(norm[:ntr].reshape(-1, 8))
            uws.append(uw.reshape(-1, 8))
            aws.append(colv[None, :].expand(ntr, cols).reshape(-1, 8))
            types.append(torch.full((ntr * cols // 8,), pi, dtype=torch.int8, device=lw.dev))
            held.append((p, w[ntr:].clone(), col))
        del gu, dn, mats
    pool = torch.cat(xs).contiguous()
    uw = torch.cat(uws).contiguous()
    aw = torch.cat(aws).contiguous()
    typ = torch.cat(types)
    del xs, uws, aws, types
    sums = [float(uw[typ == pi].double().sum()) for pi in range(3)]
    target = sum(sums) / 3
    for pi in range(3):
        if sums[pi] > 0:
            uw[typ == pi] *= target / sums[pi]
    uw /= uw.double().mean().float()
    t0 = time.monotonic()
    e8, f16, f32, hist = gpu.train(pool, k, iterations, weights=uw, assign_weights=aw)
    seconds = time.monotonic() - t0
    cb = e8.float().contiguous()
    sse = en = wsse = wen = 0.0
    for p, w, col in held:
        norm, g = gpu.normalize(w)
        ids = gpu.assign(norm, cb, colw=col)
        dec = gpu.restore_ids(ids, g, cb, w.shape[1])
        d2 = (dec.double() - w.double()) ** 2
        r2 = w.double() ** 2
        h = col.double() if col is not None else torch.ones(w.shape[1], dtype=torch.float64, device=w.device)
        sse, en = sse + float(d2.sum()), en + float(r2.sum())
        wsse, wen = wsse + float((d2 * h).sum()), wen + float((r2 * h).sum())
    info = {'k': k, 'iterations': len(hist), 'pool_vectors': len(pool), 'pool_per_codeword': len(pool) / k,
            'train_rows_per_expert': TRAIN_ROWS, 'heldout_rows_per_expert': HELD_ROWS,
            'type_totals_before_equalize': sums, 'lloyd_seconds': round(seconds, 1),
            'distortion_first_last': [hist[0]['weighted_distortion'], hist[-1]['weighted_distortion']],
            'empty_centroids_last': hist[-1]['empty_centroids'],
            'heldout_rel_mse': sse / en, 'heldout_col_weighted_rel_err': wsse / wen,
            'init': 'equidistant pool[floor(i*N/K)]', 'rounding': 'FP32 -> F16 -> E4M3FN'}
    log({'phase': 'codebook', 'layer': layer, **{k_: v for k_, v in info.items() if k_ != 'train_rows_per_expert'}})
    return e8, info


# ------------------------------------------------------------------------------------- hVQ
JITTER, JITTER_STEPS = 1e-8, 8
GAIN_REFIT = ('ls-h: g = w H_t c^T / c H_t c^T per row (F16), after each of two passes; the first pass uses RMS '
              'gains; off-diagonal products in FP32 GEMMs, row sums in FP64')


def make_hvq():
    import ds41_hvq as hv

    class Hvq32(hv.Hvq):
        def refit(self, w, ids, cb, h, gains, chunk=4):
            """Hvq.refit with the off-diagonal products as FP32 GEMMs (row sums in FP64)."""
            p, rows, cols = w.shape
            assert gains.dtype == torch.float16 and gains.is_contiguous() and ids.is_contiguous()
            hdiag = h.diagonal(dim1=-2, dim2=-1).float().contiguous()
            noff = torch.empty((p, rows), dtype=torch.float64, device=w.device)
            doff = torch.empty_like(noff)
            for s in range(0, p, chunk):
                code = cb[ids[s:s + chunk].long()].reshape(-1, rows, cols)
                off = h[s:s + chunk].float()
                off.diagonal(dim1=-2, dim2=-1).zero_()
                y = torch.bmm(code, off)
                noff[s:s + chunk] = (w[s:s + chunk] * y).double().sum(-1)
                doff[s:s + chunk] = (code * y).double().sum(-1)
                del code, off, y
            self.check(self.lib.hvq_refit(w.data_ptr(), ids.data_ptr(), cb.data_ptr(), hdiag.data_ptr(),
                                          noff.data_ptr(), doff.data_ptr(), gains.data_ptr(), p, rows, cols,
                                          self.stream()))
            return gains

    return hv, Hvq32()


def factor_safe(hv, h):
    """V with H = V V^T (ds41_hvq.factor); a matrix that fails gets its diagonal lifted in h.
    Returns (V, matrices lifted, largest lift x mean)."""
    v, info = hv.factor(h)
    bad = info != 0
    lifted, lift, step = int(bad.sum()), 0.0, 0
    while bad.any():
        if step == JITTER_STEPS:
            raise RuntimeError(f'{int(bad.sum())} matrices are not positive definite after a {lift:g} x mean lift')
        lift = JITTER * 10.0 ** step
        sub = h[bad]
        sub.diagonal(dim1=-2, dim2=-1).add_(lift * sub.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True))
        h[bad] = sub
        v2, info2 = hv.factor(sub)
        v[bad] = v2
        still = bad.clone()
        still[bad] = info2 != 0
        bad = still
        step += 1
    return v, lifted, lift


def pack13(ids):
    """int32 [rows, nv] -> (GVQ1 12-bit pair stream bytes [a & 255, (a >> 8 & 15) | (b & 15) << 4, b >> 4 & 255],
    1-bit plane bytes: bit 12 of index i is bit i % 8 of byte i // 8)."""
    a, b = ids[..., 0::2], ids[..., 1::2]
    low = torch.stack([a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255], -1).to(torch.uint8).reshape(-1)
    hi = ((ids >> 12) & 1).reshape(-1, 8)
    plane = (hi << torch.arange(8, device=ids.device, dtype=ids.dtype)).sum(-1).to(torch.uint8)
    return low, plane


def fit_shard(lw, wt, hess, cb, cb_bytes, k, layer, var, hv, hvq, experts, out_dir, rank, baseline_every, log):
    """hVQ codes and gains of this rank's experts -> GVQ1 payloads in one binary file + index + receipt."""
    e_lo = experts[0]
    bid = fmt.book_id(layer)
    path = out_dir / f'rank-{rank}.bin'
    tmp = path.with_suffix('.bin.tmp')
    index, errs = [], {g: {'hvq': 0.0, 'ivq': 0.0, 'energy': 0.0, 'energy_sampled': 0.0, 'plain_hvq': 0.0,
                           'plain_ivq': 0.0, 'plain_energy': 0.0, 'plain_energy_sampled': 0.0} for g in GROUPS}
    seconds = {'weights': 0.0, 'hessian': 0.0, 'factor': 0.0, 'assign': 0.0, 'baseline': 0.0, 'pack': 0.0}
    lifted, lift_max, checked = 0, 0.0, 0
    payloads = {}
    rng = np.random.default_rng(SEED + layer * 7 + rank)

    def tick(key, t0):
        torch.cuda.synchronize()
        seconds[key] += time.monotonic() - t0
        return time.monotonic()

    with tmp.open('wb') as f:
        for group, projs in GROUPS.items():
            cols = SHAPES[projs[0]][1]
            prior = hess.layer_mean_all if group == 'x' else None
            for b0 in range(0, len(experts), BATCH[group]):
                es = experts[b0:b0 + BATCH[group]]
                t0 = time.monotonic()
                ws = []
                for e in es:
                    gu, dn = lw.expert(e)
                    ws.append(gu.float() if group == 'x' else dn.float())
                w = torch.stack(ws).contiguous()
                del ws
                cw = [wt.col(e, projs[0]) for e in es]
                col = torch.stack([c if c is not None else torch.ones(cols, device=lw.dev) for c in cw])
                t0 = tick('weights', t0)
                sl = slice(es[0] - e_lo, es[-1] - e_lo + 1)
                r, _ = hv.correlation((hess.xx if group == 'x' else hess.aa)[sl], hess.count[sl], prior, var['hess_shrink'])
                h = hv.hessian(col, r, var['t'], var['damp'])
                del r
                t0 = tick('hessian', t0)
                v, n_lift, top = factor_safe(hv, h)
                lifted, lift_max = lifted + n_lift, max(lift_max, top)
                t0 = tick('factor', t0)
                ids, g = hvq.quantize(w, cb, h, v)
                del v
                t0 = tick('assign', t0)
                dec = hvq.restore(ids, g, cb)
                h32 = h.float()
                errs[group]['hvq'] += float(hv.output_error(w - dec, h32).sum())
                errs[group]['energy'] += float(hv.output_error(w, h32).sum())
                errs[group]['plain_hvq'] += float((w - dec).double().square().sum())
                errs[group]['plain_energy'] += float(w.double().square().sum())
                # iVQ baseline (t = 0, the same book) on every `baseline_every`-th expert, scored under H_t
                sel = [i for i, e in enumerate(es) if baseline_every and e % baseline_every == 0]
                if sel:
                    ws_, cs_ = w[sel].contiguous(), col[sel]
                    h0 = hv.hessian(cs_, torch.eye(cols, dtype=torch.float64, device=lw.dev).expand(len(sel), cols, cols), 0.0, var['damp'])
                    v0, _, _ = factor_safe(hv, h0)
                    i0, g0 = hvq.quantize(ws_, cb, h0, v0)
                    d0 = hvq.restore(i0, g0, cb)
                    errs[group]['ivq'] += float(hv.output_error(ws_ - d0, h32[sel]).sum())
                    errs[group]['energy_sampled'] += float(hv.output_error(ws_, h32[sel]).sum())
                    errs[group]['hvq_sampled'] = errs[group].get('hvq_sampled', 0.0) + float(hv.output_error(ws_ - dec[sel], h32[sel]).sum())
                    errs[group]['plain_ivq'] += float((ws_ - d0).double().square().sum())
                    errs[group]['plain_energy_sampled'] += float(ws_.double().square().sum())
                    del h0, v0, i0, g0, d0, ws_
                    t0 = tick('baseline', t0)
                del h, h32, dec
                row0 = 0
                for p in projs:
                    rows = SHAPES[p][0]
                    for i, e in enumerate(es):
                        ip, gp = ids[i, row0:row0 + rows].contiguous(), g[i, row0:row0 + rows].contiguous()
                        low, plane = pack13(ip) if k == 8192 else (pack13(ip)[0], None)
                        low = low.cpu().numpy().tobytes()
                        plane = plane.cpu().numpy().tobytes() if plane is not None else None
                        gb = gp.cpu().numpy().astype('<f2').tobytes()
                        payload = fmt.assemble_vq(gb, low, plane, rows, cols, k, cb_bytes, bid)
                        if rng.random() < 0.02:          # independent numpy encoder + decoder on sampled matrices
                            ref = fmt.encode_vq(np.frombuffer(gb, '<f2'), ip.cpu().numpy(), rows, cols, k, cb_bytes, bid)
                            if ref != payload:
                                raise RuntimeError(f'layer {layer} expert {e} {p}: device packing differs from encode_vq')
                            t = fmt.VqTensor(payload, cb_bytes, (rows, cols), bid)
                            rr = int(rng.integers(0, rows))
                            want = (cb[ip[rr].long()].reshape(-1) * gp[rr].float()).cpu().numpy()
                            if not np.array_equal(t.decode_rows(rr, 1)[0], want):
                                raise RuntimeError(f'layer {layer} expert {e} {p}: decode mismatch')
                            checked += 1
                        name = fmt.matrix_name(layer, e, p)
                        payloads[name] = (f.tell(), len(payload), hashlib.sha256(payload).hexdigest())
                        f.write(payload)
                    row0 += rows
                del ids, g, w, col
                t0 = tick('pack', t0)
    # canonical order: expert-major, gate/up/down
    for e in experts:
        for p in fmt.PROJECTIONS:
            name = fmt.matrix_name(layer, e, p)
            off, size, sha = payloads[name]
            index.append({'name': name, 'offset': off, 'payload_bytes': size, 'payload_sha256': sha})
    tmp.rename(path)
    with path.open('rb') as f:
        file_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    summary = {}
    for group, s in errs.items():
        summary[group] = {'hvq_output_rel': s['hvq'] / s['energy'], 'plain_rel_mse_hvq': s['plain_hvq'] / s['plain_energy'],
                          'sampled_experts_every': baseline_every}
        if s['energy_sampled'] > 0:
            summary[group].update({'sampled_hvq_output_rel': s['hvq_sampled'] / s['energy_sampled'],
                                   'sampled_ivq_output_rel': s['ivq'] / s['energy_sampled'],
                                   'sampled_plain_rel_mse_hvq_vs_ivq': [None, s['plain_ivq'] / s['plain_energy_sampled']]})
    return {'rank': rank, 'experts': [experts[0], experts[-1]], 'file': path.name, 'file_sha256': file_sha,
            'index': index, 'errors': summary, 'raw_error_sums': errs, 'seconds': {k_: round(v, 1) for k_, v in seconds.items()},
            'factorization': {'experts_lifted': lifted, 'largest_lift_x_mean': lift_max},
            'payloads_cross_checked': checked}


# --------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='IFM/K2-Horizon-375B-A23B at R.SOURCE_REV (BF16)')
    ap.add_argument('--calib', required=True, type=Path, help='npz with tokens [chunks, 512] (the MQ87 calibration mix)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stats', required=True, type=Path)
    ap.add_argument('--layers', default='3-60')
    ap.add_argument('--calib-chunks', type=int, default=7680, help='the first N chunks are the calibration set')
    ap.add_argument('--heldout-chunks', type=int, default=64, help='the last N chunks ride along on rank 0 without statistics')
    ap.add_argument('--batch-seqs', type=int, default=96)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--damp', type=float, default=0.0)
    ap.add_argument('--hess-shrink', type=float, default=4096.0)
    ap.add_argument('--col-power', type=float, default=0.75)
    ap.add_argument('--shrink-tokens', type=float, default=256.0)
    ap.add_argument('--col-floor', type=float, default=0.02)
    ap.add_argument('--k', type=int, default=8192)
    ap.add_argument('--iterations', type=int, default=32)
    ap.add_argument('--baseline-every', type=int, default=16)
    ap.add_argument('--stop-after', type=int, default=-1, help='stop after this layer (tests)')
    ap.add_argument('--no-ckpt', action='store_true')
    a = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(lrank)
    dev = torch.device('cuda', lrank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    log = lambda v: emit(rank, v)
    a.out.mkdir(parents=True, exist_ok=True)
    a.stats.mkdir(parents=True, exist_ok=True)
    convert = set(parse_layers(a.layers))
    var = {'t': a.t, 'damp': a.damp, 'hess_shrink': a.hess_shrink}
    assert R.E % world == 0
    shard = R.E // world
    my_experts = list(range(rank * shard, (rank + 1) * shard))

    data = np.load(a.calib)
    tokens = data['tokens']
    total, T = tokens.shape
    assert a.calib_chunks + a.heldout_chunks <= total
    calib_sha = hashlib.sha256(a.calib.read_bytes()).hexdigest()
    mine = np.arange(a.calib_chunks)[rank::world]
    held_idx = np.arange(total - a.heldout_chunks, total)
    run_id = hashlib.sha256(json.dumps({'calib': calib_sha, 'calib_chunks': a.calib_chunks, 'heldout_chunks': a.heldout_chunks,
                                        'world': world, 'source': R.SOURCE_REV}).encode()).hexdigest()
    log({'phase': 'start', 'world': world, 'sequences': a.calib_chunks, 'tokens': a.calib_chunks * T,
         'my_sequences': len(mine), 'my_tokens': len(mine) * T, 'calib_sha256': calib_sha, 'run_id': run_id,
         'convert_layers': sorted(convert), 'variant': var, 'k': a.k, 'all_ranks': True})

    src = R.Source(a.source)
    last = max(convert) if a.stop_after < 0 else a.stop_after
    ck_dir = a.out / 'ckpt'

    def converted(l):
        """Whether layer l has a receipt of this calibration and these settings; another one stops the run, so a
        reused --out never mixes conversions."""
        path = a.out / f'layer-{l:02d}' / 'receipt.json'
        if not path.exists():
            return False
        r = json.loads(path.read_text())
        want = {'calibration_run_id': run_id, 'k': a.k, 't': a.t, 'damping': a.damp, 'source_revision': R.SOURCE_REV,
                'hess_shrink': a.hess_shrink, 'col_power': a.col_power, 'shrink_tokens': a.shrink_tokens,
                'col_floor': a.col_floor}
        got = {**{k: r.get(k) for k in ('calibration_run_id', 'k', 't', 'damping', 'source_revision')},
               'hess_shrink': r.get('hessian', {}).get('shrink_tokens'),
               **{k: r.get('weighting', {}).get(k) for k in ('col_power', 'shrink_tokens', 'col_floor')}}
        diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        if diff:
            raise SystemExit(f'{path} is another conversion (receipt, this run): {diff}; use a fresh --out')
        return True

    # resume point: the latest checkpoint every rank has, with every MoE layer up to it converted
    start = 0
    for c in sorted(CKPT_LAYERS, reverse=True):
        d = ck_dir / f'layer-{c:02d}'
        if all((d / f'rank-{r}.done').exists() and (d / f'rank-{r}.done').read_text().strip() == run_id for r in range(world)) \
                and all(converted(l) for l in convert if l <= c):
            start = c + 1
            break
    held, held_routes = None, {}
    if start:
        d = ck_dir / f'layer-{start - 1:02d}'
        hid = torch.load(d / f'rank-{rank}.pt', map_location='cpu').to(dev)
        if rank == 0 and a.heldout_chunks:
            held = torch.load(d / 'heldout.pt', map_location='cpu').to(dev)
            # the layers before the checkpoint route only here: teacher.safetensors needs all 58
            held_routes = torch.load(d / 'heldout-routes.pt', map_location='cpu')
        log({'phase': 'resumed', 'from_layer': start, 'all_ranks': True})
    else:
        glob = R.Globals(src, dev, embed=True)
        hid = glob.embed[torch.as_tensor(tokens[mine].astype(np.int64), device=dev)]
        if rank == 0 and a.heldout_chunks:
            held = glob.embed[torch.as_tensor(tokens[held_idx].astype(np.int64), device=dev)]
        del glob
    torch.cuda.empty_cache()
    cos, sin = R.rope_cache(T, dev)
    stacks = R.new_stacks()
    pool = ThreadPoolExecutor(1)
    nxt_weights = pool.submit(src.read_layer, start, stacks)
    from axk2_vq_gpu import GpuVq
    from axk2_weighting import Weighting
    hv, hvq = make_hvq()
    gpu = GpuVq()
    ck_pool = ThreadPoolExecutor(1)
    t_run = time.monotonic()
    for l in range(start, last + 1):
        t_layer = time.monotonic()
        host = nxt_weights.result()
        t_read = time.monotonic() - t_layer
        lw = R.LayerWeights(l, host, dev)
        del host
        if l + 1 <= last:
            nxt_weights = pool.submit(src.read_layer, l + 1, stacks)
        layer_dir = a.out / f'layer-{l:02d}'
        done = converted(l)
        do = l in convert and not done
        col = Collector(dev) if do else None
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        for s0 in range(0, hid.shape[0], a.batch_seqs):
            R.layer_forward(lw, hid[s0:s0 + a.batch_seqs], cos, sin, hooks=col)
        if held is not None:
            sels = []
            for s0 in range(0, held.shape[0], a.batch_seqs):
                sels.append(R.layer_forward(lw, held[s0:s0 + a.batch_seqs], cos, sin))
            if lw.moe:
                held_routes[l] = torch.cat(sels).to(torch.uint8).cpu()
        torch.cuda.synchronize()
        fwd_s = time.monotonic() - t0
        peak_fwd = torch.cuda.max_memory_allocated() / 2**30
        finite = torch.tensor([int(torch.isfinite(hid[:: max(1, hid.shape[0] // 64)]).all())], device=dev)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite):
            raise FloatingPointError(f'layer {l}: non-finite hidden state')
        if not do:
            log({'phase': 'forward-only', 'layer': l, 'seconds': round(fwd_s, 1), 'read': round(t_read, 1),
                 'reason': 'done' if done else ('dense' if not lw.moe else 'not selected'),
                 'elapsed_min': round((time.monotonic() - t_run) / 60, 1)})
        else:
            # ---------------------------------------------------------------- reduce
            t0 = time.monotonic()
            for k_, v in col.d.items():
                dist.all_reduce(v)
            hess = col.hess
            cnt = hess.count.clone()
            dist.all_reduce(cnt)
            xs = torch.empty((shard, R.H, R.H), dtype=torch.float32, device=dev)
            dist.reduce_scatter_tensor(xs, hess.xx)
            hess.xx = None
            torch.cuda.empty_cache()
            as_ = torch.empty((shard, R.FF, R.FF), dtype=torch.float32, device=dev)
            dist.reduce_scatter_tensor(as_, hess.aa)
            hess.aa = None
            torch.cuda.empty_cache()
            hess.xx, hess.aa, hess.count = xs, as_, cnt[rank * shard:(rank + 1) * shard].clone()
            # gate/up prior: the layer's mean per-token moment over all experts
            msum = torch.zeros((R.H, R.H), dtype=torch.float64, device=dev)
            for e in range(shard):
                msum += xs[e].double()
            dist.all_reduce(msum)
            hess.layer_mean_all = msum / cnt.sum().clamp_min(1).double()
            del msum
            red_s = time.monotonic() - t0
            layer_dir.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                save_file({k_: v.cpu().contiguous() for k_, v in col.d.items()}, str(a.stats / f'layer-{l:02d}.safetensors'),
                          metadata={'run_id': run_id, 'calib_sha256': calib_sha, 'source_revision': R.SOURCE_REV})
            wt = Weighting('imatrix', {k_: v for k_, v in col.d.items()}, device=dev, shrink_tokens=a.shrink_tokens,
                           col_floor=a.col_floor, col_power=a.col_power)
            # ---------------------------------------------------------------- codebook
            t0 = time.monotonic()
            book = torch.empty((a.k, 8), dtype=torch.uint8, device=dev)
            info = None
            if rank == 0:
                e8, info = train_codebook(gpu, lw, wt, l, a.k, a.iterations, log)
                book.copy_(e8.view(torch.uint8))
            dist.broadcast(book, 0)
            cb_bytes = book.cpu().numpy().tobytes()
            cb = book.view(torch.float8_e4m3fn).float().contiguous()
            if rank == 0:
                (layer_dir / 'codebook.bin').write_bytes(cb_bytes)
            book_s = time.monotonic() - t0
            # ---------------------------------------------------------------- hVQ
            t0 = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            res = fit_shard(lw, wt, hess, cb, cb_bytes, a.k, l, var, hv, hvq, my_experts, layer_dir, rank, a.baseline_every, log)
            fit_s = time.monotonic() - t0
            res.update({'peak_gpu_gib_forward': round(peak_fwd, 1),
                        'peak_gpu_gib_fit': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                        'host_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)})
            (layer_dir / f'rank-{rank}.json').write_text(json.dumps(res) + '\n')
            c = cnt.double()
            hd = {'normalization': 'token: each routed token x x^T (a a^T) divided by its own mean square',
                  'precision': 'chunk Gram matrices as FP32 GEMMs, per-expert sums in FP32',
                  'tokens_routed': int(c.sum()), 'count_min': int(c.min()), 'count_median': float(c.median()),
                  'count_max': int(c.max()), 'experts_unrouted': int((c == 0).sum()),
                  'experts_below_hidden': int((c < R.H).sum()), 'experts_below_ffn': int((c < R.FF).sum())}
            del hess, col, xs, as_, wt
            torch.cuda.empty_cache()
            dist.barrier()
            if rank == 0:
                ranks = [json.loads((layer_dir / f'rank-{r}.json').read_text()) for r in range(world)]
                sums = {}
                for g in GROUPS:
                    s = {k_: sum(r_['raw_error_sums'][g].get(k_, 0.0) for r_ in ranks)
                         for k_ in ('hvq', 'energy', 'ivq', 'hvq_sampled', 'energy_sampled', 'plain_hvq', 'plain_energy',
                                    'plain_ivq', 'plain_energy_sampled')}
                    sums[g] = {'hvq_output_rel': s['hvq'] / s['energy'],
                               'sampled_hvq_output_rel': s['hvq_sampled'] / s['energy_sampled'] if s['energy_sampled'] else None,
                               'sampled_ivq_output_rel': s['ivq'] / s['energy_sampled'] if s['energy_sampled'] else None,
                               'plain_rel_mse_hvq': s['plain_hvq'] / s['plain_energy'],
                               'sampled_plain_rel_mse_ivq': s['plain_ivq'] / s['plain_energy_sampled'] if s['plain_energy_sampled'] else None}
                receipt = {'layer': l, 'method': 'hvq', 'index_bits': 13, 'k': a.k, 'codebook_id': fmt.book_id(l),
                           'codebook_name': fmt.book_name(l), 'codebook_sha256': hashlib.sha256(cb_bytes).hexdigest(),
                           't': a.t, 'damping': a.damp, 'hessian': {'shrink_tokens': a.hess_shrink, **hd},
                           'weighting': {'mode': 'imatrix', 'col_power': a.col_power, 'shrink_tokens': a.shrink_tokens,
                                         'col_floor': a.col_floor, 'gain_refit': GAIN_REFIT, 'assignment': hv.ASSIGNMENT},
                           'codebook_training': info, 'calibration_run_id': run_id, 'calib_sha256': calib_sha,
                           'calibration_tokens': a.calib_chunks * T, 'source_revision': R.SOURCE_REV,
                           'tools': source_hashes(), 'errors': sums,
                           'seconds': {'read_wait': round(t_read, 1), 'forward_collect': round(fwd_s, 1), 'reduce': round(red_s, 1),
                                       'codebook': round(book_s, 1), 'fit': round(fit_s, 1),
                                       'layer_total': round(time.monotonic() - t_layer, 1)},
                           'ranks': [{k_: r_[k_] for k_ in ('rank', 'experts', 'file', 'file_sha256', 'seconds', 'factorization',
                                                            'payloads_cross_checked', 'peak_gpu_gib_forward', 'peak_gpu_gib_fit',
                                                            'host_rss_gib')} for r_ in ranks]}
                (layer_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
                log({'phase': 'layer-done', 'layer': l, 'errors': sums, 'seconds': receipt['seconds'],
                     'fit_seconds_rank0': ranks[0]['seconds'], 'peak_gib': [ranks[0]['peak_gpu_gib_forward'], ranks[0]['peak_gpu_gib_fit']],
                     'tokens_routed': hd.get('tokens_routed'), 'count_min': hd.get('count_min'), 'count_median': hd.get('count_median'),
                     'elapsed_min': round((time.monotonic() - t_run) / 60, 1)})
            dist.barrier()
        del lw
        torch.cuda.empty_cache()
        if l in CKPT_LAYERS and l < last and not a.no_ckpt:
            d = ck_dir / f'layer-{l:02d}'
            d.mkdir(parents=True, exist_ok=True)
            cpu = hid.cpu()
            cpu_held = held.cpu() if held is not None else None

            # the routes as of this layer: later layers add to the dict before the write runs
            def write(d=d, cpu=cpu, cpu_held=cpu_held, routes=dict(held_routes)):
                torch.save(cpu, d / f'rank-{rank}.pt')
                if cpu_held is not None:
                    torch.save(cpu_held, d / 'heldout.pt')
                    torch.save(routes, d / 'heldout-routes.pt')
                (d / f'rank-{rank}.done').write_text(run_id + '\n')
            ck_pool.submit(write)
            del cpu, cpu_held
    ck_pool.shutdown(wait=True)
    if rank == 0 and held is not None and last == R.L - 1:
        # the original's final states, routes and next-token scores on the held-out sequences
        hd_dir = a.out / 'heldout'
        hd_dir.mkdir(parents=True, exist_ok=True)
        glob = R.Globals(src, dev, embed=False, head=True)
        ids = torch.as_tensor(tokens[held_idx].astype(np.int64), device=dev)
        nll, top1, n = 0.0, 0, 0
        for i in range(held.shape[0]):
            lg = glob.logits(held[i])
            lp = torch.log_softmax(lg[:-1], -1)
            nll += float(-lp.gather(1, ids[i, 1:, None]).sum())
            top1 += int((lg[:-1].argmax(-1) == ids[i, 1:]).sum())
            n += T - 1
        routes = torch.stack([held_routes[l] for l in sorted(held_routes)]) if len(held_routes) == len(R.MOE_LAYERS) else None
        out = {'final_hidden': held.cpu().contiguous(), 'input_ids': ids.cpu().to(torch.int32).contiguous()}
        if routes is not None:
            out['routes'] = routes.contiguous()
        save_file(out, str(hd_dir / 'teacher.safetensors'), metadata={'run_id': run_id, 'source_revision': R.SOURCE_REV})
        summary = {'sequences': int(held.shape[0]), 'tokens': int(held.shape[0] * T), 'scored_positions': n,
                   'nll': nll / n, 'top1_accuracy': top1 / n, 'chunk_indices': [int(held_idx[0]), int(held_idx[-1])]}
        (hd_dir / 'teacher-summary.json').write_text(json.dumps(summary, indent=1) + '\n')
        log({'phase': 'teacher-heldout', **summary})
    dist.barrier()
    log({'phase': 'finished', 'elapsed_min': round((time.monotonic() - t_run) / 60, 1), 'all_ranks': True})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
