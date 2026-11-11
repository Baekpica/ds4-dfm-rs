#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash routed-expert hVQ: one MoE layer -> one DQVL v3 blob that drops into
the `vq8sh14-q4k-mtpnative` base GGUF in place of the iVQ (or weights-only) blob.

The codebook is the iVQ release's, byte for byte; K, offsets and sizes are unchanged. Only
the indices and the row gains are refitted (ds41_hvq.py): the 8-column blocks of every
expert matrix are assigned in order with error feedback through

    H_t = D^1/2 ((1 - t) I + t R) D^1/2 + damp * mean(D) * I

D = the iVQ release's column weights (axk2_weighting `imatrix`, col_power 0.75, shrink 256,
floor 0.02, from the release's statistics), R = the expert's input correlation from
token-normalized second moments (ds41_hess.py), rare experts shrunk toward the layer mean
(gate/up) or the identity (down). Gate and up share an input and one factorization.

t = 0 needs no second moments and reproduces the iVQ conversion: same indices, same gains.

A finished layer (blob-LL.bin + receipt-LL.json) is reused only if the blob matches its
receipt and the receipt was made for the requested variant, column statistics and
calibration run; otherwise the converter raises.
"""
import argparse
from dataclasses import dataclass
import enum
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from axk2_weighting import Weighting
from convert_layer_ds41 import LayerExperts, Meter, PROJ, ROOT, SHAPES, emit, statistics
import ds41_hvq as hv
from ds41_hess import LayerHess
import ds41_vq_format as fmt
from ds41_ref import Source
from ds41_vq_student import decode_blob

TOOLS = Path(__file__).resolve().parent
EXPERTS = 384
GROUPS = {'x': ('w1', 'w3'), 'a': ('w2',)}      # projections sharing an input: gate and up read x, down reads a
BATCH = {'x': 16, 'a': 48}                      # experts per launch
JITTER = 1e-8                                   # first diagonal lift (x mean) when a factorization fails
JITTER_STEPS = 8
RANK_DEFICIENCY = ('pseudo-count shrink of the token-normalized sums before the correlation: gate/up toward the '
                   'layer mean per token, (S_e + s M) / (n_e + s); down toward the identity at the expert\'s own '
                   'mean diagonal, (S_e + s m_e I) / (n_e + s)')


class Measure(enum.IntEnum):
    """What fit_layer records beside the indices and gains."""
    NONE = 0        # nothing
    ERRORS = 1      # output error under H_t and under D^1/2 R D^1/2
    ALL = 2         # and the weight-space errors of convert_layer_ds41.py's receipts


@dataclass(frozen=True)
class Variant:
    t: float
    damp: float
    hess_shrink: float = 256.0
    col_power: float = 0.75
    shrink_tokens: float = 256.0
    col_floor: float = 0.02

    def want(self):
        return {'method': 'hvq', 't': self.t, 'damping': self.damp, 'hessian_shrink_tokens': self.hess_shrink,
                'col_power': self.col_power, 'shrink_tokens': self.shrink_tokens, 'col_floor': self.col_floor}


def source_hashes():
    files = [Path(__file__), TOOLS / 'ds41_hvq.py', TOOLS / 'cuda/ds41_hvq.cu', TOOLS / 'ds41_hess.py',
             TOOLS / 'convert_layer_ds41.py', TOOLS / 'axk2_vq_gpu.py', TOOLS / 'axk2_weighting.py',
             TOOLS / 'cuda/axk2_vq_convert.cu', TOOLS / 'ds41_vq_format.py', TOOLS / 'ds41_ref.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def open_ivq(path):
    """The iVQ release as one virtual file: an assembled GGUF or the directory of its 40 parts."""
    path = Path(path)
    return fmt.open_base(fmt.default_parts(str(path)) if path.is_dir() else str(path))


def factor_safe(h):
    """V with H = V V^T; a matrix that fails gets its diagonal lifted (in h, so the gains and the
    recorded errors use the factored matrix). Returns (V, experts lifted, largest lift)."""
    v, info = hv.factor(h)
    bad = info != 0
    lifted, lift = int(bad.sum()), 0.0
    step = 0
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


class ErrorSums:
    """Output-error sums under a set of matrices H: new blob, published blob, and the weights' own energy."""

    def __init__(self):
        self.v = {}

    def add(self, key, w, new, old, h):
        row = self.v.setdefault(key, {'new': 0.0, 'old': 0.0, 'energy': 0.0})
        row['new'] += float(hv.output_error(w - new, h).sum())
        row['energy'] += float(hv.output_error(w, h).sum())
        if old is not None:
            row['old'] += float(hv.output_error(w - old, h).sum())

    def summary(self, has_old):
        return {k: {'hvq': r['new'] / r['energy'], 'ivq': r['old'] / r['energy'] if has_old else None}
                for k, r in self.v.items()}


def fit_layer(lx, wt, hess, cb, var, hvq, old=None, stats=None, log=None, measure=Measure.ALL):
    """Indices and gains of every expert of one layer.

    lx: LayerExperts; wt: Weighting (D); hess: LayerHess or None (t = 0 only); cb [K, 8] F32;
    old: (cb, idx, gains) of the published blob for the side-by-side errors.
    -> idx {p: int16 [384, rows, cols/8]}, gains {p: f16 [384, rows]}, record."""
    if hess is None and var.t != 0:
        raise ValueError('t > 0 needs second moments')
    idx = {p: np.empty((EXPERTS, SHAPES[p][0], SHAPES[p][1] // 8), np.int16) for p in SHAPES}
    gains = {p: np.empty((EXPERTS, SHAPES[p][0]), np.float16) for p in SHAPES}
    m_new, m_old = Meter(stats), Meter(stats)
    errs = ErrorSums()
    changed = {p: {'indices': 0, 'gains': 0} for p in SHAPES}
    seconds = {'weights': 0.0, 'hessian': 0.0, 'factor': 0.0, 'assign': 0.0, 'measure': 0.0}
    lifted, lift = 0, 0.0
    drift = {g: [] for g in GROUPS}
    t_all = time.monotonic()

    def tick(key, t0):
        torch.cuda.synchronize()
        seconds[key] += time.monotonic() - t0
        return time.monotonic()

    for group, projs in GROUPS.items():
        cols = SHAPES[projs[0]][1]
        prior = hess.layer_mean() if hess is not None and group == 'x' else None
        for e0 in range(0, EXPERTS, BATCH[group]):
            es = list(range(e0, min(EXPERTS, e0 + BATCH[group])))
            t0 = time.monotonic()
            w = torch.stack([torch.cat([lx.get(e, p) for p in projs]) for e in es])
            cw = [wt.col(e, PROJ[projs[0]]) for e in es]
            col = torch.stack([c if c is not None else torch.ones(cols, device='cuda') for c in cw])
            t0 = tick('weights', t0)
            if hess is None:
                r = torch.eye(cols, dtype=torch.float64, device='cuda').expand(len(es), cols, cols)
            else:
                r, _ = hv.correlation((hess.xx if group == 'x' else hess.aa)[e0:e0 + len(es)],
                                      hess.count[e0:e0 + len(es)], prior, var.hess_shrink)
            h = hv.hessian(col, r, var.t, var.damp)
            t0 = tick('hessian', t0)
            v, n_lift, top = factor_safe(h)
            lifted, lift = lifted + n_lift, max(lift, top)
            t0 = tick('factor', t0)
            ids, g = hvq.quantize(w, cb, h, v, drift=drift[group] if measure >= Measure.ERRORS else None)
            del v
            t0 = tick('assign', t0)
            dec = hvq.restore(ids, g, cb)
            dec_old = None
            if old is not None:
                dec_old = torch.stack([torch.cat([old[0][old[1][p][e].long()].flatten(1) * old[2][p][e][:, None]
                                                  for p in projs]) for e in es])
            if measure >= Measure.ERRORS:
                errs.add(f'{group}:H_t', w, dec, dec_old, h)
                if hess is not None:
                    errs.add(f'{group}:D^1/2 R D^1/2', w, dec, dec_old, hv.hessian(col, r, 1.0, 0.0))
            row0 = 0
            for p in projs:
                rows = SHAPES[p][0]
                for i, e in enumerate(es):
                    ip, gp = ids[i, row0:row0 + rows], g[i, row0:row0 + rows]
                    if measure == Measure.ALL:
                        m_new.add(e, p, w[i, row0:row0 + rows], dec[i, row0:row0 + rows])
                    if old is not None:
                        if measure == Measure.ALL:
                            m_old.add(e, p, w[i, row0:row0 + rows], dec_old[i, row0:row0 + rows])
                        changed[p]['indices'] += int((ip != old[1][p][e]).sum())
                        changed[p]['gains'] += int((gp.float() != old[2][p][e]).sum())
                    idx[p][e] = ip.to(torch.int16).cpu().numpy()
                    gains[p][e] = gp.cpu().numpy()
                row0 += rows
            del w, h, r, ids, g, dec, dec_old
            t0 = tick('measure', t0)
            if log is not None and (e0 // BATCH[group]) % 6 == 5:
                log({'phase': 'assign', 'group': group, 'experts': es[-1] + 1,
                     'seconds': round(time.monotonic() - t_all, 1)})
    total = {p: {'indices': idx[p].size, 'gains': gains[p].size} for p in SHAPES}
    # target energy over source energy by quarter of the block order (1.0 without feedback): whether
    # feedback pushes the later columns' targets away from what the codebook was fitted to
    quarters = {}
    for group, rows in drift.items():
        if not rows:
            continue
        tgt, src = (torch.stack([r[i] for r in rows]).mean(0) for i in (0, 1))
        cut = [len(tgt) * q // 4 for q in range(5)]
        quarters[group] = [float(tgt[cut[q]:cut[q + 1]].sum() / src[cut[q]:cut[q + 1]].sum()) for q in range(4)]
    record = {'seconds': {k: round(s, 2) for k, s in seconds.items()},
              'factorization': {'experts_lifted': lifted, 'largest_lift_x_mean': lift},
              'output_error': errs.summary(old is not None) if measure >= Measure.ERRORS else None,
              'target_energy_by_quarter': quarters or None,
              'weight_error': {'hvq': m_new.summary() if measure == Measure.ALL else None,
                               'ivq': m_old.summary() if measure == Measure.ALL and old is not None else None},
              'changed_from_ivq': {p: {k: changed[p][k] / total[p][k] for k in changed[p]} for p in SHAPES}
              if old is not None else None,
              'identical_to_ivq': all(changed[p][k] == 0 for p in SHAPES for k in changed[p]) if old is not None else None}
    return idx, gains, record


def reuse(L, out_dir, want):
    """True when blob-LL.bin and receipt-LL.json exist, the blob is the receipt's and the receipt was made
    for `want`; raises listing any difference."""
    path, rpath = out_dir / f'blob-{L:02d}.bin', out_dir / f'receipt-{L:02d}.json'
    if not (path.exists() and rpath.exists()):
        return False
    r = json.loads(rpath.read_text())
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    w = r.get('weighting') or {}
    hs = r.get('hessian') or {}
    got = {'sha256': r.get('sha256'), 'tensor': r.get('tensor'), 'method': r.get('method'), 't': r.get('t'),
           'damping': r.get('damping'), 'hessian_shrink_tokens': hs.get('shrink_tokens'),
           **{k: w.get(k) for k in ('col_power', 'shrink_tokens', 'col_floor')},
           'd_statistics_sha256': r.get('d_statistics_sha256'), 'calibration_run_id': r.get('calibration_run_id')}
    need = {'sha256': checksum, 'tensor': f'blk.{L}.ffn_exps_vq.blob', **want}
    diff = {k: (got.get(k), v) for k, v in need.items() if got.get(k) != v}
    if diff:
        raise ValueError(f'existing layer {L} ({path.name}) was made for another request: {diff}')
    emit({'layer': L, 'phase': 'reuse-verified', 'sha256': checksum})
    return True


class HostExperts(LayerExperts):
    """LayerExperts from a layer the collector already holds in host memory."""

    def __init__(self, host, L, dev='cuda'):
        self.t = {k: v.to(dev) for k, v in host.items() if f'layers.{L}.ffn.experts.' in k}
        self.L = L
        self.read_seconds = 0.0


class Job:
    """One layer's conversion in three steps, so the driver can pack and write in the background:
    open (the published blob and D), fit (device), finish (pack, decode check, write, receipt)."""

    def __init__(self, L, st, var, ivq, lx=None):
        self.L, self.var, self.ivq, self.t0 = L, var, ivq, time.monotonic()
        self.K = 8192 if L < 14 else 4096
        self.stats, self.stat_files, self.stats_sha, _, _ = st
        self.wt = Weighting('imatrix', self.stats, shrink_tokens=var.shrink_tokens, col_floor=var.col_floor,
                            col_power=var.col_power)
        self.name = f'blk.{L}.ffn_exps_vq.blob'
        # the published iVQ blob: its codebook is kept, its indices and gains are the comparison
        info = open_ivq(ivq).tensor(self.name)
        raw = open_ivq(ivq).vf.pread(info.abs_offset, info.nbytes)
        self.old = decode_blob(torch.from_numpy(raw).cuda())
        lay = fmt.blob_layout(EXPERTS, self.K, SHAPES)
        self.cb_bytes = raw[lay.cb_off:lay.cb_off + self.K * 8].reshape(self.K, 8).copy()
        self.cb = torch.from_numpy(fmt.e4m3_decode(self.cb_bytes)).cuda().contiguous()
        assert torch.equal(self.cb, self.old[0]), 'codebook decode'
        self.ivq_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        self.header, self.nbytes = raw[:16].copy(), info.nbytes
        self.lx = lx if lx is not None else LayerExperts(Source(ROOT), L)
        emit({'layer': L, 'phase': 'read', 'seconds': round(time.monotonic() - self.t0, 1), 'K': self.K, 't': var.t,
              'damping': var.damp})

    def fit(self, hess, hvq, measure=Measure.ALL):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        self.hess = {'sha256': hess.digest() if hess is not None else 'none', 'shrink_tokens': self.var.hess_shrink,
                     'rank_deficiency': RANK_DEFICIENCY,
                     **(hess.describe() if hess is not None else {'normalization': 'none (t = 0)'}),
                     **(hess.meta if hess is not None else {})}
        self.idx, self.gains, self.rec = fit_layer(self.lx, self.wt, hess, self.cb, self.var, hvq, self.old, self.stats,
                                                   lambda r: emit({'layer': self.L, **r}), measure)
        self.fit_seconds = time.monotonic() - t0
        self.peak_fit = torch.cuda.max_memory_allocated() / 2 ** 30
        del self.lx, self.old
        return self

    def finish(self, out_dir, run=None, strict_decode=True, collect=None, device_check=True):
        """collect: the driver's record of this layer's second-moment collection (seconds, peak memory).
        device_check=False keeps this step off the device (the driver runs it beside the next layer)."""
        L, var, t0 = self.L, self.var, time.monotonic()
        idx, gains, rec = self.idx, self.gains, self.rec
        blob = fmt.pack_vq_layer(L=L, cb_bytes=self.cb_bytes, gains=gains, idx=idx)
        assert blob.size == self.nbytes, (blob.size, self.nbytes)
        assert np.array_equal(blob[:16], self.header), 'DQVL header'
        # decode round trip of every expert: the forward's device decoder, then the strict reader
        if device_check:
            dcb, didx, dgain = decode_blob(torch.from_numpy(blob).cuda())
            assert torch.equal(dcb, self.cb)
            for p in SHAPES:
                assert torch.equal(didx[p], torch.from_numpy(idx[p]).cuda()), (L, p, 'indices')
                assert torch.equal(dgain[p], torch.from_numpy(gains[p]).cuda().float()), (L, p, 'gains')
            del dcb, didx, dgain
        if strict_decode:
            chk = fmt.decode_vq_blob(blob, self.name)
            assert np.array_equal(chk.cb_bytes, self.cb_bytes)
            for p in SHAPES:
                assert np.array_equal(chk.idx[p], idx[p]), (L, p, 'indices (strict reader)')
                assert np.array_equal(chk.gains[p].view(np.uint16), gains[p].view(np.uint16)), (L, p, 'gains (strict reader)')
            del chk
        digest = hashlib.sha256(blob.tobytes()).hexdigest()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'blob-{L:02d}.bin'
        tmp = path.with_suffix('.tmp')
        blob.tofile(tmp)
        tmp.rename(path)
        pack_seconds = time.monotonic() - t0
        run, collect = run or {}, collect or {}
        receipt = {
            'layer': L, 'tensor': self.name, 'K': self.K, 'bits': 13 if self.K == 8192 else 12, 'bytes': int(blob.size),
            'sha256': digest,
            'method': 'hvq', 'assignment': hv.ASSIGNMENT, 't': var.t, 'damping': var.damp,
            'gain_refit': 'ls', 'gain_refit_metric': hv.GAIN_REFIT, 'mode': 'imatrix', 'codebook_scope': 'shared',
            'codebook': {'source': 'iVQ release blob, bytes unchanged',
                         'sha256': hashlib.sha256(self.cb_bytes.tobytes()).hexdigest(),
                         'ivq_blob_sha256': self.ivq_sha, 'ivq_base': str(self.ivq)},
            'weighting': self.wt.describe(), 'stats_norm': 'sum',
            'd_statistics': self.stat_files, 'd_statistics_sha256': self.stats_sha,
            'hessian': self.hess, 'statistics_sha256': self.hess['sha256'],
            'calibration_run_id': run.get('run_id'), 'calibration_set_sha256': run.get('calib_sha256'),
            'calibration_content_sha256': run.get('calib_content_sha256'), 'calibration_run': run.get('run'),
            **{k: rec[k] for k in ('factorization', 'output_error', 'target_energy_by_quarter', 'weight_error',
                                   'changed_from_ivq', 'identical_to_ivq')},
            'decode_round_trip': ' + '.join(n for n, on in (('device decoder', device_check), ('strict reader', strict_decode))
                                            if on) + ', every expert',
            'collection': collect or None,
            'seconds': (collect.get('seconds') or 0.0) + self.fit_seconds + pack_seconds,
            'timing': {'collect': collect.get('seconds'), **rec['seconds'], 'fit': round(self.fit_seconds, 2),
                       'pack_verify_write': round(pack_seconds, 2)},
            'peak_gpu_gib': {'collect': collect.get('peak_gpu_gib'), 'convert': round(self.peak_fit, 2)},
            'peak_host_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
            'tools': source_hashes(),
        }
        rtmp = out_dir / f'receipt-{L:02d}.json.partial'
        rtmp.write_text(json.dumps(receipt, indent=1) + '\n')
        rtmp.rename(out_dir / f'receipt-{L:02d}.json')
        emit({'layer': L, 'phase': 'done', 'seconds': round(receipt['seconds'], 1), 'sha256': digest,
              'identical_to_ivq': rec['identical_to_ivq'], 'output_error': rec['output_error'],
              'weight_error': rec['weight_error'], 'timing': receipt['timing'], 'peak_gpu_gib': receipt['peak_gpu_gib']})
        return receipt


def convert(L, out_dir, st, hess, var, ivq, run=None, strict_decode=True):
    """st: convert_layer_ds41.statistics() of the iVQ release's statistics (D);
    hess: LayerHess of this layer or None (t = 0); run: the second-moment collection run record."""
    return Job(L, st, var, ivq).fit(hess, hv.Hvq()).finish(out_dir, run, strict_decode)


def verify(L, out_dir):
    """A finished layer read back: the blob is its receipt's, and the forward's device decoder and the
    strict reader return the same indices, gains and codebook for every expert."""
    r = json.loads((out_dir / f'receipt-{L:02d}.json').read_text())
    blob = np.fromfile(out_dir / f'blob-{L:02d}.bin', dtype=np.uint8)
    assert hashlib.sha256(blob.tobytes()).hexdigest() == r['sha256'] and blob.size == r['bytes'], (L, 'sha256')
    chk = fmt.decode_vq_blob(blob, r['tensor'])
    assert chk.L == L and chk.K == r['K'] and chk.nexp == EXPERTS and chk.shapes == SHAPES, (L, 'header')
    assert hashlib.sha256(chk.cb_bytes.tobytes()).hexdigest() == r['codebook']['sha256'], (L, 'codebook')
    cb, idx, gains = decode_blob(torch.from_numpy(blob).cuda())
    assert torch.equal(cb, torch.from_numpy(chk.cb).cuda())
    for p in SHAPES:
        assert torch.equal(idx[p].int(), torch.from_numpy(chk.idx[p]).cuda()), (L, p, 'indices')
        assert torch.equal(gains[p], torch.from_numpy(chk.gains[p].astype(np.float32)).cuda()), (L, p, 'gains')
        assert bool(torch.isfinite(gains[p]).all()) and float(gains[p].min()) >= 0.0, (L, p, 'gain range')
    emit({'layer': L, 'phase': 'verified', 'sha256': r['sha256'], 'K': chk.K, 't': r.get('t'), 'damping': r.get('damping')})


def parse_layers(text):
    if '-' in text:
        lo, hi = text.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(',')]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layers', required=True, help='comma list or a-b range')
    p.add_argument('--d-stats', type=Path, help="the iVQ release's statistics (column weights D)")
    p.add_argument('--ivq', help='iVQ release: assembled GGUF or the directory of its parts')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--t', type=float, help='0 = iVQ (no feedback) .. 1 = the full correlation')
    p.add_argument('--verify', action='store_true', help='read finished layers back instead of converting')
    p.add_argument('--damp', type=float, default=0.0, help='diagonal lift of H_t, x mean(D)')
    p.add_argument('--hess-shrink', type=float, default=256.0, help='pseudo-count of the rare-expert prior')
    p.add_argument('--hess', type=Path, help='directory with layer-LL/ second moments (t > 0)')
    p.add_argument('--no-strict-decode', action='store_true')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.verify:
        for L in parse_layers(a.layers):
            verify(L, a.out)
        return
    if a.d_stats is None or a.ivq is None or a.t is None:
        p.error('--d-stats, --ivq and --t are required to convert')
    var = Variant(a.t, a.damp, a.hess_shrink)
    for L in parse_layers(a.layers):
        st = statistics(a.d_stats, L, 'imatrix')
        hess = LayerHess.load(a.hess / f'layer-{L:02d}') if a.hess else None
        run = (hess.meta if hess is not None else {})
        want = {**var.want(), 'd_statistics_sha256': st[2], 'calibration_run_id': run.get('run_id')}
        if reuse(L, a.out, want):
            continue
        convert(L, a.out, st, hess, var, a.ivq, run, not a.no_strict_decode)
        del hess
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
