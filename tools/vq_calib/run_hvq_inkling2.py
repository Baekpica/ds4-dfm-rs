#!/usr/bin/env python3
"""Inkling-2-Small routed-expert hVQ: calibration statistics and conversion, one MoE layer at a time on
N GPUs (torchrun, one rank per GPU).

Per layer l:
  1. every rank streams its share of the calibration sequences through layer l (inkling2_ref, the MXFP8
     original's own arithmetic) and sums, per routed expert, the diagonal statistics (count, Σx², Σa²,
     token-normalized and route-weighted variants) and the token-normalized second moments
     (ds41_hess.LayerHess: Σ x xᵀ / mean(x²) over the gate/up input, Σ a aᵀ / mean(a²) over the down
     input), plus Σx² of every non-expert projection input (Q4_K importance);
  2. the diagonal statistics are all-reduced; the second moments are reduce-scattered, so rank r holds
     experts [r·E/N, (r+1)·E/N);
  3. rank 0 trains the layer's codebook (K = 4096, E4M3, shared by the 384 experts and gate/up/down) by
     importance-weighted Lloyd (axk2_weighting `imatrix`, column power 0.75: the iVQ stage) and
     broadcasts it;
  4. every rank assigns its experts' codes and F16 row gains by hVQ (ds41_hvq: 8-column blocks in order,
     error feedback through H_t = D½((1−t)I + tR)D½, least-squares gains under H_t), packs them as GVQ1 v1
     12-bit payloads, cross-checks sampled payloads with the independent numpy encoder and decoder, and
     writes them; the moments are dropped.

Every rank keeps its sequences' residual stream (the current layer input and the inputs of the block
boundary layers, BF16) in host memory. Layers before --first are re-run without statistics (catch-up).

    torchrun --nproc-per-node 4 run_hvq_inkling2.py --source $SRC --calib $W/calib/calib.safetensors \
        --out $W/convert --stats $W/stats
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
from safetensors.torch import load_file, save_file

import inkling2_ref as R
import inkling2_vq_format as fmt

TOOLS = Path(__file__).resolve().parent
TRAIN_ROWS = {'gate_proj': 24, 'up_proj': 24, 'down_proj': 48}     # equal vectors per projection type
HELD_ROWS = {'gate_proj': 2, 'up_proj': 2, 'down_proj': 4}
SEED = 20261113
GROUPS = {'x': ('gate_proj', 'up_proj'), 'a': ('down_proj',)}
BATCH = {'x': 8, 'a': 24}
SHAPES = fmt.SHAPES


def emit(rank, value):
    if rank == 0 or value.get('all_ranks'):
        print(json.dumps({'rank': rank, 't': round(time.time(), 1), **value}), flush=True)


def source_hashes():
    files = [Path(__file__), TOOLS / 'inkling2_ref.py', TOOLS / 'inkling2_vq_format.py', TOOLS / 'ds41_hvq.py',
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
    """Per-expert diagonal statistics (FP64) and the hVQ second moments of one layer, on one rank."""

    def __init__(self, dev, hess=True):
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device=dev)
        self.d = {'count': torch.zeros(R.E, dtype=torch.int64, device=dev), 'w2': z(R.E),
                  'x2': z(R.E, R.H), 'a2': z(R.E, R.FF), 'x2n': z(R.E, R.H), 'a2n': z(R.E, R.FF),
                  'x2w': z(R.E, R.H), 'a2w': z(R.E, R.FF)}
        self.nx = {'q_in': z(R.H), 'k_in': z(R.H), 'v_in': z(R.H), 'o_in': z(R.HEADS * R.HD), 'mlp_in': z(R.H),
                   'tokens': z(1)}
        if hess:
            from ds41_hess import LayerHess
            self.hess = LayerHess(R.E, R.H, R.FF, dev)
        else:
            self.hess = None

    # hooks called by inkling2_ref
    def attn_inputs(self, l, xq, xk, xv):
        self.nx['q_in'] += xq.float().square().sum(0).double()
        self.nx['k_in'] += xk.float().square().sum(0).double()
        self.nx['v_in'] += xv.float().square().sum(0).double()
        self.nx['tokens'] += xq.shape[0]

    def o_input(self, l, o):
        self.nx['o_in'] += o.float().square().sum(0).double()

    def mlp_input(self, x):
        self.nx['mlp_in'] += x.float().square().sum(0).double()

    def expert(self, l, e, t, xe, act, we):
        d = self.d
        x2 = xe.float().square()
        a2 = act.float().square()
        w2 = we.double().square()
        d['count'][e] += xe.shape[0]
        d['w2'][e] += w2.sum()
        d['x2'][e] += x2.sum(0).double()
        d['a2'][e] += a2.sum(0).double()
        d['x2n'][e] += (x2 / x2.mean(-1, keepdim=True).clamp_min(1e-30)).sum(0).double()
        d['a2n'][e] += (a2 / a2.mean(-1, keepdim=True).clamp_min(1e-30)).sum(0).double()
        d['x2w'][e] += (x2.double() * w2[:, None]).sum(0)
        d['a2w'][e] += (a2.double() * w2[:, None]).sum(0)
        if self.hess is not None:
            self.hess.add(e, xe, act)


# ----------------------------------------------------------------------------------- data
def load_calib(path):
    t = load_file(str(path))
    ids, off = t['input_ids'].to(torch.int64), t['offsets'].to(torch.int64)
    return ids, off


def assign_ranks(off, world):
    """Sequences to ranks: longest first to the rank with the least attention-weighted load."""
    lens = (off[1:] - off[:-1]).tolist()
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    load = [0.0] * world
    out = [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda k: load[k])
        out[r].append(i)
        load[r] += lens[i] * (1.0 + lens[i] / 16384.0)
    return [sorted(o) for o in out]


# --------------------------------------------------------------------------------- codebook
def train_codebook(gpu, lw, wt, layer, k, iterations, log):
    """Pool of stratified rows of all experts (TRAIN_ROWS), RMS gains, imatrix weights (row factor g²
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


def factor_safe(h):
    """V with H = V V^T (ds41_hvq.factor); a matrix that fails gets its diagonal lifted in h (as
    convert_layer_ds41_hvq.factor_safe). Returns (V, matrices lifted, largest lift x mean)."""
    import ds41_hvq as hv
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


def pack12(ids):
    """int32 [rows, nv] -> GVQ1 12-bit pair stream bytes (FORMAT.md): [a & 255, (a >> 8 & 15) | (b & 15) << 4, b >> 4]."""
    a, b = ids[..., 0::2], ids[..., 1::2]
    out = torch.stack([a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255], -1)
    return out.to(torch.uint8).reshape(-1)


def fit_shard(lw, wt, hess, cb, cb_bytes, layer, var, hvq, experts, out_dir, rank, baseline_every, log):
    """hVQ codes and gains of this rank's experts -> GVQ1 payloads in one binary file + index + receipt."""
    import ds41_hvq as hv
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
                t0 = tick('hessian', t0)
                v, n_lift, top = factor_safe(h)
                lifted, lift_max = lifted + n_lift, max(lift_max, top)
                t0 = tick('factor', t0)
                ids, g = hvq.quantize(w, cb, h, v)
                del v
                t0 = tick('assign', t0)
                dec = hvq.restore(ids, g, cb)
                errs[group]['hvq'] += float(hv.output_error(w - dec, h).sum())
                errs[group]['energy'] += float(hv.output_error(w, h).sum())
                errs[group]['plain_hvq'] += float((w - dec).double().square().sum())
                errs[group]['plain_energy'] += float(w.double().square().sum())
                # iVQ baseline (t = 0, the same book) on every `baseline_every`-th expert, scored under H_t
                sel = [i for i, e in enumerate(es) if baseline_every and e % baseline_every == 0]
                if sel:
                    ws_, cs_ = w[sel].contiguous(), col[sel]
                    h0 = hv.hessian(cs_, torch.eye(cols, dtype=torch.float64, device=lw.dev).expand(len(sel), cols, cols), 0.0, var['damp'])
                    v0, _, _ = factor_safe(h0)
                    i0, g0 = hvq.quantize(ws_, cb, h0, v0)
                    d0 = hvq.restore(i0, g0, cb)
                    errs[group]['ivq'] += float(hv.output_error(ws_ - d0, h[sel]).sum())
                    errs[group]['energy_sampled'] += float(hv.output_error(ws_, h[sel]).sum())
                    errs[group]['hvq_sampled'] = errs[group].get('hvq_sampled', 0.0) + float(hv.output_error(ws_ - dec[sel], h[sel]).sum())
                    errs[group]['plain_ivq'] += float((ws_ - d0).double().square().sum())
                    errs[group]['plain_energy_sampled'] += float(ws_.double().square().sum())
                    del h0, v0, i0, g0, d0, ws_
                    t0 = tick('baseline', t0)
                del h, r, dec
                row0 = 0
                for p in projs:
                    rows = SHAPES[p][0]
                    for i, e in enumerate(es):
                        ip, gp = ids[i, row0:row0 + rows].contiguous(), g[i, row0:row0 + rows].contiguous()
                        low = pack12(ip).cpu().numpy().tobytes()
                        gb = gp.cpu().numpy().astype('<f2').tobytes()
                        payload = fmt.assemble_vq(gb, low, None, rows, cols, 4096, cb_bytes, bid)
                        if rng.random() < 0.02:          # independent numpy encoder + decoder on sampled matrices
                            ref = fmt.encode_vq(np.frombuffer(gb, '<f2'), ip.cpu().numpy(), rows, cols, 4096, cb_bytes, bid)
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
                if (b0 // BATCH[group]) % 4 == 3:
                    log({'phase': 'assign', 'layer': layer, 'group': group, 'experts_done': b0 + len(es)})
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
            'index': index, 'errors': summary, 'raw_error_sums': errs, 'seconds': {k: round(v, 1) for k, v in seconds.items()},
            'factorization': {'experts_lifted': lifted, 'largest_lift_x_mean': lift_max},
            'payloads_cross_checked': checked}


# --------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True,
                    help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8 at R.SOURCE_REV)')
    ap.add_argument('--calib', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stats', required=True, type=Path)
    ap.add_argument('--layers', default='0-47')
    ap.add_argument('--max-seqs', type=int, default=0, help='test runs: first N sequences only')
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--damp', type=float, default=0.0)
    ap.add_argument('--hess-shrink', type=float, default=4096.0)
    ap.add_argument('--col-power', type=float, default=0.75)
    ap.add_argument('--shrink-tokens', type=float, default=256.0)
    ap.add_argument('--col-floor', type=float, default=0.02)
    ap.add_argument('--k', type=int, default=4096)
    ap.add_argument('--iterations', type=int, default=32)
    ap.add_argument('--chunk-tokens', type=int, default=65536)
    ap.add_argument('--baseline-every', type=int, default=8)
    ap.add_argument('--save-hess-layers', default='')
    ap.add_argument('--hess-dir', type=Path, help='--save-hess-layers: where the Hessians go')
    ap.add_argument('--stop-after', type=int, default=-1, help='stop after this layer (tests)')
    ap.add_argument('--images', type=Path, help='image calibration sequences (input_ids, offsets) from build_calib_inkling2.py build-images')
    ap.add_argument('--image-features', type=Path, help='their vision features (inkling2_vision.py)')
    a = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(lrank)
    dev = torch.device('cuda', lrank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log = lambda v: emit(rank, v)
    a.out.mkdir(parents=True, exist_ok=True)
    a.stats.mkdir(parents=True, exist_ok=True)
    convert = set(parse_layers(a.layers))
    save_hess = set(parse_layers(a.save_hess_layers))
    var = {'t': a.t, 'damp': a.damp, 'hess_shrink': a.hess_shrink}
    assert R.E % world == 0
    shard = R.E // world
    my_experts = list(range(rank * shard, (rank + 1) * shard))

    ids_all, off = load_calib(a.calib)
    nseq = len(off) - 1 if not a.max_seqs else min(a.max_seqs, len(off) - 1)
    off = off[:nseq + 1]
    calib_sha = hashlib.sha256(a.calib.read_bytes()).hexdigest()
    pool_seqs = [(ids_all[off[i]:off[i + 1]], None) for i in range(nseq)]
    img_sha = None
    if a.images is not None:
        # image sequences: placeholder rows (id 200054) take the vision tower's merger outputs, in order
        it = load_file(str(a.images))
        ft = load_file(str(a.image_features))
        iids, ioff = it['input_ids'].to(torch.int64), it['offsets'].to(torch.int64)
        feats, foff = ft['features'], ft['feature_offsets'].to(torch.int64)
        for j in range(len(ioff) - 1):
            sid = iids[ioff[j]:ioff[j + 1]]
            f = feats[foff[j]:foff[j + 1]] if foff[j + 1] > foff[j] else None
            if int((sid == 200054).sum()) != (0 if f is None else f.shape[0]):
                raise ValueError(f'image sequence {j}: placeholders and features differ')
            pool_seqs.append((sid, f))
        img_sha = hashlib.sha256(a.images.read_bytes() + a.image_features.read_bytes()).hexdigest()
    lens = torch.tensor([0] + [len(x) for x, _ in pool_seqs]).cumsum(0)
    mine = assign_ranks(lens, world)[rank]
    seqs, cursor = [], 0
    for i in mine:
        n = len(pool_seqs[i][0])
        seqs.append((i, cursor, n))
        cursor += n
    n_tok = cursor
    tok = torch.cat([pool_seqs[i][0] for i in mine]) if mine else torch.zeros(0, dtype=torch.int64)
    totals = torch.tensor([n_tok, len(mine)], device=dev)
    dist.all_reduce(totals)
    run_id = hashlib.sha256(json.dumps({'calib': calib_sha, 'images': img_sha, 'max_seqs': a.max_seqs, 'world': world,
                                        'source': R.SOURCE_REV, 'transformers': R.TRANSFORMERS_REV}).encode()).hexdigest()
    log({'phase': 'start', 'world': world, 'sequences': int(totals[1]), 'tokens': int(totals[0]),
         'my_tokens': n_tok, 'my_sequences': len(mine), 'calib_sha256': calib_sha, 'run_id': run_id,
         'convert_layers': sorted(convert), 'variant': var, 'all_ranks': True})

    src = R.Source(a.source)
    glob = R.Globals(src, dev, head=False)
    hid = torch.empty((n_tok, R.H), dtype=torch.bfloat16)
    for i, c0, n in seqs:
        rows = glob.embed_ids(tok[c0:c0 + n])
        f = pool_seqs[i][1]
        if f is not None:
            rows[tok[c0:c0 + n].to(dev) == 200054] = f.to(dev)
        hid[c0:c0 + n] = rows.cpu()
    snaps = []
    pool = ThreadPoolExecutor(1)
    nxt_weights = pool.submit(src.read, src.layer_names(0))
    from axk2_vq_gpu import GpuVq
    from axk2_weighting import Weighting
    import ds41_hvq as hv
    gpu = GpuVq()
    hvq = hv.Hvq()
    last = max(convert) if a.stop_after < 0 else a.stop_after
    for l in range(0, last + 1):
        t_layer = time.monotonic()
        host = nxt_weights.result()
        if l + 1 <= last:
            nxt_weights = pool.submit(src.read, src.layer_names(l + 1))
        lw = R.LayerWeights(l, host, dev)
        del host
        do = l in convert
        layer_dir = a.out / f'layer-{l:02d}'
        done = (layer_dir / 'receipt.json').exists()
        if done:
            do = False
        col = Collector(dev, hess=do) if do else None
        nxt = torch.empty_like(hid)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        # forward in token chunks of whole sequences
        groups, cur, cur_n = [], [], 0
        for s in seqs:
            if cur and cur_n + s[2] > a.chunk_tokens:
                groups.append(cur)
                cur, cur_n = [], 0
            cur.append(s)
            cur_n += s[2]
        if cur:
            groups.append(cur)
        for grp in groups:
            c0 = grp[0][1]
            c1 = grp[-1][1] + grp[-1][2]
            n = c1 - c0
            resid_all = torch.empty((n, R.H), dtype=torch.float32, device=dev)
            x_all = torch.empty((n, R.H), dtype=torch.bfloat16, device=dev)
            for i, s0, sn in grp:
                a0 = s0 - c0
                x_in = hid[s0:s0 + sn].to(dev, non_blocking=False)
                blocks = [sp[s0:s0 + sn].to(dev) for sp in snaps]
                pos = torch.arange(sn, device=dev)
                resid, mb = R.attn_part(lw, x_in, blocks, pos, hooks=col)
                x_all[a0:a0 + sn] = R.mlp_in(lw, resid, mb)
                resid_all[a0:a0 + sn] = resid
                del x_in, blocks, resid, mb
            if col is not None:
                col.mlp_input(x_all)
            y, _ = R.moe(lw, x_all, hooks=col)
            for i, s0, sn in grp:
                a0 = s0 - c0
                nxt[s0:s0 + sn] = R.finish_layer(lw, resid_all[a0:a0 + sn], y[a0:a0 + sn]).cpu()
            del resid_all, x_all, y
        fwd_s = time.monotonic() - t0
        peak_fwd = torch.cuda.max_memory_allocated() / 2**30
        if R.is_boundary(l):
            snaps.append(hid)
        hid = nxt
        if not do:
            log({'phase': 'forward-only', 'layer': l, 'seconds': round(fwd_s, 1), 'reason': 'done' if done else 'catch-up'})
            del lw
            torch.cuda.empty_cache()
            continue
        # ---------------------------------------------------------------- reduce
        t0 = time.monotonic()
        for k_, v in col.d.items():
            dist.all_reduce(v)
        for k_, v in col.nx.items():
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
        if l in save_hess:
            hd = a.hess_dir / f'layer-{l:02d}' / f'rank-{rank}'
            hess.meta = {'layer': l, 'experts': [my_experts[0], my_experts[-1]], 'run_id': run_id}
            hess.save(hd)
            torch.save(hess.layer_mean_all.cpu(), hd / 'layer_mean.pt')
        stats_cpu = {k_: v.cpu() for k_, v in col.d.items()}
        if rank == 0:
            save_file({k_: v.contiguous() for k_, v in stats_cpu.items()}, str(a.stats / f'layer-{l:02d}.safetensors'),
                      metadata={'run_id': run_id, 'calib_sha256': calib_sha, 'source_revision': R.SOURCE_REV})
            save_file({k_: v.cpu().contiguous() for k_, v in col.nx.items()}, str(a.stats / f'layer-{l:02d}-nonexpert.safetensors'),
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
        res = fit_shard(lw, wt, hess, cb, cb_bytes, l, var, hvq, my_experts, layer_dir, rank, a.baseline_every, log)
        fit_s = time.monotonic() - t0
        res.update({'peak_gpu_gib_forward': round(peak_fwd, 1),
                    'peak_gpu_gib_fit': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                    'host_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)})
        (layer_dir / f'rank-{rank}.json').write_text(json.dumps(res) + '\n')
        c = cnt.double()
        hd = {'normalization': 'token: each routed token x x^T (a a^T) divided by its own mean square',
              'precision': 'chunk Gram matrices in FP64 (<= 32768 tokens), per-expert sums in FP32',
              'tokens_routed': int(c.sum()), 'count_min': int(c.min()), 'count_median': float(c.median()),
              'count_max': int(c.max()), 'experts_unrouted': int((c == 0).sum()),
              'experts_below_hidden': int((c < R.H).sum()), 'experts_below_ffn': int((c < R.FF).sum()),
              'effective_rank_rank0_shard': {'xx': hess.effective_rank('xx'), 'aa': hess.effective_rank('aa')} if rank == 0 else None}
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
            receipt = {'layer': l, 'method': 'hvq', 'index_bits': 12, 'k': a.k, 'codebook_id': fmt.book_id(l),
                       'codebook_name': fmt.book_name(l), 'codebook_sha256': hashlib.sha256(cb_bytes).hexdigest(),
                       't': a.t, 'damping': a.damp, 'hessian': {'shrink_tokens': a.hess_shrink, **hd},
                       'weighting': {'mode': 'imatrix', 'col_power': a.col_power, 'shrink_tokens': a.shrink_tokens,
                                     'col_floor': a.col_floor, 'gain_refit': hv.GAIN_REFIT, 'assignment': hv.ASSIGNMENT},
                       'codebook_training': info, 'calibration_run_id': run_id, 'calib_sha256': calib_sha,
                       'source_revision': R.SOURCE_REV, 'transformers_revision': R.TRANSFORMERS_REV,
                       'tools': source_hashes(), 'errors': sums,
                       'seconds': {'forward_collect': round(fwd_s, 1), 'reduce': round(red_s, 1),
                                   'codebook': round(book_s, 1), 'fit': round(fit_s, 1),
                                   'layer_total': round(time.monotonic() - t_layer, 1)},
                       'ranks': [{k_: r_[k_] for k_ in ('rank', 'experts', 'file', 'file_sha256', 'seconds', 'factorization',
                                                        'payloads_cross_checked', 'peak_gpu_gib_forward', 'peak_gpu_gib_fit',
                                                        'host_rss_gib')} for r_ in ranks]}
            (layer_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
            log({'phase': 'layer-done', 'layer': l, 'errors': sums, 'seconds': receipt['seconds'],
                 'tokens_routed': hd.get('tokens_routed'), 'count_min': hd.get('count_min'), 'count_median': hd.get('count_median')})
        dist.barrier()
        del lw
        torch.cuda.empty_cache()
    if last == R.L - 1 and not (a.out / 'final-hidden.done').exists():
        # importance of the output head input: the final normalised hidden state of every token
        glob_h = R.Globals(src, dev, head=False)
        fx = torch.zeros(R.H, dtype=torch.float64, device=dev)
        for i, s0, sn in seqs:
            fh = glob_h.final_hidden([sp[s0:s0 + sn].to(dev) for sp in snaps], hid[s0:s0 + sn].to(dev))
            fx += fh.float().square().sum(0).double()
        ntok = torch.tensor([float(n_tok)], dtype=torch.float64, device=dev)
        dist.all_reduce(fx)
        dist.all_reduce(ntok)
        if rank == 0:
            save_file({'head_in': fx.cpu(), 'tokens': ntok.cpu()}, str(a.stats / 'head.safetensors'),
                      metadata={'run_id': run_id, 'calib_sha256': calib_sha, 'source_revision': R.SOURCE_REV})
            (a.out / 'final-hidden.done').write_text(run_id + '\n')
    dist.barrier()
    log({'phase': 'finished', 'all_ranks': True})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
