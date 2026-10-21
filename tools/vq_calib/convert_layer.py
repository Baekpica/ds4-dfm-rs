#!/usr/bin/env python3
"""A.X-K2 routed-expert VQ: one MoE layer -> one independent GVQ v2 GGUF part.

Per layer: read all 256 experts x gate/up/down (FP8 + 128x128 block scale,
dequantized in FP32 on the device), sample a stratified training pool and
disjoint held-out rows, weight every pool vector by the calibration
statistics (axk2_weighting, `--mode`), train one shared codebook
(equidistant init, weighted FP64 Lloyd, F32 -> F16 -> E4M3FN), assign every
vector against the stored codebook under the mode's column metric (optionally
refitting each row gain by weighted least squares), pack 9/10 bits on the
device, and verify the written bytes through independent CPU and GPU
decoders.

Forked from the Beam-1.0 converter (convert_layer.py); the VQ method is
YoungAi's (docs/TECHNICAL.md section 4.1) with importance weights added.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from axk2_source import Source, SOURCE_REV, PROJECTIONS, dequant
from axk2_vq_gpu import GpuVq, bits_for
from axk2_weighting import Weighting, load_stats, MODES

TOOLS = Path(__file__).resolve().parent
TRAIN_ROWS = {'gate_proj': 24, 'up_proj': 24, 'down_proj': 84}
HELD_ROWS = {'gate_proj': 2, 'up_proj': 2, 'down_proj': 7}
SEED = 5001


def source_hashes():
    files = [Path(__file__), TOOLS / 'axk2_vq_gpu.py', TOOLS / 'axk2_weighting.py', TOOLS / 'axk2_source.py',
             TOOLS / 'cuda/axk2_vq_convert.cu', TOOLS / 'axk2_vq_format.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def emit(value):
    print(json.dumps(value), flush=True)


def summaries(values):
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(a.mean()), 'p50': float(np.quantile(a, .5)), 'p95': float(np.quantile(a, .95)),
            'p99': float(np.quantile(a, .99)), 'max': float(a.max())}


def shapes(cfg, proj):
    h, f = cfg['hidden_size'], cfg['moe_intermediate_size']
    return (h, f) if proj == 'down_proj' else (f, h)


class LayerWeights:
    """FP8 experts of one layer resident on the device; FP32 dequant on demand."""
    def __init__(self, source, layer, host=None, threads=16, device='cuda'):
        self.cfg = source.config
        self.layer = layer
        names = source.expert_names(layer)
        stored = [s for n in names for s in source.weight_names(n)]
        t0 = time.monotonic()
        if host is None:
            host = source.read(stored, threads=threads)
        self.read_seconds = time.monotonic() - t0
        self.t = {n: host[n].to(device, non_blocking=True) for n in stored}
        torch.cuda.synchronize()
        self.names = names
        self.source_bytes = sum(v.numel() * v.element_size() for v in self.t.values())

    def name(self, e, p):
        return f'model.layers.{self.layer}.mlp.experts.{e}.{p}.weight'

    def get(self, e, p):
        n = self.name(e, p)
        return dequant(self.t[n], self.t[n + '_scale_inv'], torch.float32).contiguous()


def read_host(source, layer, threads=16):
    names = source.expert_names(layer)
    stored = [s for n in names for s in source.weight_names(n)]
    return source.read(stored, threads=threads)


def build_pool(gpu, lw, weighting, layer, seed=SEED):
    """Stratified pool + per-vector update/assignment weights + held-out rows."""
    cfg = lw.cfg
    xs, uws, aws, types = [], [], [], []
    held, strata = [], []
    weighted_assign = weighting.weighted_assignment
    for e in range(cfg['n_routed_experts']):
        dcol2 = None
        if weighting.mode == 'output':
            d = lw.get(e, 'down_proj')
            dcol2 = (d.double() ** 2).sum(0).float()                  # ||D[:, r]||^2, r over 2048
            del d
        for pi, p in enumerate(PROJECTIONS):
            rows, cols = shapes(cfg, p)
            ntrain, ntest = TRAIN_ROWS[p], HELD_ROWS[p]
            gen = torch.Generator().manual_seed(seed + layer * 1_000_000 + e * 3 + pi)
            chosen = torch.randperm(rows, generator=gen)[:ntrain + ntest]
            w = lw.get(e, p)[chosen.cuda()].contiguous()
            norm, gains = gpu.normalize(w)
            norm = norm.reshape(ntrain + ntest, cols)
            tr = chosen[:ntrain].cuda()
            rf = weighting.row_factor(e, p, gains[:ntrain], tr, dcol2)          # [ntrain] FP64
            scale = weighting.tensor_scale(e, p)
            col = weighting.col(e, p)                                          # [cols] or None
            colv = col if col is not None else torch.ones(cols, device='cuda')
            uw = (rf[:, None] * scale * colv.double()[None, :]).float()
            xs.append(norm[:ntrain].reshape(-1, 8))
            uws.append(uw.reshape(-1, 8))
            if weighted_assign:
                aws.append(colv[None, :].expand(ntrain, cols).reshape(-1, 8))
            types.append(torch.full((ntrain * cols // 8,), pi, dtype=torch.int8, device='cuda'))
            held.append((e, p, chosen[ntrain:].tolist(), w[ntrain:].clone(), col))
            strata.append({'expert': e, 'proj': p, 'train_rows': chosen[:ntrain].tolist(),
                           'heldout_rows': chosen[ntrain:].tolist()})
    pool = torch.cat(xs).contiguous()
    uw = torch.cat(uws).contiguous()
    aw = torch.cat(aws).contiguous() if weighted_assign else None
    typ = torch.cat(types)
    totals = {}
    if weighting.mode == 'none':
        uw = None
    elif weighting.mode in ('row', 'imatrix', 'imatrix_freq'):
        # Equal total weight per projection type, as the unweighted pool has equal counts.
        sums = [float(uw[typ == pi].double().sum()) for pi in range(3)]
        target = sum(sums) / 3
        for pi in range(3):
            if sums[pi] > 0:
                uw[typ == pi] *= target / sums[pi]
        totals = {'type_totals_before_equalize': sums}
    if uw is not None:
        uw /= uw.double().mean().float()            # mean 1 (scale-free, numerically tame)
        totals['type_totals'] = [float(uw[typ == pi].double().sum()) for pi in range(3)]
    return pool, uw, aw, held, strata, totals, typ


def heldout_error(gpu, held, cb, book_of=None):
    """Unweighted and column-weighted rel. error of held-out rows under the stored book(s)."""
    sse = en = wsse = wen = 0.0
    for _, p, _, w, col in held:
        book = book_of[p] if book_of is not None else cb
        norm, g = gpu.normalize(w)
        ids = gpu.assign(norm, book, colw=col)
        dec = gpu.restore_ids(ids, g, book, w.shape[1])
        d2 = (dec.double() - w.double()) ** 2
        r2 = w.double() ** 2
        sse += float(d2.sum())
        en += float(r2.sum())
        h = col.double() if col is not None else torch.ones(w.shape[1], dtype=torch.float64, device=w.device)
        wsse += float((d2 * h).sum())
        wen += float((r2 * h).sum())
    return {'rel_mse': sse / en, 'col_weighted_rel_err': wsse / wen}


def assign_tensor(gpu, w, cb, col, refit):
    """-> (ids int32 [rows*cols/8], F16 gains [rows]) for one matrix."""
    norm, gains = gpu.normalize(w)
    ids = gpu.assign(norm, cb, colw=col)
    if refit == 'ls':
        gpu.refit_gain(w, ids, cb, gains, colw=col)
        norm = gpu.normalize_gain(w, gains)
        ids = gpu.assign(norm, cb, colw=col)
        gpu.refit_gain(w, ids, cb, gains, colw=col)
    return ids, gains


class ErrorMeter:
    """Weight-space errors of a layer, unweighted and under calibration weights."""
    def __init__(self, stats):
        self.s = stats
        self.v = {k: 0.0 for k in ('sse', 'en', 'bf16_sse', 'imx_sse', 'imx_en', 'out_sse')}
        self.by_proj = {p: {'sse': 0.0, 'en': 0.0, 'imx_sse': 0.0, 'imx_en': 0.0, 'out_sse': 0.0} for p in PROJECTIONS}

    def add(self, e, p, w, rec, dcol2=None):
        d2 = (rec.double() - w.double()) ** 2
        r2 = w.double() ** 2
        sse, en = float(d2.sum()), float(r2.sum())
        bf = float(((rec.to(torch.bfloat16).double() - w.double()) ** 2).sum())
        self.v['sse'] += sse
        self.v['en'] += en
        self.v['bf16_sse'] += bf
        bp = self.by_proj[p]
        bp['sse'] += sse
        bp['en'] += en
        out = {'rel_mse': sse / max(en, 1e-300), 'bf16_restored_rel_mse': bf / max(en, 1e-300)}
        if self.s is not None:
            cnt = int(self.s['count'][e])
            imx = (self.s['a2'][e] if p == 'down_proj' else self.s['x2'][e]).double()
            h = imx / imx.mean() if cnt > 0 and imx.mean() > 0 else torch.ones_like(imx)
            isse = float((d2.sum(0) * h).sum())
            ien = float((r2.sum(0) * h).sum())
            hw = (self.s['a2w'][e] if p == 'down_proj' else self.s['x2w'][e]).double()
            if p == 'down_proj':
                osse = float((d2.sum(0) * hw).sum())
            else:
                b = (self.s['bg'][e] if p == 'gate_proj' else self.s['bu'][e]).double()
                w2 = max(float(self.s['w2'][e]), 1e-30)
                rowf = dcol2.double() * b / w2 if dcol2 is not None else torch.ones_like(b)
                osse = float(((d2 * hw[None, :]).sum(1) * rowf).sum())
            self.v['imx_sse'] += isse
            self.v['imx_en'] += ien
            self.v['out_sse'] += osse
            bp['imx_sse'] += isse
            bp['imx_en'] += ien
            bp['out_sse'] += osse
            out.update({'imatrix_rel_err': isse / max(ien, 1e-300), 'output_proxy_sse': osse})
        return out, d2

    def summary(self):
        v = self.v
        out = {'rel_mse': v['sse'] / v['en'], 'rel_rms': (v['sse'] / v['en']) ** .5,
               'bf16_restored_rel_mse': v['bf16_sse'] / v['en'], 'cosine_estimate': 1 - v['sse'] / v['en'] / 2}
        if self.s is not None:
            out.update({'imatrix_rel_err': v['imx_sse'] / v['imx_en'], 'output_proxy_sse': v['out_sse']})
        out['by_proj'] = {p: {'rel_mse': b['sse'] / max(b['en'], 1e-300),
                              **({'imatrix_rel_err': b['imx_sse'] / max(b['imx_en'], 1e-300),
                                  'output_proxy_sse': b['out_sse']} if self.s is not None else {})}
                          for p, b in self.by_proj.items()}
        return out


BOOK_GROUPS = {'shared': ((0, 1, 2),), 'split2': ((0, 1), (2,)), 'split3': ((0,), (1,), (2,))}


def book_layout(scope, layer):
    """-> [(group projection indices, codebook name, codebook id)] (4L gate/up or shared,
    4L+2 down, 4L+3 up under split3; 4L+1 stays reserved for a residual stage)."""
    base = f'model.layers.{layer}.mlp.vq_codebook'
    if scope == 'shared':
        return [((0, 1, 2), base, 4 * layer)]
    if scope == 'split2':
        return [((0, 1), base, 4 * layer), ((2,), base + '_down', 4 * layer + 2)]
    if scope == 'split3':
        return [((0,), base, 4 * layer), ((1,), base + '_up', 4 * layer + 3), ((2,), base + '_down', 4 * layer + 2)]
    raise ValueError(f'unknown codebook scope {scope}')


def train_books(gpu, pool, uw, aw, typ, k, iterations, scope):
    """One weighted Lloyd per book group on its share of the pool -> per-group results."""
    out = []
    for group in BOOK_GROUPS[scope]:
        sel = torch.zeros(len(pool), dtype=torch.bool, device=pool.device)
        for pi in group:
            sel |= typ == pi
        if len(group) == 3:
            xs, us, as_ = pool, uw, aw
        else:
            idx = torch.nonzero(sel).squeeze(1)
            xs = pool[idx].contiguous()
            us = uw[idx].contiguous() if uw is not None else None
            as_ = aw[idx].contiguous() if aw is not None else None
        e8, f16, f32, hist = gpu.train(xs, k, iterations, weights=us, assign_weights=as_)
        out.append({'group': group, 'e4m3': e8, 'f16': f16, 'history': hist, 'pool_vectors': len(xs)})
        del xs, us, as_
    return out


def train_layer(gpu, lw, layer, k, weighting, iterations, probe_iters=None, scope='shared'):
    pool, uw, aw, held, strata, totals, typ = build_pool(gpu, lw, weighting, layer)
    probe = None
    if probe_iters:
        probe = {}
        for iters in probe_iters:
            trial = train_books(gpu, pool, uw, aw, typ, k, iters, scope)
            probe[iters] = {}
            for b in trial:
                sub = [h for h in held if PROJECTIONS.index(h[1]) in b['group']]
                key = '+'.join(PROJECTIONS[pi].split('_')[0] for pi in b['group'])
                probe[iters][key] = {'heldout_e4m3': heldout_error(gpu, sub, b['e4m3'].float().contiguous()),
                                     'heldout_f16': heldout_error(gpu, sub, b['f16'].float().contiguous()),
                                     'final_weighted_distortion': b['history'][-1]['weighted_distortion']}
            emit({'phase': 'probe', 'layer': layer, 'k': k, 'iterations': iters, 'books': scope, 'result': probe[iters]})
    books = train_books(gpu, pool, uw, aw, typ, k, iterations, scope)
    info = {'pool_vectors': len(pool), 'pool_per_codeword': len(pool) / k / len(books), 'probe': probe,
            'pool_weight_totals': totals, 'books': []}
    for b in books:
        sub = [h for h in held if PROJECTIONS.index(h[1]) in b['group']]
        info['books'].append({'group': [PROJECTIONS[pi] for pi in b['group']], 'pool_vectors': b['pool_vectors'],
                              'iterations': b['history'],
                              'heldout_e4m3': heldout_error(gpu, sub, b['e4m3'].float().contiguous()),
                              'heldout_f16': heldout_error(gpu, sub, b['f16'].float().contiguous())})
    info['heldout_e4m3'] = heldout_error(gpu, held, None, book_of={p: books[[i for i, b in enumerate(books) if PROJECTIONS.index(p) in b['group']][0]]['e4m3'].float().contiguous() for p in PROJECTIONS})
    del pool, uw, aw, typ
    return books, held, strata, info


def weighting_config(recipe):
    """Weighting knobs from recipe['weighting'] or recipe['calibration'] (selected values only)."""
    cfg = dict(recipe.get('weighting') or {})
    cal = recipe.get('calibration') or {}
    if 'mode' not in cfg and cal.get('weighting_mode') in MODES:
        cfg['mode'] = cal['weighting_mode']
    if 'gain_refit' not in cfg and cal.get('gain_refit') in ('none', 'ls'):
        cfg['gain_refit'] = cal['gain_refit']
    for key in ('shrink_tokens', 'col_power', 'col_floor', 'col_shrink'):
        if key not in cfg and isinstance(cal.get(key), (int, float)):
            cfg[key] = cal[key]
    if 'books' not in cfg and cal.get('codebook_scope') in BOOK_GROUPS:
        cfg['books'] = cal['codebook_scope']
    return cfg


def convert_layer(source, gpu, layer, k, recipe, out_dir, work_dir, mode, refit, stats_dir=None,
                  probe_iters=None, host=None, iterations=None, shrink_tokens=None, books_scope=None,
                  col_power=None, col_floor=None):
    from axk2_vq_format import BOOK, VQ, PartWriter, VqTensor, assemble_vq, encode_vq, geometry, metadata
    cfg = source.config
    recipe_id = recipe['recipe_id']
    iterations = iterations or recipe.get('lloyd_iterations', 32)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f'layer-{layer:02d}'
    work.mkdir(parents=True, exist_ok=True)
    target = out_dir / f'layer-{layer:02d}-experts.gguf'
    receipt_path = out_dir / f'layer-{layer:02d}-experts.json'
    if target.exists() and receipt_path.exists():
        data = json.loads(receipt_path.read_text())
        with target.open('rb') as f:
            checksum = hashlib.file_digest(f, 'sha256').hexdigest()
        if data['sha256'] != checksum or data['k'] != k or data['recipe_id'] != recipe_id:
            raise ValueError('existing layer receipt mismatch')
        emit({'phase': 'resume_verified', 'layer': layer, 'sha256': checksum})
        return data
    if target.exists():
        raise ValueError('artifact without receipt; inspect manually')
    bits = bits_for(k)
    stats, stats_files = (None, [])
    if stats_dir is not None:
        try:
            stats, stats_files = load_stats(stats_dir, layer)
        except FileNotFoundError:
            if mode not in ('none', 'row'):
                raise
    wcfg = weighting_config(recipe)
    weighting = Weighting(mode, stats, col_power=float(col_power if col_power is not None else wcfg.get('col_power', 1.0)),
                          col_floor=float(col_floor if col_floor is not None else wcfg.get('col_floor', 0.02)),
                          col_shrink=float(wcfg.get('col_shrink', 0.0)),
                          shrink_tokens=float(shrink_tokens if shrink_tokens is not None else wcfg.get('shrink_tokens', 256)))
    sdev = weighting.s

    t0 = time.monotonic()
    lw = LayerWeights(source, layer, host=host)
    emit({'phase': 'read', 'layer': layer, 'GiB': lw.source_bytes / 2**30, 'seconds': time.monotonic() - t0})

    # 1. codebook (checkpointed per layer, K, mode)
    wd = weighting.describe()
    scope = books_scope or wcfg.get('books', 'shared')
    layout = book_layout(scope, layer)
    tag = f"k{k}-{mode}-{scope}-i{iterations}-s{wd['shrink_tokens']:g}-p{wd['col_power']:g}-f{wd['col_floor']:g}"
    book_path = work / f'{tag}-codebook.safetensors'
    t1 = time.monotonic()
    if book_path.exists():
        saved = load_file(str(book_path))
        cb8s = [saved[f'e4m3_{i}'] for i in range(len(layout))]
        training = json.loads((work / f'{tag}-training.json').read_text())
    else:
        trained, held, strata, training = train_layer(gpu, lw, layer, k, weighting, iterations, probe_iters, scope)
        cb8s = [b['e4m3'] for b in trained]
        training.update({'init': 'equidistant pool[floor(i*N/K)]', 'accumulation': 'FP64 atomics, weighted',
                         'rounding': 'FP32 -> F16 -> E4M3FN', 'empty_centroid_rule': 'retain previous (per dimension)',
                         'tie': 'lowest codeword index', 'weighting': weighting.describe(),
                         'statistics_files': stats_files, 'train_rows': TRAIN_ROWS, 'heldout_rows': HELD_ROWS,
                         'seed': SEED, 'cross_gpu_bit_identity': False})
        training['codebook_scope'] = scope
        save_file({**{f'e4m3_{i}': b['e4m3'].cpu() for i, b in enumerate(trained)},
                   **{f'f16_{i}': b['f16'].cpu() for i, b in enumerate(trained)}}, str(book_path))
        (work / f'{tag}-training.json').write_text(json.dumps(training, indent=1) + '\n')
        (work / f'{tag}-sampling.json').write_text(json.dumps({'source_revision': SOURCE_REV, 'layer': layer,
                                                              'seed': SEED, 'strata': strata}) + '\n')
        del held
    train_s = time.monotonic() - t1
    emit({'phase': 'trained', 'layer': layer, 'k': k, 'mode': mode, 'books': scope, 'seconds': train_s,
          'heldout_e4m3': training['heldout_e4m3']})

    # 2. assignment, packing, writing
    raw_books = [b.cpu().view(torch.uint8).numpy().tobytes() for b in cb8s]
    of = {}
    for i, (group, bname, bid) in enumerate(layout):
        for pi in group:
            of[PROJECTIONS[pi]] = (raw_books[i], cb8s[i].cuda().float().contiguous(), bname, bid)
    entries = [{'name': bname, 'shape': [k, 8], 'type_id': BOOK, 'payload_bytes': len(raw_books[i])}
               for i, (_, bname, _) in enumerate(layout)]
    names = [lw.name(e, p) for e in range(cfg['n_routed_experts']) for p in PROJECTIONS]
    for name in names:
        rows, cols = shapes(cfg, name.split('.')[-2])
        entries.append({'name': name, 'shape': [rows, cols], 'type_id': VQ,
                        'payload_bytes': geometry(rows, cols, k)['size']})
    meta = metadata(codebooks={str(bid): bname for _, bname, bid in layout}, part_role='trunk', extra={
        'axk2_vq.source_revision': SOURCE_REV, 'axk2_vq.layer': layer,
        'axk2_vq.recipe_id': recipe_id, 'axk2_vq.index_bits': bits,
        'axk2_vq.weighting': json.dumps({**weighting.describe(), 'gain_refit': refit, 'codebook_scope': scope},
                                        sort_keys=True)})
    partial = target.with_suffix('.gguf.partial')
    if partial.exists():
        partial.rename(partial.with_suffix(f'.failed-{time.time_ns()}'))
    t2 = time.monotonic()
    metrics, all_rows = [], []
    counts = torch.zeros(k, dtype=torch.int64, device='cuda')
    meter = ErrorMeter(sdev)
    zero_rows = 0
    with PartWriter(partial, meta, entries) as writer:
        for i, (_, bname, _) in enumerate(layout):
            writer.write(bname, raw_books[i])
        for e in range(cfg['n_routed_experts']):
            dcol2 = None
            ws = {p: lw.get(e, p) for p in PROJECTIONS}
            if sdev is not None:
                dcol2 = (ws['down_proj'].double() ** 2).sum(0)
            for pi, p in enumerate(PROJECTIONS):
                number = 3 * e + pi
                name = lw.name(e, p)
                rows, cols = shapes(cfg, p)
                w = ws[p]
                raw_book, cbf, _, book_id = of[p]
                col = weighting.col(e, p)
                ids, gains = assign_tensor(gpu, w, cbf, col, refit)
                low, plane = gpu.pack(ids, rows, cols, bits)
                restored = gpu.restore(low, plane, gains, cbf, cols, bits)
                gain_bytes = gains.cpu().numpy().tobytes()
                low_b = low.cpu().numpy().tobytes()
                plane_b = plane.cpu().numpy().tobytes() if plane is not None else None
                payload = assemble_vq(gain_bytes, low_b, plane_b, rows, cols, k, raw_book, book_id)
                if number % 97 == 0:
                    ref = encode_vq(gains.cpu().numpy(), ids.cpu().numpy(), rows, cols, k, raw_book, book_id)
                    if ref != payload:
                        raise AssertionError(f'device/numpy packing mismatch: {name}')
                writer.write(name, payload)
                cpu = VqTensor(payload, raw_book, (rows, cols), book_id)
                for row in sorted({0, rows - 1, rows // 2, (number * 7919) % rows}):
                    if not np.array_equal(restored[row:row + 1].cpu().numpy(), cpu.decode_rows(row, 1)):
                        raise AssertionError(f'CPU/GPU packed reconstruction mismatch: {name} row {row}')
                m, d2 = meter.add(e, p, w, restored, dcol2 if p != 'down_proj' else None)
                rel = torch.sqrt(d2.sum(1) / (w.double() ** 2).sum(1).clamp_min(1e-300))
                zero_rows += int((gains == 0).sum())
                counts += torch.bincount(ids.long(), minlength=k)
                all_rows.append(rel.float().cpu().numpy())
                metrics.append({'name': name, 'shape': [rows, cols], 'payload_bytes': len(payload),
                                'payload_sha256': hashlib.sha256(payload).hexdigest(), **m,
                                'row_rel_rms_max': float(rel.max()), 'worst_row': int(rel.argmax()),
                                'gain_min': float(gains.float().min()), 'gain_max': float(gains.float().max())})
                del d2, restored, low, plane, ids
            del ws
            if (e + 1) % 64 == 0:
                emit({'phase': 'convert', 'layer': layer, 'experts': e + 1, 'seconds': time.monotonic() - t2,
                      'rel_mse_so_far': meter.v['sse'] / meter.v['en']})
    convert_s = time.monotonic() - t2
    with partial.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    rows_all = np.concatenate(all_rows)
    quality = meter.summary()
    quality.update({'row_rel_rms': summaries(rows_all), 'zero_gain_rows': zero_rows,
                    'codewords_used': int((counts > 0).sum()),
                    'codeword_counts_sha256': hashlib.sha256(counts.cpu().numpy().tobytes()).hexdigest()})
    data = {'source_revision': SOURCE_REV, 'layer': layer, 'k': k, 'index_bits': bits,
            'recipe_id': recipe_id, 'mode': mode, 'gain_refit': refit, 'codebook_scope': scope,
            'weighting': {**weighting.describe(), 'gain_refit': refit},
            'path': target.name, 'bytes': partial.stat().st_size, 'sha256': checksum,
            'codebooks': [{'name': bname, 'id': bid, 'projections': [PROJECTIONS[pi] for pi in group],
                           'sha256': hashlib.sha256(raw_books[i]).hexdigest()}
                          for i, (group, bname, bid) in enumerate(layout)],
            'executed_source_sha256': source_hashes(),
            'seconds': {'read': lw.read_seconds, 'train': train_s, 'convert_and_verify': convert_s},
            'source_tensor_bytes_read': lw.source_bytes,
            'training': training,
            'cpu_gpu_restoration': 'bit-exact on first, last, middle and a rotating row of every matrix',
            'device_vs_numpy_packing': 'byte-identical on every 97th matrix',
            'quality': quality, 'tensors': metrics}
    tmp = receipt_path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    partial.rename(target)
    tmp.rename(receipt_path)
    emit({key: value for key, value in data.items() if key not in ('tensors', 'training')})
    del lw
    torch.cuda.empty_cache()
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/models/A.X-K2')
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--k', type=int)
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--gain-refit', choices=['none', 'ls'])
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--iterations', type=int)
    p.add_argument('--shrink-tokens', type=float, help='rare-expert pseudo-count (default: recipe or 256)')
    p.add_argument('--books', choices=['shared', 'split2', 'split3'], help='codebook scope (default: recipe or shared)')
    p.add_argument('--col-power', type=float, help='column importance tempering h**p (default: recipe or 1)')
    p.add_argument('--col-floor', type=float, help='column importance floor x mean (default: recipe or 0.02)')
    p.add_argument('--probe-iters', default='')
    a = p.parse_args()
    recipe = json.loads(a.recipe.read_text())
    k = a.k or recipe['layer_k'][str(a.layer)]
    wcfg = weighting_config(recipe)
    mode = a.mode or wcfg.get('mode')
    refit = a.gain_refit or wcfg.get('gain_refit')
    if mode not in MODES or refit not in ('none', 'ls'):
        raise SystemExit('the recipe has no selected weighting yet: pass --mode and --gain-refit')
    probe = [int(x) for x in a.probe_iters.split(',') if x]
    convert_layer(Source(a.source), GpuVq(), a.layer, k, recipe, a.output, a.work, mode, refit,
                  stats_dir=a.stats, probe_iters=probe or None, iterations=a.iterations, shrink_tokens=a.shrink_tokens,
                  books_scope=a.books, col_power=a.col_power, col_floor=a.col_floor)


if __name__ == '__main__':
    main()
