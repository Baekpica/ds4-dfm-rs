#!/usr/bin/env python3
"""Weighted-Lloyd variants of one A.X-K2 MoE layer, measured on held-out activations.

For every variant (weighting mode x gain refit x Lloyd iterations) the shared
codebook is trained and every vector of the layer assigned in memory exactly
as convert_layer.py would write it (stored E4M3 book). Each variant is then
judged on the held-out tokens the calibration collector dumped
(`stats/heldout/layer-LL.safetensors`: teacher FFN input, top-8 experts and
route weights): the routed-expert output

    y = sum_k w_k * D_e (silu(G_e x) * (U_e x))        (BF16 GEMMs, FP32 sums)

with the original weights (FP8 x scale -> FP32 -> BF16, the transformers
teacher) against the reconstructed weights (F32 gain x E4M3 codeword -> BF16):
relative error ||dy||^2 / ||y||^2, cosine, per expert-frequency bucket, plus
the weight-space errors of convert_layer's receipts.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from axk2_source import Source, PROJECTIONS
from axk2_vq_gpu import GpuVq
from axk2_weighting import Weighting, load_stats
from convert_layer import LayerWeights, build_pool, heldout_error, assign_tensor, ErrorMeter, shapes, emit, train_books

BF = torch.bfloat16


def expert_forward(x, g, u, d):
    """transformers AXK2Experts arithmetic: BF16 linear, act_fn(gate) * up in BF16, BF16 down."""
    gate = F.linear(x, g)
    up = F.linear(x, u)
    return F.linear(F.silu(gate) * up, d)


class Heldout:
    def __init__(self, path, max_tokens):
        t = load_file(str(path))
        n = min(len(t['ffn_in']), max_tokens) if max_tokens else len(t['ffn_in'])
        self.x = t['ffn_in'][:n].to(BF).cuda()
        self.idx = t['topk_idx'][:n].long().cuda()
        self.w = t['topk_w'][:n].float().cuda()
        self.n = n
        self.routes = {}
        for e in torch.unique(self.idx).tolist():
            tok, slot = torch.where(self.idx == e)
            self.routes[e] = (tok, slot)


def reference(lw, ho, cfg):
    """Original per-expert outputs (BF16) and the combined FP32 routed output."""
    y = torch.zeros(ho.n, cfg['hidden_size'], dtype=torch.float32, device='cuda')
    cache = {}
    for e, (tok, slot) in ho.routes.items():
        g, u, d = (lw.get(e, p).to(BF) for p in PROJECTIONS)
        out = expert_forward(ho.x[tok], g, u, d)
        cache[e] = out
        y.index_add_(0, tok, out.float() * ho.w[tok, slot, None])
    return y, cache


def evaluate(gpu, lw, ho, cfg, k, weighting, refit, iterations, layer, y0, cache, counts, scope='shared'):
    t0 = time.monotonic()
    pool, uw, aw, held, _, totals, typ = build_pool(gpu, lw, weighting, layer)
    books = train_books(gpu, pool, uw, aw, typ, k, iterations, scope)
    del pool, uw, aw, typ
    book_of = {}
    for b in books:
        for pi in b['group']:
            book_of[PROJECTIONS[pi]] = b['e4m3'].float().contiguous()
    train_s = time.monotonic() - t0
    held_err = {}
    for b in books:
        sub = [h for h in held if PROJECTIONS.index(h[1]) in b['group']]
        held_err['+'.join(PROJECTIONS[pi].split('_')[0] for pi in b['group'])] = heldout_error(gpu, sub, b['e4m3'].float().contiguous())
    del held
    hist = books[0]['history']
    y = torch.zeros_like(y0)
    meter = ErrorMeter(weighting.s)
    E = cfg['n_routed_experts']
    err_e = torch.zeros(E, dtype=torch.float64)
    en_e = torch.zeros(E, dtype=torch.float64)
    used = torch.zeros(k, dtype=torch.int64, device='cuda')
    for e in range(E):
        ws = {p: lw.get(e, p) for p in PROJECTIONS}
        dcol2 = (ws['down_proj'].double() ** 2).sum(0) if weighting.s is not None else None
        rec = {}
        for p in PROJECTIONS:
            col = weighting.col(e, p)
            cbf = book_of[p]
            ids, gains = assign_tensor(gpu, ws[p], cbf, col, refit)
            used += torch.bincount(ids.long(), minlength=k)
            r = gpu.restore_ids(ids, gains, cbf, ws[p].shape[1])
            meter.add(e, p, ws[p], r, dcol2 if p != 'down_proj' else None)
            rec[p] = r.to(BF)
            del ids, r
        if e in ho.routes:
            tok, slot = ho.routes[e]
            out = expert_forward(ho.x[tok], rec['gate_proj'], rec['up_proj'], rec['down_proj'])
            wt = ho.w[tok, slot, None]
            y.index_add_(0, tok, out.float() * wt)
            ref = cache[e].float() * wt
            err_e[e] = float(((out.float() * wt - ref).double() ** 2).sum())
            en_e[e] = float((ref.double() ** 2).sum())
        del ws, rec
    dy = (y - y0).double()
    rel = float((dy ** 2).sum() / (y0.double() ** 2).sum())
    cos_tok = F.cosine_similarity(y, y0, dim=-1)
    rel_tok = (dy ** 2).sum(-1) / (y0.double() ** 2).sum(-1).clamp_min(1e-30)
    # Buckets by calibration routing frequency (quartiles of the expert counts).
    buckets = {}
    if counts is not None:
        c = counts.double()
        qs = torch.quantile(c, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        edges = [-1.0, float(qs[0]), float(qs[1]), float(qs[2]), float(c.max()) + 1]
        for b in range(4):
            sel = (c > edges[b]) & (c <= edges[b + 1])
            en = float(en_e[sel].sum())
            buckets[f'q{b + 1}'] = {'experts': int(sel.sum()), 'count_range': [edges[b], edges[b + 1]],
                                    'heldout_routes_energy': en,
                                    'rel_err': float(err_e[sel].sum()) / en if en > 0 else None}
    out = {'mode': weighting.mode, 'gain_refit': refit, 'iterations': iterations, 'k': k, 'books': scope,
           'book_pool_vectors': [b['pool_vectors'] for b in books],
           'output_rel_err': rel, 'output_rel_rms': rel ** .5,
           'output_cosine_global': float(F.cosine_similarity(y.flatten(), y0.flatten(), dim=0)),
           'output_cosine_token_mean': float(cos_tok.mean()), 'output_cosine_token_p05': float(torch.quantile(cos_tok.float(), 0.05)),
           'output_rel_err_token_mean': float(rel_tok.mean()), 'output_rel_err_token_p95': float(torch.quantile(rel_tok.float(), 0.95)),
           'per_expert_sum_rel_err': float(err_e.sum() / en_e.sum()),
           'buckets_by_calibration_count': buckets, 'weight': meter.summary(), 'heldout_rows': held_err,
           'codewords_used_any_book': int((used > 0).sum()),
           'final_weighted_distortion': [b['history'][-1]['weighted_distortion'] for b in books],
           'pool_weight_totals': totals, 'seconds': {'train': train_s, 'total': time.monotonic() - t0}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/models/A.X-K2')
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--k', type=int, help='default: 1024 on MoE layers 1-7, else 512 (recipe)')
    p.add_argument('--variants', default='none:none,row:none,imatrix:none,imatrix_freq:none,output:none,'
                                         'none:ls,imatrix:ls,imatrix_freq:ls,output:ls,'
                                         'imatrix_freq:ls:32:0.5,output:ls:32:0.5',
                   help='comma list of mode:refit[:iterations[:col_power[:col_floor[:shrink_tokens[:books]]]]]')
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    p.add_argument('--heldout', type=Path, help='default: STATS/heldout/layer-LL.safetensors')
    p.add_argument('--max-tokens', type=int, default=65536)
    p.add_argument('--shrink-tokens', type=float, default=256.0)
    p.add_argument('--books', help='codebook scopes to cross with every variant: shared,split2,split3')
    p.add_argument('--out', type=Path, default=Path('/workspace/axk2/work/measure'))
    a = p.parse_args()
    L = a.layer
    k = a.k or (1024 if L <= 7 else 512)
    src = Source(a.source)
    cfg = src.config
    gpu = GpuVq()
    stats, files = load_stats(a.stats, L)
    ho = Heldout(a.heldout or a.stats / 'heldout' / f'layer-{L:02d}.safetensors', a.max_tokens)
    t0 = time.monotonic()
    lw = LayerWeights(src, L)
    emit({'phase': 'loaded', 'layer': L, 'k': k, 'heldout_tokens': ho.n, 'seconds': time.monotonic() - t0})
    y0, cache = reference(lw, ho, cfg)
    results = {'layer': L, 'k': k, 'heldout_tokens': ho.n, 'statistics_files': files,
               'calibration_tokens': float(stats['tokens'].item()),
               'expert_count': {'min': int(stats['count'].min()), 'median': float(stats['count'].double().median()),
                                'max': int(stats['count'].max()), 'zero': int((stats['count'] == 0).sum())},
               'variants': []}
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f'layer-{L:02d}.json'
    specs = []
    for spec in a.variants.split(','):
        parts = spec.split(':')
        if len(parts) > 6 or not a.books:
            specs.append(spec)
            continue
        defaults = [None, 'none', str(a.iterations), '1.0', '0.02', str(a.shrink_tokens)]
        full = parts + defaults[len(parts):]
        for scope in a.books.split(','):
            specs.append(':'.join(full + [scope]))
    for spec in specs:
        parts = spec.split(':')
        mode, refit = parts[0], parts[1] if len(parts) > 1 else 'none'
        iters = int(parts[2]) if len(parts) > 2 else a.iterations
        power = float(parts[3]) if len(parts) > 3 else 1.0
        floor = float(parts[4]) if len(parts) > 4 else 0.02
        shrink = float(parts[5]) if len(parts) > 5 else a.shrink_tokens
        scope = parts[6] if len(parts) > 6 else 'shared'
        w = Weighting(mode, stats, col_power=power, col_floor=floor, shrink_tokens=shrink)
        r = evaluate(gpu, lw, ho, cfg, k, w, refit, iters, L, y0, cache, stats['count'], scope)
        results['variants'].append(r)
        r['weighting'] = w.describe()
        emit({'phase': 'variant', 'layer': L, 'mode': mode, 'refit': refit, 'iters': iters, 'power': power, 'floor': floor,
              'shrink': shrink, 'books': scope,
              'output_rel_err': r['output_rel_err'], 'cos': r['output_cosine_token_mean'],
              'tok_rel_err': r['output_rel_err_token_mean'],
              'weight_rel_mse': r['weight']['rel_mse'], 'imatrix_rel_err': r['weight'].get('imatrix_rel_err'),
              'seconds': r['seconds']['total']})
        path.write_text(json.dumps(results, indent=1) + '\n')
        torch.cuda.empty_cache()
    base = next((v for v in results['variants'] if v['mode'] == 'none' and v['gain_refit'] == 'none'
                 and v['books'] == 'shared'), None)
    if base:
        for v in results['variants']:
            v['output_rel_err_vs_baseline'] = v['output_rel_err'] / base['output_rel_err'] - 1
        path.write_text(json.dumps(results, indent=1) + '\n')
    emit({'phase': 'done', 'layer': L, 'path': str(path)})


if __name__ == '__main__':
    main()
