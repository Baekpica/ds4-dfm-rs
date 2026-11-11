#!/usr/bin/env python3
"""Choose hVQ's t and damping on a few layers: the per-token routed-output error of each
candidate setting on held-out tokens (measure_ds41.py's quantity: the teacher's own inputs
and routes, no propagation between layers), next to the iVQ blob on the same tokens.

  dump    one teacher pass over the held-out set; saves the MoE input x of the listed layers
  run     one layer: its second moments (run_hvq_ds41.py --save-hess), its held-out x, and
          every `t:damp` of --grid -> sweep-LL.jsonl (settings already there are skipped)
  choose  the setting with the lowest mean ratio to iVQ over the layers swept

This is a choice of hyperparameters, not an acceptance test: the chosen setting is applied to
all 40 layers whatever its distance from iVQ.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from axk2_weighting import Weighting
from convert_layer_ds41 import LayerExperts, SHAPES, emit, statistics
import convert_layer_ds41_hvq as cv
import ds41_hvq as hv
from ds41_hess import LayerHess
from ds41_ref import BF, Engram, EngramHash, Layer, SeqState, Source, Teacher, rope_tables
import ds41_vq_format as fmt
from ds41_vq_student import VQExperts, decode_blob
from measure_ds41 import MeasuredLayer, ROOT

PROJS = ('w1', 'w3', 'w2')


class TapLayer(Layer):
    """Layer that keeps the MoE input of its forward."""

    def moe(self, x, stats=None, routes_out=None):
        self.tap = x.clone()
        return super().moe(x, stats, routes_out)


def dump(a):
    """Teacher forward over the held-out set as measure_ds41.py runs it (one batch), layers 0..max."""
    dev = 'cuda'
    layers = cv.parse_layers(a.layers)
    teacher = Teacher(a.source, dev)
    args = teacher.a
    data = load_file(str(a.data))
    ids, off = data['input_ids'], data['offsets']
    seqs = [(int(off[i]), int(off[i + 1] - off[i])) for i in range(len(off) - 1)]
    tok = ids.to(dev)
    n = tok.numel()
    ropes = rope_tables(args, max(m for _, m in seqs) + 8, dev)
    eh = EngramHash(args, a.source, dev)
    hashes = torch.cat([eh(tok[s:s + m]) for s, m in seqs]).cpu()
    emb = teacher.src.read(['embed.weight'])['embed.weight'].to(dev)
    h = emb[tok.long()].unsqueeze(1).repeat(1, args['hc_mult'], 1).contiguous()
    del emb
    pre = torch.zeros((n, args['hc_mult']), dtype=torch.float32, device=dev)
    pre[:, 0] = 1.0
    states = [SeqState() for _ in seqs]
    a.out.mkdir(parents=True, exist_ok=True)
    for L in range(max(layers) + 1):
        t0 = time.monotonic()
        w = teacher.layer(L, teacher.host_layer(L))
        engram, rows_fn = None, None
        if L in args['engram_layer_ids']:
            li = args['engram_layer_ids'].index(L)
            engram = Engram(args, w, teacher.src.memmap(f'layers.{L}.engram.embed.weight'),
                            teacher.src.memmap(f'layers.{L}.engram.embed.scale'))
            rows_fn = lambda c0, c1, li=li, eg=engram: eg.rows(hashes[c0:c1, li])
        layer = TapLayer(args, L, w, ropes, engram)
        with torch.no_grad():
            pre = layer.forward(h, pre, seqs, states, None, rows_fn)
        if L in layers:
            save_file({'x': layer.tap.cpu().contiguous(), 'offsets': off}, str(a.out / f'x-{L:02d}.safetensors'))
        emit({'layer': L, 'phase': 'dump' if L in layers else 'forward', 'seconds': round(time.monotonic() - t0, 1)})
        del w, layer
        torch.cuda.empty_cache()


def student(cb, idx, gains):
    """expert(e) -> (w1, w3, w2) in BF16 from device index / gain tensors, as VQExperts decodes a blob."""
    def expert(e):
        return tuple((cb[idx[p][e].long()].flatten(1) * gains[p][e][:, None]).to(BF) for p in PROJS)
    return expert


def token_errors(layer, x, students):
    """-> {name: (median over tokens, energy-weighted, mean over tokens) of the routed-output relative error}."""
    layer.students, layer.acc = students, {}
    with torch.no_grad():
        layer.moe(x)
    r = layer.acc[layer.L]
    out = {}
    for name in students:
        rel = r['tok_' + name] / r['tok_energy'].clamp_min(1e-30)
        out[name] = {'median_tok': float(rel.median()), 'energy_weighted': float(r[name].sum() / r['teacher_energy'].sum()),
                     'mean_tok': float(rel.mean()), 'p90_tok': float(rel.quantile(0.9))}
    return out


def run(a):
    dev = 'cuda'
    L = a.layer
    path = a.out / f'sweep-{L:02d}.jsonl'
    a.out.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done[(row['t'], row['damping'], row.get('hess_shrink', 256.0))] = row
    grid = [tuple(float(v) for v in g.split(':')) for g in a.grid.split(',')]
    grid = [g for g in grid if (g[0], g[1], a.hess_shrink) not in done]
    if not grid and 'ivq' in {r.get('name') for r in done.values()}:
        return
    teacher = Teacher(a.source, dev)
    args = teacher.a
    host = teacher.host_layer(L)
    w = teacher.layer(L, host)
    x = load_file(str(a.dump / f'x-{L:02d}.safetensors'))['x'].to(dev)
    case_of = torch.zeros(x.shape[0], dtype=torch.long, device=dev)
    layer = MeasuredLayer(args, L, w, [None, None], None, students={}, case_of=case_of, acc={})
    st = statistics(a.d_stats, L, 'imatrix')
    hess = LayerHess.load(a.hess / f'layer-{L:02d}.safetensors', dev)
    hvq = hv.Hvq()

    def emit_row(row):
        with path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        emit(row)

    # the published blobs on the same tokens
    base = {'ivq': VQExperts(a.ivq, dev).layer(L)}
    if a.vq:
        base['vq'] = VQExperts(a.vq, dev).layer(L)
    if (0.0, 0.0, 256.0) not in done or done[(0.0, 0.0, 256.0)].get('name') != 'ivq':
        for name, err in token_errors(layer, x, base).items():
            if name == 'ivq':
                emit_row({'layer': L, 'name': 'ivq', 't': 0.0, 'damping': 0.0, 'hess_shrink': 256.0, **err})
            else:
                emit_row({'layer': L, 'name': name, 't': -1.0, 'damping': -1.0, 'hess_shrink': -1.0, **err})
    del base
    torch.cuda.empty_cache()
    for t, damp in grid:
        t0 = time.monotonic()
        var = cv.Variant(t, damp, a.hess_shrink)
        job = cv.Job(L, st, var, a.ivq, cv.HostExperts(host, L))
        job.fit(hess if t != 0 else None, hvq, measure=cv.Measure.ERRORS)
        cand = student(job.cb, {p: torch.from_numpy(job.idx[p]).to(dev) for p in PROJS},
                       {p: torch.from_numpy(job.gains[p]).to(dev).float() for p in PROJS})
        err = token_errors(layer, x, {'hvq': cand})['hvq']
        emit_row({'layer': L, 'name': 'hvq', 't': t, 'damping': damp, 'hess_shrink': a.hess_shrink, **err,
                  'output_error': job.rec['output_error'], 'changed_from_ivq': job.rec['changed_from_ivq'],
                  'identical_to_ivq': job.rec['identical_to_ivq'], 'factorization': job.rec['factorization'],
                  'fit_seconds': round(job.fit_seconds, 1), 'seconds': round(time.monotonic() - t0, 1)})
        del job, cand
        torch.cuda.empty_cache()


def choose(a):
    rows = {}
    for f in sorted(a.out.glob('sweep-*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault(r['layer'], []).append(r)
    layers = sorted(rows)
    ivq = {L: next(r for r in rows[L] if r['name'] == 'ivq') for L in layers}
    table = {}
    for L in layers:
        for r in rows[L]:
            if r['name'] != 'hvq':
                continue
            key = (r['t'], r['damping'], r.get('hess_shrink', 256.0))
            table.setdefault(key, {})[L] = r
    full = {k: v for k, v in table.items() if set(v) == set(layers)}
    score = {k: sum(v[L]['median_tok'] / ivq[L]['median_tok'] for L in layers) / len(layers) for k, v in full.items()}
    best = min(score, key=lambda k: (score[k], k))
    out = {'layers': layers, 'criterion': 'mean over the layers of median-over-tokens routed-output error / iVQ\'s',
           'ivq': {L: ivq[L]['median_tok'] for L in layers},
           'settings': [{'t': k[0], 'damping': k[1], 'hess_shrink': k[2], 'mean_ratio_to_ivq': score[k],
                         'median_tok': {L: full[k][L]['median_tok'] for L in layers},
                         'energy_weighted': {L: full[k][L]['energy_weighted'] for L in layers}}
                        for k in sorted(score, key=lambda k: score[k])],
           'chosen': {'t': best[0], 'damping': best[1], 'hess_shrink': best[2], 'mean_ratio_to_ivq': score[best]}}
    (a.out / 'choice.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['chosen']))
    for s in out['settings']:
        print(f"t {s['t']:<5} damp {s['damping']:<5} ratio {s['mean_ratio_to_ivq']:.4f}  " +
              '  '.join(f"L{L}: {s['median_tok'][L]:.4f}" for L in layers))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--source', default=ROOT)
    d.add_argument('--data', type=Path, required=True, help='held-out safetensors')
    d.add_argument('--layers', required=True)
    d.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--source', default=ROOT)
    r.add_argument('--layer', type=int, required=True)
    r.add_argument('--hess', type=Path, required=True)
    r.add_argument('--dump', type=Path, required=True)
    r.add_argument('--d-stats', type=Path, required=True)
    r.add_argument('--ivq', required=True)
    r.add_argument('--vq', default='', help='the weights-only VQ base, for the record')
    r.add_argument('--grid', required=True, help='t:damp,t:damp,...')
    r.add_argument('--hess-shrink', type=float, default=256.0)
    r.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('choose')
    c.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    {'dump': dump, 'run': run, 'choose': choose}[a.mode](a)


if __name__ == '__main__':
    main()
