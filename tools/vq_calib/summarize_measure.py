#!/usr/bin/env python3
"""Table of held-out routed-output relative error per layer and variant (work/measure/layer-LL.json)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/axk2/work/measure')
files = sorted(root.glob('layer-*.json'))
rows, layers = {}, []
for f in files:
    d = json.loads(f.read_text())
    L = d['layer']
    layers.append(L)
    for v in d['variants']:
        w = v.get('weighting', {})
        key = (v['mode'], v['gain_refit'], v.get('books', 'shared'), w.get('col_power', 1.0), w.get('shrink_tokens', 256.0))
        rows.setdefault(key, {})[L] = v
base_key = ('none', 'none', 'shared', 1.0, 256.0)
print('| mode | refit | books | power | shrink | ' + ' | '.join(f'L{L} out-err (vs base)' for L in layers) + ' | mean Δ |')
print('|---|---|---|---:|---:|' + '---:|' * len(layers) + '---:|')
for key, per in rows.items():
    cells, deltas = [], []
    for L in layers:
        v = per.get(L)
        b = rows.get(base_key, {}).get(L)
        if v is None:
            cells.append('—')
            continue
        d = v['output_rel_err'] / b['output_rel_err'] - 1 if b else float('nan')
        deltas.append(d)
        tok = v.get('output_rel_err_token_mean')
        cells.append(f"{v['output_rel_err']:.4f} ({d:+.1%})" + (f" t{tok:.3f}" if tok is not None else ''))
    mean = sum(deltas) / len(deltas) if deltas else float('nan')
    print(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]:g} | {key[4]:g} | ' + ' | '.join(cells) + f' | {mean:+.1%} |')
