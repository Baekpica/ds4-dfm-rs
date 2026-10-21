#!/usr/bin/env python3
"""Sum the per-rank statistics of collect_stats.py into stats/layer-LL.safetensors and head.safetensors."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    a = p.parse_args()
    ranks = sorted(d for d in a.stats.glob('rank-*') if d.is_dir())
    names = sorted({f.name for d in ranks for f in d.glob('*.safetensors')})
    summary = {'ranks': [d.name for d in ranks], 'layers': {}}
    for name in names:
        parts = [load_file(str(d / name)) for d in ranks if (d / name).exists()]
        if len(parts) != len(ranks):
            raise SystemExit(f'{name}: {len(parts)} of {len(ranks)} ranks')
        out = {k: sum(p[k] for p in parts) for k in parts[0]}
        save_file(out, str(a.stats / name))
        row = {'tokens': float(out['tokens'].item())}
        if 'count' in out:
            c = out['count'].double()
            row.update({'count_min': int(c.min()), 'count_p01': float(c.quantile(.01)), 'count_median': float(c.median()),
                        'count_max': int(c.max()), 'unused': int((c == 0).sum()),
                        'w2_share_top8': float(torch.sort(out['w2'], descending=True)[0][:8].sum() / out['w2'].sum())})
        summary['layers'][name] = row
    (a.stats / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'files': len(names), 'ranks': len(ranks)}))


if __name__ == '__main__':
    main()
