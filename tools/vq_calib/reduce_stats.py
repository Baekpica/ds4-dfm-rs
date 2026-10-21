#!/usr/bin/env python3
"""Sum the per-rank statistics of collect_stats.py into stats/layer-LL.safetensors and head.safetensors.

Only one finished run is reduced: every rank-* directory must carry a shard.json
of the same calibration set and world size, the ranks must be exactly
0..world-1, each must have finished (DONE), every layer file must exist in all
of them, and each reduced file's token count must equal the shards' total.
Files stamped by the collector must name their own rank's run.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def run_stamp(shard):
    return {'calib': str(shard['calib']), 'world': str(shard['world']), 'rank': str(shard['rank'])}


def file_stamp(path):
    with safe_open(str(path), 'pt') as f:
        meta = f.metadata() or {}
    stamp = {k: meta[k] for k in ('calib', 'world', 'rank') if k in meta}
    return stamp or None


def run_ranks(stats):
    """-> rank directories 0..world-1 of one finished run and their shard records."""
    found = {}
    for d in sorted(x for x in stats.glob('rank-*') if x.is_dir()):
        f = d / 'shard.json'
        if not f.exists():
            raise SystemExit(f'{d.name}: no shard.json (not written by collect_stats.py)')
        shard = json.loads(f.read_text())
        if d.name != f"rank-{shard['rank']}":
            raise SystemExit(f'{d.name}: shard.json says rank {shard["rank"]}')
        if not (d / 'DONE').exists():
            raise SystemExit(f'{d.name}: not finished (no DONE)')
        found[shard['rank']] = (d, shard)
    if not found:
        raise SystemExit(f'no rank directories under {stats}')
    runs = {(s['calib'], s['world']) for _, s in found.values()}
    if len(runs) != 1:
        raise SystemExit(f'rank directories from different runs (calib, world): {sorted(runs)}')
    world = next(iter(runs))[1]
    if set(found) != set(range(world)):
        raise SystemExit(f'ranks {sorted(found)} are not exactly 0..{world - 1}; remove stale rank directories')
    return [found[r][0] for r in range(world)], [found[r][1] for r in range(world)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    a = p.parse_args()
    ranks, shards = run_ranks(a.stats)
    total_tokens = sum(s['tokens'] for s in shards)
    names = sorted({f.name for d in ranks for f in d.glob('*.safetensors')})
    summary = {'ranks': [d.name for d in ranks], 'calib': shards[0]['calib'], 'world': shards[0]['world'],
               'calib_tokens': total_tokens, 'layers': {}, 'unstamped_files': 0}
    for name in names:
        missing = [d.name for d in ranks if not (d / name).exists()]
        if missing:
            raise SystemExit(f'{name}: missing in {missing}')
        parts = []
        for d, sh in zip(ranks, shards):
            stamp = file_stamp(d / name)
            if stamp is None:
                summary['unstamped_files'] += 1
            elif stamp != run_stamp(sh):
                raise SystemExit(f'{d.name}/{name} was written by another run: {stamp} vs {run_stamp(sh)}')
            parts.append(load_file(str(d / name)))
        out = {k: sum(p[k] for p in parts) for k in parts[0]}
        if int(out['tokens'].item()) != total_tokens:
            raise SystemExit(f'{name}: {int(out["tokens"].item())} tokens, the shards hold {total_tokens}')
        save_file(out, str(a.stats / name))
        row = {'tokens': float(out['tokens'].item())}
        if 'count' in out:
            c = out['count'].double()
            row.update({'count_min': int(c.min()), 'count_p01': float(c.quantile(.01)), 'count_median': float(c.median()),
                        'count_max': int(c.max()), 'unused': int((c == 0).sum()),
                        'w2_share_top8': float(torch.sort(out['w2'], descending=True)[0][:8].sum() / out['w2'].sum())})
        summary['layers'][name] = row
    (a.stats / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'files': len(names), 'ranks': len(ranks), 'calib_tokens': total_tokens,
                      'unstamped_files': summary['unstamped_files']}))


if __name__ == '__main__':
    main()
