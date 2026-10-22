#!/usr/bin/env python3
"""Sum the per-rank statistics of collect_stats.py / run_ds41.py into stats/layer-LL.safetensors and head.safetensors.

Only one finished run is reduced: every rank-* directory must carry a shard.json
of the same calibration set, world size and run id, the ranks must be exactly
0..world-1, each must have finished (DONE), every file the run wrote must exist
in all of them, and each reduced file's token count must equal the shards'
total. The files reduced are exactly the ones DONE lists (every rank must
agree); other files in the rank directories are left out and listed in
summary.json. Files stamped by the collector must name their own rank's run.
Rank directories written before run ids (no run_id in shard.json) are still
reduced, and their files are counted as unverified in summary.json.

summary.json records the run (its id and record) and the SHA-256 of every
reduced file; `calibration_run` checks a converter's statistics against them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run_id(run):
    """SHA-256 of a collection run's record (collector, calibration content, world, layer list, token cap,
    source): the same on every rank; stamped into shard.json, DONE and every rank file."""
    return hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()


def run_stamp(shard):
    stamp = {'calib': str(shard['calib']), 'world': str(shard['world']), 'rank': str(shard['rank'])}
    if 'run_id' in shard:
        stamp['run_id'] = shard['run_id']
    return stamp


def file_stamp(path):
    with safe_open(str(path), 'pt') as f:
        meta = f.metadata() or {}
    stamp = {k: meta[k] for k in ('calib', 'world', 'rank', 'run_id') if k in meta}
    return stamp or None


def run_ranks(stats):
    """-> rank directories 0..world-1 of one finished run, their shard records and the files the run wrote."""
    found = {}
    for d in sorted(x for x in stats.glob('rank-*') if x.is_dir()):
        f = d / 'shard.json'
        if not f.exists():
            raise SystemExit(f'{d.name}: no shard.json (not written by a collector)')
        shard = json.loads(f.read_text())
        if d.name != f"rank-{shard['rank']}":
            raise SystemExit(f'{d.name}: shard.json says rank {shard["rank"]}')
        if not (d / 'DONE').exists():
            raise SystemExit(f'{d.name}: not finished (no DONE)')
        done = json.loads((d / 'DONE').read_text())
        if 'run_id' in shard and (run_id(shard['run']) != shard['run_id'] or done.get('run_id') != shard['run_id']):
            raise SystemExit(f'{d.name}: shard.json and DONE are not from one run '
                             f'({shard["run_id"]} vs {done.get("run_id")})')
        found[shard['rank']] = (d, shard, done)
    if not found:
        raise SystemExit(f'no rank directories under {stats}')
    runs = {(s['calib'], s['world'], s.get('run_id')) for _, s, _ in found.values()}
    if len(runs) != 1:
        raise SystemExit(f'rank directories from different runs (calib, world, run id): {sorted(runs, key=str)}')
    world = next(iter(runs))[1]
    if set(found) != set(range(world)):
        raise SystemExit(f'ranks {sorted(found)} are not exactly 0..{world - 1}; remove stale rank directories')
    lists = {json.dumps(done.get('files', done.get('layers'))) for _, _, done in found.values()}
    if len(lists) != 1:
        raise SystemExit(f'ranks finished different layer lists: {sorted(lists)}')
    done = found[0][2]
    if 'files' in done:
        names = list(done['files'])
    else:  # older collectors list layers only (a resumed rank only those after its checkpoint)
        names = [f'layer-{L:02d}.safetensors' for L in done['layers']]
        if all((d / 'head.safetensors').exists() for d, _, _ in found.values()):
            names.append('head.safetensors')
    return [found[r][0] for r in range(world)], [found[r][1] for r in range(world)], names


def run_record(stats_dir, files):
    """-> (run id, run record) of the collection run the statistics `files` (as load_stats returned them, or the
    merged files a non-expert conversion read) come from. Merged files must be the ones summary.json records;
    rank files must be exactly the ranks of one run, each stamped by it. (None, None) for statistics written
    before run ids (unverified), ('none', None) without statistics."""
    if not files:
        return 'none', None
    stats_dir = Path(stats_dir)
    paths = [Path(f) for f in files]
    if all(p.parent == stats_dir for p in paths):
        sfile = stats_dir / 'summary.json'
        summary = json.loads(sfile.read_text()) if sfile.exists() else {}
        rid = summary.get('run_id')
        if rid is None:
            return None, None
        for p in paths:
            row = summary['layers'].get(p.name) or {}
            if row.get('sha256') != sha256_file(p):
                raise ValueError(f'{p} is not the file run {rid} reduced (summary.json): stale statistics')
        return rid, summary['run']
    shards = []
    for p in paths:
        f = p.parent / 'shard.json'
        shard = json.loads(f.read_text()) if f.exists() else {}
        if 'run_id' in shard and file_stamp(p) != run_stamp(shard):
            raise ValueError(f'{p} was not written by run {shard["run_id"]}: stale rank file')
        shards.append(shard)
    ids = {s.get('run_id') for s in shards}
    if ids == {None}:
        return None, None
    ranks = sorted(s.get('rank', -1) for s in shards)
    if len(ids) != 1 or ranks != list(range(shards[0]['world'])):
        raise ValueError(f'{files} are not exactly the ranks of one run (run ids {sorted(ids, key=str)}, ranks {ranks})')
    return shards[0]['run_id'], shards[0]['run']


def calibration_run(stats_dir, files):
    """-> (run id, calibration set SHA-256) of the run `files` come from (run_record)."""
    rid, run = run_record(stats_dir, files)
    return rid, (run or {}).get('calib_sha256')


def hub_revision(root):
    """Commit of a checkpoint fetched with `hf download --local-dir` (its download metadata), or None."""
    try:
        return (Path(root) / '.cache/huggingface/download/config.json.metadata').read_text().split()[0]
    except (OSError, IndexError):
        return None


def source_fingerprint(root):
    """-> (SHA-256, validated Hub revision or None) of a checkpoint's weight files, without hashing their bytes:
    every shard's name, size, modification time and safetensors header, plus the commit and etag `hf download`
    recorded for it when that record is not older than the file (huggingface_hub's own rule: a file rewritten
    after its download no longer matches its record). Replacing or rewriting a shard changes the SHA-256; the
    revision is returned only when every shard carries a valid record of one commit."""
    root = Path(root)
    rows, commits = [], set()
    for f in sorted(root.glob('*.safetensors')):
        st = f.stat()
        with f.open('rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            header = hashlib.sha256(fh.read(n)).hexdigest()
        rec = None
        try:
            commit, etag, ts = (root / '.cache/huggingface/download' / f'{f.name}.metadata').read_text().split()[:3]
            if float(ts) >= st.st_mtime:
                rec = [commit, etag]
                commits.add(commit)
        except (OSError, ValueError):
            pass
        rows.append([f.name, st.st_size, st.st_mtime_ns, header, rec])
    revision = commits.pop() if len(commits) == 1 and rows and all(r[4] for r in rows) else None
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest(), revision


def check_source(run, root, revision=None):
    """Statistics must come from the checkpoint being converted: the run's recorded source directory and, where
    both sides know it, its revision. A run recorded before run ids (None) cannot be checked."""
    if not run:
        return
    here = {'source': os.path.realpath(root), 'source_revision': revision}
    diff = {k: (run.get(k), v) for k, v in here.items() if run.get(k) is not None and v is not None and run[k] != v}
    if diff:
        raise ValueError(f'the statistics come from another checkpoint (statistics run, this conversion): {diff}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    a = p.parse_args()
    ranks, shards, names = run_ranks(a.stats)
    total_tokens = sum(s['tokens'] for s in shards)
    rid = shards[0].get('run_id')
    ignored = sorted(f'{d.name}/{f.name}' for d in ranks for f in d.glob('*.safetensors') if f.name not in names)
    summary = {'ranks': [d.name for d in ranks], 'calib': shards[0]['calib'], 'world': shards[0]['world'],
               'calib_tokens': total_tokens, 'run_id': rid, 'run': shards[0].get('run'), 'files': names,
               'layers': {}, 'unstamped_files': 0, 'unverified_files': 0, 'ignored_files': ignored}
    for name in names:
        missing = [d.name for d in ranks if not (d / name).exists()]
        if missing:
            raise SystemExit(f'{name}: missing in {missing}')
        parts = []
        for d, sh in zip(ranks, shards):
            stamp = file_stamp(d / name)
            if stamp != run_stamp(sh):
                if stamp is not None or rid is not None:
                    raise SystemExit(f'{d.name}/{name} was written by another run: {stamp} vs {run_stamp(sh)}')
                summary['unstamped_files'] += 1
            if rid is None:
                summary['unverified_files'] += 1
            parts.append(load_file(str(d / name)))
        out = {k: sum(p[k] for p in parts) for k in parts[0]}
        if int(out['tokens'].item()) != total_tokens:
            raise SystemExit(f'{name}: {int(out["tokens"].item())} tokens, the shards hold {total_tokens}')
        # atomic, so a converter checking for the merged file never reads a partial one
        tmp = a.stats / (name + '.tmp')
        save_file(out, str(tmp))
        digest = sha256_file(tmp)
        tmp.replace(a.stats / name)
        row = {'tokens': float(out['tokens'].item()), 'sha256': digest}
        if 'count' in out:
            c = out['count'].double()
            row.update({'count_min': int(c.min()), 'count_p01': float(c.quantile(.01)), 'count_median': float(c.median()),
                        'count_max': int(c.max()), 'unused': int((c == 0).sum()),
                        'w2_share_top8': float(torch.sort(out['w2'], descending=True)[0][:8].sum() / out['w2'].sum())})
        summary['layers'][name] = row
    # merged files of earlier reductions that this run did not write (calibration_run refuses them)
    summary['outputs_not_in_run'] = sorted(f.name for f in a.stats.glob('*.safetensors') if f.name not in names)
    (a.stats / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'files': len(names), 'ranks': len(ranks), 'calib_tokens': total_tokens, 'run_id': rid,
                      'unstamped_files': summary['unstamped_files'], 'unverified_files': summary['unverified_files'],
                      'ignored_files': len(ignored), 'outputs_not_in_run': len(summary['outputs_not_in_run'])}))


if __name__ == '__main__':
    main()
