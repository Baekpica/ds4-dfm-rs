#!/usr/bin/env python3
"""Check every local source file against the pinned Hub revision of skt/A.X-K2.

LFS files (the 346 shards and other large files) by SHA-256, the rest by git
blob id. `--fetch` writes the Hub's own file metadata first. Writes a
receipt; exits non-zero on any mismatch or missing file. Forked from the
Beam-1.0 source verifier.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = 'skt/A.X-K2'
REVISION = 'd0805cfd1fc647b4078c0d9f232c80d34ee0c893'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def check(args):
    root, name, row = args
    path = Path(root) / name
    if not path.is_file():
        return name, 'missing', None
    if path.stat().st_size != row['size']:
        return name, 'size', path.stat().st_size
    if row['sha256']:
        got = sha256(path)
        return name, 'ok' if got == row['sha256'] else 'sha256', got
    got = git_blob(path)
    return name, 'ok' if got == row['blob_id'] else 'blob', got


def fetch(path):
    from huggingface_hub import HfApi
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = {s.rfilename: {'size': s.size, 'sha256': s.lfs.sha256 if s.lfs else None, 'blob_id': s.blob_id}
             for s in info.siblings}
    path.write_text(json.dumps({'repo': REPO, 'revision': info.sha, 'files': files}, indent=1) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/models/A.X-K2')
    p.add_argument('--hub', type=Path, required=True)
    p.add_argument('--fetch', action='store_true', help='write --hub from the Hub API first')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    if a.fetch:
        fetch(a.hub)
    hub = json.loads(a.hub.read_text())
    if hub['revision'] != REVISION:
        raise SystemExit(f'hub metadata is for {hub["revision"]}, expected {REVISION}')
    os.nice(10)
    jobs = [(a.root, n, r) for n, r in sorted(hub['files'].items())]
    results = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for name, status, got in pool.map(check, jobs):
            results[name] = {'status': status, 'value': got}
            if status != 'ok':
                print(json.dumps({'file': name, 'status': status}), flush=True)
    bad = {n: r for n, r in results.items() if r['status'] != 'ok'}
    receipt = {'repo': hub['repo'], 'revision': hub['revision'], 'files': len(results),
               'lfs_sha256_checked': sum(1 for r in hub['files'].values() if r['sha256']),
               'mismatches': bad, 'status': 'all_files_match_hub' if not bad else 'MISMATCH', 'results': results}
    a.out.write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'results'}))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
