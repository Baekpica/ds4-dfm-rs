#!/usr/bin/env python3
"""Upload a verified A.X-K2 release directory to the public model repo and read it back.

One commit for everything. Afterwards every file listed in SHA256SUMS is
checked against the Hub: LFS files by their SHA-256, small files by their git
blob id (computed locally). Anonymous HEAD requests confirm public access.
Forked from the Beam-1.0 publisher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='Baekpica/A.X-K2-VQ-GGUF')
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--message', required=True)
    p.add_argument('--skip-upload', action='store_true')
    a = p.parse_args()
    api = HfApi()
    if not a.skip_upload:
        subprocess.run(['hf', 'upload', a.repo, str(a.release), '.', '--type', 'model',
                        '--commit-message', a.message], check=True)
    info = api.model_info(a.repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    want = {}
    for line in (a.release / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split(None, 1)
        want[name.strip()] = h
    ok, bad = 0, []
    for name, h in sorted(want.items()):
        s = remote.get(name)
        if s is None:
            bad.append(('missing', name))
            continue
        if s.lfs:
            good = s.lfs.sha256 == h and s.size == (a.release / name).stat().st_size
        else:
            good = s.blob_id == git_blob(a.release / name)
        if good:
            ok += 1
        else:
            bad.append(('mismatch', name))
    for name in ('SHA256SUMS', 'files.json'):
        s = remote.get(name)
        if s is None or (not s.lfs and s.blob_id != git_blob(a.release / name)):
            bad.append(('index', name))
    heads = {}
    for name in ('README.md', 'release.json', 'layer-01-experts.gguf', 'layer-60-experts.gguf',
                 'body-nonexpert.gguf', 'embed-tokens-bf16.gguf'):
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-I', '-L', '-w', '%{http_code}',
                            f'https://huggingface.co/{a.repo}/resolve/{info.sha}/{name}'],
                           capture_output=True, text=True, env={'PATH': '/usr/bin:/bin'})
        heads[name] = r.stdout
    out = {'repo': a.repo, 'commit': info.sha, 'private': info.private, 'remote_files': len(remote),
           'checked': len(want), 'ok': ok, 'bad': bad, 'anonymous_head': heads}
    print(json.dumps(out, indent=1))
    sys.exit(1 if bad or info.private or any(v != '200' for v in heads.values()) else 0)


if __name__ == '__main__':
    main()
