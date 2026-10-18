#!/usr/bin/env python3
"""usage: ling31_release_check.py DIR — check a Ling-3.1 VQ release's indexes.

release.json, recipe.json, tensor-manifest.json, files.json and SHA256SUMS
must agree with each other and with the files on disk. Text and JSON files
are rehashed; GGUF parts are only sized here (the host hashes them on load)."""
import hashlib
import json
import sys
from pathlib import Path

INDEXES = {'files.json', 'SHA256SUMS'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def check(root):
    errors = []
    release = json.loads((root / 'release.json').read_text())
    recipe = json.loads((root / 'recipe.json').read_text())
    manifest_id = json.loads((root / 'tensor-manifest.json').read_text())['recipe_id']
    if not release['recipe_id'] == recipe['recipe_id'] == manifest_id:
        errors.append('recipe ids differ')
    if release['recipe_sha256'] != digest(root / 'recipe.json'):
        errors.append('release.json recipe_sha256 is not recipe.json')

    files = json.loads((root / 'files.json').read_text())['files']
    listed = {e['path']: e for e in files}
    if [e['path'] for e in files] != sorted(listed):
        errors.append('files.json is not in path order')
    for part in release['parts']:
        entry = listed.get(part['path'], {})
        if (entry.get('bytes'), entry.get('sha256')) != (part['bytes'], part['sha256']):
            errors.append('release part differs from files.json: ' + part['path'])

    # Every file but the indexes and the HF download cache is listed, as it is on disk.
    present = {p.relative_to(root).as_posix() for p in root.rglob('*')
               if p.is_file() and '.cache' not in p.relative_to(root).parts}
    if present - INDEXES != set(listed):
        errors.append('files.json lists %s, disk has %s' % (sorted(set(listed) - present), sorted(present - INDEXES - set(listed))))
    for name, entry in listed.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            errors.append('size differs: ' + name)
        elif not name.endswith('.gguf') and digest(path) != entry['sha256']:
            errors.append('sha256 differs: ' + name)

    sums = {name: e['sha256'] for name, e in listed.items()}
    sums['files.json'] = digest(root / 'files.json')
    if (root / 'SHA256SUMS').read_text() != ''.join('%s  %s\n' % (sums[n], n) for n in sorted(sums)):
        errors.append('SHA256SUMS differs from files.json')
    return errors


if __name__ == '__main__':
    problems = check(Path(sys.argv[1]))
    print('\n'.join(problems) or 'release indexes agree')
    sys.exit(1 if problems else 0)
