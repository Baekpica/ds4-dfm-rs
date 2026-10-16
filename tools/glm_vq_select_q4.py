#!/usr/bin/env python3
"""Select existing, original-derived Q4_K tensors without requantization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import glm_vq_q4k as fmt


def _copy(file, offset, size, out=None):
    file.seek(offset)
    digest = hashlib.sha256()
    while size:
        chunk = file.read(min(size, 1024 * 1024))
        if not chunk:
            raise ValueError('truncated tensor payload')
        digest.update(chunk)
        if out is not None:
            out.write(chunk)
        size -= len(chunk)
    return digest.hexdigest()


def _rewrite(source, donor, output, records, imported, selected):
    with source.open('rb') as src, donor.open('rb') as old:
        meta, entries, base = fmt.read_header(src)
        _, old_entries, old_base = fmt.read_header(old)
        old_entries = {entry['name']: entry for entry in old_entries}
        links = json.loads(meta.get('glm_vq.fp8_scales', (8, '{}'))[1])
        offset = 0
        for entry in entries:
            name = entry['name']
            record = records[name]
            if list(reversed(entry['dims'])) != record['stored_shape'] or entry['typ'] != record['type_id']:
                raise ValueError('source header/manifest mismatch')
            entry['size'] = record['payload_bytes']
            if name in selected:
                replacement = imported[name]
                old_entry = old_entries[name]
                if old_entry['typ'] != fmt.Q4_K or old_entry['dims'] != entry['dims']:
                    raise ValueError('donor header/manifest mismatch')
                if name in links and records[links[name]]['payload_sha256'] != imported[links[name]]['payload_sha256']:
                    raise ValueError('donor used different FP8 scales')
                entry['typ'] = fmt.Q4_K
                entry['size'] = replacement['payload_bytes']
            entry['out_offset'] = fmt.align(offset)
            offset = entry['out_offset'] + entry['size']
        meta['glm_vq.native_policy'] = (8, fmt.POLICY)
        meta['glm_vq.fp8_scales'] = (8, json.dumps({k:v for k,v in links.items() if k not in selected}, sort_keys=True))
        with output.open('xb') as out:
            header = fmt.header(meta, entries)
            out.write(header)
            for entry in entries:
                name = entry['name']
                record = records[name]
                out.write(bytes(len(header) + entry['out_offset'] - out.tell()))
                if name in selected:
                    if _copy(src, base + entry['offset'], record['payload_bytes']) != record['payload_sha256']:
                        raise ValueError('source tensor checksum mismatch')
                    replacement = imported[name]
                    actual = _copy(old, old_base + old_entries[name]['offset'], entry['size'], out)
                    if actual != replacement['payload_sha256']:
                        raise ValueError('donor tensor checksum mismatch')
                    record.clear()
                    record.update(replacement)
                elif _copy(src, base + entry['offset'], entry['size'], out) != record['payload_sha256']:
                    raise ValueError('unchanged tensor checksum mismatch')
            out.flush()
            os.fsync(out.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('donor', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--select', type=Path, required=True, help='JSON array of exact tensor names')
    args = parser.parse_args()
    started = time.monotonic()
    source, donor, output = (p.resolve() for p in (args.source, args.donor, args.output))
    manifest = json.loads((source / 'tensor-manifest.json').read_text())
    other = json.loads((donor / 'tensor-manifest.json').read_text())
    files = json.loads((source / 'files.json').read_text())
    records = {r['name']:r for r in manifest['tensors']}
    imported = {r['name']:r for r in other['tensors']}
    names = json.loads(args.select.read_text())
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        parser.error('--select must contain a nonempty tensor-name array')
    selected = set(names)
    if len(names) != len(selected) or manifest['source_revision'] != other['source_revision']:
        parser.error('duplicate selection or different source revision')
    for name in selected:
        if not fmt.target(name) or name not in records or name not in imported:
            parser.error(f'not an eligible nonexpert tensor: {name}')
        parent, child = records[name], imported[name]
        if parent['type_id'] not in (fmt.BF16, fmt.FP8) or child['type_id'] != fmt.Q4_K:
            parser.error(f'selection must replace original precision with Q4_K: {name}')
        if child.get('parent_payload_sha256') != parent['payload_sha256']:
            parser.error(f'Q4_K parent differs: {name}')
        if any(parent[k] != child[k] for k in ['part', 'shape', 'stored_shape', 'source_dtype']):
            parser.error(f'Q4_K tensor geometry/provenance differs: {name}')
    for entry in files['files']:
        if Path(entry['path']).name != entry['path']:
            parser.error('release paths must be filenames')
    # A new report must not overwrite a hardlink to the source's report.
    files['files'] = [entry for entry in files['files'] if entry['path'] != 'q4k-selection.json']
    parts = {records[name]['part'] for name in selected}
    report = dict(recipe=fmt.POLICY, selected=sorted(selected), changed_count=len(selected),
        parent_manifest_sha256=fmt.digest(source / 'tensor-manifest.json'),
        donor_manifest_sha256=fmt.digest(donor / 'tensor-manifest.json'),
        selector_sha256=fmt.digest(Path(__file__)), scope='Reuse original-derived Q4_K payloads; no requantization')
    output.mkdir(exist_ok=False)
    for entry in files['files']:
        name = entry['path']
        if name == 'tensor-manifest.json':
            continue
        if name == 'README.md':
            (output / name).write_text('# GLM VQ selective Q4_K candidate\n\n'
                'Local candidate; speed and quality adoption pending. See q4k-selection.json.\n')
        elif name in parts:
            _rewrite(source / name, donor / name, output / name, records, imported, selected)
            print(json.dumps(dict(part=name, seconds=time.monotonic()-started)), flush=True)
        else:
            os.link(source / name, output / name)
            continue
        entry.update(bytes=(output / name).stat().st_size, sha256=fmt.digest(output / name))
    # Keep the base quantizer provenance and record this distinct selection.
    recipe = manifest.setdefault('nonexpert_recipe', dict(name=fmt.POLICY))
    recipe['kept_tensors'] = sorted(n for n,r in records.items() if fmt.target(n) and r['type_id'] != fmt.Q4_K)
    recipe['selection'] = report.copy()
    (output / 'tensor-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for entry in files['files']:
        if entry['path'] == 'tensor-manifest.json':
            entry.update(bytes=(output / entry['path']).stat().st_size, sha256=fmt.digest(output / entry['path']))
    report.update(manifest_sha256=fmt.digest(output / 'tensor-manifest.json'),
        total_q4=sum(r['type_id']==fmt.Q4_K for r in records.values()), elapsed_seconds=time.monotonic()-started)
    (output / 'q4k-selection.json').write_text(json.dumps(report, indent=2)+'\n')
    files['files'].append(dict(path='q4k-selection.json', bytes=(output / 'q4k-selection.json').stat().st_size,
        sha256=fmt.digest(output / 'q4k-selection.json')))
    (output / 'files.json').write_text(json.dumps(files, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
