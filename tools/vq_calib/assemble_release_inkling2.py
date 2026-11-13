#!/usr/bin/env python3
"""Inkling-2-Small VQ release directory: parts + release.json + tensor-manifest.json + docs + SHA256SUMS.

Every part is re-verified with the strict reader against its conversion receipt (file and payload
SHA-256) before it is linked into the release; the tensor manifest is built from those receipts and the
source mapping of the non-expert receipt must account for every source tensor exactly once.

    assemble_release_inkling2.py --parts /work/inkling2/parts --docs /work/inkling2/release-docs \
        --tools /ds4-dfm-rs/tools/vq_calib --out /work/inkling2/release
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import inkling2_vq_format as fmt

NATIVE = {'body-nonexpert.gguf': ('trunk', 'resident', True),
          'embed-tokens-bf16.gguf': ('embedding', 'ssd-rows', True),
          'vision-bf16.gguf': ('vision', 'mmap-on-demand', False),
          'mtp-q8_0.gguf': ('drafter', 'resident', False)}
TOOLS = ['inkling2_ref.py', 'inkling2_selftest.py', 'inkling2_vision.py', 'inkling2_eval.py', 'inkling2_vq_format.py',
         'run_hvq_inkling2.py', 'convert_t0_inkling2.py', 'assemble_experts_inkling2.py', 'convert_nonexpert_inkling2.py',
         'assemble_release_inkling2.py', 'build_calib_inkling2.py', 'ds41_hvq.py', 'ds41_hess.py', 'axk2_vq_gpu.py',
         'axk2_weighting.py', 'cuda/ds41_hvq.cu', 'cuda/axk2_vq_convert.cu', 'inkling2_budget.py']
KV_CONTRACT = {
    'global_layers': [l for l in range(48) if (l + 1) % 6 == 0],
    'global_kv_format': 'nvfp4: E2M1 values, an E4M3 scale per 16 values = max(amax / 6, 2^-9), global scale 1 '
                        '(the MiniMax VQ release encoder); K after the L2 clip, V after the unweighted RMS norm',
    'global_kv_bytes_per_token': 9216, 'drafter_global_kv_bytes_per_token': 1152,
    'local_layers': 'BF16 ring of 1,024 positions (sliding window), 40 layers',
    'tiering': 'resident up to a serving threshold, SSD pages past it; the global layers attend densely, so decode '
               'past the threshold reads every paged block',
    'native_context': 1048576}


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', type=Path, required=True)
    ap.add_argument('--docs', type=Path, required=True)
    ap.add_argument('--tools', type=Path, required=True)
    ap.add_argument('--extra-tools', type=Path, nargs='*', default=[])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budget', type=Path, help='memory-budget.json to include')
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    receipts = a.parts / 'receipts'
    parts, manifest = [], []
    for l in fmt.MOE_LAYERS:
        p = a.parts / f'layer-{l:02d}-experts.gguf'
        rec = json.loads((receipts / f'layer-{l:02d}-experts.json').read_text())
        s = fmt.verify_expert_part(p, l, rec, fmt.RECIPE_ID, 1)
        link(p, out / p.name)
        parts.append({'path': p.name, 'role': 'trunk', 'residency': 'resident', 'required': True, 'bytes': s['bytes'],
                      'sha256': rec['sha256'], 'layer': l, 'variant': 'R12', 'container_version': 1, 'index_bits': 12,
                      'residual': False, 'method': 'hvq', 'codebook_sha256': rec['codebook_sha256']})
        manifest.append({'name': fmt.book_name(l), 'part': p.name, 'stored_shape': [4096, 8], 'type_id': fmt.BOOK,
                         'payload_bytes': 4096 * 8, 'payload_sha256': rec['codebook_sha256'], 'derived': 'codebook'})
        for t in rec['tensors']:
            proj = t['name'].split('.')[-2]
            manifest.append({'name': t['name'], 'part': p.name, 'stored_shape': list(fmt.SHAPES[proj]), 'type_id': fmt.VQ,
                             'payload_bytes': t['payload_bytes'], 'payload_sha256': t['payload_sha256']})
    nx = json.loads((receipts / 'nonexpert-receipt.json').read_text())
    for name, (role, residency, required) in NATIVE.items():
        p = a.parts / name
        with fmt.PartReader(p) as r:
            if r.meta[fmt.NS + 'part_role'] != role:
                raise ValueError(f'{name}: part role {r.meta[fmt.NS + "part_role"]} != {role}')
            for t, e in r.entries.items():
                raw = r.raw(t)
                manifest.append({'name': t, 'part': name, 'stored_shape': list(e['shape']), 'type_id': e['type_id'],
                                 'payload_bytes': e['payload_bytes'], 'payload_sha256': hashlib.sha256(raw).hexdigest()})
                raw.release()
            size = len(r.data)
        link(p, out / name)
        parts.append({'path': name, 'role': role, 'residency': residency, 'required': required, 'bytes': size,
                      'sha256': sha_file(p)})
    mapping = nx.get('source_mapping')
    if not mapping:
        raise ValueError('non-expert receipt has no source mapping')
    names = [(m['source_checkpoint'], m['source_name']) for m in mapping]
    if len(names) != len(set(names)):
        raise ValueError('source mapping lists a source tensor twice')
    cov = nx.get('coverage') or {}
    if cov.get('main', {}).get('unmapped_count') != 0 or cov.get('mtp', {}).get('unmapped_count') != 0 \
            or cov.get('main', {}).get('mapped') != 1610 or cov.get('mtp', {}).get('mapped') != 84:
        raise ValueError(f'source coverage incomplete: {cov}')
    release = {'format': fmt.VARIANT, 'format_version': 1, 'recipe_id': fmt.RECIPE_ID, 'entrypoint': 'release.json',
               'architecture': fmt.ARCH, 'source_repo': 'thinkingmachines/Inkling-2-Small-MXFP8',
               'source_revision': fmt.SOURCE_REVISION, 'base_model': 'thinkingmachines/Inkling-2-Small',
               'parts': parts, 'kv_contract': KV_CONTRACT,
               'bytes': {'resident_trunk': sum(p['bytes'] for p in parts if p['role'] == 'trunk'),
                         'drafter': sum(p['bytes'] for p in parts if p['role'] == 'drafter'),
                         'vision': sum(p['bytes'] for p in parts if p['role'] == 'vision'),
                         'embedding_ssd': sum(p['bytes'] for p in parts if p['role'] == 'embedding'),
                         'all_parts': sum(p['bytes'] for p in parts)}}
    (out / 'release.json').write_text(json.dumps(release, indent=1) + '\n')
    (out / 'tensor-manifest.json').write_text(json.dumps({'tensors': manifest, 'source_mapping': mapping}) + '\n')
    for d in a.docs.iterdir():
        if d.is_file():
            shutil.copy2(d, out / d.name)
        elif d.is_dir():
            shutil.copytree(d, out / d.name, dirs_exist_ok=True)
    (out / 'tools').mkdir(exist_ok=True)
    for t in TOOLS:
        src = a.tools / t
        if src.exists():
            (out / 'tools' / Path(t).parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / 'tools' / t)
    for t in a.extra_tools:
        shutil.copy2(t, out / 'tools' / t.name)
    (out / 'conversion-receipts').mkdir(exist_ok=True)
    for r in receipts.glob('*.json'):
        shutil.copy2(r, out / 'conversion-receipts' / r.name)
    if a.budget:
        shutil.copy2(a.budget, out / 'memory-budget.json')
    lines = []
    for f in sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '.cache' not in p.parts):
        lines.append(f'{sha_file(f)}  {f.relative_to(out)}')
    (out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'parts': len(parts), 'bytes': release['bytes'], 'tensors': len(manifest), 'files': len(lines)}))


if __name__ == '__main__':
    main()
