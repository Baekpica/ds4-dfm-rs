#!/usr/bin/env python3
"""K2-Horizon-375B-A23B VQ release directory: parts + release.json + tensor-manifest.json + docs + SHA256SUMS.

Every expert part is re-verified with the strict reader against its conversion receipt (file and payload
SHA-256) before it is linked into the release; the native parts are re-hashed against theirs. The tensor
manifest is built from those receipts and its source mapping must account for every tensor of the source
checkpoint exactly once.

    assemble_release_k2h.py --parts $W/parts --source $SRC --docs $W/release-docs --tools tools/vq_calib \
        --calib tokens.npz --convert $W/convert --out $W/release
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

import k2h_vq_format as fmt

NATIVE = {'body-nonexpert.gguf': ('trunk', 'resident', True),
          'embed-tokens-q8_0.gguf': ('embedding', 'resident-or-host-rows', True)}
TOOLS = ['k2h_ref.py', 'k2h_vq_format.py', 'k2h_eval.py', 'k2h_chat.py', 'run_hvq_k2h.py', 'assemble_experts_k2h.py',
         'convert_nonexpert_k2h.py', 'assemble_release_k2h.py', 'ds41_hvq.py', 'ds41_hess.py', 'axk2_vq_gpu.py',
         'axk2_weighting.py', 'cuda/ds41_hvq.cu', 'cuda/axk2_vq_convert.cu']
SOURCE_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'chat_template.jinja', 'LICENSE']
DOCS = ['README.md', 'FORMAT.md', 'RECIPE.md', 'SOURCE.md', 'EVALUATION.md']
LAYERS, KV_HEADS, HEAD_DIM, NATIVE_CTX = 61, 8, 128, 524288
KV_VALUES = LAYERS * 2 * KV_HEADS * HEAD_DIM                      # K and V values per token
KV_BYTES = {'bf16': KV_VALUES * 2, 'fp8_scale_per_16': KV_VALUES * 17 // 16, 'nvfp4': KV_VALUES * 9 // 16}
KV_CONTRACT = {
    'layers': 'all 61 (full causal GQA attention, 48 query / 8 KV heads x 128; no sliding window, no indexer)',
    'kv_format': 'nvfp4: E2M1 values, an E4M3 scale per 16 values = max(amax / 6, 2^-9), global scale 1 '
                 '(the MiniMax / Inkling-2 VQ release encoder)',
    'kv_bytes_per_token': KV_BYTES['nvfp4'], 'kv_bytes_per_token_bf16': KV_BYTES['bf16'],
    'reference_emulation': 'tools/k2h_ref.py nvfp4: K after RoPE, V; the runtime placement is fixed on DGX Spark',
    'tiering': 'resident only: every decode step reads all cached K/V',
    'native_context': NATIVE_CTX, 'kv_bytes_at_native_context': KV_BYTES['nvfp4'] * NATIVE_CTX}


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def verify_layer(job):
    parts, layer = job
    receipt = json.loads((parts / 'receipts' / f'layer-{layer:02d}-experts.json').read_text())
    return fmt.verify_expert_part(parts / f'layer-{layer:02d}-experts.gguf', layer, receipt, fmt.RECIPE_ID, 0), receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--docs', type=Path, required=True)
    ap.add_argument('--tools', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--convert', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--allow-partial', action='store_true', help='tests: only the layers converted so far')
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    layers = [l for l in fmt.MOE_LAYERS if (a.parts / 'receipts' / f'layer-{l:02d}-experts.json').exists()]
    if len(layers) != len(fmt.MOE_LAYERS) and not a.allow_partial:
        raise SystemExit(f'{len(fmt.MOE_LAYERS) - len(layers)} expert parts are missing')
    parts, stored, mapping, sums = [], [], {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(verify_layer, [(a.parts, l) for l in layers]))
    conv = {}
    for l, (summary, receipt) in zip(layers, results):
        name = receipt['path']
        link(a.parts / name, out / name)
        c = receipt['conversion']
        conv[l] = c
        parts.append({'path': name, 'role': 'trunk', 'residency': 'resident', 'required': True, 'bytes': receipt['bytes'],
                      'sha256': receipt['sha256'], 'layer': l, 'variant': 'R13', 'container_version': 1, 'index_bits': 13,
                      'residual': False, 'method': 'hvq', 'codebook_sha256': receipt['codebook_sha256']})
        sums[name] = receipt['sha256']
        stored.append({'name': receipt['codebook_name'], 'part': name, 'stored_shape': [8192, 8], 'type_id': fmt.BOOK,
                       'payload_bytes': 8192 * 8, 'payload_sha256': receipt['codebook_sha256']})
        for t in receipt['tensors']:
            proj = t['name'].split('.')[-2]
            stored.append({'name': t['name'], 'part': name, 'stored_shape': list(fmt.SHAPES[proj]), 'type_id': fmt.VQ,
                           'payload_bytes': t['payload_bytes'], 'payload_sha256': t['payload_sha256']})
            mapping[t['name']] = {'part': name, 'disposition': 'consumed_vq'}
        link(a.parts / 'receipts' / f'layer-{l:02d}-experts.json', out / 'conversion-receipts' / f'layer-{l:02d}-experts.json')
    for name, (role, residency, required) in NATIVE.items():
        rp = a.parts / 'receipts' / (Path(name).stem + '.json')
        receipt = json.loads(rp.read_text())
        if sha_file(a.parts / name) != receipt['sha256'] or (a.parts / name).stat().st_size != receipt['bytes']:
            raise SystemExit(f'{name}: file differs from its receipt')
        with fmt.PartReader(a.parts / name) as r:
            if r.meta[fmt.NS + 'part_role'] != role or r.order != [t['name'] for t in receipt['tensors']]:
                raise SystemExit(f'{name}: role / inventory')
        link(a.parts / name, out / name)
        link(rp, out / 'conversion-receipts' / rp.name)
        parts.append({'path': name, 'role': role, 'residency': residency, 'required': required,
                      'bytes': receipt['bytes'], 'sha256': receipt['sha256']})
        sums[name] = receipt['sha256']
        for t in receipt['tensors']:
            stored.append({'name': t['name'], 'part': name, 'stored_shape': t['shape'], 'type_id': t['type_id'],
                           'payload_bytes': t['payload_bytes'], 'payload_sha256': t['payload_sha256']})
            if t['name'] in mapping:
                raise SystemExit(f"{t['name']}: mapped twice")
            mapping[t['name']] = {'part': name, 'disposition': 'consumed_q8_0' if t['type'] == 'q8_0' else 'preserved_f32'}
    # exactly-once source accounting
    source_names = set(json.loads((a.source / 'model.safetensors.index.json').read_text())['weight_map'])
    missing, extra = source_names - set(mapping), set(mapping) - source_names
    if extra or (missing and not a.allow_partial):
        raise SystemExit(f'source accounting: {len(missing)} unmapped, {len(extra)} unknown')
    by = lambda role: sum(p['bytes'] for p in parts if p['role'] == role)
    experts_bytes = sum(p['bytes'] for p in parts if 'layer' in p)
    total = sum(p['bytes'] for p in parts)
    release = {'format': fmt.VARIANT, 'format_version': 1, 'recipe_id': fmt.RECIPE_ID, 'entrypoint': 'release.json',
               'architecture': fmt.ARCH, 'source_repo': 'IFM/K2-Horizon-375B-A23B', 'source_revision': fmt.SOURCE_REVISION,
               'base_model': 'IFM/K2-Horizon-375B-A23B', 'parts': parts, 'kv_contract': KV_CONTRACT,
               'bytes': {'expert_parts': experts_bytes, 'resident_trunk': by('trunk'), 'embedding': by('embedding'), 'all_parts': total}}
    (out / 'release.json').write_text(json.dumps(release, indent=1) + '\n')
    (out / 'tensor-manifest.json').write_text(json.dumps({
        'recipe_id': fmt.RECIPE_ID, 'source_revision': fmt.SOURCE_REVISION, 'stored_tensors': len(stored),
        'source_tensors': len(source_names), 'source_tensors_mapped': len(mapping),
        'dispositions': {d: sum(v['disposition'] == d for v in mapping.values()) for d in ('consumed_vq', 'consumed_q8_0', 'preserved_f32')},
        'tensors': stored, 'source_mapping': [{'source': n, **mapping[n]} for n in sorted(mapping)]}) + '\n')
    # memory budget: exact part bytes, the K/V arithmetic of the recipe
    gib = lambda b: round(b / 2**30, 3)
    ctxs = [8192, 32768, 65536, 131072, 262144, 524288]
    budget = {'source': 'exact part sizes of this release and the K/V byte arithmetic of the recipe; the runtime floor, graph '
                        'scratch and prefill workspace are measured on DGX Spark, not here',
              'parts_bytes': {'experts_58_parts': experts_bytes, 'body_nonexpert': by('trunk') - experts_bytes, 'embedding_q8_0': by('embedding')},
              'parts_gib': {'experts_58_parts': gib(experts_bytes), 'body_nonexpert': gib(by('trunk') - experts_bytes),
                            'embedding_q8_0': gib(by('embedding')), 'resident_trunk': gib(by('trunk')), 'all_parts': gib(total)},
              'kv_values_per_token': KV_VALUES, 'kv_bytes_per_token': KV_BYTES,
              'kv_gib_at': {str(c): {k: gib(v * c) for k, v in KV_BYTES.items()} for c in ctxs},
              'trunk_plus_nvfp4_kv_gib_at': {str(c): gib(by('trunk') + KV_BYTES['nvfp4'] * c) for c in ctxs},
              'all_parts_plus_nvfp4_kv_gib_at': {str(c): gib(total + KV_BYTES['nvfp4'] * c) for c in ctxs},
              'dgx_spark_memory_gib': {'128 GB unified memory': gib(128e9)},
              'left_at_native_context_gib': {'embedding as host rows': gib(128e9 - by('trunk') - KV_BYTES['nvfp4'] * NATIVE_CTX),
                                             'embedding on the device': gib(128e9 - total - KV_BYTES['nvfp4'] * NATIVE_CTX)}}
    (out / 'memory-budget.json').write_text(json.dumps(budget, indent=1) + '\n')
    # calibration record
    data = np.load(a.calib)
    c0 = conv[layers[0]]
    n_cal = c0['calibration_tokens'] // data['tokens'].shape[1]
    held = data['tokens'].shape[0] - 64
    cal = {'token_file': a.calib.name, 'token_file_sha256': sha_file(a.calib), 'calib_sha256_in_receipts': c0['calib_sha256'],
           'calibration_run_id': c0['calibration_run_id'], 'sequence_length': int(data['tokens'].shape[1]),
           'calibration_chunks': [0, n_cal - 1], 'calibration_tokens': c0['calibration_tokens'],
           'calibration_chunks_per_domain': np.bincount(data['source_ids'][:n_cal], minlength=5).tolist(),
           'heldout_chunks': [held, int(data['tokens'].shape[0]) - 1],
           'heldout_chunks_per_domain': np.bincount(data['source_ids'][held:], minlength=5).tolist(),
           'domains': ['coding_tool', 'math_science', 'general_reasoning', 'long_retrieval', 'general_text'],
           'note': 'the token file itself is the MQ87 release calibration mix (manifest beside this file)'}
    (out / 'calibration').mkdir(exist_ok=True)
    (out / 'calibration' / 'calibration.json').write_text(json.dumps(cal, indent=1) + '\n')
    meta = a.calib.with_suffix(a.calib.suffix + '.json')
    if meta.exists():
        shutil.copy2(meta, out / 'calibration' / meta.name)
    ts = a.convert / 'heldout' / 'teacher-summary.json'
    if ts.exists():
        shutil.copy2(ts, out / 'calibration' / 'heldout-original-summary.json')
    for t in TOOLS:
        (out / 'tools' / t).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.tools / t, out / 'tools' / t)
    for d in DOCS:
        if (a.docs / d).exists():
            shutil.copy2(a.docs / d, out / d)
        elif not a.allow_partial:
            raise SystemExit(f'{d} is missing')
    for extra_dir in ('evaluation',):
        if (a.docs / extra_dir).is_dir():
            shutil.copytree(a.docs / extra_dir, out / extra_dir, dirs_exist_ok=True)
    for s in SOURCE_FILES:
        shutil.copy2(a.source / s, out / s)
    lines = []
    for p in sorted(x for x in out.rglob('*') if x.is_file() and x.name != 'SHA256SUMS'):
        rel = p.relative_to(out).as_posix()
        lines.append(f'{sums.get(rel) or sha_file(p)}  {rel}')
    (out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'release': str(out), 'parts': len(parts), 'expert_layers': len(layers), 'bytes': release['bytes'],
                      'gib': {k: gib(v) for k, v in release['bytes'].items()}, 'stored_tensors': len(stored),
                      'source_tensors_mapped': len(mapping), 'source_tensors_unmapped': len(missing), 'files': len(lines) + 1}))


if __name__ == '__main__':
    main()
