#!/usr/bin/env python3
"""Assemble an A.X-K2 VQ release directory from one expert variant and the shared non-expert parts.

Hard-links the GGUF parts (never writes into them, never copies them):
`layer-LL-experts.gguf` from --experts-dir (one converted variant), and
`body-nonexpert.gguf` / `embed-tokens-bf16.gguf` from --parts. Re-reads every
part with the strict reader, hashes every payload independently of the
converter receipts and cross-checks both, checks that all 60 layer receipts
come from one variant (mode, gain refit, codebook scope, column knobs) that
matches the recipe's selection, accounts for every one of the 94,172 source
tensors exactly once (FP8 `weight_scale_inv` companions as `consumed_scale`
of their quantized weight), then writes release.json, tensor-manifest.json,
weight-error-by-layer.json, conversion-receipts/, files.json and SHA256SUMS
and runs the release verification. Unchanged source assets, the card, docs
and tools are copied in.

--dry-run skips the recipe-selection check and allows @@ placeholders, to
prove the path on any variant before the selection is final.

Forked from the Beam-1.0 release assembler.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from axk2_source import Source, SOURCE_REV
from axk2_vq_format import PartReader, VARIANT, NS, book_id, verify_source_mapping, verify_release
from convert_nonexpert import policy

ASSETS = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
          'special_tokens_map.json', 'chat_template.jinja']
TOOL_GLOBS = ['*.py', '*.sh', '*.md', 'cuda/*.cu', 'cuda/LICENSE*', 'native/*.c', 'native/*.json', 'tests/*.py']
DISPOSITION = {'expert': 'consumed_vq', 'q4k': 'consumed_q4k', 'q8_0': 'consumed_q8_0',
               'preserve': 'preserved', 'embedding': 'preserved'}
# Receipt fields that must be identical in every layer of one variant (shrink_units is layer-dependent).
VARIANT_KEYS = ('mode', 'gain_refit', 'codebook_scope')
WEIGHTING_KEYS = ('mode', 'shrink_tokens', 'col_floor', 'col_shrink', 'col_power', 'gain_refit')
PLACEHOLDER_RE = re.compile(r'@@[A-Z_]+@@')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def scan(path):
    """-> (part name, role, file sha256, size, [tensor records]) via the strict reader."""
    records = []
    with PartReader(path) as r:
        role = r.meta[NS + 'part_role']
        for name, e in sorted(r.entries.items(), key=lambda kv: kv[1]['offset']):
            raw = r.raw(name)
            digest = hashlib.sha256(raw).hexdigest()
            raw.release()
            records.append({'name': name, 'part': path.name, 'stored_shape': list(e['shape']),
                            'type_id': e['type_id'], 'payload_bytes': e['payload_bytes'], 'payload_sha256': digest})
    return path.name, role, sha(path), path.stat().st_size, records


def source_mapping(src):
    cfg = src.config
    plan = {n: policy(n, cfg) for n in src.index}
    mapping = []
    for n in sorted(src.index):
        pol = plan[n]
        if pol == 'scale':
            weight = n[:-len('_scale_inv')]
            if plan.get(weight) not in ('expert', 'q4k', 'q8_0'):
                raise SystemExit(f'scale {n} belongs to an unquantized weight')
            mapping.append({'source_name': n, 'target_name': weight, 'source_shard': src.index[n],
                            'disposition': 'consumed_scale'})
        else:
            mapping.append({'source_name': n, 'target_name': n, 'source_shard': src.index[n],
                            'disposition': DISPOSITION[pol]})
    return mapping


def lloyd_iterations(receipt):
    """Lloyd iterations every book of a layer was trained with (from the per-iteration history), or None."""
    t = receipt.get('training') or {}
    histories = [b.get('iterations') for b in t.get('books') or []] or [t.get('iterations')]
    counts = {len(h) for h in histories if isinstance(h, list)}
    if len(counts) > 1:
        raise SystemExit(f'layer {receipt.get("layer")} books were trained with different iteration counts {sorted(counts)}')
    return counts.pop() if counts else None


def variant_of(receipt):
    w = receipt.get('weighting') or {}
    return {**{k: receipt.get(k) for k in VARIANT_KEYS}, **{f'weighting.{k}': w.get(k) for k in WEIGHTING_KEYS},
            'lloyd_iterations': lloyd_iterations(receipt)}


def load_receipts(experts, layers, layer_bits, recipe):
    receipts, variant = {}, None
    for l in layers:
        r = json.loads((experts / f'layer-{l:02d}-experts.json').read_text())
        if r.get('layer', l) != l or r.get('source_revision', SOURCE_REV) != SOURCE_REV:
            raise SystemExit(f'layer {l} receipt is for another layer/source')
        bits = r.get('index_bits') or {512: 9, 1024: 10}.get(r.get('k'))
        if bits != layer_bits[l] or r.get('k', 2 ** bits) != 2 ** bits:
            raise SystemExit(f'layer {l} receipt has {bits} bits, recipe wants {layer_bits[l]}')
        if r['recipe_id'] != recipe['recipe_id']:
            raise SystemExit(f'layer {l} receipt recipe_id {r["recipe_id"]} != {recipe["recipe_id"]}')
        v = variant_of(r)
        if variant is None:
            variant = v
        elif v != variant:
            raise SystemExit(f'layer {l} comes from another variant: {v} vs {variant}')
        books = r.get('codebooks') or [{'name': r['codebook_name'], 'id': 4 * l, 'sha256': r['codebook_sha256']}]
        for b in books:
            if b.get('id', book_id(l, b['name'])) != book_id(l, b['name']):
                raise SystemExit(f'layer {l} codebook {b["name"]} has ID {b["id"]}')
        receipts[l] = r
    return receipts, variant


def check_selection(recipe, variant):
    """The final recipe must name exactly the variant being released."""
    cal = recipe.get('calibration') or {}
    want = {'mode': cal.get('weighting_mode'), 'gain_refit': cal.get('gain_refit'),
            'codebook_scope': cal.get('codebook_scope'), 'weighting.col_power': cal.get('col_power'),
            'weighting.col_floor': cal.get('col_floor'), 'weighting.shrink_tokens': cal.get('shrink_tokens')}
    bad = {k: (v, variant.get(k)) for k, v in want.items()
           if v is not None and not (isinstance(v, (int, float)) and isinstance(variant.get(k), (int, float))
                                     and float(v) == float(variant[k])) and v != variant.get(k)}
    if 'lloyd_iterations' in recipe and variant.get('lloyd_iterations') != recipe['lloyd_iterations']:
        bad['lloyd_iterations'] = (recipe['lloyd_iterations'], variant.get('lloyd_iterations'))
    if bad:
        raise SystemExit(f'recipe selection does not match the variant (recipe, variant): {bad}')


def placeholders(paths):
    """Files under `paths` that still contain an @@NAME@@ placeholder."""
    out = []
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.rglob('*')) if p.is_dir() else []):
            if f.is_file() and f.suffix in ('.md', '.json', '.txt') and PLACEHOLDER_RE.search(f.read_text(errors='ignore')):
                out.append(str(f))
    return out


def link(src, dst):
    if dst.exists():
        if os.path.samefile(dst, src):
            return
        dst.unlink()
    os.link(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts', type=Path, default=Path('/workspace/axk2/parts'),
                   help='shared body-nonexpert.gguf, embed-tokens-bf16.gguf, nonexpert-receipt.json')
    p.add_argument('--experts-dir', type=Path, help='one variant: layer-LL-experts.gguf + .json (default: --parts)')
    p.add_argument('--source', default='/models/A.X-K2')
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--card', type=Path, required=True, help='model card source, published as README.md')
    p.add_argument('--docs', type=Path, required=True, help='release docs dir: RECIPE.md, FORMAT.md, SOURCE.md, licenses')
    p.add_argument('--extra', type=Path, nargs='*', default=[], help='extra files/dirs copied to the release root')
    p.add_argument('--tools', type=Path, help='copy converter sources into tools/')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--dry-run', action='store_true', help='skip the selection check, allow @@ placeholders')
    a = p.parse_args()
    experts = a.experts_dir or a.parts
    recipe = json.loads(a.recipe.read_text())
    src = Source(a.source)
    layer_bits = {int(l): b for l, b in recipe['layer_bits'].items()}
    layers = sorted(layer_bits)
    if layers != src.moe_layers():
        raise SystemExit('recipe must cover every MoE layer')
    parts = [experts / f'layer-{l:02d}-experts.gguf' for l in layers]
    parts += [a.parts / 'body-nonexpert.gguf', a.parts / 'embed-tokens-bf16.gguf']
    missing = [x for x in parts if not x.is_file()]
    if missing:
        raise SystemExit(f'missing parts: {missing}')
    receipts, variant = load_receipts(experts, layers, layer_bits, recipe)
    if not a.dry_run:
        check_selection(recipe, variant)
        left = placeholders([a.recipe, a.card, a.docs] + list(a.extra))
        if left:
            raise SystemExit(f'unfilled placeholders in: {left}')
    nonexpert = json.loads((a.parts / 'nonexpert-receipt.json').read_text())
    if not nonexpert.get('complete', False):
        raise SystemExit('non-expert receipt is from a partial (test) run')
    if nonexpert['recipe_id'] != recipe['recipe_id']:
        raise SystemExit('non-expert receipt is for another recipe')

    a.out.mkdir(parents=True, exist_ok=True)
    for x in parts:
        link(x, a.out / x.name)
    with ProcessPoolExecutor(a.workers) as pool:
        scanned = list(pool.map(scan, [a.out / x.name for x in parts]))
    tensors, release_parts = [], []
    for name, role, digest, size, records in scanned:
        release_parts.append({'path': name, 'role': role, 'required': True, 'bytes': size, 'sha256': digest})
        tensors += records
    by_part = {x['path']: x for x in release_parts}

    # Cross-check file and payload hashes against the converter receipts.
    expected = {}
    for l, r in receipts.items():
        part = by_part[f'layer-{l:02d}-experts.gguf']
        if r['sha256'] != part['sha256'] or r.get('bytes', part['bytes']) != part['bytes']:
            raise SystemExit(f'layer {l} file hash/size differs from its receipt')
        expected.update({t['name']: t['payload_sha256'] for t in r['tensors']})
        for b in r.get('codebooks') or [{'name': r['codebook_name'], 'sha256': r['codebook_sha256']}]:
            expected[b['name']] = b['sha256']
    for pname, prec in nonexpert['parts'].items():
        if prec['sha256'] != by_part[pname]['sha256']:
            raise SystemExit(f'{pname} file hash differs from the non-expert receipt')
    expected.update({n: t['payload_sha256'] for n, t in nonexpert['tensors'].items()})
    for t in tensors:
        if expected.get(t['name']) != t['payload_sha256']:
            raise SystemExit(f'payload hash differs from receipt: {t["name"]}')
    if set(expected) != {t['name'] for t in tensors}:
        raise SystemExit('receipts and parts name different tensors')

    mapping = source_mapping(src)
    source_names = sorted(src.index)
    verify_source_mapping(source_names, mapping, tensors)

    cal = recipe.get('calibration', {})
    release = {'format': VARIANT, 'format_version': 1, 'recipe_id': recipe['recipe_id'],
               'source_repo': recipe['source_repo'], 'source_revision': SOURCE_REV,
               'variant': variant, 'calibration_set_sha256': (cal.get('calibration_set') or {}).get('safetensors_sha256'),
               'nonexpert_imatrix_files_sha256': (nonexpert.get('imatrix') or {}).get('files_sha256'),
               'entrypoint': 'release.json', 'parts': release_parts}
    if a.dry_run:
        release['dry_run'] = True
    (a.out / 'release.json').write_text(json.dumps(release, indent=2) + '\n')
    counts = {}
    for t in tensors:
        counts[t['type_id']] = counts.get(t['type_id'], 0) + 1
    manifest = {'format': VARIANT, 'format_version': 1, 'recipe_id': recipe['recipe_id'],
                'source_revision': SOURCE_REV, 'tensor_count': len(tensors),
                'type_counts': {str(k): v for k, v in sorted(counts.items())},
                'source_tensor_count': len(mapping), 'tensors': tensors,
                'source_names': source_names, 'source_mapping': mapping}
    (a.out / 'tensor-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    errors = {'variant': variant, 'layers': {
        str(l): {'index_bits': r['index_bits'], 'k': r['k'], 'codebooks': [b['name'] for b in r.get('codebooks', [])],
                 **{k: r['quality'].get(k) for k in ('rel_mse', 'imatrix_rel_err', 'cosine_estimate', 'output_proxy_sse')},
                 'by_proj': r['quality'].get('by_proj'),
                 'heldout_e4m3': (r.get('training') or {}).get('heldout_e4m3')}
        for l, r in receipts.items()}}
    (a.out / 'weight-error-by-layer.json').write_text(json.dumps(errors, indent=1) + '\n')
    rec_dir = a.out / 'conversion-receipts'
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    rec_dir.mkdir()
    for l in layers:
        shutil.copyfile(experts / f'layer-{l:02d}-experts.json', rec_dir / f'layer-{l:02d}-experts.json')
    shutil.copyfile(a.parts / 'nonexpert-receipt.json', rec_dir / 'nonexpert-receipt.json')

    for f in ASSETS:
        shutil.copyfile(Path(a.source) / f, a.out / f)
    shutil.copyfile(a.recipe, a.out / 'recipe.json')
    shutil.copyfile(a.card, a.out / 'README.md')
    for f in sorted(a.docs.iterdir()):
        if f.is_file():
            shutil.copyfile(f, a.out / f.name)
    for f in a.extra:
        dst = a.out / f.name
        if f.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(f, dst)
        else:
            shutil.copyfile(f, dst)
    if a.tools:
        dst = a.out / 'tools'
        if dst.exists():
            shutil.rmtree(dst)
        for pattern in TOOL_GLOBS:
            for f in sorted(a.tools.glob(pattern)):
                if f.is_file():
                    (dst / f.relative_to(a.tools)).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst / f.relative_to(a.tools))

    files = []
    for f in sorted(a.out.rglob('*')):
        rel = f.relative_to(a.out).as_posix()
        if f.is_file() and rel not in ('files.json', 'SHA256SUMS') and not rel.startswith('.'):
            files.append({'path': rel, 'bytes': f.stat().st_size,
                          'sha256': by_part[rel]['sha256'] if rel in by_part else sha(f)})
    (a.out / 'files.json').write_text(json.dumps({'files': files}, indent=1) + '\n')
    lines = [f"{x['sha256']}  {x['path']}" for x in files]
    lines.append(f"{sha(a.out / 'files.json')}  files.json")
    (a.out / 'SHA256SUMS').write_text('\n'.join(sorted(lines, key=lambda s: s.split('  ', 1)[1])) + '\n')
    result = verify_release(a.out, a.workers)
    print(json.dumps({'release': str(a.out), 'variant': variant, 'parts': len(release_parts), 'tensors': len(tensors),
                      'source_tensors': len(mapping), 'gguf_bytes': sum(x['bytes'] for x in release_parts),
                      'dry_run': a.dry_run, 'verify': result}, indent=1))


if __name__ == '__main__':
    main()
