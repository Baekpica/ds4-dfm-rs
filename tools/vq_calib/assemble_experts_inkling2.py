#!/usr/bin/env python3
"""Inkling-2-Small VQ: one converted layer (run_hvq_inkling2.py output) -> `layer-LL-experts.gguf`.

Input `CONVERT/layer-LL/`: `codebook.bin` (the [4096, 8] E4M3 book), `rank-R.bin` + `rank-R.json` (each rank's
GVQ1 payloads and their index), `receipt.json` (the layer's conversion record). Output: the expert part
in the inkling2_vq contract (FORMAT.md, inkling2_vq_format.py): one codebook
`model.llm.layers.L.mlp.vq_codebook` (ID 4·L) followed by the 1,152 GVQ1 matrices in expert-major
gate/up/down order, and `receipts/layer-LL-experts.json` (file and every payload SHA-256 plus the
conversion record). Every payload is re-hashed while it is copied and the finished part is verified by
the strict reader (verify_expert_part, with sampled row decodes).

    assemble_experts_inkling2.py --convert /work/inkling2/convert --parts /work/inkling2/parts --layers 0-47
"""
import argparse
import hashlib
import json
from pathlib import Path

import inkling2_vq_format as fmt


def assemble(conv, parts, layer, decode_rows=2, recipe_id=fmt.RECIPE_ID):
    src = conv / f'layer-{layer:02d}'
    rec = json.loads((src / 'receipt.json').read_text())
    if rec['layer'] != layer or rec['index_bits'] != 12 or rec['k'] != 4096 or rec['codebook_id'] != fmt.book_id(layer):
        raise ValueError(f'layer {layer}: receipt does not describe a 12-bit layer {layer}')
    book = (src / 'codebook.bin').read_bytes()
    if hashlib.sha256(book).hexdigest() != rec['codebook_sha256'] or len(book) != 4096 * 8:
        raise ValueError(f'layer {layer}: codebook bytes differ from the receipt')
    ranks = sorted(src.glob('rank-*.json'), key=lambda p: int(p.stem.split('-')[1]))
    index = {}
    files = {}
    for rp in ranks:
        r = json.loads(rp.read_text())
        binp = src / r['file']
        files[r['rank']] = binp
        for row in r['index']:
            if row['name'] in index:
                raise ValueError(f'duplicate matrix {row["name"]}')
            index[row['name']] = (r['rank'], row['offset'], row['payload_bytes'], row['payload_sha256'])
    inventory = fmt.expert_inventory(layer)
    if set(index) != {n for n, _, _ in inventory}:
        raise ValueError(f'layer {layer}: rank files do not cover the expert inventory exactly')
    bname = fmt.book_name(layer)
    weighting = {**rec['weighting'], 'method': 'hvq', 't': rec['t'], 'damping': rec['damping'],
                 'hessian_shrink_tokens': rec['hessian']['shrink_tokens'], 'calibration_run_id': rec['calibration_run_id']}
    meta = fmt.metadata(codebooks={str(fmt.book_id(layer)): bname}, part_role='trunk', extra={
        fmt.NS + 'source_revision': fmt.SOURCE_REVISION, fmt.NS + 'recipe_id': recipe_id,
        fmt.NS + 'layer': layer, fmt.NS + 'index_bits': 12,
        fmt.NS + 'weighting': json.dumps(weighting, sort_keys=True)})
    entries = [{'name': bname, 'shape': [4096, 8], 'type_id': fmt.BOOK, 'payload_bytes': len(book)}]
    entries += [{'name': n, 'shape': list(s), 'type_id': t, 'payload_bytes': index[n][2]} for n, s, t in inventory]
    parts.mkdir(parents=True, exist_ok=True)
    (parts / 'receipts').mkdir(exist_ok=True)
    out = parts / f'layer-{layer:02d}-experts.gguf'
    tmp = out.with_suffix('.gguf.tmp')
    if tmp.exists():
        tmp.unlink()
    handles = {r: p.open('rb') for r, p in files.items()}
    rows = []
    try:
        with fmt.PartWriter(tmp, meta, entries) as w:
            w.write(bname, book)
            for n, _, _ in inventory:
                r, off, size, sha = index[n]
                f = handles[r]
                f.seek(off)
                payload = f.read(size)
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha:
                    raise ValueError(f'{n}: payload differs from its rank index')
                w.write(n, payload)
                rows.append({'name': n, 'payload_bytes': size, 'payload_sha256': sha})
    finally:
        for f in handles.values():
            f.close()
    with tmp.open('rb') as f:
        file_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    receipt = {'path': out.name, 'layer': layer, 'index_bits': 12, 'bytes': tmp.stat().st_size, 'sha256': file_sha,
               'codebook_name': bname, 'codebook_id': fmt.book_id(layer), 'codebook_sha256': rec['codebook_sha256'],
               'recipe_id': recipe_id, 'tensors': rows, 'conversion': rec}
    tmp.rename(out)
    summary = fmt.verify_expert_part(out, layer, receipt, recipe_id, decode_rows)
    (parts / 'receipts' / f'layer-{layer:02d}-experts.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--convert', type=Path, required=True)
    ap.add_argument('--parts', type=Path, required=True)
    ap.add_argument('--layers', default='0-47')
    ap.add_argument('--decode-rows', type=int, default=2)
    ap.add_argument('--force', action='store_true', help='rebuild parts that already exist')
    a = ap.parse_args()
    for l in fmt.parse_layers(a.layers):
        out = a.parts / f'layer-{l:02d}-experts.gguf'
        if out.exists() and not a.force:
            rp = a.parts / 'receipts' / f'layer-{l:02d}-experts.json'
            s = fmt.verify_expert_part(out, l, json.loads(rp.read_text()), fmt.RECIPE_ID, 0)
            print(json.dumps({**s, 'reused': True}), flush=True)
            continue
        if not (a.convert / f'layer-{l:02d}' / 'receipt.json').exists():
            print(json.dumps({'layer': l, 'status': 'not converted yet'}), flush=True)
            continue
        if out.exists():
            out.unlink()
        print(json.dumps(assemble(a.convert, a.parts, l, a.decode_rows)), flush=True)


if __name__ == '__main__':
    main()
