#!/usr/bin/env python3
"""Derive the named Q4_K nonexpert recipe without rewriting VQ/MTP parts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

ALIGN = 128
Q4_K, BF16, FP8 = 12, 30, 0x47514638
BLOCK, BLOCK_BYTES = 256, 144
ROWS = 128
POLICY = 'q4_k_nonexpert_v1'
ROLES = {
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'self_attn.q_a_proj', 'self_attn.q_b_proj', 'self_attn.kv_a_proj_with_mqa',
    'self_attn.kv_b_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    'mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj',
    'mlp.shared_experts.down_proj',
}


def target(name):
    if name in {'lm_head.weight', 'model.language_model.embed_tokens.weight'}:
        return True
    m = re.fullmatch(r'model\.language_model\.layers\.(\d+)\.(.+)\.weight', name)
    return bool(m and int(m[1]) < 45 and m[2] in ROLES)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def read_header(file):
    def value(fmt):
        return struct.unpack('<' + fmt, file.read(struct.calcsize('<' + fmt)))

    def text():
        n, = value('Q')
        if n > 16_000_000:
            raise ValueError('oversized GGUF string')
        return file.read(n).decode()

    magic, version, count, nm = value('4sIQQ')
    if magic != b'GGUF' or version != 3 or max(count, nm) > 100000:
        raise ValueError('unsupported GGUF header')
    meta = {}
    for _ in range(nm):
        key, typ = text(), value('I')[0]
        if key in meta or typ not in (4, 7, 8):
            raise ValueError('unsupported or duplicate metadata')
        meta[key] = (typ, text() if typ == 8 else value('I' if typ == 4 else 'B')[0])
    entries = []
    for _ in range(count):
        name, ndim = text(), value('I')[0]
        if not 1 <= ndim <= 4:
            raise ValueError('unsupported tensor rank')
        dims = value('Q' * ndim)
        typ, offset = value('IQ')
        entries.append(dict(name=name, dims=dims, typ=typ, offset=offset))
    if meta.get('glm_vq.format_version') != (4, 1) or meta.get('general.alignment') != (4, ALIGN):
        raise ValueError('not a GVQ1 part')
    return meta, entries, align(file.tell())


def header(meta, entries):
    out = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(entries), len(meta)))
    for name, (typ, val) in meta.items():
        out += string(name) + struct.pack('<I', typ)
        out += string(val) if typ == 8 else struct.pack('<I' if typ == 4 else '<B', val)
    for e in entries:
        out += string(e['name']) + struct.pack('<I', len(e['dims']))
        out += struct.pack('<' + 'Q' * len(e['dims']), *e['dims'])
        out += struct.pack('<IQ', e['typ'], e['out_offset'])
    out += bytes(align(len(out)) - len(out))
    return out


def bf16_round(x):
    bits = np.ascontiguousarray(x, dtype='<f4').view('<u4')
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return bits.view('<f4')


def fp8_table():
    b = np.arange(256, dtype=np.uint16)
    exp, mant = (b >> 3) & 15, b & 7
    val = np.where(exp == 0, mant * 2.0**-9, (1 + mant / 8) * np.exp2(exp.astype(int) - 7))
    val = np.where(b & 128, -val, val).astype('<f4')
    val[(b & 127) == 127] = np.nan
    return val


class Quantizer:
    def __init__(self, lib):
        self.lib = ct.CDLL(str(lib.resolve()))
        self.fn = self.lib.ds4q_quantize_chunk
        self.fn.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int64,
                            ct.c_int64, ct.c_int64, ct.c_void_p]
        self.fn.restype = ct.c_size_t
        self.lut = fp8_table()

    def rows(self, fd, offset, typ, cols, first, count, scale):
        width = 2 if typ == BF16 else 1
        raw = os.pread(fd, count * cols * width, offset + first * cols * width)
        if len(raw) != count * cols * width:
            raise ValueError('truncated source matrix')
        if typ == BF16:
            x = (np.frombuffer(raw, '<u2').astype('<u4') << 16).view('<f4').reshape(count, cols)
        else:
            x = self.lut[np.frombuffer(raw, np.uint8)].reshape(count, cols)
            x = bf16_round(x * scale[first // ROWS].repeat(ROWS)[None, :])
        if not np.isfinite(x).all():
            raise ValueError('nonfinite source matrix')
        out = np.empty(count * cols // BLOCK * BLOCK_BYTES, np.uint8)
        size = self.fn(Q4_K, x.ctypes.data, out.ctypes.data, 0, count, cols, None)
        if size != out.size:
            raise ValueError('Q4_K writer size mismatch')
        scales = out.reshape(-1, BLOCK_BYTES)[:, :4].copy().view('<f2')
        if not np.isfinite(scales).all():
            raise ValueError('nonfinite Q4_K scales')
        return raw, out.tobytes()


def rewrite(src, dst, records, quant, workers):
    with src.open('rb') as inp:
        meta, entries, base = read_header(inp)
        byname = {e['name']: e for e in entries}
        links = json.loads(meta.get('glm_vq.fp8_scales', (8, '{}'))[1])
        out_offset = 0
        for e in entries:
            r = records[e['name']]
            if list(reversed(e['dims'])) != r['stored_shape'] or e['typ'] != r['type_id']:
                raise ValueError('manifest/header mismatch')
            e['source_type'] = e['typ']
            e['size'] = r['payload_bytes']
            if target(e['name']):
                if len(e['dims']) != 2 or e['dims'][0] % BLOCK or e['typ'] not in (BF16, FP8):
                    raise ValueError('unsupported recipe shape/type')
                e['typ'] = Q4_K
                e['size'] = math.prod(e['dims']) // BLOCK * BLOCK_BYTES
            e['out_offset'] = align(out_offset)
            out_offset = e['out_offset'] + e['size']
        meta['glm_vq.native_policy'] = (8, POLICY)
        meta['glm_vq.fp8_scales'] = (8, json.dumps({k: v for k, v in links.items() if not target(k)}, sort_keys=True))
        with dst.open('xb') as out, ThreadPoolExecutor(max_workers=workers) as pool:
            head = header(meta, entries)
            out.write(head)
            for e in entries:
                r = records[e['name']]
                out.write(bytes(len(head) + e['out_offset'] - out.tell()))
                original, encoded = hashlib.sha256(), hashlib.sha256()
                if e['typ'] == Q4_K:
                    cols, rows = e['dims']
                    scale = None
                    if e['source_type'] == FP8:
                        se = byname[links[e['name']]]
                        raw_scale = os.pread(inp.fileno(), records[se['name']]['payload_bytes'], base + se['offset'])
                        scale = np.frombuffer(raw_scale, '<f4').reshape(rows // ROWS, cols // ROWS)
                    # Bound outstanding source/float/output buffers to one worker window.
                    for start in range(0, rows, ROWS * workers):
                        jobs = [pool.submit(quant.rows, inp.fileno(), base + e['offset'],
                                           e['source_type'], cols, first, min(ROWS, rows - first), scale)
                                for first in range(start, min(rows, start + ROWS * workers), ROWS)]
                        for job in jobs:
                            raw, data = job.result()
                            original.update(raw)
                            encoded.update(data)
                            out.write(data)
                    r.update(type_id=Q4_K, payload_bytes=e['size'], precision='Q4_K',
                             byte_exact=False, parent_payload_sha256=r['payload_sha256'])
                else:
                    inp.seek(base + e['offset'])
                    remaining = e['size']
                    while remaining:
                        data = inp.read(min(remaining, 1024 * 1024))
                        if not data:
                            raise ValueError('truncated unchanged tensor')
                        original.update(data)
                        encoded.update(data)
                        out.write(data)
                        remaining -= len(data)
                if original.hexdigest() != r['payload_sha256']:
                    raise ValueError('source tensor checksum mismatch')
                r['payload_sha256'] = encoded.hexdigest()
            out.flush()
            os.fsync(out.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lib', type=Path, default=Path('gguf-tools/libds4q.so'))
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('workers must be 1..32')
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    manifest_path = source / 'tensor-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = json.loads((source / 'files.json').read_text())
    records = {r['name']: r for r in manifest['tensors']}
    parts = {r['part'] for r in records.values() if target(r['name'])}
    before = sum(r['payload_bytes'] for r in records.values())
    quant = Quantizer(args.lib)
    for entry in files['files']:
        name = entry['path']
        if Path(name).name != name:
            raise ValueError('release path must be a filename')
        if name == 'tensor-manifest.json':
            continue
        src, dst = source / name, output / name
        if name == 'README.md':
            dst.write_text('# GLM-5.3-Flash-VQ: Q4_K candidate\n\n'
                'Local derivative for measured nonexpert quantization. Quality and speed '
                'qualification are pending; see q4k-conversion.json and tensor-manifest.json '
                'for provenance. VQ experts, codebooks, MTP and vision retain their source bytes.\n')
            entry.update(bytes=dst.stat().st_size, sha256=digest(dst))
            continue
        if name not in parts:
            os.link(src, dst)
            continue
        rewrite(src, dst, records, quant, args.workers)
        entry.update(bytes=dst.stat().st_size, sha256=digest(dst))
        print(json.dumps(dict(part=name, bytes=entry['bytes'], seconds=time.monotonic() - started)), flush=True)
    manifest['nonexpert_recipe'] = dict(name=POLICY, parent_manifest_sha256=digest(manifest_path),
        quantizer_sha256=digest(args.lib), converter_sha256=digest(Path(__file__)),
        input='source BF16; FP8 times original block scale rounded to BF16',
        preserved='VQ/books, MTP, vision, router/norm/mHC/KDA gates/indexer',
        weights='Q4_K: 256 elements per 144-byte block, no imatrix')
    (output / 'tensor-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for entry in files['files']:
        if entry['path'] == 'tensor-manifest.json':
            entry.update(bytes=(output / entry['path']).stat().st_size, sha256=digest(output / entry['path']))
    (output / 'files.json').write_text(json.dumps(files, indent=2) + '\n')
    after = sum(r['payload_bytes'] for r in records.values())
    report = dict(recipe=POLICY, converted=sum(target(n) for n in records), parts=len(parts),
                  parent_manifest_sha256=digest(manifest_path), manifest_sha256=digest(output / 'tensor-manifest.json'),
                  payload_before=before, payload_after=after, saved_bytes=before-after,
                  elapsed_seconds=time.monotonic()-started)
    (output / 'q4k-conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
