#!/usr/bin/env python3
"""Derive the Ling-3.1 VQ v2 release from v1 without the source checkpoint.

v1's trunk part keeps the original BF16 embedding and the KDA FP8 decay
and output-gate projections with their block scales, so the v2 tensors come
from the same values the original converter read: Q8_0 for the embedding,
Q4_K for every KDA decay and gate from FP8 x block scale rounded to BF16.
Build the quantizer with -ffp-contract=off (make gguf-tools/libds4q-ieee.so):
it then reproduces the v1 converter's Q4_K/Q8_0 fixtures byte for byte.
Expert and MTP parts are hard-linked; text files are copied so their later
edits cannot reach v1."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

ALIGN = 128
F32, Q8_0, Q4_K, BF16, FP8 = 0, 8, 12, 30, 0x47514638
BLOCKS = {Q8_0: (32, 34), Q4_K: (256, 144)}
SCALE_ROWS = 128
ROWS = 512
BODY = 'body-nonexpert.gguf'
V1 = 'ling31-vq8-mix8r13-60r12-mtp13-q4h-v1'
V2 = 'ling31-vq8-mix8r13-60r12-mtp13-q4h-q8e-q4fg-v2'
EMBEDDING = 'model.word_embeddings.weight'
# KDA blocks: every layer but the MLA ones at (il + 1) % 8 == 0.
KDA = re.compile(r'model\.layers\.(\d+)\.attention\.([fg])_proj\.weight')


def target(name):
    if name == EMBEDDING:
        return Q8_0
    m = KDA.fullmatch(name)
    if m and int(m[1]) < 72 and (int(m[1]) + 1) % 8:
        return Q4_K
    return None


# Hashing and copying ~10 GiB parts drops their pages: on GB10's UMA a full
# page cache shrinks the pool the next model load draws device memory from.
def digest(path):
    with path.open('rb') as f:
        value = hashlib.file_digest(f, 'sha256').hexdigest()
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def read_header(f):
    def value(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

    def text():
        n, = value('Q')
        if n > 16_000_000:
            raise ValueError('oversized GGUF string')
        return f.read(n).decode()

    magic, version, count, kv = value('4sIQQ')
    if magic != b'GGUF' or version != 3:
        raise ValueError('not GGUF v3')
    meta = {}
    for _ in range(kv):
        key, typ = text(), value('I')[0]
        if typ == 8:
            meta[key] = (typ, text())
        elif typ == 4:
            meta[key] = (typ, value('I')[0])
        else:
            raise ValueError('unsupported metadata type %d' % typ)
    entries = []
    for _ in range(count):
        name = text()
        dims = list(value('Q' * value('I')[0]))
        typ, offset = value('IQ')
        entries.append(dict(name=name, dims=dims, typ=typ, offset=offset))
    return meta, entries, align(f.tell())


def header(meta, entries):
    out = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(entries), len(meta)))
    for key, (typ, val) in meta.items():
        out += string(key) + struct.pack('<I', typ)
        out += string(val) if typ == 8 else struct.pack('<I', val)
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


def decode(raw, typ, rows, cols):
    block, size = BLOCKS[typ]
    p = np.frombuffer(raw, np.uint8).reshape(-1, size)
    d = p[:, :2].copy().view('<f2').astype('<f4')
    if typ == Q8_0:
        return (d * p[:, 2:].view(np.int8).astype('<f4')).reshape(rows, cols)
    dm = p[:, 2:4].copy().view('<f2').astype('<f4')
    s = p[:, 4:16].astype(np.uint16)
    sc = np.empty((len(p), 8), '<f4')
    mn = np.empty_like(sc)
    for j in range(8):
        sc[:, j] = s[:, j] & 63 if j < 4 else (s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)
        mn[:, j] = s[:, j + 4] & 63 if j < 4 else (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    q = p[:, 16:].reshape(-1, 4, 32)
    vals = np.stack([q & 15, q >> 4], axis=2).reshape(-1, 8, 32).astype('<f4')
    return ((d * sc)[:, :, None] * vals - (dm * mn)[:, :, None]).reshape(rows, cols)


class Quantizer:
    def __init__(self, lib):
        self.fn = ct.CDLL(str(lib.resolve())).ds4q_quantize_chunk
        self.fn.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
        self.fn.restype = ct.c_size_t
        self.lut = fp8_table()

    # Rows [first, first + count) of one source matrix: their source bytes,
    # encoding, and error sums against the BF16 values the encoder read.
    def rows(self, fd, offset, src, typ, cols, first, count, scale):
        width = 2 if src == BF16 else 1
        raw = os.pread(fd, count * cols * width, offset + first * cols * width)
        if len(raw) != count * cols * width:
            raise ValueError('truncated source matrix')
        if src == BF16:
            x = (np.frombuffer(raw, '<u2').astype('<u4') << 16).view('<f4').reshape(count, cols)
        else:
            blocks = scale[np.arange(first, first + count) // SCALE_ROWS]
            x = bf16_round(self.lut[np.frombuffer(raw, np.uint8)].reshape(count, cols) *
                           np.repeat(blocks, SCALE_ROWS, axis=1)[:, :cols])
        if not np.isfinite(x).all():
            raise ValueError('nonfinite source matrix')
        block, size = BLOCKS[typ]
        out = np.empty(count * cols // block * size, np.uint8)
        if self.fn(typ, x.ctypes.data, out.ctypes.data, 0, count, cols, None) != out.size:
            raise ValueError('quantizer size mismatch')
        y = decode(out, typ, count, cols)
        if not np.isfinite(y).all():
            raise ValueError('nonfinite encoding')
        diff = (y - x).astype(np.float64)
        energy, sse = (x.astype(np.float64) ** 2).sum(1), (diff ** 2).sum(1)
        return raw, out.tobytes(), dict(energy=energy, sse=sse, max_abs=float(np.abs(diff).max()), first=first)


# v1's stored_payload_error fields over all rows (zero rows count as exact).
def stats(parts, cols):
    energy = np.concatenate([p['energy'] for p in parts])
    sse = np.concatenate([p['sse'] for p in parts])
    row = np.sqrt(sse / np.maximum(energy, 1e-300))
    worst = int(np.argmax(row))
    return dict(relative_rmse=float(np.sqrt(sse.sum() / energy.sum())),
                row_relative_rmse_p95=float(np.quantile(row, 0.95)),
                row_relative_rmse_p99=float(np.quantile(row, 0.99)),
                worst_relative=float(row[worst]), worst_row=worst,
                worst_source_rms=float(np.sqrt(energy[worst] / cols)),
                max_abs_error=max(p['max_abs'] for p in parts), signal_energy=float(energy.sum()),
                sse=float(sse.sum()), decoder='NumPy decode of the stored bytes')


def rewrite(src, dst, records, quant, workers):
    with src.open('rb') as inp:
        meta, entries, base = read_header(inp)
        links = json.loads(meta['ling_vq.fp8_scales'][1])
        chosen = {e['name']: target(e['name']) for e in entries if target(e['name'])}
        consumed = {links[n] for n, typ in chosen.items() if n in links}
        by_name = {e['name']: e for e in entries}
        kept = [e for e in entries if e['name'] not in consumed]
        out_offset = 0
        for e in kept:
            r = records[e['name']]
            if list(e['dims']) != list(reversed(r['stored_shape'])) or e['typ'] != r['type_id']:
                raise ValueError('manifest/header mismatch: ' + e['name'])
            e['source_type'] = e['typ']
            e['size'] = r['payload_bytes']
            if e['name'] in chosen:
                typ = chosen[e['name']]
                block, size = BLOCKS[typ]
                if len(e['dims']) != 2 or e['dims'][0] % block or e['typ'] not in (BF16, FP8):
                    raise ValueError('unsupported recipe shape/type: ' + e['name'])
                e['typ'] = typ
                e['size'] = e['dims'][0] * e['dims'][1] // block * size
            e['out_offset'] = align(out_offset)
            out_offset = e['out_offset'] + e['size']
        meta['ling_vq.recipe_id'] = (8, V2)
        meta['ling_vq.native_policy'] = (8, 'conservative_Q4_K_504_head_Q4_K_embedding_Q8_0_KDA_decay_gates_Q4_K')
        meta['ling_vq.recipe_status'] = (8, 'derived_from_v1_trunk_part')
        meta['ling_vq.fp8_scales'] = (8, json.dumps({k: v for k, v in links.items() if k not in chosen},
                                                    sort_keys=True, separators=(',', ':')))
        converted = {}
        with dst.open('xb') as out, ThreadPoolExecutor(max_workers=workers) as pool:
            head = header(meta, kept)
            out.write(head)
            for e in kept:
                r = records[e['name']]
                out.write(bytes(len(head) + e['out_offset'] - out.tell()))
                original, encoded = hashlib.sha256(), hashlib.sha256()
                if e['name'] in chosen:
                    cols, rows = e['dims']
                    scale = None
                    if e['source_type'] == FP8:
                        se = by_name[links[e['name']]]
                        raw_scale = os.pread(inp.fileno(), records[se['name']]['payload_bytes'], base + se['offset'])
                        scale = np.frombuffer(raw_scale, '<f4').reshape(-(-rows // SCALE_ROWS), -(-cols // SCALE_ROWS))
                    parts = []
                    for start in range(0, rows, ROWS * workers):
                        jobs = [pool.submit(quant.rows, inp.fileno(), base + e['offset'], e['source_type'],
                                            e['typ'], cols, first, min(ROWS, rows - first), scale)
                                for first in range(start, min(rows, start + ROWS * workers), ROWS)]
                        for job in jobs:
                            raw, data, part = job.result()
                            original.update(raw)
                            encoded.update(data)
                            out.write(data)
                            parts.append(part)
                    converted[e['name']] = (e['typ'], stats(parts, cols), links.get(e['name']))
                else:
                    inp.seek(base + e['offset'])
                    left = e['size']
                    while left:
                        data = inp.read(min(left, 1 << 24))
                        if not data:
                            raise ValueError('truncated tensor ' + e['name'])
                        original.update(data)
                        encoded.update(data)
                        out.write(data)
                        left -= len(data)
                if original.hexdigest() != r['payload_sha256']:
                    raise ValueError('source tensor checksum mismatch: ' + e['name'])
                r['payload_sha256'] = encoded.hexdigest()
                r['payload_bytes'] = e['size']
            out.flush()
            os.fsync(out.fileno())
            os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(inp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return converted, consumed


# files.json lists every file but itself and SHA256SUMS (in path order);
# SHA256SUMS lists files.json too. `changed` names the files to rehash (and
# add); hard-linked parts keep v1's entries. Rerun after editing a copy.
def index(root, files, changed):
    listed = {e['path'] for e in files['files']}
    files['files'] += [dict(path=name) for name in changed if name not in listed]
    files['files'].sort(key=lambda e: e['path'])
    for entry in files['files']:
        if entry['path'] in changed:
            entry.update(bytes=(root / entry['path']).stat().st_size, sha256=digest(root / entry['path']))
    (root / 'files.json').write_text(json.dumps(files, indent=2) + '\n')
    sums = [(e['path'], e['sha256']) for e in files['files']] + [('files.json', digest(root / 'files.json'))]
    (root / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (h, n) for n, h in sorted(sums)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path, help='v1 release directory')
    p.add_argument('output', type=Path, help='new directory for the v2 release')
    p.add_argument('--lib', type=Path, default=Path('gguf-tools/libds4q-ieee.so'))
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('workers must be 1..32')
    source, output = args.source.resolve(), args.output.resolve()
    release = json.loads((source / 'release.json').read_text())
    if release.get('recipe_id') != V1:
        p.error('source is not the v1 recipe')
    started = time.monotonic()
    manifest = json.loads((source / 'tensor-manifest.json').read_text())
    files = json.loads((source / 'files.json').read_text())
    records = {r['name']: r for r in manifest['tensors']}
    output.mkdir(exist_ok=False)
    quant = Quantizer(args.lib)
    regenerated = {'release.json', 'files.json', 'tensor-manifest.json', 'recipe.json', 'SHA256SUMS', BODY}
    for entry in files['files']:
        name = entry['path']
        src, dst = source / name, output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name in regenerated:
            continue
        if name.endswith('.gguf'):
            os.link(src, dst)
        else:
            dst.write_bytes(src.read_bytes())
    converted, consumed = rewrite(source / BODY, output / BODY, records, quant, args.workers)

    # Manifest: converted records, consumed scales, their source mapping.
    for name, (typ, error, scale) in converted.items():
        r = records[name]
        r.update(type_id=typ, precision='Q8_0' if typ == Q8_0 else 'Q4_K', byte_exact=False,
                 source_scale_tensor=scale, stored_payload_error=error)
    manifest['tensors'] = [r for r in manifest['tensors'] if r['name'] not in consumed]
    disposition = {name: 'consumed_%s' % ('q8_0' if typ == Q8_0 else 'q4k') for name, (typ, _, _) in converted.items()}
    for m in manifest['source_mapping']:
        if m['target_name'] in disposition:
            m['disposition'] = disposition[m['target_name']]
        if m['source_name'] in consumed:
            owner = next(n for n, (_, _, s) in converted.items() if s == m['source_name'])
            m.update(disposition=disposition[owner], target_name=owner)
    manifest['output_tensor_count'] = len(manifest['tensors'])
    manifest['recipe_id'] = V2
    parent = digest(source / 'tensor-manifest.json')
    manifest['derivation'] = dict(parent_recipe_id=V1, parent_manifest_sha256=parent,
        parent_part_sha256=digest(source / BODY), quantizer_sha256=digest(args.lib),
        converter_sha256=digest(Path(__file__)), quantizer='gguf-tools/quants.c, -O3 -ffp-contract=off, no imatrix',
        input='v1 trunk part: original BF16 embedding; FP8 x original block scale rounded to BF16 (KDA)',
        converted=sorted(converted), consumed_scales=sorted(consumed))
    (output / 'tensor-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    recipe = json.loads((source / 'recipe.json').read_text())
    recipe.update(recipe_id=V2, status='derived_from_v1', parent_recipe_id=V1,
                  embedding_precision='Q8_0', kda_gate_precision='Q4_K', kda_decay_precision='Q4_K')
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')

    body = next(part for part in release['parts'] if part['path'] == BODY)
    body.update(bytes=(output / BODY).stat().st_size, sha256=digest(output / BODY))
    release.update(recipe_id=V2, recipe_sha256=digest(output / 'recipe.json'))
    (output / 'release.json').write_text(json.dumps(release, indent=2, sort_keys=True) + '\n')

    report = dict(recipe_id=V2, parent_recipe_id=V1, parent_part_sha256=manifest['derivation']['parent_part_sha256'],
                  part_sha256=body['sha256'], part_bytes=body['bytes'], quantizer_sha256=manifest['derivation']['quantizer_sha256'],
                  converter_sha256=manifest['derivation']['converter_sha256'],
                  tensors={n: dict(precision='Q8_0' if t == Q8_0 else 'Q4_K', **e) for n, (t, e, _) in sorted(converted.items())},
                  consumed_scales=sorted(consumed), elapsed_seconds=time.monotonic() - started)
    receipt = output / 'conversion-receipts' / 'body-nonexpert-v2.json'
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    index(output, files, {BODY, 'release.json', 'tensor-manifest.json', 'recipe.json',
                          receipt.relative_to(output).as_posix()})
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}), flush=True)


if __name__ == '__main__':
    main()
