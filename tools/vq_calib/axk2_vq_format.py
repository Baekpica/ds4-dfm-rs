#!/usr/bin/env python3
"""A.X-K2 GVQ v2 byte format and independent, mmap-backed CPU validation.

Forked from the Beam-1.0 release tool (beam_vq_format.py): the same GGUF v3
part container and GVQ1 header struct, an `axk2_vq` family namespace, and a
byte-major version 2 of the GVQ1 matrix for 9- and 10-bit indices:

  * main stream: one byte per index holding its low 8 bits, row-major vector
    order (n = r * C/8 + j);
  * plane: the high (bits - 8) bits; 9-bit: bit 8 of index n is bit n % 8 of
    byte n // 8; 10-bit: bits 8-9 of index n are the 2-bit field at bit
    2 * (n % 4) of byte n // 4 (LSB first).

Native tensors: F32, F16, BF16, Q4_K, Q8_0. Two part roles: `trunk` (device
resident) and `embedding` (the BF16 token table, read in rows from SSD by the
host). Production host integration belongs in Rust; this module never implies
a serving backend.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import struct
import numpy as np

ALIGN = HEADER = 128
F32, F16, Q8_0, Q4_K, BF16 = 0, 1, 8, 12, 30
BOOK, VQ = 0x47564342, 0x47565138
ARCH = 'axk2_vq'
NS = 'axk2_vq.'
VARIANT = 'axk2-vq8'
SOURCE_MODEL_TYPE = 'axk2'
ROLES = ('trunk', 'embedding')
GVQ_VERSION = 2
BITS = (9, 10)
FOREIGN_PREFIXES = ('glm_vq.', 'ling_vq.', 'beam_vq.', 'split.')
DISPOSITIONS = ('preserved', 'consumed_vq', 'consumed_q4k', 'consumed_q8_0', 'consumed_scale')
U64_MAX = 2**64 - 1
GVQ_HEADER = struct.Struct('<4sHBBIIIIIIQQQQ32s32s')
QUANT_BLOCKS = {Q4_K: (256, 144, 4), Q8_0: (32, 34, 2)}   # elements, bytes, leading F16 scale bytes
# Generated codebooks of an expert part: name suffix -> codebook ID offset from 4*layer
# (4L shared or gate/up, 4L+2 down, 4L+3 up under a three-way split; 4L+1 reserved for a residual stage).
BOOK_SUFFIXES = {'vq_codebook': 0, 'vq_codebook_down': 2, 'vq_codebook_up': 3}
BOOK_NAME = re.compile(r'model\.layers\.(\d+)\.mlp\.(vq_codebook(?:_down|_up)?)')


def book_id(layer, name):
    """Codebook ID for a generated book tensor name of `layer` (raises on a foreign name)."""
    m = BOOK_NAME.fullmatch(name)
    if not m or int(m[1]) != layer:
        raise ValueError(f'not a layer-{layer} codebook name: {name}')
    return 4 * layer + BOOK_SUFFIXES[m[2]]


def checked(n):
    if not isinstance(n, int) or not 0 <= n <= U64_MAX:
        raise ValueError('u64 overflow')
    return n


def align(n):
    return checked(checked(n + ALIGN - 1) // ALIGN * ALIGN)


def bits_for(k):
    if k == 512:
        return 9
    if k == 1024:
        return 10
    raise ValueError('unsupported K (GVQ v2 has K=512 / 1024)')


def geometry(rows, cols, k):
    bits = bits_for(k)
    if not 0 < rows < 2**32 or not 0 < cols < 2**32 or cols % 64:
        raise ValueError('unsupported GVQ v2 shape')
    n = checked(rows * cols // 8)
    main = align(HEADER + 2 * rows)
    main_end = checked(main + n)
    plane = align(main_end)
    end = checked(plane + n * (bits - 8) // 8)
    return dict(bits=bits, gain=HEADER, main=main, plane=plane, size=align(end),
                main_end=main_end, end=end, low_stride=cols // 8,
                high_stride=cols // 8 * (bits - 8) // 8)


def _pack_plane(ix, bits):
    """High bits of the flat index vector -> plane bytes (writer's own packing)."""
    if bits == 9:
        return np.packbits(((ix >> 8) & 1).astype(np.uint8), bitorder='little').tobytes()
    hi = ((ix >> 8) & 3).astype(np.uint8).reshape(-1, 4)
    return (hi[:, 0] | (hi[:, 1] << 2) | (hi[:, 2] << 4) | (hi[:, 3] << 6)).astype(np.uint8).tobytes()


def _header(out, rows, cols, k, codebook, book_id, precision, g):
    out[:HEADER] = GVQ_HEADER.pack(b'GVQ1', GVQ_VERSION, g['bits'], 8, rows, cols, k, precision, book_id, 0,
                                   HEADER, g['main'], g['plane'], g['size'],
                                   hashlib.sha256(codebook).digest(),
                                   hashlib.sha256(memoryview(out)[HEADER:]).digest())


def _check_gains(source_gains, rows):
    if source_gains.dtype.kind not in 'fiu':
        raise ValueError('gain must be real numeric values')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        gains = source_gains.astype('<f2')
    if gains.shape != (rows,) or not np.isfinite(gains).all() or np.signbit(gains).any():
        raise ValueError('gain must be finite nonnegative F16 per output row')
    if not np.array_equal(source_gains, gains):
        raise ValueError('gain must already be rounded to F16 before normalization')
    return gains


def encode_vq(gains, indices, rows, cols, k, codebook, book_id, precision=1):
    """Encode rounded gains/indices and raw stored codebook bytes.

    gains must already have been rounded to F16 before normalization, and
    indices assigned against this exact codebook precision. No training here.
    """
    g = geometry(rows, cols, k)
    gains = _check_gains(np.asarray(gains), rows)
    ix = np.asarray(indices)
    if ix.shape not in ((rows, cols // 8), (rows * cols // 8,)) or ix.dtype.kind not in 'iu':
        raise ValueError('index shape/type')
    if (ix < 0).any() or (ix >= k).any():
        raise ValueError('index outside codebook')
    _book_array(codebook, k, precision)
    ix = ix.astype('<u2', copy=False).reshape(-1)
    out = bytearray(g['size'])
    out[HEADER:HEADER + 2 * rows] = gains.tobytes()
    out[g['main']:g['main_end']] = (ix & 255).astype(np.uint8).tobytes()
    out[g['plane']:g['end']] = _pack_plane(ix, g['bits'])
    _header(out, rows, cols, k, codebook, book_id, precision, g)
    return bytes(out)


def assemble_vq(gain_bytes, low_bytes, high_bytes, rows, cols, k, codebook, book_id, precision=1):
    """GVQ v2 payload from already-packed streams (device packing). The result
    is cross-checked against encode_vq() on sampled tensors by the converter."""
    g = geometry(rows, cols, k)
    if len(gain_bytes) != 2 * rows or len(low_bytes) != g['main_end'] - g['main']:
        raise ValueError('gain/main stream size')
    if high_bytes is None or len(high_bytes) != g['end'] - g['plane']:
        raise ValueError('plane stream size')
    gains = np.frombuffer(gain_bytes, '<f2')
    if not np.isfinite(gains).all() or np.signbit(gains).any():
        raise ValueError('gain must be finite nonnegative F16 per output row')
    _book_array(codebook, k, precision)
    out = bytearray(g['size'])
    out[HEADER:HEADER + 2 * rows] = gain_bytes
    out[g['main']:g['main_end']] = low_bytes
    out[g['plane']:g['end']] = high_bytes
    _header(out, rows, cols, k, codebook, book_id, precision, g)
    return bytes(out)


def fp8_table():
    b = np.arange(256, dtype=np.uint16)
    exp, mant = (b >> 3) & 15, b & 7
    val = np.where(exp == 0, mant * 2.0**-9, (1 + mant / 8) * np.exp2(exp.astype(int) - 7))
    val = np.where(b & 128, -val, val).astype('<f4')
    val[(b & 127) == 127] = np.nan
    return val


FP8_TABLE = fp8_table()


def _book_array(raw, k, precision):
    if precision not in (1, 2) or len(raw) != k * 8 * precision:
        raise ValueError('codebook size/precision')
    array = FP8_TABLE[np.frombuffer(raw, np.uint8)] if precision == 1 else np.frombuffer(raw, '<f2').astype('<f4')
    if not np.isfinite(array).all():
        raise ValueError('nonfinite codebook')
    return array.reshape(k, 8)


class VqTensor:
    """Borrow payload bytes; decode only requested output rows."""
    def __init__(self, payload, book, shape=None, book_id=None):
        if len(payload) < HEADER:
            raise ValueError('truncated header')
        (magic, version, bits, vector, rows, cols, k, precision, link, flags,
         gain, main, plane, size, book_hash, payload_hash) = GVQ_HEADER.unpack_from(payload)
        if magic != b'GVQ1' or version != GVQ_VERSION or vector != 8 or flags != 0:
            raise ValueError('unsupported GVQ header')
        g = geometry(rows, cols, k)
        if bits != g['bits'] or (gain, main, plane, size) != tuple(g[n] for n in ('gain', 'main', 'plane', 'size')):
            raise ValueError('noncanonical GVQ offsets')
        if len(payload) != size or shape is not None and tuple(shape) != (rows, cols):
            raise ValueError('payload size/shape mismatch')
        if book_id is not None and link != book_id:
            raise ValueError('codebook ID mismatch')
        if hashlib.sha256(book).digest() != book_hash:
            raise ValueError('codebook checksum')
        if hashlib.sha256(memoryview(payload)[HEADER:]).digest() != payload_hash:
            raise ValueError('payload checksum')
        self.book = _book_array(book, k, precision)
        self.gains = np.frombuffer(payload, '<f2', rows, HEADER).astype('<f4')
        if not np.isfinite(self.gains).all() or np.signbit(self.gains).any():
            raise ValueError('negative/nonfinite gain')
        for start, end in [(HEADER + 2 * rows, main), (g['main_end'], plane), (g['end'], size)]:
            if any(memoryview(payload)[start:end]):
                raise ValueError('nonzero padding')
        self.payload, self.rows, self.cols, self.k, self.precision = payload, rows, cols, k, precision
        self.bits, self.book_id, self.geometry = bits, link, g

    def indices(self, first=0, count=None):
        count = self.rows - first if count is None else count
        if first < 0 or count < 0 or first + count > self.rows:
            raise ValueError('row range')
        g, nv = self.geometry, self.cols // 8
        low = np.frombuffer(self.payload, np.uint8, count * g['low_stride'], g['main'] + first * g['low_stride'])
        high = np.frombuffer(self.payload, np.uint8, count * g['high_stride'], g['plane'] + first * g['high_stride'])
        # Independent bit expansion, deliberately not the writer's shifts.
        width = self.bits - 8
        planes = np.unpackbits(high, bitorder='little').reshape(count * nv, width).astype(np.uint16)
        hi = np.sum(planes << np.arange(width, dtype=np.uint16), axis=-1, dtype=np.uint16)
        return (low.astype(np.uint16) | (hi << 8)).reshape(count, nv)

    def decode_rows(self, first=0, count=None):
        count = self.rows - first if count is None else count
        ix = self.indices(first, count)
        return (self.book[ix].reshape(count, self.cols) * self.gains[first:first+count, None]).astype('<f4')


def metadata(codebooks=None, part_role='trunk', extra=None):
    if part_role not in ROLES:
        raise ValueError('part role')
    out = {'general.architecture': ARCH, 'general.alignment': ALIGN,
           NS + 'format_version': 1, NS + 'variant': VARIANT,
           NS + 'source_model_type': SOURCE_MODEL_TYPE, NS + 'part_role': part_role,
           NS + 'codebooks': json.dumps(codebooks or {}, sort_keys=True)}
    if extra:
        if set(out) & set(extra):
            raise ValueError('cannot override family metadata')
        if any(not key.startswith((NS, 'general.')) for key in extra):
            raise ValueError('metadata outside the family namespace')
        out.update(extra)
    return out


def _string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


class PartWriter:
    """Streaming independent GGUF part writer, exclusive-create destination.

    entries: [{name, shape: HF [out,in], type_id, payload_bytes}]. Metadata
    values are strings, bools, or u32; write() must follow descriptor order.
    """
    def __init__(self, path, meta, entries):
        self.path, self.entries, self.index = Path(path), list(entries), 0
        if len({e['name'] for e in self.entries}) != len(self.entries):
            raise ValueError('duplicate tensor')
        out = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(self.entries), len(meta)))
        for key, value in meta.items():
            typ = 8 if isinstance(value, str) else 7 if isinstance(value, bool) else 4
            out += _string(key) + struct.pack('<I', typ)
            out += _string(value) if typ == 8 else struct.pack('<B' if typ == 7 else '<I', value)
        offset = 0
        self.offsets = []
        for entry in self.entries:
            offset = align(offset)
            self.offsets.append(offset)
            shape = entry['shape']
            if not 1 <= len(shape) <= 4 or any(not isinstance(d, int) or d <= 0 for d in shape):
                raise ValueError('invalid shape')
            out += _string(entry['name']) + struct.pack('<I', len(shape))
            out += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
            out += struct.pack('<IQ', entry['type_id'], offset)
            offset = checked(offset + entry['payload_bytes'])
        out += bytes(align(len(out)) - len(out))
        self.base, self.end = len(out), len(out) + offset
        self.file = self.path.open('xb')
        self.file.write(out)

    def write(self, name, payload):
        if self.index >= len(self.entries):
            raise ValueError('extra tensor')
        entry = self.entries[self.index]
        if entry['name'] != name or len(payload) != entry['payload_bytes']:
            raise ValueError('tensor order/size mismatch')
        target = self.base + self.offsets[self.index]
        self.file.write(bytes(target - self.file.tell()))
        self.file.write(payload)
        self.index += 1

    def write_chunks(self, name, chunks):
        """Native tensors can stream without a full-tensor Python copy."""
        if self.index >= len(self.entries) or self.entries[self.index]['name'] != name:
            raise ValueError('tensor order')
        expected = self.entries[self.index]['payload_bytes']
        target = self.base + self.offsets[self.index]
        self.file.write(bytes(target - self.file.tell()))
        written = 0
        for chunk in chunks:
            written += len(chunk)
            if written > expected:
                raise ValueError('oversized tensor')
            self.file.write(chunk)
        if written != expected:
            raise ValueError('truncated tensor')
        self.index += 1

    def close(self):
        try:
            if self.index != len(self.entries) or self.file.tell() != self.end:
                raise ValueError('missing tensor')
            self.file.flush()
        finally:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.file.close()
        else:
            self.close()


def _json_object(value):
    if not isinstance(value, str):
        raise ValueError('JSON metadata must be string')
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    obj = json.loads(value, object_pairs_hook=unique)
    if not isinstance(obj, dict):
        raise ValueError('link metadata must be an object')
    return obj


class PartReader:
    """Strict A.X-K2 family reader. Validation scans payloads without expansion."""
    def __init__(self, path):
        self.file = Path(path).open('rb')
        try:
            self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._parse()
        except Exception:
            if hasattr(self, 'data'):
                try:
                    self.data.close()
                except BufferError:
                    pass  # The rejecting frame owns views until its traceback is freed.
            self.file.close()
            raise

    def _parse(self):
        data, pos = self.data, 0
        def take(n):
            nonlocal pos
            end = checked(pos + n)
            if end > len(data):
                raise ValueError('truncated GGUF')
            raw, pos = data[pos:end], end
            return raw
        def unpack(fmt):
            return struct.unpack('<' + fmt, take(struct.calcsize('<' + fmt)))
        def string():
            n, = unpack('Q')
            if n > 16_000_000:
                raise ValueError('oversized string')
            return take(n).decode('utf-8')
        magic, version, count, nm = unpack('4sIQQ')
        if magic != b'GGUF' or version != 3 or count > 1000000 or nm > 100000:
            raise ValueError('unsupported GGUF header')
        meta, meta_types = {}, {}
        for _ in range(nm):
            key, typ = string(), unpack('I')[0]
            if key in meta or typ not in (4, 7, 8):
                raise ValueError('duplicate/unsupported metadata')
            value = string() if typ == 8 else unpack('I' if typ == 4 else 'B')[0]
            if typ == 7 and value not in (0, 1):
                raise ValueError('invalid boolean')
            meta[key] = value
            meta_types[key] = typ
        required = {'general.architecture': ARCH, 'general.alignment': ALIGN,
                    NS + 'format_version': 1, NS + 'variant': VARIANT,
                    NS + 'source_model_type': SOURCE_MODEL_TYPE}
        if any(meta.get(k) != v or meta_types.get(k) != (8 if isinstance(v, str) else 4) for k, v in required.items()) \
                or meta.get(NS + 'part_role') not in ROLES:
            raise ValueError('unsupported A.X-K2 family/version')
        if any(k.startswith(FOREIGN_PREFIXES) for k in meta):
            raise ValueError('foreign family/standard split metadata')
        entries = {}
        for _ in range(count):
            name, rank = string(), unpack('I')[0]
            if not name or name in entries or not 1 <= rank <= 4:
                raise ValueError('duplicate tensor/invalid rank')
            dims = unpack('Q' * rank)
            typ, offset = unpack('IQ')
            if 0 in dims or offset % ALIGN:
                raise ValueError('tensor shape/alignment')
            entries[name] = dict(name=name, shape=tuple(reversed(dims)), type_id=typ, offset=offset)
        base = align(pos)
        if base > len(data) or any(data[pos:base]):
            raise ValueError('GGUF header padding')
        cursor = 0
        for entry in sorted(entries.values(), key=lambda e: e['offset']):
            offset, typ, shape = entry['offset'], entry['type_id'], entry['shape']
            if offset != align(cursor) or any(data[base + cursor:base + offset]):
                raise ValueError('tensor gap/overlap/padding')
            elements = math.prod(shape)
            checked(elements)
            if typ == VQ:
                if len(shape) != 2 or base + offset + HEADER > len(data):
                    raise ValueError('VQ shape/truncation')
                k = struct.unpack_from('<I', data, base + offset + 16)[0]
                size = geometry(*shape, k)['size']
            elif typ in QUANT_BLOCKS:
                block, width, _ = QUANT_BLOCKS[typ]
                if len(shape) != 2 or shape[1] % block:
                    raise ValueError('native quant shape')
                size = checked(elements // block * width)
            elif typ in (F32, F16, BF16, BOOK):
                size = checked(elements * {F32: 4, F16: 2, BF16: 2, BOOK: 1}[typ])
            else:
                raise ValueError('unsupported tensor type')
            cursor = checked(offset + size)
            if base + cursor > len(data):
                raise ValueError('tensor outside file')
            entry.update(payload_bytes=size, absolute=base + offset)
        if base + cursor != len(data):
            raise ValueError('trailing/missing bytes')
        self.meta, self.entries, self.base = meta, entries, base
        self._validate_links()

    def raw(self, name):
        entry = self.entries[name]
        # Callers may borrow these views only while the reader remains open.
        return memoryview(self.data)[entry['absolute']:entry['absolute'] + entry['payload_bytes']]

    def _validate_links(self):
        links = _json_object(self.meta.get(NS + 'codebooks', '{}'))
        used, used_books = set(), set()
        for name, e in self.entries.items():
            typ, shape = e['type_id'], e['shape']
            raw = self.raw(name)
            if typ == VQ:
                link = str(struct.unpack_from('<I', raw, 24)[0])
                book_name = links.get(link)
                if not isinstance(book_name, str) or book_name not in self.entries:
                    raise ValueError('missing codebook link/tensor')
                book = self.entries[book_name]
                k, precision = struct.unpack_from('<II', raw, 16)
                if book['type_id'] != (BOOK if precision == 1 else F16) or book['shape'] != (k, 8):
                    raise ValueError('codebook type/shape')
                VqTensor(raw, self.raw(book_name), shape, int(link))
                used.add(link)
                used_books.add(book_name)
            elif typ in (F16, F32, BF16):
                for first in range(0, len(raw), 1024 * 1024):
                    chunk = raw[first:first + 1024 * 1024]
                    if typ == BF16:
                        finite = (np.frombuffer(chunk, '<u2') & 0x7f80) != 0x7f80
                    else:
                        finite = np.isfinite(np.frombuffer(chunk, '<f2' if typ == F16 else '<f4'))
                    if not finite.all():
                        raise ValueError('nonfinite native tensor')
            elif typ in QUANT_BLOCKS:
                _, width, scale_bytes = QUANT_BLOCKS[typ]
                for first in range(0, len(raw), width * 8192):
                    blocks = np.frombuffer(raw[first:first + width * 8192], np.uint8).reshape(-1, width)
                    if not np.isfinite(blocks[:, :scale_bytes].copy().view('<f2')).all():
                        raise ValueError('nonfinite native quant scale')
            elif typ == BOOK:
                _book_array(bytes(raw), shape[0], 1)
            raw.release()
        if used != set(links):
            raise ValueError('unused codebook links')
        books = [n for n, e in self.entries.items() if e['type_id'] == BOOK]
        if any(n not in used_books for n in books):
            raise ValueError('unused codebook')
        if len(books) > len(BOOK_SUFFIXES):
            raise ValueError('more than three codebooks in one part')
        if NS + 'layer' in self.meta:
            layer = self.meta[NS + 'layer']
            if {str(book_id(layer, n)): n for n in books} != links:
                raise ValueError('codebook IDs do not follow 4L / 4L+2 / 4L+3')

    def vq(self, name):
        raw = self.raw(name)
        link = str(struct.unpack_from('<I', raw, 24)[0])
        book_name = _json_object(self.meta[NS + 'codebooks'])[link]
        return VqTensor(raw, self.raw(book_name), self.entries[name]['shape'], int(link))

    def close(self):
        self.data.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def active_parts(manifest):
    """Every part is required: trunk parts are device resident, the embedding
    part is opened for host row reads and never uploaded whole."""
    if manifest.get('format') != VARIANT or manifest.get('format_version') != 1:
        raise ValueError('release format')
    parts, names = [], set()
    for part in manifest['parts']:
        name, role = part['path'], part['role']
        if not name or name in ('.', '..') or '/' in name or '\\' in name or name in names or role not in ROLES:
            raise ValueError('invalid/duplicate part')
        if part.get('required') is not True:
            raise ValueError('every A.X-K2 part is required')
        names.add(name)
        parts.append(part)
    if [p['role'] for p in parts].count('embedding') != 1 or not any(p['role'] == 'trunk' for p in parts):
        raise ValueError('need trunk parts and exactly one embedding part')
    return parts


def verify_source_mapping(source_names, mapping, tensors):
    """Every original tensor has one explicit preserved/consumed destination.

    Weights keep their HF names; an FP8 `*.weight_scale_inv` maps to its
    weight (the name without `_scale_inv`) as `consumed_scale`."""
    expected = set(source_names)
    targets = {row['name']: row for row in tensors}
    if len(targets) != len(tensors) or len(expected) != len(source_names):
        raise ValueError('duplicate source/output inventory')
    seen = set()
    for row in mapping:
        source, target = row['source_name'], row['target_name']
        disposition = row['disposition']
        if source not in expected or source in seen or target not in targets:
            raise ValueError('source mapping duplicate/missing/unknown')
        if disposition not in DISPOSITIONS:
            raise ValueError('source disposition')
        typ = targets[target]['type_id']
        if disposition == 'consumed_scale':
            if not source.endswith('_scale_inv') or target != source[:-len('_scale_inv')] or typ not in (VQ, Q4_K, Q8_0):
                raise ValueError('scale must be consumed by its quantized weight')
        elif source != target:
            raise ValueError('A.X-K2 keeps original tensor names')
        if disposition == 'preserved' and typ in (VQ, Q4_K, Q8_0):
            raise ValueError('quantized source must record consumption')
        if (disposition == 'consumed_vq' and typ != VQ or disposition == 'consumed_q4k' and typ != Q4_K
                or disposition == 'consumed_q8_0' and typ != Q8_0):
            raise ValueError('source disposition/type mismatch')
        seen.add(source)
    reached = {row['target_name'] for row in mapping}
    for name, record in targets.items():
        if name not in reached and not (BOOK_NAME.fullmatch(name) and record['type_id'] == BOOK):
            raise ValueError('unmapped output tensor')
    if seen != expected:
        raise ValueError('source inventory not completely accounted')


def _verify_part(args):
    """One release part: size, file hash, strict reader, role, inventory and payload hashes."""
    root, part, wanted = args
    path = Path(root) / part['path']
    if not path.is_file() or path.stat().st_size != part['bytes']:
        raise ValueError(f'missing part/size mismatch: {part["path"]}')
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != part['sha256']:
            raise ValueError(f'part checksum: {part["path"]}')
    seen = []
    with PartReader(path) as reader:
        if reader.meta[NS + 'part_role'] != part['role'] or set(reader.entries) != set(wanted):
            raise ValueError(f'part role/inventory mismatch: {part["path"]}')
        for name, entry in reader.entries.items():
            record = wanted[name]
            if tuple(record['stored_shape']) != entry['shape']:
                raise ValueError(f'manifest shape: {name}')
            if any(record[key] != entry[key] for key in ('type_id', 'payload_bytes')):
                raise ValueError(f'manifest type/size: {name}')
            raw = reader.raw(name)
            digest = hashlib.sha256(raw).hexdigest()
            raw.release()
            if digest != record['payload_sha256']:
                raise ValueError(f'tensor checksum: {name}')
            seen.append(name)
    return seen


def verify_release(root, workers=1):
    """Hash all files/payloads and check the exact named inventory (parts in parallel with workers > 1)."""
    root = Path(root)
    release = _json_object((root / 'release.json').read_text())
    manifest = _json_object((root / 'tensor-manifest.json').read_text())
    selected = active_parts(release)
    all_parts = {p['path'] for p in release['parts']}
    expected = {}
    for record in manifest['tensors']:
        name = record['name']
        if name in expected or record['part'] not in all_parts:
            raise ValueError('duplicate tensor/unknown manifest part')
        role = next(p['role'] for p in release['parts'] if p['path'] == record['part'])
        if (name == 'model.embed_tokens.weight') != (role == 'embedding'):
            raise ValueError('embedding/trunk part tensor mismatch')
        expected[name] = record
    jobs = [(str(root), part, {n: r for n, r in expected.items() if r['part'] == part['path']}) for part in selected]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_verify_part, jobs))
    else:
        results = [_verify_part(j) for j in jobs]
    actual = set()
    for names in results:
        if actual & set(names):
            raise ValueError('duplicate tensor across parts')
        actual |= set(names)
    if actual != set(expected):
        raise ValueError('release missing/extra tensor')
    if 'source_mapping' in manifest and 'source_names' in manifest:
        verify_source_mapping(manifest['source_names'], manifest['source_mapping'], manifest['tensors'])
    return {'parts': len(selected), 'tensors': len(actual), 'bytes': sum(p['bytes'] for p in selected),
            'status': 'verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', type=Path)
    parser.add_argument('--release', action='store_true', help='validate a release directory')
    parser.add_argument('--workers', type=int, default=16, help='parallel part checks for --release')
    args = parser.parse_args()
    if args.release:
        print(json.dumps(verify_release(args.part, args.workers)))
        return
    with PartReader(args.part) as reader:
        print(json.dumps({'part': args.part.name, 'bytes': len(reader.data), 'tensors': len(reader.entries),
                          'architecture': reader.meta['general.architecture'], 'status': 'validated'}))


if __name__ == '__main__':
    main()
