#!/usr/bin/env python3
"""Inkling-2-Small VQ (inkling2_vq) GVQ1 v1 byte format and independent, mmap-backed CPU validation.

Adapted from the MiniMax-M3.1-Flash VQ release tool (Baekpica/MiniMax-M3.1-Flash-VQ-GGUF
tools/minimax_vq_format.py at 2c37c3a3, itself from Beam-1.0's beam_vq_format.py): the GVQ1 v1 matrix
container is byte-for-byte Beam's / MiniMax's (FORMAT.md: 128-byte header `<4sHBBIIIIIIQQQQ32s32s`,
F16 row gains, 12-bit pair stream, 128-byte aligned regions); the family namespace is `inkling2_vq`.
The GVR8 residual code path is kept from the source module for container parity but the Inkling-2
recipe uses none (every part is format version 1, 12-bit).

Expert parts (`layer-LL-experts.gguf`, LL = 00..47) hold one E4M3 codebook
`model.llm.layers.L.mlp.vq_codebook` (ID 4*L, [4096, 8]) and the 1,152 GVQ1 matrices
`model.llm.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` (expert-major; gate = rows 0::2 and
up = rows 1::2 of the source `w13_weight`, [2304, 4608]; down = `w2_weight`, [4608, 2304]).

    python3 inkling2_vq_format.py PART.gguf [--receipt receipts/layer-LL-experts.json]
    python3 inkling2_vq_format.py --parts $W/parts [--layers 0-47] [--require-all]

The strict reader rejects an unsupported family/version/variant, foreign or unknown metadata
namespaces, duplicate names or keys, gaps, overlaps, truncation, trailing bytes, bad shapes, types or
alignment, non-zero padding, invalid gains or codebooks, noncanonical GVQ1 offsets, and missing,
unused or hash-mismatched codebook links. The expert-part check adds the exact metadata set, the
exact tensor inventory and order, the per-layer codebook ID and index width, and (with a receipt)
the file and every payload SHA-256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import numpy as np

ALIGN = HEADER = 128
F32, F16, Q8_0, Q4_K, BF16 = 0, 1, 8, 12, 30
FP8, BOOK, VQ, RES = 0x47514638, 0x47564342, 0x47565138, 0x47565238
RES_K, RES_BITS = 256, 8
VERSIONS = (1, 2)            # 2: R13+8 parts (GVR8 residual stage)
ARCH = 'inkling2_vq'
NS = ARCH + '.'
VARIANT = 'inkling2-vq8'
SOURCE_MODEL_TYPE = 'inkling2'
SOURCE_REVISION = '086a047e8b7ff2ac902dd52355328ddb66b16914'
RECIPE_ID = 'inkling2s-hvq8-r12x48-q4k-nvfp4kv-ssdkv-embssd-v1'
ROLES = ('trunk', 'embedding', 'vision', 'drafter')     # as mm31_nx_format.py (body, embedding, vision, DSpark parts)
FOREIGN = ('beam_vq.', 'glm_vq.', 'ling_vq.', 'axk2_vq.', 'ds41_vq.', 'minimax_m3_vq.', 'split.')
U64_MAX = 2**64 - 1
GVQ_HEADER = struct.Struct('<4sHBBIIIIIIQQQQ32s32s')

# Inkling-2-Small routed experts
MOE_LAYERS = tuple(range(0, 48))
N_EXPERTS = 384
HIDDEN, FFN = 4608, 2304
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')   # stored order per expert
SHAPES = {'gate_proj': (FFN, HIDDEN), 'up_proj': (FFN, HIDDEN), 'down_proj': (HIDDEN, FFN)}
PREFIX = 'model.llm.layers.{L}.mlp.'


def book_name(layer):
    return PREFIX.format(L=layer) + 'vq_codebook'


def matrix_name(layer, expert, proj):
    return PREFIX.format(L=layer) + f'experts.{expert}.{proj}.weight'


def book_id(layer):
    return 4 * layer


def res_book_name(layer):
    return PREFIX.format(L=layer) + 'vq_codebook_residual'


def res_book_id(layer):
    return 4 * layer + 1      # GLM's residual book ID, Beam's reserved slot


def residual_name(matrix):
    return matrix + '_residual'         # '<matrix>.weight_residual'


def expert_inventory(layer, residual=False):
    """[(name, shape, type_id)] of the matrices of one expert part, in stored order (after the books)."""
    out = []
    for e in range(N_EXPERTS):
        names = [(matrix_name(layer, e, p), SHAPES[p]) for p in PROJECTIONS]
        out += [(n, s, VQ) for n, s in names]
        if residual:
            out += [(residual_name(n), s, RES) for n, s in names]
    return out


def checked(n):
    if not isinstance(n, int) or not 0 <= n <= U64_MAX:
        raise ValueError('u64 overflow')
    return n


def align(n):
    return checked(checked(n + ALIGN - 1) // ALIGN * ALIGN)


def geometry(rows, cols, k):
    if not 0 < rows < 2**32 or not 0 < cols < 2**32 or cols % 64 or k not in (4096, 8192):
        raise ValueError('unsupported GVQ1 shape/K')
    elements = checked(rows * cols)
    main = align(HEADER + 2 * rows)
    main_end = checked(main + elements * 3 // 16)
    plane = align(main_end) if k == 8192 else 0
    end = checked(plane + elements // 64) if plane else main_end
    return dict(gain=HEADER, main=main, plane=plane, size=align(end),
                main_end=main_end, end=end, low_stride=cols * 3 // 16,
                high_stride=cols // 64 if plane else 0)


def encode_vq(gains, indices, rows, cols, k, codebook, book_id, precision=1):
    """Encode rounded gains/indices and raw stored codebook bytes (independent numpy packer).

    gains must already have been rounded to F16 before normalization, and
    indices assigned against this exact codebook precision. No training here.
    """
    g = geometry(rows, cols, k)
    source_gains = np.asarray(gains)
    if source_gains.dtype.kind not in 'fiu':
        raise ValueError('gain must be real numeric values')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        gains = source_gains.astype('<f2')
    ix = np.asarray(indices)
    if gains.shape != (rows,) or not np.isfinite(gains).all() or np.signbit(gains).any():
        raise ValueError('gain must be finite nonnegative F16 per output row')
    if not np.array_equal(source_gains, gains):
        raise ValueError('gain must already be rounded to F16 before normalization')
    if ix.shape not in ((rows, cols // 8), (rows * cols // 8,)) or ix.dtype.kind not in 'iu':
        raise ValueError('index shape/type')
    if (ix < 0).any() or (ix >= k).any():
        raise ValueError('index outside codebook')
    _book_array(codebook, k, precision)
    ix = ix.astype('<u2', copy=False).reshape(-1)
    out = bytearray(g['size'])
    out[HEADER:HEADER + 2 * rows] = gains.tobytes()
    pair = ix.reshape(-1, 2)
    low = np.empty((len(pair), 3), dtype=np.uint8)
    low[:, 0] = pair[:, 0] & 255
    low[:, 1] = ((pair[:, 0] >> 8) & 15) | ((pair[:, 1] & 15) << 4)
    low[:, 2] = (pair[:, 1] >> 4) & 255
    out[g['main']:g['main_end']] = low.tobytes()
    if g['plane']:
        out[g['plane']:g['end']] = np.packbits((ix >> 12).astype(np.uint8), bitorder='little').tobytes()
    out[:HEADER] = GVQ_HEADER.pack(b'GVQ1', 1, 12 if k == 4096 else 13, 8,
        rows, cols, k, precision, book_id, 0, HEADER, g['main'], g['plane'], g['size'],
        hashlib.sha256(codebook).digest(), hashlib.sha256(memoryview(out)[HEADER:]).digest())
    return bytes(out)


def assemble_vq(gain_bytes, low_bytes, high_bytes, rows, cols, k, codebook, book_id, precision=1):
    """GVQ1 payload from already-packed streams (device packing). The result
    is cross-checked against encode_vq() on sampled tensors by the converter."""
    g = geometry(rows, cols, k)
    if len(gain_bytes) != 2 * rows or len(low_bytes) != g['main_end'] - g['main']:
        raise ValueError('gain/main stream size')
    if (high_bytes is None) != (g['plane'] == 0) or high_bytes is not None and len(high_bytes) != g['end'] - g['plane']:
        raise ValueError('plane stream size')
    gains = np.frombuffer(gain_bytes, '<f2')
    if not np.isfinite(gains).all() or np.signbit(gains).any():
        raise ValueError('gain must be finite nonnegative F16 per output row')
    _book_array(codebook, k, precision)
    out = bytearray(g['size'])
    out[HEADER:HEADER + 2 * rows] = gain_bytes
    out[g['main']:g['main_end']] = low_bytes
    if g['plane']:
        out[g['plane']:g['end']] = high_bytes
    out[:HEADER] = GVQ_HEADER.pack(b'GVQ1', 1, 12 if k == 4096 else 13, 8,
        rows, cols, k, precision, book_id, 0, HEADER, g['main'], g['plane'], g['size'],
        hashlib.sha256(codebook).digest(), hashlib.sha256(memoryview(out)[HEADER:]).digest())
    return bytes(out)


def res_geometry(rows, cols):
    """GVR8: 128-byte header, rows * cols / 8 index bytes (row-major), zero padding to 128."""
    if not 0 < rows < 2**32 or not 0 < cols < 2**32 or cols % 64:
        raise ValueError('unsupported GVR8 shape')
    end = checked(HEADER + checked(rows * (cols // 8)))
    return dict(index=HEADER, end=end, size=align(end), stride=cols // 8)


def _gvr_header(rows, cols, book_id, size, codebook, body):
    return GVQ_HEADER.pack(b'GVR1', 1, RES_BITS, 8, rows, cols, RES_K, 1, book_id, 0, HEADER, 0, 0, size,
                           hashlib.sha256(codebook).digest(), hashlib.sha256(body).digest())


def encode_gvr(indices, rows, cols, codebook, book_id):
    """Independent numpy GVR8 encoder: K=256 residual indices -> payload."""
    g = res_geometry(rows, cols)
    ix = np.asarray(indices)
    if ix.shape not in ((rows, cols // 8), (rows * cols // 8,)) or ix.dtype.kind not in 'iu':
        raise ValueError('residual index shape/type')
    if (ix < 0).any() or (ix >= RES_K).any():
        raise ValueError('residual index outside codebook')
    _book_array(codebook, RES_K, 1)
    out = bytearray(g['size'])
    out[HEADER:g['end']] = ix.astype(np.uint8).reshape(-1).tobytes()
    out[:HEADER] = _gvr_header(rows, cols, book_id, g['size'], codebook, memoryview(out)[HEADER:])
    return bytes(out)


def assemble_gvr(index_bytes, rows, cols, codebook, book_id):
    """GVR8 payload from device index bytes (cross-checked against encode_gvr by the converter)."""
    g = res_geometry(rows, cols)
    if len(index_bytes) != g['end'] - HEADER:
        raise ValueError('residual stream size')
    _book_array(codebook, RES_K, 1)
    out = bytearray(g['size'])
    out[HEADER:g['end']] = index_bytes
    out[:HEADER] = _gvr_header(rows, cols, book_id, g['size'], codebook, memoryview(out)[HEADER:])
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
        if magic != b'GVQ1' or version != 1 or vector != 8 or flags != 0:
            raise ValueError('unsupported GVQ1 header')
        g = geometry(rows, cols, k)
        if bits != (12 if k == 4096 else 13) or (gain, main, plane, size) != tuple(g[n] for n in ('gain', 'main', 'plane', 'size')):
            raise ValueError('noncanonical GVQ1 offsets')
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
        for start, end in [(HEADER + 2 * rows, main), (g['main_end'], plane or g['main_end']), (g['end'], size)]:
            if any(memoryview(payload)[start:end]):
                raise ValueError('nonzero padding')
        self.payload, self.rows, self.cols, self.k, self.precision = payload, rows, cols, k, precision
        self.bits = bits
        self.book_id, self.geometry = link, g

    def indices(self, first=0, count=None):
        count = self.rows - first if count is None else count
        if first < 0 or count < 0 or first + count > self.rows:
            raise ValueError('row range')
        g, nv = self.geometry, self.cols // 8
        low = np.frombuffer(self.payload, np.uint8, count * g['low_stride'], g['main'] + first * g['low_stride'])
        # Independent bit expansion, deliberately not the writer's paired shifts.
        bits = np.unpackbits(low, bitorder='little').reshape(count, nv, 12)
        ix = np.sum(bits.astype(np.uint16) << np.arange(12, dtype=np.uint16), axis=-1, dtype=np.uint16)
        if g['plane']:
            high = np.frombuffer(self.payload, np.uint8, count * g['high_stride'], g['plane'] + first * g['high_stride'])
            ix |= np.unpackbits(high, bitorder='little').reshape(count, nv).astype(np.uint16) << 12
        return ix

    def decode_rows(self, first=0, count=None):
        count = self.rows - first if count is None else count
        ix = self.indices(first, count)
        return (self.book[ix].reshape(count, self.cols) * self.gains[first:first + count, None]).astype('<f4')


class ResTensor:
    """GVR8 residual stage of a GVQ1 matrix (GLM container v2): validated header, borrowed bytes."""
    def __init__(self, payload, book, shape=None, book_id=None):
        if len(payload) < HEADER:
            raise ValueError('truncated GVR1 header')
        (magic, version, bits, vector, rows, cols, k, precision, link, flags,
         index, r0, r1, size, book_hash, payload_hash) = GVQ_HEADER.unpack_from(payload)
        if magic != b'GVR1' or version != 1 or bits != RES_BITS or vector != 8 or k != RES_K \
                or precision != 1 or flags != 0:
            raise ValueError('unsupported GVR1 header')
        g = res_geometry(rows, cols)
        if (index, r0, r1, size) != (HEADER, 0, 0, g['size']):
            raise ValueError('noncanonical GVR1 offsets')
        if len(payload) != size or shape is not None and tuple(shape) != (rows, cols):
            raise ValueError('residual payload size/shape mismatch')
        if book_id is not None and link != book_id:
            raise ValueError('residual codebook ID mismatch')
        if hashlib.sha256(book).digest() != book_hash:
            raise ValueError('residual codebook checksum')
        if hashlib.sha256(memoryview(payload)[HEADER:]).digest() != payload_hash:
            raise ValueError('residual payload checksum')
        if any(memoryview(payload)[g['end']:size]):
            raise ValueError('nonzero residual padding')
        self.book = _book_array(book, RES_K, 1)
        self.payload, self.rows, self.cols, self.book_id, self.geometry = payload, rows, cols, link, g

    def indices(self, first=0, count=None):
        count = self.rows - first if count is None else count
        if first < 0 or count < 0 or first + count > self.rows:
            raise ValueError('row range')
        g = self.geometry
        return np.frombuffer(self.payload, np.uint8, count * g['stride'], HEADER + first * g['stride']) \
            .reshape(count, self.cols // 8)


def decode_r13p8_rows(vq, res, first=0, count=None):
    """(C1[i1] + C2[i2]) * gain in FP32 (the two E4M3 words add exactly), as GLM's restore."""
    count = vq.rows - first if count is None else count
    if (res.rows, res.cols) != (vq.rows, vq.cols):
        raise ValueError('residual/parent shape')
    words = vq.book[vq.indices(first, count)] + res.book[res.indices(first, count)]
    return (words.reshape(count, vq.cols) * vq.gains[first:first + count, None]).astype('<f4')


def metadata(codebooks=None, part_role='trunk', extra=None, format_version=1):
    if part_role not in ROLES:
        raise ValueError('part role')
    out = {'general.architecture': ARCH, 'general.alignment': ALIGN,
           NS + 'format_version': format_version, NS + 'variant': VARIANT,
           NS + 'source_model_type': SOURCE_MODEL_TYPE, NS + 'part_role': part_role,
           NS + 'codebooks': json.dumps(codebooks or {}, sort_keys=True)}
    if format_version not in VERSIONS:
        raise ValueError('format version')
    if extra:
        if set(out) & set(extra):
            raise ValueError('cannot override family metadata')
        if any(not k.startswith(NS) for k in extra):
            raise ValueError('extra metadata must use the inkling2_vq namespace')
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
            if typ == 4 and (not isinstance(value, int) or not 0 <= value < 2**32):
                raise ValueError(f'metadata {key}: u32 expected')
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
    """Strict inkling2_vq family reader. Validation scans payloads without expansion."""
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
                    NS + 'variant': VARIANT, NS + 'source_model_type': SOURCE_MODEL_TYPE}
        if any(meta.get(k) != v or meta_types.get(k) != (8 if isinstance(v, str) else 4) for k, v in required.items()) \
                or meta.get(NS + 'format_version') not in VERSIONS or meta_types.get(NS + 'format_version') != 4 \
                or meta.get(NS + 'part_role') not in ROLES or meta_types.get(NS + 'part_role') != 8:
            raise ValueError('unsupported inkling2_vq family/version')
        if any(k.startswith(FOREIGN) for k in meta):
            raise ValueError('foreign family/standard split metadata')
        if any(not (k.startswith(NS) or k.startswith('general.')) for k in meta):
            raise ValueError('metadata outside the general/inkling2_vq namespaces')
        entries = {}
        self.order = []
        for _ in range(count):
            name, rank = string(), unpack('I')[0]
            if not name or name in entries or not 1 <= rank <= 4:
                raise ValueError('duplicate tensor/invalid rank')
            dims = unpack('Q' * rank)
            typ, offset = unpack('IQ')
            if 0 in dims or offset % ALIGN:
                raise ValueError('tensor shape/alignment')
            entries[name] = dict(name=name, shape=tuple(reversed(dims)), type_id=typ, offset=offset)
            self.order.append(name)
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
            elif typ == RES:
                if len(shape) != 2:
                    raise ValueError('GVR8 shape')
                size = res_geometry(*shape)['size']
            elif typ in (Q4_K, Q8_0):
                block, width = (256, 144) if typ == Q4_K else (32, 34)
                if len(shape) != 2 or shape[1] % block:
                    raise ValueError('native quant shape')
                size = checked(elements // block * width)
            elif typ in (F32, F16, BF16, BOOK):
                if typ == BOOK and (len(shape) != 2 or shape[1] != 8):
                    raise ValueError('codebook shape')
                size = checked(elements * {F32: 4, F16: 2, BF16: 2, BOOK: 1}[typ])
            else:
                raise ValueError('unsupported tensor type')
            cursor = checked(offset + size)
            if base + cursor > len(data):
                raise ValueError('tensor outside file')
            entry.update(payload_bytes=size, absolute=base + offset)
        if base + cursor != len(data):
            raise ValueError('trailing/missing bytes')
        self.meta, self.meta_types, self.entries, self.base = meta, meta_types, entries, base
        self._validate_links()

    def raw(self, name):
        entry = self.entries[name]
        # Callers may borrow these views only while the reader remains open.
        return memoryview(self.data)[entry['absolute']:entry['absolute'] + entry['payload_bytes']]

    def _validate_links(self):
        links = _json_object(self.meta.get(NS + 'codebooks', '{}'))
        if any(not isinstance(k, str) or not k.isdigit() or str(int(k)) != k for k in links):
            raise ValueError('codebook link keys must be canonical decimal IDs')
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
            elif typ == RES:
                link = str(struct.unpack_from('<I', raw, 24)[0])
                book_name = links.get(link)
                if not isinstance(book_name, str) or book_name not in self.entries:
                    raise ValueError('missing residual codebook link/tensor')
                book = self.entries[book_name]
                if book['type_id'] != BOOK or book['shape'] != (RES_K, 8):
                    raise ValueError('residual codebook type/shape')
                ResTensor(raw, self.raw(book_name), shape, int(link))
                used.add(link)
                used_books.add(book_name)
            elif typ == BOOK:
                _book_array(bytes(raw), shape[0], 1)
            elif typ in (F16, F32, BF16):
                for first in range(0, len(raw), 1024 * 1024):
                    chunk = raw[first:first + 1024 * 1024]
                    if typ == BF16:
                        finite = (np.frombuffer(chunk, '<u2') & 0x7f80) != 0x7f80
                    else:
                        finite = np.isfinite(np.frombuffer(chunk, '<f2' if typ == F16 else '<f4'))
                    if not finite.all():
                        raise ValueError('nonfinite native tensor')
            elif typ in (Q4_K, Q8_0):
                width, scale_bytes = (144, 4) if typ == Q4_K else (34, 2)
                for first in range(0, len(raw), width * 8192):
                    blocks = np.frombuffer(raw[first:first + width * 8192], np.uint8).reshape(-1, width)
                    if not np.isfinite(blocks[:, :scale_bytes].copy().view('<f2')).all():
                        raise ValueError('nonfinite native quant scale')
            raw.release()
        if used != set(links):
            raise ValueError('unused codebook links')
        if any(e['type_id'] == BOOK and n not in used_books for n, e in self.entries.items()):
            raise ValueError('unused codebook')
        self._validate_residuals()

    def _validate_residuals(self):
        """GLM container v2: residual tensors need version 2 and `residuals` links; all-or-none per part:
        every 13-bit GVQ1 matrix has exactly one residual of its shape, every residual one parent."""
        key = NS + 'residuals'
        if key in self.meta and self.meta_types[key] != 8:
            raise ValueError('residual links must be a JSON string')
        links = _json_object(self.meta.get(key, '{}'))
        residuals = [n for n, e in self.entries.items() if e['type_id'] == RES]
        self.residuals = {}
        if not residuals and not links:
            return
        if self.meta[NS + 'format_version'] < 2:
            raise ValueError('residual stage needs format version 2')
        linked = set()
        for parent, child in links.items():
            p, c = self.entries.get(parent), self.entries.get(child) if isinstance(child, str) else None
            if p is None or c is None:
                raise ValueError('missing residual parent/tensor')
            raw = self.raw(parent)
            k = struct.unpack_from('<I', raw, 16)[0]
            raw.release()
            if p['type_id'] != VQ or c['type_id'] != RES or k != 8192 or p['shape'] != c['shape'] or child in linked:
                raise ValueError('residual link type/shape/duplicate')
            linked.add(child)
        parents = [n for n, e in self.entries.items() if e['type_id'] == VQ]
        if len(linked) != len(residuals) or len(residuals) != len(parents):
            raise ValueError('partial or unlinked residual stage')
        self.residuals = dict(links)

    def vq(self, name):
        raw = self.raw(name)
        link = str(struct.unpack_from('<I', raw, 24)[0])
        bname = _json_object(self.meta[NS + 'codebooks'])[link]
        return VqTensor(raw, self.raw(bname), self.entries[name]['shape'], int(link))

    def res(self, name):
        """The GVR8 residual of GVQ1 matrix `name`, or None (R13 part)."""
        child = self.residuals.get(name)
        if child is None:
            return None
        raw = self.raw(child)
        link = str(struct.unpack_from('<I', raw, 24)[0])
        bname = _json_object(self.meta[NS + 'codebooks'])[link]
        return ResTensor(raw, self.raw(bname), self.entries[child]['shape'], int(link))

    def close(self):
        self.data.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


EXPERT_META_KEYS = {'general.architecture': 8, 'general.alignment': 4, NS + 'format_version': 4,
                    NS + 'variant': 8, NS + 'source_model_type': 8, NS + 'part_role': 8, NS + 'codebooks': 8,
                    NS + 'source_revision': 8, NS + 'recipe_id': 8, NS + 'layer': 4, NS + 'index_bits': 4,
                    NS + 'weighting': 8}
R13P8_META_KEYS = {**EXPERT_META_KEYS, NS + 'residuals': 8}


def verify_expert_part(path, layer=None, receipt=None, recipe_id=RECIPE_ID, decode_rows=0):
    """Strict check of one layer-LL-experts.gguf (R13: format version 1; R13+8: version 2): the family
    reader plus the exact expert-part contract. receipt: the converter's receipt dict (file and every
    payload hash). -> summary dict."""
    path = Path(path)
    if layer is None:
        stem = path.name
        if not (stem.startswith('layer-') and stem.endswith('-experts.gguf')):
            raise ValueError('expert part file name must be layer-LL-experts.gguf')
        layer = int(stem[len('layer-'):-len('-experts.gguf')])
    if path.name != f'layer-{layer:02d}-experts.gguf':
        raise ValueError('expert part file name/layer mismatch')
    if layer not in MOE_LAYERS:
        raise ValueError(f'layer {layer} is not a MoE layer')
    with PartReader(path) as r:
        m, mt = r.meta, r.meta_types
        version = m[NS + 'format_version']
        residual = version == 2
        keys = R13P8_META_KEYS if residual else EXPERT_META_KEYS
        if set(m) != set(keys) or any(mt[k] != t for k, t in keys.items()):
            raise ValueError(f'expert part metadata set/types: {sorted(set(m) ^ set(keys))}')
        bits = m[NS + 'index_bits']
        if bits not in (12, 13) or residual and bits != 13:
            raise ValueError('index_bits')
        k = 1 << bits
        want = {NS + 'part_role': 'trunk', NS + 'source_revision': SOURCE_REVISION, NS + 'layer': layer}
        if recipe_id is not None:
            want[NS + 'recipe_id'] = recipe_id
        if any(m[key] != v for key, v in want.items()):
            raise ValueError('expert part role/revision/recipe/layer metadata')
        bname, bid = book_name(layer), book_id(layer)
        books = [(bname, (k, 8), BOOK)]
        links = {str(bid): bname}
        if residual:
            books.append((res_book_name(layer), (RES_K, 8), BOOK))
            links[str(res_book_id(layer))] = res_book_name(layer)
        if _json_object(m[NS + 'codebooks']) != links:
            raise ValueError('codebook link metadata')
        if residual and _json_object(m[NS + 'residuals']) != {n: residual_name(n) for n, _, t in
                                                              expert_inventory(layer) if t == VQ}:
            raise ValueError('residual link metadata')
        weighting = _json_object(m[NS + 'weighting'])
        inventory = books + expert_inventory(layer, residual)
        if r.order != [n for n, _, _ in inventory]:
            raise ValueError('expert part tensor inventory/order')
        offsets = [r.entries[n]['offset'] for n in r.order]
        if offsets != sorted(offsets) or offsets[0] != 0:
            raise ValueError('tensor descriptors not in storage order')
        geo = {p: geometry(*SHAPES[p], k)['size'] for p in PROJECTIONS}
        rgeo = {p: res_geometry(*SHAPES[p])['size'] for p in PROJECTIONS}
        for name, shape, typ in inventory:
            e = r.entries[name]
            if e['shape'] != shape or e['type_id'] != typ:
                raise ValueError(f'{name}: shape/type')
            if typ in (VQ, RES):
                raw = r.raw(name)
                hb, hv, hk, hprec, hlink = raw[6], raw[7], *struct.unpack_from('<III', raw, 16)
                raw.release()
                proj = name.split('.')[-2]
                want_h = (bits, 8, k, 1, bid, geo[proj]) if typ == VQ else (RES_BITS, 8, RES_K, 1, res_book_id(layer), rgeo[proj])
                if (hb, hv, hk, hprec, hlink, e['payload_bytes']) != want_h:
                    raise ValueError(f'{name}: header bits/K/precision/link/size')
        shas = {}
        for name, _, _ in books:
            raw = r.raw(name)
            shas[name] = hashlib.sha256(raw).hexdigest()
            raw.release()
        decoded = 0
        if decode_rows:
            rng = np.random.default_rng(layer)
            for name, shape, typ in inventory:
                if typ != VQ:
                    continue
                t, res = r.vq(name), r.res(name)
                for row in rng.integers(0, shape[0], decode_rows):
                    out = decode_r13p8_rows(t, res, int(row), 1) if res is not None else t.decode_rows(int(row), 1)
                    if not np.isfinite(out).all():
                        raise ValueError(f'{name}: nonfinite decode')
                    decoded += 1
                del t, res          # release the mmap views before the reader closes
        if receipt is not None:
            if receipt.get('path') != path.name or receipt.get('layer') != layer or receipt.get('index_bits') != bits:
                raise ValueError('receipt path/layer/bits')
            if receipt.get('bytes') != len(r.data):
                raise ValueError('receipt file size')
            if receipt.get('codebook_sha256') != shas[bname] or receipt.get('codebook_name') != bname \
                    or receipt.get('codebook_id') != bid:
                raise ValueError('receipt codebook')
            if residual and (receipt.get('residual_codebook_sha256') != shas[res_book_name(layer)]
                             or receipt.get('residual_codebook_name') != res_book_name(layer)
                             or receipt.get('residual_codebook_id') != res_book_id(layer)):
                raise ValueError('receipt residual codebook')
            rows = receipt.get('tensors') or []
            if [t['name'] for t in rows] != [n for n, _, _ in inventory[len(books):]]:
                raise ValueError('receipt tensor inventory')
            for t in rows:
                raw = r.raw(t['name'])
                digest = hashlib.sha256(raw).hexdigest()
                size = len(raw)
                raw.release()
                if digest != t['payload_sha256'] or size != t['payload_bytes']:
                    raise ValueError(f"{t['name']}: payload hash/size differs from the receipt")
            data_sha = hashlib.sha256(r.data).hexdigest()
            if data_sha != receipt.get('sha256'):
                raise ValueError('receipt file checksum')
        return {'part': path.name, 'layer': layer, 'variant': 'R13+8' if residual else f'R{bits}',
                'format_version': version, 'index_bits': bits, 'bytes': len(r.data),
                'tensors': len(r.entries), 'codebook_sha256': shas[bname],
                'residual_codebook_sha256': shas.get(res_book_name(layer)),
                'weighting_mode': weighting.get('mode'), 'decoded_rows_checked': decoded,
                'receipt_checked': receipt is not None, 'status': 'verified'}


def parse_layers(spec):
    out = set()
    for part in (spec or '').split(','):
        if part:
            lo, _, hi = part.partition('-')
            out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('part', type=Path, nargs='?')
    parser.add_argument('--receipt', type=Path, help='converter receipt of PART (file and payload hashes)')
    parser.add_argument('--parts', type=Path, help='verify every layer-LL-experts.gguf in this directory '
                        'against receipts/layer-LL-experts.json')
    parser.add_argument('--layers', help='with --parts: subset, e.g. 0-47')
    parser.add_argument('--require-all', action='store_true', help='with --parts: every MoE layer 0..47 must exist')
    parser.add_argument('--decode-rows', type=int, default=0, help='decode this many random rows per matrix')
    parser.add_argument('--any-recipe', action='store_true', help='do not pin the recipe id')
    args = parser.parse_args()
    rid = None if args.any_recipe else RECIPE_ID
    if args.parts:
        layers = parse_layers(args.layers) if args.layers else list(MOE_LAYERS)
        found = [l for l in layers if (args.parts / f'layer-{l:02d}-experts.gguf').exists()]
        if args.require_all and found != layers:
            raise SystemExit(json.dumps({'status': 'missing', 'missing_layers': sorted(set(layers) - set(found))}))
        stray = sorted(p.name for p in args.parts.glob('layer-*-experts.gguf*') if not p.name.endswith('.gguf'))
        results = []
        for l in found:
            rp = args.parts / 'receipts' / f'layer-{l:02d}-experts.json'
            if not rp.exists():
                raise SystemExit(f'layer {l}: no receipt {rp}')
            res = verify_expert_part(args.parts / f'layer-{l:02d}-experts.gguf', l, json.loads(rp.read_text()),
                                     rid, args.decode_rows)
            print(json.dumps(res), flush=True)
            results.append(res)
        print(json.dumps({'status': 'verified', 'layers': [r['layer'] for r in results],
                          'bytes': sum(r['bytes'] for r in results), 'stray_files': stray}))
        return
    if args.part is None:
        parser.error('PART or --parts required')
    name = args.part.name
    if name.startswith('layer-') and name.endswith('-experts.gguf'):
        receipt = json.loads(args.receipt.read_text()) if args.receipt else None
        print(json.dumps(verify_expert_part(args.part, None, receipt, rid, args.decode_rows)))
        return
    with PartReader(args.part) as reader:
        print(json.dumps({'part': name, 'bytes': len(reader.data), 'tensors': len(reader.entries),
                          'architecture': reader.meta['general.architecture'], 'status': 'validated'}))


if __name__ == '__main__':
    main()
