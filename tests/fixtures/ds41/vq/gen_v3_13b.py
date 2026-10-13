#!/usr/bin/env python3
# gen_v3_13b.py — P4-1 gate fixtures for the V4.1 VQ expert decode kernels.
#
# Writes deterministic DQVL blobs plus probe lists for tests/test_ds41_vq.cu:
#
#   v3_13b.blob       DQVL v3, nc=8192, 13-bit with the bit plane (flags=3),
#                     e=0 with all three matrices; real column geometry
#                     (5120 -> 20 rounds, 2304 -> 9 rounds), 128 rows.
#   v3_13b.probes.txt + v3_13b.ref.f32
#   v3_12b.blob       DQVL v3, nc=4096, 12-bit, no plane (flags=1).
#   v3_12b.probes.txt + v3_12b.ref.f32
#   v2_12b.blob       DQVL v2, nc=4096 (12-bit bit-window), 2304 cols.
#   v2_12b.probes.txt (the oracle for v2 is the vendored ds4vq_dequant_f32,
#                     computed inside the test — the engine's own host decoder)
#   v2_11b.blob       DQVL v2, nc=2048 (11-bit), 5120 cols.
#   v2_11b.probes.txt
#
# The layout is written from the spec, not from a decoder: v3 geometry is
# cuda_vq_row.inc.cu:19-49 (m.gr = pay+32, m.ix = m.gr + rows*2, plane at
# m.ix + rows*mrow, mrow = nidx_row*12/8) and the plane word rule is the
# kernel's round k bit `lane` of block word k; v2 geometry is vq_fmt.h:57-88.
# ref.f32 is computed from THIS script's own index/codebook/gain arrays, never
# from a decoder, so the reference is independent of the kernels under test.
#
# Probe lines: `e which row n col` — n consecutive rows from `row`, all dotted
# against a one-hot activation at `col`; ref.f32 carries the n values of each
# line in order. Probe points cover row 0 / mid / last, all three matrices,
# block boundaries (k = 256, 512), and (13-bit) both plane states, asserted
# below so a probe set that stops exercising the plane path fails loudly.
#
#   python3 gen_v3_13b.py     (writes next to this script)
import os
import struct
import sys

DIM = 8
BLOB_MAGIC = 0x4C565144  # 'DQVL'
MAT_MAGIC = 0x51565144   # 'DQVQ' (v2)
MAT3_MAGIC = 0x33565144  # 'DQV3'
GAIN_CYCLE = [0.25, 0.5, 1.0, 1.5, 2.0]
HERE = os.path.dirname(os.path.abspath(__file__))


def e4m3_to_f32(b):
    """E4M3FN byte -> float (bias 7, 3 mantissa bits; 0x7F/0xFF are NaN and
    never generated here)."""
    s = -1.0 if (b & 0x80) else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * (m / 8.0) * (2.0 ** -6)
    return s * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def f16(v):
    return struct.pack('<e', v)


def f16_to_f32(b):
    return struct.unpack('<e', b)[0]


def cb_byte(k):
    """Deterministic E4M3 codebook byte: 0x01..0x7E or 0x81..0xFE, never a
    zero (a +/-0 codeword would make the compare sign-of-zero dependent) and
    never 0x7F/0xFF (NaN)."""
    v = (k * 29) % 127
    if v == 0:
        v = 1
    return v | ((k & 1) << 7)


def cb_half(k):
    """Deterministic f16 codebook value (v2): nonzero, exact halves."""
    v = ((k % 100) + 1) * 0.25
    return -v if (k & 1) else v


def idx_of(which, row, k, nc):
    return (row * 131 + k * 37 + which * 911) % nc


def write_bits(buf, bitpos, value, nbits):
    for b in range(nbits):
        if (value >> b) & 1:
            buf[(bitpos + b) >> 3] |= 1 << ((bitpos + b) & 7)


def gains_bytes(rows):
    return b''.join(f16(GAIN_CYCLE[r % len(GAIN_CYCLE)]) for r in range(rows))


def build_v3(which_geo, nc, plane, rows):
    """which_geo: list of (which, cols) present in slot order."""
    flags = 1 | (2 if plane else 0)
    nbit = max(1, (nc - 1).bit_length())
    assert nbit in (12, 13) and plane == (nbit > 12)
    nexp = 1
    cb_off = 16 + nexp * 3 * 8
    cb = bytes(cb_byte(k) for k in range(nc * DIM))
    payloads = []
    idxs = {}  # (which, row, k) -> idx, kept for the ref
    for which, cols in which_geo:
        nidx = cols // DIM
        assert cols % 256 == 0, "kernel geometry needs cols % 256 == 0"
        mrow = nidx * 12 // 8
        prow = (nidx + 7) // 8
        pay = bytearray()
        pay += struct.pack('<IHHIIIIQ', MAT3_MAGIC, DIM, nc, rows, cols, flags, 12, cb_off)
        pay += gains_bytes(rows)
        # main streams for ALL rows, then the plane for ALL rows (the kernel's
        # m.ex = m.ix + rows*mrow, cuda_vq_row.inc.cu:41)
        main_all = bytearray()
        pl_all = bytearray(prow * rows) if plane else None
        for r in range(rows):
            main = bytearray(mrow)
            for k in range(nidx):
                v = idx_of(which, r, k, nc)
                idxs[(which, r, k)] = v
                write_bits(main, k * 12, v & 0xFFF, 12)
                if plane and ((v >> 12) & 1):
                    g, j = k >> 5, k & 31
                    pl_all[r * prow + g * 4 + (j >> 3)] |= 1 << (j & 7)
            main_all += main
        pay += main_all
        if plane:
            pay += pl_all
        pay += b'\0' * 8  # the payload's tail pad (vq_fmt.h:5)
        payloads.append((which, bytes(pay)))
    table = bytearray(nexp * 3 * 8)
    off = cb_off + len(cb)
    for which, p in payloads:
        struct.pack_into('<Q', table, which * 8, off)
        off += len(p)
    blob = struct.pack('<IIII', BLOB_MAGIC, 3, 0, nexp) + bytes(table) + cb + b''.join(p for _, p in payloads)
    return blob, idxs


def build_v2(nc, rows, cols):
    nbit = max(1, (nc - 1).bit_length())
    assert nbit in (11, 12)
    nexp = 1
    cb = b''.join(f16(cb_half(k)) for k in range(nc * DIM))
    nidx = cols // DIM
    nstream = (rows * nidx * nbit + 7) // 8 + 3  # + safety window (vq_fmt.h)
    stream = bytearray(nstream)
    for r in range(rows):
        for k in range(nidx):
            write_bits(stream, (r * nidx + k) * nbit, (r * 97 + k * 29) % nc, nbit)
    pay = struct.pack('<IHHII', MAT_MAGIC, DIM, nc, rows, cols) + cb + gains_bytes(rows) + bytes(stream)
    table = bytearray(nexp * 3 * 8)
    struct.pack_into('<Q', table, 0, 16 + nexp * 3 * 8)
    return struct.pack('<IIII', BLOB_MAGIC, 2, 0, nexp) + bytes(table) + pay


def probes_v3(which_geo, rows, plane, idxs):
    lines = []
    for which, cols in which_geo:
        for c in (7, 2055, 2303, 5119):
            if c < cols:
                lines.append((which, 0, rows, c))
        for r in (0, rows // 2, rows - 1):
            for c in (1003, 2040, 4103):
                if c < cols:
                    lines.append((which, r, 1, c))
    if plane:
        bits = set()
        blocks = set()
        for (w, row, n, c) in lines:
            for rr in range(row, row + n):
                bits.add(1 if idxs[(w, rr, c // DIM)] >= 4096 else 0)
                blocks.add((c // DIM) // 256)
        assert bits == {0, 1}, "13-bit probe set must cover both plane states"
        assert len(blocks) >= 2, "probe set must cross a block boundary"
    return lines


def ref_v3(lines, idxs, cb, gains):
    vals = []
    for (which, row, n, col) in lines:
        k, d = col // DIM, col % DIM
        for r in range(row, row + n):
            v = idxs[(which, r, k)]
            g = f16_to_f32(gains[r * 2:r * 2 + 2])
            vals.append(e4m3_to_f32(cb[v * DIM + d]) * g)
    return vals


def emit(stem, blob, lines, ref=None):
    with open(os.path.join(HERE, stem + '.blob'), 'wb') as f:
        f.write(blob)
    with open(os.path.join(HERE, stem + '.probes.txt'), 'w') as f:
        f.write('# e which row n col  (n consecutive rows at column col)\n')
        for (e, w, row, n, col) in lines:
            f.write('%d %d %d %d %d\n' % (e, w, row, n, col))
    if ref is not None:
        with open(os.path.join(HERE, stem + '.ref.f32'), 'wb') as f:
            f.write(struct.pack('<%df' % len(ref), *ref))
    nvals = sum(n for (_, _, _, n, _) in lines)
    print('%s: blob=%d bytes probes=%d values=%d' % (stem, len(blob), len(lines), nvals))


def main():
    rows13 = 128
    geo = [(0, 5120), (1, 5120), (2, 2304)]
    blob, idxs = build_v3(geo, 8192, True, rows13)
    cb = bytes(cb_byte(k) for k in range(8192 * DIM))
    lines = probes_v3(geo, rows13, True, idxs)
    emit('v3_13b', blob, [(0,) + l for l in lines], ref_v3(lines, idxs, cb, gains_bytes(rows13)))

    rows12 = 64
    geo12 = [(0, 5120)]
    blob, idxs = build_v3(geo12, 4096, False, rows12)
    cb = bytes(cb_byte(k) for k in range(4096 * DIM))
    lines = probes_v3(geo12, rows12, False, idxs)
    emit('v3_12b', blob, [(0,) + l for l in lines], ref_v3(lines, idxs, cb, gains_bytes(rows12)))

    rows2 = 64
    lines2 = [(0, 0, 0, rows2, 7), (0, 0, 0, rows2, 2055), (0, 0, 0, rows2, 2303)]
    for r in (0, rows2 // 2, rows2 - 1):
        for c in (1003, 2040):
            lines2.append((0, 0, r, 1, c))
    emit('v2_12b', build_v2(4096, rows2, 2304), lines2)

    lines2 = [(0, 0, 0, rows2, 7), (0, 0, 0, rows2, 2055), (0, 0, 0, rows2, 5119)]
    for r in (0, rows2 // 2, rows2 - 1):
        for c in (1003, 4103):
            lines2.append((0, 0, r, 1, c))
    emit('v2_11b', build_v2(2048, rows2, 5120), lines2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
