#!/usr/bin/env python3
# gen_moe.py — P4-2 gate fixture: a MoE-geometry DQVL v3 blob and its cases.
#
# Writes moe.blob + moe.cases.txt for tests/test_ds41_moe.cu. The reference
# values (moe.ref.f32) come from the Rust emulation
# (crates/ds4-core/examples/ds41_moe_ref.rs), never from this script.
#
# Blob: DQVL v3, nc=8192, 13-bit with the plane, nexp=4 experts, each with
# which 0/1 [MID=256][IN=512] (gate/up) and which 2 [OUT=512][MID=256].
#
#   expert 0 is the PROBE expert: its down matrix decodes to a single nonzero
#   column m0=8 per row — one shared codeword word carries 1.0 at byte m0%8,
#   every other down word is all-zero E4M3 — so a one-hot activation makes
#   every dot in the chain a single-term sum and the whole MoE is bit-exact
#   against the emulation. (The one remaining inexactness is the device expf
#   inside swiglu, whose f32 result is rounded to bf16 right after.)
#   expert 1 is dense: deterministic indices covering both plane states, and it
#   shares expert 0's gate/up payload slots (only the downs differ).
#
# Geometry contract: the n=1 persist kernels (the dispatch the real forward
# takes) pipeline 8-round blocks and need R = cols/256 >= 8 rounds per row —
# every real V4.1 layer has R = 20 (gate/up) / 9 (down).  IN and MID are 2048
# here to stay at that boundary; a smaller fixture decodes garbage in rows 2+
# of a span and would test a shape the engine never runs.
#
# The layout is written from the spec (cuda_vq_row.inc.cu:19-49: gains at
# pay+32, main stream after them for ALL rows, plane after that, 8 B pad),
# exactly as gen_v3_13b.py — same writer, different geometry.
#
#   python3 gen_moe.py     (writes next to this script)
import os
import struct
import sys

DIM = 8
BLOB_MAGIC = 0x4C565144  # 'DQVL'
MAT3_MAGIC = 0x33565144  # 'DQV3'
GAIN_CYCLE = [0.25, 0.5, 1.0, 1.5, 2.0]
IN, MID, OUT = 2048, 2048, 512
NC = 8192
NEXP = 2
M0 = 8                      # the probe expert's single nonzero down column
W_STAR_IDX, W_ZERO_IDX = 7000, 7001
CLAMP = 10.0
HERE = os.path.dirname(os.path.abspath(__file__))


def cb_byte(k):
    v = (k * 29) % 127
    if v == 0:
        v = 1
    return v | ((k & 1) << 7)


def idx_of(which, e, row, k):
    return (e * 5011 + row * 131 + k * 37 + which * 911) % (NC - 2)


def write_bits(buf, bitpos, value, nbits):
    for b in range(nbits):
        if (value >> b) & 1:
            buf[(bitpos + b) >> 3] |= 1 << ((bitpos + b) & 7)


def gains_bytes(rows):
    return b''.join(struct.pack('<e', GAIN_CYCLE[r % len(GAIN_CYCLE)]) for r in range(rows))


def build_payload(which, rows, cols, idx_fn, plane):
    nidx = cols // DIM
    mrow = nidx * 12 // 8
    prow = (nidx + 7) // 8
    pay = bytearray()
    pay += struct.pack('<IHHIIIIQ', MAT3_MAGIC, DIM, NC, rows, cols, 1 | (2 if plane else 0), 12, cb_off())
    pay += gains_bytes(rows)
    main_all = bytearray()
    pl_all = bytearray(prow * rows)
    for r in range(rows):
        main = bytearray(mrow)
        for k in range(nidx):
            v = idx_fn(r, k)
            write_bits(main, k * 12, v & 0xFFF, 12)
            if plane and ((v >> 12) & 1):
                g, j = k >> 5, k & 31
                pl_all[r * prow + g * 4 + (j >> 3)] |= 1 << (j & 7)
        main_all += main
    pay += main_all
    pay += pl_all
    pay += b'\0' * 8
    return bytes(pay)


def cb_off():
    return 16 + NEXP * 3 * 8


def main():
    cb = bytearray(cb_byte(k) for k in range(NC * DIM))
    # The probe words: 1.0 at byte M0%8, zeros elsewhere; and an all-zero word.
    for d in range(DIM):
        cb[W_STAR_IDX * DIM + d] = 0x38 if d == M0 % DIM else 0x00
        cb[W_ZERO_IDX * DIM + d] = 0x00

    payloads = []   # (e, which, bytes)
    plane_states = set()
    gate = build_payload(0, MID, IN, lambda r, k: idx_of(0, 1, r, k), True)
    up = build_payload(1, MID, IN, lambda r, k: idx_of(1, 1, r, k), True)
    # expert 0: every down row one nonzero column (word M0/8), rest zero words
    down0 = build_payload(2, OUT, MID,
                          lambda r, k: W_STAR_IDX if k == M0 // DIM else W_ZERO_IDX, True)
    down1 = build_payload(2, OUT, MID, lambda r, k: idx_of(2, 1, r, k), True)
    for which, cols, rows in ((0, IN, MID), (1, IN, MID), (2, MID, OUT)):
        for r in range(0, rows, 17):
            for k in range(0, cols // DIM, 7):
                plane_states.add(1 if idx_of(which, 1, r, k) >= 4096 else 0)
    assert plane_states == {0, 1}, "the dense expert must cover both plane states"
    # Both experts share the gate/up payload slots; only the downs differ.
    payloads += [(0, 0, gate), (0, 1, up), (0, 2, down0)]
    payloads += [(1, 0, gate), (1, 1, up), (1, 2, down1)]

    table = bytearray(NEXP * 3 * 8)
    off = cb_off() + len(cb)
    for (e, which, p) in payloads:
        struct.pack_into('<Q', table, (e * 3 + which) * 8, off)
        off += len(p)
    blob = struct.pack('<IIII', BLOB_MAGIC, 3, 0, NEXP) + bytes(table) + bytes(cb) + b''.join(p for _, _, p in payloads)

    # Cases. onehot: x has a single 1.0 (bf16-exact) and K=1 so the chain is
    # single-term everywhere; random: dense x, K=6 across experts.
    def lcg(seed, n, lo=-1.0, hi=1.0):
        s = seed
        out = []
        for _ in range(n):
            s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            u = ((s >> 33) & 0x7FFFFF) / float(0x800000)
            out.append(lo + (hi - lo) * u)
        return out

    cases = []
    for c in (7, 8, 255, 256, 2047, 1024):
        x = [0.0] * IN
        x[c] = 1.0
        cases.append(('onehot', [0], [1.0], x))
    cases.append(('random', [1, 0, 1, 0, 1, 0], [0.7, 1.3, 0.2, 2.0, 0.9, 1.1], lcg(12345, IN)))
    cases.append(('random', [0, 1, 0, 1, 0, 1], [1.0, 0.5, 0.25, 1.5, 0.75, 2.0], lcg(999, IN)))

    with open(os.path.join(HERE, 'moe.blob'), 'wb') as f:
        f.write(blob)
    with open(os.path.join(HERE, 'moe.cases.txt'), 'w') as f:
        f.write('# case <n> <onehot|random> K=<K>; sel <K ints>; w <K hex f32>; x <IN hex f32>\n')
        for n, (kind, sel, w, x) in enumerate(cases):
            f.write('case %d %s K=%d\n' % (n, kind, len(sel)))
            f.write('sel %s\n' % ' '.join(str(v) for v in sel))
            f.write('w %s\n' % ' '.join('%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in w))
            f.write('x %s\n' % ' '.join('%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in x))
    print('moe: blob=%d bytes cases=%d (onehot=%d random=%d) IN=%d MID=%d OUT=%d'
          % (len(blob), len(cases),
             sum(1 for c in cases if c[0] == 'onehot'),
             sum(1 for c in cases if c[0] == 'random'), IN, MID, OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
