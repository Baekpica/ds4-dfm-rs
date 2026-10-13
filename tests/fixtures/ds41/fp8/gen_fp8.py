#!/usr/bin/env python3
# gen_fp8.py — P4-3 gate fixture: an fp8_32x32 image and its cases.
#
# Writes fp8.img + fp8.cases.txt for tests/test_ds41_fp8.cu. The reference
# values (fp8.ref.f32) come from the Rust emulation
# (crates/ds4-core/examples/ds41_fp8_ref.rs), never from this script.
#
# The image is a "model map": a 128-byte pad, then two fp8 tensors in the
# on-disk layout (ds4_quantfmt.h:36-38) — the rows*cols E4M3 plane, then
# ceil(rows/32)*ceil(cols/32) E8M0 scale bytes, tile (r/32, c/32):
#
#   plain   off=128       in=1024 out=4096   the engram-wkv entry's shape
#   grouped off=...       groups=2 gdim=256 rank=64   the towers' wo_a form
#
# Geometry rationale: the plain tensor's out_dim*1 = 4096 < 32768 lets the
# ksplit search climb, and nseg = ceil(1024/512) = 2 clamps it to ksplit=2 —
# the cross-warp reduction arm the real wkv (out 25600, in 6144, ksplit 2)
# takes. The grouped tensor at rank 64 stops at ksplit=1, the other arm.
#
# Cases: onehot (a single 1.0 per token, so every dot is a single-term sum and
# the entry must be BIT-EXACT) and dense (f32 accumulation order differs from
# the emulation's f64; compare within the recorded tolerance). Scale bytes
# cycle over 120..135 so a wrong tile index changes the decode loudly.
#
#   python3 gen_fp8.py     (writes next to this script)
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLAIN_IN, PLAIN_OUT = 1024, 4096
GROUP_GROUPS, GROUP_GDIM, GROUP_RANK = 2, 256, 64
# Page-aligned offsets: the resolver's register tier page-rounds ranges, so a
# 16-byte-aligned base with non-aligned neighbors makes the next tensor's
# registration overlap and its device-copy source straddle a boundary (the
# engine's documented kv_rms_weight shape, ds4_cuda.cu:1795-1803). The test
# allocates the image with the same 4096 alignment.
PLAIN_OFF = 4096
PLAIN_LEN = (PLAIN_OUT * PLAIN_IN + (PLAIN_OUT // 32) * (PLAIN_IN // 32) + 4095) // 4096 * 4096


def sc_byte(r, c):
    """Scale byte for tile (r/32, c/32): a spread of exponents 2^-7..2^8."""
    return 120 + ((r // 32) * 7 + (c // 32) * 3) % 16


def e4_byte(r, c):
    """Deterministic E4M3 byte (finite: abs != 0x7f)."""
    v = (r * 131 + c * 37 + (r >> 3) * 11 + 5) % 126
    if v == 0:
        v = 1
    return v | (((r + c) & 1) << 7)


def build_tensor(rows, cols):
    w = bytearray(rows * cols)
    for r in range(rows):
        for c in range(cols):
            w[r * cols + c] = e4_byte(r, c)
    sc = bytearray((rows // 32) * (cols // 32))
    for tr in range(rows // 32):
        for tc in range(cols // 32):
            sc[tr * (cols // 32) + tc] = sc_byte(tr * 32, tc * 32)
    return bytes(w) + bytes(sc)


def lcg(seed, n, lo=-1.0, hi=1.0):
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = ((s >> 33) & 0x7FFFFF) / float(0x800000)
        out.append(lo + (hi - lo) * u)
    return out


def hexf(v):
    return '%08x' % struct.unpack('<I', struct.pack('<f', v))[0]


def main():
    plain = build_tensor(PLAIN_OUT, PLAIN_IN)
    grouped = build_tensor(GROUP_GROUPS * GROUP_RANK, GROUP_GDIM)
    group_off = PLAIN_OFF + PLAIN_LEN   # PLAIN_LEN is page-multiple, so this is aligned too
    img = b'\0' * PLAIN_OFF + plain + b'\0' * (PLAIN_LEN - len(plain)) + grouped

    # Cases. x is n_tok * in_dim floats; for grouped, in_dim = groups*gdim and
    # token t's onehots sit in BOTH group segments (one per group), so one case
    # covers the grid.y row offset.
    cases = []

    def onehot_plain(n, col):
        x = [0.0] * (n * PLAIN_IN)
        for t in range(n):
            x[t * PLAIN_IN + (col + t * 7) % PLAIN_IN] = 1.0
        cases.append(('onehot', 'plain', n, 0, x, col))

    def dense_plain(n, seed):
        cases.append(('dense', 'plain', n, 0, lcg(seed, n * PLAIN_IN), 0))

    def onehot_grouped(n, col):
        x = [0.0] * (n * GROUP_GROUPS * GROUP_GDIM)
        for t in range(n):
            base = t * GROUP_GROUPS * GROUP_GDIM
            x[base + (col + t) % GROUP_GDIM] = 1.0
            x[base + GROUP_GDIM + (col + 167 + t) % GROUP_GDIM] = 1.0
        cases.append(('onehot', 'grouped', n, 0, x, col))

    onehot_plain(1, 7)
    onehot_plain(1, PLAIN_IN - 1)      # last column tile
    dense_plain(1, 12345)
    dense_plain(2, 777)                # XB=1 (activations through bf16)
    dense_plain(12, 999)               # >8: the bf16-w + cuBLAS arm
    cases.append(('onehot', 'round', 2, 1, [0.0] * (2 * PLAIN_IN), 511))
    for t in range(2):
        cases[-1][4][t * PLAIN_IN + (511 + t * 7) % PLAIN_IN] = 1.0
    cases.append(('dense', 'round', 3, 1, lcg(4242, 3 * PLAIN_IN), 0))
    cases.append(('dense', 'grouped', 1, 0, lcg(31337, GROUP_GROUPS * GROUP_GDIM), 0))
    onehot_grouped(2, 33)

    with open(os.path.join(HERE, 'fp8.img'), 'wb') as f:
        f.write(img)
    with open(os.path.join(HERE, 'fp8.cases.txt'), 'w') as f:
        f.write('# fp8_32x32 fixture: image + case list\n')
        f.write('tensor plain off=%d in=%d out=%d\n' % (PLAIN_OFF, PLAIN_IN, PLAIN_OUT))
        f.write('tensor grouped off=%d groups=%d gdim=%d rank=%d\n' % (group_off, GROUP_GROUPS, GROUP_GDIM, GROUP_RANK))
        f.write('# case <n> <onehot|dense> entry=<plain|round|grouped> n=<tokens> round=<0|1> col=<onehot col>\n')
        for n, (kind, entry, nt, round_out, x, col) in enumerate(cases):
            f.write('case %d %s entry=%s n=%d round=%d col=%d\n' % (n, kind, entry, nt, round_out, col))
            f.write('x %s\n' % ' '.join(hexf(v) for v in x))
    print('fp8: img=%d bytes cases=%d (onehot=%d dense=%d) plain[%d][%d] grouped[%d][%d]x%d'
          % (len(img), len(cases),
             sum(1 for c in cases if c[0] == 'onehot'),
             sum(1 for c in cases if c[0] == 'dense'),
             PLAIN_OUT, PLAIN_IN, GROUP_GROUPS * GROUP_RANK, GROUP_GDIM, GROUP_GROUPS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
