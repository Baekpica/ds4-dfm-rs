#!/usr/bin/env python3
# gen_engram.py — P4-3 gate fixture: the engram gate kernel and the row dequant.
#
# Writes engram.img + engram.cases.txt + rows.bin for tests/test_ds41_engram.cu.
# The references (engram.ref.f32, rows.ref.f32) come from the Rust emulation
# (crates/ds4-core/examples/ds41_engram_ref.rs), never from this script.
#
# Geometry is the artifact's own (E=5120, HC=4, head_dim=256, eps=1e-20) at
# n_tok=2, so the kernel's thread-stride pattern (5120/256 = 20 steps) is the
# real one. The image is a "model map": 64 B pad, then engram_q [HC*E] f32 and
# engram_k [HC*E] f32 (the gate entry addresses them by offset).
#
# Case kinds:
#   dense  — random h/key/val: the f32 reduction order and the fast-math expf
#            differ from the emulation, and the output is bf16-quantized, so
#            the compare is at bf16-ulp distance.
#   absorb — h already on the bf16 grid and a tiny val: gate*val lands under
#            half a bf16 ulp, so h must come back BIT-EXACT.
#   zero   — h = 0: the output is bf16r(gate*val), the pure gate path.
#
#   python3 gen_engram.py     (writes next to this script)
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E, HC, NTOK = 5120, 4, 2
EPS = 1e-20
Q_OFF = 4096   # page-aligned (see gen_fp8.py: the resolver page-rounds ranges)
HEAD_DIM, N_ROWS = 256, 5


def lcg(seed, n, lo=-1.0, hi=1.0):
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = ((s >> 33) & 0x7FFFFF) / float(0x800000)
        out.append(lo + (hi - lo) * u)
    return out


def bf16r(v):
    u = struct.unpack('<I', struct.pack('<f', v))[0]
    if (u & 0x7F800000) == 0x7F800000:
        return v
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', u))[0]


def hexf(v):
    return '%08x' % struct.unpack('<I', struct.pack('<f', v))[0]


def main():
    qw = lcg(101, HC * E)
    kw = lcg(202, HC * E)
    img = b'\0' * Q_OFF + b''.join(struct.pack('<f', v) for v in qw + kw)

    cases = []
    # dense
    cases.append(('dense', lcg(303, NTOK * HC * E), lcg(404, NTOK * (HC + 1) * E)))
    # absorb: h on the bf16 grid, key O(1), val ~1e-7 (under half a bf16 ulp)
    h = [bf16r(v) for v in lcg(505, NTOK * HC * E)]
    kv = lcg(606, NTOK * (HC + 1) * E)
    for t in range(NTOK):
        base = t * (HC + 1) * E + HC * E   # token t's val slice (row HC)
        for d in range(E):
            kv[base + d] *= 1e-7
    cases.append(('absorb', h, kv))
    # zero h
    cases.append(('zero', [0.0] * (NTOK * HC * E), lcg(707, NTOK * (HC + 1) * E)))

    # Row fixture: e4m3 row bytes + ue8m0 tail, the engine's on-disk row form.
    rows = bytearray()
    for r in range(N_ROWS):
        for d in range(HEAD_DIM):
            v = (r * 131 + d * 37 + 5) % 126
            rows.append(v if v else 1)
        for d in range(HEAD_DIM // 32):
            rows.append(120 + (r * 3 + d) % 16)

    with open(os.path.join(HERE, 'engram.img'), 'wb') as f:
        f.write(img)
    with open(os.path.join(HERE, 'engram.cases.txt'), 'w') as f:
        f.write('# engram fixture: model image + gate cases\n')
        f.write('gate q_off=%d k_off=%d e=%d hc=%d eps=%s\n' % (Q_OFF, Q_OFF + HC * E * 4, E, HC, repr(EPS)))
        f.write('# case <n> <dense|absorb|zero> n=%d\n' % NTOK)
        for n, (kind, h, kv) in enumerate(cases):
            f.write('case %d %s n=%d\n' % (n, kind, NTOK))
            f.write('h %s\n' % ' '.join(hexf(v) for v in h))
            f.write('kv %s\n' % ' '.join(hexf(v) for v in kv))
    with open(os.path.join(HERE, 'rows.bin'), 'wb') as f:
        f.write(bytes(rows))
    print('engram: img=%d cases=%d rows=%d (E=%d HC=%d n=%d eps=%g)'
          % (len(img), len(cases), N_ROWS, E, HC, NTOK, EPS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
