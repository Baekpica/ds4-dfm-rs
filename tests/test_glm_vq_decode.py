#!/usr/bin/env python3
"""Independent per-projection bounds on a saved actual eight-expert fixture."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np


def bf16(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def restore(fixture, slot, bits):
    prefix = f'm{slot}_'
    rows, cols = fixture[prefix + 'shape']
    p = fixture[prefix + 'main'].reshape(-1, 3).astype(np.uint16)
    indices = np.empty(p.shape[0] * 2, dtype=np.uint16)
    indices[0::2] = p[:, 0] | (p[:, 1] & 15) << 8
    indices[1::2] = p[:, 1] >> 4 | p[:, 2] << 4
    if bits == 13:
        indices |= np.unpackbits(fixture[prefix + 'plane'], bitorder='little').astype(np.uint16) << 12
    code = fixture['book'].reshape(-1, 8)[indices]
    exponent = (code >> 3) & 15
    mantissa = code & 7
    word = np.where(exponent == 0, mantissa * 2.0**-9,
                    np.ldexp(1.0 + mantissa / 8.0, exponent.astype(np.int32) - 7))
    word = np.where(code & 128, -word, word).astype(np.float32)
    gain = fixture[prefix + 'gain'].astype(np.float32)
    return bf16(word.reshape(rows, cols) * gain[:, None])


def dot_bound(w, x, got):
    w, x = w.astype(np.float64), x.astype(np.float64)
    expected = w @ x
    absolute = np.abs(w) @ np.abs(x)
    ops, u = w.shape[1] + 32, 2.0**-24
    bound = ops * u / (1 - ops * u) * absolute
    bound += np.exp2(np.floor(np.log2(np.maximum(np.abs(expected), 2.0**-126))) - 8) + 2.0**-126
    error = np.abs(got.astype(np.float64) - expected)
    ratio = float((error / bound).max())
    assert np.isfinite(got).all() and (error <= bound).all(), ratio
    return ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--lib', type=Path, default=Path('tests/libglm_vq_decode.so'))
    parser.add_argument('--bits', type=int, choices=[12, 13], default=13)
    parser.add_argument('--warps', type=int, choices=[16, 32], default=32)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--input-row', type=int, default=0)
    args = parser.parse_args()
    f = np.load(args.fixture)
    arrays = [[np.ascontiguousarray(f[f'm{i}_{name}']) for i in range(24)] for name in ['main', 'plane', 'gain']]
    pointers = [(ct.c_void_p * 24)(*[a.ctypes.data for a in group]) for group in arrays]
    x = np.ascontiguousarray(f['input'][args.input_row], dtype=np.float32)
    weights = np.array([.11, .24, .36, .21, .32, .07, .29, .40], dtype=np.float32)
    gate = np.empty((8, 2048), dtype=np.float32)
    up = np.empty_like(gate)
    mid = np.empty((8, 2048), dtype=np.uint16)
    partial = np.empty((8, 4096), dtype=np.float32)
    output, baseline, ms = [np.empty(n, dtype=np.float32) for n in [4096, 4096, 2]]
    book = np.ascontiguousarray(f['book'])
    lib = ct.CDLL(str(args.lib.resolve()))
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(name in line for name in ['libcudart', 'libcublas'])})
    assert libraries and all('/usr/local/cuda-13.4/' in path for path in libraries), libraries
    fn = lib.test_glmvq_decode
    fn.argtypes = [ct.c_void_p] * 6 + [ct.c_uint32] * 3 + [ct.c_void_p] * 7
    fn.restype = ct.c_int
    rc = fn(*pointers, book.ctypes.data, x.ctypes.data, weights.ctypes.data,
            args.bits, args.warps, args.repeats, *[a.ctypes.data for a in [ms, gate, up, mid, partial, output, baseline]])
    assert rc == 0, rc
    inputs = bf16(x)
    mid_f32 = (mid.astype(np.uint32) << 16).view(np.float32)
    bounds = []
    for i in range(24):
        w = restore(f, i, args.bits)
        e, projection = divmod(i, 3)
        got = [gate[e], up[e], partial[e]][projection]
        bounds.append(dot_bound(w, mid_f32[e] if projection == 2 else inputs, got))
    expected = np.zeros(4096, dtype=np.float32)
    for e in range(8):
        expected = bf16(expected + bf16(partial[e] * weights[7 - e]))
    assert np.array_equal(output.view(np.uint32), expected.view(np.uint32))
    print(json.dumps(dict(bits=args.bits, warps=args.warps, input_row=args.input_row,
        matrices=24, max_bound_ratio=max(bounds), baseline_ms=float(ms[0]), direct_ms=float(ms[1]),
        full_output_relative_rms=float(np.linalg.norm(output - baseline) / np.linalg.norm(baseline)),
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.lib.read_bytes()).hexdigest(), libraries=libraries,
        scope='Saved layer3 experts and layer0 normalized input;12bit masks the actual plane for a layout case. Rotation exceeds L2; book/activation hot. Projection bounds, fused SwiGLU and ordered reduction pass; no whole-workload or broad quality claim.'
    )), flush=True)


if __name__ == '__main__':
    main()
