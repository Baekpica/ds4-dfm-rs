#!/usr/bin/env python3
"""Replay one actual nonexpert Q4_K matrix without loading the full model."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q4gate', ROOT / 'tests/test_glm_vq_q4k.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--matrix', default='model.language_model.layers.0.self_attn.q_proj.weight')
    parser.add_argument('--tokens', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=128)
    parser.add_argument('--mode', choices=['scalar', 'stage', 'pipe', 'auto'], default='auto',
                        help='Diagnostic request; pipe falls back when two buffers exceed device limits')
    parser.add_argument('--rows', type=int, help='Crop rows to exercise a partial CTA')
    parser.add_argument('--lib', type=Path, default=ROOT / 'tests/libglm_vq_linear.so')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.model / 'tensor-manifest.json').read_text())
    tensor = next(t for t in manifest['tensors'] if t['name'] == args.matrix)
    assert tensor['type_id'] == 12
    rows, cols = tensor['shape']
    with (args.model / tensor['part']).open('rb') as file:
        _, entries, base = gate.q4.read_header(file)
        entry = next(e for e in entries if e['name'] == args.matrix)
        file.seek(base + entry['offset'])
        packed = np.frombuffer(file.read(tensor['payload_bytes']), dtype=np.uint8).copy()
    assert hashlib.sha256(packed).hexdigest() == tensor['payload_sha256']
    if args.rows:
        assert 0 < args.rows <= rows
        rows = args.rows
        packed = packed[:rows * cols // 256 * 144]
    if args.input:
        x = np.fromfile(args.input, dtype=np.float32).reshape(-1, cols)
        assert len(x) >= args.tokens
        x = np.ascontiguousarray(x[:args.tokens])
    else:
        rng = np.random.default_rng(5353)
        x = gate.q4.bf16_round(rng.normal(size=(args.tokens, cols)).astype(np.float32))
    y = np.empty((args.tokens, rows), dtype=np.float32)
    elapsed = ct.c_float()
    fn = ct.CDLL(str(args.lib.resolve())).bench_glmvq_q4
    fn.argtypes = [ct.c_void_p] * 3 + [ct.c_uint32] * 5 + [ct.POINTER(ct.c_float)]
    fn.restype = ct.c_int
    assert fn(packed.ctypes.data, x.ctypes.data, y.ctypes.data, rows, cols,
              args.tokens, args.repeats, ['scalar', 'stage', 'pipe', 'auto'].index(args.mode), ct.byref(elapsed)) == 0
    assert np.isfinite(y).all()
    result = dict(matrix=args.matrix, shape=[rows, cols], tokens=args.tokens,
        repeats=args.repeats, mode=args.mode, milliseconds=elapsed.value,
        payload_gbs=packed.nbytes / elapsed.value / 1e6,
        matrix_sha256=tensor['payload_sha256'],
        library_sha256=hashlib.sha256(args.lib.read_bytes()).hexdigest(),
        input='saved activation' if args.input else 'synthetic BF16 normal; dense access is value-independent',
        cache='rotate at least64MiB of matrix copies; one launch for NCU has its own replay/cache policy')
    if args.check:
        w = gate.unpack(packed.tobytes(), rows, cols)
        expected = x.astype(np.float64) @ w.T
        absolute = np.abs(x.astype(np.float64)) @ np.abs(w.T)
        u = 2.0**-24
        ops = cols + 32
        gamma = ops * u / (1 - ops * u)
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(expected), 2.0**-126))) - 8)
        bound = gamma * absolute + ulp + 2.0**-126
        error = np.abs(y.astype(np.float64) - expected)
        result['max_bound_ratio'] = float((error / bound).max())
        assert (error <= bound).all(), result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
