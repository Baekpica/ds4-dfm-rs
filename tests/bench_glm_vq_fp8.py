#!/usr/bin/env python3
"""Replay actual FP8 weights/scales with system CUDA and explicit inputs."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE_BLOCK = 128
spec = importlib.util.spec_from_file_location('vqformat', ROOT / 'tools/glm_vq_q4k.py')
fmt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmt)


def payload(model, record):
    with (model / record['part']).open('rb') as file:
        meta, entries, base = fmt.read_header(file)
        entry = next(e for e in entries if e['name'] == record['name'])
        assert list(reversed(entry['dims'])) == record['stored_shape']
        file.seek(base + entry['offset'])
        raw = file.read(record['payload_bytes'])
    assert hashlib.sha256(raw).hexdigest() == record['payload_sha256']
    return raw, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--matrix', required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--input-row', type=int, default=0)
    p.add_argument('--tokens', type=int, default=1)
    p.add_argument('--repeats', type=int, default=64)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--mode', choices=['base', 'global', 'shared'], default='base')
    p.add_argument('--split', type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument('--check', action='store_true')
    p.add_argument('--lib', type=Path, default=ROOT / 'tests/libglm_vq_fp8.so')
    args = p.parse_args()
    manifest = json.loads((args.model / 'tensor-manifest.json').read_text())
    records = {r['name']: r for r in manifest['tensors']}
    r = records[args.matrix]
    assert r['type_id'] == fmt.FP8
    rows, cols = r['shape']
    raw, meta = payload(args.model, r)
    link = json.loads(meta['glm_vq.fp8_scales'][1])[args.matrix]
    sr = records[link]
    assert sr['shape'] == [rows // SCALE_BLOCK, cols // SCALE_BLOCK] and sr['type_id'] == 0
    scaledata, _ = payload(args.model, sr)
    w = np.frombuffer(raw, np.uint8)
    scales = np.frombuffer(scaledata, '<f4').reshape(rows // SCALE_BLOCK, cols // SCALE_BLOCK)
    if args.input:
        x = np.fromfile(args.input, '<f4').reshape(-1, cols)
        x = np.ascontiguousarray(x[args.input_row:args.input_row + args.tokens])
        assert len(x) == args.tokens
    else:
        x = fmt.bf16_round(np.random.default_rng(5353).normal(size=(args.tokens, cols)).astype(np.float32))
    y = np.empty((args.tokens, rows), np.float32)
    lib = ct.CDLL(str(args.lib.resolve()))
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(s in line for s in ['libcudart', 'libcublas'])})
    assert libraries and all('/usr/local/cuda-13.4/' in s for s in libraries), libraries
    fn = lib.bench_glmvq_fp8
    fn.argtypes = [ct.c_void_p] * 4 + [ct.c_uint32] * 7 + [ct.POINTER(ct.c_float)]
    fn.restype = ct.c_int
    ms = ct.c_float()
    rc = fn(w.ctypes.data, scales.ctypes.data, x.ctypes.data, y.ctypes.data,
            rows, cols, args.tokens, args.repeats, int(args.profile),
            ['base', 'global', 'shared'].index(args.mode), args.split, ct.byref(ms))
    assert rc == 0 and np.isfinite(y).all(), rc
    result = dict(matrix=args.matrix, shape=[rows, cols], tokens=args.tokens,
        milliseconds=ms.value, repeats=args.repeats, profile=args.profile,
        mode=args.mode, split=args.split, payload_sha256=r['payload_sha256'],
        scale_sha256=sr['payload_sha256'], input_sha256=hashlib.sha256(x).hexdigest(),
        library_sha256=hashlib.sha256(args.lib.read_bytes()).hexdigest(), libraries=libraries,
        input=str(args.input) if args.input else 'synthetic BF16 normal; dense access value-independent',
        input_row=args.input_row, cache='rotate at least64MiB weights; hot input/scales;16 warmups',
        scope='Original FP8 block-scale/BF16 boundaries; direct dot changes FP32 reduction order. Baseline production cuBLAS with32MiB workspace; no whole-model claim')
    if args.check:
        words = fmt.fp8_table()[w].reshape(rows, cols)
        values = fmt.bf16_round(words * scales.repeat(SCALE_BLOCK, axis=0).repeat(SCALE_BLOCK, axis=1))
        actual_x = fmt.bf16_round(x).astype(np.float64)
        reference = actual_x @ values.astype(np.float64).T
        absolute = np.abs(actual_x) @ np.abs(values.astype(np.float64).T)
        u, ops = 2.0**-24, cols + 32
        bound = ops * u / (1 - ops * u) * absolute
        bound += np.exp2(np.floor(np.log2(np.maximum(np.abs(reference), 2.0**-126))) - 8) + 2.0**-126
        error = np.abs(y.astype(np.float64) - reference)
        result['max_bound_ratio'] = float((error / bound).max())
        assert (error <= bound).all(), result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
