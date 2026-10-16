#!/usr/bin/env python3
"""Independent BF16 indexer arithmetic and exact chunk/carry structure."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH, RATIO, HEADS = 128, 4, 32


def bits(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def floats(x):
    return (np.asarray(x, np.uint16).astype(np.uint32) << 16).view(np.float32)


def rounded(x):
    return floats(bits(x)).astype(np.float64)


def ulp(x):
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(x), 2.0**-126))) - 7)


def parameters(model):
    spec = importlib.util.spec_from_file_location('q4tool', ROOT / 'tools/glm_vq_q4k.py')
    q4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q4)
    tensors = json.loads((model / 'tensor-manifest.json').read_text())['tensors']
    result = []
    for suffix in ['k_norm.weight', 'k_norm.bias', 'index_kpool_compress_ape']:
        name = 'model.language_model.layers.3.self_attn.indexer.' + suffix
        t = next(t for t in tensors if t['name'] == name)
        assert t['type_id'] == 30
        with (model / t['part']).open('rb') as f:
            _, entries, base = q4.read_header(f)
            entry = next(e for e in entries if e['name'] == name)
            f.seek(base + entry['offset'])
            payload = f.read(t['payload_bytes'])
        assert hashlib.sha256(payload).hexdigest() == t['payload_sha256']
        result.append(np.frombuffer(payload, np.uint16).copy())
    return result


def pool_reference(raw, gate, params):
    norm, bias, ape = [floats(x).astype(np.float64) for x in params]
    count = len(raw) // RATIO * RATIO
    x = raw[:count].astype(np.float64)
    centered = x - x.mean(-1, keepdims=True)
    normal = rounded(centered / np.sqrt((centered**2).mean(-1, keepdims=True) + 1e-6) * norm + bias)
    normal = normal.reshape(-1, RATIO, WIDTH)
    logits = gate[:count].reshape(-1, RATIO, WIDTH).astype(np.float64) + ape.reshape(RATIO, WIDTH)
    exp = np.exp(logits - logits.max(1, keepdims=True))
    prob = rounded(exp / exp.sum(1, keepdims=True))
    product = rounded(prob * normal)
    expected = rounded(product.sum(1))
    # One BF16 step at each nonlinear boundary allows an FP32 reduction to
    # cross a rounding midpoint. Propagate it with actual per-channel scale;
    # packing, completed-pool count and carried positions remain exact.
    bound = (np.abs(prob) * ulp(normal) + ulp(prob) * np.abs(normal)
             + ulp(product)).sum(1) + ulp(expected)
    return expected, bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    lib = ct.CDLL(str(args.lib.resolve()))
    pool = lib.test_gvq_pool
    pool.argtypes = [ct.c_void_p] * 8 + [ct.c_uint32] * 2
    pool.restype = ct.c_int
    score = lib.test_gvq_scores
    score.argtypes = [ct.c_void_p] * 4 + [ct.c_uint32] * 3
    score.restype = ct.c_int
    params = parameters(args.model)
    rng = np.random.default_rng(53531)
    reports = []
    for rows, chunks in [(3, [1, 2]), (8, [3, 5]),
                         (2053, [1, 2, 1024, 1024, 2]),
                         (8195, [1, 2, 5, 1023, 1024, 1025, 2048, 2048, 1019])]:
        assert sum(chunks) == rows
        raw = floats(bits(rng.normal(size=(rows, WIDTH))))
        gate = floats(bits(rng.normal(size=(rows, WIDTH)) * 2))
        raw[0] = 1  # Zero variance exercises the source epsilon.
        whole, split = [np.zeros(((rows + RATIO - 1) // RATIO, WIDTH), np.uint16) for _ in range(2)]
        states = [np.empty((2, RATIO, WIDTH), np.float32) for _ in range(2)]
        for out, state, parts in [(whole, states[0], [rows]), (split, states[1], chunks)]:
            parts = np.asarray(parts, np.uint32)
            assert pool(raw.ctypes.data, gate.ctypes.data, *[p.ctypes.data for p in params],
                        out.ctypes.data, state.ctypes.data, parts.ctypes.data, len(parts), rows) == 0
        assert np.array_equal(whole, split)
        tail = rows % RATIO
        if tail:
            for state in states:
                assert np.array_equal(state[0, :tail], raw[-tail:])
                assert np.array_equal(state[1, :tail], gate[-tail:])
            assert not np.any(whole[-1])  # Future/incomplete pool is untouched.
        expected, bound = pool_reference(raw, gate, params)
        actual = floats(whole[:rows // RATIO]).astype(np.float64)
        error = np.abs(actual - expected)
        ratio = float(np.max(error / bound)) if error.size else 0
        assert np.isfinite(actual).all() and ratio <= 1, (rows, ratio)
        reports.append(dict(rows=rows,chunk_state='exact',pool_bound_ratio=ratio,
                            pool_equal_fraction=float(np.mean(actual == expected)) if error.size else 1))
        if rows < 8:
            continue
        query_rows = min(7, rows)
        first, pools = rows - query_rows, rows // RATIO
        q = floats(bits(rng.normal(size=(query_rows, HEADS, WIDTH))))
        weights = floats(bits(rng.normal(size=(query_rows, HEADS))))
        out = np.empty((query_rows, pools), np.float32)
        assert score(q.ctypes.data, weights.ctypes.data, whole.ctypes.data,
                     out.ctypes.data, query_rows, pools, first) == 0
        # FP64 dot reference on the actual BF16 cache isolates score errors
        # from the separately assessed normalization/pooling error.
        dots = q.astype(np.float64) @ actual.T
        qs = float(np.float32(WIDTH**-.5))
        ws = (weights.astype(np.float64) * float(np.float32(HEADS**-.5))).astype(np.float32).astype(np.float64)
        expected = (np.maximum(dots * qs, 0) * ws[..., None]).sum(1)
        u = 2.0**-24
        gamma = 192 * u / (1 - 192 * u)
        magnitude = (np.abs(q.astype(np.float64)) @ np.abs(actual.T)) * qs
        bound = gamma * (magnitude * np.abs(ws[..., None])).sum(1) + 2.0**-126
        visible = np.arange(pools)[None] < (first + np.arange(query_rows)[:, None] + 1) // RATIO
        assert np.all(out[~visible] == -np.finfo(np.float32).max)
        ratio = float(np.max(np.abs(out[visible] - expected[visible]) / bound[visible]))
        assert np.isfinite(out).all() and ratio <= 1, (rows, ratio)
        reports[-1]['score_bound_ratio'] = ratio
        reports[-1]['causal_mask'] = 'exact'
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(s in line for s in ['libcuda', 'libcublas', 'libtorch'])})
    assert not any('torch' in x for x in libraries)
    print(json.dumps(dict(cases=reports,libraries=libraries)), flush=True)
    print('GLMVQ_INDEX_PASS', flush=True)


if __name__ == '__main__':
    main()
