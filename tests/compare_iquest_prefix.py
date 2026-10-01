#!/usr/bin/env python3
"""Compare a fresh native prefix dump with pinned independent reference vectors.

Requires NumPy and explicit archived inputs, but no model, Torch or quantizer.
Float/KV deltas are measurements: only embedding, routing and structural gates
decide the exit code. These three tokens do not qualify whole-model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_REPORT_SHA256 = '53fa9033bde7b7a682f4cbe6f548e7526913e32d00959f92a1e2c2c5a1d19eed'
REFERENCE_KV_SHA256 = (
    '7a7fa18890d1da4d9e9c794f427605118c2608ed1400ab047bc92fc3695e5dba',
    '270b3c2f4348fcfe5f656cecfad6d33b01339284382b674d5c37bce45123fdde',
)
TOKENS = (1, 2, 3)
KV_ROW_BYTES = 64 * 34  # K and V: 8 heads of 128 values, Q8_0 blocks of 32.


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_stage(path, shape, digest=None):
    raw = path.read_bytes()
    if len(raw) != int(np.prod(shape)) * 4:
        raise ValueError(f'{path}: incorrect stage size')
    if digest and sha256(raw) != digest:
        raise ValueError(f'{path}: reference SHA256 mismatch')
    values = np.frombuffer(raw, dtype='<f4').reshape(shape)
    if not np.isfinite(values).all() or not np.any(values):
        raise ValueError(f'{path}: nonfinite or all-zero stage')
    return values


def float_metrics(reference, actual):
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    delta = a - b
    x, y = a.reshape(len(a), -1), b.reshape(len(b), -1)
    cosine = np.sum(x * y, axis=-1) / np.maximum(
        np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-30)
    return {'max_absolute_error': float(np.abs(delta).max()),
            'relative_rmse': float(np.sqrt(np.sum(delta * delta) / np.sum(a * a))),
            'mean_cosine': float(cosine.mean()),
            'element_equal_fraction': float((a == b).mean())}


def read_kv(path, digest=None):
    raw = path.read_bytes()
    if len(raw) != len(TOKENS) * KV_ROW_BYTES or (digest and sha256(raw) != digest):
        raise ValueError(f'{path}: KV size or reference SHA256 mismatch')
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = np.ascontiguousarray(blocks[:, :2]).view('<f2')
    if not np.isfinite(scales).all() or not np.any(scales) or not np.any(blocks[:, 2:]):
        raise ValueError(f'{path}: nonfinite or all-zero KV')
    return raw


def compare(native, reference, report_path):
    raw_report = report_path.read_bytes()
    if sha256(raw_report) != REFERENCE_REPORT_SHA256:
        raise ValueError('Reference report does not match the pinned published-artifact comparison')
    source = json.loads(raw_report)
    comparisons, native_hashes = {}, {}
    for name, stage in source['stages'].items():
        path = native / (name + '.f32')
        expected = read_stage(reference / path.name, stage['shape'], stage['sha256'])
        actual = read_stage(path, stage['shape'])
        comparisons[name] = float_metrics(expected, actual)
        comparisons[name]['archived_native_relative_rmse'] = source['comparisons'][name]['relative_rmse']
        native_hashes[path.name] = sha256(path.read_bytes())
    embedding_exact = comparisons['embedding']['element_equal_fraction'] == 1

    expected_ids = np.asarray(source['comparisons']['layer1-routing']['reference_ids'], dtype='<u4')
    raw_ids = (native / 'layer1-ids.u32').read_bytes()
    if len(raw_ids) != expected_ids.nbytes:
        raise ValueError('layer1-ids.u32: incorrect routing size')
    actual_ids = np.frombuffer(raw_ids, dtype='<u4').reshape(expected_ids.shape)
    routing_exact = np.array_equal(expected_ids, actual_ids)
    comparisons['layer1-routing'] = {'order_exact': bool(routing_exact),
        'reference_ids': expected_ids.tolist(), 'native_ids': actual_ids.tolist()}
    native_hashes['layer1-ids.u32'] = sha256(raw_ids)

    for layer, digest in enumerate(REFERENCE_KV_SHA256):
        name = f'layer{layer}-kv.q8_0'
        expected, actual = read_kv(reference / name, digest), read_kv(native / name)
        a, b = np.frombuffer(expected, dtype=np.uint8), np.frombuffer(actual, dtype=np.uint8)
        comparisons[f'layer{layer}-kv'] = {'exact_bytes': expected == actual,
            'byte_equal_fraction': float((a == b).mean())}
        native_hashes[name] = sha256(actual)

    return {'scope': source['scope'], 'tokens': list(TOKENS), 'positions': [0, 1, 2],
            'reference_report_sha256': REFERENCE_REPORT_SHA256,
            'reference_weight_mode': source['weight_mode'],
            'native': str(native.resolve()), 'reference': str(reference.resolve()),
            'native_sha256': native_hashes, 'comparisons': comparisons,
            'gates': {'finite_nonzero_stages': True, 'embedding_exact': embedding_exact,
                      'routing_exact': bool(routing_exact)},
            'structural_gates_passed': bool(embedding_exact and routing_exact),
            'numerical_quality_qualified': False,
            'limitations': source['limitations'] + [
                'Numerical deltas require review; this tool imposes no global float/KV threshold.',
                'Archived routed projections use the archived native expert order. Routing mismatch invalidates those comparisons.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True, help='pinned independent comparison JSON')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.native, args.reference, args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'compared_stages': len(report['comparisons']),
                      'gates': report['gates'], 'numerical_quality_qualified': False}))
    return 0 if report['structural_gates_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
