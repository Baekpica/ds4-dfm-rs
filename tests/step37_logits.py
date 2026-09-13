#!/usr/bin/env python3
"""Compare finite full-vocabulary logits from identical Step MQ83 workloads."""
import argparse
import array
import json
import math
from pathlib import Path


def read(path, vocab):
    data = array.array('f')
    data.frombytes(path.read_bytes())
    if not data or len(data) % vocab:
        raise ValueError('incomplete vocabulary rows')
    return [data[i:i + vocab] for i in range(0, len(data), vocab)]


def compare(reference, actual):
    finite = all(math.isfinite(x) for x in reference) and all(math.isfinite(x) for x in actual)
    if not finite:
        return {'finite': False, 'passed': False}
    error = math.fsum((a - r) ** 2 for r, a in zip(reference, actual))
    ref2 = math.fsum(r * r for r in reference)
    actual2 = math.fsum(a * a for a in actual)
    dot = math.fsum(r * a for r, a in zip(reference, actual))
    ref_token = max(range(len(reference)), key=reference.__getitem__)
    token = max(range(len(actual)), key=actual.__getitem__)
    rel_rms = math.sqrt(error / ref2) if ref2 else math.inf
    cosine = dot / math.sqrt(ref2 * actual2) if ref2 and actual2 else 0
    return dict(finite=True, rel_rms=rel_rms, cosine=cosine,
                max_abs=max(abs(r - a) for r, a in zip(reference, actual)),
                reference_token=ref_token, token=token,
                passed=rel_rms <= 0.03 and cosine >= 0.999 and token == ref_token)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('actual', type=Path)
    p.add_argument('--vocab', type=int, default=128896)
    args = p.parse_args()
    reference, actual = read(args.reference, args.vocab), read(args.actual, args.vocab)
    if len(reference) != len(actual):
        raise ValueError('different row counts')
    rows = [compare(r, a) for r, a in zip(reference, actual)]
    print(json.dumps(dict(reference=str(args.reference), actual=str(args.actual), rows=rows), indent=2))
    return 0 if all(r.get('passed', False) for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
