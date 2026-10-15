#!/usr/bin/env python3
"""Q4_K converter contract and direct-dot error on the same packed weights."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q4tool', ROOT / 'tools/glm_vq_q4k.py')
q4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q4)


def unpack(data, rows, cols):
    # Independent GGML superblock decoder, including all eight six-bit pairs.
    b = np.frombuffer(data, np.uint8).reshape(-1, 144)
    d = b[:, :4].copy().view('<f2').astype(np.float64)
    s = b[:, 4:16].astype(np.uint16)
    out = np.empty((len(b), 256), np.float64)
    for group in range(8):
        scale = s[:, group] & 63 if group < 4 else (s[:, group + 4] & 15) + ((s[:, group - 4] // 64) * 16)
        minimum = s[:, group + 4] & 63 if group < 4 else s[:, group + 4] // 16 + ((s[:, group] // 64) * 16)
        codes = b[:, 16 + group // 2 * 32:48 + group // 2 * 32]
        codes = codes // 16 if group % 2 else codes % 16
        out[:, group * 32:(group + 1) * 32] = d[:, :1] * scale[:, None] * codes - d[:, 1:] * minimum[:, None]
    return out.reshape(rows, cols)


def converter_gate(quant, selection='all'):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Distinct FP8 scales catch missing/inverted scales and wrong tile axes.
        shape = [256, 256]
        weights = [np.full(shape, 0x3f80, dtype='<u2').tobytes(),
                   np.full(shape, 0x38, dtype=np.uint8).tobytes()]
        scales = np.array([[.5, 2.], [4., .25]], dtype='<f4')
        names = ['model.language_model.layers.0.self_attn.q_proj.weight',
                 'model.language_model.layers.0.mlp.gate_proj.weight']
        scale_name = names[1].replace('.weight', '.weight_scale_inv')
        records, entries, payloads = {}, [], []
        offset = 0
        for name, typ, dims, data in [
            (names[0], q4.BF16, shape, weights[0]),
            (names[1], q4.FP8, shape, weights[1]),
            (scale_name, 0, [2, 2], scales.tobytes()),
        ]:
            entries.append(dict(name=name, typ=typ, dims=tuple(reversed(dims)), out_offset=offset))
            records[name] = dict(type_id=typ, stored_shape=dims, payload_bytes=len(data),
                                 payload_sha256=hashlib.sha256(data).hexdigest())
            payloads.append(data)
            offset = q4.align(offset + len(data))
        meta = {'general.architecture': (8, 'glm5_next'), 'general.alignment': (4, 128),
                'glm_vq.format_version': (4, 1),
                'glm_vq.fp8_scales': (8, json.dumps({names[1]: scale_name}))}
        src, dst = tmp / 'in.gguf', tmp / 'out.gguf'
        with src.open('wb') as f:
            head = q4.header(meta, entries)
            f.write(head)
            for e, data in zip(entries, payloads):
                f.write(bytes(len(head) + e['out_offset'] - f.tell()))
                f.write(data)
        keep = {names[1]} if selection == 'preserve' else set()
        q4.rewrite(src, dst, records, quant, 2, keep)
        with dst.open('rb') as f:
            meta, entries, base = q4.read_header(f)
            expected_links = {names[1]: scale_name} if keep else {}
            assert json.loads(meta['glm_vq.fp8_scales'][1]) == expected_links
            for i, e in enumerate(entries):
                f.seek(base + e['offset'])
                data = f.read(records[e['name']]['payload_bytes'])
                if i == 2:
                    assert data == scales.tobytes()
                    continue
                if e['name'] in keep:
                    assert e['typ'] == q4.FP8 and data == weights[i]
                    continue
                expected = np.ones(shape, np.float32) if i == 0 else scales.repeat(128, 0).repeat(128, 1)
                packed = np.empty(len(data), np.uint8)
                assert quant.fn(12, expected.ctypes.data, packed.ctypes.data, 0, *shape, None) == len(data)
                # The unchanged C quantizer isolates the converter's tile,
                # scale and dtype plumbing from its own weight-rounding loss.
                assert data == packed.tobytes()
    print('GLMVQ_Q4_CONVERTER_PASS', flush=True)


def dot_gate(quant, lib):
    fn = ct.CDLL(str(lib.resolve())).test_glmvq_q4
    fn.argtypes = [ct.c_void_p] * 3 + [ct.c_uint32] * 3
    fn.restype = ct.c_int
    rng = np.random.default_rng(5353)
    for rows, cols in [(17, 256), (256, 512), (256, 2048), (256, 4096), (256, 8192)]:
        w = q4.bf16_round(rng.normal(0, .02, (rows, cols)).astype(np.float32))
        packed = np.empty(rows * cols // 256 * 144, np.uint8)
        assert quant.fn(12, w.ctypes.data, packed.ctypes.data, 0, rows, cols, None) == packed.size
        restored = unpack(packed.tobytes(), rows, cols)
        for tokens in [1, 3, 8]:
            x = q4.bf16_round(rng.normal(size=(tokens, cols)).astype(np.float32))
            got = np.empty((tokens, rows), np.float32)
            assert fn(packed.ctypes.data, x.ctypes.data, got.ctypes.data, rows, cols, tokens) == 0
            expected = x.astype(np.float64) @ restored.T
            absolute = np.abs(x.astype(np.float64)) @ np.abs(restored.T)
            # Weight decode, FMA/tree reduction, then a single BF16 rounding.
            u = 2.0**-24
            ops = cols + 32
            gamma = ops * u / (1 - ops * u)
            ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(expected), 2.0**-126))) - 8)
            bound = gamma * absolute + ulp + 2.0**-126
            error = np.abs(got.astype(np.float64) - expected)
            assert np.isfinite(got).all() and (error <= bound).all(), (rows, cols, tokens, float((error-bound).max()))
            print(json.dumps(dict(shape=[rows, cols], tokens=tokens, max_abs=float(error.max()),
                                  max_bound_ratio=float((error / bound).max()))), flush=True)
    print('GLMVQ_Q4_DOT_PASS', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quant', type=Path, default=ROOT / 'gguf-tools/libds4q.so')
    p.add_argument('--cuda', type=Path, default=ROOT / 'tests/libglm_vq_linear.so')
    p.add_argument('--scope', choices=['all', 'preserve'], default='all')
    args = p.parse_args()
    quant = q4.Quantizer(args.quant)
    converter_gate(quant, args.scope)
    if args.scope == 'all':
        dot_gate(quant, args.cuda)


if __name__ == '__main__':
    main()
