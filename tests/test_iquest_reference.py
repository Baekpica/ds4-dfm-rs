#!/usr/bin/env python3
"""Independent NumPy parity for the IQuest numerical reference, CPU only."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.POINTER(ctypes.c_float)
UINT = ctypes.POINTER(ctypes.c_uint)
BYTE = ctypes.POINTER(ctypes.c_uint8)
WRAPPER = r'''
#include "ds4_iquest_ref.h"
void ref_rms(float *o,const float *x,const float *w,unsigned n){iquest_rms(o,x,w,n);}
void ref_rope(float *x,unsigned h,unsigned p,float t){iquest_rope(x,h,p,t);}
void ref_router(unsigned *i,float *w,const float *x){iquest_router(i,w,x);}
void ref_pack(void *o,const float *x,unsigned n){iquest_pack(o,x,n);}
void ref_attn(float *o,const float *q,const void *c,const float *s,unsigned p,unsigned n,unsigned w){iquest_attn(o,q,c,s,p,n,w);}
void ref_add(float *o,const float *x,const float *n,const float *a,unsigned count,unsigned first){iquest_attn_add(o,x,n,a,count,first);}
void ref_ffn(float *o,const float *r,const float *f,unsigned n,unsigned first){iquest_ffn_add(o,r,f,n,first);}
unsigned ref_full(unsigned l){return iquest_full(l);}
uint16_t ref_half(float x){return iquest_half_bits(x);}
float ref_unhalf(uint16_t x){return iquest_half(x);}
'''


def ptr(x):
    return x.ctypes.data_as(FLOAT)


def bind(path):
    lib = ctypes.CDLL(str(path))
    signatures = {
        'ref_rms': [FLOAT, FLOAT, FLOAT, ctypes.c_uint],
        'ref_rope': [FLOAT, ctypes.c_uint, ctypes.c_uint, ctypes.c_float],
        'ref_router': [UINT, FLOAT, FLOAT],
        'ref_pack': [ctypes.c_void_p, FLOAT, ctypes.c_uint],
        'ref_attn': [FLOAT, FLOAT, ctypes.c_void_p, FLOAT] + [ctypes.c_uint] * 3,
        'ref_add': [FLOAT] * 4 + [ctypes.c_uint] * 2,
        'ref_ffn': [FLOAT] * 3 + [ctypes.c_uint] * 2,
        'ref_full': [ctypes.c_uint],
        'ref_half': [ctypes.c_float],
        'ref_unhalf': [ctypes.c_uint16],
    }
    for name, sig in signatures.items():
        getattr(lib, name).argtypes = sig
        getattr(lib, name).restype = None
    lib.ref_full.restype = ctypes.c_uint
    lib.ref_half.restype = ctypes.c_uint16
    lib.ref_unhalf.restype = ctypes.c_float
    return lib


def unpack(raw):
    blocks = raw.reshape(-1, 34)
    scale = np.ascontiguousarray(blocks[:, :2]).view('<f2').astype(np.float32).reshape(-1, 1)
    return (scale * blocks[:, 2:].view(np.int8)).reshape(-1)


def pack_ref(x):
    rows = x.reshape(-1, 32)
    scale = np.max(np.abs(rows), axis=1) / np.float32(127)
    inverse = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
    values = rows * inverse[:, None]
    quant = np.copysign(np.floor(np.abs(values) + np.float32(0.5)), values).astype(np.int8)
    raw = np.empty((len(rows), 34), dtype=np.uint8)
    raw[:, :2] = scale.astype('<f2').view(np.uint8).reshape(-1, 2)
    raw[:, 2:] = quant.view(np.uint8)
    return raw.reshape(-1)


def round_bf16(x):
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def main():
    rng = np.random.default_rng(20261001)
    evidence = {}
    with tempfile.TemporaryDirectory(prefix='iquest-ref-') as tmp:
        tmp = Path(tmp)
        (tmp / 'wrapper.c').write_text(WRAPPER)
        subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-I', str(ROOT),
                        str(tmp / 'wrapper.c'), '-lm', '-o', str(tmp / 'ref.so')], check=True)
        lib = bind(tmp / 'ref.so')
        assert [i for i in range(88) if lib.ref_full(i)] == [0] + list(range(1, 85, 4)) + [85, 86, 87]
        evidence['layer_pattern'] = {'full': 25, 'sliding': 63}

        values = np.concatenate([rng.normal(size=10000).astype(np.float32),
                                 np.array([0, -0., 2**-24, 2**-25, 65504, -65504, np.inf, -np.inf], dtype=np.float32)])
        actual = np.array([lib.ref_half(float(x)) for x in values], dtype=np.uint16)
        np.testing.assert_array_equal(actual, values.astype(np.float16).view(np.uint16))
        bits = np.arange(65536, dtype=np.uint16)
        finite = (bits & 0x7c00) != 0x7c00
        restored = np.array([lib.ref_unhalf(int(x)) for x in bits[finite]], dtype=np.float32)
        np.testing.assert_array_equal(restored, bits[finite].view(np.float16).astype(np.float32))
        evidence['fp16_scale_conversion'] = {'finite_half_bit_patterns': int(finite.sum()), 'passed': True}

        x = rng.normal(size=3072).astype(np.float32)
        w = rng.uniform(0.7, 1.3, size=3072).astype(np.float32)
        actual = np.empty_like(x)
        lib.ref_rms(ptr(actual), ptr(x), ptr(w), len(x))
        expected = x * np.float32(1 / np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-6)) * w
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)
        evidence['rms_max_error'] = float(np.max(np.abs(actual - expected)))

        rope_errors = []
        for theta in (10000., 1000000.):
            for position in (0, 1, 4095, 4096, 524287):
                source = rng.normal(size=(48, 128)).astype(np.float32)
                actual = source.copy()
                lib.ref_rope(ptr(actual), 48, position, theta)
                freq = np.float32(1) / np.power(np.float32(theta), np.arange(16, dtype=np.float32) / np.float32(16))
                angle = np.float32(position) * freq
                expected = source.copy()
                a, b = source[:, :16], source[:, 16:32]
                expected[:, :16] = a * np.cos(angle) - b * np.sin(angle)
                expected[:, 16:32] = b * np.cos(angle) + a * np.sin(angle)
                np.testing.assert_array_equal(actual[:, 32:], source[:, 32:])
                maximum = float(np.max(np.abs(actual - expected)))
                if position < 4095:
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                rope_errors.append({'theta': theta, 'position': position, 'max_numpy_error': maximum})
        evidence['rope_frequency_implementation_drift'] = rope_errors

        logits = rng.normal(size=256).astype(np.float32) * 50
        ids, weights = np.empty(8, dtype=np.uint32), np.empty(8, dtype=np.float32)
        lib.ref_router(ids.ctypes.data_as(UINT), ptr(weights), ptr(logits))
        expected_ids = np.argsort(-logits, kind='stable')[:8]
        expected = np.exp(logits[expected_ids].astype(np.float64) - logits[expected_ids[0]])
        expected /= expected.sum()
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(weights, expected, atol=3e-7, rtol=3e-7)
        evidence['router_max_error'] = float(np.max(np.abs(weights - expected)))

        x = rng.normal(size=2048).astype(np.float32)
        x[:32] = 0
        packed = np.empty(2048 // 32 * 34, dtype=np.uint8)
        lib.ref_pack(packed.ctypes.data, ptr(x), len(x))
        np.testing.assert_array_equal(packed, pack_ref(x))
        evidence['q8_pack_exact'] = True

        attn_errors = []
        for position, capacity, window in ((12, 16, 0), (31, 7, 4), (31, 7, 7)):
            query = rng.normal(size=(48, 128)).astype(np.float32)
            sink = rng.normal(size=(8, 128)).astype(np.float32)
            keys = rng.normal(size=(position + 1, 8, 128)).astype(np.float32)
            values = rng.normal(size=keys.shape).astype(np.float32)
            cache = np.zeros((capacity, 64 * 34), dtype=np.uint8)
            for token in range(position + 1):
                cache[token % capacity, :32*34] = pack_ref(keys[token])
                cache[token % capacity, 32*34:] = pack_ref(values[token])
            actual = np.empty_like(query)
            lib.ref_attn(ptr(actual), ptr(query), cache.ctypes.data, ptr(sink), position, capacity, window)
            expected = np.empty_like(query)
            combined_only = np.empty_like(query)
            start = max(0, position + 1 - window) if window else 0
            for h in range(48):
                kh = h // 6
                kk = np.stack([unpack(cache[t % capacity, :32*34]).reshape(8, 128)[kh] for t in range(start, position + 1)])
                vv = np.stack([unpack(cache[t % capacity, 32*34:]).reshape(8, 128)[kh] for t in range(start, position + 1)])
                scores = kk.astype(np.float64) @ query[h] / np.sqrt(128)
                maximum = scores.max()
                normal_lse = maximum + np.log(np.exp(scores - maximum).sum())
                probs = np.exp(scores - normal_lse)
                ordinary = round_bf16(probs @ vv)
                sink_logit = query[h].astype(np.float64) @ sink[kh] / np.sqrt(128)
                factor = 1 / (1 + np.exp(sink_logit - normal_lse))
                expected[h] = round_bf16(ordinary.astype(np.float64) * factor)
                combined_only[h] = round_bf16((probs @ vv) * factor)
            assert np.all((actual.view(np.uint32) & 0xffff) == 0), 'Sink output must store BF16-rounded values'
            equal = float((actual == expected).mean())
            assert equal >= 0.99, f'Two-stage sink rounding differs on {1-equal:.3%} of outputs'
            negative_equal = float((combined_only == expected).mean())
            assert negative_equal < 0.99, 'Fixture must distinguish combined final-only sink rounding'
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0.008)
            attn_errors.append({'position': position, 'capacity': capacity, 'window': window,
                                'bf16_element_equal_fraction': equal,
                                'negative_control_combined_final_only_equal_fraction': negative_equal,
                                'max_error': float(np.max(np.abs(actual - expected)))})
        evidence['learned_key_sink_q8_attention'] = attn_errors

        raw, normalized, branch = [rng.normal(size=3072).astype(np.float32) for _ in range(3)]
        actual = np.empty_like(raw)
        for first in (0, 1):
            lib.ref_add(ptr(actual), ptr(raw), ptr(normalized), ptr(branch), 3072, first)
            np.testing.assert_array_equal(actual, (raw if first else normalized) + branch)
            lib.ref_ffn(ptr(actual), ptr(raw), ptr(branch), 3072, first)
            np.testing.assert_allclose(actual, raw + branch * np.float32(1 if first else 0.53881590608), atol=2e-7, rtol=2e-7)
        evidence['normalized_residual_and_scale'] = True
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
