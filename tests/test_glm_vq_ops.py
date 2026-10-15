#!/usr/bin/env python3
"""Source BF16 arithmetic gates, without loading model weights.

Oracle: Transformers 5.19.0 modeling_glm5_next.py, SHA256
a56b3f59f79c4430505bbc85557e0abdc35098561b831ab79cf14661aac1f6a5.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path

import torch
import torch.nn.functional as F

STREAMS = 4
MIX = STREAMS * (STREAMS + 2)
EPS = 1e-5
HC_EPS = 1e-6
SINKHORN = 20


class Ops:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        p, u, f = ct.c_void_p, ct.c_uint32, ct.c_float
        for name, args in {
            'rms': [p] * 4 + [u, u, f, u, p],
            'swiglu': [p] * 3 + [u, f, p],
            'hc_split': [p] * 6 + [u, u, u, f, p],
            'hc_expand': [p] * 4 + [u, u, p],
        }.items():
            fn = getattr(self.lib, 'test_glmvq_' + name)
            fn.argtypes, fn.restype = args, ct.c_int

    def call(self, name, *args):
        args = [x.data_ptr() if isinstance(x, torch.Tensor) else x for x in args]
        rc = getattr(self.lib, 'test_glmvq_' + name)(*args, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, (name, rc)
        torch.cuda.synchronize()


def check(name, got, expected, max_ulp=0):
    assert torch.isfinite(got).all(), name
    # The graph stores BF16 values in F32 buffers; reject missing boundaries.
    assert torch.equal(got, got.bfloat16().float()), (name, 'not BF16 rounded')
    a, b = got.bfloat16().view(torch.int16).int(), expected.bfloat16().view(torch.int16).int()
    # Monotonic bit order, with both signed zeros mapped to zero.
    a = torch.where(a < 0, -32768 - a, a)
    b = torch.where(b < 0, -32768 - b, b)
    ulp = (a - b).abs()
    report = {'op': name, 'elements': got.numel(), 'different': int((ulp != 0).sum()),
              'max_bf16_ulp': int(ulp.max()), 'max_abs': float((got - expected.float()).abs().max())}
    print(json.dumps(report), flush=True)
    assert report['max_bf16_ulp'] <= max_ulp, report


def bf16(*shape):
    return torch.randn(shape, device='cuda', dtype=torch.bfloat16)


def rms(ops):
    # Lines 66-82: round normalized input BEFORE the weighted product.
    for rows in [1, 7, 33]:
        for width in [128, 1536, 4096, 16384]:
            x = bf16(rows, width)
            w = (1 + torch.randn(width, device='cuda') * .2).bfloat16()
            gate = bf16(rows, width)
            xf, gf = x.float(), gate.float()
            norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
            out = torch.empty_like(xf)
            ops.call('rms', xf, w, None, out, rows, width, EPS, 0)
            check(f'rms/{rows}/{width}', out, w * norm.bfloat16(), 1)
            # Lines 375-394: gated norm keeps the weighted/gated path FP32.
            ops.call('rms', xf, w, gf, out, rows, width, EPS, 1)
            check(f'gated_rms/{rows}/{width}', out,
                  (norm * w.float() * torch.sigmoid(gf)).bfloat16(), 1)
            ops.call('rms', xf, None, None, out, rows, width, EPS, 2)
            torch.testing.assert_close(out, norm, rtol=5e-7, atol=1e-7)


def swiglu(ops):
    # Lines 98-105: only gate's upper side clamps; SiLU rounds before product.
    gate = torch.cat([bf16(131072) * 8,
                      torch.tensor([-100, -32, -11, -10, 0, 10, 11, 32, 100], device='cuda')]).bfloat16()
    up = (bf16(gate.numel()) * 8).clamp(-32, 32)
    out = torch.empty_like(gate, dtype=torch.float32)
    ops.call('swiglu', gate.float(), up.float(), out, gate.numel(), 10.)
    expected = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    check('swiglu', out, expected)


def hc(ops):
    # Lines 286-329 and 1300-1352: F32 mixing, then separate BF16 operations.
    for rows in [1, 7, 33]:
        width = 4096
        residual, branch = bf16(rows, STREAMS, width), bf16(rows, width)
        mix = torch.randn((rows, MIX), device='cuda') * 3
        scale = torch.tensor([.3, .4, .5], device='cuda')
        base = torch.randn(MIX, device='cuda')
        pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + HC_EPS
        post = 2 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
        comb = (mix[:, 8:] * scale[2] + base[8:]).view(rows, 4, 4).softmax(-1) + HC_EPS
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
        for _ in range(SINKHORN - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
            comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
        expected_split = torch.cat([pre, post, comb.flatten(1)], -1)
        split, out = torch.empty_like(mix), torch.empty_like(branch, dtype=torch.float32)
        ops.call('hc_split', mix, residual.float(), scale, base, out, split,
                 rows, width, SINKHORN, HC_EPS)
        torch.testing.assert_close(split, expected_split, rtol=2e-6, atol=2e-7)
        check(f'hc_collapse/{rows}', out, (pre.unsqueeze(-1) * residual).sum(1).bfloat16(), 1)

        expanded = torch.empty_like(residual, dtype=torch.float32)
        # Isolate expansion from the small F32 softmax/reduction differences.
        ops.call('hc_expand', branch.float(), residual.float(), expected_split,
                 expanded, rows, width)
        expected = post.bfloat16().unsqueeze(-1) * branch.unsqueeze(1) + torch.matmul(
            comb.bfloat16().transpose(-1, -2), residual)
        check(f'hc_expand/{rows}', expanded, expected, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--case', choices=['all', 'rms', 'swiglu', 'hc'], default='all')
    args = parser.parse_args()
    torch.manual_seed(5353)
    ops = Ops(args.lib)
    for name, test in [('rms', rms), ('swiglu', swiglu), ('hc', hc)]:
        if args.case in ['all', name]:
            test(ops)
    print('GLMVQ_SOURCE_OPS_PASS', flush=True)


if __name__ == '__main__':
    main()
