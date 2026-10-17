#!/usr/bin/env python3
"""Bounded native linear gate: source BF16 and explicit FP32 projections."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from test_glm_vq_cuda import fixture


class Linear:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        p, u = ct.c_void_p, ct.c_uint32
        self.fn = self.lib.test_glmvq_linear
        self.fn.argtypes = [p] * 6 + [u] * 4 + [p] * 2 + [u] * 2 + [p]
        self.fn.restype = ct.c_int

    def run(self, x, shape, *, weight=None, scale=None, components=None, compute=0):
        rows, cols = shape
        components = components or [None] * 4
        precision = 1
        storage = 0 if scale is None else 1
        if components[0] is not None:
            main, plane, book, gain, rows, cols, precision = components
            components = [torch.from_numpy(t.copy()).cuda() if t is not None else None
                          for t in [main, plane, book, gain]]
            storage = 2
        ptrs = [t.data_ptr() if t is not None else None for t in [weight, scale, *components]]
        out = torch.empty((x.shape[0], rows), dtype=torch.float32, device='cuda')
        rc = self.fn(*ptrs, rows, cols, precision, storage, x.data_ptr(), out.data_ptr(),
                     x.shape[0], compute, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
        return out


def compare(name, got, expected, *, float32=False):
    ref = expected.float()
    delta = got - ref
    report = {'case': name, 'max_abs': float(delta.abs().max()),
              'rms': float(delta.square().mean().sqrt()),
              'rel_rms': float(delta.norm() / ref.norm().clamp_min(1e-20))}
    print(json.dumps(report), flush=True)
    if float32:
        torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-4)
    else:
        assert torch.equal(got, got.bfloat16().float()), name
        # GEMM reduction order may change near cancellation; bound vector
        # error and per-entry error relative to this row's signal scale.
        assert report['rel_rms'] < .002
        bound = .008 * ref.abs() + .0001 * ref.square().mean(-1, keepdim=True).sqrt()
        assert torch.all(delta.abs() <= bound), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(5353)
    torch.backends.cuda.matmul.allow_tf32 = False
    linear = Linear(args.lib)
    for rows, cols in [(2048, 4096), (4096, 2048), (12288, 4096), (8192, 128)]:
        w = (torch.randn((rows, cols), device='cuda') * .01).bfloat16()
        raw = w.to(torch.float8_e4m3fn)
        scale = torch.rand((rows // 128, cols // 128), device='cuda') * .1
        restored = (raw.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
        for tokens in [1, 4, 32, 512]:
            x = torch.randn((tokens, cols), device='cuda').bfloat16().float()
            for name, weight, scales, reference in [('bf16', w, None, w), ('fp8', raw, scale, restored)]:
                got = linear.run(x, w.shape, weight=weight, scale=scales)
                compare(f'{name}/{rows}/{cols}/{tokens}', got, F.linear(x.bfloat16(), reference))
    for rows, cols in [(24, 16384), (288, 4096)]:
        w = (torch.randn((rows, cols), device='cuda') * .01).bfloat16()
        for tokens in [1, 7, 32]:
            x = torch.randn((tokens, cols), device='cuda')
            compare(f'f32/{rows}/{cols}/{tokens}', linear.run(x, w.shape, weight=w, compute=1),
                    F.linear(x, w.float()), float32=True)
    manifest = json.loads((args.fixtures / 'manifest.json').read_text())
    for entry in manifest['fixtures']:
        folder = args.fixtures / entry['id']
        gold = np.load(folder / 'expected.npz')
        x = torch.from_numpy(gold['linear_input_f32']).cuda().bfloat16().float()
        components = fixture(folder / 'fixture.gguf')
        w = torch.from_numpy(gold['vq_expected_f32']).cuda().bfloat16()
        compare(f'gvq1/{entry["id"]}', linear.run(x, w.shape, components=components), F.linear(x.bfloat16(), w))
    print('GLMVQ_LINEAR_PASS', flush=True)


if __name__ == '__main__':
    main()
