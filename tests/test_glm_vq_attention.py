#!/usr/bin/env python3
"""Source eager attention arithmetic over interleaved GLM K/V rows."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    args = parser.parse_args()
    lib = ct.CDLL(str(args.lib.resolve()))
    fn = lib.test_glmvq_attention
    fn.argtypes = [ct.c_void_p] * 5 + [ct.c_uint32] * 5 + [ct.c_void_p]
    fn.restype = ct.c_int
    torch.manual_seed(5353)
    torch.backends.cuda.matmul.allow_tf32 = False
    dim = 256
    for rows, keys, heads in [(1, 1, 64), (1, 2048, 64), (7, 33, 3), (64, 512, 3)]:
        q = torch.randn((rows, heads, dim), device='cuda').bfloat16()
        kv = torch.randn((keys, heads, 2 * dim), device='cuda').bfloat16()
        positions = torch.arange(keys - rows, keys, device='cuda', dtype=torch.int32)
        for sparse in [False, True]:
            mask = torch.arange(keys, device='cuda')[None] <= positions[:, None]
            selected = None
            if sparse:
                selected = torch.randint(-1, keys + 2, (rows, min(keys + 3, 2051)), device='cuda', dtype=torch.int32)
                selected[:, 0] = 0  # Every query has at least one visible key.
                membership = torch.zeros((rows, keys), device='cuda', dtype=torch.int32)
                valid = (selected >= 0) & (selected < keys)
                membership.scatter_add_(1, selected.long().clamp(0, keys - 1), valid.int())
                mask &= membership.bool()
            query = q.permute(1, 0, 2)
            key, value = [x.permute(1, 0, 2) for x in kv.split(dim, -1)]
            scores = (query @ key.transpose(-1, -2)) * (dim ** -.5)
            scores = scores.masked_fill(~mask[None], torch.finfo(torch.bfloat16).min)
            probabilities = scores.float().softmax(-1).bfloat16()
            expected = (probabilities @ value).permute(1, 0, 2).float()
            out = torch.empty_like(q, dtype=torch.float32)
            qf, kvf = q.float(), kv.float()
            rc = fn(qf.data_ptr(), kvf.data_ptr(), out.data_ptr(), positions.data_ptr(),
                    selected.data_ptr() if selected is not None else None,
                    selected.shape[1] if selected is not None else 0, rows, keys, heads, dim,
                    torch.cuda.current_stream().cuda_stream)
            assert rc == 0, rc
            error = out - expected
            report = {'rows': rows, 'keys': keys, 'heads': heads, 'sparse': sparse,
                      'max_abs': float(error.abs().max()), 'rel_rms': float(error.norm() / expected.norm())}
            print(json.dumps(report), flush=True)
            assert torch.equal(out, out.bfloat16().float())
            assert report['rel_rms'] < .002
            bound = .008 * expected.abs() + .001 * expected.square().mean(-1, keepdim=True).sqrt()
            assert torch.all(error.abs() <= bound), report
    print('GLMVQ_ATTENTION_PASS', flush=True)


if __name__ == '__main__':
    main()
