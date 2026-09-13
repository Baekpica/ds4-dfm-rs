#!/usr/bin/env python3
"""Actual Step F16 projector oracle using pinned source equations and RoPE.

The native F16 GEMM rounds its input to half, accumulates/output in F32;
--contract fp32 provides the separate unrounded-input source control.
No language model is loaded. Run serially under the host memory guard.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Union

import gguf
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DIM, FFN, HEADS, HEAD, LAYERS = 1536, 8960, 16, 96, 47


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sidecar', type=Path)
    parser.add_argument('source', type=Path, help='pinned vision_encoder.py')
    parser.add_argument('pixels', type=Path, help='normalized CHW little-endian F32')
    parser.add_argument('output', type=Path)
    parser.add_argument('--edge', type=int, choices=[504, 728], required=True)
    parser.add_argument('--contract', choices=['fp16-input', 'fp32'], default='fp16-input')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    reader = gguf.GGUFReader(str(args.sidecar))
    tensors = {t.name: t for t in reader.tensors}
    assert len(tensors) == 667
    hashes = {}

    def load(name):
        t = tensors[name]
        assert t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
        hashes[name] = hashlib.sha256(t.data).hexdigest()
        return torch.from_numpy(np.array(t.data, dtype=np.float32, copy=True).reshape(tuple(reversed(t.shape)))).to(device)

    def rounded(x):
        return x.half().float() if args.contract == 'fp16-input' else x

    def linear(x, weight, bias=None):
        y = F.linear(rounded(x), load(weight))
        return y + load(bias) if bias else y

    def norm(x, prefix):
        return F.layer_norm(x, (DIM,), load(prefix + '.weight'), load(prefix + '.bias'), 1e-5)

    def save(name, x):
        x.detach().cpu().contiguous().numpy().astype('<f4').tofile(args.output / (name + '.f32'))

    # Execute the actual source's interleaved pair/x-y frequency construction.
    source = args.source.read_bytes()
    names = {'rotate_half', 'apply_rotary_emb', 'EncoderRope2D'}
    definitions = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(definitions) == len(names)
    scope = dict(torch=torch, nn=nn, Union=Union)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(args.source), 'exec'), scope)
    rope = scope['EncoderRope2D'](HEAD, 52, 52).to(device)
    pixels = np.fromfile(args.pixels, dtype='<f4')
    assert pixels.size == args.edge * args.edge * 3
    x = torch.from_numpy(pixels.copy()).reshape(1, 3, args.edge, args.edge).to(device)
    save('pixels', x)
    x = F.conv2d(rounded(x), load('v.patch_embd.weight'), stride=14)
    grid = args.edge // 14
    x = x.flatten(2).transpose(1, 2)
    position = load('v.position_embd.weight')
    if grid != 52:
        position = F.interpolate(position.reshape(52, 52, DIM).permute(2, 0, 1).unsqueeze(0),
                                 size=(grid, grid), mode='bilinear', align_corners=False)
        position = position.squeeze(0).permute(1, 2, 0).reshape(grid * grid, DIM)
    x = norm(x + position, 'v.pre_ln')
    save('pre', x)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for layer in range(LAYERS):
            prefix = f'v.blk.{layer}.'
            a = norm(x, prefix + 'ln1')
            qkv = linear(a, prefix + 'attn_qkv.weight', prefix + 'attn_qkv.bias')
            q, k, v = [t.reshape(1, grid * grid, HEADS, HEAD).transpose(1, 2) for t in qkv.chunk(3, -1)]
            q, k = rope(q, k, (grid, grid))
            a = F.scaled_dot_product_attention(q, k, v, scale=HEAD ** -0.5)
            a = a.transpose(1, 2).reshape(1, grid * grid, DIM)
            x = x + linear(a, prefix + 'attn_out.weight', prefix + 'attn_out.bias') * load(prefix + 'ls1.weight')
            save(f'attn-{layer}', x)
            a = linear(norm(x, prefix + 'ln2'), prefix + 'ffn_up.weight', prefix + 'ffn_up.bias')
            a = a * torch.sigmoid(1.702 * a)
            x = x + linear(a, prefix + 'ffn_down.weight', prefix + 'ffn_down.bias') * load(prefix + 'ls2.weight')
            save(f'layer-{layer}', x)
            print(f'edge {args.edge} layer {layer}', flush=True)
        x = x.transpose(1, 2).reshape(1, DIM, grid, grid)
        for i in range(2):
            x = F.conv2d(rounded(x), load(f'mm.{i}.weight'), stride=2, padding=1) + load(f'mm.{i}.bias')[None, :, None, None]
            save(f'down-{i}', x.flatten(2).transpose(1, 2))
        x = linear(x.flatten(2).transpose(1, 2), 'mm.model.fc.weight')
        save('projection', x)
    metadata = dict(source_revision='5f6244077ac62e04eec3f320501ff8c2b293373a',
                    source_sha256=hashlib.sha256(source).hexdigest(),
                    sidecar=str(args.sidecar), weights=hashes, torch=torch.__version__,
                    device=torch.cuda.get_device_name(), contract=args.contract, edge=args.edge,
                    pixel_sha256=hashlib.sha256(args.pixels.read_bytes()).hexdigest())
    (args.output / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('complete', tuple(x.shape), flush=True)


if __name__ == '__main__':
    main()
