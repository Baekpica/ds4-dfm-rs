"""Pinned FP8 block-scaled safetensors source for skt/A.X-K2.

Reads tensors by byte range from the shard headers (no safetensors import on
the hot path), in parallel threads, straight into pinned host memory, and
dequantizes the 128x128 block-scaled E4M3 weights:

    W[i, j] = float32(fp8[i, j]) * weight_scale_inv[i // 128, j // 128]

`weight_scale_inv` multiplies despite its name (DeepSeek-V3 convention, also
transformers' FP8 integration and llama.cpp convert_hf_to_gguf.py
`dequant_simple`: scale.repeat_interleave(block, dim) for each dim, cropped
to the weight shape when a dimension is not a multiple of 128, e.g.
kv_a_proj_with_mqa [576, 7168] with a [5, 56] scale). The product is formed
in FP32 (a 4-bit significand times a 24-bit one rounds once, <= 2^-24 rel).
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct

import numpy as np
import torch

SOURCE_REPO = 'skt/A.X-K2'
SOURCE_REV = 'd0805cfd1fc647b4078c0d9f232c80d34ee0c893'
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
BLOCK = 128
DTYPES = {'F8_E4M3': (torch.float8_e4m3fn, 1), 'BF16': (torch.bfloat16, 2),
          'F32': (torch.float32, 4), 'F16': (torch.float16, 2)}


def dequant(w, scale, out_dtype=torch.float32):
    """E4M3 weight [R, C] and F32 scale [ceil(R/128), ceil(C/128)] -> out_dtype."""
    rows, cols = w.shape
    if scale.shape != ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK):
        raise ValueError(f'scale shape {tuple(scale.shape)} for weight {tuple(w.shape)}')
    s = scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:rows, :cols]
    return (w.float() * s).to(out_dtype)


class Source:
    def __init__(self, root='/models/A.X-K2'):
        self.root = Path(root)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.config = json.loads((self.root / 'config.json').read_text())
        self._headers = {}

    def header(self, shard):
        if shard not in self._headers:
            with (self.root / shard).open('rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                meta = json.loads(f.read(n))
            meta.pop('__metadata__', None)
            self._headers[shard] = (8 + n, meta)
        return self._headers[shard]

    def info(self, name):
        shard = self.index[name]
        base, meta = self.header(shard)
        t = meta[name]
        start, end = t['data_offsets']
        return shard, base + start, end - start, t['dtype'], tuple(t['shape'])

    def has(self, name):
        return name in self.index

    def read(self, names, threads=16, pin=True):
        """Read many tensors in parallel -> dict name -> CPU tensor in the stored dtype."""
        names = list(names)
        infos = {n: self.info(n) for n in names}
        out = {}
        for n, (shard, off, size, dtype, shape) in infos.items():
            torch_dtype, width = DTYPES[dtype]
            if size != width * int(np.prod(shape)):
                raise ValueError(f'size mismatch {n}')
            out[n] = torch.empty(shape, dtype=torch_dtype, pin_memory=pin)

        def load(n):
            shard, off, size, _, _ = infos[n]
            view = out[n].view(torch.uint8).reshape(-1).numpy()
            fd = os.open(self.root / shard, os.O_RDONLY)
            try:
                got = 0
                while got < size:
                    chunk = os.preadv(fd, [memoryview(view[got:])], off + got)
                    if chunk <= 0:
                        raise IOError(f'short read {n}')
                    got += chunk
            finally:
                os.close(fd)
            return n

        with ThreadPoolExecutor(threads) as pool:
            for _ in pool.map(load, names):
                pass
        return out

    def weight_names(self, name):
        """The stored tensors that make up one logical weight (with its scale, if FP8)."""
        scale = name + '_scale_inv'
        return [name, scale] if scale in self.index else [name]

    def load(self, names, device='cuda', dtype=torch.bfloat16, threads=16):
        """Logical weights -> dict name -> dequantized tensor on `device` in `dtype`.
        FP8 weights are dequantized in FP32 on the device, then cast once."""
        stored = [s for n in names for s in self.weight_names(n)]
        host = self.read(stored, threads=threads)
        out = {}
        for n in names:
            w = host[n].to(device, non_blocking=True)
            scale = n + '_scale_inv'
            if scale in host:
                out[n] = dequant(w, host[scale].to(device, non_blocking=True), dtype)
            else:
                out[n] = w.to(dtype) if dtype is not None else w
        return out

    def expert_names(self, layer):
        return [f'model.layers.{layer}.mlp.experts.{e}.{p}.weight'
                for e in range(self.config['n_routed_experts']) for p in PROJECTIONS]

    def moe_layers(self):
        k = self.config['first_k_dense_replace']
        return list(range(k, self.config['num_hidden_layers']))
