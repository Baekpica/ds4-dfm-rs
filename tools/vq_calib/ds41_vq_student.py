"""Routed experts of a DeepSeek-V4.1-Flash VQ conversion for the ds41_ref forward.

`VQExperts(path)` reads DQVL v3 blobs from an assembled base GGUF or from a directory of
`blob-LL.bin` files written by convert_layer_ds41.py (the next layer's bytes on a host
thread), decodes them on the device (ds41_vq_format layout: 12-bit LSB-first pairs plus the
13th-bit plane) and hands the forward `expert(e) -> (w1, w3, w2)` in BF16,
W = e4m3(cb[idx]) * f16 gain.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch

import ds41_vq_format as fmt

BF = torch.bfloat16
SHAPES = {'w1': (2304, 5120), 'w3': (2304, 5120), 'w2': (5120, 2304)}


def decode_blob(blob, nexp=384, chunk=64):
    """blob: uint8 tensor on the device -> (cb f32 [K, 8], {p: idx int16 [nexp, rows, nidx]},
    {p: gains f32 [nexp, rows]})."""
    hdr = blob[:16].cpu().numpy().view(np.uint32)
    assert hdr[0] == fmt.DQVL_MAGIC and hdr[1] == 3 and hdr[3] == nexp, hdr
    # K from the first payload header (u16 nc at +6), found through slot[0][w1]
    slot0 = int(blob[16:24].cpu().numpy().view(np.uint64)[0])
    nc = int(blob[slot0 + 6:slot0 + 8].cpu().numpy().view(np.uint16)[0])
    lay = fmt.blob_layout(nexp, nc, SHAPES)
    assert lay.total == blob.numel(), (lay.total, blob.numel())
    cb = blob[lay.cb_off:lay.cb_off + nc * 8].view(torch.float8_e4m3fn).float().view(nc, 8)
    body = blob[lay.body:lay.body + nexp * lay.expert_stride].view(nexp, lay.expert_stride)
    idx, gains = {}, {}
    for p, (rows, cols) in SHAPES.items():
        nidx = cols // 8
        g0 = lay.pay_off[p] + fmt.PAYLOAD_HDR
        m0 = g0 + rows * 2
        m1 = m0 + rows * lay.mrow[p]
        gains[p] = body[:, g0:m0].contiguous().view(torch.float16).float()
        out = torch.empty((nexp, rows, nidx), dtype=torch.int16, device=blob.device)
        for e0 in range(0, nexp, chunk):
            e1 = min(nexp, e0 + chunk)
            m = body[e0:e1, m0:m1].reshape(e1 - e0, rows, -1, 3).int()
            a = m[..., 0] | ((m[..., 1] & 0xF) << 8)
            b = (m[..., 1] >> 4) | (m[..., 2] << 4)
            ix = torch.stack([a, b], -1).reshape(e1 - e0, rows, nidx)
            if lay.prow[p]:
                pl = body[e0:e1, m1:m1 + rows * lay.prow[p]].reshape(e1 - e0, rows, -1).int()
                bits = (pl.unsqueeze(-1) >> torch.arange(8, device=blob.device)) & 1
                ix = ix | (bits.reshape(e1 - e0, rows, -1)[..., :nidx] << 12)
            out[e0:e1] = ix.to(torch.int16)
        idx[p] = out
    return cb, idx, gains


class VQExperts:
    def __init__(self, path, device='cuda'):
        self.path, self.dev = path, device
        self.blobs = os.path.isdir(path)
        self.g = None if self.blobs else fmt.open_base(path if path.endswith('.gguf') else None)
        self.pool = ThreadPoolExecutor(1)
        self.pending = {}

    def _read(self, L):
        if self.blobs:
            return np.fromfile(os.path.join(self.path, f'blob-{L:02d}.bin'), dtype=np.uint8)
        t = self.g.tensor(f'blk.{L}.ffn_exps_vq.blob')
        return self.g.vf.pread(t.abs_offset, t.nbytes)

    def prefetch(self, L):
        if 0 <= L < 40 and L not in self.pending:
            self.pending[L] = self.pool.submit(self._read, L)

    def layer(self, L):
        self.prefetch(L)
        raw = self.pending.pop(L).result()
        self.prefetch(L + 1)
        blob = torch.from_numpy(raw).to(self.dev)
        del raw
        cb, idx, gains = decode_blob(blob)
        del blob

        def expert(e):
            return tuple((cb[idx[p][e].long()].flatten(1) * gains[p][e][:, None]).to(BF) for p in ('w1', 'w3', 'w2'))
        return expert
