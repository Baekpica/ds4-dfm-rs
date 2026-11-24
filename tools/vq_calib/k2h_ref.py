#!/usr/bin/env python3
"""K2-Horizon-375B-A23B reference forward, layer-streamed from the original BF16 checkpoint.

The arithmetic is the manual forward of the MQ87 release's calibration collector
(`run_k2_bf16_imatrix.py`), bit-identical to transformers' K2HorizonDecoderLayer on layers 0 and 3:

    h += o(attn(q, k, v))      q/k/v = proj(rmsnorm(h)); partial NeoX RoPE on the first 64 of 128 head dims
                               (theta 1e7), GQA 48 query / 8 KV heads, causal SDPA in BF16, scale 128^-1/2
    h += mlp(rmsnorm(h))       layers 0-2: SwiGLU 16,384; layers 3-60: shared expert + 8 of 192 routed experts
                               (FFN 1,792): score = sigmoid(router x), selection by score + bias, weight =
                               2.5 score / sum of the 8 scores, routed sum in BF16 in ascending expert order

RMSNorm runs in F32 and returns BF16; every linear is a BF16 GEMM. One MoE layer is one safetensors shard
(model-000{l+1}-of-00061), 12.9 GB; `Source.read_layer` reads it with pread into pinned host buffers.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

SOURCE_REPO = 'IFM/K2-Horizon-375B-A23B'
SOURCE_REV = 'd33e3ae45281865ebf9f044b12d3635b1d1e17fe'
L, DENSE = 61, 3
E, TOPK = 192, 8
H, FF, DENSE_FF = 6144, 1792, 16384
HEADS, KV_HEADS, HD = 48, 8, 128
ROPE, THETA = 64, 10_000_000.0
EPS = 1e-6
ROUTER_SCALE = 2.5
VOCAB = 250624
MOE_LAYERS = tuple(range(DENSE, L))
PROJ = ('gate_proj', 'up_proj', 'down_proj')
EXPERT_SHAPES = {'gate_proj': (FF, H), 'up_proj': (FF, H), 'down_proj': (H, FF)}


# ------------------------------------------------------------------------------------ source
def _pread_into(fd, view, offset):
    done, n = 0, len(view)
    while done < n:
        r = os.preadv(fd, [view[done:done + (1 << 30)]], offset + done)
        if r <= 0:
            raise IOError('short read')
        done += r


class Source:
    """Tensor byte ranges of the BF16 shards; reads go straight into host buffers."""

    def __init__(self, path):
        self.path = Path(path)
        self.map = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers, self.fds = {}, {}

    def header(self, shard):
        if shard not in self.headers:
            with open(self.path / shard, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                meta = json.loads(f.read(n))
            meta.pop('__metadata__', None)
            self.headers[shard] = (8 + n, meta)
        return self.headers[shard]

    def info(self, name):
        shard = self.map[name]
        base, meta = self.header(shard)
        m = meta[name]
        if m['dtype'] != 'BF16':
            raise ValueError(f'{name}: {m["dtype"]}')
        lo, hi = m['data_offsets']
        return shard, base + lo, hi - lo, tuple(m['shape'])

    def fd(self, shard):
        if shard not in self.fds:
            self.fds[shard] = os.open(self.path / shard, os.O_RDONLY)
        return self.fds[shard]

    def close(self, shard=None):
        for s in ([shard] if shard else list(self.fds)):
            if s in self.fds:
                os.close(self.fds.pop(s))

    def read_into(self, name, out):
        """out: a contiguous host int16 tensor of the tensor's element count."""
        shard, off, size, shape = self.info(name)
        if out.numel() * 2 != size:
            raise ValueError(f'{name}: buffer size')
        _pread_into(self.fd(shard), memoryview(out.numpy()).cast('B'), off)

    def tensor(self, name):
        """One tensor as a host BF16 tensor."""
        shard, off, size, shape = self.info(name)
        out = torch.empty(shape, dtype=torch.int16)
        self.read_into(name, out)
        return out.view(torch.bfloat16)

    @staticmethod
    def names(l):
        p = f'model.layers.{l}.'
        base = {'attn_norm': p + 'input_layernorm.weight', 'ffn_norm': p + 'post_attention_layernorm.weight',
                'q': p + 'self_attn.q_proj.weight', 'k': p + 'self_attn.k_proj.weight',
                'v': p + 'self_attn.v_proj.weight', 'o': p + 'self_attn.o_proj.weight'}
        if l < DENSE:
            base.update({'dense_gate': p + 'mlp.gate_proj.weight', 'dense_up': p + 'mlp.up_proj.weight',
                         'dense_down': p + 'mlp.down_proj.weight'})
        else:
            base.update({'router': p + 'mlp.gate.weight', 'router_bias': p + 'mlp.gate.bias',
                         'shared_gate': p + 'mlp.shared_experts.gate_proj.weight',
                         'shared_up': p + 'mlp.shared_experts.up_proj.weight',
                         'shared_down': p + 'mlp.shared_experts.down_proj.weight'})
        return base

    @staticmethod
    def expert_name(l, e, proj):
        return f'model.layers.{l}.mlp.experts.{e}.{proj}.weight'

    def read_layer(self, l, stacks=None, threads=12, experts=True):
        """Host tensors of layer l: the small ones as BF16 tensors, the expert stacks (when `experts`) in
        `stacks` (pinned int16 [E, FF, H] x2 and [E, H, FF], allocated when None)."""
        host = {k: self.tensor(n) for k, n in self.names(l).items()}
        if l >= DENSE and experts:
            if stacks is None:
                stacks = new_stacks()
            jobs = [(self.expert_name(l, e, p), stacks[p][e]) for p in PROJ for e in range(E)]
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda j: self.read_into(*j), jobs))
            host['stacks'] = stacks
        self.close(self.map[self.names(l)['q']])
        return host


def new_stacks(pin=True):
    return {p: torch.empty((E, *EXPERT_SHAPES[p]), dtype=torch.int16, pin_memory=pin) for p in PROJ}


class LayerWeights:
    def __init__(self, l, host, dev):
        self.l, self.dev = l, dev
        for k, v in host.items():
            if k != 'stacks':
                setattr(self, k, v.to(dev))
        self.moe = l >= DENSE
        if self.moe and 'stacks' in host:
            st = host['stacks']
            self.gate = st['gate_proj'].to(dev, non_blocking=True).view(torch.bfloat16)
            self.up = st['up_proj'].to(dev, non_blocking=True).view(torch.bfloat16)
            self.down = st['down_proj'].to(dev, non_blocking=True).view(torch.bfloat16)
            torch.cuda.synchronize(dev)

    def expert(self, e):
        """(gate over up [2 FF, H], down [H, FF]), BF16."""
        return torch.cat([self.gate[e], self.up[e]]), self.down[e]


# ----------------------------------------------------------------------------------- forward
def rmsnorm(x, weight):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS)
    return (y * weight.float()).to(x.dtype)


def _split_to_interleaved(x):
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def _interleaved_to_split(x):
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def _rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def rope_cache(seq, dev, dtype=torch.bfloat16, first=0):
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, device=dev, dtype=torch.float32) / ROPE))
    pos = torch.arange(first, first + seq, device=dev, dtype=torch.float32)
    freqs = pos[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def apply_partial_rope(query, key, cos, sin):
    q_rot, q_pass = torch.split(_split_to_interleaved(query), [ROPE, HD - ROPE], dim=-1)
    k_rot, k_pass = torch.split(_split_to_interleaved(key), [ROPE, HD - ROPE], dim=-1)
    q_rot = _interleaved_to_split(q_rot)
    k_rot = _interleaved_to_split(k_rot)
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    query = _interleaved_to_split(torch.cat((_split_to_interleaved(q_rot), q_pass), dim=-1))
    key = _interleaved_to_split(torch.cat((_split_to_interleaved(k_rot), k_pass), dim=-1))
    return query, key


def nvfp4(x):
    """NVFP4 store/load of K or V rows (the runtime's K/V contract, the MiniMax / Inkling-2 encoder):
    per 16 consecutive values of the last dimension an E4M3 scale max(amax / 6, 2^-9) and E2M1 codes
    (0, 0.5, 1, 1.5, 2, 3, 4, 6 with sign), nearest, ties to the even code. BF16 in, BF16 out."""
    shape = x.shape
    v = x.float().reshape(-1, 16)
    scale = (v.abs().amax(-1, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
    scale = scale.to(torch.float8_e4m3fn).float()
    scale = torch.where(scale > 0, scale, torch.full_like(scale, 2.0 ** -9))
    t = (v / scale).clamp(-6.0, 6.0)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    a = t.abs()
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2)          # midpoints; right=False: ties go down
    lo = grid[idx]
    # ties to even code index: a midpoint between codes i and i+1 goes to the even one
    mid = (grid[1:] + grid[:-1]) / 2
    tie_up = torch.zeros_like(a, dtype=torch.bool)
    for i in range(7):
        if (i + 1) % 2 == 0:
            tie_up |= (a == mid[i])
    q = torch.where(tie_up, grid[(idx + 1).clamp_max(7)], lo)
    return (torch.sign(t) * q * scale).reshape(shape).to(x.dtype)


def attention(lw, x, cos, sin, kv=None):
    """x [B, T, H] BF16 (already normalized). kv: None or 'nvfp4' (K after RoPE and V through the codec)."""
    b, t, _ = x.shape
    q = F.linear(x, lw.q).view(b, t, HEADS, HD).transpose(1, 2)
    k = F.linear(x, lw.k).view(b, t, KV_HEADS, HD).transpose(1, 2)
    v = F.linear(x, lw.v).view(b, t, KV_HEADS, HD).transpose(1, 2)
    q, k = apply_partial_rope(q, k, cos, sin)
    if kv == 'nvfp4':
        k, v = nvfp4(k), nvfp4(v)
    elif kv is not None:
        raise ValueError(kv)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                                         scale=HD ** -0.5, enable_gqa=True)
    out = out.transpose(1, 2).reshape(b, t, HEADS * HD)
    return F.linear(out, lw.o)


def dense(lw, x):
    return F.linear(F.silu(F.linear(x, lw.dense_gate)) * F.linear(x, lw.dense_up), lw.dense_down)


def route(lw, flat):
    scores = torch.sigmoid(F.linear(flat, lw.router).float())
    sel = torch.topk(scores + lw.router_bias.float(), k=TOPK, dim=-1, sorted=False).indices
    w = scores.gather(1, sel)
    w = w / w.sum(dim=-1, keepdim=True)
    return sel, (w * ROUTER_SCALE).to(flat.dtype)


def moe(lw, flat, hooks=None, experts=None, routes=None):
    """flat [n, H] BF16 (post-attention RMSNorm output) -> (mlp output [n, H] BF16, selected experts [n, 8]).
    hooks.expert(e, x, act, w): the tokens routed to expert e, their down input and route weights.
    experts(e) -> (gate, up, down) replaces the layer's own expert weights (VQ students).
    routes: (sel, weights) to force (the original's routes)."""
    sel, wts = route(lw, flat) if routes is None else routes
    flat_routes = sel.reshape(-1)
    counts = torch.bincount(flat_routes, minlength=E)
    order = torch.argsort(flat_routes, stable=True)
    offs = [0] + counts.cumsum(0).tolist()
    final = torch.zeros_like(flat)
    for e in range(E):
        b0, b1 = offs[e], offs[e + 1]
        if b0 == b1:
            continue
        pos = order[b0:b1]
        tok = torch.div(pos, TOPK, rounding_mode='floor')
        slot = pos.remainder(TOPK)
        x = flat[tok]
        gate, up, down = (lw.gate[e], lw.up[e], lw.down[e]) if experts is None else experts(e)
        act = F.silu(F.linear(x, gate)) * F.linear(x, up)
        w = wts[tok, slot, None]
        final.index_add_(0, tok, (F.linear(act, down) * w).to(final.dtype))
        if hooks is not None:
            hooks.expert(e, x, act, w[:, 0])
    shared = F.silu(F.linear(flat, lw.shared_gate)) * F.linear(flat, lw.shared_up)
    final.add_(F.linear(shared, lw.shared_down))
    return final, sel


def layer_forward(lw, h, cos, sin, hooks=None, experts=None, kv=None):
    """In place on h [B, T, H] BF16. -> selected experts [B*T, 8] (MoE layers) or None."""
    h.add_(attention(lw, rmsnorm(h, lw.attn_norm), cos, sin, kv))
    x = rmsnorm(h, lw.ffn_norm)
    if not lw.moe:
        h.add_(dense(lw, x))
        return None
    y, sel = moe(lw, x.reshape(-1, H), hooks, experts)
    h.add_(y.view_as(h))
    return sel


class Globals:
    """Embedding, final norm and output head."""

    def __init__(self, src, dev, embed=True, head=False):
        self.dev = dev
        self.embed = src.tensor('model.embed_tokens.weight').to(dev) if embed else None
        self.norm = src.tensor('model.norm.weight').to(dev)
        self.head = src.tensor('lm_head.weight').to(dev) if head else None

    def logits(self, h):
        """h [n, H] BF16 final hidden state -> [n, VOCAB] F32."""
        return F.linear(rmsnorm(h, self.norm), self.head).float()
