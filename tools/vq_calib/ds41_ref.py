#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash reference forward in plain PyTorch, for calibration statistics
and teacher-forced evaluation of VQ conversions.

It follows the official `inference/model.py` and `inference/kernel.py` of
deepseek-ai/DeepSeek-V4.1-Flash-1017 on the prefill path (start_pos 0, text only,
one sequence at a time for attention): hyper-connections with Sinkhorn mixes,
Engram n-gram lookups on layers 1 and 14, sliding-window attention plus
compressed KV (ratio 2 on layers 2-19, ratio 1 on 20-39) with the two-level
indexer (top-512, candidate blocks from layer 20), the sqrtsoftplus router with
correction bias, SwiGLU experts with the training clamps, and the shared expert.

The tilelang kernels are replaced by exact emulations on any CUDA GPU:
- FP8/FP4 weights are dequantized to BF16 (exact: e4m3/e2m1 times a power-of-two
  scale fits BF16), activations take the kernels' FP8 quant/dequant with UE8M0
  per-32 scales (`act_quant`), so `F.linear` in BF16 with FP32 accumulation is the
  kernels' GEMM up to summation order;
- the KV, index-key, index-query and compressed-KV quant/dequant steps
  (`act_quant(..., inplace=True)`, `fp4_act_quant`) are emulated value by value;
- `sparse_attn` gathers the selected KV rows, keeps scores in FP32, casts the
  probabilities to BF16 for the PV product and adds the sink to the denominator.

Experts come from a provider, so the same forward runs the FP4 teacher and any
dequantized VQ student. Statistics (`Stats`) are FP64 sums on the device.
"""
import glob
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

BF = torch.bfloat16
FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ----------------------------------------------------------------------------- source

class Source:
    """Tensor name -> shard map for the HF checkpoint (one shard per layer)."""

    def __init__(self, root):
        self.root = root
        self.loc, self.hdr = {}, {}
        self.refresh()
        self.args = json.load(open(os.path.join(root, 'inference', 'config.json')))

    def refresh(self):
        for f in sorted(glob.glob(os.path.join(self.root, 'model-*.safetensors'))):
            if f in self.hdr:
                continue
            with open(f, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                h = json.loads(fh.read(n))
            self.hdr[f] = (n, h)
            for k in h:
                if k != '__metadata__':
                    self.loc[k] = f

    def names(self, prefix):
        return sorted(k for k in self.loc if k.startswith(prefix))

    def read(self, names, pin=False):
        by_file = {}
        for k in names:
            by_file.setdefault(self.loc[k], []).append(k)
        out = {}
        for f, ks in by_file.items():
            with safe_open(f, 'pt') as fh:
                for k in ks:
                    t = fh.get_tensor(k)
                    out[k] = t.pin_memory() if pin else t
        return out

    def memmap(self, name):
        """Raw uint8 memmap of a 2-D tensor (Engram tables), shape [rows, row_bytes]."""
        f = self.loc[name]
        n, h = self.hdr[f]
        meta = h[name]
        a, b = meta['data_offsets']
        rows = meta['shape'][0]
        return np.memmap(f, dtype=np.uint8, mode='r', offset=8 + n + a, shape=(rows, (b - a) // rows))


# ----------------------------------------------------------------------------- numerics

def pow2_ceil(v):
    """2**ceil(log2(v)) for v > 0 (the kernels' fast_round_scale)."""
    m, e = torch.frexp(v)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(v), e)


def act_quant(x, block=32):
    """FP8 e4m3 quant/dequant with UE8M0 per-`block` scales (act_quant, round_scale, inplace)."""
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], shp[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / 448.0)
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(shp).to(x.dtype)


_FP4_BOUNDS = {}


def fp4_round(y):
    """Round to the e2m1 grid, ties to even (cvt.rn.satfinite)."""
    dev = y.device
    if dev not in _FP4_BOUNDS:
        _FP4_BOUNDS[dev] = (torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev),
                            torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev))
    b, v = _FP4_BOUNDS[dev]
    a = y.abs()
    i = torch.bucketize(a, b, right=False)
    i = i + ((a == 0.75) | (a == 1.75) | (a == 3.5)).long()
    return torch.copysign(v[i], y)


def fp4_quant(x, block=32, e4m3_scale=False):
    """fp4_act_quant(..., inplace=True): E8M0 scales (indexer) or E4M3 scales (compressed KV)."""
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], shp[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6 * 2 ** -9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = pow2_ceil(amax.clamp_min(6 * 2 ** -126) / 6.0)
    y = fp4_round((xf / s).clamp(-6.0, 6.0)) * s
    return y.reshape(shp).to(x.dtype)


def fp8_dequant(w, s, block=32):
    """FP8 e4m3 [O, I] with E8M0 [ceil(O/b), ceil(I/b)] block scales -> BF16 (exact)."""
    O, I = w.shape
    sf = s.float().repeat_interleave(block, 0)[:O].repeat_interleave(block, 1)[:, :I]
    return (w.float() * sf).to(BF)


def fp4_dequant(w, s, out=BF):
    """Packed e2m1 [O, I/2] (low nibble first) with E8M0 [O, I/32] scales -> [O, I] (exact)."""
    u = w.view(torch.uint8)
    tab = torch.tensor(FP4_TABLE, device=w.device, dtype=torch.float32)
    x = torch.stack([tab[(u & 15).long()], tab[(u >> 4).long()]], -1).flatten(-2)
    O, I = x.shape
    x = x.view(O, I // 32, 32) * s.float().unsqueeze(-1)
    return x.view(O, I).to(out)


def rms_norm(x, w, eps=1e-20):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def lin_q(x, w, chunk=65536):
    """linear() for FP8/FP4 weights: FP8 activation quant/dequant, BF16 GEMM, BF16 out."""
    if x.shape[0] <= chunk:
        return F.linear(act_quant(x), w)
    return torch.cat([F.linear(act_quant(x[i:i + chunk]), w) for i in range(0, x.shape[0], chunk)])


def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def rope_(x, f, inverse=False):
    """In-place rotary on the last dim of x [S, ..., 64]; f [S, 32] complex."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        f = f.conj()
    if xc.ndim == 3:
        f = f[:, None, :]
    x.copy_(torch.view_as_real(xc * f).flatten(-2))
    return x


def window_idxs(s, win, device):
    end = torch.arange(s, device=device)[:, None]
    idx = (end - win + 1).clamp(min=0) + torch.arange(min(s, win), device=device)
    return torch.where(idx > end, -1, idx)


def sparse_attn(q, kv, sink, idxs, scale, chunk=256):
    """q [s, H, D] bf16, kv [n, D] bf16, idxs [s, k] (-1 = none) -> [s, H, D] bf16."""
    s = q.shape[0]
    out = torch.empty_like(q)
    for c0 in range(0, s, chunk):
        c1 = min(s, c0 + chunk)
        ii = idxs[c0:c1]
        valid = ii >= 0
        kvg = kv[ii.clamp(min=0)].float()                                   # [c, k, D]
        sc = torch.einsum('chd,ckd->chk', q[c0:c1].float(), kvg) * scale
        sc = sc.masked_fill(~valid[:, None, :], float('-inf'))
        m = sc.amax(-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(sc - m)
        l = p.sum(-1) + torch.exp(sink[None, :] - m.squeeze(-1))
        o = torch.einsum('chk,ckd->chd', p.to(BF).float(), kvg)
        out[c0:c1] = (o / l.unsqueeze(-1)).to(BF)
    return out


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float('-inf'))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    nb = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(nb, device=logits.device) == last, float('inf'))
    top = scores.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float('-inf'))
    return keep                                                             # [c, nb] block mask


# ----------------------------------------------------------------------------- hyper-connections

def hc_mixes(x, fn, scale, base, hc=4, iters=20, eps=1e-6, norm_eps=1e-20):
    xf = x.flatten(1).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, fn) * rs
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, pre):
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(f, residual, post, comb):
    # sum_j comb[j, k] * residual[j] as one FP32 batched product (no [n, hc, hc, D] temporary)
    y = post.unsqueeze(-1) * f.unsqueeze(-2) + torch.einsum('njk,njd->nkd', comb, residual.float())
    return y.to(f.dtype)


# ----------------------------------------------------------------------------- Engram hashing

def find_next_prime(start, seen):
    from sympy import isprime
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


class EngramHash:
    """inference/engram.py NgramHashState for whole sequences (start_pos 0, no image spans)."""

    def __init__(self, args, tok_dir, device):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir)
        lookup, vocab = self.compressed_map(tok)
        assert vocab == args['engram_compressed_vocab_size'], (vocab, args['engram_compressed_vocab_size'])
        self.layer_ids = tuple(args['engram_layer_ids'])
        self.n = args['engram_max_ngram_size']
        nh = args['engram_n_heads']
        primes, seen = [], set()
        for _ in self.layer_ids:
            per = []
            for _ in range(self.n - 1):
                sizes, cur = [], args['engram_vocab_size'] - 1
                for _ in range(nh):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(sizes)
            primes.append(per)
        flat = [[p for pn in layer for p in pn] for layer in primes]
        offsets = [np.cumsum([0, *s[:-1]]) for s in flat]
        bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
        mult = []
        for L in self.layer_ids:
            g = np.random.default_rng(10007 * L)
            mult.append(torch.tensor(g.integers(0, bound, size=(self.n,), dtype=np.int64) * 2 + 1))
        self.primes = torch.tensor(primes, device=device)                  # [nl, n-1, nh]
        self.offsets = torch.tensor(np.array(offsets), device=device)      # [nl, (n-1)*nh]
        self.mult = torch.stack(mult).to(device)                          # [nl, n]
        self.map = torch.tensor(lookup, device=device)
        self.pad = int(self.map[args['engram_pad_id']])

    @staticmethod
    def compressed_map(tokenizer):
        from tokenizers import Regex, normalizers
        sentinel = ''
        norm = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r'[ \t\r\n]+'), ' '), normalizers.Replace(Regex(r'^ $'), sentinel),
            normalizers.Strip(), normalizers.Replace(sentinel, ' ')])
        backend = tokenizer.backend_tokenizer
        key_to_new, lookup = {}, [0] * len(tokenizer)
        for tid in range(len(tokenizer)):
            text = backend.decode([tid], skip_special_tokens=False)
            if '�' in text:
                key = backend.id_to_token(tid)
            else:
                nrm = norm.normalize_str(text)
                key = nrm if nrm else text
            nid = key_to_new.get(key)
            if nid is None:
                nid = len(key_to_new)
                key_to_new[key] = nid
            lookup[tid] = nid
        return lookup, len(key_to_new)

    def __call__(self, ids):
        """ids [s] one sequence -> [s, n_layers, (n-1)*n_heads] int64 global row ids."""
        c = self.map[ids.long()]
        s = c.shape[0]
        pos = torch.arange(s, device=c.device)
        toks = []
        for shift in range(self.n):
            src = c[(pos - shift).clamp_min(0)]
            toks.append(torch.where(pos < shift, self.pad, src))
        toks = torch.stack(toks, -1)                                       # [s, n]
        prod = toks.unsqueeze(1) * self.mult                               # [s, nl, n]
        rolling, hs = prod[..., 0], []
        for i in range(1, self.n):
            rolling = torch.bitwise_xor(rolling, prod[..., i])
            hs.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hs, -1) + self.offsets


# ----------------------------------------------------------------------------- weights

class Teacher:
    """Per-layer weights of the HF checkpoint; experts stay packed FP4 on the device."""

    def __init__(self, root, device):
        self.src = Source(root)
        self.a = self.src.args
        self.dev = device

    def host_layer(self, L):
        # the Engram tables (2 x ~100 GB) are read in rows through Source.memmap, never whole
        names = [k for k in self.src.names(f'layers.{L}.') if '.engram.embed.' not in k]
        return self.src.read(names, pin=True)

    def layer(self, L, host):
        p = f'layers.{L}.'
        w = {k[len(p):]: v.to(self.dev, non_blocking=True) for k, v in host.items()
             if not k.startswith(p + 'engram.embed.')}
        out = {}
        for k, v in w.items():
            if '.experts.' in k and not k.startswith('ffn.shared_experts'):
                continue
            if k.endswith('.scale'):
                continue
            s = w.get(k[:-len('weight')] + 'scale') if k.endswith('.weight') else None
            if s is not None and v.dtype == torch.float8_e4m3fn:
                out[k] = fp8_dequant(v, s)
            else:
                out[k] = v
        pk = {k: v for k, v in w.items() if '.experts.' in k and not k.startswith('ffn.shared_experts')}

        def expert(e):
            g = f'ffn.experts.{e}.'
            return (fp4_dequant(pk[g + 'w1.weight'], pk[g + 'w1.scale']),
                    fp4_dequant(pk[g + 'w3.weight'], pk[g + 'w3.scale']),
                    fp4_dequant(pk[g + 'w2.weight'], pk[g + 'w2.scale']))
        out['experts'] = expert
        return out


# ----------------------------------------------------------------------------- statistics

class Stats:
    """FP64 sums for one MoE layer (docs/vq-calibration.md, Statistics)."""

    def __init__(self, a, device):
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device=device)
        E, H, Fm = a['n_routed_experts'], a['dim'], a['moe_inter_dim']
        self.t = {'tokens': z(1), 'count': torch.zeros(E, dtype=torch.int64, device=device),
                  'w1': z(E), 'w2': z(E), 'x2': z(E, H), 'x2w': z(E, H), 'a2': z(E, Fm), 'a2w': z(E, Fm),
                  'ffn_in': z(H), 'shared_a': z(Fm)}


# ----------------------------------------------------------------------------- layer

class SeqState:
    """What attention hands down the stack for one sequence (SharedAttentionRuntime)."""
    __slots__ = ('compress_kv', 'index_k', 'topk_idxs', 'cand')

    def __init__(self):
        self.compress_kv = self.index_k = self.topk_idxs = self.cand = None


class Layer:
    def __init__(self, a, L, w, rope_cache, engram=None):
        self.a, self.L, self.w = a, L, w
        self.ratio = a['compress_ratios'][L]
        self.kv_source = L in a['kv_source_layers']
        self.index_source = L in a['index_source_layers']
        self.cand_source = L == a['candidate_source_layer']
        self.uses_cand = 0 <= a['candidate_source_layer'] < L
        self.freqs = rope_cache[1 if self.ratio else 0]
        self.engram = engram

    # -- attention ------------------------------------------------------------
    def compressor(self, x):
        w, r = self.w, self.ratio
        if r == 1:
            return rms_norm(F.linear(x, w['attn.compressor.wkv.weight']), w['attn.compressor.norm.weight'])
        s = x.shape[0]
        xf = x.float()
        kv = F.linear(xf, w['attn.compressor.wkv.weight'].float())
        sc = F.linear(xf, w['attn.compressor.wgate.weight'].float())
        cut = s - s % r
        kv = kv[:cut].view(-1, r, kv.shape[-1])
        sc = sc[:cut].view(-1, r, sc.shape[-1])
        kv = (kv * sc.softmax(dim=1)).sum(dim=1)
        return rms_norm(kv.to(BF), w['attn.compressor.norm.weight'])

    def indexer(self, x, qr, latent, st, q_chunk=512):
        a, w, r = self.a, self.w, self.ratio
        s = x.shape[0]
        rd = a['rope_head_dim']
        clen = s // r
        if self.kv_source and latent is not None and latent.shape[0] > 0:
            k = rms_norm(F.linear(latent, w['attn.indexer.wk.weight']), w['attn.indexer.k_norm.weight'])
            rope_(k[..., -rd:], self.freqs[:s - s % r:r])
            st.index_k = fp4_quant(k, 32)
        topk = min(a['index_topk'], clen)
        nb = (clen + a['candidate_block_size'] - 1) // a['candidate_block_size']
        need_scores = topk < clen or (self.cand_source and nb > a['candidate_topk_blocks'])
        if not need_scores:
            # every reachable compressed position is selected: no scores needed
            pos = torch.arange(clen, device=x.device)
            lens = (torch.arange(1, s + 1, device=x.device) // r)[:, None]
            if self.cand_source:
                st.cand = None
            return torch.where(pos[None, :] < lens, pos[None, :] + s, -1).int()
        H, Dh = a['index_n_heads'], a['index_head_dim']
        q = lin_q(qr, w['attn.indexer.wq_b.weight']).view(s, H, Dh)
        rope_(q[..., -rd:], self.freqs[:s])
        q = fp4_quant(q, 32)
        wts = F.linear(x, w['attn.indexer.weights_proj.weight']) * (Dh ** -0.5 * H ** -0.5)
        ik = st.index_k[:clen]
        out = torch.empty((s, topk), dtype=torch.int32, device=x.device)
        cand = [] if (self.cand_source and nb > a['candidate_topk_blocks']) else None
        bsz = a['candidate_block_size']
        for c0 in range(0, s, q_chunk):
            c1 = min(s, c0 + q_chunk)
            sc = torch.einsum('shd,td->sht', q[c0:c1], ik)
            sc = (sc.relu_() * wts[c0:c1].unsqueeze(-1)).sum(dim=1)
            lens = (torch.arange(c0 + 1, c1 + 1, device=x.device) // r).unsqueeze(-1)
            sc.masked_fill_(torch.arange(clen, device=x.device) >= lens, float('-inf'))
            if cand is not None:
                cand.append(select_candidate_blocks(sc, lens, a['candidate_topk_blocks'], bsz))
            elif self.uses_cand and st.cand is not None:
                keep = st.cand[c0:c1].repeat_interleave(bsz, dim=-1)[:, :clen]
                sc = sc.masked_fill(~keep, float('-inf'))
            idx = sc.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            out[c0:c1] = torch.where(idx < lens, idx + s, -1).int()
        if self.cand_source:
            st.cand = torch.cat(cand) if cand is not None else None
        return out

    def attention(self, x, st):
        """x [s, D] (attn_norm output of one sequence) -> [s, D]."""
        a, w = self.a, self.w
        s = x.shape[0]
        rd, H, Dh = a['rope_head_dim'], a['n_heads'], a['head_dim']
        f = self.freqs[:s]
        qr = rms_norm(lin_q(x, w['attn.wq_a.weight']), w['attn.q_norm.weight'])
        q = lin_q(qr, w['attn.wq_b.weight']).view(s, H, Dh)
        rope_(q[..., -rd:], f)
        kv = rms_norm(lin_q(x, w['attn.wkv.weight']), w['attn.kv_norm.weight'])
        rope_(kv[..., -rd:], f)
        kv = act_quant(kv, 32)
        idx = window_idxs(s, a['window_size'], x.device).int()
        if self.ratio:
            r = self.ratio
            latent = None
            if self.kv_source:
                latent = self.compressor(x)
            if self.index_source:
                st.topk_idxs = (torch.empty((s, 0), dtype=torch.int32, device=x.device) if s // r == 0
                                else self.indexer(x, qr, latent, st))
            if latent is not None:
                rope_(latent[..., -rd:], self.freqs[:s - s % r:r])
                st.compress_kv = fp4_quant(latent, 16, e4m3_scale=True)
            kv = torch.cat([kv, st.compress_kv[:s // r]])
            idx = torch.cat([idx, st.topk_idxs], dim=-1)
        o = sparse_attn(q, kv, w['attn.attn_sink'].float(), idx, Dh ** -0.5)
        rope_(o[..., -rd:], f, inverse=True)
        G = a['o_groups']
        o = act_quant(o.view(s, G, -1), 32)
        o = torch.einsum('sgd,grd->sgr', o, w['attn.wo_a.weight'].view(G, a['o_lora_rank'], -1))
        return lin_q(o.flatten(1), w['attn.wo_b.weight'])

    # -- MoE ------------------------------------------------------------------
    def moe(self, x, stats=None, routes_out=None):
        a, w = self.a, self.w
        N, D = x.shape
        E, K = a['n_routed_experts'], a['n_activated_experts']
        lim = a['swiglu_limit']
        idx_l, wt_l = [], []
        for c0 in range(0, N, 65536):
            sc = F.softplus(F.linear(x[c0:c0 + 65536].float(), w['ffn.gate.weight'].float())).sqrt()
            ii = (sc + w['ffn.gate.bias']).topk(K, dim=-1)[1]
            ww = sc.gather(1, ii)
            ww = ww / (ww.sum(-1, keepdim=True) + 1e-20) * a['route_scale']
            idx_l.append(ii)
            wt_l.append(ww)
        idx, wt = torch.cat(idx_l), torch.cat(wt_l)
        if routes_out is not None:
            routes_out.append((idx.to(torch.int16).cpu(), wt.float().cpu()))
        y = torch.zeros((N, D), dtype=torch.float32, device=x.device)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E).tolist()
        tok_all = order // K
        wt_all = wt.reshape(-1)[order]
        t = stats.t if stats is not None else None
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            tok = tok_all[start:start + n]
            we = wt_all[start:start + n]
            start += n
            w1, w3, w2 = w['experts'](e)
            xe = x[tok]
            xq = act_quant(xe)
            g = F.linear(xq, w1).float()
            u = F.linear(xq, w3).float()
            if lim > 0:
                u = u.clamp(-lim, lim)
                g = g.clamp(max=lim)
            act = F.silu(g) * u
            if t is not None:
                w2d = we.double() ** 2
                t['count'][e] += n
                t['w1'][e] += we.double().sum()
                t['w2'][e] += w2d.sum()
                xf = xe.float().pow(2)
                t['x2'][e] += xf.sum(0).double()
                t['x2w'][e] += (w2d.float()[None] @ xf)[0].double()
                af = act.pow(2)
                t['a2'][e] += af.sum(0).double()
                t['a2w'][e] += (w2d.float()[None] @ af)[0].double()
                del xf, af
            y.index_add_(0, tok, F.linear(act_quant((we[:, None] * act).to(BF)), w2).float())
            del w1, w3, w2, xe, xq, g, u, act
        for c0 in range(0, N, 65536):
            xc = x[c0:c0 + 65536]
            xq = act_quant(xc)
            g = F.linear(xq, w['ffn.shared_experts.w1.weight']).float()
            u = F.linear(xq, w['ffn.shared_experts.w3.weight']).float()
            if lim > 0:
                u = u.clamp(-lim, lim)
                g = g.clamp(max=lim)
            act = F.silu(g) * u
            if t is not None:
                t['shared_a'] += act.pow(2).sum(0).double()
                t['ffn_in'] += xc.float().pow(2).sum(0).double()
            y[c0:c0 + 65536] += F.linear(act_quant(act.to(BF)), w['ffn.shared_experts.w2.weight']).float()
        if t is not None:
            t['tokens'] += N
        return y.to(BF)

    # -- block ----------------------------------------------------------------
    def forward(self, h, pre_mix, seqs, states, stats=None, engram_rows=None, chunk=65536, routes_out=None):
        """h [N, hc, D] bf16 (updated in place), pre_mix [N, hc] fp32, seqs [(start, len)].
        Returns the next pre_mix."""
        a, w = self.a, self.w
        N = h.shape[0]
        hcp = dict(hc=a['hc_mult'], iters=a['hc_sinkhorn_iters'], eps=a['hc_eps'], norm_eps=a['norm_eps'])
        if self.engram is not None:
            for c0 in range(0, N, chunk):
                h[c0:c0 + chunk] = self.engram(h[c0:c0 + chunk], engram_rows(c0, min(N, c0 + chunk)))
        # attention sub-block
        att_pre = torch.empty_like(pre_mix)
        att_post = torch.empty_like(pre_mix)
        att_comb = torch.empty((N, a['hc_mult'], a['hc_mult']), dtype=torch.float32, device=h.device)
        xin = torch.empty((N, h.shape[2]), dtype=BF, device=h.device)
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            p_, q_, c_ = hc_mixes(h[c0:c1], w['hc_attn_fn'], w['hc_attn_scale'], w['hc_attn_base'], **hcp)
            att_pre[c0:c1], att_post[c0:c1], att_comb[c0:c1] = p_, q_, c_
            xin[c0:c1] = rms_norm(hc_pre(h[c0:c1], pre_mix[c0:c1]), w['attn_norm.weight'])
        att = torch.empty_like(xin)
        for i, (st, n) in enumerate(seqs):
            att[st:st + n] = self.attention(xin[st:st + n], states[i])
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            h[c0:c1] = hc_post(att[c0:c1], h[c0:c1], att_post[c0:c1], att_comb[c0:c1])
        del att, att_post, att_comb
        # FFN sub-block
        ffn_pre = torch.empty_like(pre_mix)
        ffn_post = torch.empty_like(pre_mix)
        ffn_comb = torch.empty((N, a['hc_mult'], a['hc_mult']), dtype=torch.float32, device=h.device)
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            p_, q_, c_ = hc_mixes(h[c0:c1], w['hc_ffn_fn'], w['hc_ffn_scale'], w['hc_ffn_base'], **hcp)
            ffn_pre[c0:c1], ffn_post[c0:c1], ffn_comb[c0:c1] = p_, q_, c_
            xin[c0:c1] = rms_norm(hc_pre(h[c0:c1], att_pre[c0:c1]), w['ffn_norm.weight'])
        y = self.moe(xin, stats, routes_out)
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            h[c0:c1] = hc_post(y[c0:c1], h[c0:c1], ffn_post[c0:c1], ffn_comb[c0:c1])
        return ffn_pre


class Engram:
    """Layer Engram: table rows come from the host (memmap gather), the rest on the device."""

    def __init__(self, a, w, table, scale):
        self.a, self.w = a, w
        self.table, self.scale = table, scale            # np.memmap [rows, 256] u8, [rows, 8] u8 (e8m0)

    def rows(self, ids):
        """ids [n, cols] int64 (CPU) -> BF16 [n, cols*256] on the device."""
        flat = ids.reshape(-1).numpy()
        order = np.argsort(flat, kind='stable')
        v = np.empty((flat.size, self.table.shape[1]), dtype=np.uint8)
        s = np.empty((flat.size, self.scale.shape[1]), dtype=np.uint8)
        v[order] = self.table[flat[order]]
        s[order] = self.scale[flat[order]]
        dev = self.w['engram.wkv.weight'].device
        vt = torch.from_numpy(v).to(dev).view(torch.float8_e4m3fn).float()
        st = torch.from_numpy(s).to(dev).view(torch.float8_e8m0fnu).float()
        n, c = ids.shape
        x = vt.view(n * c, -1, 32) * st.unsqueeze(-1)
        return x.view(n, -1).to(BF)

    def __call__(self, h, emb):
        a, w = self.a, self.w
        hc, D = a['hc_mult'], a['dim']
        kv = lin_q(emb, w['engram.wkv.weight'])
        key, value = kv.split([hc * D, D], dim=-1)
        key = key.float().unflatten(-1, (hc, D))
        weight = w['engram.q_weight'].float() * w['engram.k_weight'].float()
        hf = h.float()
        eps = a['norm_eps']
        rstd = torch.rsqrt(hf.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (hf * weight * key).sum(-1) * rstd * D ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


def rope_tables(a, max_len, device):
    """[ratio 0 (base theta, no YaRN), ratio > 0 (compress theta, YaRN)]."""
    rd = a['rope_head_dim']
    return [freqs_cis(rd, max_len, 0, a['rope_theta'], a['rope_factor'], a['beta_fast'], a['beta_slow'], device),
            freqs_cis(rd, max_len, a['original_seq_len'], a['compress_rope_theta'], a['rope_factor'],
                      a['beta_fast'], a['beta_slow'], device)]
