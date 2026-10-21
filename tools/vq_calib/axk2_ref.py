"""Layer-streamed A.X-K2 reference forward (BF16) with calibration-statistics hooks.

Arithmetic follows transformers `models/axk2/modeling_axk2.py` (checked layer
by layer against `AXK2DecoderLayer` in `selftest`): gated RMSNorm, MLA with the
fused q/gate projection (`q_b_proj` = HF `q_gate_proj`, input
`[q_a_layernorm(q_a), q_a]`), interleaved RoPE on the MLA rope slice and
half-split RoPE in the DSA indexer, YaRN softmax scale, top-2048 sparse
selection, sigmoid output gate, group-limited sigmoid router with the
correction bias (masked groups -inf, as in Megatron/vLLM/llama.cpp; NOT
transformers' 0.0, see route()), BF16
SwiGLU experts, shared expert.

Deviations (all inside BF16 rounding): routed-expert outputs are accumulated in
FP32 and rounded once (transformers index_adds in BF16); sequences longer than
2,048 tokens run the indexer and attention in query chunks (same selection).

Weights come from a provider with `layer(L) -> dict` (dequantized BF16 on the
device) and `experts(L) -> callable(e) -> (gate_up [4096, 7168], down [7168, 2048])`,
so the same forward serves the FP8 teacher and a VQ/Q4_K student.
"""
import math

import torch
import torch.nn.functional as F

from axk2_source import Source, dequant

BF = torch.bfloat16
EPS = 1e-6
NOPE, ROPE, VDIM, QK = 128, 64, 128, 192
HEADS, Q_LORA, KV_LORA = 64, 1536, 512
IDX_HEADS, IDX_DIM, IDX_TOPK = 64, 128, 2048


# ---------------------------------------------------------------- weights

def layer_names(cfg, L):
    p = f'model.layers.{L}.'
    names = [p + 'input_layernorm.norm.weight', p + 'input_layernorm.W_down.weight', p + 'input_layernorm.W_up.weight']
    a = p + 'self_attn.'
    names += [a + n for n in ('q_a_proj.weight', 'q_a_layernorm.weight', 'q_b_proj.weight', 'kv_a_proj_with_mqa.weight',
                              'kv_a_layernorm.weight', 'kv_b_proj.weight', 'o_proj.weight', 'indexer.wq_b.weight',
                              'indexer.wk.weight', 'indexer.k_norm.weight', 'indexer.k_norm.bias',
                              'indexer.weights_proj.weight')]
    if L < cfg['first_k_dense_replace']:
        names += [p + 'post_attention_layernorm.weight'] + [p + f'mlp.{k}_proj.weight' for k in ('gate', 'up', 'down')]
    else:
        names += [p + 'post_attention_layernorm.norm.weight', p + 'post_attention_layernorm.W_down.weight',
                  p + 'post_attention_layernorm.W_up.weight', p + 'mlp.gate.weight', p + 'mlp.gate.e_score_correction_bias']
        names += [p + f'mlp.shared_experts.{k}_proj.weight' for k in ('gate', 'up', 'down')]
    return names


class TeacherWeights:
    """FP8 source: non-expert weights dequantized to BF16 (router/bias/norms as stored, upcast
    where the reference upcasts); routed experts kept FP8 on the device, dequantized per expert."""
    def __init__(self, root='/models/A.X-K2', device='cuda'):
        self.src = Source(root)
        self.cfg = self.src.config
        self.device = device

    def host_layer(self, L, threads=16):
        names = layer_names(self.cfg, L)
        stored = [s for n in names for s in self.src.weight_names(n)]
        if L >= self.cfg['first_k_dense_replace']:
            stored += [s for n in self.src.expert_names(L) for s in self.src.weight_names(n)]
        return self.src.read(stored, threads=threads)

    def layer(self, L, host=None):
        host = host if host is not None else self.host_layer(L)
        dev = self.device
        out = {}
        p = f'model.layers.{L}.'
        for n in layer_names(self.cfg, L):
            w = host[n].to(dev, non_blocking=True)
            key = n[len(p):]
            s = n + '_scale_inv'
            if s in host:
                out[key] = dequant(w, host[s].to(dev, non_blocking=True), BF)
            elif w.dtype == torch.float32:
                out[key] = w                      # e_score_correction_bias
            else:
                out[key] = w.to(BF)
        if L >= self.cfg['first_k_dense_replace']:
            fp8 = {}
            for n in self.src.expert_names(L):
                fp8[n] = (host[n].to(dev, non_blocking=True), host[n + '_scale_inv'].to(dev, non_blocking=True))

            def expert(e, fp8=fp8):
                q = f'model.layers.{L}.mlp.experts.{e}.'
                g = dequant(*fp8[q + 'gate_proj.weight'], BF)
                u = dequant(*fp8[q + 'up_proj.weight'], BF)
                d = dequant(*fp8[q + 'down_proj.weight'], BF)
                return torch.cat([g, u], 0), d
            out['experts'] = expert
        return out


# ---------------------------------------------------------------- arithmetic

def rms_norm(x, w, eps=EPS):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def gated_norm(x, w, prefix, eps=EPS):
    y = rms_norm(x, w[prefix + '.norm.weight'], eps)
    g = F.linear(F.silu(F.linear(y, w[prefix + '.W_down.weight'])), w[prefix + '.W_up.weight'])
    return (y * torch.sigmoid(g.float())).to(y.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Rope:
    """cos/sin exactly as transformers AXK2RotaryEmbedding (YaRN), cast to BF16."""
    def __init__(self, root='/models/A.X-K2', device='cuda'):
        from transformers import AutoConfig
        from transformers.models.axk2.modeling_axk2 import AXK2RotaryEmbedding
        self.hf_cfg = AutoConfig.from_pretrained(root)
        self.emb = AXK2RotaryEmbedding(self.hf_cfg).to(device)
        self.device = device
        from transformers.models.axk2.modeling_axk2 import yarn_apply_mscale
        self.scaling = yarn_apply_mscale(self.hf_cfg.rope_parameters, QK ** -0.5)
        self._cache = {}

    def __call__(self, T):
        if T not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            pos = torch.arange(T, device=self.device)[None]
            dummy = torch.empty(1, dtype=BF, device=self.device)
            cos, sin = self.emb(dummy, pos)
            self._cache[T] = (cos[0], sin[0])          # [T, 64] BF16
        return self._cache[T]


def rope_interleave(q, k, cos, sin):
    """q [B,H,T,64], k [B,1,T,64]; cos/sin [T,64] -> de-interleaved rotated halves (transformers)."""
    c = cos[..., : cos.shape[-1] // 2]
    s = sin[..., : sin.shape[-1] // 2]
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    return (torch.cat([q1 * c - q2 * s, q2 * c + q1 * s], dim=-1),
            torch.cat([k1 * c - k2 * s, k2 * c + k1 * s], dim=-1))


class Stats:
    """FP64 sums on the device. All updates take only valid (non-padding) rows."""
    def __init__(self, cfg, L, device, moe):
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device=device)
        H, E, Fm = cfg['hidden_size'], cfg['n_routed_experts'], cfg['moe_intermediate_size']
        self.t = {'tokens': z(1), 'attn_in': z(H), 'qb_in': z(2 * Q_LORA), 'kvb_in': z(KV_LORA),
                  'o_in': z(HEADS * VDIM), 'ffn_in': z(H)}
        if moe:
            self.t.update({'shared_a': z(Fm), 'count': torch.zeros(E, dtype=torch.int64, device=device),
                           'w1': z(E), 'w2': z(E), 'x2': z(E, H), 'x2w': z(E, H), 'a2': z(E, Fm),
                           'a2w': z(E, Fm), 'bg': z(E, Fm), 'bu': z(E, Fm)})
        else:
            self.t['dense_a'] = z(cfg['intermediate_size'])

    def sq(self, key, x):
        self.t[key] += x.float().pow(2).sum(0).double()


def attention_short(w, x, cos, sin, scaling, stats=None, lengths=None):
    """x [B, T, H] right-padded, T <= 2048: every causal key is selected (top-k covers all)."""
    B, T, _ = x.shape
    a = 'self_attn.'
    qc = F.linear(x, w[a + 'q_a_proj.weight'])
    qr = rms_norm(qc, w[a + 'q_a_layernorm.weight'])
    qin = torch.cat([qr, qc], dim=-1)
    qg = F.linear(qin, w[a + 'q_b_proj.weight']).view(B, T, HEADS, QK + VDIM)
    q, gate = torch.split(qg, [QK, VDIM], dim=-1)
    q = q.transpose(1, 2)
    gate = gate.reshape(B, T, HEADS * VDIM)
    q_pass, q_rot = torch.split(q, [NOPE, ROPE], dim=-1)
    ckv = F.linear(x, w[a + 'kv_a_proj_with_mqa.weight'])
    kv_pass, k_rot = torch.split(ckv, [KV_LORA, ROPE], dim=-1)
    k_pass = rms_norm(kv_pass, w[a + 'kv_a_layernorm.weight'])
    k_rot = k_rot.reshape(B, 1, T, ROPE)
    q_rot, k_rot = rope_interleave(q_rot, k_rot, cos, sin)
    query = torch.cat((q_pass, q_rot), dim=-1)
    kv = F.linear(k_pass, w[a + 'kv_b_proj.weight']).view(B, T, HEADS, NOPE + VDIM).transpose(1, 2)
    k_nope, v = torch.split(kv, [NOPE, VDIM], dim=-1)
    key = torch.cat([k_nope, k_rot.expand(-1, HEADS, -1, -1)], dim=-1)
    o = F.scaled_dot_product_attention(query, key, v, is_causal=True, scale=scaling)
    o = o.transpose(1, 2).reshape(B, T, HEADS * VDIM)
    o = (o * torch.sigmoid(gate.float())).to(o.dtype)
    if stats is not None:
        valid = valid_mask(lengths, T, x.device)
        stats.sq('attn_in', x[valid])
        stats.sq('qb_in', qin[valid])
        stats.sq('kvb_in', k_pass[valid])
        stats.sq('o_in', o[valid])
    return F.linear(o, w[a + 'o_proj.weight'])


def valid_mask(lengths, T, device):
    return torch.arange(T, device=device)[None, :] < torch.as_tensor(lengths, device=device)[:, None]


def attention_long(w, x, cos, sin, scaling, stats=None, q_chunk=512, idx_chunk=256):
    """x [1, T, H], any T: DSA indexer top-2048 selection per query, chunked over queries."""
    _, T, _ = x.shape
    a = 'self_attn.'
    x2 = x[0]
    qc = F.linear(x2, w[a + 'q_a_proj.weight'])
    qr = rms_norm(qc, w[a + 'q_a_layernorm.weight'])
    qin = torch.cat([qr, qc], dim=-1)
    qg = F.linear(qin, w[a + 'q_b_proj.weight']).view(T, HEADS, QK + VDIM)
    q, gate = torch.split(qg, [QK, VDIM], dim=-1)
    q = q.transpose(0, 1)[None]                                    # [1, H, T, 192]
    gate = gate.reshape(T, HEADS * VDIM)
    q_pass, q_rot = torch.split(q, [NOPE, ROPE], dim=-1)
    ckv = F.linear(x2, w[a + 'kv_a_proj_with_mqa.weight'])
    kv_pass, k_rot = torch.split(ckv, [KV_LORA, ROPE], dim=-1)
    k_pass = rms_norm(kv_pass, w[a + 'kv_a_layernorm.weight'])
    k_rot = k_rot.reshape(1, 1, T, ROPE)
    q_rot, k_rot = rope_interleave(q_rot, k_rot, cos, sin)
    query = torch.cat((q_pass, q_rot), dim=-1)
    kv = F.linear(k_pass, w[a + 'kv_b_proj.weight']).view(T, HEADS, NOPE + VDIM).transpose(0, 1)[None]
    k_nope, v = torch.split(kv, [NOPE, VDIM], dim=-1)
    key = torch.cat([k_nope, k_rot.expand(-1, HEADS, -1, -1)], dim=-1)
    del kv, k_nope, qg

    # indexer (transformers AXK2Indexer, bf16 projections, fp32 scores)
    iq = F.linear(qr, w[a + 'indexer.wq_b.weight']).view(T, IDX_HEADS, IDX_DIM)
    iq_rot, iq_pass = torch.split(iq, [ROPE, IDX_DIM - ROPE], dim=-1)
    ik = F.layer_norm(F.linear(x2, w[a + 'indexer.wk.weight']), (IDX_DIM,), w[a + 'indexer.k_norm.weight'],
                      w[a + 'indexer.k_norm.bias'], 1e-6)
    ik_rot, ik_pass = torch.split(ik, [ROPE, IDX_DIM - ROPE], dim=-1)
    c, s = cos[:, None, :], sin[:, None, :]
    iq_rot = iq_rot * c + rotate_half(iq_rot) * s
    ik_rot = ik_rot * cos + rotate_half(ik_rot) * sin
    iq = torch.cat([iq_rot, iq_pass], dim=-1).float()              # [T, 64, 128]
    ik = torch.cat([ik_rot, ik_pass], dim=-1).float()              # [T, 128]
    hw = F.linear(x2.float(), w[a + 'indexer.weights_proj.weight'].float()) * (IDX_HEADS ** -0.5)   # [T, 64]

    out = torch.empty((T, HEADS * VDIM), dtype=x.dtype, device=x.device)
    for s0 in range(0, T, q_chunk):
        s1 = min(T, s0 + q_chunk)
        n = s1 - s0
        allowed = torch.zeros((n, s1), dtype=torch.bool, device=x.device)
        for i0 in range(s0, s1, idx_chunk):
            i1 = min(s1, i0 + idx_chunk)
            sc = torch.einsum('shd,td->sht', iq[i0:i1], ik[:s1]) * (IDX_DIM ** -0.5)
            sc = torch.einsum('sh,sht->st', hw[i0:i1], F.relu(sc))
            causal = torch.arange(s1, device=x.device)[None, :] <= torch.arange(i0, i1, device=x.device)[:, None]
            sc = sc.masked_fill(~causal, float('-inf'))
            sel = sc.topk(min(IDX_TOPK, s1), dim=-1).indices
            rows = torch.zeros((i1 - i0, s1), dtype=torch.bool, device=x.device).scatter_(1, sel, True)
            allowed[i0 - s0:i1 - s0] = rows & causal
            del sc
        o = F.scaled_dot_product_attention(query[:, :, s0:s1], key[:, :, :s1], v[:, :, :s1],
                                           attn_mask=allowed[None, None], scale=scaling)
        out[s0:s1] = o[0].transpose(0, 1).reshape(n, HEADS * VDIM)
    o = (out * torch.sigmoid(gate.float())).to(out.dtype)
    if stats is not None:
        stats.sq('attn_in', x2)
        stats.sq('qb_in', qin)
        stats.sq('kvb_in', k_pass)
        stats.sq('o_in', o)
    return F.linear(o, w[a + 'o_proj.weight'])[None]


def mlp(x, wg, wu, wd, stats=None, key=None):
    a = F.silu(F.linear(x, wg)) * F.linear(x, wu)
    if stats is not None:
        stats.sq(key, a)
    return F.linear(a, wd)


def route(cfg, w, x):
    """x [N, H] BF16 -> (top-k ids [N, 8], weights [N, 8] F32 incl. routed_scaling_factor)."""
    logits = F.linear(x.float(), w['mlp.gate.weight'].float())
    scores = logits.sigmoid()
    sfc = scores + w['mlp.gate.e_score_correction_bias']
    E, G, TG = cfg['n_routed_experts'], cfg['n_group'], cfg['topk_group']
    gs = sfc.view(-1, G, E // G).topk(2, dim=-1)[0].sum(dim=-1)
    gi = torch.topk(gs, k=TG, dim=-1, sorted=False)[1]
    gm = torch.zeros_like(gs).scatter_(1, gi, 1)
    sm = gm.unsqueeze(-1).expand(-1, G, E // G).reshape(-1, E)
    # Masked groups get -inf, as in Megatron (training), vLLM grouped_topk and llama.cpp
    # build_moe_ffn. transformers' AXK2/DeepseekV3 router fills 0.0, which hands the top-k to
    # masked experts whenever the biased scores are negative: every token in layers 24, 41, 48
    # (all correction biases < -1.3) and some tokens in 34-37, 42, 49, 51, 52, 56, 57, 60.
    sfc = sfc.masked_fill(~sm.bool(), float('-inf'))
    idx = torch.topk(sfc, k=cfg['num_experts_per_tok'], dim=-1, sorted=False)[1]
    wt = scores.gather(1, idx)
    wt = wt / (wt.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, wt * cfg['routed_scaling_factor']


def silu_grad(g):
    s = torch.sigmoid(g)
    return s * (1 + g * (1 - s))


def moe(cfg, w, x, stats=None, chunk=262144, routes=None):
    """x [N, H] BF16 (valid rows only) -> [N, H] BF16. `routes` optional precomputed (ids, wts)."""
    N, H = x.shape
    out = torch.empty_like(x)
    all_idx, all_wt = [], []
    for c0 in range(0, N, chunk):
        xc = x[c0:c0 + chunk]
        if routes is None:
            idx, wt = route(cfg, w, xc)
        else:
            idx, wt = routes[0][c0:c0 + chunk], routes[1][c0:c0 + chunk]
        all_idx.append(idx)
        all_wt.append(wt)
        acc = torch.zeros((len(xc), H), dtype=torch.float32, device=x.device)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=cfg['n_routed_experts']).tolist()
        tok_all = order // cfg['num_experts_per_tok']
        wt_all = wt.reshape(-1)[order]
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            tok = tok_all[start:start + n]
            we = wt_all[start:start + n]
            start += n
            gu, dn = w['experts'](e)
            xe = x[c0:c0 + chunk][tok]
            g, u = F.linear(xe, gu).chunk(2, dim=-1)
            a = F.silu(g) * u
            y = F.linear(a, dn)
            acc.index_add_(0, tok, y.float() * we[:, None])
            if stats is not None:
                t = stats.t
                w2 = (we.double() ** 2)
                t['count'][e] += n
                t['w1'][e] += we.double().sum()
                t['w2'][e] += w2.sum()
                xf = xe.float().pow(2)
                t['x2'][e] += xf.sum(0).double()
                t['x2w'][e] += (w2.float()[None] @ xf)[0].double()
                af = a.float().pow(2)
                t['a2'][e] += af.sum(0).double()
                t['a2w'][e] += (w2.float()[None] @ af)[0].double()
                gf = g.float()
                t['bg'][e] += (w2.float()[None] @ (silu_grad(gf) * u.float()).pow(2))[0].double()
                t['bu'][e] += (w2.float()[None] @ F.silu(gf).pow(2))[0].double()
        sh = mlp(xc, w['mlp.shared_experts.gate_proj.weight'], w['mlp.shared_experts.up_proj.weight'],
                 w['mlp.shared_experts.down_proj.weight'], stats, 'shared_a')
        out[c0:c0 + chunk] = acc.to(x.dtype) + sh
        del acc
    return out, (torch.cat(all_idx), torch.cat(all_wt))


class Layer:
    def __init__(self, cfg, L, weights, rope):
        self.cfg, self.L, self.w, self.rope = cfg, L, weights, rope
        self.dense = L < cfg['first_k_dense_replace']

    def attention(self, h, seqs, stats=None, max_tokens=65536):
        """h [N, H] flat residual stream; seqs = list of (start, length). Returns the attention
        sublayer output [N, H] (before the residual add)."""
        w = self.w
        out = torch.empty_like(h)
        short = sorted([s for s in seqs if s[1] <= IDX_TOPK], key=lambda s: s[1])
        long_ = [s for s in seqs if s[1] > IDX_TOPK]
        i = 0
        while i < len(short):
            T = short[i][1]
            j = i
            while j < len(short) and (j - i + 1) * short[j][1] <= max_tokens:
                j += 1
            j = max(j, i + 1)
            batch = short[i:j]
            T = batch[-1][1]
            idx = torch.cat([torch.arange(st, st + n, device=h.device) for st, n in batch])
            x = torch.zeros((len(batch), T, h.shape[1]), dtype=h.dtype, device=h.device)
            valid = valid_mask([n for _, n in batch], T, h.device)
            x[valid] = gated_norm(h[idx], w, 'input_layernorm')
            cos, sin = self.rope(T)
            y = attention_short(w, x, cos, sin, self.rope.scaling, stats, [n for _, n in batch])
            out[idx] = y[valid]
            i = j
        for st, n in long_:
            x = gated_norm(h[st:st + n], w, 'input_layernorm')[None]
            cos, sin = self.rope(n)
            out[st:st + n] = attention_long(w, x, cos, sin, self.rope.scaling, stats)[0]
        return out

    def ffn(self, x, stats=None, routes=None):
        w = self.w
        if stats is not None:
            stats.sq('ffn_in', x)
        if self.dense:
            return mlp(x, w['mlp.gate_proj.weight'], w['mlp.up_proj.weight'], w['mlp.down_proj.weight'],
                       stats, 'dense_a'), None
        return moe(self.cfg, w, x, stats, routes=routes)

    def post_norm(self, h):
        if self.dense:
            return rms_norm(h, self.w['post_attention_layernorm.weight'])
        return gated_norm(h, self.w, 'post_attention_layernorm')

    def forward(self, h, seqs, stats=None, keep_ffn_in=False, chunk=131072):
        """Full decoder layer over a flat [N, H] buffer, in place where possible."""
        a = self.attention(h, seqs, stats)
        h += a
        del a
        ffn_in = torch.empty_like(h) if keep_ffn_in else None
        routes = []
        for c0 in range(0, h.shape[0], chunk):
            x = self.post_norm(h[c0:c0 + chunk])
            if keep_ffn_in:
                ffn_in[c0:c0 + chunk] = x
            y, r = self.ffn(x, stats)
            if r is not None:
                routes.append(r)
            h[c0:c0 + chunk] += y
        if stats is not None:
            stats.t['tokens'] += h.shape[0]
        r = (torch.cat([x[0] for x in routes]), torch.cat([x[1] for x in routes])) if routes else None
        return h, ffn_in, r
