"""Inkling-2-Small reference forward (text), layer-streamed, for hVQ statistics and teacher-forced scoring.

The arithmetic follows transformers' `modeling_inkling2.py` at 109a7d19bf667168837f32929e00b791e195d14d
(eager attention, the Python short-convolution path, the eager experts loop), operation by operation and in
the same dtypes, but computed per sequence in query blocks so a 32K-token sequence never materialises a
[48, T, T] score tensor. `inkling2_selftest.py` checks it against transformers' decoder layers.

Source: thinkingmachines/Inkling-2-Small-MXFP8 at 086a047e8b7ff2ac902dd52355328ddb66b16914. Routed experts
are MXFP8: e4m3 times a ue8m0 scale per row and 32 columns, `w = fp8 * 2^(s - 127)`, exact in BF16.
`w13_weight` stores gate and up rows interleaved (row 2i is gate i, row 2i + 1 is up i); transformers
de-interleaves it into `gate_up_proj` = [gate; up] (conversion_mapping.py `Interleave(dim=1)`).

Layer l is a full-attention ("hybrid") layer when (l + 1) % 6 == 0, otherwise a sliding-window layer
(window 1024). Every layer is MoE: 384 experts, top 8, sigmoid scores, a selection-only bias, weights
renormalised and scaled by 8, softcapped SwiGLU. Block attention residuals: layer l reads the inputs of
the block-boundary layers before it (l % 6 == 0) plus its own input through learned queries.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading

import torch
import torch.nn.functional as F
from safetensors import safe_open

SRC = Path('/models/Inkling-2-Small-MXFP8')
SOURCE_REPO = 'thinkingmachines/Inkling-2-Small-MXFP8'
SOURCE_REV = '086a047e8b7ff2ac902dd52355328ddb66b16914'
TRANSFORMERS_REV = '109a7d19bf667168837f32929e00b791e195d14d'

H, FF, E, L, TOPK = 4608, 2304, 384, 48, 8
HEADS, KVH, HD, DREL = 48, 8, 128, 16
REP = HEADS // KVH
WINDOW, REL_EXTENT = 1024, 1024
BLOCK = 6
EPS = 1e-6
QK_CLIP, REL_CLIP = 22.627416997969522, 8.0
GATE_CAP, UP_CAP, LOGIT_CAP = 4.0, 25.0, 30.0
ROUTE_SCALE = 8.0
VOCAB, VOCAB_REAL = 201024, 200058
LOG_FLOOR, LOG_ALPHA = 128000, 0.1
GLOBAL = tuple(l for l in range(L) if (l + 1) % BLOCK == 0)
BF16_MIN = torch.finfo(torch.bfloat16).min


def is_global(l):
    return (l + 1) % BLOCK == 0


def is_boundary(l):
    return l % BLOCK == 0


# ----------------------------------------------------------------------------------------- source
class Source:
    """Pinned MXFP8 checkpoint: tensor names -> shard, threaded reads into (pinned) host memory."""

    def __init__(self, root=SRC, index=None):
        self.root = Path(root)
        index = Path(index) if index else self.root / 'model.safetensors.index.json'
        self.map = json.loads(index.read_text())['weight_map']
        self.config = json.loads((self.root / 'config.json').read_text())
        self._local = threading.local()

    def _handle(self, shard):
        cache = getattr(self._local, 'files', None)
        if cache is None:
            cache = self._local.files = {}
        if shard not in cache:
            cache[shard] = safe_open(str(self.root / shard), framework='pt', device='cpu')
        return cache[shard]

    def shape(self, name):
        return tuple(self._handle(self.map[name]).get_slice(name).get_shape())

    def read(self, names, threads=16, pin=False, row_chunks=8):
        """{name: CPU tensor}. Large tensors are read as row ranges in parallel."""
        out, jobs = {}, []
        for n in names:
            f = self._handle(self.map[n])
            sl = f.get_slice(n)
            shape = tuple(sl.get_shape())
            dtype = {'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16, 'F32': torch.float32,
                     'U8': torch.uint8}[sl.get_dtype()]
            t = torch.empty(shape, dtype=dtype, pin_memory=pin)
            out[n] = t
            rows = shape[0]
            step = max(1, -(-rows // row_chunks)) if t.numel() * t.element_size() > (64 << 20) else rows
            for r0 in range(0, rows, step):
                jobs.append((n, r0, min(rows, r0 + step)))

        def work(job):
            n, r0, r1 = job
            src = self._handle(self.map[n]).get_slice(n)[r0:r1]
            out[n][r0:r1].copy_(src)

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(work, jobs))
        return out

    def layer_names(self, l, experts=True):
        p = f'model.llm.layers.{l}.'
        names = [k for k in self.map if k.startswith(p) and (experts or '.mlp.experts.' not in k)]
        q = [f'model.llm.attn_res.mlp_queries.{l}']
        if l > 0:
            q.append(f'model.llm.attn_res.attn_queries.{l - 1}')
        return sorted(names + q)


def dequant_mx(w8, s8, dtype=torch.bfloat16):
    """MXFP8 [R, C] e4m3 with a ue8m0 scale per row and 32 columns -> dtype (exact for BF16)."""
    r, c = w8.shape[-2:]
    lead = w8.shape[:-2]
    w = w8.reshape(*lead, r, c // 32, 32).to(torch.float32)
    sc = torch.exp2(s8.to(torch.float32) - 127.0)
    return (w * sc[..., None]).reshape(*lead, r, c).to(dtype)


# ----------------------------------------------------------------------------------- primitives
def rms(x, w):
    """Inkling2RMSNorm: FP32 normalisation, weight product in FP32; caller casts."""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return w.float() * y


def unweighted_rms(x32):
    return x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + EPS)


def l2_clip(x, thr, scale=None):
    x32 = x.float()
    if scale is not None:
        x32 = x32 * scale.float()
    n = torch.linalg.vector_norm(x32, dim=-1, keepdim=True)
    return (x32 * (thr / (n + 1e-6)).clamp(max=1.0)).to(x.dtype)


def short_conv(x, w):
    """Inkling2ShortConvolution: causal depthwise conv (4 taps, weight [C, 1, 4]) in FP32, plus the input,
    divided by sqrt(5). x [T, C] BF16 -> BF16."""
    t, c = x.shape
    x32 = x.float()
    y = F.conv1d(x32.t().unsqueeze(0), w.float().reshape(c, 1, -1), padding=w.shape[-1] - 1, groups=c)[0, :, :t].t()
    return ((y + x32) / math.sqrt(w.shape[-1] + 1)).to(x.dtype)


def dyn_conv(x, din, static_w, proj_w):
    """Inkling2DynamicShortConvolution: taps = static [4, C] (tap = backward distance) + per-token
    weights projected from din, shared within each KV head. x [T, 1024] BF16, din [T, 4608] BF16."""
    t, c = x.shape
    k = static_w.shape[0]
    dyn = F.linear(din, proj_w).float().view(t, KVH, 1, k)
    w = static_w.float().t().reshape(KVH, c // KVH, k) + dyn
    x32 = x.float()
    ctx = F.pad(x32.t(), (k - 1, 0))
    win = ctx.unfold(-1, k, 1).flip(-1).transpose(0, 1).reshape(t, KVH, c // KVH, k)
    y = (win * w).sum(-1).reshape(t, c)
    return ((y + x32) / math.sqrt(2 * k + 1)).to(x.dtype)


def attn_res(blocks, partial, query, chunk=16384):
    """Inkling2BlockAttentionResidual: RMS-normalised stack of the block inputs and the partial state,
    attended by `query` [Q, H] (FP32) with scale H^-1/2. blocks: list of [T, H]; -> [T, Q, H] FP32.
    The same F.scaled_dot_product_attention call as transformers ([1, T, Q, H] queries over [1, T, n, H]),
    in token chunks."""
    t = partial.shape[0]
    q = query.float().reshape(-1, H)
    out = torch.empty((t, q.shape[0], H), dtype=torch.float32, device=partial.device)
    for c0 in range(0, t, chunk):
        c1 = min(t, c0 + chunk)
        st = torch.stack([b[c0:c1].float() for b in blocks] + [partial[c0:c1].float()], dim=-2)
        st = unweighted_rms(st)
        qs = q.expand(c1 - c0, -1, -1)
        out[c0:c1] = F.scaled_dot_product_attention(qs[None], st[None], st[None], scale=H ** -0.5)[0]
    return out


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


def swiglu(g, u):
    g, u = g.float(), u.float()
    return softcap(g, GATE_CAP) * torch.sigmoid(g) * softcap(u, UP_CAP)


# KV storage formats for the K/V-quantization study (the engine's NVFP4 is vLLM's encoder, as MiniMax's)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def kv_round(x, fmt):
    """Round K or V head vectors [..., 128] (BF16) as stored: 'bf16' (unchanged), 'fp8' (E4M3, per-vector
    F32 scale amax / 448), 'nvfp4' (E2M1 values, an E4M3 scale per 16 = max(amax / 6, 2^-9), global 1)."""
    if fmt == 'bf16':
        return x
    x32 = x.float()
    if fmt == 'fp8':
        amax = x32.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        s = amax / 448.0
        return ((x32 / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
    if fmt == 'nvfp4':
        g = x32.reshape(*x32.shape[:-1], -1, 16)
        amax = g.abs().amax(-1, keepdim=True)
        s = (amax / 6.0).clamp_min(2.0 ** -9).to(torch.float8_e4m3fn).float()
        v = (g / s).clamp(-6.0, 6.0)
        grid = E2M1.to(x.device)
        mag = v.abs()
        # round to nearest E2M1 magnitude, ties to even code
        d = (mag[..., None] - grid).abs()
        idx = d.argmin(-1)
        tie = (d.gather(-1, (idx + 1).clamp(max=7)[..., None])[..., 0] == d.gather(-1, idx[..., None])[..., 0]) & (idx < 7)
        idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
        q = grid[idx] * v.sign()
        return (q * s).reshape(x32.shape).to(x.dtype)
    raise ValueError(fmt)


def attention(q, k, v, rel, sinks, window, qblock=None):
    """Eager attention of one sequence. q [T, 48, 128], k/v [T, 8, 128], rel [T, 48, 1024] (relative
    profiles), sinks [T, 48], all BF16. Scores in BF16 (matmul then x scale, + bias, + mask), softmax with
    the sink logit in FP32, probabilities in BF16 -> [T, 48, 128] BF16."""
    t = q.shape[0]
    qblock = qblock or (t if t <= 2048 else (1024 if window else max(256, min(1024, (1 << 27) // (HEADS * t)))))
    out = torch.empty_like(q)
    # repeat_kv: query head h reads KV head h // 6, as transformers' eager path
    kk = k.transpose(0, 1).repeat_interleave(REP, dim=0)               # [48, T, 128]
    vv = v.transpose(0, 1).repeat_interleave(REP, dim=0)
    qq = q.transpose(0, 1)                                             # [48, T, 128]
    scale = HD ** -0.5
    for i0 in range(0, t, qblock):
        i1 = min(t, i0 + qblock)
        j0 = 0 if window is None else max(0, i0 - window + 1)
        bq, bk = i1 - i0, i1 - j0
        if window is None and qblock >= t:
            j0, bk = 0, t                                              # one block: the full [T, T] as eager
        s = torch.matmul(qq[:, i0:i1], kk[:, j0:j0 + bk].transpose(1, 2)) * scale    # [48, bq, bk] BF16
        dist = torch.arange(i0, i1, device=q.device)[:, None] - torch.arange(j0, j0 + bk, device=q.device)[None, :]
        prof = rel[i0:i1].transpose(0, 1)                                            # [48, bq, 1024]
        bias = prof.gather(-1, dist.clamp(0, REL_EXTENT - 1).expand(HEADS, -1, -1))
        bias = bias.masked_fill((dist < 0) | (dist >= REL_EXTENT), 0.0)
        s = s + bias
        allowed = dist >= 0
        if window is not None:
            allowed &= dist < window
        s = s + torch.where(allowed, 0.0, BF16_MIN).to(s.dtype)                     # eager adds the mask
        logits = torch.cat([s, sinks[i0:i1].t()[:, :, None]], -1).float()
        p = logits.softmax(-1)[..., :-1].to(q.dtype)
        out[i0:i1] = torch.matmul(p, vv[:, j0:j0 + bk]).transpose(0, 1)              # [bq, 48, 128]
    return out


# ------------------------------------------------------------------------------------- the layer
class LayerWeights:
    """One layer's tensors on a device. Experts stay MXFP8 (dequantised per expert on demand) unless a
    student replaces them (`student(e) -> (gate_up [2*FF, H] in [gate; up] order, down [H, FF]) BF16`)."""

    def __init__(self, l, host, device, student=None):
        self.l, self.dev = l, device
        p = f'model.llm.layers.{l}.'
        g = lambda n: host[p + n].to(device, non_blocking=True)
        self.glob = is_global(l)
        a = {}
        for n in ('wq_du', 'wk_dv', 'wv_dv', 'wr_du', 'wo_ud', 'gate_attn', 'attn_sink_proj'):
            a[n] = g(f'attn.{n}.weight')
        a['q_scale'] = g('attn.q_l2_clip.scale')
        a['rel_scale'] = g('attn.rel_l2_clip.scale')
        a['rel_proj'] = g('attn.rel_logits_proj.proj')
        if self.glob:
            for kv in ('k', 'v'):
                a[f'{kv}_static'] = g(f'attn.{kv}_sconv.static_w')
                a[f'{kv}_wproj'] = g(f'attn.{kv}_sconv.weight_proj')
                a[f'{kv}_norm'] = g(f'{kv}_input_norm.weight')
            a['q_norm'] = g('q_input_norm.weight')
        else:
            a['k_conv'] = g('attn.k_sconv.weight')
            a['v_conv'] = g('attn.v_sconv.weight')
            a['norm'] = g('attn_norm.weight')
        a['attn_conv'] = g('attn_sconv.weight')
        a['mlp_conv'] = g('mlp_sconv.weight')
        a['mlp_norm'] = g('mlp_norm.weight')
        a['router'] = g('mlp.gate.weight').float()
        a['router_bias'] = g('mlp.gate.bias').float()
        a['mlp_query'] = host[f'model.llm.attn_res.mlp_queries.{l}'].to(device).float()
        if l > 0:
            a['attn_query'] = host[f'model.llm.attn_res.attn_queries.{l - 1}'].to(device).float()
        self.a = a
        self.student = student
        if student is None:
            self.w13 = g('mlp.experts.w13_weight')
            self.s13 = g('mlp.experts.w13_weight_scale')
            self.w2 = g('mlp.experts.w2_weight')
            self.s2 = g('mlp.experts.w2_weight_scale')

    def expert(self, e):
        """-> (gate_up [2*FF, H] = [gate; up], down [H, FF]) BF16 on the device."""
        if self.student is not None:
            return self.student(e)
        w13 = dequant_mx(self.w13[e], self.s13[e])
        gu = torch.cat([w13[0::2], w13[1::2]])
        return gu, dequant_mx(self.w2[e], self.s2[e])


def attn_part(lw, x_in, blocks, positions, kv_fmt='bf16', hooks=None):
    """Attention half of a layer for one sequence. x_in [T, H] BF16 (layer input), blocks: block-state
    list before this layer. Returns (h_mlp_in FP32 for mlp_input, residual FP32, blocks for mlp_input)."""
    a, l = lw.a, lw.l
    t = x_in.shape[0]
    if l > 0:
        res_in = attn_res(blocks, x_in, a['attn_query'])                    # [T, Q, H] FP32
    if lw.glob:
        xq = rms(res_in[:, 0], a['q_norm']).to(torch.bfloat16)
        xk = rms(res_in[:, 1], a['k_norm']).to(torch.bfloat16)
        xv = rms(res_in[:, 2], a['v_norm']).to(torch.bfloat16)
    else:
        src = res_in[:, 0] if l > 0 else x_in
        xq = xk = xv = rms(src, a['norm']).to(torch.bfloat16)
    if hooks is not None:
        hooks.attn_inputs(l, xq, xk, xv)
    q = F.linear(xq, a['wq_du'])
    kr = F.linear(xk, a['wk_dv'])
    vr = F.linear(xv, a['wv_dv'])
    if lw.glob:
        k = dyn_conv(kr, xk, a['k_static'], a['k_wproj'])
        v = dyn_conv(vr, xv, a['v_static'], a['v_wproj'])
    else:
        k = short_conv(kr, a['k_conv'])
        v = short_conv(vr, a['v_conv'])
    r = F.linear(xq, a['wr_du'])
    gate = F.linear(xq, a['gate_attn'])
    sinks = F.linear(xq, a['attn_sink_proj'])
    q = l2_clip(q.view(t, HEADS, HD), QK_CLIP, a['q_scale'])
    k = l2_clip(k.view(t, KVH, HD), QK_CLIP)
    v = unweighted_rms(v.view(t, KVH, HD).float()).to(torch.bfloat16)
    r = l2_clip(r.view(t, HEADS, DREL), REL_CLIP, a['rel_scale'])
    if lw.glob:
        tau = 1.0 + LOG_ALPHA * torch.log(((positions + 1).float() / LOG_FLOOR).clamp(min=1.0))
        q = (q.float() * tau[:, None, None]).to(q.dtype)
        r = (r.float() * tau[:, None, None]).to(r.dtype)
    if kv_fmt != 'bf16' and (lw.glob or kv_fmt.endswith('-all')):
        fmt = kv_fmt.removesuffix('-all')
        k, v = kv_round(k, fmt), kv_round(v, fmt)
    rel = torch.matmul(r, a['rel_proj'])                                    # [T, 48, 1024] BF16
    o = attention(q, k, v, rel, sinks, None if lw.glob else WINDOW)
    o = o.reshape(t, HEADS * HD)
    o = (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    if hooks is not None:
        hooks.o_input(l, o)
    o = F.linear(o, a['wo_ud'])
    o = short_conv(o, a['attn_conv'])
    resid = o.float()
    if is_boundary(l):
        blocks = blocks + [x_in]
    else:
        resid = x_in.float() + resid
    return resid, blocks


def mlp_in(lw, resid, blocks):
    """mlp_input attention residual and post_attention_layernorm -> expert input x [T, H] BF16."""
    h = attn_res(blocks, resid, lw.a['mlp_query'])[:, 0]
    return rms(h, lw.a['mlp_norm']).to(torch.bfloat16)


def route(lw, x):
    logits = F.linear(x.float(), lw.a['router'])
    scores = torch.sigmoid(logits)
    idx = torch.topk(scores + lw.a['router_bias'], TOPK, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
    return idx, w


def moe(lw, x, hooks=None, routes=None):
    """Routed experts of all tokens x [N, H] BF16 -> [N, H] BF16 (FP32 sum in ascending expert order).
    hooks.expert(e, tokens, x_e, act_e, weights_e) sees every expert's inputs."""
    idx, w = routes if routes is not None else route(lw, x)
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E).tolist()
    tok = order // TOPK
    wflat = w.reshape(-1)[order]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    start = 0
    for e in range(E):
        n = counts[e]
        if n == 0:
            continue
        t = tok[start:start + n]
        we = wflat[start:start + n]
        start += n
        xe = x[t]
        gu_w, dn_w = lw.expert(e)
        gu = F.linear(xe, gu_w)
        act = swiglu(gu[:, :FF], gu[:, FF:]).to(torch.bfloat16)
        y = F.linear(act, dn_w)
        out.index_add_(0, t, y.float() * we[:, None])
        if hooks is not None:
            hooks.expert(lw.l, e, t, xe, act, we)
        del gu_w, dn_w, gu, act, y
    return out.to(torch.bfloat16), (idx, w)


def finish_layer(lw, resid, moe_out):
    m = short_conv(moe_out, lw.a['mlp_conv'])
    return (resid + m.float()).to(torch.bfloat16)


# ------------------------------------------------------------------------------- whole model
class Globals:
    """Embedding (host), embed norm, final norm, output query and unembedding (device)."""

    def __init__(self, src, device, head=True):
        names = ['model.llm.embed_norm.weight', 'model.llm.norm.weight', 'model.llm.attn_res.output_query']
        if head:
            names.append('model.llm.unembed.weight')
        t = src.read(names)
        self.embed_norm = t['model.llm.embed_norm.weight'].to(device)
        self.norm = t['model.llm.norm.weight'].to(device)
        self.out_query = t['model.llm.attn_res.output_query'].to(device).float()
        self.unembed = t['model.llm.unembed.weight'].to(device) if head else None
        self.embed = src.read(['model.llm.embed.weight'])['model.llm.embed.weight']    # host BF16
        self.dev = device

    def embed_ids(self, ids):
        rows = self.embed[ids.cpu().long()].to(self.dev)
        return rms(rows, self.embed_norm).to(torch.bfloat16)

    def final_hidden(self, blocks, hidden):
        h = attn_res(blocks, hidden, self.out_query)[:, 0]
        return rms(h, self.norm).to(torch.bfloat16)

    def logits(self, final_h):
        z = F.linear(final_h, self.unembed).float()
        return softcap(z, LOGIT_CAP)[..., :VOCAB_REAL]


def run_sequence(src, glob, layers, ids, device, kv_fmt='bf16', keep=None):
    """Whole-model forward of one sequence with per-layer LayerWeights from `layers(l)`.
    -> final normalised hidden [T, H] BF16 (and the per-layer inputs listed in `keep`)."""
    pos = torch.arange(ids.numel(), device=device)
    hidden = glob.embed_ids(ids)
    blocks, kept = [], {}
    for l in range(L):
        if keep is not None and l in keep:
            kept[l] = (hidden.clone(), [b.clone() for b in blocks])
        lw = layers(l)
        resid, mblocks = attn_part(lw, hidden, blocks, pos, kv_fmt)
        x = mlp_in(lw, resid, mblocks)
        y, _ = moe(lw, x)
        new = finish_layer(lw, resid, y)
        if is_boundary(l):
            blocks = blocks + [hidden]
        hidden = new
    return glob.final_hidden(blocks, hidden), kept
