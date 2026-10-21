#!/usr/bin/env python3
"""Greedy generation with A.X-K2 VQ release parts (student only), pipelined over GPUs.

Weights come from a release directory (60 GVQ v2 expert parts, body-nonexpert.gguf,
embed-tokens-bf16.gguf) through eval_e2e.py's independent GGUF reader. Each layer
lives on one GPU: non-experts dequantized to BF16, routed experts kept as their
packed index bytes, F16 gains and E4M3 books. Prefill restores the selected
experts to BF16 (gain x codeword rounded to BF16, as eval_e2e) and runs the
axk2_ref arithmetic; decode steps run a GEMV straight from the packed indices
(cuda/gen_vq.cu, same BF16 weight rounding, F32 accumulation in another order).
The KV cache holds the per-head keys and values expanded from the latent, i.e.
exactly the tensors axk2_ref.attention_short attends over. Prompt + generation
must stay within 2,048 tokens, where the DSA top-2048 selection keeps every key.

Results differ slightly between GPU architectures, CUDA versions, kernels and
batch shapes (near-tied argmaxes flip and the continuation then diverges), so
recipe comparisons here are statistical, not bit-exact.
"""
import argparse
import ctypes as ct
import hashlib
import json
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from axk2_ref import BF, NOPE, ROPE, VDIM, QK, HEADS, KV_LORA, Rope, gated_norm, layer_names, mlp, moe, rms_norm, \
    rope_interleave, route, valid_mask
from eval_e2e import BF16, F32, Gguf, Q4_K, Q8_0, dequant_q4k, dequant_q8, tensor_bytes

ROOT = '/models/A.X-K2'
TOOLS = Path(__file__).resolve().parent
GVQ = struct.Struct('<4sHBBIIIIIIQQQQ32s32s')
EOS = 163704
MAX_TOKENS = 2048
PROJ = ('gate_proj', 'up_proj', 'down_proj')


def build():
    src = TOOLS / 'cuda/gen_vq.cu'
    major, minor = torch.cuda.get_device_capability(0)
    lib = TOOLS / f'cuda/libgen_vq_sm{major}{minor}.so'
    cmd = ['nvcc', '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC', f'-arch=sm_{major}{minor}', str(src),
           '-o', str(lib)]
    stamp = hashlib.sha256(src.read_bytes() + json.dumps(cmd).encode()).hexdigest()
    st = lib.with_suffix('.build.json')
    if not lib.exists() or not st.exists() or json.loads(st.read_text())['stamp'] != stamp:
        subprocess.run(cmd, check=True)
        st.write_text(json.dumps({'stamp': stamp, 'command': cmd}) + '\n')
    so = ct.CDLL(str(lib))
    p, i, l = ct.c_void_p, ct.c_int, ct.c_long
    so.gen_vq_gemv_pairs.argtypes = [p, p, p, p, i, l, l, i, i, p, p, i, p, p, p]
    so.gen_vq_decode_bf16.argtypes = [p, p, p, p, i, i, i, p, p]
    return so


LIB = None


def check(code):
    if code:
        raise RuntimeError(f'CUDA kernel error {code}')


class Bank:
    """One projection of one MoE layer: packed indices of all 256 experts on one device."""
    def __init__(self, g, L, proj, dev):
        links = json.loads(g.meta['axk2_vq.codebooks'])
        mains, planes, gains, first = [], [], [], None
        for e in range(256):
            name = f'model.layers.{L}.mlp.experts.{e}.{proj}.weight'
            _, _, off = g.t[name]
            base = g.base + off
            h = GVQ.unpack_from(g.m, base)
            magic, ver, bits, dim, R, C, K, prec, bid = h[:9]
            go, mo, po = h[10], h[11], h[12]
            assert magic == b'GVQ1' and ver == 2 and dim == 8 and prec == 1 and K == 1 << bits
            key = (bits, R, C, bid)
            if first is None:
                first = key
            assert key == first, f'{name}: {key} != {first}'
            n = R * C // 8
            gains.append(np.frombuffer(g.m, np.uint8, 2 * R, base + go))
            mains.append(np.frombuffer(g.m, np.uint8, n, base + mo))
            planes.append(np.frombuffer(g.m, np.uint8, n * (bits - 8) // 8, base + po))
        self.bits, self.R, self.C, self.book_id = first
        self.n = self.R * self.C // 8
        self.dev = dev
        self.main = torch.from_numpy(np.stack(mains)).to(dev)
        self.plane = torch.from_numpy(np.stack(planes)).to(dev)
        self.gain = torch.from_numpy(np.stack(gains)).view(torch.float16).to(dev).contiguous()
        bname = links[str(self.book_id)]
        btyp, bshape, _ = g.t[bname]
        raw = g.raw(bname, tensor_bytes(btyp, bshape)).copy()
        self.book = torch.from_numpy(raw).view(torch.float8_e4m3fn).float().reshape(-1, 8).to(dev).contiguous()
        assert self.book.shape[0] == 1 << self.bits

    def decode(self, e):
        out = torch.empty((self.R, self.C), dtype=BF, device=self.dev)
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        pl = self.plane[e].data_ptr()
        check(LIB.gen_vq_decode_bf16(self.main[e].data_ptr(), pl, self.gain[e].data_ptr(), self.book.data_ptr(),
                                     self.bits, self.R, self.C, out.data_ptr(), stream))
        return out

    def gemv(self, slot_expert, slot_off, x):
        """x [P, C] BF16 rows grouped by expert slot -> F32 [P, R]."""
        P = x.shape[0]
        y = torch.empty((P, self.R), dtype=torch.float32, device=self.dev)
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        check(LIB.gen_vq_gemv_pairs(self.main.data_ptr(), self.plane.data_ptr(), self.gain.data_ptr(),
                                    self.book.data_ptr(), self.bits, self.n, self.n * (self.bits - 8) // 8,
                                    self.R, self.C, slot_expert.data_ptr(), slot_off.data_ptr(),
                                    len(slot_expert), x.contiguous().data_ptr(), y.data_ptr(), stream))
        return y


class Student:
    def __init__(self, release, n_gpus=8):
        self.release = Path(release)
        self.cfg = json.loads(Path(ROOT, 'config.json').read_text())
        self.body = Gguf(self.release / 'body-nonexpert.gguf')
        emb = Gguf(self.release / 'embed-tokens-bf16.gguf')
        typ, shape, _ = emb.t['model.embed_tokens.weight']
        self.embed = emb.raw('model.embed_tokens.weight', tensor_bytes(typ, shape)).view(np.uint16).reshape(shape)
        n_layers = self.cfg['num_hidden_layers']
        split = np.array_split(np.arange(n_layers), n_gpus)
        self.dev_of = {int(L): f'cuda:{g}' for g, part in enumerate(split) for L in part}
        self.rope = {}
        for g in range(n_gpus):
            with torch.cuda.device(g):
                self.rope[f'cuda:{g}'] = Rope(ROOT, f'cuda:{g}')
        self.layers = []
        t0 = time.monotonic()
        for L in range(n_layers):
            dev = self.dev_of[L]
            with torch.cuda.device(dev):
                p = f'model.layers.{L}.'
                names = [n for n in layer_names(self.cfg, L) if '.indexer.' not in n]
                w = {n[len(p):]: self.native(n, dev) for n in names}
                if L >= self.cfg['first_k_dense_replace']:
                    g = Gguf(self.release / f'layer-{L:02d}-experts.gguf')
                    banks = {pr: Bank(g, L, pr, dev) for pr in PROJ}
                    w['banks'] = banks

                    def expert(e, banks=banks):
                        return (torch.cat([banks['gate_proj'].decode(e), banks['up_proj'].decode(e)], 0),
                                banks['down_proj'].decode(e))
                    w['experts'] = expert
                self.layers.append(w)
            if L % 10 == 0:
                print(json.dumps({'load_layer': L, 'device': dev, 'seconds': round(time.monotonic() - t0, 1)}),
                      flush=True)
        last = self.dev_of[n_layers - 1]
        self.head_dev = last
        with torch.cuda.device(last):
            self.norm = self.native('model.norm.weight', last)
            self.lm_head = self.native('lm_head.weight', last)

    def native(self, name, dev):
        typ, shape, _ = self.body.t[name]
        raw = self.body.raw(name, tensor_bytes(typ, shape))
        if typ == Q4_K:
            return dequant_q4k(raw, shape, dev).to(BF)
        if typ == Q8_0:
            return dequant_q8(raw, shape, dev).to(BF)
        dt = {F32: torch.float32, BF16: torch.bfloat16, 1: torch.float16}[typ]
        t = torch.from_numpy(raw.copy()).view(dt).reshape(shape).to(dev)
        return t if typ == F32 else t.to(BF)

    def embed_rows(self, ids):
        rows = self.embed[np.asarray(ids, dtype=np.int64)]
        return torch.from_numpy(np.ascontiguousarray(rows)).view(torch.bfloat16)

    # ------------------------------------------------------------ attention
    def attn_prefill(self, w, x, lengths, rope):
        B, T, _ = x.shape
        a = 'self_attn.'
        cos, sin = rope(T)
        qc = F.linear(x, w[a + 'q_a_proj.weight'])
        qr = rms_norm(qc, w[a + 'q_a_layernorm.weight'])
        qg = F.linear(torch.cat([qr, qc], dim=-1), w[a + 'q_b_proj.weight']).view(B, T, HEADS, QK + VDIM)
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
        o = F.scaled_dot_product_attention(query, key, v, is_causal=True, scale=rope.scaling)
        o = o.transpose(1, 2).reshape(B, T, HEADS * VDIM)
        o = (o * torch.sigmoid(gate.float())).to(o.dtype)
        return F.linear(o, w[a + 'o_proj.weight']), key, v

    def attn_step(self, w, x, pos, kc, vc, rope, cos_tab, sin_tab):
        B = x.shape[0]
        a = 'self_attn.'
        qc = F.linear(x, w[a + 'q_a_proj.weight'])
        qr = rms_norm(qc, w[a + 'q_a_layernorm.weight'])
        qg = F.linear(torch.cat([qr, qc], dim=-1), w[a + 'q_b_proj.weight']).view(B, HEADS, QK + VDIM)
        q, gate = torch.split(qg, [QK, VDIM], dim=-1)
        q = q.unsqueeze(2)
        gate = gate.reshape(B, HEADS * VDIM)
        q_pass, q_rot = torch.split(q, [NOPE, ROPE], dim=-1)
        ckv = F.linear(x, w[a + 'kv_a_proj_with_mqa.weight'])
        kv_pass, k_rot = torch.split(ckv, [KV_LORA, ROPE], dim=-1)
        k_pass = rms_norm(kv_pass, w[a + 'kv_a_layernorm.weight'])
        k_rot = k_rot.reshape(B, 1, 1, ROPE)
        cos, sin = cos_tab[pos][:, None, None, :], sin_tab[pos][:, None, None, :]
        q_rot, k_rot = rope_interleave(q_rot, k_rot, cos, sin)
        query = torch.cat((q_pass, q_rot), dim=-1)
        kv = F.linear(k_pass, w[a + 'kv_b_proj.weight']).view(B, HEADS, NOPE + VDIM)
        k_nope, v = torch.split(kv, [NOPE, VDIM], dim=-1)
        key = torch.cat([k_nope, k_rot.reshape(B, 1, ROPE).expand(B, HEADS, ROPE)], dim=-1)
        ar = torch.arange(B, device=x.device)
        kc[ar, :, pos] = key
        vc[ar, :, pos] = v
        T = int(pos.max()) + 1
        mask = torch.arange(T, device=x.device)[None, :] <= pos[:, None]
        o = F.scaled_dot_product_attention(query, kc[:, :, :T], vc[:, :, :T], attn_mask=mask[:, None, None, :],
                                           scale=rope.scaling)
        o = o.reshape(B, HEADS * VDIM)
        o = (o * torch.sigmoid(gate.float())).to(o.dtype)
        return F.linear(o, w[a + 'o_proj.weight'])

    # ------------------------------------------------------------ ffn
    def post_norm(self, w, L, h):
        if L < self.cfg['first_k_dense_replace']:
            return rms_norm(h, w['post_attention_layernorm.weight'])
        return gated_norm(h, w, 'post_attention_layernorm')

    def ffn_prefill(self, w, L, x):
        if L < self.cfg['first_k_dense_replace']:
            return mlp(x, w['mlp.gate_proj.weight'], w['mlp.up_proj.weight'], w['mlp.down_proj.weight'])
        return moe(self.cfg, w, x)[0]

    def ffn_step(self, w, L, x):
        if L < self.cfg['first_k_dense_replace']:
            return mlp(x, w['mlp.gate_proj.weight'], w['mlp.up_proj.weight'], w['mlp.down_proj.weight'])
        idx, wt = route(self.cfg, w, x)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // idx.shape[1]
        we = wt.reshape(-1)[order]
        uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
        off = torch.zeros(len(uniq) + 1, dtype=torch.int32, device=x.device)
        off[1:] = torch.cumsum(counts, 0)
        uniq = uniq.to(torch.int32)
        xp = x[tok]
        b = w['banks']
        g = b['gate_proj'].gemv(uniq, off, xp).to(BF)
        u = b['up_proj'].gemv(uniq, off, xp).to(BF)
        a_ = F.silu(g) * u
        y = b['down_proj'].gemv(uniq, off, a_).to(BF)
        acc = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
        acc.index_add_(0, tok, y.float() * we[:, None])
        sh = mlp(x, w['mlp.shared_experts.gate_proj.weight'], w['mlp.shared_experts.up_proj.weight'],
                 w['mlp.shared_experts.down_proj.weight'])
        return acc.to(x.dtype) + sh

    # ------------------------------------------------------------ passes
    @torch.no_grad()
    def prefill(self, prompts, t_cache=None):
        """prompts: list of token lists. Returns logits of the last prompt token [B, V] and, with t_cache,
        per-layer caches sized t_cache; also all-position logits when t_cache is None."""
        B = len(prompts)
        lengths = [len(p) for p in prompts]
        T = max(lengths)
        assert (t_cache or T) <= MAX_TOKENS, 'beyond 2,048 tokens DSA selection is not the identity'
        ids = [t for p in prompts for t in p]
        dev0 = self.dev_of[0]
        h = self.embed_rows(ids).to(dev0)
        caches = []
        for L, w in enumerate(self.layers):
            dev = self.dev_of[L]
            with torch.cuda.device(dev):
                h = h.to(dev)
                valid = valid_mask(lengths, T, dev)
                x = torch.zeros((B, T, h.shape[1]), dtype=h.dtype, device=dev)
                x[valid] = gated_norm(h, w, 'input_layernorm')
                y, key, v = self.attn_prefill(w, x, lengths, self.rope[dev])
                h = h + y[valid]
                if t_cache:
                    kc = torch.zeros((B, HEADS, t_cache, QK), dtype=BF, device=dev)
                    vc = torch.zeros((B, HEADS, t_cache, VDIM), dtype=BF, device=dev)
                    kc[:, :, :T] = key
                    vc[:, :, :T] = v
                    caches.append((kc, vc))
                del x, y, key, v
                h = h + self.ffn_prefill(w, L, self.post_norm(w, L, h))
        with torch.cuda.device(self.head_dev):
            h = h.to(self.head_dev)
            if t_cache:
                last = torch.tensor([sum(lengths[:i + 1]) - 1 for i in range(B)], device=self.head_dev)
                return F.linear(rms_norm(h[last], self.norm), self.lm_head).float(), caches, lengths
            return h, lengths

    @torch.no_grad()
    def logits_all(self, h_final, chunk=4096):
        with torch.cuda.device(self.head_dev):
            out = []
            for c0 in range(0, h_final.shape[0], chunk):
                out.append(F.linear(rms_norm(h_final[c0:c0 + chunk], self.norm), self.lm_head).float().argmax(-1))
            return torch.cat(out)

    @torch.no_grad()
    def step(self, tokens, pos, caches):
        """tokens [B] next input ids, pos [B] their positions -> logits [B, V]."""
        h = self.embed_rows(tokens.tolist()).to(self.dev_of[0])
        for L, w in enumerate(self.layers):
            dev = self.dev_of[L]
            with torch.cuda.device(dev):
                h = h.to(dev)
                p = pos.to(dev)
                rope = self.rope[dev]
                cos_tab, sin_tab = rope(MAX_TOKENS)
                kc, vc = caches[L]
                h = h + self.attn_step(w, gated_norm(h, w, 'input_layernorm'), p, kc, vc, rope, cos_tab, sin_tab)
                h = h + self.ffn_step(w, L, self.post_norm(w, L, h))
        with torch.cuda.device(self.head_dev):
            h = h.to(self.head_dev)
            return F.linear(rms_norm(h, self.norm), self.lm_head).float()

    @torch.no_grad()
    def generate(self, prompts, max_new, temperature=0.0, top_p=1.0, seed=0):
        B = len(prompts)
        t_cache = max(len(p) for p in prompts) + max_new
        assert t_cache <= MAX_TOKENS
        t0 = time.monotonic()
        logits, caches, lengths = self.prefill(prompts, t_cache)
        t_prefill = time.monotonic() - t0
        gen = torch.Generator(device=self.head_dev).manual_seed(seed)
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool)
        pos = torch.tensor(lengths, dtype=torch.long)
        t1 = time.monotonic()
        steps = 0
        for _ in range(max_new):
            if temperature > 0:
                probs = torch.softmax(logits / temperature, -1)
                sp, si = torch.sort(probs, descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p
                sp = sp * keep
                nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen))[:, 0]
            else:
                nxt = logits.argmax(-1)
            nxt = nxt.cpu()
            for b in range(B):
                if not done[b]:
                    out[b].append(int(nxt[b]))
                    if int(nxt[b]) == EOS:
                        done[b] = True
            steps += 1
            if bool(done.all()):
                break
            logits = self.step(nxt, pos, caches)
            pos = pos + 1
        t_dec = time.monotonic() - t1
        return out, {'prefill_s': t_prefill, 'decode_s': t_dec, 'steps': steps,
                     'generated_tokens': sum(len(o) for o in out)}


# ---------------------------------------------------------------- prompts and metrics

TOOLS_SPEC = [
    {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get the current weather for a city.',
     'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'City name'},
                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}},
    {'type': 'function', 'function': {'name': 'search_web', 'description': 'Search the web and return result snippets.',
     'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
    {'type': 'function', 'function': {'name': 'calculator', 'description': 'Evaluate an arithmetic expression.',
     'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}},
    {'type': 'function', 'function': {'name': 'send_email', 'description': 'Send an email.',
     'parameters': {'type': 'object', 'properties': {'to': {'type': 'string'}, 'subject': {'type': 'string'},
                    'body': {'type': 'string'}}, 'required': ['to', 'subject', 'body']}}},
]


def long_contexts():
    readme = Path(ROOT, 'README.md').read_text()
    a = readme[readme.index('## Key Features'):readme.index('### Think-Fusion')]
    a = re.sub(r'<[^>]+>', '', a)
    beam = (Path(__file__).resolve().parents[2] / 'docs/beam-vq.md').read_text()
    b = beam[beam.index('## Execution on GB10'):beam.index('## Output protocol')] + \
        beam[beam.index('## Performance'):beam.index('## Quality')]
    b = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', b)
    return a.strip(), b.strip()


def prompt_set():
    ctx_a, ctx_b = long_contexts()
    P = []

    def add(cat, lang, think, content, tools=False, summary=None):
        P.append({'id': f'{len(P):02d}', 'category': cat, 'lang': lang, 'think': think, 'tools': tools,
                  'messages': [{'role': 'user', 'content': content}], 'summary': summary or content[:80]})
    ko = ['한국의 사계절 특징을 각각 두 문장으로 설명해줘.',
          "다음 문장을 정중한 존댓말로 바꿔줘: '내일 회의 시간 좀 알려줘. 그리고 자료도 미리 보내줘.'",
          '초등학생에게 광합성을 설명하듯이 쉽게 설명해줘.',
          '서울에서 부산까지 KTX로 이동할 때의 장점과 단점을 정리해줘.',
          "'지속 가능한 도시'를 주제로 다섯 줄짜리 짧은 시를 써줘.",
          '회사 동료에게 보낼 정중한 회의 연기 요청 이메일을 작성해줘.']
    for c in ko:
        add('ko_chat', 'ko', False, c)
    add('ko_longctx', 'ko', False, ctx_a + '\n\n위 글에서 Gated Norm이 해결하려는 문제가 무엇인지 한국어로 두세 문장으로 설명해줘.',
        summary='[A.X K2 README 영어 발췌 약 800토큰] + Gated Norm이 해결하려는 문제를 한국어로 설명')
    add('ko_longctx', 'ko', False, ctx_b + '\n\n위 문서에 따르면 Beam 모델의 디코드 속도는 얼마이고 무엇이 병목인가? 한국어로 답해줘.',
        summary='[Beam VQ 문서 영어 발췌 약 800토큰] + 디코드 속도와 병목을 한국어로 답변')
    for c in ['철수는 사과 12개를 가지고 있었는데 친구 3명에게 똑같이 나눠주고 2개를 먹었다. 남은 사과는 몇 개이고 각 친구는 몇 개를 받았을까?',
              '어떤 수에 3을 곱하고 7을 더했더니 31이 되었다. 그 수는 무엇인가?',
              "'나는 어제 친구와 영화를 봤다'를 영어로 번역하고 두 언어의 문장 구조 차이를 설명해줘.",
              '기후 변화가 한국 농업에 미치는 영향을 세 가지로 분석해줘.']:
        add('ko_think', 'ko', True, c)
    for c in ['Explain the difference between TCP and UDP in a short paragraph.',
              'Write a haiku about autumn leaves.',
              'Summarize the plot of Romeo and Juliet in three sentences.',
              'Give me five practical tips for improving sleep quality.',
              "Rewrite this sentence to be more formal: 'Hey, can you send me the report asap?'",
              'What are the main causes of inflation? Answer briefly.']:
        add('en_chat', 'en', False, c)
    add('code', 'en', False, 'Write a Python function that checks whether a string has balanced parentheses, brackets and braces. Include a few test cases.')
    add('code', 'en', False, 'Implement binary search in C and explain its time complexity.')
    add('code', 'en', True, 'Write a Python function that returns the n-th Fibonacci number in O(log n) time.')
    add('code', 'en', True, 'What does this JavaScript print and why? console.log([1, 2, 3].map(parseInt))')
    for c in ['What is the sum of all positive integers less than 100 that are divisible by 3 or 5?',
              'A train travels 120 km in 1.5 hours. What is its average speed in meters per second?',
              'Find all real solutions of x^2 - 5x + 6 = 0.']:
        add('math_think', 'en', True, c)
    add('math_think', 'ko', True, '주사위 두 개를 던질 때 두 눈의 합이 7이 될 확률을 구해줘.')
    add('tool', 'en', False, "What's the weather like in Seoul right now?", tools=True)
    add('tool', 'ko', False, '서울과 부산의 현재 날씨를 각각 알려줘.', tools=True)
    add('tool', 'en', False, 'Compute 1234 * 5678 using the calculator.', tools=True)
    add('tool', 'en', True, "Send an email to kim@example.com with the subject 'Meeting' saying the meeting moved to 3pm.", tools=True)
    add('ja_chat', 'ja', False, '日本の首都とその有名な観光地を3つ教えてください。')
    add('zh_chat', 'zh', False, '请用三句话介绍一下长城。')
    return P


def render(tok, p):
    kw = {'tools': TOOLS_SPEC} if p['tools'] else {}
    text = tok.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=p['think'], **kw)
    return tok(text, add_special_tokens=False).input_ids


TOOL_NAMES = {t['function']['name'] for t in TOOLS_SPEC}


def metrics(p, ids, text, max_new):
    m = {'tokens': len(ids), 'finished': bool(ids and ids[-1] == EOS)}
    body = ids[:-1] if m['finished'] else ids
    grams = [tuple(body[i:i + 4]) for i in range(max(0, len(body) - 3))]
    m['rep4'] = (1 - len(set(grams)) / len(grams)) if grams else 0.0
    g8 = {}
    for i in range(max(0, len(body) - 7)):
        k = tuple(body[i:i + 8])
        g8[k] = g8.get(k, 0) + 1
    m['max_8gram_count'] = max(g8.values()) if g8 else 0
    m['replacement_chars'] = text.count('�')
    answer = text
    if p['think']:
        m['think_closed'] = '</think>' in text
        answer = text.split('</think>', 1)[1] if m['think_closed'] else ''
    if p['lang'] == 'ko':
        hangul = len(re.findall(r'[가-힣]', answer))
        letters = len(re.findall(r'[A-Za-z가-힣぀-ヿ一-鿿]', answer))
        m['hangul_share'] = hangul / letters if letters else 0.0
    if p['tools']:
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
        ok = 0
        for c in calls:
            try:
                obj = json.loads(c.strip())
                ok += int(obj.get('name') in TOOL_NAMES and isinstance(obj.get('arguments'), dict))
            except Exception:
                pass
        m['tool_calls'] = len(calls)
        m['tool_calls_valid'] = ok
        m['tool_ok'] = bool(calls) and ok == len(calls)
    return m


def aggregate(rows):
    def mean(k, rs):
        v = [r['metrics'][k] for r in rs if k in r['metrics']]
        return (sum(float(x) for x in v) / len(v)) if v else None
    out = {}
    cats = sorted({r['category'] for r in rows})
    for c in ['ALL'] + cats:
        rs = rows if c == 'ALL' else [r for r in rows if r['category'] == c]
        out[c] = {'n': len(rs), **{k: mean(k, rs) for k in ('finished', 'think_closed', 'rep4', 'max_8gram_count',
                                                             'hangul_share', 'replacement_chars', 'tool_ok', 'tokens')}}
    return out


def run_recipe(name, release, prompts, tok, args, log):
    st = Student(release)
    res = {}
    # sanity: teacher-forced prefill argmax vs the eval_e2e goldens (same release bytes)
    if args.goldens and Path(args.goldens).exists():
        from safetensors.torch import load_file
        fin = load_file(str(args.final))
        ids, off = fin['input_ids'].tolist(), fin['offsets'].tolist()
        man = json.loads(Path(args.manifest).read_text())['samples']
        gold = json.loads(Path(args.goldens).read_text())
        cases = [i for i in range(len(off) - 1) if off[i + 1] - off[i] <= MAX_TOKENS]
        seqs = [ids[off[i]:off[i + 1]] for i in cases]
        agree = total = 0
        t0 = time.monotonic()
        for c0 in range(0, len(seqs), 8):
            hfin, lens = st.prefill(seqs[c0:c0 + 8])
            am = st.logits_all(hfin).cpu().tolist()
            pos = 0
            for j, i in enumerate(cases[c0:c0 + 8]):
                key = f"{i:02d}-{man[i]['domain']}"
                g = [r[0] for r in gold[key]['student_top8_ids']]
                a = am[pos:pos + lens[j]]
                agree += sum(int(x == y) for x, y in zip(a, g))
                total += lens[j]
                pos += lens[j]
        res['sanity_prefill_vs_goldens'] = {'cases': len(cases), 'positions': total, 'argmax_agreement': agree / total,
                                            'seconds': time.monotonic() - t0}
        log({'recipe': name, **res['sanity_prefill_vs_goldens']})
        # incremental decode step vs prefill logits of the same position
        s = seqs[0][:600]
        full_logits, _, _ = st.prefill([s], t_cache=len(s) + 2)
        pre_logits, caches, lens = st.prefill([s[:-1]], t_cache=len(s) + 2)
        step_logits = st.step(torch.tensor([s[-1]]), torch.tensor([len(s) - 1]), caches)
        d = (step_logits - full_logits).abs()
        res['sanity_step_vs_prefill'] = {'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
                                         'logit_scale_std': float(full_logits.std()),
                                         'same_argmax': bool(step_logits.argmax() == full_logits.argmax())}
        log({'recipe': name, **res['sanity_step_vs_prefill']})
    rows = []
    for think in (False, True):
        group = [p for p in prompts if p['think'] == think]
        max_new = args.max_new_think if think else args.max_new
        toks = [render(tok, p) for p in group]
        outs, timing = st.generate(toks, max_new)
        timing['tokens_per_s'] = timing['generated_tokens'] / max(timing['decode_s'], 1e-9)
        log({'recipe': name, 'think': think, **timing})
        res[f'timing_think_{think}'] = timing
        for p, t_in, o in zip(group, toks, outs):
            text = tok.decode(o, skip_special_tokens=False)
            rows.append({'id': p['id'], 'category': p['category'], 'lang': p['lang'], 'think': think,
                         'summary': p['summary'], 'prompt_tokens': len(t_in), 'output_ids': o, 'output': text,
                         'metrics': metrics(p, o, text, max_new)})
    rows.sort(key=lambda r: r['id'])
    if args.sample_korean:
        group = [p for p in prompts if p['lang'] == 'ko']
        samp = []
        for think in (False, True):
            sub = [p for p in group if p['think'] == think]
            if not sub:
                continue
            outs, timing = st.generate([render(tok, p) for p in sub], args.max_new_think if think else args.max_new,
                                       temperature=0.9, top_p=0.95, seed=1234)
            for p, o in zip(sub, outs):
                text = tok.decode(o, skip_special_tokens=False)
                samp.append({'id': p['id'], 'category': p['category'], 'think': think, 'output': text,
                             'metrics': metrics(p, o, text, 0)})
        res['sampled_korean'] = samp
    del st
    torch.cuda.empty_cache()
    return rows, res


def main():
    global LIB, ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default=ROOT, help='A.X-K2 checkpoint directory (config, tokenizer, README)')
    ap.add_argument('--recipe', action='append', required=True, help='NAME=RELEASE_DIR (repeatable)')
    ap.add_argument('--out', type=Path, default=Path('/workspace/axk2/work/gen'))
    ap.add_argument('--max-new', type=int, default=512)
    ap.add_argument('--max-new-think', type=int, default=768)
    ap.add_argument('--goldens', default='/workspace/axk2/variants/R-final/eval-release/goldens.json')
    ap.add_argument('--goldens-for', default='R2', help='recipe name whose release made the goldens')
    ap.add_argument('--final', type=Path, default=Path('/workspace/axk2/stats/heldout/final.safetensors'))
    ap.add_argument('--manifest', default='/workspace/axk2/calib/frozen/heldout-81681f74_manifest.json')
    ap.add_argument('--sample-korean', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ROOT = a.source
    torch.backends.cuda.matmul.allow_tf32 = False
    LIB = build()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT)
    prompts = prompt_set()
    if a.limit:
        prompts = prompts[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'prompts.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=1) + '\n')
    logf = (a.out / 'log.jsonl').open('a')

    def log(v):
        s = json.dumps(v, ensure_ascii=False)
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()
    summary = {}
    for spec in a.recipe:
        name, _, rel = spec.partition('=')
        args = argparse.Namespace(**vars(a))
        if name != a.goldens_for:
            args.goldens = None
        rows, res = run_recipe(name, rel, prompts, tok, args, log)
        with (a.out / f'{name}.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        summary[name] = {'release': rel, 'aggregate': aggregate(rows), **res}
        (a.out / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
        log({'recipe': name, 'aggregate_all': summary[name]['aggregate']['ALL']})


if __name__ == '__main__':
    main()
