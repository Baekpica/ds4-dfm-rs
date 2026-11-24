#!/usr/bin/env python3
"""K2-Horizon-375B-A23B VQ: a forward of the release bytes, measured against the original BF16 model.

The student is rebuilt from the parts alone: the GVQ1 13-bit routed experts of every MoE layer as
bf16(f32(gain) * codeword) (decoded on the GPU; one sampled row per matrix group is compared with the
independent numpy decoder), the Q8_0 matrices of the body and the embedding as bf16(f32(d) * q), the router and
norms as stored (F32). It runs k2h_ref's arithmetic (BF16 GEMMs, F32 RMSNorm) with BF16 K/V or with K (after
RoPE) and V passed through the NVFP4 codec (k2h_ref.nvfp4), teacher-forced on the held-out sequences whose
original final states and routes run_hvq_k2h.py saved (`convert/heldout/teacher.safetensors`).

    eval     KL(original || student), same top-1, NLL and router top-8 overlap, per K/V variant and domain
    goldens  per-layer residual streams, routes and final logits of the release bytes on fixed prompts, for
             the runtime port (every K/V variant)
    longctx  one long document through the release bytes with every K/V variant: next-token loss and the drift
             from BF16 K/V per position bucket

    k2h_eval.py eval --source $SRC --parts $W/parts --teacher $W/convert/heldout/teacher.safetensors --out $W/eval
    k2h_eval.py goldens --source $SRC --parts $W/parts --out $W/goldens
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import k2h_ref as R
import k2h_vq_format as fmt

DOMAINS = ['coding_tool', 'math_science', 'general_reasoning', 'long_retrieval', 'general_text']
VARIANTS = {'kv-bf16': None, 'kv-fp8': 'fp8', 'kv-fp8k-nvfp4v': ('fp8', 'nvfp4'), 'kv-nvfp4': 'nvfp4'}
KV_FORMATS = {'kv-bf16': 'K and V rows as computed (BF16)',
              'kv-fp8': 'E4M3 values with a power-of-two scale per 16 values (2^ceil(log2(amax / 448))): an FP8 reference '
                        'point, 8.5 bits per value; K after RoPE, V',
              'kv-fp8k-nvfp4v': 'K as kv-fp8, V as kv-nvfp4 (6.5 bits per value on average)',
              'kv-nvfp4': 'the recipe: E2M1 values with an E4M3 scale per 16 values, max(amax / 6, 2^-9) '
                          '(k2h_ref.nvfp4, the MiniMax / Inkling-2 encoder), 4.5 bits per value; K after RoPE, V'}


def kv_round(x, kind):
    """K or V rows [..., 128] BF16 through a storage format and back."""
    if kind is None:
        return x
    if kind == 'nvfp4':
        return R.nvfp4(x)
    if kind == 'fp8':
        g = x.float().reshape(*x.shape[:-1], -1, 16)
        amax = g.abs().amax(-1, keepdim=True)
        s = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(2.0 ** -100))))
        return ((g / s).to(torch.float8_e4m3fn).float() * s).reshape(x.shape).to(x.dtype)
    raise ValueError(kind)


def layer_forward(lw, h, cos, sin, kv=None):
    """k2h_ref.layer_forward with K (after RoPE) and V passed through kv_round. In place on h [B, T, H]."""
    x = R.rmsnorm(h, lw.attn_norm)
    b, t, _ = x.shape
    q = F.linear(x, lw.q).view(b, t, R.HEADS, R.HD).transpose(1, 2)
    k = F.linear(x, lw.k).view(b, t, R.KV_HEADS, R.HD).transpose(1, 2)
    v = F.linear(x, lw.v).view(b, t, R.KV_HEADS, R.HD).transpose(1, 2)
    q, k = R.apply_partial_rope(q, k, cos, sin)
    kk, kv_ = kv if isinstance(kv, tuple) else (kv, kv)
    k, v = kv_round(k, kk), kv_round(v, kv_)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                                         scale=R.HD ** -0.5, enable_gqa=True)
    h.add_(F.linear(out.transpose(1, 2).reshape(b, t, R.HEADS * R.HD), lw.o))
    x = R.rmsnorm(h, lw.ffn_norm)
    if not lw.moe:
        h.add_(R.dense(lw, x))
        return None
    y, sel = R.moe(lw, x.reshape(-1, R.H))
    h.add_(y.view_as(h))
    return sel


class Release:
    """Weights of the release parts, decoded on one device."""

    def __init__(self, parts, dev):
        self.parts, self.dev = Path(parts), dev
        self.body = fmt.PartReader(self.parts / 'body-nonexpert.gguf')
        self.checked = 0

    def native(self, reader, name):
        e = reader.entries[name]
        raw = reader.raw(name)
        data = torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).to(self.dev)
        raw.release()
        if e['type_id'] == fmt.F32:
            return data.view(torch.float32).reshape(e['shape'])
        if e['type_id'] != fmt.Q8_0:
            raise ValueError(f'{name}: type {e["type_id"]}')
        blk = data.view(-1, 34)
        d = blk[:, :2].contiguous().view(torch.float16).float()
        q = blk[:, 2:].contiguous().view(torch.int8).float()
        return (q * d).reshape(e['shape']).to(torch.bfloat16)

    def embedding(self):
        with fmt.PartReader(self.parts / 'embed-tokens-q8_0.gguf') as r:
            return self.native(r, 'model.embed_tokens.weight')

    def head(self):
        return self.native(self.body, 'model.norm.weight'), self.native(self.body, 'lm_head.weight')

    def experts(self, l, chunk=24):
        out = {p: torch.empty((R.E, *fmt.SHAPES[p]), dtype=torch.bfloat16, device=self.dev) for p in fmt.PROJECTIONS}
        shift = torch.arange(8, device=self.dev, dtype=torch.int64)
        with fmt.PartReader(self.parts / f'layer-{l:02d}-experts.gguf') as r:
            raw = r.raw(fmt.book_name(l))
            book = bytes(raw)
            raw.release()
            cb = torch.frombuffer(bytearray(book), dtype=torch.uint8).to(self.dev).view(torch.float8_e4m3fn).float().reshape(-1, 8)
            k = cb.shape[0]
            for p in fmt.PROJECTIONS:
                rows, cols = fmt.SHAPES[p]
                g = fmt.geometry(rows, cols, k)
                for e0 in range(0, R.E, chunk):
                    es = range(e0, min(R.E, e0 + chunk))
                    bufs = []
                    for e in es:
                        raw = r.raw(fmt.matrix_name(l, e, p))
                        bufs.append(np.frombuffer(raw, np.uint8).copy())
                        raw.release()
                    pay = torch.from_numpy(np.stack(bufs)).to(self.dev)
                    gains = pay[:, g['gain']:g['gain'] + 2 * rows].contiguous().view(torch.float16).float()
                    low = pay[:, g['main']:g['main_end']].reshape(len(es), -1, 3).to(torch.int64)
                    a = low[..., 0] | ((low[..., 1] & 15) << 8)
                    b = (low[..., 1] >> 4) | (low[..., 2] << 4)
                    ix = torch.stack([a, b], -1).reshape(len(es), -1)
                    if g['plane']:
                        hi = pay[:, g['plane']:g['end']].to(torch.int64)
                        ix |= (((hi[..., None] >> shift) & 1).reshape(len(es), -1)) << 12
                    w = cb[ix].reshape(len(es), rows, cols) * gains[:, :, None]
                    out[p][e0:e0 + len(es)] = w.to(torch.bfloat16)
                    if e0 == 0:            # the independent numpy decoder on one row
                        t = fmt.VqTensor(bufs[0], book, (rows, cols), fmt.book_id(l))
                        rr = (l * 131 + 7) % rows
                        if not np.array_equal(t.decode_rows(rr, 1)[0], w[0, rr].cpu().numpy()):
                            raise RuntimeError(f'layer {l} {p}: GPU decode differs from VqTensor')
                        self.checked += 1
                        del t
                    del pay, low, a, b, ix, w, gains, bufs
        return out

    def layer(self, l):
        lw = SimpleNamespace(l=l, dev=self.dev, moe=l >= R.DENSE)
        for key, name in R.Source.names(l).items():
            setattr(lw, key, self.native(self.body, name))
        if lw.moe:
            lw.router = lw.router.to(torch.bfloat16)      # the source's BF16 values, stored as F32
            ex = self.experts(l)
            lw.gate, lw.up, lw.down = ex['gate_proj'], ex['up_proj'], ex['down_proj']
        return lw


def overlap(sel, ref):
    """Mean fraction of the reference's 8 experts the student also selected. sel, ref [n, 8]."""
    a = torch.zeros((sel.shape[0], R.E), dtype=torch.bool, device=sel.device).scatter_(1, sel.long(), True)
    b = torch.zeros_like(a).scatter_(1, ref.long(), True)
    return float((a & b).sum(-1).float().mean() / R.TOPK)


def cmd_eval(a):
    dev = torch.device('cuda', a.gpu)
    t0 = time.time()
    teacher = load_file(str(a.teacher))
    ids = teacher['input_ids'].to(torch.int64)
    n_seq, T = ids.shape
    if a.max_seqs:
        n_seq = min(n_seq, a.max_seqs)
        ids = ids[:n_seq]
    routes = teacher.get('routes')
    data = np.load(a.calib)
    held_idx = np.arange(data['tokens'].shape[0] - teacher['input_ids'].shape[0], data['tokens'].shape[0])[:n_seq]
    if not np.array_equal(data['tokens'][held_idx], ids.numpy()):
        raise RuntimeError('held-out ids differ from the calibration file')
    dom = data['source_ids'][held_idx]
    rel = Release(a.parts, dev)
    emb = rel.embedding()
    h = {v: emb[ids.to(dev)].clone() for v in VARIANTS}
    del emb
    cos, sin = R.rope_cache(T, dev)
    route_rows = {v: [] for v in VARIANTS}
    for l in range(R.L):
        lw = rel.layer(l)
        for v, kv in VARIANTS.items():
            sels = [layer_forward(lw, h[v][s0:s0 + a.batch_seqs], cos, sin, kv=kv) for s0 in range(0, n_seq, a.batch_seqs)]
            if lw.moe and routes is not None:
                ref = routes[l - R.DENSE][:n_seq * T].to(dev)
                route_rows[v].append(overlap(torch.cat(sels), ref))
        del lw
        torch.cuda.empty_cache()
        print(json.dumps({'layer': l, 'elapsed': round(time.time() - t0, 1),
                          **{f'overlap_{v}': (route_rows[v][-1] if route_rows[v] and l >= R.DENSE else None) for v in VARIANTS}}), flush=True)
    # the original's scores (source norm and head) against the students' (release norm and Q8_0 head)
    src = R.Source(a.source)
    glob = R.Globals(src, dev, embed=False, head=True)
    norm_s, head_s = rel.head()
    th = teacher['final_hidden'][:n_seq]
    tot = {v: {'kl': 0.0, 'same': 0, 'nll': 0.0, 'n': 0, 'n_nll': 0, 'cos': 0.0,
               'dom': {d: {'kl': 0.0, 'same': 0, 'nll': 0.0, 'nll_t': 0.0, 'n': 0, 'n_nll': 0} for d in range(len(DOMAINS))},
               'late': {'kl': 0.0, 'same': 0, 'n': 0}, 'kls': []} for v in VARIANTS}
    t_nll, t_top1, kv_kl, kv_same = 0.0, 0, 0.0, 0
    for i in range(n_seq):
        tl = glob.logits(th[i].to(dev))
        tlp = torch.log_softmax(tl, -1)
        tgt = ids[i, 1:].to(dev)
        t_nll += float(-tlp[:-1].gather(1, tgt[:, None]).sum())
        t_top1 += int((tl[:-1].argmax(-1) == tgt).sum())
        slps = {}
        for v in VARIANTS:
            sl = F.linear(R.rmsnorm(h[v][i], norm_s), head_s).float()
            slp = torch.log_softmax(sl, -1)
            slps[v] = slp
            kl = (tlp.exp() * (tlp - slp)).sum(-1)
            same = tl.argmax(-1) == sl.argmax(-1)
            nll = -slp[:-1].gather(1, tgt[:, None])[:, 0]
            r = tot[v]
            r['kl'] += float(kl.sum()); r['same'] += int(same.sum()); r['nll'] += float(nll.sum())
            r['n'] += T; r['n_nll'] += T - 1
            r['cos'] += float(F.cosine_similarity(h[v][i].float(), th[i].to(dev).float(), dim=-1).sum())
            r['late']['kl'] += float(kl[T // 2:].sum()); r['late']['same'] += int(same[T // 2:].sum()); r['late']['n'] += T - T // 2
            d = r['dom'][int(dom[i])]
            d['kl'] += float(kl.sum()); d['same'] += int(same.sum()); d['nll'] += float(nll.sum())
            d['nll_t'] += float(-tlp[:-1].gather(1, tgt[:, None]).sum()); d['n'] += T; d['n_nll'] += T - 1
            r['kls'].append(kl.cpu())
        a_, b_ = slps['kv-bf16'], slps['kv-nvfp4']
        kv_kl += float((a_.exp() * (a_ - b_)).sum())
        kv_same += int((a_.argmax(-1) == b_.argmax(-1)).sum())
        del tl, tlp, slps
    n, n_nll = n_seq * T, n_seq * (T - 1)
    summary = {'sequences': n_seq, 'tokens': n, 'scored_positions': n, 'nll_positions': n_nll,
               'heldout_chunks': [int(held_idx[0]), int(held_idx[-1])],
               'original': {'nll': t_nll / n_nll, 'top1_accuracy': t_top1 / n_nll},
               'student_vs_original': {}, 'payload_groups_cross_checked': rel.checked,
               'kv_nvfp4_vs_kv_bf16_same_weights': {'kl': kv_kl / n, 'same_top1': kv_same / n},
               'kv_formats': KV_FORMATS,
               'seconds': round(time.time() - t0, 1)}
    for v in VARIANTS:
        r = tot[v]
        kls = torch.cat(r['kls'])
        summary['student_vs_original'][v] = {
            'kl': r['kl'] / n, 'kl_median': float(kls.median()), 'same_top1': r['same'] / n, 'nll': r['nll'] / n_nll,
            'final_hidden_cosine': r['cos'] / n,
            'second_half_positions': {'kl': r['late']['kl'] / r['late']['n'], 'same_top1': r['late']['same'] / r['late']['n']},
            'router_top8_overlap': {'mean': float(np.mean(route_rows[v])) if route_rows[v] else None,
                                    'min': float(np.min(route_rows[v])) if route_rows[v] else None,
                                    'per_layer': [round(x, 4) for x in route_rows[v]]},
            'domains': {DOMAINS[d]: {'sequences': x['n'] // T, 'kl': x['kl'] / x['n'], 'same_top1': x['same'] / x['n'],
                                     'nll': x['nll'] / x['n_nll'], 'nll_original': x['nll_t'] / x['n_nll']}
                        for d, x in r['dom'].items() if x['n']}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k_: v for k_, v in summary.items() if k_ not in ('student_vs_original', 'kv_formats')}))
    for v in VARIANTS:
        s = summary['student_vs_original'][v]
        print(json.dumps({'variant': v, **{k_: s[k_] for k_ in ('kl', 'kl_median', 'same_top1', 'nll', 'final_hidden_cosine')},
                          'overlap_mean': s['router_top8_overlap']['mean']}))


PROMPTS = {
    'raw_paris': {'kind': 'raw', 'text': 'The capital of France is'},
    'chat_capital': {'kind': 'chat', 'messages': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}]},
    'chat_code': {'kind': 'chat', 'messages': [{'role': 'user', 'content': 'Write a Python function that returns the n-th Fibonacci number.'}],
                  'kwargs': {'reasoning_effort': 'low'}},
}


def cmd_goldens(a):
    from tokenizers import Tokenizer
    import k2h_chat
    dev = torch.device('cuda', a.gpu)
    tk = Tokenizer.from_file(str(a.source / 'tokenizer.json'))
    rel = Release(a.parts, dev)
    emb = rel.embedding()
    cases = {}
    for name, p in PROMPTS.items():
        if p['kind'] == 'raw':
            text, ids = p['text'], tk.encode(p['text'], add_special_tokens=False).ids
        else:
            text = k2h_chat.render(a.source / 'chat_template.jinja', p['messages'], add_generation_prompt=True, **p.get('kwargs', {}))
            ids = tk.encode(text, add_special_tokens=False).ids
        t = torch.tensor(ids, dtype=torch.int64, device=dev)
        cases[name] = {'text': text, 'ids': ids, 'cs': R.rope_cache(len(ids), dev),
                       'h': {v: emb[t][None].clone() for v in VARIANTS},
                       'hidden': {v: [emb[t].clone().cpu()] for v in VARIANTS}, 'routes': {v: [] for v in VARIANTS}}
    del emb
    for l in range(R.L):
        lw = rel.layer(l)
        for c in cases.values():
            for v, kv in VARIANTS.items():
                sel = layer_forward(lw, c['h'][v], *c['cs'], kv=kv)
                c['hidden'][v].append(c['h'][v][0].clone().cpu())
                if sel is not None:
                    c['routes'][v].append(sel.sort(-1).values.to(torch.uint8).cpu())
        del lw
        torch.cuda.empty_cache()
    norm_s, head_s = rel.head()
    a.out.mkdir(parents=True, exist_ok=True)
    index = {'release_parts': str(a.parts), 'arithmetic': 'k2h_ref (BF16 GEMMs, F32 RMSNorm); experts bf16(f32(gain) * codeword); '
             'Q8_0 as bf16(f32(d) * q); K/V per variant (K after RoPE, V)', 'kv_formats': KV_FORMATS,
             'tensors': {'input_ids': '[T] int32', 'hidden': '[62, T, 6144] BF16: the embedding rows, then the residual stream after each of the 61 layers',
                         'routes': '[58, T, 8] uint8: the selected experts of MoE layers 3..60, ascending',
                         'logits': '[T, 250624] F32 of the release norm and Q8_0 head on the final residual stream',
                         'top_ids / top_logits': '[T, 32] the 32 largest logits per position'},
             'cases': {}}
    for name, c in cases.items():
        for v in VARIANTS:
            logits = F.linear(R.rmsnorm(c['h'][v][0], norm_s), head_s).float()
            top = logits.topk(32, -1)
            path = a.out / f'{name}.{v}.safetensors'
            save_file({'input_ids': torch.tensor(c['ids'], dtype=torch.int32), 'hidden': torch.stack(c['hidden'][v]).contiguous(),
                       'routes': torch.stack(c['routes'][v]).contiguous(), 'logits': logits.cpu().contiguous(),
                       'top_ids': top.indices.to(torch.int32).cpu().contiguous(), 'top_logits': top.values.cpu().contiguous()}, str(path))
            nxt = int(logits[-1].argmax())
            index['cases'].setdefault(name, {'text': c['text'], 'input_ids': c['ids'], 'tokens': len(c['ids']), 'variants': {}})['variants'][v] = {
                'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'next_token_id': nxt,
                'next_token': tk.decode([nxt]), 'next_token_logit': float(logits[-1, nxt]),
                'greedy_per_position': tk.decode(logits.argmax(-1).tolist())}
    (a.out / 'index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({n: {v: [x['variants'][v]['next_token'], x['variants'][v]['greedy_per_position'][-80:]] for v in VARIANTS}
                      for n, x in index['cases'].items()}, ensure_ascii=False, indent=1))


def cmd_longctx(a):
    """One long document through the release bytes with every K/V variant: next-token loss per position bucket and
    the drift of each K/V format from BF16 K/V on the same weights (no original needed)."""
    from tokenizers import Tokenizer
    dev = torch.device('cuda', a.gpu)
    t0 = time.time()
    tk = Tokenizer.from_file(str(a.source / 'tokenizer.json'))
    text = a.text.read_text(encoding='utf-8', errors='replace')[:a.tokens * 12]
    ids = tk.encode(text, add_special_tokens=False).ids[:a.tokens]
    T = len(ids)
    t = torch.tensor(ids, dtype=torch.int64, device=dev)
    rel = Release(a.parts, dev)
    emb = rel.embedding()
    h = {v: emb[t][None].clone() for v in VARIANTS}
    del emb
    cos, sin = R.rope_cache(T, dev)
    for l in range(R.L):
        lw = rel.layer(l)
        for v, kv in VARIANTS.items():
            layer_forward(lw, h[v], cos, sin, kv=kv)
        del lw
        torch.cuda.empty_cache()
        if l % 10 == 0:
            print(json.dumps({'layer': l, 'elapsed': round(time.time() - t0, 1)}), flush=True)
    norm_s, head_s = rel.head()
    edges = [e for e in (0, 512, 2048, 4096, 8192, 16384, 32768) if e < T] + [T]
    nll = {v: torch.zeros(T - 1, dtype=torch.float64) for v in VARIANTS}
    kl = {v: torch.zeros(T, dtype=torch.float64) for v in VARIANTS}
    same = {v: torch.zeros(T, dtype=torch.bool) for v in VARIANTS}
    for c0 in range(0, T, 1024):
        c1 = min(T, c0 + 1024)
        ref = torch.log_softmax(F.linear(R.rmsnorm(h['kv-bf16'][0, c0:c1], norm_s), head_s).float(), -1)
        for v in VARIANTS:
            lp = ref if v == 'kv-bf16' else torch.log_softmax(F.linear(R.rmsnorm(h[v][0, c0:c1], norm_s), head_s).float(), -1)
            n1 = min(c1, T - 1)
            if n1 > c0:
                nll[v][c0:n1] = -lp[:n1 - c0].gather(1, t[c0 + 1:n1 + 1, None])[:, 0].double().cpu()
            kl[v][c0:c1] = (ref.exp() * (ref - lp)).sum(-1).double().cpu()
            same[v][c0:c1] = (ref.argmax(-1) == lp.argmax(-1)).cpu()
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = {'positions': [lo, hi]}
        for v in VARIANTS:
            row[v] = {'nll': float(nll[v][lo:min(hi, T - 1)].mean()), 'kl_vs_kv_bf16': float(kl[v][lo:hi].mean()),
                      'same_top1_vs_kv_bf16': float(same[v][lo:hi].float().mean())}
        buckets.append(row)
    summary = {'text': a.text.name, 'text_sha256': hashlib.sha256(a.text.read_bytes()).hexdigest(), 'tokens': T, 'bos': False,
               'what': 'release weights only; each K/V variant against BF16 K/V on the same weights, and the next-token loss of the text',
               'kv_formats': KV_FORMATS,
               'all_positions': {v: {'nll': float(nll[v].mean()), 'kl_vs_kv_bf16': float(kl[v].mean()),
                                     'same_top1_vs_kv_bf16': float(same[v].float().mean())} for v in VARIANTS},
               'buckets': buckets, 'seconds': round(time.time() - t0, 1)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'longctx.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != 'kv_formats'}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('eval', 'goldens', 'longctx'):
        p = sub.add_parser(name)
        p.add_argument('--source', type=Path, required=True)
        p.add_argument('--parts', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--gpu', type=int, default=0)
    p = sub.choices['eval']
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--batch-seqs', type=int, default=32)
    p.add_argument('--max-seqs', type=int, default=0)
    p = sub.choices['longctx']
    p.add_argument('--text', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=8192)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    {'eval': cmd_eval, 'goldens': cmd_goldens, 'longctx': cmd_longctx}[a.cmd](a)


if __name__ == '__main__':
    main()
