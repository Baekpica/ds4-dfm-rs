#!/usr/bin/env python3
"""Stand-in calibration statistics for pipeline tests (NOT for conversion).

Embedding rows of random tokens play the hidden states of one MoE layer: gated
post-attention RMSNorm, the grouped sigmoid router and the routed experts
(transformers AXK2 arithmetic), accumulated in the collector's statistics
format (docs/vq-calibration.md), plus a held-out dump of disjoint tokens.
"""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axk2_source import Source, PROJECTIONS  # noqa: E402

BF = torch.bfloat16


def gated_norm(x, w, eps=1e-6):
    xf = x.float()
    y = (w['norm'] * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF))
    g = F.linear(F.silu(F.linear(y, w['W_down'])), w['W_up'])
    return (y * torch.sigmoid(g.float())).to(BF)


def route(x, gate_w, bias, cfg):
    logits = F.linear(x.float(), gate_w.float())
    scores = logits.sigmoid()
    choice = scores + bias
    G, TG, E = cfg['n_group'], cfg['topk_group'], cfg['n_routed_experts']
    gs = choice.view(-1, G, E // G).topk(2, dim=-1)[0].sum(-1)
    gi = torch.topk(gs, k=TG, dim=-1, sorted=False)[1]
    gm = torch.zeros_like(gs).scatter_(1, gi, 1)
    sm = gm.unsqueeze(-1).expand(-1, G, E // G).reshape(-1, E)
    choice = choice.masked_fill(~sm.bool(), float('-inf'))   # Megatron/vLLM/llama.cpp semantics
    idx = torch.topk(choice, k=cfg['num_experts_per_tok'], dim=-1, sorted=False)[1]
    wt = scores.gather(1, idx)
    wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * cfg['routed_scaling_factor']
    return idx, wt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--tokens', type=int, default=16384)
    p.add_argument('--heldout', type=int, default=4096)
    p.add_argument('--out', type=Path, default=Path('/workspace/axk2/work/lloyd-test/fakestats'))
    p.add_argument('--source', default='/models/A.X-K2')
    a = p.parse_args()
    src = Source(a.source)
    cfg = src.config
    L = a.layer
    emb = src.read(['model.embed_tokens.weight'])['model.embed_tokens.weight']
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 150000, (a.tokens + a.heldout,), generator=g)
    h = emb[ids].cuda()
    del emb
    pre = f'model.layers.{L}.'
    nw = src.load([pre + 'post_attention_layernorm.norm.weight', pre + 'post_attention_layernorm.W_down.weight',
                   pre + 'post_attention_layernorm.W_up.weight', pre + 'mlp.gate.weight'])
    bias = src.read([pre + 'mlp.gate.e_score_correction_bias'])[pre + 'mlp.gate.e_score_correction_bias'].cuda().float()
    w = {'norm': nw[pre + 'post_attention_layernorm.norm.weight'], 'W_down': nw[pre + 'post_attention_layernorm.W_down.weight'],
         'W_up': nw[pre + 'post_attention_layernorm.W_up.weight']}
    x = gated_norm(h * 8, w)          # embedding rows are small; scale toward a residual-stream size
    idx, wt = route(x, nw[pre + 'mlp.gate.weight'], bias, cfg)
    E, H, Fm = cfg['n_routed_experts'], cfg['hidden_size'], cfg['moe_intermediate_size']
    z = lambda *s: torch.zeros(s, dtype=torch.float64, device='cuda')
    st = {'tokens': z(1) + a.tokens, 'count': torch.zeros(E, dtype=torch.int64, device='cuda'), 'w1': z(E), 'w2': z(E),
          'x2': z(E, H), 'x2w': z(E, H), 'a2': z(E, Fm), 'a2w': z(E, Fm), 'bg': z(E, Fm), 'bu': z(E, Fm),
          'ffn_in': x[:a.tokens].float().pow(2).sum(0).double()}
    T = a.tokens
    names = src.expert_names(L)
    for e in range(E):
        tok, slot = torch.where(idx[:T] == e)
        if len(tok) == 0:
            continue
        ws = src.load([f'{pre}mlp.experts.{e}.{q}.weight' for q in PROJECTIONS])
        G_, U_, D_ = (ws[f'{pre}mlp.experts.{e}.{q}.weight'] for q in PROJECTIONS)
        xe = x[tok]
        r = wt[tok, slot].double()
        gg = F.linear(xe, G_).float()
        uu = F.linear(xe, U_).float()
        sg = torch.sigmoid(gg)
        act = gg * sg
        a_ = (F.silu(gg.to(BF)) * uu.to(BF)).float()
        dsilu = sg * (1 + gg * (1 - sg))
        st['count'][e] += len(tok)
        st['w1'][e] += r.sum()
        st['w2'][e] += (r ** 2).sum()
        xf = xe.float().double() ** 2
        st['x2'][e] += xf.sum(0)
        st['x2w'][e] += (xf * r[:, None] ** 2).sum(0)
        a2 = a_.double() ** 2
        st['a2'][e] += a2.sum(0)
        st['a2w'][e] += (a2 * r[:, None] ** 2).sum(0)
        st['bg'][e] += ((dsilu * uu).double() ** 2 * r[:, None] ** 2).sum(0)
        st['bu'][e] += (act.double() ** 2 * r[:, None] ** 2).sum(0)
    out = a.out / 'rank-0'
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in st.items()}, str(out / f'layer-{L:02d}.safetensors'))
    hd = a.out / 'heldout'
    hd.mkdir(parents=True, exist_ok=True)
    save_file({'ffn_in': x[T:].cpu().contiguous(), 'topk_idx': idx[T:].to(torch.int32).cpu().contiguous(),
               'topk_w': wt[T:].float().cpu().contiguous()}, str(hd / f'layer-{L:02d}.safetensors'))
    c = st['count'].float()
    print({'layer': L, 'count_min': int(c.min()), 'count_median': float(c.median()), 'count_max': int(c.max()),
           'unused': int((c == 0).sum())})


if __name__ == '__main__':
    main()
