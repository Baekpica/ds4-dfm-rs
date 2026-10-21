#!/usr/bin/env python3
"""axk2_ref layer forward vs transformers AXK2DecoderLayer on the same dequantized weights.

T = 2500 > 2048 exercises the DSA top-k selection (long path); a right-padded
batch of shorter sequences exercises the short path. Layers 0 (dense) and 1 (MoE).
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from axk2_ref import BF, Layer, Rope, TeacherWeights, layer_names, attention_long, gated_norm

ROOT = '/models/A.X-K2'


def hf_layer(cfg_hf, L, w, n_experts):
    from transformers.models.axk2.modeling_axk2 import AXK2DecoderLayer
    with torch.device('meta'):
        layer = AXK2DecoderLayer(cfg_hf, L)
    sd = {}
    for k, v in w.items():
        if k == 'experts':
            continue
        k2 = k.replace('W_down', 'mlp.fc1').replace('W_up', 'mlp.fc2').replace('self_attn.q_b_proj.', 'self_attn.q_gate_proj.')
        if k2 == 'self_attn.indexer.weights_proj.weight':
            v = v.float()
        sd[k2] = v
    if 'experts' in w:
        gu, dn = zip(*[w['experts'](e) for e in range(n_experts)])
        sd['mlp.experts.gate_up_proj'] = torch.stack(gu)
        sd['mlp.experts.down_proj'] = torch.stack(dn)
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise SystemExit(f'HF layer {L}: missing {missing} unexpected {unexpected}')
    return layer.eval()


def compare(a, b):
    d = a.float() - b.float()
    return {'max_abs': d.abs().max().item(), 'rel_rms': (d.pow(2).mean() / b.float().pow(2).mean()).sqrt().item(),
            'cosine': F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item(),
            'bit_identical': bool(torch.equal(a, b))}


@torch.no_grad()
def main():
    import argparse
    from transformers import AutoConfig
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('layers', nargs='*', default=['0', '1'])
    p.add_argument('--source', default=ROOT, help='A.X-K2 checkpoint directory')
    a = p.parse_args()
    cfg_hf = AutoConfig.from_pretrained(a.source)
    cfg_hf._attn_implementation = 'sdpa'
    teacher = TeacherWeights(a.source)
    cfg = teacher.cfg
    rope = Rope(a.source)
    torch.manual_seed(0)
    T = 2500
    out = {}
    for L in [int(x) for x in a.layers]:
        t0 = time.time()
        w = teacher.layer(L)
        mine = Layer(cfg, L, w, rope)
        # realistic-scale residual: rows drawn N(0, 1) * 0.05
        h = (torch.randn(T, cfg['hidden_size'], device='cuda') * 0.05).to(BF)
        h_mine, _, routes = mine.forward(h.clone(), [(0, T)])
        layer = hf_layer(cfg_hf, L, w, cfg['n_routed_experts'])
        pos = torch.arange(T, device='cuda')[None]
        cos, sin = rope.emb(h[None], pos)
        mask = torch.ones(T, T, dtype=torch.bool, device='cuda').tril()[None, None]
        y_hf = layer(h[None], attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
        y_hf = (y_hf[0] if isinstance(y_hf, tuple) else y_hf)[0]
        r = compare(h_mine, y_hf)
        # sublayer check: attention output alone
        x = gated_norm(h, w, 'input_layernorm')[None]
        a_mine = attention_long(w, x, rope(T)[0], rope(T)[1], rope.scaling)[0]
        a_hf, _ = layer.self_attn(layer.input_layernorm(h[None]), position_embeddings=(cos, sin), attention_mask=mask,
                                 position_ids=pos)
        r['attention'] = compare(a_mine, a_hf[0])
        # short path: right-padded batch of three sequences
        lens = [700, 1300, 2048]
        hs = (torch.randn(sum(lens), cfg['hidden_size'], device='cuda') * 0.05).to(BF)
        seqs, st = [], 0
        for n in lens:
            seqs.append((st, n))
            st += n
        a_short = mine.attention(hs, seqs)
        diffs = []
        for (st, n) in seqs:
            p = torch.arange(n, device='cuda')[None]
            c2, s2 = rope.emb(hs[None], p)
            m2 = torch.ones(n, n, dtype=torch.bool, device='cuda').tril()[None, None]
            ah, _ = layer.self_attn(layer.input_layernorm(hs[st:st + n][None]), position_embeddings=(c2, s2),
                                    attention_mask=m2, position_ids=p)
            diffs.append(compare(a_short[st:st + n], ah[0]))
        r['attention_short_batch'] = diffs
        if routes is not None:
            # HF router on the same post-norm input
            hpost = layer.post_attention_layernorm((h + layer.self_attn(layer.input_layernorm(h[None]),
                                                   position_embeddings=(cos, sin), attention_mask=mask,
                                                   position_ids=pos)[0][0])[None])[0]
            _, wts, ids = layer.mlp.gate(hpost)
            a_sorted = torch.sort(routes[0], dim=-1)[0]
            b_sorted = torch.sort(ids, dim=-1)[0]
            r['route_set_exact'] = (a_sorted == b_sorted).all(-1).float().mean().item()
        r['seconds'] = time.time() - t0
        out[L] = r
        print(json.dumps({'layer': L, **r}), flush=True)
        del layer, w, mine
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    main()
