#!/usr/bin/env python3
"""Per-layer routed-expert output error of VQ students on the teacher's own inputs.

The teacher (original FP4/FP8 experts) runs the held-out set layer by layer. At every MoE
layer the routed output y = sum_k w_k E_k(x) of each token is also computed with each
student's experts on the same input x and the same routes (the teacher's), and
||y_s - y_t||^2 / ||y_t||^2 is summed per layer and per held-out case. Unlike the
end-to-end KL this has no error propagation between layers: it measures the experts.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ds41_ref import BF, Engram, EngramHash, Layer, SeqState, Teacher, act_quant, rope_tables
from ds41_vq_student import VQExperts

ROOT = '/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'


class MeasuredLayer(Layer):
    """Layer whose MoE also evaluates student experts on the teacher's routes."""

    def __init__(self, *args, students=None, case_of=None, acc=None, **kw):
        super().__init__(*args, **kw)
        self.students, self.case_of, self.acc = students, case_of, acc

    def expert_out(self, experts, e, x, we):
        w1, w3, w2 = experts(e)
        xq = act_quant(x)
        g = F.linear(xq, w1).float()
        u = F.linear(xq, w3).float()
        lim = self.a['swiglu_limit']
        u = u.clamp(-lim, lim)
        g = g.clamp(max=lim)
        return F.linear(act_quant((we[:, None] * (F.silu(g) * u)).to(BF)), w2).float()

    def moe(self, x, stats=None, routes_out=None):
        a, w = self.a, self.w
        N, D = x.shape
        E, K = a['n_routed_experts'], a['n_activated_experts']
        sc = F.softplus(F.linear(x.float(), w['ffn.gate.weight'].float())).sqrt()
        idx = (sc + w['ffn.gate.bias']).topk(K, dim=-1)[1]
        wt = sc.gather(1, idx)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * a['route_scale']
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E).tolist()
        tok_all = order // K
        wt_all = wt.reshape(-1)[order]
        yt = torch.zeros((N, D), dtype=torch.float32, device=x.device)
        ys = {n: torch.zeros((N, D), dtype=torch.float32, device=x.device) for n in self.students}
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            tok = tok_all[start:start + n]
            we = wt_all[start:start + n]
            start += n
            xe = x[tok]
            yt.index_add_(0, tok, self.expert_out(w['experts'], e, xe, we))
            for name, ex in self.students.items():
                ys[name].index_add_(0, tok, self.expert_out(ex, e, xe, we))
        cid = self.case_of
        tok_e = yt.pow(2).sum(-1)
        et = torch.zeros(int(cid.max()) + 1, dtype=torch.float64, device=x.device).index_add_(0, cid, tok_e.double())
        row = {'teacher_energy': et.cpu(), 'tok_energy': tok_e.cpu()}
        for name in self.students:
            tok_d = (ys[name] - yt).pow(2).sum(-1)
            row[name] = torch.zeros_like(et).index_add_(0, cid, tok_d.double()).cpu()
            row['tok_' + name] = tok_d.cpu()                  # per-token error energy (each token equal)
        self.acc[self.L] = row
        # the teacher's own output continues the forward
        out = yt
        for c0 in range(0, N, 65536):
            xc = x[c0:c0 + 65536]
            xq = act_quant(xc)
            g = F.linear(xq, w['ffn.shared_experts.w1.weight']).float()
            u = F.linear(xq, w['ffn.shared_experts.w3.weight']).float()
            lim = a['swiglu_limit']
            u = u.clamp(-lim, lim)
            g = g.clamp(max=lim)
            out[c0:c0 + 65536] += F.linear(act_quant((F.silu(g) * u).to(BF)), w['ffn.shared_experts.w2.weight']).float()
        return out.to(BF)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--student', action='append', default=[], help='name=GGUF path or blob directory')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = 'cuda'
    teacher = Teacher(a.source, dev)
    args = teacher.a
    data = load_file(str(a.data))
    ids, off = data['input_ids'], data['offsets']
    seqs = [(int(off[i]), int(off[i + 1] - off[i])) for i in range(len(off) - 1)]
    tok = ids.to(dev)
    N = tok.numel()
    case_of = torch.cat([torch.full((n,), i, dtype=torch.long) for i, (_, n) in enumerate(seqs)]).to(dev)
    cases = [r['case'] for r in json.load(open(a.manifest))['samples']]
    students = {s.split('=', 1)[0]: VQExperts(s.split('=', 1)[1], dev) for s in a.student}
    ropes = rope_tables(args, max(n for _, n in seqs) + 8, dev)
    eh = EngramHash(args, a.source, dev)
    hashes = torch.cat([eh(tok[s:s + n]) for s, n in seqs]).cpu()
    emb = teacher.src.read(['embed.weight'])['embed.weight'].to(dev)
    h = emb[tok.long()].unsqueeze(1).repeat(1, args['hc_mult'], 1).contiguous()
    del emb
    pre_mix = torch.zeros((N, args['hc_mult']), dtype=torch.float32, device=dev)
    pre_mix[:, 0] = 1.0
    states = [SeqState() for _ in seqs]
    acc = {}
    t0 = time.monotonic()
    for L in range(args['n_layers']):
        w = teacher.layer(L, teacher.host_layer(L))
        engram, rows_fn = None, None
        if L in args['engram_layer_ids']:
            li = args['engram_layer_ids'].index(L)
            engram = Engram(args, w, teacher.src.memmap(f'layers.{L}.engram.embed.weight'),
                            teacher.src.memmap(f'layers.{L}.engram.embed.scale'))
            rows_fn = lambda c0, c1, li=li, eg=engram: eg.rows(hashes[c0:c1, li])
        st = {n: s.layer(L) for n, s in students.items()}
        layer = MeasuredLayer(args, L, w, ropes, engram, students=st, case_of=case_of, acc=acc)
        with torch.no_grad():
            pre_mix = layer.forward(h, pre_mix, seqs, states, None, rows_fn)
        r = acc[L]
        rel = {n: r['tok_' + n] / r['tok_energy'].clamp_min(1e-30) for n in students}
        print(json.dumps({'layer': L, 'seconds': round(time.monotonic() - t0, 1),
                          **{n: round(float(r[n].sum() / r['teacher_energy'].sum()), 5) for n in students},
                          **{n + '_median_tok': round(float(rel[n].median()), 5) for n in students}}), flush=True)
        del w, layer, st
        torch.cuda.empty_cache()
    out = {'cases': cases, 'layers': {L: {k: v.tolist() for k, v in r.items() if not k.startswith('tok_')}
                                      for L, r in acc.items()}}
    a.out.write_text(json.dumps(out) + '\n')
    from safetensors.torch import save_file
    save_file({f'{k}-{L:02d}': v.contiguous() for L, r in acc.items() for k, v in r.items() if k.startswith('tok_')},
              str(a.out.with_suffix('.tokens.safetensors')))


if __name__ == '__main__':
    main()
