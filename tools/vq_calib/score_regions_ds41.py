#!/usr/bin/env python3
"""score_ds41.py restricted to the positions a model generates.

Post-trained chat models are never trained to predict prompt text (system prompts, user
turns, tool schemas, tool results), and DeepSeek-V4.1-Flash assigns some of those tokens
probabilities near 1e-8 that small numerical changes move a lot. This splits every
held-out case into
  generated: all positions of raw-text cases, and the assistant spans of chat cases
             (targets after `<｜Assistant｜>` until the next `<｜User｜>` / `<｜System｜>`);
  prompt:    the remaining positions of chat cases,
and reports KL(teacher ‖ student), top-1 agreement and NLL for each.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

from ds41_ref import Source

ROOT = '/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--teacher', required=True)
    p.add_argument('--student', action='append', default=[], help='name=final.safetensors')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    tk = Tokenizer.from_file(f'{a.source}/tokenizer.json')
    A, U, S = (tk.token_to_id(x) for x in ('<｜Assistant｜>', '<｜User｜>', '<｜System｜>'))
    head = Source(a.source).read(['head.weight'])['head.weight'].to(dev).float()
    t = load_file(a.teacher)
    st = {n: load_file(f) for n, f in (s.split('=', 1) for s in a.student)}
    ids, off = t['input_ids'].long(), t['offsets']
    rows = json.load(open(a.manifest))['samples']
    acc = {}
    for i, r in enumerate(rows):
        s, e = int(off[i]), int(off[i + 1])
        seq = ids[s:e].tolist()
        if r.get('format') == 'chat':
            role, gen = 0, []
            for tok in seq:
                if tok == A:
                    role = 1
                elif tok in (U, S):
                    role = 0
                gen.append(role)
            gen = torch.tensor(gen[1:], dtype=torch.bool)
            gen &= torch.tensor([x not in (U, S) for x in seq[1:]])   # predicting the next turn marker is a prompt event
        else:
            gen = torch.ones(e - s - 1, dtype=torch.bool)
        gen = gen.to(dev)
        tgt = ids[s + 1:e].to(dev)
        lt = F.log_softmax(F.linear(t['final'][s:e - 1].to(dev).float(), head), -1)
        nt = -lt.gather(1, tgt[:, None])[:, 0]
        at = lt.argmax(-1)
        for name, f in st.items():
            ls = F.log_softmax(F.linear(f['final'][s:e - 1].to(dev).float(), head), -1)
            kl = (lt.exp() * (lt - ls)).sum(-1)
            top = (ls.argmax(-1) == at).float()
            ns = -ls.gather(1, tgt[:, None])[:, 0]
            for region, m in (('generated', gen), ('prompt', ~gen), ('all', torch.ones_like(gen))):
                q = acc.setdefault(region, {}).setdefault(name, [0, 0.0, 0.0, 0.0, 0.0])
                q[0] += int(m.sum())
                q[1] += float(kl[m].sum())
                q[2] += float(top[m].sum())
                q[3] += float(ns[m].sum())
                q[4] += float(nt[m].sum())
            del ls
    res = {region: {name: {'tokens': q[0], 'kl': q[1] / q[0], 'top1': q[2] / q[0], 'nll': q[3] / q[0],
                           'teacher_nll': q[4] / q[0]} for name, q in d.items()} for region, d in acc.items()}
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    for region, d in res.items():
        for name, v in d.items():
            print(f"{region:9s} {name:4s} tokens {v['tokens']:6d}  KL {v['kl']:.4f}  top-1 {100 * v['top1']:.2f}%  "
                  f"NLL {v['teacher_nll']:.3f} -> {v['nll']:.3f}")


if __name__ == '__main__':
    main()
