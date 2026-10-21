#!/usr/bin/env python3
"""Teacher-forced scores of DeepSeek-V4.1-Flash students against the teacher, from the
final normed hidden states written by `run_ds41.py eval`.

Per held-out case and overall (token-weighted): KL(teacher || student) in nats per
token over the full 129,280-token vocabulary, next-token top-1 agreement with the
teacher, and the NLL of the actual next token for teacher and student. Logits are
`head.weight` (FP32) times the final hidden state, as in inference/model.py.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ds41_ref import Source

ROOT = '/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--teacher', type=Path, required=True, help='teacher final.safetensors')
    p.add_argument('--student', action='append', default=[], help='name=path/final.safetensors')
    p.add_argument('--manifest', type=Path, help='held-out manifest (case names)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    head = Source(a.source).read(['head.weight'])['head.weight'].to(dev).float()
    t = load_file(str(a.teacher))
    ids, off = t['input_ids'].long(), t['offsets']
    students = {}
    for s in a.student:
        name, _, path = s.partition('=')
        students[name] = load_file(path)
        assert torch.equal(students[name]['input_ids'], t['input_ids']), name
    cases = None
    if a.manifest:
        m = json.load(open(a.manifest))
        rows = m.get('samples', m.get('rows')) if isinstance(m, dict) else m
        cases = [r.get('case') or r.get('name') or r.get('family') or str(i) for i, r in enumerate(rows)]
    n_seq = len(off) - 1
    res = {'cases': [], 'overall': {}}
    tot = {k: {'kl': 0.0, 'top1': 0, 'nll': 0.0} for k in students}
    tot_t, tot_n = 0.0, 0
    for i in range(n_seq):
        s, e = int(off[i]), int(off[i + 1])
        # predict tokens s+1..e-1 from positions s..e-2
        acc = {k: {'kl': 0.0, 'top1': 0, 'nll': 0.0} for k in students}
        nll_t, n = 0.0, 0
        for c0 in range(s, e - 1, a.chunk):
            c1 = min(e - 1, c0 + a.chunk)
            tgt = ids[c0 + 1:c1 + 1].to(dev)
            lt = F.log_softmax(F.linear(t['final'][c0:c1].to(dev).float(), head), -1)
            nll_t += -lt.gather(1, tgt[:, None]).sum().item()
            pt = lt.exp()
            at = lt.argmax(-1)
            for k, st in students.items():
                ls = F.log_softmax(F.linear(st['final'][c0:c1].to(dev).float(), head), -1)
                acc[k]['kl'] += (pt * (lt - ls)).sum().item()
                acc[k]['top1'] += (ls.argmax(-1) == at).sum().item()
                acc[k]['nll'] += -ls.gather(1, tgt[:, None]).sum().item()
                del ls
            n += c1 - c0
            del lt, pt
        row = {'case': cases[i] if cases else str(i), 'tokens': n, 'teacher_nll': nll_t / n}
        for k in students:
            row[k] = {'kl': acc[k]['kl'] / n, 'top1': acc[k]['top1'] / n, 'nll': acc[k]['nll'] / n}
            for q in ('kl', 'top1', 'nll'):
                tot[k][q] += acc[k][q]
        res['cases'].append(row)
        tot_t += nll_t
        tot_n += n
        print(json.dumps(row), flush=True)
    res['overall'] = {'tokens': tot_n, 'teacher_nll': tot_t / tot_n, 'teacher_ppl': float(torch.tensor(tot_t / tot_n).exp())}
    for k in students:
        o = {q: tot[k][q] / tot_n for q in ('kl', 'top1', 'nll')}
        o['ppl'] = float(torch.tensor(o['nll']).exp())
        res['overall'][k] = o
    print(json.dumps(res['overall']), flush=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
