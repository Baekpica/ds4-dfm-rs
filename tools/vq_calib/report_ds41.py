#!/usr/bin/env python3
"""Tables for the DeepSeek-V4.1-Flash iVQ card from score_ds41.py output and the routes the
eval passes wrote: overall and per-family teacher-forced KL / top-1 / NLL, and the mean
top-6 routed-expert overlap with the teacher per layer."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

GROUPS = [('WikiText-2', ['wikitext2']), ('Web en / ko / zh / ja / es', ['web_en_short', 'web_ko', 'web_zh', 'web_ja', 'web_es']),
          ('Math web', ['math_raw']), ('Code files (Python, C)', ['code_raw']),
          ('Chat ko / zh / ja', ['chat_ko', 'chat_zh', 'chat_ja']), ('Reasoning (think)', ['reason_general', 'reason_math']),
          ('Instruction following', ['chat_if']), ('Tool calls / agents (DSML)', ['agent_tool', 'ds4_prompts']),
          ('SWE agentless', ['agent_swe']), ('Competitive programming (think)', ['code_chat']),
          ('Long 8K (web, finance)', ['web_en_long', 'long_finance'])]


def family(row):
    f = row.get('domain') or row.get('family')
    if f == 'web_en':
        return 'web_en_long' if row['n_tokens'] > 4096 else 'web_en_short'
    return f


def overlap(a, b):
    """mean |S_a ∩ S_b| / k over tokens, a/b int [N, k]."""
    eq = (a.long().unsqueeze(-1) == b.long().unsqueeze(-2)).any(-1)
    return float(eq.float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--routes', action='append', default=[], help='name=routes.safetensors (first is the teacher)')
    p.add_argument('--finals', action='append', default=[], help='name=final.safetensors (first is the teacher)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sc = json.loads(a.score.read_text())
    rows = json.loads(a.manifest.read_text())['samples']
    names = [k for k in sc['overall'] if k not in ('tokens', 'teacher_nll', 'teacher_ppl')]
    res = {'overall': sc['overall'], 'groups': [], 'routes': {}}
    for title, fams in GROUPS:
        sel = [i for i, r in enumerate(rows) if family(r) in fams]
        if not sel:
            continue
        n = sum(sc['cases'][i]['tokens'] for i in sel)
        g = {'group': title, 'cases': len(sel), 'tokens': n,
             'teacher_nll': sum(sc['cases'][i]['teacher_nll'] * sc['cases'][i]['tokens'] for i in sel) / n}
        for k in names:
            g[k] = {q: sum(sc['cases'][i][k][q] * sc['cases'][i]['tokens'] for i in sel) / n for q in ('kl', 'top1', 'nll')}
        res['groups'].append(g)
    if a.routes:
        rs = [(s.split('=', 1)[0], load_file(s.split('=', 1)[1])) for s in a.routes]
        tname, t = rs[0]
        for name, r in rs[1:]:
            res['routes'][name] = {L: overlap(t[f'idx-{L:02d}'], r[f'idx-{L:02d}']) for L in range(40)}
    if a.finals:
        fs = [(x.split('=', 1)[0], load_file(x.split('=', 1)[1])['final'].float()) for x in a.finals]
        t = fs[0][1]
        res['final_cos'] = {k: float(torch.nn.functional.cosine_similarity(t, f, dim=-1).mean()) for k, f in fs[1:]}
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    hdr = '| Model | KL | Top-1 | NLL | PPL |'
    print(hdr + '\n|---|---:|---:|---:|---:|')
    o = sc['overall']
    print(f"| Teacher (FP4/FP8) | 0 | 100% | {o['teacher_nll']:.3f} | {o['teacher_ppl']:.2f} |")
    for k in names:
        print(f"| {k} | {o[k]['kl']:.4f} | {100 * o[k]['top1']:.2f}% | {o[k]['nll']:.3f} | {o[k]['ppl']:.2f} |")
    print()
    print('| Group | Cases | Tokens | ' + ' | '.join(f'{k} KL / top-1' for k in names) + ' |')
    print('|---|---:|---:|' + '---|' * len(names))
    for g in res['groups']:
        print(f"| {g['group']} | {g['cases']} | {g['tokens']:,} | " +
              ' | '.join(f"{g[k]['kl']:.3f} / {100 * g[k]['top1']:.1f}%" for k in names) + ' |')
    print('final hidden cosine', res.get('final_cos'))
    for k, v in res['routes'].items():
        print(k, 'route overlap', {L: round(x, 3) for L, x in v.items() if L % 5 == 4 or L in (0, 1)})


if __name__ == '__main__':
    main()
