#!/usr/bin/env python3
"""Markdown tables for EVALUATION.md from the evaluation summaries and the conversion receipts.

    report_inkling2.py --eval /work/inkling2/eval --convert /work/inkling2/convert --out /work/inkling2/eval/tables.md
"""
import argparse
import json
from pathlib import Path

VARIANTS = [('kv-fp8', 'Original weights, K/V FP8'), ('kv-nvfp4', 'Original weights, K/V NVFP4'),
            ('ivq', 'iVQ experts (same books, t = 0)'), ('hvq', 'hVQ experts'),
            ('release', 'Release: hVQ experts + Q4_K body + NVFP4 K/V')]


def load(eval_dir, name, part):
    p = eval_dir / name / part / 'summary.json'
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, pct=False, nd=4):
    if v is None:
        return '—'
    return f'{100 * v:.2f}%' if pct else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', type=Path, required=True)
    ap.add_argument('--convert', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = []
    for part, title in (('text', 'Text held-out (26 cases)'), ('image', 'Image held-out (8 cases)')):
        rows = [(n, d, load(a.eval, n, part)) for n, d in VARIANTS]
        rows = [r for r in rows if r[2]]
        if not rows:
            continue
        s0 = rows[0][2]['all']
        out.append(f'### {title}: {s0["positions"]:,} positions, {s0.get("generated", 0):,} generated\n')
        if part == 'image':
            # prompt positions of the image cases are mostly image placeholders (their next token is another
            # placeholder, which the model never predicts): only the answer tokens are scored
            out.append('Answer tokens only: the prompt positions are image placeholders.\n')
            out.append('| vs the original | KL, generated | Same top-1, generated | NLL, generated: original → variant |')
            out.append('|---|---:|---:|---:|')
            for n, d, s in rows:
                x = s['all']
                out.append(f"| {d} | {fmt(x.get('kl_gen'))} | {fmt(x.get('top1_gen'), True)} | {fmt(x.get('nll_teacher_gen'))} → {fmt(x.get('nll_gen'))} |")
            out.append('')
            continue
        out.append('| vs the original | KL, all | KL, generated | Same top-1, all | Same top-1, generated | NLL, all | NLL, generated |')
        out.append('|---|---:|---:|---:|---:|---:|---:|')
        t = rows[0][2]['all']
        out.append(f"| Original (teacher) | 0 | 0 | 100% | 100% | {fmt(t['nll_teacher'])} | {fmt(t.get('nll_teacher_gen'))} |")
        for n, d, s in rows:
            x = s['all']
            out.append(f"| {d} | {fmt(x['kl'])} | {fmt(x.get('kl_gen'))} | {fmt(x['top1'], True)} | {fmt(x.get('top1_gen'), True)} "
                       f"| {fmt(x['nll'])} | {fmt(x.get('nll_gen'))} |")
        out.append('')
        best = next((r for r in rows if r[0] == 'release'), rows[-1])
        out.append(f'#### By domain ({best[1]})\n')
        out.append('| Domain | Positions | KL | KL, generated | Same top-1 | NLL: original → variant |')
        out.append('|---|---:|---:|---:|---:|---:|')
        for dom, x in best[2]['by_domain'].items():
            out.append(f"| {dom} | {x['positions']:,} | {fmt(x['kl'])} | {fmt(x.get('kl_gen'))} | {fmt(x['top1'], True)} "
                       f"| {fmt(x['nll_teacher'])} → {fmt(x['nll'])} |")
        out.append('')
        ov = {n: s.get('route_overlap_by_layer') for n, _, s in rows if s.get('route_overlap_by_layer')}
        if ov:
            names = [n for n in ('ivq', 'hvq', 'release') if n in ov]
            if names:
                out.append('#### Routed-expert set overlap with the original (mean over tokens)\n')
                out.append('| Layers | ' + ' | '.join(names) + ' |')
                out.append('|---|' + '---:|' * len(names))
                for lo, hi in ((0, 11), (12, 23), (24, 35), (36, 47), (0, 47)):
                    vals = []
                    for n in names:
                        v = [ov[n][str(l)] if str(l) in ov[n] else ov[n].get(l) for l in range(lo, hi + 1)]
                        v = [x for x in v if x is not None]
                        vals.append(fmt(sum(v) / len(v), True) if v else '—')
                    out.append(f'| {lo}–{hi} | ' + ' | '.join(vals) + ' |')
                out.append('')
    # in-sample per-layer table from the receipts
    recs = sorted((json.loads(p.read_text()) for p in a.convert.glob('layer-*/receipt.json')), key=lambda r: r['layer'])
    if recs:
        out.append('### Per layer, on the calibration statistics (every 8th expert also converted at t = 0)\n')
        out.append('| Layer | Tokens routed | Min / median per expert | gate/up error under H: iVQ → hVQ | down: iVQ → hVQ | '
                   'Book held-out rel. MSE | Seconds |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|')
        for r in recs:
            e, h, b = r['errors'], r['hessian'], r.get('codebook_training') or {}
            out.append(f"| {r['layer']} | {h['tokens_routed']:,} | {h['count_min']:,} / {int(h['count_median']):,} "
                       f"| {e['x']['sampled_ivq_output_rel']:.3f} → {e['x']['sampled_hvq_output_rel']:.3f} "
                       f"| {e['a']['sampled_ivq_output_rel']:.3f} → {e['a']['sampled_hvq_output_rel']:.3f} "
                       f"| {b.get('heldout_rel_mse', float('nan')):.3f} | {r['seconds']['layer_total']:.0f} |")
        n = len(recs)
        mx = sum(r['errors']['x']['sampled_hvq_output_rel'] / r['errors']['x']['sampled_ivq_output_rel'] for r in recs) / n
        ma = sum(r['errors']['a']['sampled_hvq_output_rel'] / r['errors']['a']['sampled_ivq_output_rel'] for r in recs) / n
        out.append(f'\nMean hVQ / iVQ ratio over {n} layers: gate/up {mx:.3f}, down {ma:.3f}.\n')
    a.out.write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
