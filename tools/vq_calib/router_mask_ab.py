#!/usr/bin/env python3
"""Held-out NLL of the FP8 teacher with transformers' 0.0 group mask vs Megatron/vLLM/llama.cpp -inf.

Both runs are the collector's held-out pass (stats-maskzero/heldout/final.safetensors
and stats/heldout/final.safetensors); the head is the source final norm and BF16
lm_head. Also reports how the layer routes differ where the dumps overlap.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from axk2_ref import rms_norm
from axk2_source import Source


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zero', type=Path, default=Path('/workspace/axk2/stats-maskzero'))
    p.add_argument('--inf', type=Path, default=Path('/workspace/axk2/stats'))
    p.add_argument('--manifest', type=Path, default=Path('/workspace/axk2/calib/frozen/heldout-81681f74_manifest.json'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = 'cuda'
    w = Source().load(['model.norm.weight', 'lm_head.weight'], dev)
    names = [s['domain'] for s in json.loads(a.manifest.read_text())['samples']]
    res = {'cases': {}, 'layers': {}}
    nll = {}
    for tag, root in (('mask_zero_transformers', a.zero), ('mask_neg_inf_reference', a.inf)):
        f = load_file(str(root / 'heldout' / 'final.safetensors'))
        ids, off = f['input_ids'].long().to(dev), f['offsets'].tolist()
        h = f['final_h'].to(dev)
        per = []
        for i in range(len(off) - 1):
            st, en = off[i], off[i + 1]
            lg = F.linear(rms_norm(h[st:en], w['model.norm.weight']), w['lm_head.weight']).float()
            lp = F.log_softmax(lg, -1)
            per.append(-lp[:-1].gather(1, ids[st + 1:en, None])[:, 0])
        nll[tag] = per
    tot = {k: torch.cat(v) for k, v in nll.items()}
    for i, name in enumerate(names):
        res['cases'][f'{i:02d}-{name}'] = {k: float(v[i].mean()) for k, v in nll.items()}
    res['all'] = {k: float(v.mean()) for k, v in tot.items()}
    res['all']['ppl'] = {k: float(torch.exp(v.mean())) for k, v in tot.items()}
    for L in (24, 34, 37, 41, 48, 60):
        z = load_file(str(a.zero / 'heldout' / f'layer-{L:02d}.safetensors'))['topk_idx'].long()
        n = load_file(str(a.inf / 'heldout' / f'layer-{L:02d}.safetensors'))['topk_idx'].long()
        same = (torch.sort(z, -1)[0] == torch.sort(n, -1)[0]).all(-1).float().mean().item()
        res['layers'][L] = {'route_set_identical': same, 'experts_used_zero': int(z.unique().numel()),
                            'experts_used_inf': int(n.unique().numel())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({'all': res['all'], 'layers': res['layers']}, indent=1))


if __name__ == '__main__':
    main()
