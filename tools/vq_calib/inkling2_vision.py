#!/usr/bin/env python3
"""Inkling-2-Small vision features for calibration and evaluation: transformers' Inkling2VisionModel
(pinned commit 109a7d19, BF16, the checkpoint's own weights) over processor outputs.

Inkling2Model.forward replaces the normalised text embedding of every image placeholder (id 200054) by
one merger output row, in order; this module computes those rows so the layer-streamed reference can
inject them (`features` [n_image_tokens, 4608] BF16 per sequence).

    inkling2_vision.py --source $SRC --images $W/calib/images.safetensors --out $W/calib/image_features.safetensors
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import inkling2_ref as R

IMAGE_TOKEN = 200054
RENAMES = [('model.visual.backbone.embeddings.position_embedding', 'embeddings.position_embedding.weight'),
           ('model.visual.backbone.embeddings.', 'embeddings.'),
           ('model.visual.backbone.ln_pre.', 'pre_layernorm.'),
           ('model.visual.proj_in.', 'merger.linear_1.'),
           ('model.visual.proj_out.', 'merger.linear_2.'),
           ('model.visual.backbone.layers.', 'layers.')]
LAYER_RENAMES = [('.ln_1.', '.layernorm_before.'), ('.ln_2.', '.layernorm_after.'), ('.mlp.c_fc.', '.mlp.fc1.'),
                 ('.mlp.c_proj.', '.mlp.fc2.'), ('.ls_1.gamma', '.lambda_1'), ('.ls_2.gamma', '.lambda_2')]


def target_name(name):
    for a, b in RENAMES:
        if name.startswith(a):
            name = b + name[len(a):]
            break
    else:
        raise ValueError(name)
    if name.startswith('layers.'):
        for a, b in LAYER_RENAMES:
            name = name.replace(a, b)
    return name


def load_vision(src, dev, attn='sdpa'):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    vc = dict(src.config['vision_config'])
    cfg = Inkling2VisionConfig(**{k: v for k, v in vc.items() if k != 'temporal_patch_size'})
    cfg._attn_implementation = attn
    model = Inkling2VisionModel(cfg).to(torch.bfloat16)
    names = sorted(k for k in src.map if k.startswith('model.visual.'))
    host = src.read(names)
    sd = {target_name(n): t for n, t in host.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if 'inv_freq' not in m]
    if missing or unexpected:
        raise RuntimeError(f'vision state dict: missing {missing} unexpected {unexpected}')
    return model.to(dev).eval(), {n: hashlib.sha256(host[n].view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                                 for n in names[:3]}


@torch.no_grad()
def features(model, pixel_values, grid_thw):
    """-> merger output rows [sum(t*h*w), 4608] BF16 (one per image token, image order)."""
    out = model(pixel_values=pixel_values.to(next(model.parameters()).device), image_grid_thw=grid_thw.cpu())
    return out.pooler_output.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True,
                    help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8 at R.SOURCE_REV)')
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda')
    src = R.Source(a.source)
    model, _ = load_vision(src, dev)
    t = load_file(str(a.images))
    ids, off = t['input_ids'].long(), t['offsets'].long()
    pv, grid = t['pixel_values'], t['image_grid_thw'].long()
    per_seq = t['image_counts'].long()                         # images per sequence
    tok_per_image = grid.prod(-1)
    img_off = torch.cat([torch.zeros(1, dtype=torch.long), per_seq.cumsum(0)])
    pv_off = torch.cat([torch.zeros(1, dtype=torch.long), tok_per_image.cumsum(0)])
    feats, foff = [], [0]
    for s in range(len(off) - 1):
        i0, i1 = int(img_off[s]), int(img_off[s + 1])
        n_place = int((ids[off[s]:off[s + 1]] == IMAGE_TOKEN).sum())
        if i1 == i0:
            if n_place:
                raise ValueError(f'sequence {s}: placeholders without images')
            foff.append(foff[-1])
            continue
        f = features(model, pv[int(pv_off[i0]):int(pv_off[i1])], grid[i0:i1])
        if f.shape[0] != n_place:
            raise ValueError(f'sequence {s}: {f.shape[0]} features for {n_place} placeholders')
        feats.append(f.cpu())
        foff.append(foff[-1] + f.shape[0])
    out = {'features': torch.cat(feats) if feats else torch.zeros((0, R.H), dtype=torch.bfloat16),
           'feature_offsets': torch.tensor(foff, dtype=torch.int64)}
    save_file(out, str(a.out), metadata={'images_sha256': hashlib.sha256(a.images.read_bytes()).hexdigest(),
                                         'source_revision': R.SOURCE_REV, 'transformers_revision': R.TRANSFORMERS_REV})
    print(json.dumps({'sequences': len(off) - 1, 'image_tokens': int(out['features'].shape[0])}))


if __name__ == '__main__':
    main()
