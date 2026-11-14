#!/usr/bin/env python3
"""inkling2_ref.py against transformers' Inkling2 decoder layers (pinned commit, eager attention, eager
experts), on the same weights and the same inputs.

The inputs are real: a tokenized text through the reference's own embedding and the layers before the one
under test, so the block states and hidden state are those the layer sees in the model. Every compared
layer runs both implementations on identical BF16 inputs; reported are the relative RMS and max abs
differences of the layer output, the share of bit-identical outputs, and the routed-expert agreement.

Usage: inkling2_selftest.py --layers 0,5,6,11 [--tokens 512] [--out selftest.json]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

import inkling2_ref as R


def tf_layer(cfg, l, host, dev):
    from transformers.models.inkling2 import modeling_inkling2 as M
    cls = M.Inkling2FullDecoderLayer if R.is_global(l) else M.Inkling2SlidingDecoderLayer
    layer = cls(cfg, l).to(torch.bfloat16)
    p = f'model.llm.layers.{l}.'
    sd = {}
    ren = {'wq_du': 'q_proj', 'wk_dv': 'k_proj', 'wv_dv': 'v_proj', 'wr_du': 'r_proj', 'wo_ud': 'o_proj',
           'gate_attn': 'gate_proj', 'attn_sink_proj': 'sink_proj'}
    for a, b in ren.items():
        sd[f'self_attn.{b}.weight'] = host[p + f'attn.{a}.weight']
    sd['self_attn.q_l2_clip.scale'] = host[p + 'attn.q_l2_clip.scale']
    sd['self_attn.rel_l2_clip.scale'] = host[p + 'attn.rel_l2_clip.scale']
    sd['self_attn.rel_logits_proj.proj'] = host[p + 'attn.rel_logits_proj.proj']
    if R.is_global(l):
        for kv in ('k', 'v'):
            sd[f'self_attn.{kv}_sconv.static_weight'] = host[p + f'attn.{kv}_sconv.static_w']
            sd[f'self_attn.{kv}_sconv.weight_proj.weight'] = host[p + f'attn.{kv}_sconv.weight_proj']
            sd[f'{kv}_input_layernorm.weight'] = host[p + f'{kv}_input_norm.weight']
        sd['q_input_layernorm.weight'] = host[p + 'q_input_norm.weight']
    else:
        for kv in ('k', 'v'):
            sd[f'self_attn.{kv}_sconv.conv1d.weight'] = host[p + f'attn.{kv}_sconv.weight']
        sd['input_layernorm.weight'] = host[p + 'attn_norm.weight']
    sd['post_attention_convolution.conv1d.weight'] = host[p + 'attn_sconv.weight']
    sd['post_mlp_convolution.conv1d.weight'] = host[p + 'mlp_sconv.weight']
    sd['post_attention_layernorm.weight'] = host[p + 'mlp_norm.weight']
    sd['mlp.gate.weight'] = host[p + 'mlp.gate.weight']
    sd['mlp.gate.e_score_correction_bias'] = host[p + 'mlp.gate.bias']
    w13, s13 = host[p + 'mlp.experts.w13_weight'], host[p + 'mlp.experts.w13_weight_scale']
    w2, s2 = host[p + 'mlp.experts.w2_weight'], host[p + 'mlp.experts.w2_weight_scale']
    gu = torch.empty((R.E, 2 * R.FF, R.H), dtype=torch.bfloat16, device=dev)
    dn = torch.empty((R.E, R.H, R.FF), dtype=torch.bfloat16, device=dev)
    for e in range(R.E):
        w = R.dequant_mx(w13[e].to(dev), s13[e].to(dev))
        # transformers' Interleave(dim=1): row 2i -> gate i, row 2i + 1 -> up i
        gu[e, :R.FF], gu[e, R.FF:] = w[0::2], w[1::2]
        dn[e] = R.dequant_mx(w2[e].to(dev), s2[e].to(dev))
    sd['mlp.experts.gate_up_proj'] = gu
    sd['mlp.experts.down_proj'] = dn
    sd['mlp_input.query'] = host[f'model.llm.attn_res.mlp_queries.{l}']
    if l > 0:
        sd['attention_input.query'] = host[f'model.llm.attn_res.attn_queries.{l - 1}']
    # from_pretrained keeps these modules in FP32 (_keep_in_fp32_modules_strict): convert them before
    # loading so F32 source values (attn-res queries, router bias) are never rounded to BF16
    for name in ('post_attention_convolution', 'post_mlp_convolution', 'attention_input', 'mlp_input'):
        mod = getattr(layer, name, None)
        if mod is not None:
            mod.float()
    if not R.is_global(l):
        layer.self_attn.k_sconv.conv1d.float()
        layer.self_attn.v_sconv.conv1d.float()
    layer.mlp.gate.float()
    missing, unexpected = layer.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    if unexpected or [m for m in missing if 'rotary' not in m]:
        raise RuntimeError(f'state dict mismatch: missing {missing} unexpected {unexpected}')
    return layer.to(dev).eval()


def masks(t, dev):
    i = torch.arange(t, device=dev)
    d = i[:, None] - i[None, :]
    lo = torch.finfo(torch.bfloat16).min
    full = torch.where(d >= 0, 0.0, lo).to(torch.bfloat16)[None, None]
    slide = torch.where((d >= 0) & (d < R.WINDOW), 0.0, lo).to(torch.bfloat16)[None, None]
    return full, slide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True,
                    help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8 at R.SOURCE_REV)')
    ap.add_argument('--layers', default='0,5,6,11')
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda')
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    from transformers import AutoTokenizer
    src = R.Source(a.source)
    cfg = Inkling2TextConfig(**{k: v for k, v in src.config['text_config'].items() if k != 'torch_dtype'})
    cfg._attn_implementation = 'eager'
    cfg._experts_implementation = 'eager'
    tok = AutoTokenizer.from_pretrained(str(a.source))
    text = (Path(__file__).resolve().parents[2] / 'README.md').read_text()[:20000]
    ids = torch.tensor(tok.encode(text)[:a.tokens], device=dev)
    t = ids.numel()
    glob = R.Globals(src, dev, head=False)
    want = sorted(int(x) for x in a.layers.split(','))
    pos = torch.arange(t, device=dev)
    hidden, blocks = glob.embed_ids(ids), []
    full, slide = masks(t, dev)
    rows = []
    for l in range(max(want) + 1):
        host = src.read(src.layer_names(l))
        lw = R.LayerWeights(l, host, dev)
        resid, mblocks = R.attn_part(lw, hidden, blocks, pos)
        x = R.mlp_in(lw, resid, mblocks)
        y, (idx, w) = R.moe(lw, x)
        new = R.finish_layer(lw, resid, y)
        if l in want:
            t0 = time.time()
            ref = tf_layer(cfg, l, host, dev)
            bs = torch.stack(blocks, dim=-2)[None] if blocks else hidden.new_zeros(1, t, 0, R.H)
            cap = {}
            def grab(module, inputs, output):
                cap['idx'] = output[2]
            hook = ref.mlp.gate.register_forward_hook(grab)
            with torch.no_grad():
                got = ref(hidden[None], bs, attention_mask=full if R.is_global(l) else slide, conv_mask=None,
                          past_key_values=None, position_ids=pos[None])[0]
            hook.remove()
            d = (got.float() - new.float())
            same_routes = (torch.sort(cap['idx'], -1).values == torch.sort(idx, -1).values).all(-1).float().mean()
            row = {'layer': l, 'global': R.is_global(l), 'boundary': R.is_boundary(l), 'tokens': t,
                   'rel_rms': float(d.square().mean().sqrt() / new.float().square().mean().sqrt()),
                   'max_abs': float(d.abs().max()), 'out_rms': float(new.float().square().mean().sqrt()),
                   'bit_identical': float((got == new).float().mean()),
                   'same_expert_sets': float(same_routes), 'seconds': round(time.time() - t0, 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del ref
        if R.is_boundary(l):
            blocks = blocks + [hidden]
        hidden = new
        del lw, host
        torch.cuda.empty_cache()
    res = {'transformers_rev': R.TRANSFORMERS_REV, 'source_rev': R.SOURCE_REV, 'rows': rows}
    if a.out:
        open(a.out, 'w').write(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
