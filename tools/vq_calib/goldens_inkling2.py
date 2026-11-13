#!/usr/bin/env python3
"""Reference outputs of the released bytes for the ds4-dfm-rs `inkling2_vq` port: per-layer hidden states,
routes and final logits of a few prompts, computed by inkling2_ref.py with the GVQ1 expert parts and the
Q4_K body exactly as stored (every weight restored to BF16 as the runtime restores it), with K/V in BF16
and in the NVFP4 contract. All prompts and both K/V formats run layer-synchronously in one pass.

    goldens_inkling2.py --parts /work/inkling2/parts --out /work/inkling2/goldens
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import inkling2_ref as R
from inkling2_eval import Body, GvqExperts, q4k_dequant


def prompts(tok):
    chat = tok.apply_chat_template([{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}],
                                   tokenize=True, reasoning_effort='none')
    chat = chat['input_ids'] if hasattr(chat, 'keys') else chat
    chat = [int(t) for t in (chat[0] if chat and isinstance(chat[0], list) else chat)]
    out = {'raw_paris': ([200028] + tok.encode('The capital of France is'), None),
           'chat_capital': (list(chat) + [200001], None)}
    probe = Path('/work/inkling2/test/vision_probe.pt')
    if probe.exists():
        p = torch.load(probe)
        out['image_red_blue'] = (p['ids'].tolist() + [200001], p['features'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    dev = torch.device('cuda')
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(R.SRC))
    src = R.Source()
    glob = R.Globals(src, dev, head=False)
    body = Body(a.parts / 'body-nonexpert.gguf', dev)
    states = []
    for name, (ids, feats) in prompts(tok).items():
        for kv in ('bf16', 'nvfp4'):
            ids_t = torch.tensor(ids, device=dev)
            h = glob.embed_ids(ids_t)
            if feats is not None:
                h[ids_t == 200054] = feats.to(dev)
            states.append({'name': name, 'kv': kv, 'ids': ids, 'h': h, 'blocks': [], 'layers': [], 'routes': [],
                           'pos': torch.arange(len(ids), device=dev)})
    for l in range(R.L):
        host = body.override(src.read(src.layer_names(l)))
        lw = R.LayerWeights(l, host, dev, student=GvqExperts(a.parts, l, dev))
        for st in states:
            resid, mb = R.attn_part(lw, st['h'], st['blocks'], st['pos'], st['kv'])
            x = R.mlp_in(lw, resid, mb)
            y, (idx, _) = R.moe(lw, x)
            new = R.finish_layer(lw, resid, y)
            if R.is_boundary(l):
                st['blocks'] = st['blocks'] + [st['h']]
            st['h'] = new
            st['layers'].append(new.cpu())
            st['routes'].append(torch.sort(idx, -1).values.to(torch.int16).cpu())
        del lw, host
        torch.cuda.empty_cache()
    e = body.r.entries['model.llm.unembed.weight']
    raw = body.r.raw('model.llm.unembed.weight')
    head = q4k_dequant(raw, e['shape'], dev).to(torch.bfloat16)
    raw.release()
    index = {}
    for st in states:
        fh = glob.final_hidden(st['blocks'], st['h'])
        z = R.softcap(torch.nn.functional.linear(fh, head).float(), R.LOGIT_CAP)[:, :R.VOCAB_REAL]
        top = torch.topk(z, 32, -1)
        f = a.out / f"{st['name']}.kv-{st['kv']}.safetensors"
        save_file({'input_ids': torch.tensor(st['ids'], dtype=torch.int32), 'layer_out': torch.stack(st['layers']),
                   'routes': torch.stack(st['routes']), 'final_hidden': fh.cpu(), 'logits_top32_ids': top.indices.int().cpu(),
                   'logits_top32': top.values.cpu(), 'logits_last': z[-1].cpu()}, str(f))
        index[f.name] = {'prompt': st['name'], 'kv': st['kv'], 'tokens': len(st['ids']),
                         'next_token': tok.decode([int(z[-1].argmax())]), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
        print(json.dumps(index[f.name], ensure_ascii=False), flush=True)
    (a.out / 'index.json').write_text(json.dumps({
        'recipe': 'release bytes: GVQ1 12-bit experts + Q4_K body; every weight restored to BF16 (bf16(gain * codeword) for '
                  'experts, Q4_K dequantised exactly); attention, MoE and residual arithmetic as inkling2_ref.py',
        'layer_out': 'hidden state after each of the 48 layers [48, T, 4608] BF16',
        'routes': 'sorted top-8 expert ids per layer and token [48, T, 8]',
        'files': index}, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
