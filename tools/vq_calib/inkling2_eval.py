#!/usr/bin/env python3
"""Inkling-2-Small teacher-forced evaluation: the MXFP8 original against converted variants.

    teacher   inkling2_ref with the source experts; saves every position's final normalised hidden state
              (BF16) and every layer's routes, so a variant is scored against the teacher's logits
              (recomputed from the saved states and the source head) without rerunning the teacher.
    variant   the same forward with any of: routed experts from GVQ1 expert parts (--experts DIR), K/V
              stored as FP8 or NVFP4 in the global layers (--kv-fmt), the non-expert body from a Q4_K part
              (--body PART: dequantised exactly), one layer swapped (--only-layers).
    score     KL(teacher || variant), same top-1, NLL of the actual next token: all positions, generated
              positions (raw text, and assistant tokens of chat rows), per domain and per case; routed
              expert-set overlap per layer.

Sequences are split over the visible GPUs (one process per GPU, `--rank/--world`), layer-synchronously:
each layer is loaded once per process and applied to all its sequences.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import inkling2_ref as R

MSG_MODEL, END_MSG, END_SAMPLING, IMAGE = 200001, 200010, 200006, 200054


# ------------------------------------------------------------------------------------ data
def load_set(path, features=None):
    t = load_file(str(path))
    ids, off = t['input_ids'].long(), t['offsets'].long()
    feats = foff = None
    if features is not None:
        f = load_file(str(features))
        feats, foff = f['features'], f['feature_offsets'].long()
    seqs = []
    for i in range(len(off) - 1):
        s = {'i': i, 'ids': ids[off[i]:off[i + 1]]}
        if feats is not None and foff[i + 1] > foff[i]:
            s['features'] = feats[foff[i]:foff[i + 1]]
        seqs.append(s)
    return seqs


def generated_mask(ids, chat):
    """Targets (positions 1..T-1) inside assistant messages for chat rows; all targets for raw rows."""
    t = ids.numel()
    if not chat:
        return torch.ones(t - 1, dtype=torch.bool)
    inside = torch.zeros(t, dtype=torch.bool)
    on = False
    for j, tok in enumerate(ids.tolist()):
        if tok == MSG_MODEL:
            on = True
            continue
        if on:
            inside[j] = True
            if tok in (END_MSG, END_SAMPLING):
                on = tok == END_MSG and j + 1 < t and ids[j + 1] == MSG_MODEL
                if tok == END_SAMPLING:
                    on = False
    return inside[1:]


# --------------------------------------------------------------------------------- students
class GvqExperts:
    """Routed experts of one layer from `layer-LL-experts.gguf` (GVQ1 12-bit), decoded on the device as the
    runtime restores them: bf16(f32(gain) * f32(codeword))."""

    def __init__(self, parts_dir, layer, dev):
        import inkling2_vq_format as fmt
        self.fmt, self.dev = fmt, dev
        self.reader = fmt.PartReader(Path(parts_dir) / f'layer-{layer:02d}-experts.gguf')
        self.layer = layer
        book = bytes(self.reader.raw(fmt.book_name(layer)))
        self.cb = torch.from_numpy(np.frombuffer(book, np.uint8).copy()).to(dev).view(torch.float8_e4m3fn).float().reshape(-1, 8)

    def matrix(self, e, p):
        fmt = self.fmt
        name = fmt.matrix_name(self.layer, e, p)
        raw = self.reader.raw(name)
        rows, cols = fmt.SHAPES[p]
        g = fmt.geometry(rows, cols, 4096)
        buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(self.dev)
        raw.release()
        gains = buf[g['gain']:g['gain'] + 2 * rows].view(torch.float16).float()
        low = buf[g['main']:g['main_end']].view(-1, 3).int()
        a = low[:, 0] | ((low[:, 1] & 15) << 8)
        b = (low[:, 1] >> 4) | (low[:, 2] << 4)
        idx = torch.stack([a, b], -1).reshape(rows, cols // 8)
        return (self.cb[idx.long()].reshape(rows, cols) * gains[:, None]).to(torch.bfloat16)

    def __call__(self, e):
        gate = self.matrix(e, 'gate_proj')
        up = self.matrix(e, 'up_proj')
        return torch.cat([gate, up]), self.matrix(e, 'down_proj')


def q4k_dequant(raw, shape, dev):
    """GGML Q4_K blocks (144 bytes per 256 weights) -> F32 [rows, cols] on the device."""
    rows, cols = shape
    b = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).view(-1, 144)
    d = b[:, 0:2].contiguous().view(torch.float16).float()[:, 0]
    dmin = b[:, 2:4].contiguous().view(torch.float16).float()[:, 0]
    s = b[:, 4:16].int()
    q = b[:, 16:144].int()
    sc = torch.empty((b.shape[0], 8), dtype=torch.int32, device=dev)
    mn = torch.empty_like(sc)
    sc[:, :4], mn[:, :4] = s[:, 0:4] & 63, s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    q = q.view(-1, 4, 32)
    lo, hi = q & 15, q >> 4
    vals = torch.stack([lo, hi], 2).reshape(-1, 8, 32).float()                  # sub-blocks 2j (low), 2j+1 (high)
    w = d[:, None, None] * sc[:, :, None].float() * vals - dmin[:, None, None] * mn[:, :, None].float()
    return w.reshape(rows, cols)


class Body:
    """Non-expert tensors from body-nonexpert.gguf: Q4_K dequantised (F32 -> BF16), others as stored."""

    def __init__(self, path, dev):
        import inkling2_vq_format as fmt
        self.r = fmt.PartReader(path)
        self.dev = dev

    def override(self, host):
        out = dict(host)
        for name in host:
            e = self.r.entries.get(name)
            if e is None or e['type_id'] != 12:
                continue
            raw = self.r.raw(name)
            out[name] = q4k_dequant(raw, e['shape'], self.dev).to(torch.bfloat16).cpu()
            raw.release()
        return out


# ----------------------------------------------------------------------------------- forward
def run(source, seqs, dev, kv_fmt='bf16', experts=None, body=None, only_layers=None, keep_routes=True, log=print):
    src = R.Source(source)
    glob = R.Globals(src, dev, head=False)
    states = []
    for s in seqs:
        ids = s['ids'].to(dev)
        h = glob.embed_ids(ids)
        if 'features' in s:
            h[ids == IMAGE] = s['features'].to(dev)
        states.append({'h': h, 'blocks': [], 'pos': torch.arange(ids.numel(), device=dev), 'routes': {}})
    for l in range(R.L):
        t0 = time.monotonic()
        host = src.read(src.layer_names(l))
        if body is not None and (only_layers is None or l in only_layers):
            host = body.override(host)
        student = None
        if experts is not None and (only_layers is None or l in only_layers):
            student = GvqExperts(experts, l, dev)
        lw = R.LayerWeights(l, host, dev, student=student)
        for st in states:
            resid, mb = R.attn_part(lw, st['h'], st['blocks'], st['pos'], kv_fmt)
            x = R.mlp_in(lw, resid, mb)
            y, (idx, _) = R.moe(lw, x)
            if keep_routes:
                st['routes'][l] = torch.sort(idx, -1).values.to(torch.int16).cpu()
            new = R.finish_layer(lw, resid, y)
            if R.is_boundary(l):
                st['blocks'] = st['blocks'] + [st['h']]
            st['h'] = new
        del lw, host, student
        torch.cuda.empty_cache()
        log(json.dumps({'layer': l, 'seconds': round(time.monotonic() - t0, 1)}))
    outs = []
    for st in states:
        outs.append({'final': glob.final_hidden(st['blocks'], st['h']).cpu(), 'routes': st['routes']})
    return outs


# ----------------------------------------------------------------------------------- scoring
def summarize(rows):
    def agg(rs):
        n = sum(r['positions'] for r in rs)
        g = sum(r['generated'] for r in rs)
        out = {'positions': n, 'generated': g,
               'kl': sum(r['kl'] for r in rs) / max(n, 1), 'top1': sum(r['top1'] for r in rs) / max(n, 1),
               'nll_teacher': sum(r['nll_teacher'] for r in rs) / max(n, 1), 'nll': sum(r['nll'] for r in rs) / max(n, 1)}
        if g:
            out.update({'kl_gen': sum(r['kl_gen'] for r in rs) / g, 'top1_gen': sum(r['top1_gen'] for r in rs) / g,
                        'nll_teacher_gen': sum(r['nll_teacher_gen'] for r in rs) / g, 'nll_gen': sum(r['nll_gen'] for r in rs) / g})
        return out
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r['domain'], []).append(r)
    layers = sorted({l for r in rows for l in r['route_overlap']})
    ov = {l: float(np.mean([r['route_overlap'][l] for r in rows if l in r['route_overlap']])) for l in layers}
    return {'all': agg(rows), 'by_domain': {d: agg(rs) for d, rs in sorted(by_dom.items(), key=lambda x: str(x[0]))},
            'route_overlap_by_layer': ov}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['teacher', 'variant', 'merge'])
    ap.add_argument('--source', type=Path, required=True,
                    help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8 at R.SOURCE_REV)')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--features', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--teacher', type=Path, help='variant: the teacher output directory')
    ap.add_argument('--experts', type=Path)
    ap.add_argument('--body', type=Path)
    ap.add_argument('--kv-fmt', default='bf16')
    ap.add_argument('--only-layers', default='')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = load_set(a.data, a.features)
    manifest = json.loads(a.manifest.read_text())['samples'] if a.manifest else None
    if a.mode == 'merge':
        rows = []
        for p in sorted(a.out.glob('rows-rank*.json')):
            rows += json.loads(p.read_text())
        res = summarize(rows)
        (a.out / 'summary.json').write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res['all']))
        return
    mine = [s for k, s in enumerate(seqs) if k % a.world == a.rank]
    dev = torch.device('cuda')
    only = set(int(x) for x in a.only_layers.split(',') if x) or None
    body = Body(a.body, dev) if a.body else None
    log = lambda m: print(m, flush=True)
    outs = run(a.source, mine, dev, a.kv_fmt, a.experts, body, only, log=log)
    if a.mode == 'teacher':
        for s, o in zip(mine, outs):
            save_file({'final': o['final'], **{f'routes.{l}': r for l, r in o['routes'].items()}},
                      str(a.out / f'seq-{s["i"]:05d}.safetensors'))
        return
    teacher = []
    for s in mine:
        t = load_file(str(a.teacher / f'seq-{s["i"]:05d}.safetensors'))
        teacher.append({'final': t['final'], 'routes': {int(k.split('.')[1]): v for k, v in t.items() if k.startswith('routes.')}})
    src = R.Source(a.source)
    head = src.read(['model.llm.unembed.weight'])['model.llm.unembed.weight'].to(dev)
    if body is not None:
        e = body.r.entries.get('model.llm.unembed.weight')
        if e is not None and e['type_id'] == 12:
            raw = body.r.raw('model.llm.unembed.weight')
            head_v = q4k_dequant(raw, e['shape'], dev).to(torch.bfloat16)
            raw.release()
        else:
            head_v = head
    else:
        head_v = head
    # the teacher's logits use the source head, the variant's its own (the Q4_K head with --body)
    rows = _score_heads(mine, teacher, outs, manifest, dev, head, head_v)
    (a.out / f'rows-rank{a.rank}.json').write_text(json.dumps(rows) + '\n')


def _score_heads(seqs, teacher, variant, manifest, dev, head_t, head_v, chunk=1024):
    rows = []
    for s, t, v in zip(seqs, teacher, variant):
        ids = s['ids']
        meta = manifest[s['i']] if manifest else {}
        chat = meta.get('format', 'raw') != 'raw'
        gen = generated_mask(ids, chat).to(dev)
        tgt = ids[1:].to(dev)
        n = ids.numel() - 1
        acc = dict.fromkeys(['kl', 'top1', 'nll_teacher', 'nll', 'kl_gen', 'top1_gen', 'nll_teacher_gen', 'nll_gen'], 0.0)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            zt = R.softcap(F.linear(t['final'][c0:c1].to(dev), head_t).float(), R.LOGIT_CAP)[:, :R.VOCAB_REAL]
            zv = R.softcap(F.linear(v['final'][c0:c1].to(dev), head_v).float(), R.LOGIT_CAP)[:, :R.VOCAB_REAL]
            lt, lv = zt.log_softmax(-1), zv.log_softmax(-1)
            kl = (lt.exp() * (lt - lv)).sum(-1)
            top = (zt.argmax(-1) == zv.argmax(-1)).float()
            nt = -lt.gather(-1, tgt[c0:c1, None])[:, 0]
            nv = -lv.gather(-1, tgt[c0:c1, None])[:, 0]
            g = gen[c0:c1].float()
            for k_, val in (('kl', kl), ('top1', top), ('nll_teacher', nt), ('nll', nv)):
                acc[k_] += float(val.sum())
                acc[k_ + '_gen'] += float((val * g).sum())
        overlap = {}
        for l, rt in t['routes'].items():
            rv = v['routes'].get(l)
            if rv is not None:
                overlap[l] = float((rt[:, :, None] == rv[:, None, :]).any(-1).float().mean())
        rows.append({'i': s['i'], 'case': meta.get('case'), 'domain': meta.get('domain'), 'positions': n,
                     'generated': int(gen.sum()), **acc, 'route_overlap': overlap})
    return rows


if __name__ == '__main__':
    main()
