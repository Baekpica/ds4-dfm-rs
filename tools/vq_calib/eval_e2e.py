#!/usr/bin/env python3
"""Teacher-forced FP8-teacher vs release comparison for A.X-K2 VQ, layer streamed.

The teacher is the collector's held-out pass (stats/heldout/final.safetensors:
the last hidden state of every held-out token under the FP8 source) put through
the source final norm and BF16 lm_head. The student runs the same tokens
through every layer with weights read from GGUF parts by this file's own
reader (independent of axk2_vq_format.py): GVQ v2 experts restored from their
bytes, Q4_K / Q8_0 dequantized from their blocks, native tensors as stored.
`--experts teacher` or `--nonexpert teacher` swap one side back to the FP8
source to isolate the two quantization stages.

Reports per case: KL(teacher || student), top-1 agreement, NLL of the next
token for both, and per-layer hidden cosine is not available here (the
teacher's per-layer states are not stored); routing agreement per layer is
measured against the teacher routes in stats/heldout/layer-LL.safetensors.
"""
import argparse
import json
import mmap
from pathlib import Path
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from axk2_ref import BF, Layer, Rope, TeacherWeights, layer_names, rms_norm

ROOT = '/models/A.X-K2'
Q8_0, Q4_K, F32, F16, BF16, BOOK, VQ = 8, 12, 0, 1, 30, 0x47564342, 0x47565138


class Gguf:
    """Minimal independent GGUF v3 directory reader (u32/bool/string metadata only)."""
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        m, pos = self.m, 0

        def take(fmt):
            nonlocal pos
            v = struct.unpack_from('<' + fmt, m, pos)
            pos += struct.calcsize('<' + fmt)
            return v

        def string():
            nonlocal pos
            n, = take('Q')
            s = m[pos:pos + n].decode()
            pos += n
            return s
        magic, ver, nt, nkv = take('4sIQQ')
        assert magic == b'GGUF' and ver == 3
        self.meta = {}
        for _ in range(nkv):
            k = string()
            t, = take('I')
            self.meta[k] = string() if t == 8 else take('I' if t == 4 else 'B')[0]
        self.t = {}
        for _ in range(nt):
            name = string()
            nd, = take('I')
            dims = take('Q' * nd)
            typ, off = take('IQ')
            self.t[name] = (typ, tuple(reversed(dims)), off)
        align = self.meta.get('general.alignment', 32)
        self.base = (pos + align - 1) // align * align

    def raw(self, name, nbytes):
        typ, shape, off = self.t[name]
        a = self.base + off
        return np.frombuffer(self.m, np.uint8, nbytes, a)


def dequant_q4k(raw, shape, dev):
    p = torch.from_numpy(raw.copy()).to(dev).reshape(-1, 144)
    d = p[:, 0:2].contiguous().view(torch.float16).float()
    dm = p[:, 2:4].contiguous().view(torch.float16).float()
    s = p[:, 4:16].to(torch.int32)
    sc = torch.empty((len(p), 8), dtype=torch.float32, device=dev)
    mn = torch.empty_like(sc)
    for j in range(8):
        if j < 4:
            sc[:, j] = (s[:, j] & 63).float()
            mn[:, j] = (s[:, j + 4] & 63).float()
        else:
            sc[:, j] = ((s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)).float()
            mn[:, j] = ((s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)).float()
    q = p[:, 16:].reshape(-1, 4, 32).to(torch.int32)
    vals = torch.stack([q & 15, q >> 4], dim=2).reshape(-1, 8, 32).float()
    return ((d * sc)[:, :, None] * vals - (dm * mn)[:, :, None]).reshape(shape)


def dequant_q8(raw, shape, dev):
    p = torch.from_numpy(raw.copy()).to(dev).reshape(-1, 34)
    d = p[:, 0:2].contiguous().view(torch.float16).float()
    q = p[:, 2:].contiguous().view(torch.int8).float()
    return (d * q).reshape(shape)


def tensor_bytes(typ, shape):
    n = int(np.prod(shape))
    return {Q4_K: n // 256 * 144, Q8_0: n // 32 * 34, F32: 4 * n, F16: 2 * n, BF16: 2 * n, BOOK: n}[typ]


def decode_gvq2(raw, book, dev):
    """GVQ v2 matrix bytes + E4M3 book [K, 8] (F32 on dev) -> F32 [R, C] on dev."""
    h = struct.unpack_from('<4sHBBIIIIIIQQQQ32s32s', raw)
    magic, ver, bits, dim, R, C, K = h[:7]
    assert magic == b'GVQ1' and ver == 2 and dim == 8 and K == 1 << bits and bits in (9, 10)
    g_off, m_off, p_off = h[10], h[11], h[12]
    n = R * C // 8
    buf = torch.from_numpy(np.asarray(raw).copy()).to(dev)
    gains = buf[g_off:g_off + 2 * R].view(torch.float16).float()
    idx = buf[m_off:m_off + n].long()
    plane = buf[p_off:p_off + n * (bits - 8) // 8].long()
    if bits == 9:
        hi = (plane[:, None] >> torch.arange(8, device=dev)) & 1
    else:
        hi = (plane[:, None] >> (2 * torch.arange(4, device=dev))) & 3
    idx = idx | (hi.reshape(-1)[:n] << 8)
    return book[idx].reshape(R, C) * gains[:, None]


class StudentWeights:
    def __init__(self, parts, teacher, experts='parts', nonexpert='parts', dev='cuda'):
        self.parts, self.teacher, self.dev = (Path(parts) if parts else None), teacher, dev
        self.experts_from, self.nonexpert_from = experts, nonexpert
        self.cfg = teacher.cfg
        self.body = Gguf(self.parts / 'body-nonexpert.gguf') if nonexpert == 'parts' else None

    def native(self, name):
        typ, shape, _ = self.body.t[name]
        raw = self.body.raw(name, tensor_bytes(typ, shape))
        if typ == Q4_K:
            return dequant_q4k(raw, shape, self.dev).to(BF)
        if typ == Q8_0:
            return dequant_q8(raw, shape, self.dev).to(BF)
        dt = {F32: torch.float32, F16: torch.float16, BF16: torch.bfloat16}[typ]
        t = torch.from_numpy(raw.copy()).view(dt).reshape(shape).to(self.dev)
        return t if typ == F32 else t.to(BF)

    def layer(self, L, host=None):
        if self.nonexpert_from == 'parts':
            p = f'model.layers.{L}.'
            w = {n[len(p):]: self.native(n) for n in layer_names(self.cfg, L)}
            if L >= self.cfg['first_k_dense_replace'] and self.experts_from == 'teacher':
                w['experts'] = self.teacher.layer(L, host)['experts']
        else:
            w = self.teacher.layer(L, host)
        if L >= self.cfg['first_k_dense_replace'] and self.experts_from == 'parts':
            g = Gguf(self.parts / f'layer-{L:02d}-experts.gguf')
            links = json.loads(g.meta['axk2_vq.codebooks'])
            books = {}
            for bid, bname in links.items():
                btyp, bshape, _ = g.t[bname]
                b = torch.from_numpy(g.raw(bname, tensor_bytes(btyp, bshape)).copy()).view(torch.float8_e4m3fn)
                books[int(bid)] = b.reshape(-1, 8).float().to(self.dev)

            def expert(e, g=g, books=books, L=L):
                mats = []
                for proj in ('gate_proj', 'up_proj', 'down_proj'):
                    name = f'model.layers.{L}.mlp.experts.{e}.{proj}.weight'
                    typ, shape, off = g.t[name]
                    hdr = struct.unpack_from('<4sHBBIIIIIIQQQQ32s32s', g.m, g.base + off)
                    mats.append(decode_gvq2(g.raw(name, hdr[13]), books[hdr[8]], self.dev).to(BF))
                return torch.cat([mats[0], mats[1]], 0), mats[2]
            w['experts'] = expert
        return w


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=ROOT, help='A.X-K2 checkpoint directory')
    p.add_argument('--parts', type=Path)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    p.add_argument('--experts', choices=['parts', 'teacher'], default='parts')
    p.add_argument('--nonexpert', choices=['parts', 'teacher'], default='parts')
    p.add_argument('--manifest', type=Path, default=Path('/workspace/axk2/calib/heldout_manifest.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--goldens', action='store_true')
    a = p.parse_args()
    dev = 'cuda'
    teacher = TeacherWeights(a.source, dev)
    cfg = teacher.cfg
    rope = Rope(a.source, dev)
    fin = load_file(str(a.stats / 'heldout' / 'final.safetensors'))
    ids, off = fin['input_ids'].long(), fin['offsets'].tolist()
    seqs = [(off[i], off[i + 1] - off[i]) for i in range(len(off) - 1)]
    student = StudentWeights(a.parts, teacher, a.experts, a.nonexpert, dev)
    table = teacher.src.load(['model.embed_tokens.weight'], dev)['model.embed_tokens.weight']
    h = table[ids.to(dev)]
    del table
    per_layer = []
    t0 = time.monotonic()
    for L in range(cfg['num_hidden_layers']):
        w = student.layer(L)
        layer = Layer(cfg, L, w, rope)
        h, ffn_in, routes = layer.forward(h, seqs, keep_ffn_in=True)
        row = {'layer': L, 'seconds': time.monotonic() - t0}
        tf = a.stats / 'heldout' / f'layer-{L:02d}.safetensors'
        dump = load_file(str(tf)) if tf.exists() else {}
        if 'ffn_in' in dump:
            ref = dump['ffn_in'].to(dev).float()
            row['ffn_in_cosine'] = F.cosine_similarity(ffn_in.float(), ref, dim=-1).mean().item()
            row['ffn_in_rel_err'] = ((ffn_in.float() - ref).pow(2).sum() / ref.pow(2).sum()).item()
            del ref
        del ffn_in
        if routes is not None and 'topk_idx' in dump:
            tr = dump['topk_idx'].to(dev).long()
            sa = torch.sort(routes[0], -1)[0]
            sb = torch.sort(tr, -1)[0]
            row['route_set_exact'] = (sa == sb).all(-1).float().mean().item()
            row['route_overlap'] = (torch.stack([(routes[0][:, None, :] == tr[:, k:k + 1, None]).any(-1)
                                                  for k in range(tr.shape[1])], -1).float().mean().item())
        per_layer.append(row)
        print(json.dumps(row), flush=True)
        del w, layer
        torch.cuda.empty_cache()
    # heads
    src_norm = teacher.src.load(['model.norm.weight', 'lm_head.weight'], dev)
    if a.nonexpert == 'parts':
        s_norm, s_head = student.native('model.norm.weight'), student.native('lm_head.weight')
    else:
        s_norm, s_head = src_norm['model.norm.weight'], src_norm['lm_head.weight']
    ht = fin['final_h'].to(dev)
    names = json.loads(a.manifest.read_text())['samples'] if a.manifest.exists() else [{} for _ in seqs]
    res = {'per_layer': per_layer, 'cases': {}, 'config': {'experts': a.experts, 'nonexpert': a.nonexpert,
                                                           'parts': str(a.parts)}}
    tot = {'n': 0, 'kl': 0., 'top1': 0, 'nt': 0., 'ns': 0., 'nn': 0}
    goldens = {}
    by_domain = {}
    def score(st, n, chunk=2048):
        # Full-vocabulary logits per position, in chunks (an 11K-token case is 7 GB per F32 tensor).
        kl, top1, nll_t, nll_s, kt_i, kt_v, ks_i, ks_v = [], [], [], [], [], [], [], []
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            lt = F.linear(rms_norm(ht[st + c0:st + c1], src_norm['model.norm.weight']), src_norm['lm_head.weight']).float()
            ls = F.linear(rms_norm(h[st + c0:st + c1], s_norm), s_head).float()
            pt, ps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
            kl.append((pt.exp() * (pt - ps)).sum(-1))
            top1.append(lt.argmax(-1) == ls.argmax(-1))
            last = min(c1, n - 1)                         # next-token targets exist up to position n-2
            if last > c0:
                tgt = ids[st + c0 + 1:st + last + 1].to(dev)
                nll_t.append(-pt[:last - c0].gather(1, tgt[:, None])[:, 0])
                nll_s.append(-ps[:last - c0].gather(1, tgt[:, None])[:, 0])
            if a.goldens:
                kt, ks = torch.topk(lt, 8, -1), torch.topk(ls, 8, -1)
                kt_i.append(kt.indices.cpu()); kt_v.append(kt.values.cpu())
                ks_i.append(ks.indices.cpu()); ks_v.append(ks.values.cpu())
            del lt, ls, pt, ps
        g = (torch.cat(kt_i), torch.cat(kt_v), torch.cat(ks_i), torch.cat(ks_v)) if a.goldens else None
        return torch.cat(kl), torch.cat(top1), torch.cat(nll_t), torch.cat(nll_s), g

    for i, (st, n) in enumerate(seqs):
        kl, top1, nll_t, nll_s, g = score(st, n)
        meta = names[i] if i < len(names) else {}
        key = f"{i:02d}-{meta.get('domain', 'case')}"
        res['cases'][key] = {'tokens': n, 'domain': meta.get('domain'), 'kl_mean': kl.mean().item(),
                             'top1_agreement': top1.float().mean().item(), 'nll_teacher': nll_t.mean().item(),
                             'nll_student': nll_s.mean().item(), 'ppl_teacher': float(np.exp(nll_t.mean().item())),
                             'ppl_student': float(np.exp(nll_s.mean().item()))}
        d = by_domain.setdefault(meta.get('domain', 'case'), {'n': 0, 'kl': 0., 'top1': 0})
        d['n'] += n
        d['kl'] += kl.sum().item()
        d['top1'] += int(top1.sum())
        tot['n'] += n
        tot['kl'] += kl.sum().item()
        tot['top1'] += int(top1.sum())
        tot['nt'] += nll_t.sum().item()
        tot['ns'] += nll_s.sum().item()
        tot['nn'] += n - 1
        if a.goldens:
            goldens[key] = {'ids': ids[st:st + n].tolist(), 'teacher_top8_ids': g[0].tolist(),
                            'teacher_top8_logits': [[round(v, 4) for v in r] for r in g[1].tolist()],
                            'student_top8_ids': g[2].tolist(),
                            'student_top8_logits': [[round(v, 4) for v in r] for r in g[3].tolist()]}
        print(json.dumps({'case': key, **res['cases'][key]}), flush=True)
    res['all'] = {'positions': tot['n'], 'kl_mean': tot['kl'] / tot['n'], 'top1_agreement': tot['top1'] / tot['n'],
                  'nll_teacher': tot['nt'] / tot['nn'], 'nll_student': tot['ns'] / tot['nn'],
                  'seconds': time.monotonic() - t0}
    res['by_domain'] = {k: {'tokens': v['n'], 'kl_mean': v['kl'] / v['n'], 'top1': v['top1'] / v['n']}
                        for k, v in by_domain.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    if a.goldens:
        (a.out / 'goldens.json').write_text(json.dumps(goldens) + '\n')
    print(json.dumps({'all': res['all']}), flush=True)


if __name__ == '__main__':
    main()
