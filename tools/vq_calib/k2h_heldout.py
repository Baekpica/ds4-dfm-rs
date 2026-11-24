#!/usr/bin/env python3
"""K2-Horizon-375B-A23B VQ: held-out references for a host that does not have the original checkpoint.

The conversion host saved the original's final residual streams on 64 held-out sequences
(`heldout/teacher.safetensors`). Turning them into next-token scores needs the original's final norm and
output head, which the release stores only as Q8_0. This tool carries them over and scores a runtime.

    original  (conversion host) the original's `model.norm.weight` and `lm_head.weight` as one BF16 file; its
              top-64 next-token log-probabilities, its argmax and the log-probability of every true next
              token; an index of the sequences (chunk, domain, loss)
    student   (any GPU with the release parts) the reference student's final residual streams on the same
              sequences, per K/V variant: k2h_eval's forward of the release bytes
    score     (anywhere) a candidate against a reference: KL(reference || candidate), same top-1, NLL.
              Reference: `original` (teacher file + original head) or a student file (through the release's
              own norm and Q8_0 head). Candidate: a directory of next-token logits, `seq-NNNN.npy`
              [positions <= 512, 250624] float16/float32 for held-out sequence NNNN (row p: the logits after
              tokens 0..p); ds4-bench frontier dumps, `seq-NNNN/frontier_FFFFFF.logits.json` (the logits after
              F tokens, compared at position F - 1); or a file of final residual streams (through the
              release's norm and Q8_0 head).
    texts     the held-out sequences decoded to text files, with whether each re-tokenizes to the same ids
              (the sequences are 512-token chunks without BOS; a runtime fed text must reproduce the ids)

    k2h_heldout.py original --source $SRC --teacher heldout/teacher.safetensors --calib tokens.npz --out heldout
    k2h_heldout.py student --parts $REL --teacher heldout/teacher.safetensors --out heldout
    k2h_heldout.py score --reference original --teacher heldout/teacher.safetensors \\
        --original-head heldout/original-head.safetensors --logits engine-logits/ --index heldout/heldout-index.json
    k2h_heldout.py score --reference heldout/student-final.kv-nvfp4.safetensors --release $REL --logits engine-logits/
    k2h_heldout.py score --reference heldout/student-final.kv-nvfp4.safetensors --release $REL --frontiers bench-dumps/
    k2h_heldout.py texts --tokenizer $REL/tokenizer.json --teacher heldout/teacher.safetensors --out heldout

Scores run in k2h_ref's arithmetic (RMSNorm in F32, the head as a BF16 GEMM, log-softmax in F32).
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import k2h_ref as R

DOMAINS = ['coding_tool', 'math_science', 'general_reasoning', 'long_retrieval', 'general_text']
TOPK = 64


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def logits_of(hidden, norm, head):
    """Final residual stream [T, H] BF16 -> log-probabilities [T, VOCAB] F32."""
    return torch.log_softmax(F.linear(R.rmsnorm(hidden, norm), head).float(), -1)


# ------------------------------------------------------------------------------------ original
def cmd_original(a):
    dev = torch.device(a.device)
    src = R.Source(a.source)
    norm, head = src.tensor('model.norm.weight'), src.tensor('lm_head.weight')
    a.out.mkdir(parents=True, exist_ok=True)
    hp = a.out / 'original-head.safetensors'
    save_file({'model.norm.weight': norm.contiguous(), 'lm_head.weight': head.contiguous()}, str(hp),
              metadata={'source_repo': R.SOURCE_REPO, 'source_revision': R.SOURCE_REV, 'dtype': 'BF16, bytes as stored'})
    teacher = load_file(str(a.teacher))
    ids, th = teacher['input_ids'].to(torch.int64), teacher['final_hidden']
    n, T = ids.shape
    norm_d, head_d = norm.to(dev), head.to(dev)
    top_ids = torch.empty((n, T, TOPK), dtype=torch.int32)
    top_lp = torch.empty((n, T, TOPK), dtype=torch.float32)
    nxt = torch.empty((n, T - 1), dtype=torch.float32)
    rows = []
    data = np.load(a.calib) if a.calib else None
    first = data['tokens'].shape[0] - n if data is not None else None
    if data is not None and not np.array_equal(data['tokens'][first:], ids.numpy()):
        raise SystemExit('held-out ids differ from the calibration file')
    for i in range(n):
        lp = logits_of(th[i].to(dev), norm_d, head_d)
        t = lp.topk(TOPK, -1)
        top_ids[i], top_lp[i] = t.indices.to(torch.int32).cpu(), t.values.cpu()
        nl = lp[:-1].gather(1, ids[i, 1:, None].to(dev))[:, 0]
        nxt[i] = nl.cpu()
        row = {'sequence': i, 'nll': float(-nl.double().mean()),
               'top1_accuracy': float((lp[:-1].argmax(-1).cpu() == ids[i, 1:]).float().mean())}
        if data is not None:
            row.update({'chunk': int(first + i), 'domain': DOMAINS[int(data['source_ids'][first + i])]})
        rows.append(row)
    tp = a.out / 'original-topk.safetensors'
    save_file({'input_ids': ids.to(torch.int32), 'top_ids': top_ids, 'top_logprobs': top_lp, 'next_logprob': nxt}, str(tp),
              metadata={'source_revision': R.SOURCE_REV, 'what': 'original BF16 model, log-softmax over the full vocabulary'})
    index = {'sequences': n, 'positions': T, 'source_revision': R.SOURCE_REV,
             'original': {'nll': float(-nxt.double().mean()), 'top1_accuracy': float(np.mean([r['top1_accuracy'] for r in rows]))},
             'files': {'teacher.safetensors': 'final_hidden [64, 512, 6144] BF16 (the residual stream after layer 60, before model.norm), '
                                              'input_ids [64, 512] int32, routes [58, 32768, 8] uint8 (MoE layers 3..60, sequence-major)',
                       hp.name: {'sha256': sha_file(hp), 'tensors': 'model.norm.weight [6144], lm_head.weight [250624, 6144], BF16'},
                       tp.name: {'sha256': sha_file(tp), 'tensors': f'top_ids / top_logprobs [64, 512, {TOPK}], next_logprob [64, 511] '
                                                                     '(log-probability of token p + 1 at position p), input_ids'}},
             'rows': rows}
    (a.out / 'heldout-index.json').write_text(json.dumps(index, indent=1) + '\n')
    print(json.dumps({k: v for k, v in index.items() if k != 'rows'}, indent=1))


# ------------------------------------------------------------------------------------- student
def cmd_student(a):
    import k2h_eval as EV
    import k2h_vq_format as fmt
    dev = torch.device(a.device)
    t0 = time.time()
    ids = load_file(str(a.teacher))['input_ids'].to(torch.int64)
    n, T = ids.shape
    rel = EV.Release(a.parts, dev)
    emb = rel.embedding()
    variants = a.variants.split(',')
    h = {v: emb[ids.to(dev)].clone() for v in variants}
    del emb
    cos, sin = R.rope_cache(T, dev)
    for l in range(R.L):
        lw = rel.layer(l)
        for v in variants:
            for s0 in range(0, n, a.batch_seqs):
                EV.layer_forward(lw, h[v][s0:s0 + a.batch_seqs], cos, sin, kv=EV.VARIANTS[v])
        del lw
        torch.cuda.empty_cache()
        if l % 10 == 0:
            print(json.dumps({'layer': l, 'elapsed': round(time.time() - t0, 1)}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    out = {}
    for v in variants:
        p = a.out / f'student-final.{v}.safetensors'
        save_file({'final_hidden': h[v].cpu().contiguous(), 'input_ids': ids.to(torch.int32)}, str(p),
                  metadata={'variant': v, 'kv_format': EV.KV_FORMATS[v], 'recipe_id': fmt.RECIPE_ID,
                            'what': 'reference forward of the release bytes (k2h_eval), residual stream after layer 60'})
        out[p.name] = sha_file(p)
    (a.out / 'student-final.json').write_text(json.dumps({'files': out, 'seconds': round(time.time() - t0, 1),
                                                          'payload_groups_cross_checked': rel.checked}, indent=1) + '\n')
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------------------- score
def release_head(release, dev):
    import k2h_eval as EV
    return EV.Release(release, dev).head()


def cmd_score(a):
    dev = torch.device(a.device)
    if a.reference == 'original':
        if not (a.teacher and a.original_head):
            raise SystemExit('--reference original needs --teacher and --original-head')
        t = load_file(str(a.teacher))
        ref_h, ids = t['final_hidden'], t['input_ids'].to(torch.int64)
        oh = load_file(str(a.original_head))
        ref_norm, ref_head = oh['model.norm.weight'].to(dev), oh['lm_head.weight'].to(dev)
        rel_head = None
    else:
        t = load_file(a.reference)
        ref_h, ids = t['final_hidden'], t['input_ids'].to(torch.int64)
        if not a.release:
            raise SystemExit('a student reference needs --release (its norm and Q8_0 head)')
        rel_head = release_head(a.release, dev)
        ref_norm, ref_head = rel_head
    cand_h = None
    if a.hidden:
        c = load_file(str(a.hidden))
        if not torch.equal(c['input_ids'].to(torch.int64), ids):
            raise SystemExit('candidate input_ids differ from the reference')
        cand_h = c['final_hidden']
        if rel_head is None:
            if not a.release:
                raise SystemExit('--hidden needs --release (the candidate is scored through its norm and Q8_0 head)')
            rel_head = release_head(a.release, dev)
    elif not (a.logits or a.frontiers):
        raise SystemExit('give --logits DIR, --frontiers DIR or --hidden FILE')
    index = json.loads(a.index.read_text()) if a.index else None
    n, T = ids.shape

    def frontier_files(i):
        return sorted((a.frontiers / f'seq-{i:04d}').glob('frontier_*.logits.json')) if a.frontiers else []

    seqs = [i for i in range(n) if cand_h is not None or (a.logits and (a.logits / f'seq-{i:04d}.npy').exists()) or frontier_files(i)]
    if not seqs:
        raise SystemExit('no candidate sequences found')
    rows, kls = [], []
    for i in seqs:
        rlp = logits_of(ref_h[i].to(dev), ref_norm, ref_head)
        if cand_h is not None:
            clp = logits_of(cand_h[i].to(dev), *rel_head)
            pos = torch.arange(clp.shape[0], device=dev)
        elif a.logits:
            raw = torch.from_numpy(np.load(a.logits / f'seq-{i:04d}.npy')).to(dev).float()
            if raw.ndim != 2 or raw.shape[1] != R.VOCAB or raw.shape[0] > T:
                raise SystemExit(f'seq-{i:04d}.npy: shape {tuple(raw.shape)}')
            clp = torch.log_softmax(raw, -1)
            pos = torch.arange(clp.shape[0], device=dev)
        else:
            lgs, fr = [], []
            for f in frontier_files(i):
                d = json.loads(f.read_text())
                if d['vocab'] != R.VOCAB or not 1 <= d['frontier_tokens'] <= T or any(x is None for x in d['logits']):
                    raise SystemExit(f'{f}: vocabulary, frontier or non-finite logits')
                lgs.append(d['logits'])
                fr.append(d['frontier_tokens'] - 1)
            clp = torch.log_softmax(torch.tensor(lgs, dtype=torch.float32, device=dev), -1)
            pos = torch.tensor(fr, device=dev)
        p = clp.shape[0]
        rlp = rlp[pos]
        kl = (rlp.exp() * (rlp - clp)).sum(-1)
        same = rlp.argmax(-1) == clp.argmax(-1)
        has = pos < T - 1                                   # positions with a true next token in the sequence
        tgt = ids[i].to(dev)[(pos[has] + 1)][:, None]
        row = {'sequence': i, 'positions': p, 'nll_positions': int(has.sum()), 'kl': float(kl.double().mean()),
               'same_top1': float(same.float().mean()),
               'nll_reference': float(-rlp[has].gather(1, tgt).double().mean()) if bool(has.any()) else None,
               'nll_candidate': float(-clp[has].gather(1, tgt).double().mean()) if bool(has.any()) else None}
        if index:
            row['domain'] = index['rows'][i].get('domain')
        rows.append(row)
        kls.append(kl.cpu())
    w = np.array([r['positions'] for r in rows], dtype=np.float64)
    # NLL averages over the positions with a next token (a frontier at T excludes its last row).
    w_nll = np.array([r['nll_positions'] for r in rows], dtype=np.float64)
    def avg(k, sel=None):
        keep = [j for j, r in enumerate(rows) if (sel is None or sel[j]) and r[k] is not None]
        weights = w_nll if k.startswith('nll') else w
        return float(np.average([rows[j][k] for j in keep], weights=[weights[j] for j in keep])) if keep else None
    summary = {'reference': a.reference, 'candidate': str(a.hidden or a.logits or a.frontiers), 'sequences': len(rows), 'positions': int(w.sum()),
               'kl': avg('kl'), 'kl_median': float(torch.cat(kls).median()), 'same_top1': avg('same_top1'),
               'nll_reference': avg('nll_reference'), 'nll_candidate': avg('nll_candidate')}
    if index:
        summary['domains'] = {}
        for d in DOMAINS:
            sel = [r.get('domain') == d for r in rows]
            if any(sel):
                summary['domains'][d] = {'sequences': int(sum(sel)), 'kl': avg('kl', sel), 'same_top1': avg('same_top1', sel),
                                         'nll_reference': avg('nll_reference', sel), 'nll_candidate': avg('nll_candidate', sel)}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({**summary, 'rows': rows}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------------------- texts
def cmd_texts(a):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(a.tokenizer))
    ids = load_file(str(a.teacher))['input_ids'].tolist()
    (a.out / 'text').mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seq in enumerate(ids):
        text = tk.decode(seq, skip_special_tokens=False)
        back = tk.encode(text, add_special_tokens=False).ids
        (a.out / 'text' / f'seq-{i:04d}.txt').write_text(text, encoding='utf-8')
        rows.append({'sequence': i, 'retokenizes_to_same_ids': back == seq, 'retokenized_tokens': len(back)})
    out = {'what': 'the held-out sequences decoded with the model tokenizer (no BOS); a runtime fed these texts sees the same ids '
                   'only where retokenizes_to_same_ids is true', 'same': sum(r['retokenizes_to_same_ids'] for r in rows),
           'sequences': len(rows), 'rows': rows}
    (a.out / 'text' / 'index.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('original')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--calib', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('student')
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variants', default='kv-bf16,kv-nvfp4')
    p.add_argument('--batch-seqs', type=int, default=32)
    p = sub.add_parser('score')
    p.add_argument('--reference', required=True, help='`original` or a student-final.*.safetensors file')
    p.add_argument('--teacher', type=Path)
    p.add_argument('--original-head', type=Path)
    p.add_argument('--release', type=Path, help='the release directory (body-nonexpert.gguf)')
    p.add_argument('--logits', type=Path, help='directory of seq-NNNN.npy [positions, 250624]')
    p.add_argument('--frontiers', type=Path, help='directory of seq-NNNN/frontier_FFFFFF.logits.json (ds4-bench --dump-frontier-logits-dir)')
    p.add_argument('--hidden', type=Path, help='safetensors with final_hidden [64, 512, 6144] and input_ids')
    p.add_argument('--index', type=Path, help='heldout-index.json, for the per-domain rows')
    p.add_argument('--out', type=Path)
    p = sub.add_parser('texts')
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    for q in sub.choices.values():
        q.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    {'original': cmd_original, 'student': cmd_student, 'score': cmd_score, 'texts': cmd_texts}[a.cmd](a)


if __name__ == '__main__':
    main()
