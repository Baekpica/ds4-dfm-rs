#!/usr/bin/env python3
"""Calibration statistics for A.X-K2 VQ/Q4_K conversion: one data-parallel rank per GPU.

Each rank pushes its shard of the calibration sequences through the FP8 teacher
(axk2_ref, BF16 math) layer by layer and accumulates FP64 sums of squared
inputs for every quantized tensor (docs/vq-calibration.md, Statistics): per routed
expert the gate/up input x and down input a, unweighted (llama.cpp imatrix)
and weighted by the squared route weight, plus gate/up row output sensitivities.
Rank 0 also pushes the held-out set and dumps, per layer, the teacher FFN input
and routing of every held-out token, and the final hidden states.

Layer L+1 is read into pinned host memory while layer L runs. The residual
stream is checkpointed every 8 layers; `--resume` continues from it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from axk2_ref import BF, Layer, Rope, Stats, TeacherWeights, rms_norm

ROOT = '/models/A.X-K2'


def emit(v):
    print(json.dumps(v), flush=True)


def load_set(path):
    t = load_file(str(path))
    ids, off = t['input_ids'], t['offsets']
    return ids, off


def shard(offsets, rank, world, limit=None):
    """Greedy token-balanced assignment, longest first, deterministic."""
    n = len(offsets) - 1
    lens = (offsets[1:] - offsets[:-1]).tolist()
    order = sorted(range(n), key=lambda i: (-lens[i], i))
    load = [0] * world
    mine = []
    for i in order:
        r = min(range(world), key=lambda k: (load[k], k))
        load[r] += lens[i]
        if r == rank:
            mine.append(i)
    mine.sort()
    if limit:
        out, tot = [], 0
        for i in mine:
            if tot + lens[i] > limit:
                continue
            out.append(i)
            tot += lens[i]
        mine = out
    return mine


def flat(ids, offsets, rows):
    toks, seqs, st = [], [], 0
    for i in rows:
        a, b = int(offsets[i]), int(offsets[i + 1])
        toks.append(ids[a:b])
        seqs.append((st, b - a))
        st += b - a
    return torch.cat(toks), seqs


def embed(src, token_ids, device):
    table = src.read(['model.embed_tokens.weight'], threads=16)['model.embed_tokens.weight'].to(device)
    h = table[token_ids.to(device).long()]
    del table
    torch.cuda.empty_cache()
    return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=ROOT, help='A.X-K2 checkpoint directory')
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--heldout', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', default='0-60')
    p.add_argument('--max-tokens', type=int, default=0, help='per-rank token cap (smoke tests)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--ffn-chunk', type=int, default=262144)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    teacher = TeacherWeights(a.source, dev)
    cfg = teacher.cfg
    rope = Rope(a.source, dev)
    lo, _, hi = a.layers.partition('-')
    layers = list(range(int(lo), int(hi or lo) + 1))
    rdir = a.out / f'rank-{a.rank}'
    rdir.mkdir(parents=True, exist_ok=True)
    hdir = a.out / 'heldout'
    if a.rank == 0 and a.heldout:
        hdir.mkdir(parents=True, exist_ok=True)

    ids, off = load_set(a.calib)
    rows = shard(off, a.rank, a.world, a.max_tokens or None)
    tok, seqs = flat(ids, off, rows)
    (rdir / 'shard.json').write_text(json.dumps({'calib': str(a.calib), 'rows': rows, 'tokens': len(tok),
                                                 'world': a.world, 'rank': a.rank}) + '\n')
    # Every rank file names its run, so reduce_stats.py can refuse files left by another one.
    stamp = {'calib': str(a.calib), 'world': str(a.world), 'rank': str(a.rank)}
    ho = None
    if a.rank == 0 and a.heldout:
        hids, hoff = load_set(a.heldout)
        htok, hseqs = flat(hids, hoff, list(range(len(hoff) - 1)))
        ho = {'tok': htok, 'seqs': hseqs}
    emit({'rank': a.rank, 'phase': 'shard', 'sequences': len(rows), 'tokens': len(tok),
          'max_len': max(n for _, n in seqs), 'heldout_tokens': len(ho['tok']) if ho else 0})

    start_layer = layers[0]
    ck = rdir / 'h-checkpoint.pt'
    if a.resume and ck.exists():
        state = torch.load(ck, map_location=dev, weights_only=True)
        h, start_layer = state['h'], state['next_layer']
        if ho is not None:
            ho['h'] = state['h_ho']
        emit({'rank': a.rank, 'phase': 'resumed', 'next_layer': start_layer})
    else:
        h = embed(teacher.src, tok, dev)
        if ho is not None:
            ho['h'] = embed(teacher.src, ho['tok'], dev)
    todo = [L for L in layers if L >= start_layer]

    reader = ThreadPoolExecutor(1)
    pending = reader.submit(teacher.host_layer, todo[0]) if todo else None
    t_all = time.monotonic()
    for i, L in enumerate(todo):
        t0 = time.monotonic()
        host = pending.result()
        pending = reader.submit(teacher.host_layer, todo[i + 1]) if i + 1 < len(todo) else None
        t_read = time.monotonic() - t0
        w = teacher.layer(L, host)
        del host
        layer = Layer(cfg, L, w, rope)
        stats = Stats(cfg, L, dev, moe=not layer.dense)
        with torch.no_grad():
            h, _, routes = layer.forward(h, seqs, stats, chunk=a.ffn_chunk)
            if ho is not None:
                ho['h'], ffn_in, r = layer.forward(ho['h'], ho['seqs'], None, keep_ffn_in=True, chunk=a.ffn_chunk)
                dump = {'ffn_in': ffn_in.cpu().contiguous()}
                if r is not None:
                    dump['topk_idx'] = r[0].to(torch.int32).cpu().contiguous()
                    dump['topk_w'] = r[1].float().cpu().contiguous()
                save_file(dump, str(hdir / f'layer-{L:02d}.safetensors'))
                del ffn_in, dump
        torch.cuda.synchronize()
        t = {k: v.cpu().contiguous() for k, v in stats.t.items()}
        save_file(t, str(rdir / f'layer-{L:02d}.safetensors'), metadata=stamp)
        extra = {}
        if routes is not None:
            cnt = stats.t['count'].float()
            extra = {'expert_count_min': int(cnt.min()), 'expert_count_median': float(cnt.median()),
                     'expert_count_max': int(cnt.max()), 'experts_unused': int((cnt == 0).sum())}
        emit({'rank': a.rank, 'layer': L, 'seconds': time.monotonic() - t0, 'read_wait': t_read,
              'tokens': int(stats.t['tokens'].item()), 'h_rms': h[::97].float().pow(2).mean().sqrt().item(),
              'gpu_peak_gib': torch.cuda.max_memory_allocated() / 2**30, **extra})
        del w, layer, stats, routes
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if (L + 1) % 8 == 0 and L != todo[-1]:
            state = {'h': h, 'next_layer': L + 1}
            if ho is not None:
                state['h_ho'] = ho['h']
            torch.save(state, ck.with_suffix('.tmp'))
            os.replace(ck.with_suffix('.tmp'), ck)
    if todo and todo[-1] == cfg['num_hidden_layers'] - 1:
        norm = teacher.src.load(['model.norm.weight'], dev)['model.norm.weight']
        x = rms_norm(h, norm)
        save_file({'head_in': x.float().pow(2).sum(0).double().cpu(), 'tokens': torch.tensor([float(len(x))], dtype=torch.float64)},
                  str(rdir / 'head.safetensors'), metadata=stamp)
        if ho is not None:
            save_file({'final_h': ho['h'].cpu().contiguous(), 'input_ids': ho['tok'].to(torch.int32),
                       'offsets': torch.tensor([0] + [s + n for s, n in ho['seqs']], dtype=torch.int64)},
                      str(hdir / 'final.safetensors'))
    (rdir / 'DONE').write_text(json.dumps({'seconds': time.monotonic() - t_all, 'layers': todo}) + '\n')
    emit({'rank': a.rank, 'phase': 'done', 'seconds': time.monotonic() - t_all})


if __name__ == '__main__':
    main()
