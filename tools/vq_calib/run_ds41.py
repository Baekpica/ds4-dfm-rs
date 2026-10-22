#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash calibration statistics and teacher-forced held-out passes, one
data-parallel rank per GPU (ds41_ref forward).

collect: each rank pushes its token-balanced shard of the calibration sequences through
  the FP4/FP8 teacher layer by layer and writes `rank-R/layer-LL.safetensors` (FP64 sums,
  same keys as the A.X-K2 collector where they exist), stamped with its run.
eval: one rank pushes a held-out set through the teacher or a VQ student (experts from a
  GGUF) and writes the final normed hidden states, the per-layer routes and
  `final.safetensors` for score_ds41.py.

Layer L+1 is read into pinned host memory while layer L runs. Shards that are still
downloading are waited for. Layers run from 0 (a later first layer would see embeddings);
eval runs a partial range only with --debug-partial. shard.json, DONE and every stamp
carry the run id (reduce_stats.py).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from ds41_ref import BF, EngramHash, Engram, Layer, SeqState, Stats, Teacher, hc_pre, rms_norm, rope_tables
from reduce_stats import hub_revision, run_id, sha256_file

ROOT = '/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'


def emit(v):
    print(json.dumps(v), flush=True)


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


def wait_for(teacher, names_prefix, need_any=True, poll=20):
    while True:
        teacher.src.refresh()
        if teacher.src.names(names_prefix):
            return
        emit({'phase': 'waiting', 'for': names_prefix})
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'eval'])
    p.add_argument('--source', default=ROOT)
    p.add_argument('--tok', default=ROOT, help='directory with tokenizer.json (Engram token map)')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=1)
    p.add_argument('--data', type=Path, required=True, help='calibration or held-out safetensors')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', default='0-39')
    p.add_argument('--max-tokens', type=int, default=0, help='per-rank token cap (smoke tests)')
    p.add_argument('--rows', default='', help='comma list of row ids to use (smoke tests)')
    p.add_argument('--experts', default='teacher', help="'teacher' or a GGUF path (VQ student)")
    p.add_argument('--chunk', type=int, default=65536)
    p.add_argument('--dump-h', default='', help='debug: positions "a:b" of h to save after every layer')
    p.add_argument('--debug-partial', action='store_true', help='eval: allow --layers not starting at 0 (debugging)')
    a = p.parse_args()
    lo, _, hi = a.layers.partition('-')
    layers = list(range(int(lo), int(hi or lo) + 1))
    if layers[0] != 0 and not (a.mode == 'eval' and a.debug_partial):
        raise SystemExit(f'--layers {a.layers} skips layers 0-{layers[0] - 1}, whose outputs are its input'
                         + ('' if a.mode == 'collect' else ' (eval: --debug-partial runs it anyway)'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = 'cuda'
    teacher = Teacher(a.source, dev)
    args = teacher.a
    rdir = a.out / (f'rank-{a.rank}' if a.mode == 'collect' else '')
    rdir.mkdir(parents=True, exist_ok=True)

    data = load_file(str(a.data))
    ids, off = data['input_ids'], data['offsets']
    if a.rows:
        rows = [int(x) for x in a.rows.split(',')]
    elif a.mode == 'collect':
        rows = shard(off, a.rank, a.world, a.max_tokens or None)
    else:
        rows = list(range(len(off) - 1))
    toks, seqs, st0 = [], [], 0
    for i in rows:
        s, e = int(off[i]), int(off[i + 1])
        toks.append(ids[s:e])
        seqs.append((st0, e - s))
        st0 += e - s
    tok = torch.cat(toks).to(dev)
    N = tok.numel()
    # reduce_stats.py accepts only files stamped by this run: same inputs, world, rank and layer list
    run = {'collector': 'run_ds41.py', 'mode': a.mode, 'calib_sha256': sha256_file(a.data), 'world': a.world,
           'layers': layers, 'max_tokens': a.max_tokens, 'rows': a.rows, 'experts': a.experts,
           'source': os.path.realpath(a.source), 'source_revision': hub_revision(a.source),
           'tok': os.path.realpath(a.tok)}
    rid = run_id(run)
    stamp = {'calib': str(a.data), 'world': str(a.world), 'rank': str(a.rank), 'mode': a.mode,
             'experts': a.experts, 'run_id': rid}
    (rdir / 'DONE').unlink(missing_ok=True)
    (rdir / 'shard.json').write_text(json.dumps({'calib': str(a.data), 'rows': rows, 'tokens': N, 'world': a.world,
                                                 'rank': a.rank, 'run': run, 'run_id': rid}) + '\n')
    emit({'rank': a.rank, 'phase': 'shard', 'sequences': len(rows), 'tokens': N,
          'max_len': max(n for _, n in seqs)})

    max_len = max(n for _, n in seqs)
    ropes = rope_tables(args, max_len + 8, dev)
    eh = EngramHash(args, a.tok, dev)
    hashes = torch.cat([eh(tok[s:s + n]) for s, n in seqs]).cpu()          # [N, n_layers, cols]
    emit({'rank': a.rank, 'phase': 'engram-hash', 'shape': list(hashes.shape)})

    student = None
    if a.experts != 'teacher':
        from ds41_vq_student import VQExperts
        student = VQExperts(a.experts, dev)

    emb = teacher.src.read(['embed.weight'])['embed.weight'].to(dev)
    h = emb[tok.long()].unsqueeze(1).repeat(1, args['hc_mult'], 1).contiguous()
    del emb
    pre_mix = torch.zeros((N, args['hc_mult']), dtype=torch.float32, device=dev)
    pre_mix[:, 0] = 1.0
    states = [SeqState() for _ in seqs]

    reader = ThreadPoolExecutor(1)

    def fetch(L):
        wait_for(teacher, f'layers.{L}.ffn.experts.383.w2.scale')
        if L in args['engram_layer_ids']:
            wait_for(teacher, f'layers.{L}.engram.wkv.scale')
        return teacher.host_layer(L)

    pending = reader.submit(fetch, layers[0])
    t_all = time.monotonic()
    routes_all = {}
    for i, L in enumerate(layers):
        t0 = time.monotonic()
        host = pending.result()
        pending = reader.submit(fetch, layers[i + 1]) if i + 1 < len(layers) else None
        t_read = time.monotonic() - t0
        w = teacher.layer(L, host)
        del host
        if student is not None:
            w['experts'] = student.layer(L)
        engram = None
        rows_fn = None
        if L in args['engram_layer_ids']:
            li = args['engram_layer_ids'].index(L)
            wait_for(teacher, f'layers.{L}.engram.embed.scale')
            table = teacher.src.memmap(f'layers.{L}.engram.embed.weight')
            scale = teacher.src.memmap(f'layers.{L}.engram.embed.scale')
            engram = Engram(args, w, table, scale)
            rows_fn = lambda c0, c1, li=li, eg=engram: eg.rows(hashes[c0:c1, li])
        layer = Layer(args, L, w, ropes, engram)
        stats = Stats(args, dev) if a.mode == 'collect' else None
        routes = [] if a.mode == 'eval' else None
        with torch.no_grad():
            pre_mix = layer.forward(h, pre_mix, seqs, states, stats, rows_fn, a.chunk, routes)
        torch.cuda.synchronize()
        if a.dump_h:
            d0, d1 = (int(x) for x in a.dump_h.split(':'))
            save_file({'h': h[d0:d1].cpu().contiguous(), 'pre': pre_mix[d0:d1].cpu().contiguous()},
                      str(rdir / f'h-{L:02d}.safetensors'))
        extra = {}
        if stats is not None:
            t = {k: v.cpu().contiguous() for k, v in stats.t.items()}
            save_file(t, str(rdir / f'layer-{L:02d}.safetensors'), metadata=stamp)
            cnt = stats.t['count'].float()
            extra = {'expert_count_min': int(cnt.min()), 'expert_count_median': float(cnt.median()),
                     'expert_count_max': int(cnt.max()), 'experts_unused': int((cnt == 0).sum())}
        if routes:
            routes_all[f'idx-{L:02d}'] = routes[0][0]
            routes_all[f'wt-{L:02d}'] = routes[0][1]
        emit({'rank': a.rank, 'layer': L, 'seconds': round(time.monotonic() - t0, 1), 'read_wait': round(t_read, 1),
              'h_rms': h[::97].float().pow(2).mean().sqrt().item(),
              'gpu_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1), **extra})
        del w, layer, stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    files = [f'layer-{L:02d}.safetensors' for L in layers] if a.mode == 'collect' else []
    if layers[-1] == args['n_layers'] - 1 and a.mode == 'eval':
        files += ['final.safetensors', 'routes.safetensors']
        nw = teacher.src.read(['norm.weight'])['norm.weight'].to(dev)
        x = rms_norm(hc_pre(h, pre_mix), nw)
        save_file({'final': x.cpu().contiguous(), 'input_ids': tok.to(torch.int32).cpu(),
                   'offsets': torch.tensor([0] + [s + n for s, n in seqs], dtype=torch.int64)},
                  str(rdir / 'final.safetensors'), metadata=stamp)
        save_file(routes_all, str(rdir / 'routes.safetensors'), metadata=stamp)
    (rdir / 'DONE').write_text(json.dumps({'seconds': time.monotonic() - t_all, 'layers': layers, 'files': files,
                                           'run_id': rid}) + '\n')
    emit({'rank': a.rank, 'phase': 'done', 'seconds': round(time.monotonic() - t_all, 1)})


if __name__ == '__main__':
    main()
