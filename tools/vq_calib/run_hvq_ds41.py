#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash hVQ on one GPU, layer by layer: collect a layer's per-expert second
moments (ds41_hess.py), convert the layer (convert_layer_ds41_hvq.py), drop the moments.
A layer's moments are 48 GiB in FP32 (384 x 5120^2 + 384 x 2304^2); 40 layers would be
1.9 TB, so they stay on the device and never reach the disk.

Same teacher stream as the iVQ statistics: the FP4/FP8 original through ds41_ref, the
calibration sequences dealt into `--world` token-balanced shards exactly as
`run_ds41.py collect` deals them to its ranks (the iVQ run had 8), so every forward batch
is one the iVQ collector ran. The shards' residual streams wait in pinned host memory
(41 KiB per token) and take turns on the device, one layer at a time:

    layer L weights -> for each shard: host -> device, forward (+ moments), device -> host
                    -> fit layer L (device) -> pack / decode check / write (host thread)

Resume: a layer whose blob and receipt exist for the requested variant, column statistics
and calibration run is not converted again and its moments are not collected; the forward
still runs through it, because later layers read its output. The loop stops after the
last layer that is still wanted.

`--save-hess DIR` writes the listed layers' moments to DIR instead of converting (the few
layers the hyperparameters are chosen on, sweep_hvq_ds41.py).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from convert_layer_ds41 import statistics
import convert_layer_ds41_hvq as cv
import ds41_hvq as hv
from ds41_hess import LayerHess, NORMALIZATION, PRECISION
from ds41_ref import EngramHash, Engram, Layer, SeqState, Stats, Teacher, rope_tables
from reduce_stats import hub_revision, run_id, sha256_file, source_fingerprint
from run_ds41 import ROOT, emit, shard, wait_for


def content_sha256(data):
    """sha256(input_ids bytes + offsets bytes): the calibration set's identity (file bytes vary with metadata order)."""
    import hashlib
    return hashlib.sha256(data['input_ids'].numpy().tobytes() + data['offsets'].numpy().tobytes()).hexdigest()


class Shard:
    """One shard of the calibration set. Its residual stream, mixes and attention state live in
    host memory (the stream pinned) between its turns on the device."""

    def __init__(self, rows, ids, off, emb, hc, hasher, dev):
        toks, self.seqs, start = [], [], 0
        for i in rows:
            s, e = int(off[i]), int(off[i + 1])
            toks.append(ids[s:e])
            self.seqs.append((start, e - s))
            start += e - s
        tok = torch.cat(toks).to(dev)
        self.rows, self.tokens = rows, tok.numel()
        self.hashes = torch.cat([hasher(tok[s:s + n]) for s, n in self.seqs]).cpu()
        self.h = torch.empty((self.tokens, hc, emb.shape[1]), dtype=emb.dtype, pin_memory=True)
        self.h.copy_(emb[tok.long()].unsqueeze(1).repeat(1, hc, 1))
        self.pre = torch.zeros((self.tokens, hc), dtype=torch.float32, pin_memory=True)
        self.pre[:, 0] = 1.0
        self.states = [{} for _ in self.seqs]

    def load(self, dev):
        states = []
        for saved in self.states:
            st = SeqState()
            for k, v in saved.items():
                setattr(st, k, v.to(dev))
            states.append(st)
        return self.h.to(dev, non_blocking=True), self.pre.to(dev, non_blocking=True), states

    def store(self, h, pre, states):
        self.h.copy_(h)
        self.pre.copy_(pre)
        self.states = [{k: getattr(st, k).cpu() for k in SeqState.__slots__ if getattr(st, k) is not None}
                       for st in states]


class Collector:
    """The teacher forward over every shard, one layer per step."""

    def __init__(self, source, tok_dir, calib, world, max_tokens=0, chunk=65536, dev='cuda'):
        self.dev, self.chunk = dev, chunk
        self.teacher = Teacher(source, dev)
        self.args = a = self.teacher.a
        data = load_file(str(calib))
        ids, off = data['input_ids'], data['offsets']
        hasher = EngramHash(a, tok_dir, dev)
        emb = self.teacher.src.read(['embed.weight'])['embed.weight'].to(dev)
        self.shards = [Shard(shard(off, r, world, max_tokens or None), ids, off, emb, a['hc_mult'], hasher, dev)
                       for r in range(world)]
        del emb
        torch.cuda.empty_cache()
        self.tokens = sum(s.tokens for s in self.shards)
        self.ropes = rope_tables(a, max(n for s in self.shards for _, n in s.seqs) + 8, dev)
        self.reader = ThreadPoolExecutor(1)
        self.pending, self.next = None, 0
        emit({'phase': 'shards', 'world': world, 'tokens': self.tokens, 'per_shard': [s.tokens for s in self.shards],
              'sequences': sum(len(s.seqs) for s in self.shards)})

    def fetch(self, L):
        wait_for(self.teacher, f'layers.{L}.ffn.experts.383.w2.scale')
        if L in self.args['engram_layer_ids']:
            wait_for(self.teacher, f'layers.{L}.engram.wkv.scale')
        return self.teacher.host_layer(L)

    def step(self, L, moments, probe=()):
        """Run layer L over every shard. -> (Stats with .hess, or None; the layer's host tensors)."""
        assert L == self.next, (L, self.next)
        a, dev = self.args, self.dev
        host = self.pending.result() if self.pending is not None else self.fetch(L)
        self.pending = self.reader.submit(self.fetch, L + 1) if L + 1 < a['n_layers'] else None
        w = self.teacher.layer(L, host)
        engram, li = None, None
        if L in a['engram_layer_ids']:
            li = a['engram_layer_ids'].index(L)
            wait_for(self.teacher, f'layers.{L}.engram.embed.scale')
            engram = Engram(a, w, self.teacher.src.memmap(f'layers.{L}.engram.embed.weight'),
                            self.teacher.src.memmap(f'layers.{L}.engram.embed.scale'))
        layer = Layer(a, L, w, self.ropes, engram)
        stats = None
        if moments:
            stats = Stats(a, dev)
            stats.hess = LayerHess(a['n_routed_experts'], a['dim'], a['moe_inter_dim'], dev, probe)
        for sh in self.shards:
            h, pre, states = sh.load(dev)
            rows_fn = (lambda c0, c1, hs=sh.hashes: engram.rows(hs[c0:c1, li])) if engram is not None else None
            with torch.no_grad():
                pre = layer.forward(h, pre, sh.seqs, states, stats, rows_fn, self.chunk)
            sh.store(h, pre, states)
            del h, pre, states
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.next = L + 1
        return stats, host


def agreement(stats, ref):
    """This run's diagonal statistics against the iVQ release's (the same calibration set and teacher on
    other hardware and batch kernels): how close the two teacher streams are."""
    t = {k: v.cpu() for k, v in stats.items()}
    out = {'count_l1': float((t['count'] - ref['count']).abs().sum() / ref['count'].sum()),
           'tokens': [float(t['tokens']), float(ref['tokens'])]}
    for k in ('x2', 'a2'):
        out[k + '_rel'] = float((t[k] - ref[k]).norm() / ref[k].norm())
        prof, rprof = t[k] / t[k].sum(1, keepdim=True).clamp_min(1e-300), ref[k] / ref[k].sum(1, keepdim=True).clamp_min(1e-300)
        out[k + '_profile_rel'] = float((prof - rprof).norm() / rprof.norm())
    return out


def parse_probe(text):
    """'0:3,17;20:5' -> {0: (3, 17), 20: (5,)}"""
    out = {}
    for part in filter(None, text.split(';')):
        layer, _, experts = part.partition(':')
        out[int(layer)] = tuple(int(e) for e in experts.split(','))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--tok', default=ROOT, help='directory with tokenizer.json (Engram token map)')
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--d-stats', type=Path, required=True, help="the iVQ release's statistics (column weights D)")
    p.add_argument('--ivq', required=True, help='iVQ release: assembled GGUF or the directory of its parts')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', default='0-39')
    p.add_argument('--t', type=float, default=None)
    p.add_argument('--damp', type=float, default=0.0)
    p.add_argument('--hess-shrink', type=float, default=256.0)
    p.add_argument('--world', type=int, default=8, help='shards; the iVQ statistics ran 8 ranks')
    p.add_argument('--max-tokens', type=int, default=0, help='per-shard token cap (smoke tests)')
    p.add_argument('--chunk', type=int, default=65536)
    p.add_argument('--save-hess', type=Path, help='write the layers\' moments here instead of converting')
    p.add_argument('--probe', default='', help="accumulation precision probe, 'layer:expert,expert;layer:...'")
    a = p.parse_args()
    if a.save_hess is None and a.t is None:
        raise SystemExit('--t is required unless --save-hess')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    layers = cv.parse_layers(a.layers)
    probe = parse_probe(a.probe)
    data = load_file(str(a.calib))
    # one record per collection: the same inputs, shards and arithmetic give the same id
    run = {'collector': 'run_hvq_ds41.py', 'mode': 'collect-hess', 'calib_sha256': sha256_file(a.calib), 'world': a.world,
           'layers': list(range(max(layers) + 1)), 'max_tokens': a.max_tokens, 'rows': '', 'experts': 'teacher',
           'source': os.path.realpath(a.source), 'source_revision': hub_revision(a.source),
           'source_files': source_fingerprint(a.source)[0], 'tok': os.path.realpath(a.tok),
           'normalization': NORMALIZATION, 'precision': PRECISION}
    info = {'run_id': run_id({**run, 'layers': 'prefix'}), 'calib_sha256': run['calib_sha256'],
            'calib_content_sha256': content_sha256(data), 'run': run}
    del data
    var = cv.Variant(a.t, a.damp, a.hess_shrink) if a.t is not None else None
    todo = []
    for L in layers:
        if a.save_hess is not None:
            if not (a.save_hess / f'layer-{L:02d}.safetensors').exists():
                todo.append(L)
            continue
        st = statistics(a.d_stats, L, 'imatrix')
        if not cv.reuse(L, a.out, {**var.want(), 'd_statistics_sha256': st[2], 'calibration_run_id': info['run_id']}):
            todo.append(L)
    emit({'phase': 'plan', 'run_id': info['run_id'], 'layers_wanted': todo, 'variant': var.want() if var else None})
    if not todo:
        return
    out = a.save_hess if a.save_hess is not None else a.out
    (out / 'stats').mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()
    col = Collector(a.source, a.tok, a.calib, a.world, a.max_tokens, a.chunk)
    emit({'phase': 'ready', 'seconds': round(time.monotonic() - t_all, 1)})
    hvq = hv.Hvq() if var is not None else None
    writer, pending = ThreadPoolExecutor(1), None
    for L in range(max(todo) + 1):
        need = L in todo
        t0 = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        stats, host = col.step(L, need, probe.get(L, ()))
        collect = {'seconds': round(time.monotonic() - t0, 2),
                   'peak_gpu_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), 'moments': need}
        if not need:
            emit({'layer': L, 'phase': 'forward', **collect})
            del host
            continue
        hess = stats.hess
        hess.meta = {'run_id': info['run_id'], 'calib_content_sha256': info['calib_content_sha256'], 'layer': L,
                     'world': a.world, 'tokens': col.tokens}
        st = statistics(a.d_stats, L, 'imatrix')
        collect['d_statistics_agreement'] = agreement(stats.t, st[0])
        save_file({k: v.cpu().contiguous() for k, v in stats.t.items()}, str(out / 'stats' / f'layer-{L:02d}.safetensors'),
                  metadata={'run_id': info['run_id'], 'calib': str(a.calib), 'world': str(a.world)})
        if hess.probe:
            (out / 'stats' / f'probe-{L:02d}.json').write_text(json.dumps(hess.probe_report(), indent=1) + '\n')
            hess.probe = {}
        emit({'layer': L, 'phase': 'collected', **collect, **hess.describe()})
        del stats
        torch.cuda.empty_cache()
        if a.save_hess is not None:
            del host
            t1 = time.monotonic()
            tmp = a.save_hess / f'layer-{L:02d}.safetensors.tmp'
            hess.save(tmp, {'run': run, 'collect': collect, 'sha256': hess.digest()})
            tmp.rename(a.save_hess / f'layer-{L:02d}.safetensors')
            emit({'layer': L, 'phase': 'saved', 'seconds': round(time.monotonic() - t1, 1)})
            del hess
            torch.cuda.empty_cache()
            continue
        job = cv.Job(L, st, var, a.ivq, cv.HostExperts(host, L))
        del host
        job.fit(hess, hvq)
        del hess
        torch.cuda.empty_cache()
        if pending is not None:
            pending.result()
        # pack, strict decode check, write and receipt on the host while the next layer runs
        pending = writer.submit(job.finish, a.out, info, True, collect, False)
        del job
    if pending is not None:
        pending.result()
    emit({'phase': 'done', 'seconds': round(time.monotonic() - t_all, 1), 'layers': todo})


if __name__ == '__main__':
    main()
