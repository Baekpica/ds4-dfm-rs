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

Host memory: the streams of all but one shard in one pinned arena (41 KiB per token, rounded up
to a power of two: 128 GiB for the 3.65M-token set in 8 shards), the attention state, and two
layers of pinned weights (~20 GiB): 170 GiB resident. Under a container limit the run hands
clean page cache back before the limit is reached (relieve_memory).

`--sweep-grid` fits every `t:damp` of the grid on the listed layers and scores each on
held-out tokens (sweep_hvq_ds41.py) while the layer's moments are on the device, instead of
converting: the choice of t and damping. `--save-hess DIR` writes the moments of the listed
layers to DIR (48 GiB each) for sweep_hvq_ds41.py run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from convert_layer_ds41 import statistics
import convert_layer_ds41_hvq as cv
import ds41_hvq as hv
from ds41_hess import LayerHess, NORMALIZATION, PRECISION
from ds41_ref import EngramHash, Engram, Layer, SeqState, Stats, Teacher, rope_tables
from reduce_stats import hub_revision, run_id, sha256_file, source_fingerprint
from run_ds41 import ROOT, emit, shard, wait_for
import sweep_hvq_ds41 as sw


def content_sha256(data):
    """sha256(input_ids bytes + offsets bytes): the calibration set's identity (file bytes vary with metadata order)."""
    import hashlib
    return hashlib.sha256(data['input_ids'].numpy().tobytes() + data['offsets'].numpy().tobytes()).hexdigest()


class Pool:
    """Pinned host memory for the residual streams of the shards that are off the device: one arena
    cut into `slots` equal slots (one fewer than there are shards, since one shard is always on the
    device). torch rounds a pinned allocation up to a power of two, so a 17.4 GiB buffer per shard
    would take 32 GiB each; the arena asks for that power of two once and uses all of it."""

    def __init__(self, slots, tokens, hc, dim, dtype=torch.bfloat16):
        row = hc * dim * torch.empty(0, dtype=dtype).element_size()
        size = 1 << (slots * tokens * row - 1).bit_length()
        self.arena = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.tokens = size // slots // row
        self.shape, self.dtype, self.step = (self.tokens, hc, dim), dtype, self.tokens * row
        self.free = list(range(slots))
        self.bytes = size

    def take(self):
        """-> (slot, its [tokens, hc, dim] view)."""
        i = self.free.pop()
        return i, self.arena[i * self.step:(i + 1) * self.step].view(self.dtype).view(self.shape)

    def give(self, slot):
        self.free.append(slot)


ENGRAM_WINDOW = 1 << 24      # table rows read at a time when the cache is built (4 GiB)


def build_engram_cache(src, a, shards, out):
    """Every shard's Engram table rows, gathered once into `out` (L{layer}-s{shard}-{w,s}.npy).

    A layer's table is 98 GiB and the calibration tokens touch nearly all of its pages in hashed
    order. Read through a memory map that is fast only while the whole table stays in the page
    cache; under a smaller memory limit every chunk of every shard reads it from disk again. Here
    each table is read once, front to back, and the rows are kept per shard in token order."""
    out.mkdir(parents=True, exist_ok=True)
    for li, L in enumerate(a['engram_layer_ids']):
        names = [(out / f'L{L:02d}-s{r}-w.npy', out / f'L{L:02d}-s{r}-s.npy') for r in range(len(shards))]
        if all(w.exists() and s.exists() for w, s in names):
            continue
        t0 = time.monotonic()
        table = src.memmap(f'layers.{L}.engram.embed.weight')
        scale = src.memmap(f'layers.{L}.engram.embed.scale')
        ids = torch.cat([sh.hashes[:, li].reshape(-1) for sh in shards]).numpy()
        order = np.argsort(ids, kind='stable')
        ranked = ids[order]
        rows_w = np.empty((ids.size, table.shape[1]), np.uint8)
        rows_s = np.empty((ids.size, scale.shape[1]), np.uint8)
        for r0 in range(0, table.shape[0], ENGRAM_WINDOW):
            r1 = min(table.shape[0], r0 + ENGRAM_WINDOW)
            lo, hi = np.searchsorted(ranked, [r0, r1])
            if lo == hi:
                continue
            rows_w[order[lo:hi]] = np.ascontiguousarray(table[r0:r1])[ranked[lo:hi] - r0]
            rows_s[order[lo:hi]] = np.ascontiguousarray(scale[r0:r1])[ranked[lo:hi] - r0]
        # spot check against the table itself
        probe = np.random.default_rng(L).integers(0, ids.size, 256)
        assert np.array_equal(rows_w[probe], table[np.sort(ids[probe])][np.argsort(np.argsort(ids[probe]))]), 'engram cache'
        start = 0
        for sh, (wpath, spath) in zip(shards, names):
            n = sh.hashes.shape[0] * sh.hashes.shape[2]
            for path, rows in ((wpath, rows_w), (spath, rows_s)):
                tmp = path.with_suffix('.tmp.npy')
                np.save(tmp, rows[start:start + n])
                tmp.rename(path)
            start += n
        del rows_w, rows_s
        emit({'phase': 'engram-cache', 'layer': L, 'rows': int(ids.size), 'seconds': round(time.monotonic() - t0, 1)})


class Shard:
    """One shard of the calibration set: its sequences, Engram rows, and between its turns on the
    device its residual stream (a pool buffer), mixes and attention state in host memory."""

    def __init__(self, rows, ids, off, hasher, dev):
        toks, self.seqs, start = [], [], 0
        for i in rows:
            s, e = int(off[i]), int(off[i + 1])
            toks.append(ids[s:e])
            self.seqs.append((start, e - s))
            start += e - s
        self.tok = torch.cat(toks)
        self.rows, self.tokens = rows, self.tok.numel()
        tok = self.tok.to(dev)
        self.hashes = torch.cat([hasher(tok[s:s + n]) for s, n in self.seqs]).cpu()
        self.slot = self.buf = self.pre = self.states = self.live = None

    def embed(self, emb, hc, dev):
        """The stream entering layer 0."""
        h = emb[self.tok.to(dev).long()].unsqueeze(1).repeat(1, hc, 1).contiguous()
        pre = torch.zeros((self.tokens, hc), dtype=torch.float32, device=dev)
        pre[:, 0] = 1.0
        return h, pre, [SeqState() for _ in self.seqs]

    def load(self, dev, pool):
        """Host -> device; the host buffer goes back to the pool."""
        h = self.buf[:self.tokens].to(dev)
        pool.give(self.slot)
        states = []
        for saved in self.states:
            st = SeqState()
            for k, v in saved.items():
                setattr(st, k, v.to(dev))
            states.append(st)
        pre = self.pre.to(dev)
        self.slot = self.buf = self.pre = self.states = None
        return h, pre, states

    def store(self, pool):
        """Device (self.live) -> host."""
        h, pre, states = self.live
        self.slot, self.buf = pool.take()
        self.buf[:self.tokens].copy_(h)
        self.pre = pre.cpu()
        self.states = [{k: getattr(st, k).cpu() for k in SeqState.__slots__ if getattr(st, k) is not None}
                       for st in states]
        self.live = None


class Collector:
    """The teacher forward over every shard, one layer per step.

    A layer starts with the shard the previous layer left on the device; every other shard comes up
    from its host buffer, which then takes the shard that just finished:

        device:  [done s]  +  load s'   ->  store s into the buffer s' left  ->  forward s'
    """

    def __init__(self, source, tok_dir, calib, world, max_tokens=0, chunk=65536, dev='cuda', engram_cache=None):
        self.dev, self.chunk, self.engram_cache = dev, chunk, engram_cache
        self.teacher = Teacher(source, dev)
        self.args = a = self.teacher.a
        data = load_file(str(calib))
        ids, off = data['input_ids'], data['offsets']
        hasher = EngramHash(a, tok_dir, dev)
        self.shards = [Shard(shard(off, r, world, max_tokens or None), ids, off, hasher, dev) for r in range(world)]
        self.tokens = sum(s.tokens for s in self.shards)
        if engram_cache is not None:
            build_engram_cache(self.teacher.src, a, self.shards, engram_cache)
        self.pool = Pool(world - 1, max(s.tokens for s in self.shards), a['hc_mult'], a['dim']) if world > 1 else None
        self.ropes = rope_tables(a, max(n for s in self.shards for _, n in s.seqs) + 8, dev)
        self.reader = ThreadPoolExecutor(1)
        self.pending, self.next, self.parked = None, 0, None
        emit({'phase': 'shards', 'world': world, 'tokens': self.tokens, 'per_shard': [s.tokens for s in self.shards],
              'sequences': sum(len(s.seqs) for s in self.shards),
              'pinned_stream_gib': round(self.pool.bytes / 2 ** 30, 1) if self.pool else 0})

    def fetch(self, L):
        wait_for(self.teacher, f'layers.{L}.ffn.experts.383.w2.scale')
        if L in self.args['engram_layer_ids']:
            wait_for(self.teacher, f'layers.{L}.engram.wkv.scale')
        return self.teacher.host_layer(L)

    def engram_rows(self, L, li, i, engram, w):
        """(c0, c1) -> the Engram rows of shard i's tokens c0..c1: from the table through its memory
        map, or from the shard's gathered rows (build_engram_cache), which are in token order."""
        sh = self.shards[i]
        if self.engram_cache is None:
            return lambda c0, c1: engram.rows(sh.hashes[c0:c1, li])
        cols = sh.hashes.shape[2]
        local = Engram(self.args, w, np.load(self.engram_cache / f'L{L:02d}-s{i}-w.npy', mmap_mode='r'),
                       np.load(self.engram_cache / f'L{L:02d}-s{i}-s.npy', mmap_mode='r'))
        rows = lambda c0, c1: local.rows(torch.arange(c0 * cols, c1 * cols).view(-1, cols))
        if local.table.shape[0] != sh.tokens * cols or not torch.equal(rows(0, 4), engram.rows(sh.hashes[0:4, li])):
            raise ValueError(f'{self.engram_cache}: layer {L} shard {i} is not this run\'s Engram rows')
        return rows

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
        emb = self.teacher.src.read(['embed.weight'])['embed.weight'].to(dev) if L == 0 else None
        order = [i for i in range(len(self.shards)) if i != self.parked]
        if self.parked is not None:
            order.insert(0, self.parked)
        done = None
        for i in order:
            sh = self.shards[i]
            if sh.live is not None:
                h, pre, states = sh.live
                sh.live = None
            elif L == 0:
                h, pre, states = sh.embed(emb, a['hc_mult'], dev)
            else:
                torch.cuda.empty_cache()                 # the last forward's cached blocks, before 17 GiB more
                h, pre, states = sh.load(dev, self.pool)
            if done is not None:
                done.store(self.pool)
                torch.cuda.empty_cache()
            rows_fn = self.engram_rows(L, li, i, engram, w) if engram is not None else None
            with torch.no_grad():
                pre = layer.forward(h, pre, sh.seqs, states, stats, rows_fn, self.chunk)
            sh.live = (h, pre, states)
            done = sh
            del h, pre, states
            relieve_memory()
        del emb
        self.parked = order[-1]
        torch.cuda.synchronize()
        self.next = L + 1
        return stats, host


CGROUP = Path('/sys/fs/cgroup')
RELIEF_ABOVE, RELIEF_TO = 0.90, 0.84     # shares of the container's memory limit


def cgroup_memory():
    """Memory charged to this container (GiB), when it runs under a cgroup v2 limit."""
    try:
        return round(int((CGROUP / 'memory.current').read_text()) / 2 ** 30, 1)
    except (OSError, ValueError):
        return None


def relieve_memory():
    """Under a container memory limit, give clean page cache back before the limit is reached. At the
    limit every allocation of the run waits in direct reclaim: with the pinned arena holding half the
    limit, one layer's pack and write once took 17 minutes instead of 40 s. No limit, no effect."""
    try:
        limit = (CGROUP / 'memory.max').read_text().strip()
        if limit == 'max':
            return
        used, limit = int((CGROUP / 'memory.current').read_text()), int(limit)
        if used > RELIEF_ABOVE * limit:
            (CGROUP / 'memory.reclaim').write_text(str(used - int(RELIEF_TO * limit)))
    except (OSError, ValueError):
        pass


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


def swept(path, grid):
    """True when sweep-LL.jsonl already holds every setting of the grid."""
    if not path.exists():
        return False
    have = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        have.add((row['name'], row['t'], row['damping'], row['hess_shrink']))
    return all(('hvq', *g) in have for g in grid)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--tok', default=ROOT, help='directory with tokenizer.json (Engram token map)')
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--d-stats', type=Path, required=True, help="the iVQ release's statistics (column weights D)")
    p.add_argument('--ivq', required=True, help='iVQ release: assembled GGUF or the directory of its parts')
    p.add_argument('--out', type=Path, required=True, help='blobs and receipts; the diagonal statistics under stats/')
    p.add_argument('--layers', default='0-39')
    p.add_argument('--t', type=float, default=None)
    p.add_argument('--damp', type=float, default=0.0)
    p.add_argument('--hess-shrink', type=float, default=256.0)
    p.add_argument('--world', type=int, default=8, help='shards; the iVQ statistics ran 8 ranks')
    p.add_argument('--max-tokens', type=int, default=0, help='per-shard token cap (smoke tests)')
    p.add_argument('--chunk', type=int, default=65536)
    p.add_argument('--engram-cache', type=Path, help='gather the Engram rows of every shard here once (hosts '
                   'whose page cache cannot hold a 98 GiB table); empty: read the tables through their memory map')
    p.add_argument('--save-hess', type=Path, help="write the layers' moments here instead of converting")
    p.add_argument('--sweep-grid', default='', help="'t:damp[:shrink],...': sweep the layers in place instead of converting")
    p.add_argument('--sweep-dump', type=Path, help='held-out MoE inputs (sweep_hvq_ds41.py dump)')
    p.add_argument('--vq', default='', help='sweep: the weights-only VQ base, for the record')
    p.add_argument('--probe', default='', help="accumulation precision probe, 'layer:expert,expert;layer:...'")
    a = p.parse_args()
    mode = 'save' if a.save_hess is not None else 'sweep' if a.sweep_grid else 'convert'
    if mode == 'convert' and a.t is None:
        raise SystemExit('--t is required to convert')
    if mode == 'sweep' and a.sweep_dump is None:
        raise SystemExit('--sweep-grid needs --sweep-dump')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    layers = cv.parse_layers(a.layers)
    probe = parse_probe(a.probe)
    grid = sw.parse_grid(a.sweep_grid, a.hess_shrink) if mode == 'sweep' else []
    data = load_file(str(a.calib))
    # one record per collection: the same inputs, shards and arithmetic give the same id. A layer's
    # moments do not depend on the layers after it, so the id does not name the layers run.
    run = {'collector': 'run_hvq_ds41.py', 'mode': 'collect-hess', 'calib_sha256': sha256_file(a.calib), 'world': a.world,
           'layers': 'from 0', 'max_tokens': a.max_tokens, 'rows': '', 'experts': 'teacher',
           'source': os.path.realpath(a.source), 'source_revision': hub_revision(a.source),
           'source_files': source_fingerprint(a.source)[0], 'tok': os.path.realpath(a.tok),
           'normalization': NORMALIZATION, 'precision': PRECISION}
    info = {'run_id': run_id(run), 'calib_sha256': run['calib_sha256'],
            'calib_content_sha256': content_sha256(data), 'run': run}
    del data
    var = cv.Variant(a.t, a.damp, a.hess_shrink) if mode == 'convert' else None
    todo = []
    for L in layers:
        if mode == 'save':
            wanted = not (a.save_hess / f'layer-{L:02d}' / 'meta.json').exists()
        elif mode == 'sweep':
            wanted = not swept(a.out / f'sweep-{L:02d}.jsonl', grid)
        else:
            st = statistics(a.d_stats, L, 'imatrix')
            wanted = not cv.reuse(L, a.out, {**var.want(), 'd_statistics_sha256': st[2], 'calibration_run_id': info['run_id']})
        if wanted:
            todo.append(L)
    emit({'phase': 'plan', 'mode': mode, 'run_id': info['run_id'], 'layers_wanted': todo, 'variant': var.want() if var else None})
    if not todo:
        return
    (a.out / 'stats').mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()
    col = Collector(a.source, a.tok, a.calib, a.world, a.max_tokens, a.chunk, engram_cache=a.engram_cache)
    emit({'phase': 'ready', 'seconds': round(time.monotonic() - t_all, 1)})
    hvq = hv.Hvq() if mode != 'save' else None
    writer, pending = ThreadPoolExecutor(1), None
    for L in range(max(todo) + 1):
        need = L in todo
        t0 = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        stats, host = col.step(L, need, probe.get(L, ()))
        collect = {'seconds': round(time.monotonic() - t0, 2),
                   'peak_gpu_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), 'moments': need,
                   'cgroup_memory_gib': cgroup_memory()}
        if not need:
            emit({'layer': L, 'phase': 'forward', **collect})
            del host
            continue
        hess = stats.hess
        hess.meta = {'run_id': info['run_id'], 'calib_content_sha256': info['calib_content_sha256'], 'layer': L,
                     'world': a.world, 'tokens': col.tokens}
        st = statistics(a.d_stats, L, 'imatrix')
        collect['d_statistics_agreement'] = agreement(stats.t, st[0])
        save_file({k: v.cpu().contiguous() for k, v in stats.t.items()}, str(a.out / 'stats' / f'layer-{L:02d}.safetensors'),
                  metadata={'run_id': info['run_id'], 'calib': str(a.calib), 'world': str(a.world)})
        if hess.probe:
            (a.out / 'stats' / f'probe-{L:02d}.json').write_text(json.dumps(hess.probe_report(), indent=1) + '\n')
            hess.probe = {}
        emit({'layer': L, 'phase': 'collected', **collect, **hess.describe()})
        del stats
        torch.cuda.empty_cache()
        if mode == 'save':
            del host
            t1 = time.monotonic()
            hess.save(a.save_hess / f'layer-{L:02d}', {'run': run, 'collect': collect, 'sha256': hess.digest()})
            emit({'layer': L, 'phase': 'saved', 'seconds': round(time.monotonic() - t1, 1)})
        elif mode == 'sweep':
            t1 = time.monotonic()
            sw.sweep_layer(L, col.teacher, host, hess, sw.held_out_x(a.sweep_dump, L), st, a.ivq, a.vq, grid, a.out, hvq)
            emit({'layer': L, 'phase': 'swept', 'seconds': round(time.monotonic() - t1, 1)})
            del host
        else:
            job = cv.Job(L, st, var, a.ivq, cv.HostExperts(host, L))
            del host
            job.fit(hess, hvq)
            if pending is not None:
                pending.result()
            # pack, strict decode check, write and receipt on the host while the next layer runs
            pending = writer.submit(job.finish, a.out, info, True, collect, False)
            del job
        del hess
        torch.cuda.empty_cache()
        relieve_memory()
    if pending is not None:
        pending.result()
    emit({'phase': 'done', 'seconds': round(time.monotonic() - t_all, 1), 'layers': todo})


if __name__ == '__main__':
    main()
