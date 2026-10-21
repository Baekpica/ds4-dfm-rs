#!/usr/bin/env python3
"""Convert every routed-expert layer of a pinned A.X-K2 recipe, resumably, on one or more GPUs.

With `--gpus 0,1,...` one worker process per GPU takes the layers in
round-robin order (layer list sorted, worker i gets layers i, i+n, ...).
Inside a worker, layer N+1's FP8 experts are read into pinned host memory
while layer N converts. Finished layers are verified against their receipts
and skipped, so a rerun resumes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent


def parse_layers(spec, all_layers):
    if not spec:
        return all_layers
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    return [l for l in all_layers if l in out]


def worker(a, layers):
    import torch
    from axk2_source import Source, SOURCE_REV
    from axk2_vq_gpu import GpuVq
    from convert_layer import convert_layer, emit, read_host
    spec = a.recipe.read_bytes()
    recipe = json.loads(spec)
    if recipe['source_revision'] != SOURCE_REV or recipe['status'] != 'selected_for_full_conversion':
        raise ValueError('recipe must be pinned and selected before full conversion')
    source = Source(a.source)
    gpu = GpuVq()
    a.work.mkdir(parents=True, exist_ok=True)
    state_path = a.work / f'run-status-{a.worker_id}.json'
    state = {'status': 'running', 'started_unix': time.time(), 'recipe_sha256': hashlib.sha256(spec).hexdigest(),
             'mode': a.mode, 'gain_refit': a.gain_refit, 'layers': layers, 'completed_layers': []}
    reader = ThreadPoolExecutor(1)
    done = lambda l: (a.output / f'layer-{l:02d}-experts.json').exists()
    todo = [l for l in layers if not done(l)]
    pending = reader.submit(read_host, source, todo[0], recipe.get('read_threads', 16)) if todo else None
    for i, layer in enumerate(layers):
        host = None
        if not done(layer):
            t = time.monotonic()
            host = pending.result()
            emit({'phase': 'prefetched', 'layer': layer, 'wait_seconds': time.monotonic() - t})
            nxt = next((l for l in layers[i + 1:] if not done(l)), None)
            pending = reader.submit(read_host, source, nxt, recipe.get('read_threads', 16)) if nxt is not None else None
        receipt = convert_layer(source, gpu, layer, recipe['layer_k'][str(layer)], recipe, a.output, a.work,
                                a.mode, a.gain_refit, stats_dir=a.stats, host=host, shrink_tokens=a.shrink_tokens,
                                books_scope=a.books, col_power=a.col_power, col_floor=a.col_floor)
        del host
        state['completed_layers'].append({'layer': layer, 'k': receipt['k'], 'sha256': receipt['sha256'],
                                          'rel_mse': receipt['quality']['rel_mse'],
                                          'imatrix_rel_err': receipt['quality'].get('imatrix_rel_err')})
        state['last_update_unix'] = time.time()
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        torch.cuda.empty_cache()
    state['status'] = 'completed'
    state['finished_unix'] = time.time()
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    emit({'phase': 'worker_done', 'worker': a.worker_id, 'layers': len(state['completed_layers'])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/models/A.X-K2')
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--stats', type=Path, default=Path('/workspace/axk2/stats'))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layers', help='subset, e.g. 1-7,30')
    p.add_argument('--mode', help='weighting mode (default: the recipe selection)')
    p.add_argument('--gain-refit', help='none | ls (default: the recipe selection)')
    p.add_argument('--shrink-tokens', type=float)
    p.add_argument('--books', choices=['shared', 'split2', 'split3'])
    p.add_argument('--col-power', type=float)
    p.add_argument('--col-floor', type=float)
    p.add_argument('--gpus', help='comma list of GPU ids: one worker process per GPU')
    p.add_argument('--worker-id', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--worker-layers', help=argparse.SUPPRESS)
    a = p.parse_args()
    sys.path.insert(0, str(TOOLS))
    from convert_layer import weighting_config
    from axk2_weighting import MODES
    recipe = json.loads(a.recipe.read_text())
    wcfg = weighting_config(recipe)
    a.mode = a.mode or wcfg.get('mode')
    a.gain_refit = a.gain_refit or wcfg.get('gain_refit')
    if a.mode not in MODES or a.gain_refit not in ('none', 'ls'):
        raise SystemExit('no weighting selected in the recipe: pass --mode and --gain-refit')
    all_layers = sorted(int(l) for l in recipe['layer_k'])
    if a.worker_layers is not None:
        worker(a, [int(x) for x in a.worker_layers.split(',') if x])
        return
    layers = parse_layers(a.layers, all_layers)
    if not a.gpus:
        worker(a, layers)
        return
    gpus = [g for g in a.gpus.split(',') if g]
    procs = []
    a.work.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(gpus):
        mine = layers[i::len(gpus)]
        if not mine:
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), '--source', a.source, '--recipe', str(a.recipe),
               '--stats', str(a.stats), '--work', str(a.work), '--output', str(a.output),
               '--mode', a.mode, '--gain-refit', a.gain_refit, '--worker-id', str(i),
               '--worker-layers', ','.join(map(str, mine))]
        if a.shrink_tokens is not None:
            cmd += ['--shrink-tokens', str(a.shrink_tokens)]
        if a.books:
            cmd += ['--books', a.books]
        if a.col_power is not None:
            cmd += ['--col-power', str(a.col_power)]
        if a.col_floor is not None:
            cmd += ['--col-floor', str(a.col_floor)]
        log = (a.work / f'worker-{i}-gpu{g}.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append((g, mine, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(TOOLS))))
        print(json.dumps({'phase': 'spawned', 'gpu': g, 'layers': mine}), flush=True)
    failed = []
    for g, mine, proc in procs:
        if proc.wait() != 0:
            failed.append({'gpu': g, 'layers': mine, 'returncode': proc.returncode})
    missing = [l for l in layers if not (a.output / f'layer-{l:02d}-experts.json').exists()]
    print(json.dumps({'phase': 'done', 'failed_workers': failed, 'missing_layers': missing}), flush=True)
    if failed or missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
