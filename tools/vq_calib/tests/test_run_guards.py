"""CPU test of the run guards (synthetic rank directories, checkpoints and receipts; no model, no GPU):
reduce_stats.py reduces exactly one run's files, collect_stats.py resumes only its own inputs and never
skips layers, run_ds41.py refuses partial ranges, converters reuse a layer only for the requested
variant, and both assemblers require one calibration run."""
import hashlib, json, os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True
import torch
from safetensors.torch import save_file
import reduce_stats as rs
import collect_stats as cs
import run_ds41
import convert_layer as cl
import convert_layer_ds41 as cl41
import assemble_release as ar
import assemble_ds41_ivq as a41

E, H = 4, 8


def refused(fn, what, *needles):
    try:
        fn()
    except (SystemExit, ValueError) as e:
        msg = str(e)
        assert all(n in msg for n in needles), (what, msg)
        return msg
    raise AssertionError('accepted: ' + what)


def stats_tensors(tokens, seed):
    g = torch.Generator().manual_seed(seed)
    return {'tokens': torch.tensor([float(tokens)], dtype=torch.float64),
            'count': torch.randint(0, 50, (E,), generator=g, dtype=torch.int64),
            'w2': torch.rand(E, generator=g, dtype=torch.float64), 'x2': torch.rand(E, H, generator=g, dtype=torch.float64)}


def make_run(stats, world=2, layers=(0, 1, 2), calib='/c/calib.safetensors', legacy=False, head=False, seed=0):
    """Rank directories as the collectors write them -> run id (None for a legacy run)."""
    run = {'collector': 'test', 'calib_sha256': hashlib.sha256(calib.encode()).hexdigest(), 'world': world,
           'layers': list(layers), 'max_tokens': 0}
    rid = None if legacy else rs.run_id(run)
    files = [f'layer-{L:02d}.safetensors' for L in layers] + (['head.safetensors'] if head else [])
    for r in range(world):
        d = stats / f'rank-{r}'
        d.mkdir(parents=True, exist_ok=True)
        shard = {'calib': calib, 'rows': [r], 'tokens': 100 + r, 'world': world, 'rank': r}
        stamp = {'calib': calib, 'world': str(world), 'rank': str(r)}
        if not legacy:
            shard.update(run=run, run_id=rid)
            stamp['run_id'] = rid
        (d / 'shard.json').write_text(json.dumps(shard))
        for i, f in enumerate(files):
            save_file(stats_tensors(100 + r, seed + 7 * i + r), str(d / f), metadata=stamp)
        done = {'seconds': 1.0, 'layers': list(layers)}
        if not legacy:
            done.update(files=files, run_id=rid)
        (d / 'DONE').write_text(json.dumps(done))
    return rid


def reduce(stats):
    sys.argv = ['reduce_stats.py', '--stats', str(stats)]
    rs.main()
    return json.loads((stats / 'summary.json').read_text())


_tmp = tempfile.TemporaryDirectory(prefix='run-guards-')
tmp = Path(_tmp.name)

# ---- reduce_stats.py: exactly the run's files -------------------------------------------------
s = tmp / 'stats-a'
make_run(s, layers=(0, 1, 2, 3, 4), seed=1)            # an earlier, wider run (and its reduction)
reduce(s)
rid = make_run(s, layers=(0, 1, 2), seed=2)             # the current run, narrower, same calib and world
summary = reduce(s)
assert summary['run_id'] == rid and summary['files'] == [f'layer-{L:02d}.safetensors' for L in (0, 1, 2)]
assert sorted(summary['layers']) == summary['files'], summary['layers'].keys()
assert summary['ignored_files'] == ['rank-0/layer-03.safetensors', 'rank-0/layer-04.safetensors',
                                    'rank-1/layer-03.safetensors', 'rank-1/layer-04.safetensors']
assert summary['outputs_not_in_run'] == ['layer-03.safetensors', 'layer-04.safetensors']
assert summary['unverified_files'] == 0 and summary['unstamped_files'] == 0
for name, row in summary['layers'].items():
    assert row['sha256'] == rs.sha256_file(s / name) and row['tokens'] == 201
print('reduce: stale layer files outside the run are left out and reported: ok')

# a stale file inside the run's layer list (another run's stamp), an unstamped one, a stale DONE
t = tmp / 'stats-b'
make_run(t, layers=(0, 1, 2, 3), seed=3)
other = sorted(t.glob('rank-1/layer-03.safetensors'))[0].read_bytes()
make_run(t, layers=(0, 1, 2), seed=4)
(t / 'rank-1/layer-02.safetensors').write_bytes(other)   # same calib/world/rank, other layer list
refused(lambda: reduce(t), 'file of another run', 'rank-1/layer-02.safetensors', 'another run')
save_file(stats_tensors(101, 9), str(t / 'rank-1/layer-02.safetensors'))
refused(lambda: reduce(t), 'unstamped file in a stamped run', 'rank-1/layer-02.safetensors')
make_run(t, layers=(0, 1, 2), seed=4)
done = json.loads((t / 'rank-0/DONE').read_text())
(t / 'rank-0/DONE').write_text(json.dumps({**done, 'run_id': 'f' * 64}))
refused(lambda: reduce(t), 'stale DONE', 'rank-0', 'not from one run')
(t / 'rank-0/DONE').write_text(json.dumps({**done, 'layers': [0, 1], 'files': done['files'][:2]}))
refused(lambda: reduce(t), 'ranks with different layer lists', 'different layer lists')
print('reduce: foreign/unstamped files in the run, stale DONE, disagreeing ranks refused: ok')

# legacy rank directories (no run ids): reduced, counted as unverified; head kept when every rank has it
u = tmp / 'stats-legacy'
make_run(u, layers=(0, 1), legacy=True, head=True, seed=5)
summary = reduce(u)
assert summary['run_id'] is None and summary['unverified_files'] == 6 and summary['unstamped_files'] == 0
assert summary['files'] == ['layer-00.safetensors', 'layer-01.safetensors', 'head.safetensors']
print('reduce: legacy run reduced as unverified: ok')

# ---- reduce_stats.calibration_run: the statistics a converter reads belong to one run ----------
assert rs.calibration_run(s, [str(s / 'layer-01.safetensors')])[0] == rid
assert rs.calibration_run(s, []) == ('none', None)
assert rs.calibration_run(u, [str(u / 'layer-01.safetensors')]) == (None, None)
refused(lambda: rs.calibration_run(s, [str(s / 'layer-03.safetensors')]), 'stale merged file', 'stale statistics')
v = tmp / 'stats-ranks'
rid_v = make_run(v, layers=(0, 1), seed=6)
assert rs.calibration_run(v, [str(p) for p in sorted(v.glob('rank-*/layer-01.safetensors'))])[0] == rid_v
make_run(tmp / 'stats-old3', world=3, layers=(0, 1), seed=7)
os.rename(tmp / 'stats-old3/rank-2', v / 'rank-2')         # stale rank of a 3-GPU run
refused(lambda: rs.calibration_run(v, [str(p) for p in sorted(v.glob('rank-*/layer-01.safetensors'))]),
        'stale rank directory', 'not exactly the ranks of one run')
print('calibration_run: merged and rank statistics verified, stale ones refused: ok')

# ---- collect_stats.py: checkpoint fingerprint and layer ranges ----------------------------------
fp = {'collector': 'collect_stats.py', 'calib_sha256': 'c' * 64, 'world': 2, 'max_tokens': 0, 'source': '/m',
      'source_revision': 'r', 'rank': 0, 'rows_sha256': 'a' * 64, 'heldout_sha256': None}
ck = tmp / 'h-checkpoint.pt'
L60 = list(range(61))
refused(lambda: cs.resume_point(ck, fp, list(range(8, 61)), False, 'cpu'), 'range not starting at 0', 'skips layers 0-7')
refused(lambda: cs.resume_point(ck, fp, list(range(8, 61)), True, 'cpu'), 'range not at 0, no checkpoint', 'skips')
assert cs.resume_point(ck, fp, L60, True, 'cpu') == (None, 0)
torch.save({'h': torch.ones(3), 'next_layer': 8, 'fingerprint': fp, 'layers': L60, 'run_id': 'x'}, ck)
state, start = cs.resume_point(ck, fp, L60, True, 'cpu')
assert start == 8 and torch.equal(state['h'], torch.ones(3))
assert cs.resume_point(ck, fp, list(range(8, 61)), True, 'cpu')[1] == 8       # a new run continuing at 8
refused(lambda: cs.resume_point(ck, fp, list(range(16, 61)), True, 'cpu'), 'range start != next layer', 'belongs to the run')
refused(lambda: cs.resume_point(ck, fp, list(range(31)), True, 'cpu'), 'other layer list', 'belongs to the run')
for key, value in (('calib_sha256', 'd' * 64), ('rows_sha256', 'b' * 64), ('world', 3), ('heldout_sha256', 'e' * 64),
                   ('source', '/other')):
    refused(lambda: cs.resume_point(ck, {**fp, key: value}, L60, True, 'cpu'), f'checkpoint of other {key}', key)
torch.save({'h': torch.ones(3), 'next_layer': 8}, ck)
refused(lambda: cs.resume_point(ck, fp, L60, True, 'cpu'), 'checkpoint without fingerprint', 'none recorded')
print('collect_stats: mismatched or legacy checkpoints and skipped layers refused: ok')

# ---- run_ds41.py: no partial layer ranges (eval only with --debug-partial) ----------------------
base_args = ['--data', str(tmp / 'none.safetensors'), '--out', str(tmp / 'ds41-out'), '--layers', '8-39']
sys.argv = ['run_ds41.py', 'collect'] + base_args
refused(run_ds41.main, 'ds41 collect from layer 8', 'skips layers 0-7')
sys.argv = ['run_ds41.py', 'collect', '--debug-partial'] + base_args
refused(run_ds41.main, 'ds41 collect from layer 8 with --debug-partial', 'skips layers 0-7')
sys.argv = ['run_ds41.py', 'eval'] + base_args
refused(run_ds41.main, 'ds41 eval from layer 8', '--debug-partial')


class Reached(Exception):
    pass


def no_teacher(*a, **k):
    raise Reached()


run_ds41.Teacher = no_teacher
sys.argv = ['run_ds41.py', 'eval', '--debug-partial'] + base_args
try:
    run_ds41.main()
    raise AssertionError('no teacher')
except Reached:
    pass
assert not (tmp / 'ds41-out').exists()
print('run_ds41: partial ranges refused (eval allowed with --debug-partial): ok')

# ---- convert_layer.py resume: Lloyd iterations and calibration run -----------------------------
out, work = tmp / 'parts', tmp / 'work'
out.mkdir()
(out / 'layer-01-experts.gguf').write_bytes(b'part')
stats_sha = cl.stats_fingerprint([str(s / 'layer-01.safetensors')])
receipt = {'sha256': hashlib.sha256(b'part').hexdigest(), 'k': 512, 'recipe_id': 'r', 'gain_refit': 'ls',
           'codebook_scope': 'split3', 'calibration_run_id': rid,
           'weighting': {'mode': 'imatrix', 'shrink_tokens': 256.0, 'col_power': 0.75, 'col_floor': 0.02, 'col_shrink': 0.0},
           'training': {'books': [{'iterations': [{}] * 32}] * 3, 'statistics_sha256': stats_sha}}
(out / 'layer-01-experts.json').write_text(json.dumps(receipt))
recipe = {'recipe_id': 'r', 'lloyd_iterations': 32}
src = SimpleNamespace(config={})


def resume(iterations=None, stats=s):
    return cl.convert_layer(src, None, 1, 512, recipe, out, work, 'imatrix', 'ls', stats_dir=stats, iterations=iterations,
                            books_scope='split3', col_power=0.75)


assert resume()['sha256'] == receipt['sha256']
assert resume(32)['sha256'] == receipt['sha256']
refused(lambda: resume(16), 'resume with other iterations', 'lloyd_iterations', '(32, 16)')
w2 = tmp / 'stats-a2'
make_run(w2, layers=(0, 1, 2), seed=2, calib='/c/other.safetensors')
reduce(w2)
refused(lambda: resume(stats=w2), 'resume from another calibration run', 'calibration_run_id')
(out / 'layer-01-experts.json').write_text(json.dumps({**receipt, 'training': {'books': [
    {'iterations': [{}] * 32}, {'iterations': [{}] * 16}, {'iterations': [{}] * 32}]}}))
refused(resume, 'books with mixed iterations', 'different iteration counts')
print('convert_layer: resume refused for other Lloyd iterations or calibration run: ok')

# ---- convert_layer_ds41.py reuse --------------------------------------------------------------
b = tmp / 'ds41'
b.mkdir()
blob = bytes(range(256)) * 4
want = {'mode': 'imatrix', 'gain_refit': 'ls', 'lloyd_iterations': 32, 'col_power': 0.75, 'shrink_tokens': 256.0,
        'col_floor': 0.02, 'statistics_sha256': 'f' * 64, 'calibration_run_id': rid}


def ds41_receipt(L, **over):
    r = {'layer': L, 'tensor': f'blk.{L}.ffn_exps_vq.blob', 'K': 4096, 'sha256': hashlib.sha256(blob).hexdigest(),
         'mode': 'imatrix', 'gain_refit': 'ls', 'lloyd_iterations': 32, 'codebook_scope': 'shared',
         'weighting': {'mode': 'imatrix', 'shrink_tokens': 256.0, 'shrink_units': 256.0 + L, 'col_floor': 0.02,
                       'col_shrink': 0.0, 'col_power': 0.75, 'tensor_scale': None},
         'statistics_sha256': 'f' * 64, 'calibration_run_id': rid, 'calibration_set_sha256': 'c' * 64,
         'training': {'history': [{}] * 32}}
    r.update(over)
    return r


(b / 'blob-20.bin').write_bytes(blob)
assert not cl41.reuse(21, b, want)
(b / 'receipt-20.json').write_text(json.dumps(ds41_receipt(20)))
assert cl41.reuse(20, b, want)
for key, value in (('lloyd_iterations', 16), ('mode', 'imatrix_freq'), ('gain_refit', 'none'), ('col_power', 1.0),
                   ('shrink_tokens', 64.0), ('col_floor', 0.05), ('statistics_sha256', 'e' * 64),
                   ('calibration_run_id', 'a' * 64)):
    refused(lambda: cl41.reuse(20, b, {**want, key: value}), f'ds41 reuse with other {key}', key)
legacy = ds41_receipt(20)
for k in ('statistics_sha256', 'calibration_run_id', 'calibration_set_sha256'):
    del legacy[k]
(b / 'receipt-20.json').write_text(json.dumps(legacy))
assert cl41.reuse(20, b, want)                                   # older receipt: unverified, not refused
(b / 'blob-20.bin').write_bytes(blob[:-1] + b'\0')
refused(lambda: cl41.reuse(20, b, want), 'blob that is not the receipt\'s', 'sha256')
print('convert_layer_ds41: reuse refused for any other variant, statistics or blob: ok')

# ---- assemblers: one calibration run -----------------------------------------------------------
rec = {l: {'calibration_run_id': rid, 'calibration_set_sha256': 'c' * 64} for l in range(4)}
assert ar.calibration_of(rec, False) == {'run_id': rid, 'calibration_set_sha256': 'c' * 64, 'unverified_layers': []}
refused(lambda: ar.calibration_of({**rec, 2: {'calibration_run_id': 'a' * 64, 'calibration_set_sha256': 'c' * 64}}, True),
        'mixed calibration runs', 'different calibration runs')
mixed = {**rec, 3: {}}
refused(lambda: ar.calibration_of(mixed, False), 'legacy receipt without the flag', '[3]', '--allow-unverified-calibration')
cal = ar.calibration_of(mixed, True)
assert cal['run_id'] == rid and cal['unverified_layers'] == [3]
variant = {'mode': 'imatrix'}
ar.check_selection({'calibration': {'calibration_run_id': rid, 'calibration_set': {'safetensors_sha256': 'c' * 64}}},
                   variant, ar.calibration_of(rec, False))
refused(lambda: ar.check_selection({'calibration': {'calibration_run_id': 'a' * 64}}, variant, ar.calibration_of(rec, False)),
        'recipe names another run', 'calibration_run_id')
refused(lambda: ar.check_selection({'calibration': {'calibration_run_id': rid}}, variant, cal),
        'recipe run with unverified layers', 'unverified_layers')
refused(lambda: ar.check_selection({'calibration': {'calibration_set': {'safetensors_sha256': 'd' * 64}}}, variant,
                                   ar.calibration_of(rec, False)), 'recipe names another calibration set', 'calibration_set_sha256')

d41 = tmp / 'ds41-release'
d41.mkdir()


def write_receipts(**per_layer):
    for L in range(40):
        (d41 / f'receipt-{L:02d}.json').write_text(json.dumps(ds41_receipt(L, **per_layer.get(str(L), {}))))


write_receipts()
_, v41, c41 = a41.load_receipts(d41, False)
assert c41 == {'run_id': rid, 'calibration_set_sha256': 'c' * 64, 'unverified_layers': []} and v41['lloyd_iterations'] == 32
write_receipts(**{'7': {'calibration_run_id': 'a' * 64}})
refused(lambda: a41.load_receipts(d41, True), 'ds41 mixed calibration runs', 'different calibration runs')
write_receipts(**{'7': {'training': {'history': [{}] * 16}}})
refused(lambda: a41.load_receipts(d41, True), 'ds41 mixed iterations', 'different variants')
write_receipts(**{'7': {'gain_refit': 'none'}})
refused(lambda: a41.load_receipts(d41, True), 'ds41 mixed gain refit', 'different variants')
write_receipts(**{str(L): {'calibration_run_id': None, 'calibration_set_sha256': None} for L in range(40)})
refused(lambda: a41.load_receipts(d41, False), 'ds41 legacy receipts without the flag', '--allow-unverified-calibration')
assert a41.load_receipts(d41, True)[2] == {'run_id': None, 'calibration_set_sha256': None, 'unverified_layers': list(range(40))}
print('assemblers: mixed calibration runs refused, legacy receipts only with the flag: ok')
print('all run guards: ok')
