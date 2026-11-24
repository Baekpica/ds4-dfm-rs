#!/usr/bin/env python3
"""K2-Horizon-375B-A23B VQ: the non-expert parts of the release, from the original BF16 checkpoint (CPU).

    body   -> parts/body-nonexpert.gguf      (part_role trunk): per layer the attention q/k/v/o projections,
              the dense MLP (layers 0-2) or the shared expert (layers 3-60) as Q8_0; the router weight and
              bias and every RMSNorm weight as F32; `model.norm.weight` (F32) and `lm_head.weight` (Q8_0)
    embed  -> parts/embed-tokens-q8_0.gguf   (part_role embedding): `model.embed_tokens.weight` as Q8_0

Source tensor names are kept. Q8_0 is the encoder of this checkout's gguf-tools/quants.c
(ds4q_quantize_q8_0, the same arithmetic as ggml's quantize_row_q8_0_ref): per 32 weights
d = amax / 127 in F32 (stored F16), q = roundf(x * (1 / d)). It runs here as torch CPU tensor operations in
F32 (roundf through FP64, exact); `--check-c` builds the C function and compares sampled tensors byte for byte.
Every written part is re-read by the strict family reader and sampled blocks are decoded against the source.

    convert_nonexpert_k2h.py --source $SRC --parts $W/parts [--only body|embed]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import k2h_ref as R
import k2h_vq_format as fmt

NS = fmt.NS
TOOLS = Path(__file__).resolve().parent
ENCODER = ('Q8_0 as ds4-dfm-rs gguf-tools/quants.c ds4q_quantize_q8_0 (ggml quantize_row_q8_0_ref): per 32 weights '
           'd = amax / 127 in F32, stored F16; q = roundf(x * (1 / d)); torch CPU tensor operations')
CHUNK = 1 << 23


def q8_0(x):
    """x F32 [..., 32 n] -> (uint8 [blocks, 34], sum of squared error, sum of squares) of the stored blocks."""
    v = x.reshape(-1, 32)
    d = v.abs().amax(1) / 127.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    t = (v * inv[:, None]).double()
    q = torch.where(t >= 0, torch.floor(t + 0.5), torch.ceil(t - 0.5))
    d16 = d.to(torch.float16)
    out = np.empty((v.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d16.numpy().view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.to(torch.int8).numpy().view(np.uint8)
    err = (q.float() * d16.float()[:, None] - v).double().square().sum()
    return out, float(err), float(v.double().square().sum())


def body_inventory():
    out = []
    for l in range(R.L):
        p = f'model.layers.{l}.'
        out.append((p + 'input_layernorm.weight', 'f32'))
        out += [(p + f'self_attn.{n}_proj.weight', 'q8_0') for n in 'qkvo']
        out.append((p + 'post_attention_layernorm.weight', 'f32'))
        if l < R.DENSE:
            out += [(p + f'mlp.{n}_proj.weight', 'q8_0') for n in ('gate', 'up', 'down')]
        else:
            out += [(p + 'mlp.gate.weight', 'f32'), (p + 'mlp.gate.bias', 'f32')]
            out += [(p + f'mlp.shared_experts.{n}_proj.weight', 'q8_0') for n in ('gate', 'up', 'down')]
    out += [('model.norm.weight', 'f32'), ('lm_head.weight', 'q8_0')]
    return out


EMBED_INVENTORY = [('model.embed_tokens.weight', 'q8_0')]
PARTS = {'body': ('body-nonexpert.gguf', 'trunk', body_inventory),
         'embed': ('embed-tokens-q8_0.gguf', 'embedding', lambda: EMBED_INVENTORY)}


def build_c_check(work):
    """ds4q_quantize_q8_0's loop as a shared object (the function is static in quants.c)."""
    work.mkdir(parents=True, exist_ok=True)
    src = work / 'q8_0_ref.c'
    src.write_text('''#include <math.h>
#include <stdint.h>
#include <string.h>
static uint16_t f32_to_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
void q8_0_ref(const float *src, uint8_t *out, int64_t nblocks) {
    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) { const float av = fabsf(x[j]); if (av > amax) amax = av; }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = f32_to_f16(d);
        memcpy(out, &hd, 2);
        int8_t *qs = (int8_t *)(out + 2);
        for (int j = 0; j < 32; j++) qs[j] = (int8_t)roundf(x[j] * id);
        out += 34;
    }
}
''')
    lib = work / 'libq8_0_ref.so'
    subprocess.run(['gcc', '-std=c11', '-O2', '-fPIC', '-shared', '-ffp-contract=off', str(src), '-lm', '-o', str(lib)], check=True)
    dll = ct.CDLL(str(lib))
    dll.q8_0_ref.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64]
    return dll


def convert(src, kind, parts, pool, check=None):
    fname, role, inv = PARTS[kind]
    inventory = inv()
    out = parts / fname
    rpath = parts / 'receipts' / (out.stem + '.json')
    rpath.parent.mkdir(parents=True, exist_ok=True)
    entries, shapes = [], {}
    for name, typ in inventory:
        _, _, size, shape = src.info(name)
        elements = size // 2
        shapes[name] = shape
        if typ == 'q8_0':
            if len(shape) != 2 or shape[1] % 32:
                raise ValueError(f'{name}: Q8_0 shape {shape}')
            entries.append({'name': name, 'shape': list(shape), 'type_id': fmt.Q8_0, 'payload_bytes': elements // 32 * 34})
        else:
            entries.append({'name': name, 'shape': list(shape), 'type_id': fmt.F32, 'payload_bytes': elements * 4})
    policy = {'q8_0': 'attention q/k/v/o, dense MLP (layers 0-2), shared experts (layers 3-60), output head' if kind == 'body'
              else 'token embedding [250624, 6144]; 6,528 bytes per row (192 blocks of 34)',
              'f32': 'router weight and bias, RMSNorm weights (BF16 source values, exact)' if kind == 'body' else None,
              'encoder': ENCODER, 'names': 'source tensor names'}
    extra = {NS + 'source_revision': fmt.SOURCE_REVISION, NS + 'recipe_id': fmt.RECIPE_ID,
             NS + 'native_policy': json.dumps({k: v for k, v in policy.items() if v}, sort_keys=True)}
    if kind == 'embed':
        extra[NS + 'residency'] = 'resident-or-host-rows'
    meta = fmt.metadata(part_role=role, extra=extra)
    tmp = out.with_suffix('.gguf.tmp')
    if tmp.exists():
        tmp.unlink()
    rows, t0, checked = [], time.time(), 0
    with fmt.PartWriter(tmp, meta, entries) as w:
        for (name, typ), entry in zip(inventory, entries):
            x = src.tensor(name)
            if typ == 'f32':
                payload = x.float().numpy().astype('<f4').tobytes()
                row = {'rel_rms_error': 0.0}
            else:
                flat = x.reshape(-1)
                jobs = [(s, min(flat.numel(), s + CHUNK)) for s in range(0, flat.numel(), CHUNK)]
                res = list(pool.map(lambda j: q8_0(flat[j[0]:j[1]].float()), jobs))
                payload = b''.join(r[0].tobytes() for r in res)
                err, en = sum(r[1] for r in res), sum(r[2] for r in res)
                row = {'rel_rms_error': (err / en) ** 0.5 if en else 0.0}
                if check is not None and (x.numel() <= 40_000_000 and len(rows) % 16 == 0 or name in ('lm_head.weight', 'model.embed_tokens.weight')):
                    n = min(flat.numel(), 32 * 2_000_000)
                    f = np.ascontiguousarray(flat[:n].float().numpy())
                    ref = np.empty(n // 32 * 34, dtype=np.uint8)
                    check.q8_0_ref(f.ctypes.data, ref.ctypes.data, n // 32)
                    if ref.tobytes() != payload[:len(ref)]:
                        raise RuntimeError(f'{name}: torch Q8_0 differs from the C encoder')
                    checked += 1
            if len(payload) != entry['payload_bytes']:
                raise RuntimeError(f'{name}: payload size')
            w.write(name, payload)
            rows.append({'name': name, 'shape': list(shapes[name]), 'type': typ, 'type_id': entry['type_id'],
                         'payload_bytes': len(payload), 'payload_sha256': hashlib.sha256(payload).hexdigest(), **row})
            del x, payload
    with tmp.open('rb') as f:
        file_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    size = tmp.stat().st_size
    tmp.rename(out)
    # strict family reader + independent numpy decode of sampled Q8_0 blocks against the source
    rng = np.random.default_rng(20261124)
    decoded = 0
    with fmt.PartReader(out) as r:
        if r.order != [n for n, _ in inventory]:
            raise RuntimeError('part inventory/order')
        for row in rows:
            e = r.entries[row['name']]
            if e['type_id'] != row['type_id'] or e['payload_bytes'] != row['payload_bytes'] or list(e['shape']) != row['shape']:
                raise RuntimeError(f"{row['name']}: descriptor")
        for row in [rows[i] for i in rng.choice(len(rows), min(len(rows), 48), replace=False)]:
            raw = r.raw(row['name'])
            x = src.tensor(row['name']).float().numpy()
            if row['type'] == 'f32':
                ok = np.array_equal(np.frombuffer(raw, '<f4'), x.reshape(-1))
                blk = None
            else:
                blk = np.frombuffer(raw, np.uint8).reshape(-1, 34)
                pick = rng.integers(0, len(blk), 4096)
                d = blk[pick, :2].copy().view('<f2').astype(np.float32)
                q = blk[pick, 2:].view(np.int8).astype(np.float32)
                ref = x.reshape(-1, 32)[pick]
                # half a step of the stored scale, plus the F16 rounding of the scale itself (relative 2^-11
                # for normal scales, absolute 2^-25 per unit of q for subnormal ones)
                ok = bool(np.all(np.abs(q * d - ref) <= d * 0.502 + np.abs(ref) * 2e-3 + 127 * 2.0 ** -25))
            del blk
            raw.release()
            if not ok:
                raise RuntimeError(f"{row['name']}: decode check")
            decoded += 1
    receipt = {'path': out.name, 'part_role': role, 'bytes': size, 'sha256': file_sha, 'recipe_id': fmt.RECIPE_ID,
               'source_revision': fmt.SOURCE_REVISION, 'encoder': ENCODER, 'native_policy': json.loads(extra[NS + 'native_policy']),
               'tensor_count': len(rows), 'q8_0_tensors': sum(r_['type'] == 'q8_0' for r_ in rows),
               'f32_tensors': sum(r_['type'] == 'f32' for r_ in rows),
               'q8_0_rel_rms_error_max': max((r_['rel_rms_error'] for r_ in rows), default=0.0),
               'c_encoder_checked_tensors': checked, 'decoded_tensors_checked': decoded,
               'seconds': round(time.time() - t0, 1), 'tensors': rows}
    rpath.write_text(json.dumps(receipt, indent=1) + '\n')
    return {k: v for k, v in receipt.items() if k != 'tensors'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--parts', type=Path, required=True)
    ap.add_argument('--only', choices=sorted(PARTS))
    ap.add_argument('--threads', type=int, default=20)
    ap.add_argument('--check-c', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    src = R.Source(a.source)
    check = build_c_check(a.parts / 'receipts' / 'c-check') if a.check_c else None
    with ThreadPoolExecutor(a.threads) as pool:
        for kind in ([a.only] if a.only else ['embed', 'body']):
            out = a.parts / PARTS[kind][0]
            if out.exists():
                print(json.dumps({'part': out.name, 'status': 'exists'}), flush=True)
                continue
            print(json.dumps(convert(src, kind, a.parts, pool, check)), flush=True)
            src.close()


if __name__ == '__main__':
    main()
