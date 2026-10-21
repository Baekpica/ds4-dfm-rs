#!/usr/bin/env python3
"""A.X-K2 non-expert tensors -> body-nonexpert.gguf (trunk) + embed-tokens-bf16.gguf (embedding).

Policy (recipe.json `nonexpert`):
  * Q4_K (ds4-dfm-rs gguf-tools/quants.c, -ffp-contract=off; the GGML
    `quantize_row_q4_K_impl` path when a per-column importance vector is
    given, the reference path otherwise): attention q_a/q_b/kv_a/kv_b/o of all
    61 layers, the shared experts of the 60 MoE layers, the dense FFN of
    layer 0, lm_head.
  * Q8_0: indexer wq_b and wk.
  * Preserved bytes and dtype: router weight (BF16) and correction bias (F32),
    indexer weights_proj and k_norm, q_a/kv_a layernorms, every RMSNorm and
    gated-norm gate (W_down, W_up, norm), the final norm.
  * The token embedding stays BF16 in its own part, read in rows from SSD by
    the host; it is never uploaded whole.
FP8 sources are dequantized with axk2_source.dequant (FP32, bit-identical to
llama.cpp's dequant_simple). Every quantized payload is decoded twice (numpy
and an independent C decoder) and both must agree before the error
statistics are recorded. With --imatrix-dir, each projection's encoder gets
the mean x^2 of its input columns from the calibration statistics.

Forked from the Beam-1.0 nonexpert converter.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from axk2_source import Source, SOURCE_REV, dequant
from axk2_vq_format import PartWriter, PartReader, metadata, F32, F16, BF16, Q4_K, Q8_0

TOOLS = Path(__file__).resolve().parent
DS4 = Path(__file__).resolve().parents[2]          # repository root (tools/vq_calib/../..)
TYPES = {'BF16': BF16, 'F32': F32, 'F16': F16}
QBYTES = {Q4_K: (256, 144), Q8_0: (32, 34)}
IMATRIX_FLOOR = 1e-8          # relative to the vector mean; keeps all-zero blocks well defined

DECODE_C = r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "quants.h"
void axk2_decode_q4(const uint8_t *x, float *y, size_t blocks) {
    for (size_t b=0;b<blocks;b++,x+=144,y+=256) {
        uint16_t hd,hm; memcpy(&hd,x,2); memcpy(&hm,x+2,2);
        const float d=ds4q_f16_to_f32(hd), m=ds4q_f16_to_f32(hm);
        const uint8_t *s=x+4, *q=x+16;
        for (int g=0;g<8;g++) {
            const int sc=g<4 ? (s[g]&63) : ((s[g+4]&15)|((s[g-4]>>6)<<4));
            const int mn=g<4 ? (s[g+4]&63) : ((s[g+4]>>4)|((s[g]>>6)<<4));
            for (int j=0;j<32;j++) {
                const int v=(q[(g/2)*32+j] >> ((g%2)*4))&15;
                y[g*32+j]=(d*sc)*v-m*mn;
            }
        }
    }
}
void axk2_decode_q8(const uint8_t *x, float *y, size_t blocks) {
    for (size_t b=0;b<blocks;b++,x+=34,y+=32) {
        uint16_t hd; memcpy(&hd,x,2);
        const float d=ds4q_f16_to_f32(hd);
        const int8_t *q=(const int8_t *)(x+2);
        for (int j=0;j<32;j++) y[j]=d*(float)q[j];
    }
}
'''

# Which calibration statistic feeds which projection (docs/DESIGN.md).
IMATRIX_KEY = {
    'self_attn.q_a_proj.weight': 'attn_in',
    'self_attn.kv_a_proj_with_mqa.weight': 'attn_in',
    'self_attn.q_b_proj.weight': 'qb_in',
    'self_attn.kv_b_proj.weight': 'kvb_in',
    'self_attn.o_proj.weight': 'o_in',
    'mlp.shared_experts.gate_proj.weight': 'ffn_in',
    'mlp.shared_experts.up_proj.weight': 'ffn_in',
    'mlp.shared_experts.down_proj.weight': 'shared_a',
    'mlp.gate_proj.weight': 'ffn_in',
    'mlp.up_proj.weight': 'ffn_in',
    'mlp.down_proj.weight': 'dense_a',
}
Q4K_TAILS = {f'self_attn.{p}.weight' for p in ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj')} | \
            {f'mlp.shared_experts.{p}_proj.weight' for p in ('gate', 'up', 'down')}
DENSE_TAILS = {f'mlp.{p}_proj.weight' for p in ('gate', 'up', 'down')}
Q8_TAILS = {'self_attn.indexer.wq_b.weight', 'self_attn.indexer.wk.weight'}
PRESERVE_TAILS = {'mlp.gate.weight', 'mlp.gate.e_score_correction_bias',
                  'self_attn.indexer.weights_proj.weight', 'self_attn.indexer.k_norm.weight',
                  'self_attn.indexer.k_norm.bias', 'self_attn.q_a_layernorm.weight',
                  'self_attn.kv_a_layernorm.weight', 'post_attention_layernorm.weight'} | \
                 {f'{n}.{p}.weight' for n in ('input_layernorm', 'post_attention_layernorm')
                  for p in ('norm', 'W_down', 'W_up')}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build():
    out = TOOLS / 'native'
    out.mkdir(exist_ok=True)
    helper = out / 'axk2_quant_decode.c'
    helper.write_text(DECODE_C)
    lib = out / 'libaxk2quant.so'
    cmd = ['gcc', '-std=c11', '-O3', '-fPIC', '-shared', '-ffp-contract=off', '-I', str(DS4 / 'gguf-tools'),
           str(DS4 / 'gguf-tools/quants.c'), str(helper), '-lm', '-pthread', '-o', str(lib)]
    subprocess.run(cmd, check=True)
    receipt = {'command': cmd,
               'ds4_dfm_rs_head': subprocess.check_output(['git', '-C', str(DS4), 'rev-parse', 'HEAD'], text=True).strip(),
               'sources': {str(p): sha(p) for p in [DS4 / 'gguf-tools/quants.c', DS4 / 'gguf-tools/quants.h', helper]},
               'library_sha256': sha(lib),
               'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0]}
    (out / 'nonexpert-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return lib, receipt


class Quantizer:
    def __init__(self, lib, threads):
        self.lib = ct.CDLL(str(lib))
        self.fn = self.lib.ds4q_quantize_chunk
        self.fn.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
        self.fn.restype = ct.c_size_t
        self.lib.axk2_decode_q4.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.lib.axk2_decode_q8.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.threads = threads

    def encode(self, typ, x, imatrix=None):
        x = np.ascontiguousarray(x, dtype='<f4')
        rows, cols = x.shape
        block, width = QBYTES[typ]
        if cols % block or not np.isfinite(x).all():
            raise ValueError('unsupported/nonfinite matrix')
        qw = None
        if imatrix is not None:
            qw = np.ascontiguousarray(imatrix, dtype='<f4')
            if qw.shape != (cols,) or not np.isfinite(qw).all() or (qw < 0).any():
                raise ValueError('imatrix shape/values')
        row_bytes = cols // block * width
        out = np.zeros(rows * row_bytes, np.uint8)
        step = max(1, (rows + self.threads - 1) // self.threads)

        def run(first):
            n = min(step, rows - first)
            got = self.fn(typ, x[first:].ctypes.data, out[first * row_bytes:].ctypes.data, 0, n, cols,
                          qw.ctypes.data if qw is not None else None)
            if got != n * row_bytes:
                raise ValueError('encoder size')

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(run, range(0, rows, step)))
        scale_bytes = 4 if typ == Q4_K else 2
        scales = out.reshape(-1, width)[:, :scale_bytes].copy().view('<f2')
        if not np.isfinite(scales).all():
            raise ValueError('nonfinite encoded scales')
        return out.tobytes()

    def decode_c(self, typ, raw, shape):
        b = np.frombuffer(raw, np.uint8)
        out = np.empty(shape, dtype='<f4')
        block = QBYTES[typ][0]
        fn = self.lib.axk2_decode_q4 if typ == Q4_K else self.lib.axk2_decode_q8
        fn(b.ctypes.data, out.ctypes.data, out.size // block)
        return out


def decode_q4_np(raw, shape):
    p = np.frombuffer(raw, np.uint8).reshape(-1, 144)
    d = p[:, :2].copy().view('<f2').astype('<f4')
    dm = p[:, 2:4].copy().view('<f2').astype('<f4')
    s = p[:, 4:16].astype(np.uint16)
    sc = np.empty((len(p), 8), dtype=np.float32)
    mn = np.empty_like(sc)
    for j in range(8):
        sc[:, j] = s[:, j] & 63 if j < 4 else (s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)
        mn[:, j] = s[:, j + 4] & 63 if j < 4 else (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    q = p[:, 16:].reshape(-1, 4, 32)
    vals = np.stack([q & 15, q >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    # Same association as the C decoder: (d*sc)*v - m*mn in F32.
    out = (d * sc)[:, :, None] * vals - (dm * mn)[:, :, None]
    return out.reshape(shape)


def decode_q8_np(raw, shape):
    p = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    d = p[:, :2].copy().view('<f2').astype('<f4')
    q = p[:, 2:].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(shape)


def layer_of(name):
    m = re.fullmatch(r'model\.layers\.(\d+)\.(.+)', name)
    return (int(m[1]), m[2]) if m else (None, name)


def policy(name, cfg):
    """-> 'q4k' | 'q8_0' | 'preserve' | 'embedding' | 'expert' | 'scale'"""
    if name.endswith('_scale_inv'):
        return 'scale'
    if name == 'model.embed_tokens.weight':
        return 'embedding'
    if name == 'lm_head.weight':
        return 'q4k'
    if name == 'model.norm.weight':
        return 'preserve'
    if '.mlp.experts.' in name:
        return 'expert'
    layer, tail = layer_of(name)
    if layer is None:
        raise ValueError(f'unexpected top-level tensor {name}')
    if tail in Q4K_TAILS:
        return 'q4k'
    if tail in DENSE_TAILS:
        if layer >= cfg['first_k_dense_replace']:
            raise ValueError(f'dense projection on a MoE layer: {name}')
        return 'q4k'
    if tail in Q8_TAILS:
        return 'q8_0'
    if tail in PRESERVE_TAILS:
        return 'preserve'
    raise ValueError(f'no policy for {name}')


class Imatrix:
    """Per-column mean x^2 from stats/layer-LL.safetensors and stats/head.safetensors."""
    def __init__(self, root):
        from safetensors.numpy import load_file
        self.root, self.load, self.cache, self.files = Path(root), load_file, {}, {}

    def _file(self, stem):
        if stem not in self.cache:
            path = self.root / f'{stem}.safetensors'
            self.cache[stem] = self.load(str(path))
            self.files[path.name] = sha(path)
        return self.cache[stem]

    def vector(self, name):
        if name == 'lm_head.weight':
            stats, key = self._file('head'), 'head_in'
        else:
            layer, tail = layer_of(name)
            key = IMATRIX_KEY.get(tail)
            if key is None:
                return None
            stats = self._file(f'layer-{layer:02d}')
        tokens = float(np.asarray(stats['tokens']).reshape(-1)[0])
        if tokens <= 0:
            raise ValueError(f'no tokens in the statistics for {name}')
        h = np.asarray(stats[key], dtype=np.float64) / tokens
        h = np.maximum(h, IMATRIX_FLOOR * max(h.mean(), 1e-300))
        return h.astype(np.float32), key


def parse_layers(spec, n):
    if not spec:
        return set(range(n))
    lo, _, hi = spec.partition('-')
    return set(range(int(lo), int(hi or lo) + 1))


def load_matrix(src, name):
    """Logical weight -> float32 numpy on the CPU (FP8 dequantized exactly)."""
    stored = src.weight_names(name)
    host = src.read(stored, threads=8, pin=False)
    w = host[name]
    if name + '_scale_inv' in host:
        return dequant(w, host[name + '_scale_inv']).numpy(), 'F8_E4M3'
    return w.float().numpy(), src.info(name)[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/models/A.X-K2')
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--imatrix-dir', type=Path, help='stats/ directory (layer-LL.safetensors, head.safetensors)')
    p.add_argument('--layers', help='subset of decoder layers, e.g. 0-1 (test runs); default all')
    p.add_argument('--no-head', action='store_true', help='skip lm_head and model.norm (test runs)')
    p.add_argument('--no-embedding', action='store_true', help='skip the embedding part (test runs)')
    p.add_argument('--threads', type=int, default=96)
    p.add_argument('--torch-threads', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.torch_threads)
    recipe = json.loads(a.recipe.read_text())
    src = Source(a.source)
    cfg = src.config
    lib, build_receipt = build()
    quant = Quantizer(lib, a.threads)
    imatrix = Imatrix(a.imatrix_dir) if a.imatrix_dir else None
    layers = parse_layers(a.layers, cfg['num_hidden_layers'])
    names = sorted(src.index)
    plan = {n: policy(n, cfg) for n in names}
    full = not a.layers and not a.no_head and not a.no_embedding

    def wanted(n):
        layer, _ = layer_of(n)
        if layer is not None:
            return layer in layers
        return not a.no_head
    body = [n for n in names if plan[n] in ('q4k', 'q8_0', 'preserve') and wanted(n)]
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    entries, stats, mapping = [], {}, []
    for n in body:
        _, _, size, dtype, shape = src.info(n)
        if plan[n] in ('q4k', 'q8_0'):
            typ = Q4_K if plan[n] == 'q4k' else Q8_0
            block, width = QBYTES[typ]
            if len(shape) != 2 or shape[1] % block:
                raise ValueError(f'{plan[n]} shape {n}')
            entries.append({'name': n, 'shape': list(shape), 'type_id': typ,
                            'payload_bytes': shape[0] * shape[1] // block * width})
        else:
            if dtype not in TYPES:
                raise ValueError(f'preserved tensor {n} has stored dtype {dtype}')
            entries.append({'name': n, 'shape': list(shape), 'type_id': TYPES[dtype], 'payload_bytes': size})
    meta = metadata(part_role='trunk', extra={
        'axk2_vq.source_revision': SOURCE_REV, 'axk2_vq.recipe_id': recipe['recipe_id'],
        'axk2_vq.native_policy': recipe['nonexpert']['policy_id'],
        'axk2_vq.nonexpert_imatrix': 'calibration mean x^2 per input column' if imatrix else 'none'})
    body_path = a.output / 'body-nonexpert.gguf'
    partial = body_path.with_suffix('.gguf.partial')
    if partial.exists():
        partial.rename(partial.with_suffix(f'.failed-{time.time_ns()}'))
    with PartWriter(partial, meta, entries) as writer:
        for i, n in enumerate(body):
            t1 = time.monotonic()
            if plan[n] in ('q4k', 'q8_0'):
                typ = Q4_K if plan[n] == 'q4k' else Q8_0
                x, src_dtype = load_matrix(src, n)
                h = key = None
                if imatrix is not None and typ == Q4_K:
                    got = imatrix.vector(n)
                    if got is None:
                        raise ValueError(f'no imatrix key for {n}')
                    h, key = got
                    if h.shape != (x.shape[1],):
                        raise ValueError(f'imatrix length {h.shape} for {n} {x.shape}')
                raw = quant.encode(typ, x, h)
                y = (decode_q4_np if typ == Q4_K else decode_q8_np)(raw, x.shape)
                if not np.array_equal(y, quant.decode_c(typ, raw, x.shape)):
                    raise AssertionError(f'independent decode mismatch {n}')
                err = (y.astype(np.float64) - x) ** 2
                den = (x.astype(np.float64) ** 2)
                row = np.sqrt(err.sum(1) / np.maximum(den.sum(1), 1e-300))
                rec = {'type': 'Q4_K' if typ == Q4_K else 'Q8_0', 'source_dtype': src_dtype, 'shape': list(x.shape),
                       'rel_mse': float(err.sum() / max(den.sum(), 1e-300)),
                       'row_rel_rms_p99': float(np.quantile(row, .99)), 'row_rel_rms_max': float(row.max()),
                       'imatrix': key, 'payload_sha256': hashlib.sha256(raw).hexdigest()}
                if imatrix is not None and typ == Q4_K:
                    rec['imatrix_weighted_rel_err'] = float((err.sum(0) @ h) / max(den.sum(0) @ h, 1e-300))
                elif imatrix is not None:
                    got = imatrix.vector(n)
                    if got is not None:
                        rec['imatrix_weighted_rel_err'] = float((err.sum(0) @ got[0]) / max(den.sum(0) @ got[0], 1e-300))
                stats[n] = rec
                mapping.append({'source_name': n, 'target_name': n,
                                'disposition': 'consumed_q4k' if typ == Q4_K else 'consumed_q8_0'})
                if n + '_scale_inv' in src.index:
                    mapping.append({'source_name': n + '_scale_inv', 'target_name': n, 'disposition': 'consumed_scale'})
                del x, y, err, den
            else:
                t = src.read([n], threads=4, pin=False)[n]
                raw = t.view(torch.uint8).numpy().tobytes()
                stats[n] = {'type': src.info(n)[3], 'shape': list(src.info(n)[4]), 'preserved': True,
                            'payload_sha256': hashlib.sha256(raw).hexdigest()}
                mapping.append({'source_name': n, 'target_name': n, 'disposition': 'preserved'})
            writer.write(n, raw)
            stats[n]['part'] = body_path.name
            stats[n]['seconds'] = time.monotonic() - t1
            if plan[n] != 'preserve':
                print(json.dumps({'tensor': n, 'i': i + 1, 'of': len(body), **{k: v for k, v in stats[n].items()
                                  if k in ('type', 'rel_mse', 'imatrix_weighted_rel_err', 'seconds')}}), flush=True)
    partial.rename(body_path)
    with PartReader(body_path):
        pass                                   # strict re-read of the written part

    parts = [body_path]
    if not a.no_embedding:
        n = 'model.embed_tokens.weight'
        _, _, size, dtype, shape = src.info(n)
        if dtype != 'BF16':
            raise ValueError('embedding must be BF16')
        emb_path = a.output / 'embed-tokens-bf16.gguf'
        partial = emb_path.with_suffix('.gguf.partial')
        if partial.exists():
            partial.rename(partial.with_suffix(f'.failed-{time.time_ns()}'))
        meta = metadata(part_role='embedding', extra={'axk2_vq.source_revision': SOURCE_REV,
                                                      'axk2_vq.recipe_id': recipe['recipe_id'],
                                                      'axk2_vq.residency': 'ssd-rows'})
        with PartWriter(partial, meta, [{'name': n, 'shape': list(shape), 'type_id': BF16, 'payload_bytes': size}]) as writer:
            raw = src.read([n], threads=16, pin=False)[n].view(torch.uint8).numpy().tobytes()
            writer.write(n, raw)
            stats[n] = {'type': 'BF16', 'shape': list(shape), 'preserved': True, 'part': emb_path.name,
                        'payload_sha256': hashlib.sha256(raw).hexdigest()}
            mapping.append({'source_name': n, 'target_name': n, 'disposition': 'preserved'})
        partial.rename(emb_path)
        with PartReader(emb_path):
            pass
        parts.append(emb_path)

    quantized = [s for s in stats.values() if s['type'] in ('Q4_K', 'Q8_0')]
    q4 = [s for s in quantized if s['type'] == 'Q4_K']
    q8 = [s for s in quantized if s['type'] == 'Q8_0']
    receipt = {'source_revision': SOURCE_REV, 'recipe_id': recipe['recipe_id'],
               'policy_id': recipe['nonexpert']['policy_id'], 'complete': full, 'layers': sorted(layers),
               'build': build_receipt,
               'imatrix': {'dir': str(a.imatrix_dir), 'files_sha256': imatrix.files, 'floor': IMATRIX_FLOOR}
                          if imatrix else None,
               'executed_source_sha256': {f.name: sha(f) for f in [Path(__file__), TOOLS / 'axk2_vq_format.py',
                                                                    TOOLS / 'axk2_source.py']},
               'parts': {pth.name: {'bytes': pth.stat().st_size, 'sha256': sha(pth)} for pth in parts},
               'q4k_tensors': len(q4), 'q8_0_tensors': len(q8), 'preserved_tensors': len(stats) - len(quantized),
               'q4k_rel_mse_mean': float(np.mean([s['rel_mse'] for s in q4])) if q4 else None,
               'q4k_rel_mse_max': float(np.max([s['rel_mse'] for s in q4])) if q4 else None,
               'q8_0_rel_mse_max': float(np.max([s['rel_mse'] for s in q8])) if q8 else None,
               'seconds': time.monotonic() - t0, 'source_mapping': mapping, 'tensors': stats}
    (a.output / 'nonexpert-receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('tensors', 'build', 'source_mapping')}, indent=1))


if __name__ == '__main__':
    main()
