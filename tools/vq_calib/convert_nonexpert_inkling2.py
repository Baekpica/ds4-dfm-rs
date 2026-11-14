#!/usr/bin/env python3
"""Inkling-2-Small hVQ release: every non-routed-expert part (CPU only).

Ported from the MiniMax-M3.1-Flash VQ release's mm31_nx_convert.py (Baekpica/MiniMax-M3.1-Flash-VQ-GGUF
tools/); container and metadata from the family module inkling2_vq_format.py (namespace inkling2_vq).

    embed   -> parts/embed-tokens-bf16.gguf   (part_role embedding, residency ssd-rows)
               model.llm.embed.weight [201024, 4608], BF16 bytes as stored.
    vision  -> parts/vision-bf16.gguf         (part_role vision, residency mmap-on-demand)
               every model.visual.* tensor (23-layer ViT, embeddings, proj_in / proj_out), BF16 as stored.
    mtp     -> parts/mtp-q8_0.gguf            (part_role drafter)
               mtp.safetensors (BF16) under the prefix `mtp.`: Q8_0 for the 5 layers' self_attn
               {q,k,v,o}_proj, k_proj_ctx, v_proj_ctx and mlp {gate,up,down}_proj; every norm (incl.
               dflare_norm, k_norm_ctx), markov_head.*, fuse_weight, mask_embed and norm.weight as stored;
               the source mtp_config (config.json) and the SHA-256 of mtp.safetensors in the metadata.
    body    -> parts/body-nonexpert.gguf      (part_role trunk)
               Q4_K (GGML quantize_row_q4_K_impl with per-input-column importance):
               attn.{wq_du,wk_dv,wv_dv,wr_du,gate_attn,wo_ud}.weight (48 layers), model.llm.unembed.weight;
               as stored: attn.attn_sink_proj, the short convolutions (attn.{k,v}_sconv.weight, attn_sconv,
               mlp_sconv: source [C, 1, 4], kept 3-D; attn.{k,v}_sconv.{static_w,weight_proj}), clip scales,
               rel_logits_proj, norms, router mlp.gate.weight (BF16) + mlp.gate.bias (F32), model.llm.norm,
               model.llm.embed_norm and the F32 block attention-residual queries model.llm.attn_res.*.
               Importance = calibration mean x^2 per input column (stats/layer-LL-nonexpert q_in -> wq_du,
               wr_du, gate_attn; k_in -> wk_dv; v_in -> wv_dv; o_in -> wo_ud; stats/head head_in -> unembed),
               or --unit-importance (test builds, never the release body).
    merge   -> parts/receipts/nonexpert-receipt.json from the per-part receipts, with the source mapping
               checked against every tensor of the main checkpoint and of mtp.safetensors, exactly once
               (the routed experts, converted by the hVQ expert tool, are accounted as consumed_vq /
               consumed_scale).
    verify  -> re-run the strict re-read and decode check of written parts against their receipts.

The release body (parts/body-nonexpert.gguf) is written only from complete calibration statistics (the
statistics run writes no DONE file): all 48 layer-LL-nonexpert.safetensors and head.safetensors present and
readable, one source revision, one calib_sha256 equal to the SHA-256 of --calib, and one token count equal
to the --calib text tokens plus the --images image-sequence tokens; it then writes the merged receipt.
Test builds (--unit-importance, --layers, --no-head) go elsewhere (--output), never into parts/.

Encoders: this checkout's gguf-tools/quants.c (ds4q_quantize_chunk), built here with
gcc -O3 -ffp-contract=off. Every quantized payload is decoded by numpy and by an independent C decoder;
both must agree bit for bit before the relative MSE is recorded. Every written part is re-read with the
strict inkling2_vq_format.PartReader: layout, metadata round trip, payload SHA-256 (preserved tensors
against their source bytes) and a second numpy + C decode of every quantized payload from the file,
which must reproduce the decode measured at encode time (SHA-256 of the decoded F32 stream).

    python3 convert_nonexpert_inkling2.py --source $SRC --work $W embed      # parts under $W/parts
    python3 convert_nonexpert_inkling2.py --source $SRC --work $W vision
    python3 convert_nonexpert_inkling2.py --source $SRC --work $W mtp
    python3 convert_nonexpert_inkling2.py --source $SRC --work $W body --stats $W/stats   # release body, then merge
    python3 convert_nonexpert_inkling2.py --source $SRC --work $W body --unit-importance --output $W/test/parts-unit/body-nonexpert.gguf
    python3 convert_nonexpert_inkling2.py --source $SRC --work $W merge
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import inkling2_vq_format as fmt                    # noqa: E402

F32, F16, Q8_0, Q4_K, BF16 = fmt.F32, fmt.F16, fmt.Q8_0, fmt.Q4_K, fmt.BF16
TYPE_NAMES = {F32: 'F32', F16: 'F16', Q8_0: 'Q8_0', Q4_K: 'Q4_K', BF16: 'BF16'}
NS = fmt.NS

SOURCE = None                             # --source: the source checkpoint directory
SOURCE_REPO = 'thinkingmachines/Inkling-2-Small-MXFP8'
DS4 = TOOLS.parents[1]                    # this checkout (gguf-tools/quants.c)
WORK = PARTS = CALIB = IMAGES = BUILD = None   # --work and the paths under it (configure)
P = 'model.llm.'
EMBED = P + 'embed.weight'
UNEMBED = P + 'unembed.weight'
FINAL_NORM = P + 'norm.weight'
EMBED_NORM = P + 'embed_norm.weight'
ATTN_RES = P + 'attn_res.'
VISUAL = 'model.visual.'
MTP_FILE = 'mtp.safetensors'
MTP_PREFIX = 'mtp.'
N_LAYERS, HIDDEN, VOCAB, MTP_LAYERS = 48, 4608, 201024, 5
QBYTES = {Q4_K: (256, 144), Q8_0: (32, 34)}
ELEMENT_BYTES = {F32: 4, F16: 2, BF16: 2}
IMATRIX_FLOOR = 1e-8          # relative to the vector mean (A.X-K2 / Beam / MiniMax); keeps all-zero columns defined
PART_FILES = {'embed': 'embed-tokens-bf16.gguf', 'vision': 'vision-bf16.gguf',
              'mtp': 'mtp-q8_0.gguf', 'body': 'body-nonexpert.gguf'}
ENCODER = ('ds4-dfm-rs gguf-tools/quants.c ds4q_quantize_chunk, gcc -O3 -ffp-contract=off; '
           'Q4_K: GGML quantize_row_q4_K_impl path (per-column importance); Q8_0: GGML quantize_row_q8_0_ref')

DECODE_C = r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "quants.h"
void ink2_nx_decode_q4(const uint8_t *x, float *y, size_t blocks) {
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
void ink2_nx_decode_q8(const uint8_t *x, float *y, size_t blocks) {
    for (size_t b=0;b<blocks;b++,x+=34,y+=32) {
        uint16_t hd; memcpy(&hd,x,2);
        const float d=ds4q_f16_to_f32(hd);
        const int8_t *q=(const int8_t *)(x+2);
        for (int j=0;j<32;j++) y[j]=d*(float)q[j];
    }
}
'''


def sha_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sha_bytes(buf):
    return hashlib.sha256(buf).hexdigest()


# hashed at import: the receipts name the code that ran, even if the files are edited during a run
EXECUTED_SOURCES = {p.name: sha_file(p) for p in (Path(__file__).resolve(), TOOLS / 'inkling2_vq_format.py')}


def git_blob_sha1(path):
    raw = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def payload_bytes(type_id, shape):
    n = math.prod(shape)
    if type_id in QBYTES:
        block, width = QBYTES[type_id]
        if len(shape) != 2 or shape[1] % block:
            raise ValueError(f'{TYPE_NAMES[type_id]} needs a 2-D shape with columns % {block} == 0: {shape}')
        return n // block * width
    if type_id in ELEMENT_BYTES:
        return n * ELEMENT_BYTES[type_id]
    raise ValueError(f'unsupported tensor type {type_id}')


_CONFIG = {}


def config():
    if not _CONFIG:
        path = SOURCE / 'config.json'
        text = path.read_text()
        cfg = json.loads(text)
        tc = cfg['text_config']
        if (tc['num_hidden_layers'], tc['hidden_size'], tc['vocab_size']) != (N_LAYERS, HIDDEN, VOCAB):
            raise ValueError('config.json does not describe Inkling-2-Small')
        _CONFIG.update(path=path, text=text, cfg=cfg, local=frozenset(tc['local_layer_ids']))
    return _CONFIG


def global_layers():
    return sorted(set(range(N_LAYERS)) - config()['local'])


def revision_tree():
    tree = SOURCE / '.cache/huggingface/trees' / f'{fmt.SOURCE_REVISION}.json'
    return json.loads(tree.read_text())['files'] if tree.exists() else {}


# ---------------------------------------------------------------- source (numpy, pread)

DTYPES = {'F8_E4M3': ('u1', 1), 'U8': ('u1', 1), 'BF16': ('<u2', 2), 'F32': ('<f4', 4), 'F16': ('<f2', 2)}
NATIVE = {'BF16': BF16, 'F32': F32, 'F16': F16}


class NpSource:
    """A safetensors checkpoint: sharded (model.safetensors.index.json) or one file (`single`)."""
    def __init__(self, root, label, single=None):
        self.root, self.label, self._headers = Path(root), label, {}
        if single is None:
            self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        else:
            self.index = {n: single for n in self.header(single)[1]}

    def header(self, shard):
        if shard not in self._headers:
            with (self.root / shard).open('rb') as f:
                n = int.from_bytes(f.read(8), 'little')
                meta = json.loads(f.read(n))
            meta.pop('__metadata__', None)
            self._headers[shard] = (8 + n, meta)
        return self._headers[shard]

    def info(self, name):
        shard = self.index[name]
        base, meta = self.header(shard)
        t = meta[name]
        start, end = t['data_offsets']
        return shard, base + start, end - start, t['dtype'], tuple(t['shape'])

    def all_infos(self):
        return {n: self.info(n) for n in self.index}

    def _pread(self, shard, off, view):
        fd = os.open(self.root / shard, os.O_RDONLY)
        try:
            got = 0
            while got < len(view):
                n = os.preadv(fd, [view[got:]], off + got)
                if n <= 0:
                    raise IOError(f'short read {shard}@{off + got}')
                got += n
        finally:
            os.close(fd)

    def read_bytes(self, name):
        shard, off, size, dtype, shape = self.info(name)
        if size != DTYPES[dtype][1] * math.prod(shape):
            raise ValueError(f'size mismatch {name}')
        buf = np.empty(size, np.uint8)
        self._pread(shard, off, memoryview(buf))
        return buf

    def stored(self, name):
        _, _, _, dtype, shape = self.info(name)
        return self.read_bytes(name).view(DTYPES[dtype][0]).reshape(shape)


def load_f32(src, name):
    """Logical weight -> (float32 [N, K], source dtype). BF16 widened exactly."""
    dtype = src.info(name)[3]
    if dtype == 'BF16':
        x = (src.stored(name).astype('<u4') << 16).view('<f4')
    elif dtype == 'F32':
        x = src.stored(name).astype('<f4')
    else:
        raise ValueError(f'cannot quantize {name} from {dtype}')
    if not np.isfinite(x).all():
        raise ValueError(f'nonfinite source {name}')
    return x, dtype


# ---------------------------------------------------------------- encoder / decoders

def build_native(out=None):
    out = out or BUILD
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    helper = out / 'inkling2_nx_decode.c'
    if not helper.exists() or helper.read_text() != DECODE_C:
        tmp = helper.with_name(helper.name + f'.tmp{os.getpid()}')
        tmp.write_text(DECODE_C)
        os.replace(tmp, helper)
    lib = out / 'libinkling2nxquant.so'
    tmp = out / f'libinkling2nxquant.so.tmp{os.getpid()}'
    cmd = ['gcc', '-std=c11', '-O3', '-fPIC', '-shared', '-ffp-contract=off', '-I', str(DS4 / 'gguf-tools'),
           str(DS4 / 'gguf-tools/quants.c'), str(helper), '-lm', '-pthread', '-o', str(tmp)]
    subprocess.run(cmd, check=True)
    os.replace(tmp, lib)

    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(DS4), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    dirty = git('status', '--porcelain', '--', 'gguf-tools/quants.c', 'gguf-tools/quants.h')
    receipt = {'command': cmd[:-1] + [str(lib)], 'ds4_dfm_rs_head': git('rev-parse', 'HEAD'),
               'quants_clean_at_head': None if dirty is None else dirty == '',
               'sources': {str(p): sha_file(p) for p in [DS4 / 'gguf-tools/quants.c', DS4 / 'gguf-tools/quants.h', helper]},
               'library_sha256': sha_file(lib),
               'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0]}
    (out / 'nonexpert-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return lib, receipt


class Quantizer:
    def __init__(self, lib, threads):
        self.lib = ct.CDLL(str(lib))
        self.fn = self.lib.ds4q_quantize_chunk
        self.fn.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
        self.fn.restype = ct.c_size_t
        self.lib.ink2_nx_decode_q4.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.lib.ink2_nx_decode_q8.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.threads = threads

    def encode(self, typ, x, imatrix=None):
        x = np.ascontiguousarray(x, dtype='<f4')
        rows, cols = x.shape
        block, width = QBYTES[typ]
        if cols % block or not np.isfinite(x).all():
            raise ValueError('unsupported/nonfinite matrix')
        qw = None
        if imatrix is not None:
            if typ != Q4_K:
                raise ValueError('importance is only used by the Q4_K encoder')
            qw = np.ascontiguousarray(imatrix, dtype='<f4')
            if qw.shape != (cols,) or not np.isfinite(qw).all() or (qw <= 0).any():
                raise ValueError('importance shape/values')
        row_bytes = cols // block * width
        out = np.zeros(rows * row_bytes, np.uint8)
        step = max(1, (rows + self.threads - 1) // self.threads)

        def run(first):
            n = min(step, rows - first)
            got = self.fn(typ, x[first:].ctypes.data, out[first * row_bytes:].ctypes.data, 0, n, cols,
                          qw.ctypes.data if qw is not None else None)
            if got != n * row_bytes:
                raise ValueError('encoder size')

        if self.threads > 1 and rows > step:
            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(run, range(0, rows, step)))
        else:
            for first in range(0, rows, step):
                run(first)
        scale_bytes = 4 if typ == Q4_K else 2
        if not np.isfinite(out.reshape(-1, width)[:, :scale_bytes].copy().view('<f2')).all():
            raise ValueError('nonfinite encoded scales')
        return out.tobytes()

    def decode_c(self, typ, raw, shape):
        b = np.frombuffer(raw, np.uint8)
        out = np.empty(shape, dtype='<f4')
        if b.size != out.size // QBYTES[typ][0] * QBYTES[typ][1]:
            raise ValueError('decode size')
        fn = self.lib.ink2_nx_decode_q4 if typ == Q4_K else self.lib.ink2_nx_decode_q8
        fn(b.ctypes.data, out.ctypes.data, out.size // QBYTES[typ][0])
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
    out = (d * sc)[:, :, None] * vals - (dm * mn)[:, :, None]      # (d*sc)*v - m*mn in F32, as the C decoder
    return out.reshape(shape)


def decode_q8_np(raw, shape):
    p = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    d = p[:, :2].copy().view('<f2').astype('<f4')
    q = p[:, 2:].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(shape)


def decode_both(q, typ, piece, shape):
    """numpy and C decodes of one row block; must agree bit for bit and be finite."""
    y = (decode_q4_np if typ == Q4_K else decode_q8_np)(piece, shape)
    yc = q.decode_c(typ, piece, shape)
    if not np.array_equal(y.view(np.uint32), yc.view(np.uint32)):
        raise AssertionError('numpy and C decoders disagree')
    if not np.isfinite(y).all():
        raise ValueError('nonfinite decode')
    return y


def row_chunks(rows, cols):
    step = max(1, (1 << 22) // cols)
    return step, list(range(0, rows, step))


def measure(q, typ, x, raw, weight=None):
    """numpy and C decodes must agree bit for bit; error statistics in FP64 (row chunks, q.threads threads;
    the per-chunk partial sums are combined in chunk order, so the result does not depend on scheduling).
    decoded_sha256: SHA-256 of the decoded F32 tensor (row-major, little endian)."""
    rows, cols = x.shape
    block, width = QBYTES[typ]
    row_bytes = cols // block * width
    rb = np.frombuffer(raw, np.uint8)
    step, starts = row_chunks(rows, cols)
    row_rel = np.empty(rows)

    def chunk(r0):
        r1 = min(rows, r0 + step)
        y = decode_both(q, typ, rb[r0 * row_bytes:r1 * row_bytes], (r1 - r0, cols))
        xs = x[r0:r1].astype(np.float64)
        d = y.astype(np.float64) - xs
        e = d * d
        xx = xs * xs
        er, dr = e.sum(1), xx.sum(1)
        row_rel[r0:r1] = np.sqrt(er / np.maximum(dr, 1e-300))
        cols_e = (e.sum(0), xx.sum(0)) if weight is not None else None
        return float(er.sum()), float(dr.sum()), float(np.abs(d).max()), cols_e, y

    h, parts = hashlib.sha256(), []
    if q.threads > 1 and len(starts) > 1:
        with ThreadPoolExecutor(q.threads) as pool:
            for res in pool.map(chunk, starts):
                h.update(res[4].data)
                parts.append(res[:4])
    else:
        for r0 in starts:
            res = chunk(r0)
            h.update(res[4].data)
            parts.append(res[:4])
    err_t = sum(p[0] for p in parts)
    den_t = sum(p[1] for p in parts)
    max_abs = max(p[2] for p in parts)
    rec = {'rel_mse': err_t / max(den_t, 1e-300), 'rmse': (err_t / x.size) ** 0.5,
           'weight_rms': (den_t / x.size) ** 0.5, 'max_abs_err': max_abs,
           'row_rel_rms_p50': float(np.quantile(row_rel, .5)), 'row_rel_rms_p99': float(np.quantile(row_rel, .99)),
           'row_rel_rms_max': float(row_rel.max()), 'decoders_agree': True, 'decoded_sha256': h.hexdigest()}
    if weight is not None:
        col_err = np.zeros(cols)
        col_den = np.zeros(cols)
        for p in parts:
            col_err += p[3][0]
            col_den += p[3][1]
        w = np.asarray(weight, np.float64)
        rec['importance_weighted_rel_mse'] = float((col_err @ w) / max(col_den @ w, 1e-300))
    return rec


def decode_digest(q, typ, raw, shape, threads):
    """Decode check of a payload read back from a part: numpy == C bit for bit, finite; -> decoded SHA-256."""
    rows, cols = shape
    block, width = QBYTES[typ]
    row_bytes = cols // block * width
    rb = np.frombuffer(raw, np.uint8)
    if rb.size != rows * row_bytes:
        raise ValueError('payload size')
    step, starts = row_chunks(rows, cols)

    def chunk(r0):
        r1 = min(rows, r0 + step)
        return decode_both(q, typ, rb[r0 * row_bytes:r1 * row_bytes], (r1 - r0, cols))

    h = hashlib.sha256()
    if threads > 1 and len(starts) > 1:
        with ThreadPoolExecutor(threads) as pool:
            for y in pool.map(chunk, starts):
                h.update(y.data)
    else:
        for r0 in starts:
            h.update(chunk(r0).data)
    return h.hexdigest()


# ---------------------------------------------------------------- worker side

_W = {}


def _worker_init(lib, threads, sources):
    _W['q'] = Quantizer(lib, threads)
    _W['src'] = sources


def quant_job(job):
    t0 = time.monotonic()
    src = _W['src'][job['src']]
    x, src_dtype = load_f32(src, job['source_name'])
    if list(x.shape) != job['shape']:
        raise ValueError(f'shape {x.shape} != {job["shape"]} for {job["source_name"]}')
    typ = job['type_id']
    q = _W['q']
    raw = q.encode(typ, x, job.get('qw'))
    rec = {'type': TYPE_NAMES[typ], 'source_dtype': src_dtype}
    rec.update(measure(q, typ, x, raw, job.get('measure_w')))
    rec['payload_sha256'] = sha_bytes(raw)
    rec['seconds'] = round(time.monotonic() - t0, 3)
    return job['name'], raw, rec


# ---------------------------------------------------------------- plans (policy)

EXPERT_RE = re.compile(re.escape(P) + r'layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)(_scale)?')
LAYER_RE = re.compile(re.escape(P) + r'layers\.(\d+)\.(.+)')
Q4K_IMPORTANCE = {'attn.wq_du.weight': 'q_in', 'attn.wr_du.weight': 'q_in', 'attn.gate_attn.weight': 'q_in',
                  'attn.wk_dv.weight': 'k_in', 'attn.wv_dv.weight': 'v_in', 'attn.wo_ud.weight': 'o_in'}
PRESERVE_ALL = {'attn.attn_sink_proj.weight', 'attn.q_l2_clip.scale', 'attn.rel_l2_clip.scale',
                'attn.rel_logits_proj.proj', 'attn_sconv.weight', 'mlp_sconv.weight', 'mlp_norm.weight',
                'mlp.gate.weight', 'mlp.gate.bias'}
PRESERVE_LOCAL = {'attn_norm.weight', 'attn.k_sconv.weight', 'attn.v_sconv.weight'}
PRESERVE_GLOBAL = {'q_input_norm.weight', 'k_input_norm.weight', 'v_input_norm.weight', 'attn.k_sconv.static_w',
                   'attn.k_sconv.weight_proj', 'attn.v_sconv.static_w', 'attn.v_sconv.weight_proj'}
F32_TAILS = {'mlp.gate.bias'}
STATS_KEYS = {'q_in': HIDDEN, 'k_in': HIDDEN, 'v_in': HIDDEN, 'o_in': 6144, 'mlp_in': HIDDEN}
EXPERT_SHAPES = {('w13_weight', ''): ('F8_E4M3', (fmt.N_EXPERTS, 2 * fmt.FFN, HIDDEN)),
                 ('w13_weight', '_scale'): ('U8', (fmt.N_EXPERTS, 2 * fmt.FFN, HIDDEN // 32)),
                 ('w2_weight', ''): ('F8_E4M3', (fmt.N_EXPERTS, HIDDEN, fmt.FFN)),
                 ('w2_weight', '_scale'): ('U8', (fmt.N_EXPERTS, HIDDEN, fmt.FFN // 32))}


def want_dtype(name, dtype, want):
    if dtype != want:
        raise ValueError(f'{name} is {dtype}, expected {want}')


def main_policy(name, shape, dtype):
    """Main-checkpoint tensor -> (part, kind, importance key); kind in q4k | preserve | vq | scale."""
    m = EXPERT_RE.fullmatch(name)
    if m:
        want, want_shape = EXPERT_SHAPES[(m[2], m[3] or '')]
        want_dtype(name, dtype, want)
        if tuple(shape) != want_shape or not 0 <= int(m[1]) < N_LAYERS:
            raise ValueError(f'expert tensor {name} {shape}')
        return 'experts', 'scale' if m[3] else 'vq', None
    if name == EMBED:
        want_dtype(name, dtype, 'BF16')
        return 'embed', 'preserve', None
    if name.startswith(VISUAL):
        return 'vision', 'preserve', None
    if name == UNEMBED:
        want_dtype(name, dtype, 'BF16')
        return 'body', 'q4k', 'head_in'
    if name in (FINAL_NORM, EMBED_NORM):
        want_dtype(name, dtype, 'BF16')
        return 'body', 'preserve', None
    if name.startswith(ATTN_RES):
        want_dtype(name, dtype, 'F32')
        return 'body', 'preserve', None
    m = LAYER_RE.fullmatch(name)
    if not m:
        raise ValueError(f'no policy for {name}')
    L, tail = int(m[1]), m[2]
    if not 0 <= L < N_LAYERS:
        raise ValueError(f'layer {name}')
    glob = L not in config()['local']
    if tail in Q4K_IMPORTANCE:
        want_dtype(name, dtype, 'BF16')
        if len(shape) != 2:
            raise ValueError(f'{name}: not a matrix')
        return 'body', 'q4k', Q4K_IMPORTANCE[tail]
    if tail in PRESERVE_ALL or tail in (PRESERVE_GLOBAL if glob else PRESERVE_LOCAL):
        want_dtype(name, dtype, 'F32' if tail in F32_TAILS else 'BF16')
        return 'body', 'preserve', None
    if tail in PRESERVE_LOCAL | PRESERVE_GLOBAL:
        raise ValueError(f'{name}: a {"local" if glob else "global"}-layer tensor in a '
                         f'{"global" if glob else "local"} (sliding-window) layer')
    raise ValueError(f'no policy for {name}')


MTP_Q8 = {'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
          'self_attn.o_proj.weight', 'self_attn.k_proj_ctx.weight', 'self_attn.v_proj_ctx.weight',
          'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'}
MTP_LAYER_PRESERVE = {'input_layernorm.weight', 'post_attention_layernorm.weight', 'dflare_norm.weight',
                      'self_attn.q_norm.weight', 'self_attn.k_norm.weight', 'self_attn.k_norm_ctx.weight'}
MTP_TOP_PRESERVE = {'fuse_weight', 'mask_embed', 'norm.weight'}


def mtp_policy(name, shape, dtype):
    """mtp.safetensors tensor -> q8_0 | preserve."""
    want_dtype(name, dtype, 'BF16')
    m = re.fullmatch(r'layers\.(\d+)\.(.+)', name)
    if m:
        if not 0 <= int(m[1]) < MTP_LAYERS:
            raise ValueError(f'mtp layer {name}')
        if m[2] in MTP_Q8:
            if len(shape) != 2:
                raise ValueError(f'{name}: not a matrix')
            return 'q8_0'
        if m[2] in MTP_LAYER_PRESERVE:
            return 'preserve'
        raise ValueError(f'no policy for mtp tensor {name}')
    if name.startswith('markov_head.') or name in MTP_TOP_PRESERVE:
        return 'preserve'
    raise ValueError(f'no policy for mtp tensor {name}')


def target_name(checkpoint, source_name):
    return MTP_PREFIX + source_name if checkpoint == 'mtp' else source_name


def layer_of(name):
    m = re.match(re.escape(P) + r'layers\.(\d+)\.', name)
    return int(m[1]) if m else None


def body_order(name):
    L = layer_of(name)
    if L is not None:
        return (0, L, natural_key(name))
    rank = 0 if name.startswith(ATTN_RES) else {EMBED_NORM: 1, FINAL_NORM: 2, UNEMBED: 3}[name]
    return (1, rank, natural_key(name))


def check_body_inventory(names):
    """Every layer has exactly its local / global non-expert set; the globals are all there."""
    have = {}
    for n in names:
        L = layer_of(n)
        if L is not None:
            have.setdefault(L, set()).add(LAYER_RE.fullmatch(n)[2])
    for L in range(N_LAYERS):
        want = set(Q4K_IMPORTANCE) | PRESERVE_ALL | (PRESERVE_LOCAL if L in config()['local'] else PRESERVE_GLOBAL)
        if have.get(L, set()) != want:
            raise ValueError(f'layer {L} non-expert inventory: missing {sorted(want - have.get(L, set()))}, '
                             f'unexpected {sorted(have.get(L, set()) - want)}')
    for n in (EMBED_NORM, FINAL_NORM, UNEMBED, ATTN_RES + 'output_query'):
        if n not in names:
            raise ValueError(f'missing {n}')


def shape_patterns(names):
    return sorted({re.sub(r'\.\d+(?=\.|$)', '.N', n) for n in names}, key=natural_key)


# ---------------------------------------------------------------- statistics

def stats_stem(name):
    return 'head' if name == UNEMBED else f'layer-{layer_of(name):02d}-nonexpert'


def sequence_tokens(path):
    from safetensors import safe_open
    with safe_open(str(path), framework='numpy') as f:
        off = f.get_tensor('offsets')
    return int(off[-1] - off[0])


def preflight_stats(sdir, stems, release, calib, images=None):
    """Every needed statistics file exists, parses and comes from one calibration (source revision, calib
    SHA-256, token count); the release body also needs the full release calibration set: the statistics'
    calib_sha256 is that of `calib` (the text set) and their token count that of `calib` plus the image
    calibration sequences `images` (run_hvq_inkling2.py --images; None / '' for text-only statistics)."""
    from safetensors import safe_open
    sdir = Path(sdir)
    what = 'the release body' if release else 'this build'
    if not sdir.is_dir():
        raise SystemExit(f'--stats {sdir} is not a directory: no calibration statistics; refusing to write {what}')
    missing = [f'{s}.safetensors' for s in stems if not (sdir / f'{s}.safetensors').is_file()]
    if missing:
        raise SystemExit(f'{sdir}: calibration statistics incomplete, {len(missing)} of {len(stems)} files missing '
                         f'({", ".join(missing[:5])}{", ..." if len(missing) > 5 else ""}); refusing to write {what}')
    files = {}
    for s in stems:
        path = sdir / f'{s}.safetensors'
        try:
            with safe_open(str(path), framework='numpy') as f:
                keys, meta = set(f.keys()), f.metadata() or {}
                tokens = float(np.asarray(f.get_tensor('tokens'), np.float64).reshape(-1)[0]) if 'tokens' in keys else 0.0
        except Exception as e:      # truncated / being written
            raise SystemExit(f'{path}: unreadable statistics ({e}); refusing to write {what}')
        want = {'head_in', 'tokens'} if s == 'head' else set(STATS_KEYS) | {'tokens'}
        if not want <= keys:
            raise SystemExit(f'{path}: missing keys {sorted(want - keys)}')
        if meta.get('source_revision') != fmt.SOURCE_REVISION:
            raise SystemExit(f'{path}: statistics from revision {meta.get("source_revision")}, not {fmt.SOURCE_REVISION}')
        if not tokens > 0:
            raise SystemExit(f'{path}: no tokens')
        files[s] = {'tokens': tokens, 'calib_sha256': meta.get('calib_sha256'), 'run_id': meta.get('run_id')}
    calibs = {v['calib_sha256'] for v in files.values()}
    toks = {v['tokens'] for v in files.values()}
    if len(calibs) != 1 or None in calibs:
        raise SystemExit(f'{sdir}: statistics from different calibration sets {sorted(map(str, calibs))}')
    if len(toks) != 1:
        raise SystemExit(f'{sdir}: statistics over different token counts {sorted(toks)}')
    out = {'calib_sha256': calibs.pop(), 'tokens': toks.pop(),
           'run_ids': sorted({str(v['run_id']) for v in files.values()})}
    if release:
        calib = Path(calib)
        if not calib.is_file():
            raise SystemExit(f'--calib {calib} does not exist')
        csha = sha_file(calib)
        if csha != out['calib_sha256']:
            raise SystemExit(f'statistics come from calibration set {out["calib_sha256"]}, not {calib} ({csha}); '
                             'refusing to write the release body')
        ctok, itok, isha = sequence_tokens(calib), 0, None
        if images:
            images = Path(images)
            if not images.is_file():
                raise SystemExit(f'--images {images} does not exist (pass --images "" for text-only statistics)')
            itok, isha = sequence_tokens(images), sha_file(images)
        if out['tokens'] != ctok + itok:
            raise SystemExit(f'statistics cover {out["tokens"]:.0f} tokens; the calibration set has {ctok} text + '
                             f'{itok} image-sequence tokens = {ctok + itok} (a --max-seqs test run, or other '
                             'images?); refusing to write the release body')
        out.update({'calib_path': str(calib), 'calib_tokens': ctok, 'images_path': str(images) if images else None,
                    'image_tokens': itok, 'images_sha256': isha})
    return out


class Stats:
    """Per-column mean x^2 = (FP64 sum of squares) / tokens, floored at IMATRIX_FLOOR x mean."""
    def __init__(self, root):
        self.root, self.cache, self.files = Path(root), {}, {}

    def _file(self, stem):
        if stem not in self.cache:
            from safetensors import safe_open
            path = self.root / f'{stem}.safetensors'
            with safe_open(str(path), framework='numpy') as f:
                self.cache[stem] = {k: f.get_tensor(k) for k in f.keys() if k in STATS_KEYS or k in ('tokens', 'head_in')}
                meta = f.metadata() or {}
            tokens = float(np.asarray(self.cache[stem]['tokens'], np.float64).reshape(-1)[0])
            self.files[path.name] = {'sha256': sha_file(path), 'tokens': tokens,
                                     'keys': sorted(self.cache[stem]), 'metadata': meta}
        return self.cache[stem]

    def vector(self, name, key, cols):
        stem = stats_stem(name)
        st = self._file(stem)
        tokens = float(np.asarray(st['tokens'], np.float64).reshape(-1)[0])
        if not tokens > 0:
            raise ValueError(f'{stem}: no tokens')
        if key not in st:
            raise ValueError(f'{stem}.safetensors has no {key} (needed by {name})')
        s = np.asarray(st[key], np.float64)
        if s.shape != (cols,) or not np.isfinite(s).all() or (s < 0).any():
            raise ValueError(f'{stem}.{key}: shape {s.shape} (want ({cols},)) / nonfinite / negative')
        h = s / tokens
        h = np.maximum(h, IMATRIX_FLOOR * max(h.mean(), 1e-300))
        return h.astype(np.float32)


# ---------------------------------------------------------------- part writing and verification

def source_files_record(src, names):
    files = revision_tree()
    out = {}
    for shard in sorted({src.index[n] for n in names}):
        t = files.get(shard, {})
        size = (src.root / shard).stat().st_size
        if t.get('size') is not None and t['size'] != size:
            raise ValueError(f'{shard}: size {size} != revision tree {t["size"]}')
        out[shard] = {'bytes': size, 'lfs_sha256 (revision tree)': t.get('lfs_sha256')}
    return out


def verify_part(path, tensors, meta, lib, threads):
    """Strict re-read (inkling2_vq_format.PartReader) + metadata round trip + every payload SHA-256 (preserved
    tensors against their source bytes) + numpy/C decode of every quantized payload from the file, which must
    reproduce the decode measured at encode time. -> verification record."""
    q = None
    checked = {'quantized_decode_checked': 0, 'preserved_bit_exact': 0}
    with fmt.PartReader(path) as r:
        if r.order != [t['name'] for t in tensors]:
            raise AssertionError(f'{path}: stored order')
        offsets = [r.entries[n]['offset'] for n in r.order]
        if offsets != sorted(offsets) or (offsets and offsets[0] != 0):
            raise AssertionError(f'{path}: descriptors not in storage order')
        if meta is not None and dict(r.meta) != dict(meta):
            raise AssertionError(f'{path}: metadata round trip')
        for t in tensors:
            e = r.entries[t['name']]
            if list(e['shape']) != t['stored_shape'] or e['type_id'] != t['type_id'] \
                    or e['payload_bytes'] != t['payload_bytes']:
                raise AssertionError(f're-read mismatch {t["name"]}')

        def digest(name):
            raw = r.raw(name)
            try:
                return hashlib.sha256(raw).hexdigest()
            finally:
                raw.release()
        with ThreadPoolExecutor(max(1, min(threads, 8))) as tp:
            digests = dict(zip(r.order, tp.map(digest, r.order)))
        for t in tensors:
            if digests[t['name']] != t['payload_sha256']:
                raise AssertionError(f'payload hash {t["name"]}')
            if t.get('preserved'):
                if digests[t['name']] != t['source_sha256']:
                    raise AssertionError(f'preserved tensor differs from its source bytes: {t["name"]}')
                checked['preserved_bit_exact'] += 1
            elif t['type_id'] in QBYTES:
                if q is None:
                    q = Quantizer(lib, threads)
                raw = r.raw(t['name'])
                try:
                    payload = bytes(raw)
                finally:
                    raw.release()
                if decode_digest(q, t['type_id'], payload, t['stored_shape'], threads) != t['decoded_sha256']:
                    raise AssertionError(f'decode from the file differs from the measured decode: {t["name"]}')
                del payload
                checked['quantized_decode_checked'] += 1
            else:
                raise AssertionError(f'{t["name"]}: neither preserved nor quantized')
        meta_back = dict(r.meta)
        n = len(r.entries)
    if checked['quantized_decode_checked'] + checked['preserved_bit_exact'] != n:
        raise AssertionError('verification count')
    return meta_back, {'reader': 'inkling2_vq_format.PartReader (strict: layout, metadata namespace, finite '
                                 'native payloads and quant scales)', 'metadata_roundtrip': meta is not None,
                       'payload_sha256_rechecked': n, **checked,
                       'decode_check': 'numpy and C decoders on the payload read back from the file, bit-equal, '
                                       'finite, decoded SHA-256 equal to the encode-time measurement'}


def write_part(plan, meta, out_path, receipt_path, sources, pool, lib, threads, extra_receipt):
    """plan: ordered [{name, source_name, src, type_id, stored_shape, source_shape, source_dtype, disposition, job?}]."""
    t0 = time.monotonic()
    out_path = Path(out_path)
    if out_path.exists():
        raise SystemExit(f'{out_path} exists; remove it first (or pass --overwrite)')
    partial = out_path.with_name(out_path.name + '.partial')
    if partial.exists():
        partial.rename(partial.with_name(partial.name + f'.failed-{time.time_ns()}'))
    futures = {}
    for p in sorted((p for p in plan if p.get('job')), key=lambda p: -math.prod(p['stored_shape'])):
        futures[p['name']] = pool.submit(quant_job, p['job'])
    entries = [{'name': p['name'], 'shape': list(p['stored_shape']), 'type_id': p['type_id'],
                'payload_bytes': payload_bytes(p['type_id'], p['stored_shape'])} for p in plan]
    tensors, mapping = [], []
    try:
        with fmt.PartWriter(partial, meta, entries) as w:
            for i, (p, entry) in enumerate(zip(plan, entries)):
                n = p['name']
                rec = {'name': n, 'part': out_path.name, 'type': TYPE_NAMES[p['type_id']], 'type_id': p['type_id'],
                       'stored_shape': list(p['stored_shape']), 'source_shape': list(p['source_shape']),
                       'source_dtype': p['source_dtype'], 'source_checkpoint': p['src'],
                       'source_name': p['source_name'], 'source_file': sources[p['src']].index[p['source_name']]}
                if p.get('job'):
                    name, raw, qrec = futures.pop(n).result()
                    if name != n or len(raw) != entry['payload_bytes'] or sha_bytes(raw) != qrec['payload_sha256']:
                        raise AssertionError(f'encoded payload {n}')
                    w.write(n, raw)
                    digest = qrec['payload_sha256']
                    rec.update({k: v for k, v in qrec.items() if k not in ('type', 'source_dtype')})
                    rec['importance'] = p['job'].get('importance')
                    del raw
                    print(json.dumps({'part': out_path.name, 'i': i + 1, 'of': len(plan), 'tensor': n,
                                      'type': rec['type'], 'rel_mse': rec['rel_mse'],
                                      **({'importance_weighted_rel_mse': rec['importance_weighted_rel_mse']}
                                         if 'importance_weighted_rel_mse' in rec else {}),
                                      'seconds': rec['seconds']}), flush=True)
                else:
                    buf = sources[p['src']].read_bytes(p['source_name'])
                    if buf.size != entry['payload_bytes']:
                        raise AssertionError(f'source size {n}')
                    digest = sha_bytes(buf)
                    w.write(n, buf)
                    del buf
                    rec['preserved'] = True
                    rec['source_sha256'] = digest
                rec['payload_bytes'] = entry['payload_bytes']
                rec['payload_sha256'] = digest
                if p['stored_shape'] != p['source_shape']:
                    rec['reshaped'] = 'row-major bytes identical'
                tensors.append(rec)
                mapping.append({'source_checkpoint': p['src'], 'source_name': p['source_name'], 'target_name': n,
                                'part': out_path.name, 'disposition': p['disposition']})
            w.file.flush()
            os.fsync(w.file.fileno())
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    os.replace(partial, out_path)
    meta_back, verification = verify_part(out_path, tensors, meta, lib, threads)
    for t in tensors:
        if t['type_id'] in QBYTES:
            t['reread_decode_check'] = True
    receipt = {'part': out_path.name, 'path': str(out_path), 'role': meta[NS + 'part_role'],
               'bytes': out_path.stat().st_size, 'sha256': sha_file(out_path),
               'source_repo': SOURCE_REPO, 'source_revision': fmt.SOURCE_REVISION, 'recipe_id': fmt.RECIPE_ID,
               'metadata': meta_back, 'tensor_count': len(tensors),
               'executed_source_sha256': dict(EXECUTED_SOURCES),
               'verification': verification,
               'seconds': round(time.monotonic() - t0, 1), **extra_receipt,
               'tensors': tensors, 'source_mapping': mapping}
    receipt['summary'] = summarize(tensors)
    write_json(receipt_path, receipt)
    q = [t for t in tensors if t['type'] in ('Q4_K', 'Q8_0')]
    print(json.dumps({'part': out_path.name, 'bytes': receipt['bytes'], 'GiB': receipt['bytes'] / 2**30,
                      'sha256': receipt['sha256'], 'tensors': len(tensors), 'quantized': len(q),
                      'seconds': receipt['seconds'], 'receipt': str(receipt_path), 'verification': verification,
                      'summary': {t: {k: v for k, v in s.items() if k != 'by_tensor_class'}
                                  for t, s in receipt['summary'].items()}}, indent=1), flush=True)
    return receipt


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


def summarize(tensors):
    out = {}
    for typ in ('Q4_K', 'Q8_0'):
        q = [t for t in tensors if t['type'] == typ]
        if not q:
            continue
        r = np.array([t['rel_mse'] for t in q])
        worst = max(q, key=lambda t: t['rel_mse'])
        num = sum(t['rel_mse'] * t['weight_rms'] ** 2 * math.prod(t['stored_shape']) for t in q)
        den = sum(t['weight_rms'] ** 2 * math.prod(t['stored_shape']) for t in q)
        s = {'tensors': len(q), 'rel_mse_mean': float(r.mean()), 'rel_mse_median': float(np.median(r)),
             'rel_mse_min': float(r.min()), 'rel_mse_max': float(r.max()), 'rel_mse_pooled': float(num / den),
             'worst': worst['name'], 'decoders_agree': all(t['decoders_agree'] for t in q),
             'reread_decode_checked': all(t.get('reread_decode_check') for t in q)}
        iw = [t['importance_weighted_rel_mse'] for t in q if 'importance_weighted_rel_mse' in t]
        if iw:
            s['importance_weighted_rel_mse_mean'] = float(np.mean(iw))
            s['importance_weighted_rel_mse_max'] = float(np.max(iw))
            s['importance_weighted_worst'] = max((t for t in q if 'importance_weighted_rel_mse' in t),
                                                 key=lambda t: t['importance_weighted_rel_mse'])['name']
        groups = {}
        for t in q:
            g = re.sub(r'\.\d+\.', '.N.', t['name'])
            groups.setdefault(g, []).append(t)
        s['by_tensor_class'] = {
            g: {'n': len(v), 'mean': float(np.mean([t['rel_mse'] for t in v])),
                'max': float(np.max([t['rel_mse'] for t in v])),
                **({'importance_weighted_mean': float(np.mean([t['importance_weighted_rel_mse'] for t in v])),
                    'importance_weighted_max': float(np.max([t['importance_weighted_rel_mse'] for t in v]))}
                   if all('importance_weighted_rel_mse' in t for t in v) else {})}
            for g, v in sorted(groups.items())}
        out[typ] = s
    return out


def native_policy(**kw):
    return json.dumps(kw, sort_keys=True)


def family_extra(extra):
    return {NS + 'source_revision': fmt.SOURCE_REVISION, NS + 'recipe_id': fmt.RECIPE_ID, **extra}


def make_pool(lib, workers, threads, sources):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'), initializer=_worker_init,
                               initargs=(str(lib), threads, sources))


def default_receipt(out_path):
    """<dir>/receipts/<stem>.json: parts/receipts/ for the release, the same layout for test directories."""
    out_path = Path(out_path)
    return out_path.parent / 'receipts' / (out_path.stem + '.json')


def prep_output(a, kind):
    out = Path(a.output) if a.output else PARTS / PART_FILES[kind]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and a.overwrite:
        out.unlink()
    return out, Path(a.receipt) if a.receipt else default_receipt(out)


def preserved_entry(src, name, checkpoint='main'):
    _, _, _, dtype, shape = src.info(name)
    if dtype not in NATIVE:
        raise ValueError(f'{name}: cannot preserve {dtype}')
    if not 1 <= len(shape) <= 4:
        raise ValueError(f'{name}: rank {len(shape)} (GGUF allows 1-4 dims; none expected in Inkling-2-Small)')
    return {'name': target_name(checkpoint, name), 'source_name': name, 'src': checkpoint, 'type_id': NATIVE[dtype],
            'stored_shape': list(shape), 'source_shape': list(shape), 'source_dtype': dtype,
            'disposition': 'preserved'}


# ---------------------------------------------------------------- commands

def cmd_embed(a):
    src = NpSource(SOURCE, 'main')
    _, _, _, dtype, shape = src.info(EMBED)
    if dtype != 'BF16' or shape != (VOCAB, HIDDEN):
        raise ValueError(f'embedding {dtype} {shape}')
    out, rpath = prep_output(a, 'embed')
    plan = [preserved_entry(src, EMBED)]
    meta = fmt.metadata(part_role='embedding', extra=family_extra({
        NS + 'residency': 'ssd-rows',
        NS + 'native_policy': native_policy(
            policy_id='inkling2-embedding-bf16-ssd-rows-v1', source_bytes={'BF16': [f'{EMBED} [{VOCAB}, {HIDDEN}]']},
            residency=f'ssd-rows: the host reads one {2 * HIDDEN:,}-byte BF16 row per token id; never uploaded whole '
                      f'(model.llm.embed_norm, in the body part, is applied after the lookup)')}))
    write_part(plan, meta, out, rpath, {'main': src}, None, None, 1,
               {'source_files': source_files_record(src, [EMBED])})


def cmd_vision(a):
    src = NpSource(SOURCE, 'main')
    infos = src.all_infos()
    names = sorted((n for n in infos if main_policy(n, infos[n][4], infos[n][3])[0] == 'vision'), key=natural_key)
    plan = [preserved_entry(src, n) for n in names]
    dtypes = sorted({p['source_dtype'] for p in plan})
    meta = fmt.metadata(part_role='vision', extra=family_extra({
        NS + 'residency': 'mmap-on-demand',
        NS + 'reshaped_tensors': json.dumps({}),
        NS + 'native_policy': native_policy(
            policy_id='inkling2-vision-source-precision-v1',
            quantized='none (DeepSeek V4.1 / MiniMax practice: every vision tensor at original source precision)',
            source_bytes={d: shape_patterns([p['name'] for p in plan if p['source_dtype'] == d]) for d in dtypes},
            residency='mmap-on-demand: mapped, read only when an image arrives; not resident')}))
    write_part(plan, meta, *prep_output(a, 'vision'), {'main': src}, None, None, 1,
               {'source_files': source_files_record(src, names), 'reshaped_tensors': {}})


def cmd_mtp(a):
    src = NpSource(SOURCE, 'mtp', single=MTP_FILE)
    infos = src.all_infos()
    cfg = config()
    mtp_cfg = cfg['cfg']['mtp_config']
    if (mtp_cfg['num_hidden_layers'], mtp_cfg['hidden_size']) != (MTP_LAYERS, HIDDEN):
        raise ValueError('mtp_config')
    mtp_cfg_text = json.dumps(mtp_cfg)              # source key order
    mtp_path = SOURCE / MTP_FILE
    t0 = time.monotonic()
    mtp_sha = sha_file(mtp_path)
    tree = revision_tree().get(MTP_FILE, {})
    if tree.get('lfs_sha256') is not None and tree['lfs_sha256'] != mtp_sha:
        raise SystemExit(f'{mtp_path}: sha256 {mtp_sha} != revision tree {tree["lfs_sha256"]}')
    print(json.dumps({'mtp_safetensors_sha256': mtp_sha, 'matches_revision_tree': tree.get('lfs_sha256') == mtp_sha,
                      'seconds': round(time.monotonic() - t0, 1)}), flush=True)
    lib, build = build_native()
    plan, per_layer = [], {}
    for n in sorted(infos, key=natural_key):
        _, _, _, dtype, shape = infos[n]
        kind = mtp_policy(n, shape, dtype)
        m = re.fullmatch(r'layers\.(\d+)\.(.+)', n)
        if m:
            per_layer.setdefault(int(m[1]), set()).add(m[2])
        if kind == 'q8_0':
            t = target_name('mtp', n)
            plan.append({'name': t, 'source_name': n, 'src': 'mtp', 'type_id': Q8_0, 'stored_shape': list(shape),
                         'source_shape': list(shape), 'source_dtype': dtype, 'disposition': 'consumed_q8_0',
                         'job': {'name': t, 'source_name': n, 'src': 'mtp', 'type_id': Q8_0, 'shape': list(shape),
                                 'importance': None}})
        else:
            plan.append(preserved_entry(src, n, 'mtp'))
    for L in range(MTP_LAYERS):
        if per_layer.get(L) != MTP_Q8 | MTP_LAYER_PRESERVE:
            raise ValueError(f'mtp layer {L} inventory: {sorted(per_layer.get(L, set()) ^ (MTP_Q8 | MTP_LAYER_PRESERVE))}')
    if sorted(per_layer) != list(range(MTP_LAYERS)):
        raise ValueError(f'mtp layers {sorted(per_layer)}')
    preserved = [p['name'] for p in plan if not p.get('job')]
    meta = fmt.metadata(part_role='drafter', extra=family_extra({
        NS + 'mtp_config': mtp_cfg_text,
        NS + 'mtp_safetensors_sha256': mtp_sha,
        NS + 'native_policy': native_policy(
            policy_id='inkling2-mtp-q8_0-v1',
            name_mapping='tensor X of mtp.safetensors is stored as mtp.X',
            Q8_0={'encoder': ENCODER, 'tensors': [
                'mtp.layers.{0-4}.self_attn.{q_proj,k_proj,v_proj,o_proj,k_proj_ctx,v_proj_ctx}.weight (BF16 source)',
                'mtp.layers.{0-4}.mlp.{gate_proj,up_proj,down_proj}.weight (BF16 source)']},
            source_bytes={'BF16': shape_patterns(preserved)})}))
    with make_pool(lib, a.workers, a.threads, {'mtp': src}) as pool:
        write_part(plan, meta, *prep_output(a, 'mtp'), {'mtp': src}, pool, lib, verify_threads(a),
                   {'build': build, 'source_files': source_files_record(src, list(infos)),
                    'mtp_safetensors': {'path': str(mtp_path), 'bytes': mtp_path.stat().st_size, 'sha256': mtp_sha,
                                        'lfs_sha256 (revision tree)': tree.get('lfs_sha256')},
                    'mtp_config': {'source': f'{cfg["path"]} ["mtp_config"]', 'config_sha256': sha_file(cfg['path']),
                                   'config_git_blob_sha1': git_blob_sha1(cfg['path']),
                                   'config_blob_id (revision tree)': revision_tree().get('config.json', {}).get('blob_id'),
                                   'stored_json': mtp_cfg_text}})


def parse_layers(spec):
    if not spec:
        return set(range(N_LAYERS))
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out |= set(range(int(lo), int(hi or lo) + 1))
    if not out <= set(range(N_LAYERS)):
        raise SystemExit(f'--layers outside 0-{N_LAYERS - 1}')
    return out


def verify_threads(a):
    return max(1, min(28, a.workers * a.threads))


def cmd_body(a):
    unit = a.unit_importance
    if unit and a.stats:
        raise SystemExit('--stats and --unit-importance are exclusive')
    if not unit and not a.stats:
        raise SystemExit('--stats DIR or --unit-importance is required')
    out_path = Path(a.output) if a.output else PARTS / PART_FILES['body']
    final = out_path.resolve() == (PARTS / PART_FILES['body']).resolve()
    if final and (unit or a.layers or a.no_head):
        raise SystemExit('the release body (parts/body-nonexpert.gguf) needs calibration importance and every '
                         'tensor; write test builds elsewhere with --output')
    if not final and out_path.resolve().parent == PARTS.resolve():
        raise SystemExit(f'test builds do not go into {PARTS}; pass --output elsewhere')
    src = NpSource(SOURCE, 'main')
    infos = src.all_infos()
    layers = parse_layers(a.layers)
    all_body = []
    for n, (_, _, _, dtype, shape) in infos.items():
        if main_policy(n, shape, dtype)[0] == 'body':
            all_body.append(n)
    check_body_inventory(set(all_body))
    names = []
    for n in all_body:
        L = layer_of(n)
        if (L is not None and L not in layers) or (L is None and a.no_head):
            continue
        names.append(n)
    names.sort(key=body_order)
    stats, pre = None, None
    if not unit:
        stems = [f'layer-{L:02d}-nonexpert' for L in range(N_LAYERS)] + ['head'] if final else \
            sorted({stats_stem(n) for n in names if main_policy(n, infos[n][4], infos[n][3])[1] == 'q4k'})
        pre = preflight_stats(a.stats, stems, final, a.calib, a.images)
        stats = Stats(a.stats)
    lib, build = build_native()
    plan = []
    for n in names:
        _, _, _, dtype, shape = infos[n]
        _, kind, key = main_policy(n, shape, dtype)
        if kind == 'preserve':
            plan.append(preserved_entry(src, n))
            continue
        cols = shape[1]
        job = {'name': n, 'source_name': n, 'src': 'main', 'type_id': Q4_K, 'shape': list(shape)}
        if stats is not None:
            h = stats.vector(n, key, cols)
            job['measure_w'] = h
            job['qw'] = h
            job['importance'] = {'source': 'calibration', 'key': key, 'file': f'{stats_stem(n)}.safetensors',
                                 'used_by_encoder': True, 'vector_f32_sha256': sha_bytes(h.tobytes())}
        else:
            job['qw'] = np.ones(cols, np.float32)
            job['importance'] = {'source': 'unit', 'key': key, 'used_by_encoder': True}
        plan.append({'name': n, 'source_name': n, 'src': 'main', 'type_id': Q4_K, 'stored_shape': list(shape),
                     'source_shape': list(shape), 'source_dtype': dtype, 'disposition': 'consumed_q4k', 'job': job})
    if stats is not None:       # the vectors come from the files the preflight accepted (the statistics run writes live)
        for fname, f in stats.files.items():
            if f['tokens'] != pre['tokens'] or f['metadata'].get('calib_sha256') != pre['calib_sha256'] \
                    or f['metadata'].get('source_revision') != fmt.SOURCE_REVISION:
                raise SystemExit(f'{fname} changed after the statistics preflight')
    importance =('unit importance (all-ones per-column weights through the same GGML importance path); '
                  'TEST BUILD, not the release body') if unit else \
        ('calibration mean x^2 per input column = FP64 sum of squares / tokens, floored at 1e-8 x mean: '
         'q_in -> attn.wq_du, attn.wr_du, attn.gate_attn; k_in -> attn.wk_dv; v_in -> attn.wv_dv '
         '(q_in = k_in = v_in in sliding-window layers: one attn_norm input); o_in -> attn.wo_ud (the gated '
         'attention output); head_in -> model.llm.unembed (the final normalised hidden state)')
    meta = fmt.metadata(part_role='trunk', extra=family_extra({
        NS + 'nonexpert_importance': 'unit (test build)' if unit else 'calibration mean x^2 per input column',
        NS + 'reshaped_tensors': json.dumps({}),
        NS + 'native_policy': native_policy(
            policy_id='inkling2-nonexpert-q4k-imatrix-v1',
            Q4_K={'encoder': ENCODER, 'importance': importance, 'tensors': [
                'model.llm.layers.{0-47}.attn.{wq_du,wk_dv,wv_dv,wr_du,gate_attn,wo_ud}.weight (BF16 source)',
                f'{UNEMBED} [{VOCAB}, {HIDDEN}] (BF16 source)']},
            source_bytes={'BF16': ['attn.attn_sink_proj.weight', 'attn.q_l2_clip.scale', 'attn.rel_l2_clip.scale',
                                   'attn.rel_logits_proj.proj',
                                   'attn.{k,v}_sconv.weight [1024, 1, 4] (sliding-window layers)',
                                   'attn.{k,v}_sconv.{static_w,weight_proj} (global layers)',
                                   'attn_sconv.weight, mlp_sconv.weight [4608, 1, 4]',
                                   'attn_norm (sliding-window layers), {q,k,v}_input_norm (global layers), mlp_norm',
                                   'mlp.gate.weight (router)', FINAL_NORM, EMBED_NORM],
                          'F32': ['mlp.gate.bias (router bias)',
                                  ATTN_RES + '{attn_queries.N, mlp_queries.N, output_query}']},
            shapes='source shapes kept; the [C, 1, 4] short-convolution weights stay 3-D (GGUF allows 4 dims)',
            global_layers=global_layers())}))
    complete_set = layers == set(range(N_LAYERS)) and not a.no_head
    extra = {'build': build, 'complete': final, 'all_tensors': complete_set, 'layers': sorted(layers),
             'head': not a.no_head,
             'importance': {'source': 'unit' if unit else 'calibration', 'description': importance,
                            'floor': IMATRIX_FLOOR}}
    if stats is not None:
        sdir = Path(a.stats)
        extra['importance'].update({'dir': str(sdir), 'files': stats.files, 'preflight': pre})
        for f in ('DONE', 'summary.json'):
            if (sdir / f).exists():
                extra['importance'][f + '_sha256'] = sha_file(sdir / f)
    extra['source_files'] = source_files_record(src, names)
    out, rpath = prep_output(a, 'body')
    with make_pool(lib, a.workers, a.threads, {'main': src}) as pool:
        write_part(plan, meta, out, rpath, {'main': src}, pool, lib, verify_threads(a), extra)
    if final:
        merge(PARTS, allow_test_body=False)


# ---------------------------------------------------------------- merge

def expert_rows(main):
    """Accounting rows of the routed experts (converted by the hVQ expert tool, run_hvq_inkling2.py)."""
    rows = []
    for n in sorted(main.index, key=natural_key):
        m = EXPERT_RE.fullmatch(n)
        if not m:
            continue
        _, _, _, dtype, shape = main.info(n)
        main_policy(n, shape, dtype)            # dtype / shape check
        L = int(m[1])
        projs = ('gate_proj', 'up_proj') if m[2] == 'w13_weight' else ('down_proj',)
        weight = n[:-len('_scale')] if m[3] else n
        proj = '{%s}' % ','.join(projs) if len(projs) > 1 else projs[0]
        rows.append({'source_checkpoint': 'main', 'source_name': n,
                     'target_name': fmt.PREFIX.format(L=L) + 'experts.{0..%d}.%s.weight' % (fmt.N_EXPERTS - 1, proj),
                     'target_count': fmt.N_EXPERTS * len(projs),
                     'part': f'layer-{L:02d}-experts.gguf',
                     'disposition': 'consumed_scale' if m[3] else 'consumed_vq',
                     **({'scale_of': weight} if m[3] else
                        {'layout': 'gate = rows 0::2, up = rows 1::2 of w13_weight[E]' if m[2] == 'w13_weight'
                         else 'down = w2_weight[E]'}),
                     'converted_by': 'hVQ expert parts (run_hvq_inkling2.py), not this tool'})
    return rows


def merge(parts_dir, allow_test_body=False):
    """Combine the per-part receipts; check the source mapping against both checkpoints."""
    parts_dir = Path(parts_dir)
    receipts_dir = parts_dir / 'receipts'
    release = parts_dir.resolve() == PARTS.resolve()
    if release and allow_test_body:
        raise SystemExit('--allow-test-body is for test directories, not the release parts')
    main, mtp = NpSource(SOURCE, 'main'), NpSource(SOURCE, 'mtp', single=MTP_FILE)
    sources = {'main': main, 'mtp': mtp}
    parts, pending, tensors, mapping, test_body = {}, [], [], [], False
    for kind, fname in PART_FILES.items():
        rp = receipts_dir / (Path(fname).stem + '.json')
        if not rp.exists():
            pending.append(fname)
            continue
        r = json.loads(rp.read_text())
        path = parts_dir / fname
        if r['part'] != fname or not path.exists() or path.stat().st_size != r['bytes']:
            raise SystemExit(f'{fname}: receipt does not match the file')
        if r['tensor_count'] != len(r['tensors']) or r['source_revision'] != fmt.SOURCE_REVISION \
                or r['recipe_id'] != fmt.RECIPE_ID:
            raise SystemExit(f'{fname}: incomplete receipt / other revision or recipe')
        if kind == 'body' and not r.get('complete'):
            if not (allow_test_body and r.get('all_tensors')):
                raise SystemExit(f'{fname}: not a release body (test build or subset); '
                                 'a test directory takes --allow-test-body with an all-tensor test body')
            test_body = True
        parts[fname] = r
        tensors += r['tensors']
        mapping += r['source_mapping']
    with ThreadPoolExecutor(4) as tp:
        hashes = dict(zip(parts, tp.map(lambda f: sha_file(parts_dir / f), parts)))
    for f, r in parts.items():
        if hashes[f] != r['sha256']:
            raise SystemExit(f'{f}: file sha256 differs from its receipt')
    # inventory checks
    stored = {}
    for t in tensors:
        if t['name'] in stored:
            raise SystemExit(f'duplicate stored tensor {t["name"]}')
        stored[t['name']] = t
    expected = {ck: set(s.index) for ck, s in sources.items()}
    seen = {ck: set() for ck in sources}
    want_disp = {Q4_K: 'consumed_q4k', Q8_0: 'consumed_q8_0', BF16: 'preserved', F32: 'preserved', F16: 'preserved'}
    targets = []
    for row in mapping:
        ck, s, t, d = row['source_checkpoint'], row['source_name'], row['target_name'], row['disposition']
        if ck not in expected or s not in expected[ck] or s in seen[ck] or t not in stored:
            raise SystemExit(f'source mapping duplicate/unknown: {row}')
        st = stored[t]
        if t != target_name(ck, s) or want_disp[st['type_id']] != d or st['part'] != row['part'] \
                or st['source_checkpoint'] != ck or st['source_name'] != s:
            raise SystemExit(f'bad mapping {row}')
        seen[ck].add(s)
        targets.append(t)
    if sorted(targets) != sorted(stored):
        raise SystemExit('stored tensor without (or with several) source rows')
    experts = expert_rows(main)
    for row in experts:
        if row['source_name'] in seen['main']:
            raise SystemExit(f'expert tensor mapped twice: {row["source_name"]}')
        seen['main'].add(row['source_name'])
    mapping = mapping + experts
    unmapped = {ck: sorted(expected[ck] - seen[ck], key=natural_key) for ck in expected}
    complete = not pending and not any(unmapped.values())
    if not pending and any(unmapped.values()):
        raise SystemExit(f'source tensors not accounted: { {k: v[:10] for k, v in unmapped.items()} }')
    expert_parts = [f'layer-{L:02d}-experts.gguf' for L in range(N_LAYERS)]
    summary = summarize(tensors)
    receipt = {
        'receipt': 'inkling2-vq8 non-expert parts', 'complete': complete, 'pending_parts': pending,
        'release': release and complete and not test_body, 'test_build': test_body,
        'source_repo': SOURCE_REPO, 'source_revision': fmt.SOURCE_REVISION, 'recipe_id': fmt.RECIPE_ID,
        'format': {'architecture': fmt.ARCH, 'variant': fmt.VARIANT, 'format_version': 1, 'alignment': fmt.ALIGN,
                   'source_model_type': fmt.SOURCE_MODEL_TYPE},
        'parts': {f: {'role': r['role'], 'bytes': r['bytes'], 'GiB': r['bytes'] / 2**30, 'sha256': r['sha256'],
                      'tensors': r['tensor_count'], 'residency': r['metadata'].get(NS + 'residency'),
                      'native_policy': json.loads(r['metadata'][NS + 'native_policy']),
                      **({'importance': r['importance']} if 'importance' in r else {}),
                      **({'build': r['build']} if 'build' in r else {}),
                      **({'mtp_safetensors': r['mtp_safetensors']} if 'mtp_safetensors' in r else {}),
                      'verification': r.get('verification'),
                      'source_files': r.get('source_files'), 'executed_source_sha256': r['executed_source_sha256']}
                  for f, r in parts.items()},
        'summary': summary, 'summary_by_part': {f: r['summary'] for f, r in parts.items()},
        'coverage': {
            'main': {'source_tensors': len(main.index), 'mapped': len(seen['main']),
                     'mapped_to_these_parts': len(seen['main']) - len(experts),
                     'routed_expert_tensors_accounted': len(experts),
                     'routed_expert_parts': {'present': sum((parts_dir / p).exists() for p in expert_parts),
                                             'of': len(expert_parts)},
                     'unmapped': unmapped['main'][:50], 'unmapped_count': len(unmapped['main'])},
            'mtp': {'source_file': MTP_FILE, 'source_tensors': len(mtp.index), 'mapped': len(seen['mtp']),
                    'unmapped': unmapped['mtp'][:50], 'unmapped_count': len(unmapped['mtp'])}},
        'dispositions': {d: sum(1 for r in mapping if r['disposition'] == d)
                         for d in ('preserved', 'consumed_q4k', 'consumed_q8_0', 'consumed_vq', 'consumed_scale')},
        'source_mapping': mapping,
        'tensors': [{k: t[k] for k in ('name', 'part', 'type', 'type_id', 'stored_shape', 'source_shape',
                                       'source_dtype', 'source_checkpoint', 'source_name', 'payload_bytes',
                                       'payload_sha256', 'rel_mse', 'importance_weighted_rel_mse', 'rmse',
                                       'weight_rms', 'row_rel_rms_p99', 'row_rel_rms_max', 'max_abs_err',
                                       'decoded_sha256', 'importance', 'reshaped') if k in t}
                    for t in tensors]}
    out = receipts_dir / 'nonexpert-receipt.json'
    write_json(out, receipt)
    print(json.dumps({'receipt': str(out), 'complete': complete, 'release': receipt['release'],
                      'test_build': test_body, 'pending_parts': pending,
                      'parts': {f: {k: v for k, v in p.items() if k in ('bytes', 'GiB', 'sha256', 'tensors')}
                                for f, p in receipt['parts'].items()},
                      'coverage': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                                   for k, v in receipt['coverage'].items()},
                      'dispositions': receipt['dispositions'],
                      'summary': {t: {k: v for k, v in s.items() if k != 'by_tensor_class'}
                                  for t, s in summary.items()}}, indent=1))
    return receipt


def cmd_merge(a):
    merge(Path(a.parts) if a.parts else PARTS, a.allow_test_body)


def cmd_verify(a):
    """Re-verify written parts against their receipts (strict re-read, payload hashes, decode check)."""
    lib = None
    for part in a.part:
        part = Path(part)
        rp = Path(a.receipt) if a.receipt and len(a.part) == 1 else default_receipt(part)
        r = json.loads(rp.read_text())
        if r['part'] != part.name or part.stat().st_size != r['bytes']:
            raise SystemExit(f'{part}: receipt {rp} does not match')
        if lib is None and any(t['type_id'] in QBYTES for t in r['tensors']):
            lib, _ = build_native()
        t0 = time.monotonic()
        _, v = verify_part(part, r['tensors'], r['metadata'], lib, a.threads)
        sha = sha_file(part)
        if sha != r['sha256']:
            raise SystemExit(f'{part}: file sha256 {sha} != receipt {r["sha256"]}')
        print(json.dumps({'part': str(part), 'receipt': str(rp), 'sha256': sha, 'status': 'verified',
                          'seconds': round(time.monotonic() - t0, 1), **v}), flush=True)


def configure(source, work):
    """Point the converter at the source checkpoint and the work directory."""
    global SOURCE, WORK, PARTS, CALIB, IMAGES, BUILD
    SOURCE, WORK = source, work
    PARTS, BUILD = work / 'parts', work / 'build' / 'nonexpert-native'
    CALIB, IMAGES = work / 'calib' / 'calib.safetensors', work / 'calib' / 'images.safetensors'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8)')
    ap.add_argument('--work', type=Path, help='work directory: parts/, calib/, build/')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('embed', 'vision', 'mtp', 'body'):
        s = sub.add_parser(name)
        s.add_argument('--output', help='default WORK/parts/<part>.gguf')
        s.add_argument('--receipt', help='default <output dir>/receipts/<stem>.json')
        s.add_argument('--overwrite', action='store_true')
        s.add_argument('--workers', type=int, default=4, help='encoder processes (workers x threads <= 32 here)')
        s.add_argument('--threads', type=int, default=7, help='encoder / measure threads per worker')
        if name == 'body':
            s.add_argument('--stats', help='statistics directory (layer-LL-nonexpert.safetensors, head.safetensors)')
            s.add_argument('--unit-importance', action='store_true', help='test build: all-ones importance')
            s.add_argument('--calib', help='default WORK/calib/calib.safetensors; '
                           'release body: the text calibration set the statistics must come from (SHA-256, tokens)')
            s.add_argument('--images', help='default WORK/calib/images.safetensors; '
                           'release body: the image calibration sequences the statistics also cover (tokens); '
                                '"" for text-only statistics')
            s.add_argument('--layers', help='test builds only: subset of layers, e.g. 0,5,47 or 0-2')
            s.add_argument('--no-head', action='store_true',
                           help='test builds only: skip unembed, the norms and the attn_res queries')
    s = sub.add_parser('merge')
    s.add_argument('--parts', help='parts directory with receipts/ (default WORK/parts)')
    s.add_argument('--allow-test-body', action='store_true',
                   help='test directories: accept an all-tensor test body (unit importance); marks the receipt test_build')
    s = sub.add_parser('verify')
    s.add_argument('part', nargs='+')
    s.add_argument('--receipt', help='receipt of a single PART (default <dir>/receipts/<stem>.json)')
    s.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    # verify builds the C decoder under --work too; every other command also reads --source.
    if not a.work or (a.cmd != 'verify' and not a.source):
        ap.error(f'{a.cmd} needs --work' + ('' if a.cmd == 'verify' else ' and --source'))
    configure(a.source, a.work)
    if a.cmd == 'body':
        a.calib = str(CALIB) if a.calib is None else a.calib
        a.images = str(IMAGES) if a.images is None else a.images
    {'embed': cmd_embed, 'vision': cmd_vision, 'mtp': cmd_mtp, 'body': cmd_body, 'merge': cmd_merge,
     'verify': cmd_verify}[a.cmd](a)


if __name__ == '__main__':
    main()
