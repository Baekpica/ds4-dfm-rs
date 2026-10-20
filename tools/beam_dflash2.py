#!/usr/bin/env python3
"""Convert reflection/Beam-1.0-501B-A23B-DFlash2 (BF16 safetensors) into the
GGUF the Beam engine drafts with: matrices Q8_0, norms and conv base kernels
F32 (the device kernels read them as floats), the selector codebooks BF16 (the
host reads their rows), the DFlash2 config as metadata.

  beam_dflash2.py SRC_DIR OUT.gguf [--revision SHA]

Matrices keep their HF [out, in] order as GGUF [in, out] dims; Q8_0 blocks are
llama.cpp's (32 inputs, f16 amax/127 scale, rounded int8). fc.weight is split by
its five 5120-wide input blocks (fc.K.weight, one per target layer) so the
engine accumulates the context projection layer by layer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ALIGN = 32
F32, Q8_0, BF16 = 0, 8, 30
U32, F32_META, STRING = 4, 6, 8
QK = 32


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def meta_value(typ, val):
    if typ == STRING:
        return string(val)
    return struct.pack('<I' if typ == U32 else '<f', val)


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def safetensors(path):
    with path.open('rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        head = json.loads(f.read(n))
    head.pop('__metadata__', None)
    return 8 + n, head


def bf16_to_f32(raw, shape):
    bits = np.frombuffer(raw, dtype='<u2').astype('<u4') << 16
    return bits.view('<f4').reshape(shape)


def q8_0(x):
    rows, cols = x.shape
    assert cols % QK == 0, cols
    blocks = x.reshape(rows, cols // QK, QK)
    d = np.abs(blocks).max(axis=-1, keepdims=True) / 127.0
    q = np.where(d > 0, np.round(blocks / np.where(d > 0, d, 1)), 0).clip(-127, 127).astype('<i1')
    out = np.empty((rows, cols // QK), dtype=[('d', '<f2'), ('q', '<i1', QK)])
    out['d'] = d[..., 0].astype('<f2')
    out['q'] = q
    return out.tobytes()


def quantized(name, shape):
    # 2-D weights of the draft trunk and the selector projection.
    return len(shape) == 2 and name.endswith('.weight') and not name.startswith('candidate_selector.') \
        or name == 'candidate_selector.hidden_projection.weight'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--revision', default='')
    a = p.parse_args()
    cfg = json.loads((a.src / 'config.json').read_text())
    df = cfg['dflash_config']
    st = a.src / 'model.safetensors'
    with st.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    base, head = safetensors(st)
    taps = len(df['target_layer_ids'])
    hidden = cfg['hidden_size']
    fc = head.pop('fc.weight')
    assert fc['shape'] == [hidden, taps * hidden], fc['shape']
    for k in range(taps):
        head[f'fc.{k}.weight'] = {'dtype': 'BF16', 'shape': [hidden, hidden], 'fc': (fc, k)}
    names = sorted(head)
    entries, offset = [], 0
    for name in names:
        shape = head[name]['shape']
        assert head[name]['dtype'] == 'BF16', name
        typ = Q8_0 if quantized(name, shape) else BF16 if name.endswith('_codebook') else F32
        count = int(np.prod(shape))
        nbytes = count // QK * 34 if typ == Q8_0 else count * (2 if typ == BF16 else 4)
        entries.append({'name': name, 'shape': shape, 'typ': typ, 'off': offset, 'bytes': nbytes})
        offset = align(offset + nbytes)
    meta = {
        'general.architecture': (STRING, 'beam_dflash2'),
        'general.name': (STRING, 'Beam-1.0-501B-A23B-DFlash2'),
        'general.license': (STRING, 'apache-2.0'),
        'general.source.repo': (STRING, 'reflection/Beam-1.0-501B-A23B-DFlash2'),
        'general.source.revision': (STRING, a.revision),
        'general.source.sha256': (STRING, digest),
        'general.alignment': (U32, ALIGN),
        'beam_dflash2.block_count': (U32, cfg['num_hidden_layers']),
        'beam_dflash2.embedding_length': (U32, cfg['hidden_size']),
        'beam_dflash2.feed_forward_length': (U32, cfg['intermediate_size']),
        'beam_dflash2.attention.head_count': (U32, cfg['num_attention_heads']),
        'beam_dflash2.attention.head_count_kv': (U32, cfg['num_key_value_heads']),
        'beam_dflash2.attention.key_length': (U32, cfg['head_dim']),
        'beam_dflash2.attention.sliding_window': (U32, cfg['sliding_window']),
        'beam_dflash2.attention.layer_norm_rms_epsilon': (F32_META, cfg['rms_norm_eps']),
        'beam_dflash2.rope.freq_base': (F32_META, float(cfg['rope_theta'])),
        'beam_dflash2.vocab_size': (U32, cfg['vocab_size']),
        'beam_dflash2.block_size': (U32, df['block_size']),
        'beam_dflash2.mask_token_id': (U32, df['mask_token_id']),
        'beam_dflash2.conv_kernel_size': (U32, df['conv_kernel_size']),
        'beam_dflash2.conv_group_size': (U32, df['conv_group_size']),
        'beam_dflash2.selector_rank': (U32, df['selector_rank']),
        'beam_dflash2.selector_top_k': (U32, df['selector_top_k']),
        'beam_dflash2.final_logit_softcapping': (F32_META, float(df['final_logit_softcapping'])),
        'beam_dflash2.target_layers': (STRING, ','.join(str(i) for i in df['target_layer_ids'])),
    }
    out = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(entries), len(meta)))
    for key, (typ, val) in meta.items():
        out += string(key) + struct.pack('<I', typ) + meta_value(typ, val)
    for e in entries:
        dims = list(reversed(e['shape']))
        out += string(e['name']) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
        out += struct.pack('<IQ', e['typ'], e['off'])
    out += bytes(align(len(out)) - len(out))
    with st.open('rb') as src, a.out.open('wb') as dst:
        dst.write(out)
        for e in entries:
            meta_t = head[e['name']]
            whole, k = meta_t.get('fc', (meta_t, None))
            begin, end = whole['data_offsets']
            src.seek(base + begin)
            raw = src.read(end - begin)
            if k is not None:
                block = bf16_to_f32(raw, whole['shape'])[:, k * hidden:(k + 1) * hidden]
                data = q8_0(np.ascontiguousarray(block))
            elif e['typ'] == Q8_0:
                rows = e['shape'][0]
                data = q8_0(bf16_to_f32(raw, (rows, -1)))
            elif e['typ'] == F32:
                data = bf16_to_f32(raw, (-1,)).tobytes()
            else:
                data = raw
            assert len(data) == e['bytes'], e['name']
            dst.write(data)
            dst.write(bytes(align(e['bytes']) - e['bytes']))
    print(json.dumps({'out': str(a.out), 'tensors': len(entries), 'bytes': a.out.stat().st_size,
                      'source_sha256': digest}))


if __name__ == '__main__':
    main()
