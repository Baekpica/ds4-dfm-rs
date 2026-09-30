"""Independent Q8 decode and BF16 source-equation DSpark fixtures.

Requires NumPy, the pinned draft artifact and an output directory. No target
model is loaded: four synthetic tap rows and a synthetic anchor isolate the
draft backbone. This is not an acceptance or language-quality benchmark.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def bf(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000).view(np.float32)


def directory(path):
    f = path.open('rb')
    def scalar(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))[0]
    def string():
        return f.read(scalar('Q')).decode()
    def value(typ):
        if typ == 8:
            return string()
        if typ == 9:
            item, count = scalar('I'), scalar('Q')
            return [value(item) for _ in range(count)]
        return scalar({0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[typ])
    assert f.read(4) == b'GGUF' and scalar('I') == 3
    tensors, keys = scalar('Q'), scalar('Q')
    meta = {}
    for _ in range(keys):
        name = string()
        meta[name] = value(scalar('I'))
    layout = {}
    for _ in range(tensors):
        name = string()
        dims = [scalar('Q') for _ in range(scalar('I'))]
        layout[name] = (dims[::-1], scalar('I'), scalar('Q'))
    alignment = meta.get('general.alignment', 32)
    start = (f.tell() + alignment - 1) // alignment * alignment
    f.close()
    return meta, layout, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gguf', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    meta, layout, start = directory(args.gguf)
    assert meta['general.architecture'] == 'naive_n05_flash_dspark'
    raw = np.memmap(args.gguf, dtype=np.uint8, mode='r')
    weights = {}
    for name, (shape, typ, offset) in layout.items():
        count = int(np.prod(shape))
        if typ == 8:
            block = raw[start + offset:start + offset + count // 32 * 34].reshape(-1, 34)
            scale = np.ascontiguousarray(block[:, :2]).view('<f2').astype(np.float32)
            weights[name] = bf((scale * block[:, 2:].view(np.int8)).reshape(shape))
        else:
            assert typ == 0
            weights[name] = raw[start + offset:start + offset + count * 4].view('<f4').reshape(shape)
    def linear(x, name):
        return bf(x @ weights[name].T)
    def rms(x, name):
        inv = np.float32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5))
        return bf(bf(x * inv) * weights[name])
    freq = np.array([10000.0 ** (-2 * i / 128) for i in range(64)], dtype=np.float32)
    def rope(x, positions):
        phase = np.asarray(positions, dtype=np.float32)[:, None] * freq[None]
        c, s = bf(np.cos(phase))[:, None], bf(np.sin(phase))[:, None]
        a, b = x[:, :, :64], x[:, :, 64:]
        return np.concatenate([bf(bf(a * c) - bf(b * s)), bf(bf(b * c) + bf(a * s))], axis=-1)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    taps = bf(rng.normal(0, .05, (4, 32768)).astype(np.float32))
    anchor = bf(rng.normal(0, .05, (4096,)).astype(np.float32))
    taps.tofile(args.output / 'tap.f32')
    anchor.tofile(args.output / 'anchor.f32')
    feature = rms(linear(taps, 'fc.weight'), 'enc.output_norm.weight')
    reports = []
    for pos in [17, 1048, 1048569]:
        positions = np.arange(pos - 4, pos + 7)
        hidden = np.broadcast_to(weights['mask_embedding.weight'], (7, 4096)).copy()
        hidden[0] = anchor
        for il in range(5):
            prefix = f'blk.{il}.'
            norm = rms(hidden, prefix + 'attn_norm.weight')
            q = rope(rms(linear(norm, prefix + 'attn_q.weight').reshape(7, 32, 128),
                         prefix + 'attn_q_norm.weight'), positions[4:])
            ck = rms(linear(feature, prefix + 'attn_k.weight').reshape(4, 4, 128), prefix + 'attn_k_norm.weight')
            nk = rms(linear(norm, prefix + 'attn_k.weight').reshape(7, 4, 128), prefix + 'attn_k_norm.weight')
            k = rope(np.concatenate([ck, nk]), positions)
            v = np.concatenate([linear(feature, prefix + 'attn_v.weight').reshape(4, 4, 128),
                                linear(norm, prefix + 'attn_v.weight').reshape(7, 4, 128)])
            np.concatenate([k[:4].reshape(4, 512), v[:4].reshape(4, 512)], axis=-1).tofile(args.output / f'{pos}.kv{il}.f32')
            repeated_k, repeated_v = np.repeat(k, 8, axis=1), np.repeat(v, 8, axis=1)
            scores = bf(bf(np.einsum('qhd,khd->hqk', q, repeated_k)) * np.float32(128 ** -.5))
            allowed = np.abs(positions[4:, None] - positions[None]) < 1024
            scores = np.where(allowed[None], scores, -np.inf)
            exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs = bf(exps / np.sum(exps, axis=-1, keepdims=True))
            heads = bf(np.einsum('hqk,khd->qhd', probs, repeated_v)).reshape(7, 4096)
            hidden = bf(hidden + linear(heads, prefix + 'attn_output.weight'))
            norm = rms(hidden, prefix + 'ffn_norm.weight')
            gate, up = linear(norm, prefix + 'ffn_gate.weight'), linear(norm, prefix + 'ffn_up.weight')
            mid = bf(bf(gate / (1 + np.exp(-gate))) * up)
            hidden = bf(hidden + linear(mid, prefix + 'ffn_down.weight'))
        hidden = rms(hidden, 'output_norm.weight')
        assert np.isfinite(hidden).all()
        hidden.tofile(args.output / f'{pos}.hidden.f32')
        previous = 198
        markov = weights['markov_w1.weight'][previous]
        bias = linear(markov, 'markov_w2.weight')
        bias.tofile(args.output / f'{pos}.markov.f32')
        conf = bf(np.concatenate([hidden[1], markov]) @ weights['confidence_head.weight'].T + weights['confidence_head.bias'])
        conf.tofile(args.output / f'{pos}.confidence.f32')
        reports.append({'absolute_start': pos, 'context_rows': 4, 'noise_rows': 7, 'confidence': float(conf[0])})
    (args.output / 'reference.json').write_text(json.dumps({'scope': __doc__, 'cases': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
