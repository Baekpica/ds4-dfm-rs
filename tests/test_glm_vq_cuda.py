#!/usr/bin/env python3
"""Compare native GVQ1 restore with independent packed goldens; no full model."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import struct

import numpy as np
import torch

VQ = 0x47565138
BOOK = 0x47564342


def fixture(path):
    """Read only the tiny fixture container. Production validation is Rust."""
    data = path.read_bytes()
    magic, version, tensors, fields = struct.unpack_from('<4sIQQ', data)
    assert (magic, version) == (b'GGUF', 3)
    pos = 24

    def read(fmt):
        nonlocal pos
        value = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return value

    def string():
        nonlocal pos
        size, = read('<Q')
        value = data[pos:pos + size].decode()
        pos += size
        return value

    meta = {}
    for _ in range(fields):
        key = string()
        typ, = read('<I')
        meta[key] = string() if typ == 8 else read({4: '<I', 7: '<B'}[typ])[0]
    entries = {}
    for _ in range(tensors):
        name = string()
        rank, = read('<I')
        dims = read('<' + rank * 'Q')
        typ, off = read('<IQ')
        entries[name] = (typ, dims, off)
    base = (pos + 127) // 128 * 128
    typ, dims, off = entries['weight']
    assert typ == VQ
    h = struct.unpack_from('<4sHBBIIIIIIQQQQ32s32s', data, base + off)
    _, _, _, _, rows, cols, k, precision, book_id, _, gain, main, plane, size, _, _ = h
    assert dims == (cols, rows)
    book_name = json.loads(meta['glm_vq.codebooks'])[str(book_id)]
    typ, _, book_off = entries[book_name]
    assert typ == (BOOK if precision == 1 else 1)
    payload = data[base + off:base + off + size]
    m = np.frombuffer(payload, dtype='u1', count=rows * cols * 3 // 16, offset=main)
    e = np.frombuffer(payload, dtype='u1', count=rows * cols // 64, offset=plane) if plane else None
    g = np.frombuffer(payload, dtype='<f2', count=rows, offset=gain)
    b = np.frombuffer(data, dtype='u1' if precision == 1 else '<f2', count=k * 8, offset=base + book_off)
    return m, e, b, g, rows, cols, precision


class Kernels:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        ptr, u32 = ct.c_void_p, ct.c_uint32
        self.lib.test_glmvq_restore.argtypes = [ptr] * 4 + [u32] * 4 + [ptr] * 2
        self.lib.test_glmvq_fp8.argtypes = [ptr, ptr, u32, u32, u32, ptr, ptr]
        self.lib.test_glmvq_restore.restype = self.lib.test_glmvq_fp8.restype = ct.c_int

    def restore(self, m, e, b, g, rows, cols, precision, dtype):
        tensors = [torch.from_numpy(x.copy()).cuda() if x is not None else None for x in [m, e, b, g]]
        out = torch.empty((rows, cols), dtype=dtype, device='cuda')
        args = [x.data_ptr() if x is not None else None for x in tensors]
        rc = self.lib.test_glmvq_restore(*args, rows, cols, precision, int(dtype == torch.bfloat16), out.data_ptr(), torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
        torch.cuda.synchronize()
        return out

    def fp8(self, raw, scales, dtype):
        w, s = torch.from_numpy(raw.copy()).cuda(), torch.from_numpy(scales.copy()).cuda()
        out = torch.empty(raw.shape, dtype=dtype, device='cuda')
        rc = self.lib.test_glmvq_fp8(w.data_ptr(), s.data_ptr(), *raw.shape, int(dtype == torch.bfloat16), out.data_ptr(), torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
        torch.cuda.synchronize()
        return out


def exact(actual, expected):
    expected = torch.from_numpy(expected.copy()).to(actual.dtype)
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.cpu().view(bits), expected.view(bits)), 'restore bits differ'


def synthetic(kernels, dtypes=(torch.float32, torch.bfloat16)):
    rng = np.random.default_rng(5353)
    for k in [4096, 8192]:
        for precision in [1, 2]:
            for rows, cols in [(3, 64), (2048, 4096), (4096, 2048)]:
                idx = rng.integers(k, size=(rows, cols // 8), dtype=np.uint16)
                # Exercise low/high extrema and row-boundary plane addressing.
                idx[:, 0], idx[:, -1] = 0, k - 1
                low = idx.ravel() & 4095
                m = np.empty(low.size * 3 // 2, dtype='u1')
                a, b = low[::2], low[1::2]
                m[::3], m[1::3], m[2::3] = a & 255, (a >> 8) | ((b & 15) << 4), b >> 4
                e = np.packbits((idx.ravel() >> 12).astype('u1'), bitorder='little') if k == 8192 else None
                g = rng.uniform(0.01, 2, rows).astype('<f2')
                # Include half subnormals without relying on a shared conversion helper.
                g[0] = np.float16(2 ** -24)
                words = rng.normal(size=(k, 8)).astype('f4')
                if precision == 1:
                    book = torch.from_numpy(words).to(torch.float8_e4m3fn)
                    words = book.float().numpy()
                    book = book.view(torch.uint8).numpy()
                else:
                    book = words.astype('<f2')
                    words = book.astype('f4')
                expected = (words[idx] * g.astype('f4')[:, None, None]).reshape(rows, cols)
                for dtype in dtypes:
                    exact(kernels.restore(m, e, book, g, rows, cols, precision, dtype), expected)
                print(json.dumps({'synthetic': [rows, cols, k, precision],
                                  'outputs': list(map(str, dtypes)), 'result': 'bit-exact PASS'}), flush=True)


def fp8_scales(kernels):
    # The format permits positive finite F32 scales, including subnormals.
    raw = np.full((128, 384), 0x38, dtype='u1')  # E4M3 1.0.
    scales = np.array([[2 ** -140, 2 ** -133, 2 ** -126]], dtype='f4')
    expected = np.repeat(np.repeat(scales, 128, 0), 128, 1)
    for dtype in [torch.float32, torch.bfloat16]:
        exact(kernels.fp8(raw, scales, dtype), expected)
    print('GLMVQ_FP8_SUBNORMAL_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--case', choices=['all', 'scales', 'packing', 'fixtures', 'vq-bf16'], default='all')
    args = parser.parse_args()
    kernels = Kernels(args.lib)
    if args.case in ['all', 'fixtures', 'vq-bf16'] and args.fixtures is None:
        parser.error('--fixtures is required for the golden restore gate')
    if args.case == 'vq-bf16':
        synthetic(kernels, (torch.bfloat16,))
        manifest = json.loads((args.fixtures / 'manifest.json').read_text())
        for entry in manifest['fixtures']:
            folder = args.fixtures / entry['id']
            expected = np.load(folder / 'expected.npz')
            exact(kernels.restore(*fixture(folder / 'fixture.gguf'), torch.bfloat16), expected['vq_expected_f32'])
        print('GLMVQ_BF16_RESTORE_PASS', flush=True)
        return
    if args.case in ['all', 'scales']:
        fp8_scales(kernels)
    if args.case in ['all', 'packing']:
        synthetic(kernels)
    if args.case in ['scales', 'packing']:
        return
    manifest = json.loads((args.fixtures / 'manifest.json').read_text())
    assert len(manifest['fixtures']) == 10
    for entry in manifest['fixtures']:
        folder = args.fixtures / entry['id']
        expected = np.load(folder / 'expected.npz')
        components = fixture(folder / 'fixture.gguf')
        for dtype in [torch.float32, torch.bfloat16]:
            out = kernels.restore(*components, dtype)
            exact(out, expected['vq_expected_f32'])
            exact(kernels.fp8(expected['source_fp8'], expected['source_scale_f32'], dtype), expected['source_expected_f32'])
        x = torch.from_numpy(expected['linear_input_f32']).cuda().bfloat16()
        y = torch.nn.functional.linear(x, out).float().cpu().numpy()
        np.testing.assert_allclose(y, expected['linear_expected_bf16_as_f32'], **entry['linear_tolerance'])
        print(json.dumps({'fixture': entry['id'], 'vq_native_fp8': 'bit-exact PASS', 'bf16_linear': 'PASS'}), flush=True)
    print('GLMVQ_CUDA_RESTORE_PASS', flush=True)


if __name__ == '__main__':
    main()
