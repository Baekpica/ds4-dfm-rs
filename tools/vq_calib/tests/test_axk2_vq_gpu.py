#!/usr/bin/env python3
"""Synthetic tests for the A.X-K2 weighted Lloyd kernels (run on one GPU).

1. weights == 1 reproduces the unweighted path bit for bit (assignment, Lloyd books);
2. weighted Lloyd lowers the weighted distortion against the unweighted Lloyd;
3. the unweighted assignment equals Beam's kernel at K = 4096 (same arithmetic);
4. device v2 packing == an independent numpy packer for 9 and 10 bits; restore == numpy decode;
5. gain refit never increases the weighted row error for fixed indices;
6. assignment throughput at K = 512 / 1024 (vectors per second).
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axk2_vq_gpu import GpuVq  # noqa: E402

BEAM_CU = None    # Beam-1.0's beam_vq_convert.cu for the K = 4096 parity check (--beam-cu)
WORK = Path('/workspace/axk2/work/lloyd-test')


def mixture(n, k_true=600, seed=0, device='cuda'):
    g = torch.Generator(device='cpu').manual_seed(seed)
    centers = torch.randn(k_true, 8, generator=g) * 1.0
    scales = torch.rand(k_true, 8, generator=g) * 0.5 + 0.1
    which = torch.randint(0, k_true, (n,), generator=g)
    x = centers[which] + torch.randn(n, 8, generator=g) * scales[which]
    return x.float().contiguous().to(device)


def numpy_pack(ids, bits):
    ids = ids.astype(np.uint32)
    low = (ids & 255).astype(np.uint8)
    if bits == 9:
        plane = np.packbits(((ids >> 8) & 1).astype(np.uint8), bitorder='little')
    elif bits == 10:
        two = ((ids >> 8) & 3).astype(np.uint8).reshape(-1, 4)
        plane = (two[:, 0] | (two[:, 1] << 2) | (two[:, 2] << 4) | (two[:, 3] << 6)).astype(np.uint8)
    else:
        plane = None
    return low, plane


def numpy_unpack(low, plane, bits, n):
    ix = low.astype(np.uint32)
    if bits == 9:
        ix |= np.unpackbits(plane, bitorder='little')[:n].astype(np.uint32) << 8
    elif bits == 10:
        bitsarr = np.unpackbits(plane, bitorder='little').reshape(-1, 4, 2)
        two = (bitsarr[:, :, 0] | (bitsarr[:, :, 1] << 1)).reshape(-1)[:n].astype(np.uint32)
        ix |= two << 8
    return ix


def beam_assign(x, cb):
    lib_path = WORK / 'libbeam_ref.so'
    if not lib_path.exists():
        major, minor = torch.cuda.get_device_capability()
        subprocess.run(['nvcc', '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC', f'-arch=sm_{major}{minor}',
                        str(BEAM_CU), '-o', str(lib_path)], check=True)
    lib = ct.CDLL(str(lib_path))
    p = ct.c_void_p
    lib.bvq_assign.argtypes = [p, p, p, p, ct.c_int64, ct.c_int, p]
    ids = torch.empty(len(x), dtype=torch.int32, device=x.device)
    rc = lib.bvq_assign(x.data_ptr(), cb.data_ptr(), ids.data_ptr(), None, len(x), len(cb),
                        torch.cuda.current_stream().cuda_stream)
    assert rc == 0
    torch.cuda.synchronize()
    return ids


def main():
    global WORK, BEAM_CU
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--beam-cu', type=Path, default=BEAM_CU,
                   help='Beam-1.0 beam_vq_convert.cu for the K=4096 parity check (skipped when not given)')
    a = p.parse_args()
    WORK, BEAM_CU = a.work, a.beam_cu
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    gpu = GpuVq(vpt=2)
    x = mixture(400_000)
    k = 512
    # Heteroscedastic column weights per vector (like per-column imatrix), plus row factors.
    g = torch.Generator(device='cpu').manual_seed(1)
    colw = (torch.rand(8, generator=g) * 4 + 0.05).float()
    colw = colw[torch.randint(0, 8, (8,), generator=g)]
    aw = (colw[None, :] * (0.5 + torch.rand(len(x), 1, generator=g))).float().contiguous().cuda()
    rf = (torch.rand(len(x), 1, generator=g) ** 3 * 10 + 0.01).float().cuda()
    uw = (aw * rf).contiguous()

    # 1. weights == 1 vs unweighted: assignment and whole Lloyd runs.
    ones = torch.ones_like(x)
    cb0 = x[(torch.arange(k) * len(x) // k).cuda()].clone()
    a0 = gpu.assign(x, cb0)
    a1 = gpu.assign(x, cb0, weights=ones)
    a2 = gpu.assign(x, cb0, colw=torch.ones(8, device='cuda'))
    e8u, f16u, f32u, hu = gpu.train(x, k, 12)
    e8o, f16o, f32o, ho = gpu.train(x, k, 12, weights=ones, assign_weights=ones)
    out['ones_bit_identical'] = {
        'assign_per_vector': bool(torch.equal(a0, a1)), 'assign_per_column': bool(torch.equal(a0, a2)),
        'lloyd_f32_book': bool(torch.equal(f32u, f32o)), 'lloyd_e4m3_book': bool(torch.equal(e8u.view(torch.uint8), e8o.view(torch.uint8)))}

    # 2. weighted Lloyd lowers the weighted distortion it optimizes.
    e8w, f16w, f32w, hw = gpu.train(x, k, 24, weights=uw, assign_weights=aw)
    e8n, f16n, f32n, hn = gpu.train(x, k, 24)
    def wdist(book):
        ids = gpu.assign(x, book, weights=aw)
        return gpu.distortion(x, book, ids, uw)
    def udist(book):
        ids = gpu.assign(x, book)
        return gpu.distortion(x, book, ids)
    out['weighted_vs_unweighted'] = {
        'weighted_distortion_weighted_book': wdist(f32w), 'weighted_distortion_unweighted_book': wdist(f32n),
        'unweighted_distortion_weighted_book': udist(f32w), 'unweighted_distortion_unweighted_book': udist(f32n),
        'monotone_weighted_history': all(b['weighted_distortion'] <= a['weighted_distortion'] * (1 + 1e-9)
                                         for a, b in zip(hw, hw[1:]))}
    out['weighted_vs_unweighted']['gain'] = 1 - out['weighted_vs_unweighted']['weighted_distortion_weighted_book'] / \
        out['weighted_vs_unweighted']['weighted_distortion_unweighted_book']

    # 3. Beam kernel parity at K=4096 (unweighted arithmetic is the same).
    cb4 = x[(torch.arange(4096) * len(x) // 4096).cuda()].clone()
    out['beam_parity_k4096'] = (bool(torch.equal(beam_assign(x, cb4), gpu.assign(x, cb4)))
                                if BEAM_CU is not None and BEAM_CU.exists() else 'skipped: no --beam-cu')

    # 4. packing / restore parity for 9 and 10 bits on a matrix shape [rows, cols].
    for bits, rows, cols in ((9, 64, 7168), (10, 64, 7168), (9, 448, 2048), (10, 448, 2048)):
        kk = 1 << bits
        ids = torch.randint(0, kk, (rows * cols // 8,), dtype=torch.int32, device='cuda')
        low, plane = gpu.pack(ids, rows, cols, bits)
        nl, npl = numpy_pack(ids.cpu().numpy(), bits)
        ok_pack = np.array_equal(low.cpu().numpy(), nl) and np.array_equal(plane.cpu().numpy(), npl)
        ok_unpack = np.array_equal(numpy_unpack(nl, npl, bits, ids.numel()), ids.cpu().numpy().astype(np.uint32))
        book = torch.randn(kk, 8, device='cuda').to(torch.float16).to(torch.float8_e4m3fn).float().contiguous()
        gain = (torch.rand(rows, device='cuda') * 0.01).to(torch.float16)
        rest = gpu.restore(low, plane, gain, book, cols, bits).cpu().numpy()
        ref = (book.cpu().numpy()[numpy_unpack(nl, npl, bits, ids.numel())].reshape(rows, cols)
               * gain.float().cpu().numpy()[:, None]).astype(np.float32)
        rest_ids = gpu.restore_ids(ids, gain, book, cols).cpu().numpy()
        out[f'pack_{bits}bit_{rows}x{cols}'] = {'device_eq_numpy': bool(ok_pack), 'numpy_roundtrip': bool(ok_unpack),
                                                'restore_eq_numpy': bool(np.array_equal(rest, ref)),
                                                'restore_ids_eq': bool(np.array_equal(rest_ids, ref))}

    # 5. gain refit on a real-ish matrix: weighted row error must not increase.
    rows, cols = 256, 7168
    w = (torch.randn(rows, cols, device='cuda') * (torch.rand(rows, 1, device='cuda') + 0.1) * 0.01).contiguous()
    hcol = (torch.rand(cols, device='cuda') ** 4 * 50 + 0.01).contiguous()
    norm, gain = gpu.normalize(w)
    book = f32w.contiguous()
    ids = gpu.assign(norm, book, colw=hcol)
    def row_err(gn):
        rec = gpu.restore_ids(ids, gn, book, cols)
        return ((rec.double() - w.double()) ** 2 * hcol.double()).sum(1)
    before = row_err(gain)
    refit = gpu.refit_gain(w, ids, book, gain.clone(), colw=hcol)
    after = row_err(refit)
    out['gain_refit'] = {'weighted_error_before': float(before.sum()), 'weighted_error_after': float(after.sum()),
                         'rows_worse': int((after > before * (1 + 1e-3)).sum())}

    # 6. throughput.
    big = mixture(896 * 8960, seed=3)
    thr = {}
    for kk in (512, 1024):
        book = big[(torch.arange(kk) * len(big) // kk).cuda()].clone()
        hc = torch.rand(7168, device='cuda').contiguous()
        for vpt in (1, 2, 4):
            gpu.vpt = vpt
            for mode in ('none', 'colw'):
                kw = {} if mode == 'none' else {'colw': hc}
                gpu.assign(big, book, **kw)
                torch.cuda.synchronize()
                t = time.monotonic()
                for _ in range(3):
                    gpu.assign(big, book, **kw)
                torch.cuda.synchronize()
                thr[f'k{kk}_vpt{vpt}_{mode}'] = 3 * len(big) / (time.monotonic() - t)
    gpu.vpt = 2
    pool = mixture(16_500_000, seed=4)
    pw = torch.rand_like(pool)
    t = time.monotonic()
    gpu.train(pool, 512, 4, weights=pw, assign_weights=pw)
    torch.cuda.synchronize()
    thr['lloyd_iteration_seconds_16.5M_k512_weighted'] = (time.monotonic() - t) / 4
    out['throughput_vectors_per_s'] = thr
    print(json.dumps(out, indent=1))
    (WORK / 'synthetic-tests.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
