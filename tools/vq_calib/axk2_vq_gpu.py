"""A.X-K2 VQ GPU primitives: weighted Lloyd, weighted nearest assignment,
FP32 normalization, least-squares gain refit, GVQ v2 pack and restore.

Forked from the Beam-1.0 converter (beam_vq_gpu.py). Differences: FP32
source rows (FP8 x block scale dequantized in FP32), any power-of-two K from
256 to 8192, importance weights in both Lloyd steps (per-vector update
weights; per-vector or per-column assignment weights), the byte-major
9/10-bit GVQ v2 layout, and an optional least-squares row-gain refit.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'cuda/axk2_vq_convert.cu'


def build():
    major, minor = torch.cuda.get_device_capability()
    target = ROOT / f'cuda/libaxk2_vq_convert_sm{major}{minor}.so'
    command = ['nvcc', '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC',
               f'-arch=sm_{major}{minor}', str(SOURCE), '-o', str(target)]
    stamp = hashlib.sha256(SOURCE.read_bytes() + json.dumps(command).encode()).hexdigest()
    stamp_path = target.with_suffix('.build.json')
    if not target.exists() or not stamp_path.exists() or json.loads(stamp_path.read_text())['stamp'] != stamp:
        tmp = target.with_suffix(f'.{time.time_ns()}.tmp.so')
        subprocess.run(command[:-1] + [str(tmp)], check=True)
        tmp.rename(target)
        stamp_path.write_text(json.dumps({'stamp': stamp, 'command': command}, indent=2) + '\n')
    return target


def bits_for(k):
    bits = int(k).bit_length() - 1
    if 1 << bits != k or not 8 <= bits <= 13:
        raise ValueError(f'K must be a power of two, got {k}')
    return bits


class GpuVq:
    def __init__(self, vpt=2):
        self.lib = ct.CDLL(str(build()))
        self.vpt = vpt
        ptr, i64, integer = ct.c_void_p, ct.c_int64, ct.c_int
        self.lib.axvq_assign.argtypes = [ptr, ptr, integer, integer, ptr, ptr, ptr, i64, integer, integer, ptr]
        self.lib.axvq_update.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i64, integer, ptr]
        self.lib.axvq_normalize_f32.argtypes = [ptr, ptr, ptr, integer, integer, ptr, ptr]
        self.lib.axvq_normalize_gain.argtypes = [ptr, ptr, ptr, i64, integer, ptr]
        self.lib.axvq_refit_gain.argtypes = [ptr, ptr, ptr, ptr, ptr, integer, integer, ptr]
        self.lib.axvq_restore.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, integer, integer, ptr]
        self.lib.axvq_restore_ids.argtypes = [ptr, ptr, ptr, ptr, i64, integer, ptr]
        self.lib.axvq_pack.argtypes = [ptr, ptr, ptr, i64, integer, ptr]

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA converter error {code}')

    @staticmethod
    def _ptr(t):
        return t.data_ptr() if t is not None else None

    # ------------------------------------------------------------ assignment
    def assign(self, x, cb, weights=None, colw=None, distances=False):
        """x [N, 8] F32. weights: per-vector [N, 8] F32, or colw: per-column
        [nvec * 8] F32 for a row-major matrix whose rows hold nvec vectors
        (vector i uses columns of vector slot i % nvec). Returns int32 ids."""
        assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32 and x.shape[1] == 8
        assert cb.is_cuda and cb.is_contiguous() and cb.dtype == torch.float32 and cb.shape[1] == 8
        mode, w, nvec = 0, None, 0
        if weights is not None:
            assert weights.shape == x.shape and weights.dtype == torch.float32 and weights.is_contiguous()
            mode, w = 1, weights
        elif colw is not None:
            assert colw.dtype == torch.float32 and colw.is_contiguous() and colw.numel() % 8 == 0
            mode, w, nvec = 2, colw, colw.numel() // 8
            assert len(x) % nvec == 0
        ids = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
        dist = torch.empty(x.shape[0], dtype=torch.float32, device=x.device) if distances else None
        self.check(self.lib.axvq_assign(x.data_ptr(), self._ptr(w), mode, nvec, cb.data_ptr(), ids.data_ptr(),
                                        self._ptr(dist), len(x), len(cb), self.vpt, self.stream()))
        return (ids, dist) if distances else ids

    # ----------------------------------------------------------------- Lloyd
    def train(self, pool, k, iterations, weights=None, assign_weights=None, log=None):
        """Equidistant init (pool vector floor(i*N/K)), weighted FP64 Lloyd,
        then F32 -> F16 -> E4M3FN. `weights` [N, 8] are the update (centroid)
        weights, `assign_weights` [N, 8] the assignment metric; either may be
        None (= 1). History rows carry the update-weighted distortion of the
        assignment made in that iteration."""
        assert len(pool) >= k
        positions = (torch.arange(k, dtype=torch.int64) * len(pool)) // k
        cb = pool[positions.to(pool.device)].clone()
        sums = torch.empty((k, 8), dtype=torch.float64, device=pool.device)
        dens = torch.empty((k, 8), dtype=torch.float64, device=pool.device)
        history = []
        for iteration in range(iterations):
            start = time.monotonic()
            ids = self.assign(pool, cb, weights=assign_weights)
            distortion = self.distortion(pool, cb, ids, weights)
            self.check(self.lib.axvq_update(pool.data_ptr(), self._ptr(weights), ids.data_ptr(), cb.data_ptr(),
                                            sums.data_ptr(), dens.data_ptr(), len(pool), k, self.stream()))
            torch.cuda.synchronize()
            row = {'iteration': iteration + 1, 'seconds': time.monotonic() - start,
                   'weighted_distortion': distortion,
                   'empty_centroids': int((dens.sum(1) == 0).sum())}
            history.append(row)
            if log:
                log(row)
        f16 = cb.to(torch.float16)
        e4m3 = f16.to(torch.float8_e4m3fn)
        if not torch.isfinite(f16.float()).all() or not torch.isfinite(e4m3.float()).all():
            raise ValueError('nonfinite codebook')
        return e4m3, f16, cb, history

    @staticmethod
    def distortion(x, cb, ids, weights=None, chunk=1 << 22):
        total = 0.0
        for s in range(0, len(x), chunk):
            d = (x[s:s + chunk] - cb[ids[s:s + chunk].long()]).double().pow(2)
            if weights is not None:
                d = d * weights[s:s + chunk].double()
            total += float(d.sum())
        return total

    # ------------------------------------------------------- normalization
    def normalize(self, w):
        """w: [rows, cols] F32 on the device -> (normalized [rows*cols/8, 8] F32, F16 gains)."""
        assert w.ndim == 2 and w.dtype == torch.float32 and w.is_contiguous() and w.is_cuda
        x = torch.empty(w.shape, dtype=torch.float32, device=w.device)
        gain = torch.empty(w.shape[0], dtype=torch.float16, device=w.device)
        error = torch.zeros(1, dtype=torch.int32, device=w.device)
        self.check(self.lib.axvq_normalize_f32(w.data_ptr(), x.data_ptr(), gain.data_ptr(),
                                               *w.shape, error.data_ptr(), self.stream()))
        if error.item():
            raise ValueError('nonfinite source or gain overflow')
        return x.reshape(-1, 8), gain

    def normalize_gain(self, w, gain):
        x = torch.empty(w.shape, dtype=torch.float32, device=w.device)
        self.check(self.lib.axvq_normalize_gain(w.data_ptr(), gain.data_ptr(), x.data_ptr(), w.numel(),
                                                w.shape[1], self.stream()))
        return x.reshape(-1, 8)

    def refit_gain(self, w, ids, cb, gain, colw=None):
        """In place: least-squares F16 gain per row for fixed indices (weighted by colw [cols])."""
        assert gain.dtype == torch.float16 and gain.is_contiguous()
        if colw is not None:
            assert colw.dtype == torch.float32 and colw.numel() == w.shape[1] and colw.is_contiguous()
        self.check(self.lib.axvq_refit_gain(w.data_ptr(), ids.data_ptr(), cb.data_ptr(), self._ptr(colw),
                                            gain.data_ptr(), *w.shape, self.stream()))
        return gain

    # ------------------------------------------------------ packing/restore
    def pack(self, ids, rows, cols, bits):
        assert ids.is_cuda and ids.dtype == torch.int32 and ids.numel() == rows * cols // 8
        n = rows * cols // 8
        low = torch.empty(n, dtype=torch.uint8, device=ids.device)
        plane = torch.empty(n * (bits - 8) // 8, dtype=torch.uint8, device=ids.device) if bits > 8 else None
        self.check(self.lib.axvq_pack(ids.data_ptr(), low.data_ptr(), self._ptr(plane), n, bits, self.stream()))
        return low, plane

    def restore(self, low, plane, gain, cb, cols, bits):
        out = torch.empty((len(gain), cols), dtype=torch.float32, device=gain.device)
        self.check(self.lib.axvq_restore(low.data_ptr(), self._ptr(plane), gain.data_ptr(), cb.data_ptr(),
                                         out.data_ptr(), out.numel(), cols, bits, self.stream()))
        return out

    def restore_ids(self, ids, gain, cb, cols):
        out = torch.empty((len(gain), cols), dtype=torch.float32, device=gain.device)
        self.check(self.lib.axvq_restore_ids(ids.data_ptr(), gain.data_ptr(), cb.data_ptr(), out.data_ptr(),
                                             out.numel(), cols, self.stream()))
        return out
