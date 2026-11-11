"""hVQ: shared-codebook VQ assignment with error feedback through the input second moment
(GPTQ / GPTVQ family), for the DeepSeek-V4.1 routed experts.

iVQ weighs column k of an expert matrix by a diagonal importance D (axk2_weighting). hVQ
also uses the correlation R between input channels: with

    H_t = D^1/2 ((1 - t) I + t R) D^1/2  (+ damp * mean(D) * I)

a row's output error is (w - q) H_t (w - q)^T. The 8-column blocks the format fixes are
quantized in order; each block's error moves the targets of the columns not yet quantized
(cuda/ds41_hvq.cu has the algebra). The codebook is not touched: only indices and row
gains change.

t = 0 has no feedback and is the iVQ conversion bit for bit: the same column-weighted
distance, the same least-squares gains, the same two passes (RMS gains, refit, again).

Statistics. R comes from per-expert sums of x x^T over the tokens routed to the expert,
each token divided by its mean square (as x2n / a2n: every routed token weighs the same).
A rare expert's sum is shrunk with a pseudo-count of `shrink` tokens toward the layer mean
(gate/up: experts share the hidden space) or toward the identity (down: an expert's own
channels), so R has full rank for any routed count.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from axk2_vq_gpu import GpuVq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'cuda/ds41_hvq.cu'
DIM = 8
NMET = DIM + DIM * (DIM - 1) // 2
ASSIGNMENT = ('8-column blocks in ascending order; block target w + r V_BB^-1 with r the feedback of the earlier '
              'blocks through V (H_t = V V^T, V upper triangular); nearest codeword under V_BB V_BB^T; '
              'ties to the lowest index')
GAIN_REFIT = 'ls-h: g = w H_t c^T / c H_t c^T per row (F16), after each of two passes; the first pass uses RMS gains'


def build():
    major, minor = torch.cuda.get_device_capability()
    target = ROOT / f'cuda/libds41_hvq_sm{major}{minor}.so'
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


def correlation(s, count, prior, shrink):
    """Shrunk correlation matrices, FP64.

    s [P, C, C] token-normalized sums, count [P] routed tokens, prior [C, C] (per token) or
    None for the identity scaled to the expert's own mean diagonal. Returns (R [P, C, C]
    with unit diagonal, the shrunk diagonal [P, C])."""
    s = s.double()
    n = count.double().view(-1, 1, 1)
    if prior is None:
        mean = s.diagonal(dim1=-2, dim2=-1).mean(-1) / n.view(-1).clamp_min(1.0)
        mean = torch.where(count.view(-1) > 0, mean, torch.ones_like(mean))
        cov = s / (n + shrink)
        cov.diagonal(dim1=-2, dim2=-1).add_((shrink * mean / (n.view(-1) + shrink))[:, None])
    else:
        cov = (s + shrink * prior.double()) / (n + shrink)
    cov = 0.5 * (cov + cov.mT)
    diag = cov.diagonal(dim1=-2, dim2=-1).clone()
    scale = torch.where(diag > 0, diag.clamp_min(1e-300).rsqrt(), torch.zeros_like(diag))
    r = cov * scale[:, :, None] * scale[:, None, :]
    r.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return r, diag


def participation(r):
    """C^2 / ||R||_F^2 of correlation matrices r [P, C, C]: C for uncorrelated channels, 1 for one direction."""
    c = r.shape[-1]
    return (c * c) / r.square().sum((-2, -1))


def hessian(col, r, t, damp):
    """H_t [P, C, C] FP64 from column weights col [P, C] (F32) and correlations r.
    The diagonal is col (+ damp * mean(col)) exactly; the off-diagonal part is
    t * sqrt(col_i col_j) r_ij, exact zeros at t = 0."""
    d = col.double()
    root = d.sqrt()
    h = r * (t * root[:, :, None]) * root[:, None, :]
    h.diagonal(dim1=-2, dim2=-1).copy_(d + damp * d.mean(-1, keepdim=True))
    return h


def factor(h):
    """V upper triangular with H = V V^T: the Cholesky factor of H reversed in both
    dimensions, reversed back. Returns (V, failures [P] int)."""
    low, info = torch.linalg.cholesky_ex(h.flip(-2, -1))
    return low.flip(-2, -1), info


def block_tables(v):
    """Per 8-column block of V [P, C, C] (FP64): the inverse of its diagonal block
    [P, C/8, 64] and the metric V_BB V_BB^T as 8 diagonal + 28 doubled upper entries
    [P, C/8, 36], both F32."""
    p, c, _ = v.shape
    nblk = c // DIM
    blocks = v.view(p, nblk, DIM, nblk, DIM).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()
    eye = torch.eye(DIM, dtype=v.dtype, device=v.device).expand(p, nblk, DIM, DIM)
    inv = torch.linalg.solve_triangular(blocks, eye, upper=True)
    metric = blocks @ blocks.mT
    upper = torch.triu_indices(DIM, DIM, offset=1, device=v.device)
    met = torch.cat([metric.diagonal(dim1=-2, dim2=-1), 2.0 * metric[:, :, upper[0], upper[1]]], -1)
    return inv.reshape(p, nblk, DIM * DIM).float().contiguous(), met.float().contiguous()


class Hvq:
    def __init__(self):
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError('hVQ needs FP32 GEMMs: set torch.backends.cuda.matmul.allow_tf32 = False')
        self.vq = GpuVq()
        self.lib = ct.CDLL(str(build()))
        ptr, integer = ct.c_void_p, ct.c_int
        self.lib.hvq_panel.argtypes = [ptr] * 9 + [integer] * 6 + [ptr]
        self.lib.hvq_refit.argtypes = [ptr] * 7 + [integer] * 3 + [ptr]
        self.panel = int(self.lib.hvq_max_panel())

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA hVQ error {code}')

    # ---------------------------------------------------------------- one pass
    def run_pass(self, w, gains, v, vinv, met, cb, panel=None):
        """w [P, rows, cols] F32, gains [P, rows] F16, v [P, cols, cols] F32 (upper),
        tables from block_tables, cb [K, 8] F32 -> (ids [P, rows, cols/8] int32,
        residual w - gain * codeword [P, rows, cols] F32)."""
        p, rows, cols = w.shape
        nblk = cols // DIM
        panel = panel or self.panel
        for x, dt in ((w, torch.float32), (v, torch.float32), (vinv, torch.float32), (met, torch.float32),
                      (cb, torch.float32), (gains, torch.float16)):
            assert x.is_cuda and x.is_contiguous() and x.dtype == dt
        assert v.shape == (p, cols, cols) and vinv.shape == (p, nblk, DIM * DIM) and met.shape == (p, nblk, NMET)
        assert gains.shape == (p, rows) and cb.shape[1] == DIM and 0 < panel <= self.panel
        ids = torch.empty((p, rows, nblk), dtype=torch.int32, device=w.device)
        res = torch.empty_like(w)
        for j0 in range(0, nblk, panel):
            j1 = min(nblk, j0 + panel)
            c0, c1 = j0 * DIM, j1 * DIM
            if c0:
                # feedback of every earlier panel into this one
                rpan = torch.bmm(res[:, :, :c0], v[:, :c0, c0:c1]).contiguous()
            else:
                rpan = torch.zeros((p, rows, c1), dtype=torch.float32, device=w.device)
            self.check(self.lib.hvq_panel(w.data_ptr(), rpan.data_ptr(), res.data_ptr(), gains.data_ptr(),
                                          v.data_ptr(), vinv.data_ptr(), met.data_ptr(), cb.data_ptr(),
                                          ids.data_ptr(), p, rows, cols, len(cb), j0, j1, self.stream()))
        return ids, res

    # -------------------------------------------------------------- gain refit
    def refit(self, w, ids, cb, h, gains, chunk=4):
        """In place: least-squares F16 gains under H for fixed indices. h [P, cols, cols] FP64."""
        p, rows, cols = w.shape
        assert gains.dtype == torch.float16 and gains.is_contiguous() and ids.is_contiguous()
        hdiag = h.diagonal(dim1=-2, dim2=-1).float().contiguous()
        noff = torch.empty((p, rows), dtype=torch.float64, device=w.device)
        doff = torch.empty_like(noff)
        for s in range(0, p, chunk):
            code = cb[ids[s:s + chunk].long()].reshape(-1, rows, cols).double()
            off = h[s:s + chunk].clone()
            off.diagonal(dim1=-2, dim2=-1).zero_()
            y = torch.bmm(code, off)
            noff[s:s + chunk] = (w[s:s + chunk].double() * y).sum(-1)
            doff[s:s + chunk] = (code * y).sum(-1)
            del code, off, y
        self.check(self.lib.hvq_refit(w.data_ptr(), ids.data_ptr(), cb.data_ptr(), hdiag.data_ptr(),
                                      noff.data_ptr(), doff.data_ptr(), gains.data_ptr(), p, rows, cols,
                                      self.stream()))
        return gains

    # ------------------------------------------------------------- conversion
    def rms_gains(self, w):
        """F16 row RMS gains of w [P, rows, cols] (the iVQ / YoungAi start)."""
        p, rows, cols = w.shape
        _, gains = self.vq.normalize(w.view(p * rows, cols))
        return gains.view(p, rows).contiguous()

    def quantize(self, w, cb, h, v, passes=2, info=None):
        """Two passes as iVQ: RMS gains, assign, refit, assign, refit.
        w [P, rows, cols] F32, h [P, cols, cols] FP64 and its factor v (FP64).
        Returns (ids int32 [P, rows, cols/8], gains F16 [P, rows]); a dict passed as `info`
        receives the last pass's target drift."""
        vinv, met = block_tables(v)
        v32 = v.float().contiguous()
        gains = self.rms_gains(w)
        ids = None
        for i in range(passes):
            ids, res = self.run_pass(w, gains, v32, vinv, met, cb)
            if info is not None and i == passes - 1:
                info['target_drift'] = self.target_drift(w, res, v32, vinv)
            del res
            self.refit(w, ids, cb, h, gains)
        return ids, gains

    @staticmethod
    def target_drift(w, res, v, vinv, tail=0.25):
        """How far the feedback moves the targets from the weights the codebook was fitted to: the squared norm
        of the corrected targets w~ over the last `tail` of the block order, relative to the weights' there.
        With e = (w - q) V, block B's target is q_B + e_B V_BB^-1."""
        p, rows, cols = w.shape
        nblk = cols // DIM
        b0 = int(nblk * (1 - tail))
        c0 = b0 * DIM
        e = torch.bmm(res, v[:, :, c0:].contiguous()).view(p, rows, nblk - b0, DIM)
        back = torch.einsum('prbi,pbij->prbj', e, vinv.view(p, nblk, DIM, DIM)[:, b0:]).reshape(p, rows, cols - c0)
        target = w[:, :, c0:] - res[:, :, c0:] + back
        return float(target.double().square().sum() / w[:, :, c0:].double().square().sum())

    def restore(self, ids, gains, cb):
        """W^ [P, rows, cols] F32 = f32(gain) * codeword, as the runtime decodes."""
        p, rows, nblk = ids.shape
        return cb[ids.long()].reshape(p, rows, nblk * DIM) * gains.float()[:, :, None]


def output_error(diff, h, chunk=4):
    """sum over rows of d H d^T for d = diff [P, rows, cols] (F32), H [P, cols, cols] FP64 -> [P] FP64."""
    out = torch.empty(diff.shape[0], dtype=torch.float64, device=diff.device)
    for s in range(0, diff.shape[0], chunk):
        d = diff[s:s + chunk].double()
        out[s:s + chunk] = (torch.bmm(d, h[s:s + chunk]) * d).sum((-2, -1))
    return out
