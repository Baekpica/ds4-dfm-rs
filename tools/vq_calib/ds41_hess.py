"""Per-expert input second moments of one DeepSeek-V4.1 MoE layer, for hVQ (ds41_hvq.py).

For expert e, over the calibration tokens routed to it:

    xx[e] = sum x x^T / mean(x^2)     gate/up input (the FFN-normalized hidden state), 5120^2
    aa[e] = sum a a^T / mean(a^2)     down input a = silu(gate) * up,                  2304^2

Every token is divided by its own mean square, as the collector's x2n / a2n, so a routed
token weighs the same however large its activations (from layer 10 on 1% of the tokens
carry 94-100% of the raw energy). The route weight is a per-token scalar and cancels.

A layer is 40 GiB + 8 GiB in FP32 and lives on the device only: collected, used by the
converter and dropped. Each chunk's Gram matrix is computed in FP64 and added to the FP32
sum (`probe` keeps FP64 and all-FP32 sums of a few experts to measure what that costs).
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NORMALIZATION = 'token: each routed token x x^T (a a^T) divided by its own mean square, summed per expert'
PRECISION = 'chunk Gram matrices in FP64 (<= 32768 tokens), per-expert sums in FP32'
MIN_MEAN_SQUARE = 1e-30
CHUNK = 32768


def normalized(x, dtype):
    """Rows of x scaled to unit mean square."""
    x = x.to(dtype)
    return x * x.square().mean(-1, keepdim=True).clamp_min(MIN_MEAN_SQUARE).rsqrt()


class LayerHess:
    def __init__(self, experts, dim, inter, device='cuda', probe=()):
        self.xx = torch.zeros((experts, dim, dim), dtype=torch.float32, device=device)
        self.aa = torch.zeros((experts, inter, inter), dtype=torch.float32, device=device)
        self.count = torch.zeros(experts, dtype=torch.int64, device=device)
        self.meta = {}
        self.mean = None
        z = lambda n: torch.zeros((n, n), dtype=torch.float64, device=device)
        z32 = lambda n: torch.zeros((n, n), dtype=torch.float32, device=device)
        self.probe = {e: {'xx64': z(dim), 'aa64': z(inter), 'xx32': z32(dim), 'aa32': z32(inter)} for e in probe}

    # ------------------------------------------------------------ collection
    def add(self, e, x, act):
        """x [n, dim] expert input, act [n, inter] down input, the n tokens routed to expert e."""
        n = x.shape[0]
        ref = self.probe.get(e)
        for c0 in range(0, n, CHUNK):
            for key, src in (('xx', x), ('aa', act)):
                u = normalized(src[c0:c0 + CHUNK], torch.float64)
                gram = u.T @ u
                getattr(self, key)[e] += gram.float()
                if ref is not None:
                    ref[key + '64'] += gram
                    u32 = normalized(src[c0:c0 + CHUNK], torch.float32)
                    ref[key + '32'] += u32.T @ u32
                del u, gram
        self.count[e] += n

    def probe_report(self):
        """Per probed expert: what FP32 sums of FP64 Gram matrices, and all-FP32 sums, lose against FP64."""
        rows = []
        for e, ref in self.probe.items():
            row = {'expert': e, 'tokens': int(self.count[e])}
            for key in ('xx', 'aa'):
                exact = ref[key + '64']
                scale = exact.diagonal().clamp_min(1e-300).rsqrt()
                corr = exact * scale[:, None] * scale[None, :]
                for name, got in (('gram64_sum32', getattr(self, key)[e].double()), ('gram32_sum32', ref[key + '32'].double())):
                    gs = got.diagonal().clamp_min(1e-300).rsqrt()
                    gc = got * gs[:, None] * gs[None, :]
                    row[f'{key}_{name}'] = {
                        'rel_frobenius': float((got - exact).norm() / exact.norm()),
                        'max_abs_correlation_error': float((gc - corr).abs().max()),
                        'max_rel_diagonal_error': float(((got.diagonal() - exact.diagonal()).abs() / exact.diagonal().clamp_min(1e-300)).max())}
            rows.append(row)
        return rows

    # ------------------------------------------------------------- converter
    def layer_mean(self):
        """Per-token mean of xx over the layer's experts, FP64 [dim, dim]: the prior of rare experts."""
        if self.mean is None:
            total = torch.zeros(self.xx.shape[1:], dtype=torch.float64, device=self.xx.device)
            for e in range(self.xx.shape[0]):
                total += self.xx[e]
            self.mean = total / self.count.sum().clamp_min(1).double()
        return self.mean

    # ---------------------------------------------------------------- record
    def digest(self, threads=4, group=8, stream=None):
        """SHA-256 over the per-expert SHA-256 of xx, then aa (F32 little endian), then count (int64).
        The sums leave the device through one pinned buffer per shape; `stream` keeps the copies off the
        stream other work runs on."""
        h = hashlib.sha256()
        with ThreadPoolExecutor(threads) as ex:
            for t in (self.xx, self.aa):
                if not t.shape[0]:
                    continue
                buf = torch.empty((min(group, t.shape[0]),) + tuple(t.shape[1:]), dtype=torch.float32, pin_memory=True)
                for s in range(0, t.shape[0], group):
                    part = t[s:s + group]
                    host = buf[:part.shape[0]]
                    if stream is None:
                        host.copy_(part)
                    else:
                        with torch.cuda.stream(stream):
                            host.copy_(part)
                    for d in ex.map(lambda a: hashlib.sha256(a).digest(), host.numpy()):
                        h.update(d)
                del buf
        h.update(self.count.cpu().numpy().tobytes())
        return h.hexdigest()

    def digest_async(self):
        """digest() on a host thread and a side stream, so a fit can read the sums meanwhile. -> Future."""
        torch.cuda.synchronize()
        return ThreadPoolExecutor(1).submit(self.digest, 4, 8, torch.cuda.Stream())

    def describe(self):
        c = self.count.double()
        return {'normalization': NORMALIZATION, 'precision': PRECISION, 'tokens_routed': int(c.sum()),
                'count_min': int(c.min()), 'count_median': float(c.median()), 'count_max': int(c.max()),
                'experts_below_dim': int((c < self.xx.shape[1]).sum()), 'experts_below_inter': int((c < self.aa.shape[1]).sum())}

    def save(self, path, meta=None):
        save_file({'xx': self.xx.cpu(), 'aa': self.aa.cpu(), 'count': self.count.cpu()}, str(path),
                  metadata={'meta': json.dumps({**self.meta, **(meta or {})})})

    @classmethod
    def load(cls, path, device='cuda'):
        with safe_open(str(path), 'pt') as f:
            meta = json.loads((f.metadata() or {}).get('meta', '{}'))
            count = f.get_tensor('count')
            xs, as_ = f.get_slice('xx'), f.get_slice('aa')
            e, dim, _ = xs.get_shape()
            inter = as_.get_shape()[1]
            self = cls(0, dim, inter, device)
            self.xx = torch.empty((e, dim, dim), dtype=torch.float32, device=device)
            self.aa = torch.empty((e, inter, inter), dtype=torch.float32, device=device)
            for s in range(0, e, 32):
                self.xx[s:s + 32] = xs[s:s + 32].to(device)
                self.aa[s:s + 32] = as_[s:s + 32].to(device)
        self.count = count.to(device)
        self.meta = meta
        return self
