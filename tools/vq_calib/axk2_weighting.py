"""Importance weights for A.X-K2's shared per-layer codebook (DESIGN.md
"Weighting modes"), from the calibration statistics of one MoE layer.

Vector i = (expert e, projection p, row r, column block j), normalized
v = w / g_r. The pool weight of dimension d (column k = 8j + d) is

    omega[r, k] = row_factor[e, p, r] * tensor_scale[e, p] * col[e, p, k]

with `col` normalized to mean 1 per tensor. Assignment uses `col` only (the
row factor and tensor scale cancel in the argmin); the gain refit uses `col`.

| mode         | row_factor                                  | tensor_scale            | col            |
|--------------|---------------------------------------------|-------------------------|----------------|
| none         | 1                                           | 1                       | 1              |
| row          | g_r^2                                       | 1                       | 1              |
| imatrix      | g_r^2                                       | 1                       | x2 or a2       |
| imatrix_freq | g_r^2                                       | mean(x2w or a2w)        | x2w or a2w     |
| output       | down: g_r^2; gate/up: g_r^2 ||D[:,r]||^2    | mean(x2w or a2w)        | x2w or a2w     |
|              | (bg or bu)[r] / w2                          |                         |                |

(row, imatrix and imatrix_freq give each projection type the same layer total
in the converter's pool, as the unweighted pool has equal counts; output mode
is in layer-output units already.)

Rare experts. Per-expert profiles are shrunk toward a prior with a
pseudo-count of `shrink_tokens` routed tokens (lambda; in w^2 units for the
route-weighted sums, lambda * sum(w2) / sum(count)):

  * gate/up columns live in the shared hidden space, so the prior is the
    layer mean per token: col_e = (S_e + lambda m) / (n_e + lambda),
    m = sum_e S_e / sum_e n_e;
  * down columns and gate/up row sensitivities index an expert's own
    intermediate channels (not aligned across experts), so their prior is
    the expert's own flat profile (its channel mean).

The tensor scale (the expert's routed frequency) is not shrunk: rare experts
weigh little in the shared book; it is only floored at 1e-3 x the median so a
never-routed expert keeps a trace of weight.

`col` may further be tempered (h ** col_power), shrunk toward 1 (col_shrink)
and is floored at col_floor x its mean: a VQ dimension with ~zero weight can
take any codeword value, so an unfloored profile lets low-importance columns
err by more than their own magnitude (seen on stand-in statistics: down rel
MSE 1.1).
"""
from pathlib import Path

import torch
from safetensors.torch import load_file

MODES = ('none', 'row', 'imatrix', 'imatrix_freq', 'output')
PROJ = ('gate_proj', 'up_proj', 'down_proj')


def load_stats(stats_dir, layer):
    """stats/layer-LL.safetensors, or the sum of stats/rank-*/layer-LL.safetensors."""
    stats_dir = Path(stats_dir)
    merged = stats_dir / f'layer-{layer:02d}.safetensors'
    if merged.exists():
        return load_file(str(merged)), [str(merged)]
    ranks = sorted(stats_dir.glob(f'rank-*/layer-{layer:02d}.safetensors'))
    if not ranks:
        raise FileNotFoundError(f'no statistics for layer {layer} under {stats_dir}')
    total = None
    for path in ranks:
        t = load_file(str(path))
        if total is None:
            total = {k: v.clone() for k, v in t.items()}
        else:
            for k, v in t.items():
                total[k] += v
    return total, [str(p) for p in ranks]


class Weighting:
    def __init__(self, mode, stats=None, device='cuda', shrink_tokens=256.0, col_floor=0.02, col_shrink=0.0,
                 col_power=1.0):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode}')
        if mode not in ('none', 'row') and stats is None:
            raise ValueError(f'mode {mode} needs statistics')
        self.mode, self.device = mode, device
        self.shrink_tokens, self.col_floor, self.col_shrink, self.col_power = shrink_tokens, col_floor, col_shrink, col_power
        self.s = {k: v.to(device) for k, v in stats.items()} if stats is not None else None
        self.cols, self.scale, self.bmed, self.b, self.lam_units = {}, None, {}, {}, None
        if self.s is None or mode in ('none', 'row'):
            return
        s = self.s
        count = s['count'].double()
        weighted = mode != 'imatrix'
        n = s['w2'].double() if weighted else count                         # [E]
        lam = float(shrink_tokens)
        if weighted:
            lam = lam * float(s['w2'].double().sum() / count.sum().clamp_min(1))
        self.lam_units = lam
        # gate/up: shared hidden space -> layer-mean prior per unit of n.
        S = (s['x2w'] if weighted else s['x2']).double()                    # [E, H]
        m = S.sum(0) / n.sum().clamp_min(1e-30)                             # [H]
        hx = (S + lam * m[None, :]) / (n[:, None] + lam)
        # down: the expert's own channels -> flat prior at the expert's per-unit mean.
        A = (s['a2w'] if weighted else s['a2']).double()                    # [E, F]
        flat = A.mean(1) / n.clamp_min(1e-30)                               # [E]
        ha = (A + lam * flat[:, None]) / (n[:, None] + lam)
        ha[n == 0] = 1.0
        self.cols = {'gate_proj': hx, 'up_proj': hx, 'down_proj': ha}
        if mode in ('imatrix_freq', 'output'):
            self.scale = {}
            for p in PROJ:
                raw = (s['a2w'] if p == 'down_proj' else s['x2w']).double().mean(1)     # [E], unshrunk
                live = raw[count > 0]
                med = float(live.median()) if len(live) else 1.0
                self.scale[p] = torch.clamp(raw, min=1e-3 * med)
        if mode == 'output':
            w2 = s['w2'].double()
            for p, key in (('gate_proj', 'bg'), ('up_proj', 'bu')):
                B = s[key].double()                                         # [E, F] sums of w^2 s^2
                per = B.mean(1) / w2.clamp_min(1e-30)                       # expert's flat per-unit mean
                live = per[count > 0]
                self.bmed[p] = float(live.median()) if len(live) else 1.0
                shr = (B + lam * per[:, None]) / (w2[:, None] + lam)       # per unit of w^2
                shr[w2 == 0] = self.bmed[p]
                self.b[p] = shr

    @property
    def weighted_assignment(self):
        return self.mode not in ('none', 'row')

    def col(self, e, p):
        """[cols] F32 column profile (mean 1), or None for uniform."""
        if not self.weighted_assignment:
            return None
        h = self.cols[p][e]
        mean = h.mean()
        if not torch.isfinite(mean) or mean <= 0:
            return None
        h = h / mean
        if self.col_power != 1.0:
            h = h ** self.col_power
            h = h / h.mean()
        if self.col_shrink:
            h = (1 - self.col_shrink) * h + self.col_shrink
        h = torch.clamp(h, min=self.col_floor)
        return (h / h.mean()).float().contiguous()

    def tensor_scale(self, e, p):
        if self.scale is None:
            return 1.0
        return float(self.scale[p][e])

    def row_factor(self, e, p, gains, rows, dcol2=None):
        """gains: F16 [n] gains of the sampled rows `rows` (LongTensor) of tensor (e, p);
        dcol2: [F] squared column norms of expert e's down_proj (output mode, gate/up)."""
        g2 = gains.double() ** 2
        if self.mode == 'none':
            return torch.ones_like(g2)
        if self.mode != 'output' or p == 'down_proj':
            return g2
        # Gate/up errors reach the layer output through ||D[:,r]||^2 * E[w^2 s_r^2] / E[w^2]
        # (s = silu'(g) u for gate, silu(g) for up): the same output units as down's g_r^2 a2w.
        return g2 * dcol2[rows].double() * self.b[p][e][rows]

    def describe(self):
        return {'mode': self.mode, 'shrink_tokens': self.shrink_tokens, 'shrink_units': self.lam_units,
                'col_floor': self.col_floor, 'col_shrink': self.col_shrink, 'col_power': self.col_power,
                'tensor_scale': 'unshrunk mean, floored at 1e-3 x median' if self.scale is not None else None}
