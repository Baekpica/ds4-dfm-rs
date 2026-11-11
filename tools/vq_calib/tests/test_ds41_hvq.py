#!/usr/bin/env python3
"""hVQ (ds41_hvq.py, ds41_hess.py, cuda/ds41_hvq.cu) on synthetic problems; one GPU, no model files.

    python3 tests/test_ds41_hvq.py

 1. t = 0 is the iVQ assignment and gain refit bit for bit (shapes, K, column profiles, zero rows).
 2. A pass follows GPTQ: an FP64 NumPy reference written with the Cholesky factor of the inverse
    (the textbook form) is driven by the kernel's own choices; every choice is the reference's
    nearest codeword, and the residuals and the error agree.
 3. On correlated inputs the feedback lowers the output error ||(W - Q) X^T||^2, on the calibration
    sample and on fresh inputs; on uncorrelated inputs it changes nothing to speak of.
 4. The gain refit is the minimizer of the H error for fixed indices (FP64 formula, F16 rounding).
 5. Fewer tokens than columns: the shrunk correlation factors; without shrink and damping it needs a lift.
 6. The panel size does not matter: the lazy feedback is the same sum.
 7. Second moments: LayerHess sums are the FP64 sums, token-normalized; save / load; digest.
 8. DQVL v3 pack -> strict decode round trip of hVQ indices and gains (12- and 13-bit).
"""
import os
from pathlib import Path
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False

import convert_layer_ds41_hvq as cv  # noqa: E402
import ds41_hvq as hv  # noqa: E402
import ds41_vq_format as fmt  # noqa: E402
from ds41_hess import LayerHess  # noqa: E402

DEV = 'cuda'
FAILS = []


def check(cond, msg):
    print(('  PASS ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def codebook(k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, 8, generator=g).to(torch.float8_e4m3fn).float().contiguous().to(DEV)


def profile(p, cols, seed):
    """Column weights with mean 1 and a floor, like axk2_weighting's."""
    g = torch.Generator().manual_seed(seed)
    col = torch.rand(p, cols, generator=g) ** 3 + 0.02
    return (col / col.mean(-1, keepdim=True)).float().contiguous().to(DEV)


def correlated(n, cols, rho, seed, rank=0, law=7):
    """n tokens (sample `seed`) of one input law: AR(1) channel correlation rho, plus `rank`
    shared factors, times per-channel scales; `law` seeds the loadings and scales. FP64."""
    g, fixed = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(law)
    z = torch.randn(n, cols, generator=g, dtype=torch.float64)
    x = z.clone()
    for c in range(1, cols):
        x[:, c] = rho * x[:, c - 1] + (1 - rho * rho) ** 0.5 * z[:, c]
    if rank:
        x = x + torch.randn(n, rank, generator=g, dtype=torch.float64) @ torch.randn(rank, cols, generator=fixed, dtype=torch.float64)
    scale = torch.rand(cols, generator=fixed, dtype=torch.float64) * 1.5 + 0.25
    return (x * scale).to(DEV)


def ivq(h, w, cb, col):
    """convert_layer_ds41.py's assignment of one matrix: RMS gains, assign, refit, assign, refit."""
    norm, g = h.vq.normalize(w)
    ids = h.vq.assign(norm, cb, colw=col)
    h.vq.refit_gain(w, ids, cb, g, colw=col)
    ids = h.vq.assign(h.vq.normalize_gain(w, g), cb, colw=col)
    h.vq.refit_gain(w, ids, cb, g, colw=col)
    return ids.view(w.shape[0], -1), g


def moments(x):
    """(sum x x^T, token count) of one expert, batched [1, C, C]."""
    return (x.T @ x)[None], torch.tensor([x.shape[0]], device=DEV)


def out_error(w, q, x):
    """||(W - Q) X^T||^2 / ||W X^T||^2 on tokens x [n, cols]."""
    return float(((w - q).double() @ x.T).square().sum() / (w.double() @ x.T).square().sum())


# ----------------------------------------------------------------------------- 1
def test_t0(h):
    print('[1] t = 0 is iVQ', flush=True)
    for rows, cols, k, p, seed in ((512, 128, 512, 3, 1), (300, 64, 256, 2, 2), (256, 192, 4096, 2, 3), (768, 128, 8192, 1, 4)):
        g = torch.Generator().manual_seed(seed)
        w = (torch.randn(p, rows, cols, generator=g) * (torch.rand(p, rows, 1, generator=g) * 3 + 0.01)).to(DEV)
        w[0, 7] = 0.0                                             # a zero row keeps gain +0
        cb, col = codebook(k, seed), profile(p, cols, seed)
        ref = [ivq(h, w[i].contiguous(), cb, col[i].contiguous()) for i in range(p)]
        x = correlated(2000, cols, 0.9, seed)
        r, _ = hv.correlation(moments(x)[0].expand(p, cols, cols), torch.full((p,), 2000, device=DEV), None, 0.0)
        hm = hv.hessian(col, r, 0.0, 0.0)                         # the correlation must not matter at t = 0
        v, info = hv.factor(hm)
        ids, gains = h.quantize(w, cb, hm, v)
        same_ids = all(torch.equal(ids[i], ref[i][0]) for i in range(p))
        same_gain = all(torch.equal(gains[i].view(torch.int16), ref[i][1].view(torch.int16)) for i in range(p))
        check(int(info.abs().sum()) == 0 and same_ids and same_gain,
              f'rows {rows} cols {cols} K {k} x{p}: indices and F16 gains identical to the iVQ kernels')
    # uniform columns: the unweighted assignment
    w = torch.randn(1, 256, 64, generator=torch.Generator().manual_seed(9)).to(DEV)
    cb = codebook(256, 9)
    norm, g = h.vq.normalize(w[0].contiguous())
    ids0 = h.vq.assign(norm, cb).view(256, -1)
    one = torch.ones(1, 64, device=DEV)
    hm = hv.hessian(one, torch.eye(64, dtype=torch.float64, device=DEV)[None], 0.0, 0.0)
    v, _ = hv.factor(hm)
    vinv, met = hv.block_tables(v)
    ids, _ = h.run_pass(w, h.rms_gains(w), v.float().contiguous(), vinv, met, cb)
    check(torch.equal(ids[0], ids0), 'uniform columns: one pass is the unweighted nearest assignment')


# ----------------------------------------------------------------------------- 2
def gptq_follow(w, g, cb, hm, ids):
    """Textbook GPTQ in FP64 (U upper with H^-1 = U^T U), fed the choices `ids`.
    -> (worst relative excess of a choice over the reference's best distance, share of exact
    agreements, residual W - Q, error sum)."""
    rows, cols = w.shape
    u = np.linalg.cholesky(np.linalg.inv(hm)).T
    t = w.copy()
    worst, agree, q = 0.0, 0, np.empty_like(w)
    for j in range(cols // 8):
        b = slice(8 * j, 8 * j + 8)
        tinv = np.linalg.inv(u[b, b])
        x = t[:, b] / g[:, None]
        d = (((x[:, None, :] - cb[None, :, :]) @ tinv) ** 2).sum(-1)
        best = d.argmin(1)
        pick = ids[:, j]
        agree += int((best == pick).sum())
        dp, db = d[np.arange(rows), pick], d[np.arange(rows), best]
        worst = max(worst, float(((dp - db) / np.maximum(db, 1e-30)).max()))
        q[:, b] = g[:, None] * cb[pick]
        e = (t[:, b] - q[:, b]) @ tinv
        t[:, 8 * j + 8:] -= e @ u[b, 8 * j + 8:]
    diff = w - q
    return worst, agree / ids.size, diff, float(np.einsum('rc,cd,rd->', diff, hm, diff))


def test_reference(h):
    print('[2] a pass against FP64 GPTQ', flush=True)
    for rows, cols, k, rho, t, damp, seed in ((256, 128, 256, 0.9, 1.0, 0.01, 11), (256, 192, 64, 0.6, 0.7, 0.0, 12),
                                              (512, 64, 1024, 0.95, 0.9, 0.0, 13)):
        g0 = torch.Generator().manual_seed(seed)
        w = (torch.randn(1, rows, cols, generator=g0) * (torch.rand(1, rows, 1, generator=g0) + 0.3)).to(DEV)
        cb, col = codebook(k, seed), profile(1, cols, seed)
        x = correlated(3000, cols, rho, seed, rank=2)
        r, _ = hv.correlation(*moments(x), None, 0.0)
        hm = hv.hessian(col, r, t, damp)
        v, info = hv.factor(hm)
        vinv, met = hv.block_tables(v)
        gains = h.rms_gains(w)
        ids, res = h.run_pass(w, gains, v.float().contiguous(), vinv, met, cb)
        worst, agree, diff, err = gptq_follow(w[0].double().cpu().numpy(), gains[0].double().cpu().numpy(),
                                              cb.double().cpu().numpy(), hm[0].cpu().numpy(), ids[0].cpu().numpy())
        res_ok = np.abs(res[0].double().cpu().numpy() - diff).max() < 1e-5 * float(w.abs().max())
        got = float(hv.output_error(res, hm).sum())
        check(int(info) == 0 and worst < 2e-4 and agree > 0.995 and res_ok and abs(got - err) < 1e-5 * err,
              f'cols {cols} K {k} rho {rho} t {t}: every choice within {worst:.1e} of the reference optimum, '
              f'{100 * agree:.2f}% the same codeword, error {got:.6g} vs {err:.6g}')
        # H = V V^T and the block metric is the Schur complement the textbook form uses
        hv_err = float((v[0] @ v[0].T - hm[0]).abs().max() / hm[0].abs().max())
        check(hv_err < 1e-12 and float(v[0].tril(-1).abs().max()) == 0.0, f'V upper, V V^T = H to {hv_err:.1e}')


# ----------------------------------------------------------------------------- 3
def test_feedback(h):
    print('[3] feedback on correlated inputs', flush=True)
    rows, cols, k, n = 512, 256, 256, 8192
    rates = {}
    for rho, rank in ((0.0, 0), (0.6, 0), (0.9, 0), (0.9, 4)):
        seed = 21
        g0 = torch.Generator().manual_seed(seed)
        w = (torch.randn(1, rows, cols, generator=g0) * (torch.rand(1, rows, 1, generator=g0) + 0.3)).to(DEV)
        # a codebook fitted to the normalized weights (a few Lloyd steps): ~1 bit per weight
        pool, _ = h.vq.normalize(w[0].contiguous())
        _, _, cbf, _ = h.vq.train(pool, k, 8)
        cb = cbf.to(torch.float8_e4m3fn).float().contiguous()
        x, fresh = correlated(n, cols, rho, 31, rank), correlated(n, cols, rho, 32, rank)
        s, cnt = moments(x)
        col = s[0].diagonal() / s[0].diagonal().mean()
        col = col.float()[None].contiguous()
        r, _ = hv.correlation(s, cnt, None, 0.0)
        row = {}
        for t, damp in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.01)):
            hm = hv.hessian(col, r, t, damp)
            v, _ = hv.factor(hm)
            ids, gains = h.quantize(w, cb, hm, v)
            q = h.restore(ids, gains, cb)
            row[t] = (out_error(w[0], q[0], x), out_error(w[0], q[0], fresh))
        rates[(rho, rank)] = row
        print(f'    rho {rho} rank {rank}: calibration / fresh output error  t=0 {row[0.0][0]:.4f} / {row[0.0][1]:.4f}  '
              f't=0.5 {row[0.5][0]:.4f} / {row[0.5][1]:.4f}  t=1 {row[1.0][0]:.4f} / {row[1.0][1]:.4f}', flush=True)
    r0, r6, r9, r9f = (rates[key] for key in ((0.0, 0), (0.6, 0), (0.9, 0), (0.9, 4)))
    check(abs(r0[1.0][1] / r0[0.0][1] - 1) < 0.03, 'uncorrelated inputs: feedback changes the fresh error by < 3%')
    for name, row, need in (('rho 0.6', r6, 0.95), ('rho 0.9', r9, 0.75), ('rho 0.9 + 4 factors', r9f, 0.75)):
        check(row[1.0][0] < need * row[0.0][0] and row[1.0][1] < need * row[0.0][1],
              f'{name}: t = 1 lowers the output error to {row[1.0][0] / row[0.0][0]:.2f}x (calibration) and '
              f'{row[1.0][1] / row[0.0][1]:.2f}x (fresh inputs) of t = 0')
        check(row[1.0][1] < row[0.5][1] < row[0.0][1], f'{name}: t = 0.5 lies between')
    check(r9[1.0][1] / r9[0.0][1] < r6[1.0][1] / r6[0.0][1], 'stronger correlation, larger gain')


# ----------------------------------------------------------------------------- 4
def test_gain(h):
    print('[4] gain refit', flush=True)
    rows, cols, k = 384, 128, 256
    g0 = torch.Generator().manual_seed(41)
    w = (torch.randn(2, rows, cols, generator=g0) * (torch.rand(2, rows, 1, generator=g0) + 0.3)).to(DEV)
    cb, col = codebook(k, 41), profile(2, cols, 41)
    x = correlated(4000, cols, 0.8, 41, rank=2)
    r, _ = hv.correlation(moments(x)[0].expand(2, cols, cols), torch.full((2,), 4000, device=DEV), None, 0.0)
    hm = hv.hessian(col, r, 0.8, 0.0)
    v, _ = hv.factor(hm)
    vinv, met = hv.block_tables(v)
    gains = h.rms_gains(w)
    ids, _ = h.run_pass(w, gains, v.float().contiguous(), vinv, met, cb)
    before = hv.output_error(w - h.restore(ids, gains, cb), hm)
    start = gains.clone()
    h.refit(w, ids, cb, hm, gains)
    after = hv.output_error(w - h.restore(ids, gains, cb), hm)
    code = cb[ids.long()].reshape(2, rows, cols).double()
    y = code @ hm
    exact = ((w.double() * y).sum(-1) / (code * y).sum(-1)).to(torch.float16)
    off = (gains.view(torch.int16).int() - exact.view(torch.int16).int()).abs()
    check(int(off.max()) <= 1 and float((off > 0).float().mean()) < 1e-3,
          f'F16 gains are the FP64 minimizer w H c / c H c ({int((off > 0).sum())} of {off.numel()} one ulp away)')
    check(bool((after <= before).all()) and float(after.sum()) < float(before.sum()),
          f'refit lowers the H error: {float(before.sum()):.5g} -> {float(after.sum()):.5g}')
    bump = hv.output_error(w - h.restore(ids, (gains.float() * 1.01).to(torch.float16), cb), hm)
    check(bool((bump >= after).all()), 'a 1% larger gain is worse on every problem')
    # a refit that is not a positive finite F16 keeps the previous gain
    neg = (-w).contiguous()
    kept = start.clone()
    h.refit(neg, ids, cb, hm, kept)
    check(torch.equal(kept.view(torch.int16), start.view(torch.int16)), 'non-positive refit keeps the previous gain')


# ----------------------------------------------------------------------------- 5
def test_rank(h):
    print('[5] rank-deficient second moments', flush=True)
    cols, k, rows = 128, 256, 256
    few = correlated(20, cols, 0.9, 51, rank=2)                  # 20 tokens, 128 columns
    many = correlated(20000, cols, 0.9, 52, rank=2)
    s, cnt = moments(few)
    prior = (many.T @ many) / many.shape[0]
    col = profile(1, cols, 51)
    raw, _ = hv.correlation(s, cnt, None, 0.0)
    check(int(torch.linalg.matrix_rank(raw[0])) <= 20, 'unshrunk correlation of 20 tokens has rank <= 20')
    _, info = hv.factor(hv.hessian(col, raw, 1.0, 0.0))
    check(int(info) != 0, 't = 1 without shrink or damping does not factor')
    hm = hv.hessian(col, raw, 1.0, 0.0)
    v, lifted, lift = cv.factor_safe(hm)
    check(lifted == 1 and lift > 0 and float((v[0] @ v[0].T - hm[0]).abs().max() / hm[0].abs().max()) < 1e-9,
          f'factor_safe lifts the diagonal ({lift:g} x mean) and factors the lifted matrix')
    w = torch.randn(1, rows, cols, generator=torch.Generator().manual_seed(53)).to(DEV)
    cb = codebook(k, 53)
    for name, pr in (('layer mean (gate/up)', prior), ('identity (down)', None)):
        r, _ = hv.correlation(s, cnt, pr, 256.0)
        hm = hv.hessian(col, r, 1.0, 0.0)
        v, info = hv.factor(hm)
        ok = int(info) == 0 and float(torch.linalg.eigvalsh(r[0]).min()) > 1e-6
        ids, gains = h.quantize(w, cb, hm, v)
        check(ok and bool(torch.isfinite(gains.float()).all()), f'shrunk toward the {name}: full rank, factors at t = 1')
    r, _ = hv.correlation(torch.zeros_like(s), torch.zeros_like(cnt), None, 256.0)
    check(torch.equal(r[0], torch.eye(cols, dtype=torch.float64, device=DEV)), 'an expert without tokens gets the identity')
    r, _ = hv.correlation(torch.zeros_like(s), torch.zeros_like(cnt), prior, 256.0)
    want = prior / prior.diagonal().sqrt()[:, None] / prior.diagonal().sqrt()[None, :]
    check(float((r[0] - want).abs().max()) < 1e-12, '... or the layer-mean correlation')


# ----------------------------------------------------------------------------- 6
def test_panel(h):
    print('[6] panel size', flush=True)
    rows, cols, k = 256, 256, 256
    w = torch.randn(1, rows, cols, generator=torch.Generator().manual_seed(61)).to(DEV)
    cb, col = codebook(k, 61), profile(1, cols, 61)
    x = correlated(4000, cols, 0.9, 61)
    r, _ = hv.correlation(*moments(x), None, 0.0)
    hm = hv.hessian(col, r, 0.9, 0.0)
    v, _ = hv.factor(hm)
    vinv, met = hv.block_tables(v)
    gains = h.rms_gains(w)
    out = {}
    for panel in (1, 5, 16):
        ids, res = h.run_pass(w, gains, v.float().contiguous(), vinv, met, cb, panel)
        worst, agree, _, _ = gptq_follow(w[0].double().cpu().numpy(), gains[0].double().cpu().numpy(),
                                         cb.double().cpu().numpy(), hm[0].cpu().numpy(), ids[0].cpu().numpy())
        out[panel] = (ids, float(hv.output_error(res, hm).sum()), worst)
    same = float((out[1][0] == out[16][0]).float().mean())
    check(all(o[2] < 2e-4 for o in out.values()) and abs(out[1][1] / out[16][1] - 1) < 5e-3 and same > 0.98,
          f'panels of 1, 5, 16 blocks: each a valid GPTQ pass, {100 * same:.2f}% identical codewords, '
          f'errors {out[1][1]:.6g} / {out[5][1]:.6g} / {out[16][1]:.6g}')


# ----------------------------------------------------------------------------- 7
def test_moments():
    print('[7] second moments', flush=True)
    dim, inter, experts = 64, 32, 3
    hs = LayerHess(experts, dim, inter, DEV, probe=(1,))
    g = torch.Generator().manual_seed(71)
    exact = {e: [torch.zeros(dim, dim, dtype=torch.float64), torch.zeros(inter, inter, dtype=torch.float64), 0] for e in range(experts)}
    for e, n, scale in ((0, 500, 1.0), (1, 70000, 1.0), (1, 300, 1e4), (2, 3, 1e-3)):
        x = (torch.randn(n, dim, generator=g) * scale * (torch.rand(n, 1, generator=g) * 100 + 0.01)).to(torch.bfloat16)
        act = torch.randn(n, inter, generator=g) * scale
        hs.add(e, x.to(DEV), act.to(DEV))
        for i, src in enumerate((x, act)):
            u = src.double()
            u = u / u.square().mean(-1, keepdim=True).sqrt()
            exact[e][i] += u.T @ u
        exact[e][2] += n
    ok = all(float((hs.xx[e].double().cpu() - exact[e][0]).norm() / exact[e][0].norm()) < 1e-6
             and float((hs.aa[e].double().cpu() - exact[e][1]).norm() / exact[e][1].norm()) < 1e-6
             and int(hs.count[e]) == exact[e][2] for e in range(experts))
    check(ok, 'sums are the FP64 sums of token-normalized x x^T and a a^T; counts')
    trace = float(hs.xx[1].diagonal().sum() / (hs.count[1] * dim))
    check(abs(trace - 1) < 1e-5, f'every token has unit mean square whatever its scale (trace / (n dim) = {trace:.7f})')
    rep = hs.probe_report()[0]
    check(rep['xx_gram64_sum32']['max_abs_correlation_error'] < 1e-5 and rep['expert'] == 1,
          f"precision probe: correlation error {rep['xx_gram64_sum32']['max_abs_correlation_error']:.1e} (FP64 Gram, FP32 sum), "
          f"{rep['xx_gram32_sum32']['max_abs_correlation_error']:.1e} (all FP32)")
    mean = sum(exact[e][0] for e in range(experts)) / sum(exact[e][2] for e in range(experts))
    check(float((hs.layer_mean().cpu() - mean).norm() / mean.norm()) < 1e-6, 'layer mean per token')
    with tempfile.TemporaryDirectory() as d:
        hs.meta = {'run_id': 'r'}
        hs.save(Path(d) / 'layer-00', {'layer': 0})
        back = LayerHess.load(Path(d) / 'layer-00', DEV)
        check(torch.equal(back.xx, hs.xx) and torch.equal(back.aa, hs.aa) and torch.equal(back.count, hs.count)
              and back.meta == {'run_id': 'r', 'layer': 0} and back.digest() == hs.digest(), 'save / load / digest')
    hs.xx[2, 0, 0] += 1
    check(back.digest() != hs.digest(), 'digest sees one changed element')
    flat = LayerHess(2, 8, 4, DEV)
    flat.count += 5
    flat.xx[0], flat.aa[0] = torch.eye(8, device=DEV), torch.eye(4, device=DEV)      # uncorrelated: rank = columns
    flat.xx[1], flat.aa[1] = torch.ones(8, 8, device=DEV), torch.ones(4, 4, device=DEV)   # one direction: rank 1
    d = flat.describe()
    check(d['effective_rank'] == {'xx': [1.0, 1.0, 8.0], 'aa': [1.0, 1.0, 4.0]} and d['count_median'] == 5.0,
          'effective rank: columns for uncorrelated channels, 1 for a single direction')


# ----------------------------------------------------------------------------- 8
def test_roundtrip(h):
    print('[8] DQVL v3 round trip', flush=True)
    shapes = {'w1': (256, 128), 'w3': (256, 128), 'w2': (128, 64)}
    nexp = 3
    for k in (4096, 8192):
        g0 = torch.Generator().manual_seed(k)
        cb_bytes = fmt.e4m3_encode(torch.randn(k, 8, generator=g0).numpy())
        cb = torch.from_numpy(fmt.e4m3_decode(cb_bytes)).to(DEV).contiguous()
        idx, gains, dec = {}, {}, {}
        for p, (rows, cols) in shapes.items():
            w = (torch.randn(nexp, rows, cols, generator=g0) * 0.05).to(DEV)
            x = correlated(2000, cols, 0.8, k)
            r, _ = hv.correlation(moments(x)[0].expand(nexp, cols, cols), torch.full((nexp,), 2000, device=DEV), None, 256.0)
            hm = hv.hessian(profile(nexp, cols, k), r, 0.9, 0.0)
            v, _ = hv.factor(hm)
            ids, gn = h.quantize(w, cb, hm, v)
            idx[p], gains[p] = ids.to(torch.int16).cpu().numpy(), gn.cpu().numpy()
            dec[p] = h.restore(ids, gn, cb)
        blob = fmt.pack_vq_layer(L=5, cb_bytes=cb_bytes, gains=gains, idx=idx)
        lay = fmt.decode_vq_blob(blob, 'test')
        same = all(np.array_equal(lay.idx[p], idx[p]) and np.array_equal(lay.gains[p].view(np.uint16), gains[p].view(np.uint16))
                   for p in shapes) and np.array_equal(lay.cb_bytes, cb_bytes)
        deq = all(torch.equal(fmt.dequant_experts(lay, range(nexp), p, DEV), dec[p]) for p in shapes)
        used = max(int(idx[p].max()) for p in shapes)
        check(same and deq and lay.K == k and lay.bits == (13 if k == 8192 else 12) and blob.size == lay.layout.total,
              f'K {k}: indices (max {used}), F16 gains and codebook bytes read back; dequantized weights equal')
        check(np.array_equal(fmt.pack_vq_layer(lay), blob), f'K {k}: repacking the decoded layer gives the same bytes')


def main():
    h = hv.Hvq()
    test_t0(h)
    test_reference(h)
    test_feedback(h)
    test_gain(h)
    test_rank(h)
    test_panel(h)
    test_moments()
    test_roundtrip(h)
    print(f'{len(FAILS)} failed' if FAILS else 'all passed', flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
