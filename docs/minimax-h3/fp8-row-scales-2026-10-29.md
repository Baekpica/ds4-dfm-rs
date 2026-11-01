# The FP8 arm's per-row scales: cuBLASLt refuses, and the route that is left (2026-10-29)

The tensorwise FP8 arm is ported and 29.3% of an evaluation faster
(`fp8-route-2026-10-29.md`), and its render is a different sample of the same
shot. This unit asked the reference's own higher-accuracy alternative -
`granularity="channel"`, per-output-channel weight scales and per-token
activation scales - whether this port can reach it at all. The answer is no
through cuBLASLt, for a reason worth recording, and yes through a route the
reference does not need but this port can afford; the measurements that had to
be re-established first are in sections 1 and 2.

## 1. The host had a stray resident, and it was mine

A `gdb -batch -ex run -ex bt ... --args ./tests/h3_dit_gate ...` in the Spark
checkout started at 12:43:11 and hung (`Sl`, 0.9% CPU,
`futex_do_wait`, no progress for 58 minutes) while holding **18,655 MiB** of
the GPU. Every capture and render between then and 13:42 ran beside it, and it
cost real time: the same BF16 render that produced `5b5e735c...` at DENOISE
54.932 s beside the stray took **45.061 s** with the GPU empty. It is killed;
the GPU is idle (208 MHz, 7 W, no compute applications) and the two renders
below are the clean ones.

This is the campaign's own leak, not the operator's work: `h3_dit_gate` in
the Spark checkout is this port's gate. Check `nvidia-smi
--query-compute-apps` before any measurement, not just `pgrep -x h3-rs`.

## 2. The BF16 capture's 10x swiglu anomaly is reproducible and is not a shape mixture

`h3d_swiglu_kernel` runs 7.38-7.49 ms per call in five of six captures and
74.87-72.97 in the BF16 arm's steps=4 capture - twice, in two independent
pairs (`fp8-bf16-4`, `fp8-bf16b-4`). Grouping every swiglu call by grid
answers the obvious explanation: the kernel has exactly two shapes, `(11, 56)`
block 256 with two calls per capture and `(18799, 56)` block 256 for the rest,
and the slow capture uses the same grid, block and call count as the fast ones.
It is not two variants sharing a name and it is not the steps=2/steps=4
difference: the FP8 arm's steps=4 capture at the same shape and the same
period runs 7.46 ms.

So the anomaly stays unexplained and confined to that one arm's longer
capture. It does not touch the render (the render is the arbiter and matches
its own capture rate), but the BF16 nsys column's swiglu is taken from the
steps=2 captures, and any future BF16 attribution should re-check it.

## 3. The arm re-measured, on a clean GPU, interleaved

Job A, one binary, the canonical fast env of `steady-state-2026-10-28.md`
(`H3_DIT_SOL=0.9 H3_VAE_ATTENTION=tensor H3_VAE_GEMM=bf16`), four renders in
the order bf16, fp8, bf16, fp8:

| arm | round 1 DENOISE | round 2 DENOISE | MP4 |
|---|---:|---:|---|
| BF16 | 45.061 s | 45.388 s | `5b5e735c1ebcbf2e99abd3a69f210affd0844406c473bdedca707cabeb224876` |
| FP8 tensorwise | 31.766 s | 31.849 s | `4611c9478c00ee28c159e04d9a057ee19bb08fe827aad7d79ef2f855392532aa` |

Both arms repeat to 0.3-0.7% and to the same digest, so the pair settles the
ratio: **1.422x, 13.41 s of DENOISE saved, 29.7%**, against the 29.3% the
per-evaluation kernel table gives (`fp8-route-2026-10-29.md`). The BF16
absolute is host-state dependent: 50.55 s in the steady-state record, 54.93
beside the stray, 45.2 s here. Only the pair's own numbers are quoted.

The first attempt at this pair is worth one line because it is the trap: it
omitted `H3_DIT_SOL=0.9`, which reverts the DiT to dense attention (the value
is read once per load, `dit.rs:1499-1520`), and produced DENOISE 84.359 s and
a third digest. An arm is only the arm if its whole env is set.

## 4. What the FP8 arm's output costs, measured

**Pixel metrics** (FFmpeg on the decoded frames, BF16 as reference): PSNR y
**21.720 dB**, u 28.510, v 31.188, average 23.144, min 19.973; SSIM Y
**0.715096**, U 0.845, V 0.891, all 0.766. Luma RMS 20.919 over the 107
frames - the figure the port record carries, reproduced exactly, and now with
its scale attached: 21.7 dB is a different picture, not a noisy one.

**It is not softness and it is not a delay.** Sliding the whole clip against
itself peaks at zero offset (k=0: 21.720, k=+1: 21.675, k=-1: 21.568), so
there is no global time shift. Blurring the BF16 arm towards the FP8 arm buys
0.3 dB (21.72 -> 22.06 at sigma 2.0), so it is not the FP8 arm being a blurred
BF16 arm either. Looking at frames 10, 50 and 100 of both: same framing, same
light, the fox at a different phase of its stride, the tree line redrawn, and
**each arm carries its own local artefact** - the BF16 frame at 100 smears the
belly and forelegs into the snow, the FP8 frame at 10 does the same to the
under-body. The port record's "visibly softer" is a summary the frames do not
support as written; at four steps from a distilled checkpoint the sampler is
chaotic in the sense `lora-merge-2026-10-28.md` §PSNR already established, and
both clips are finished.

**Detail energy.** The campaign's instrument (`~/detail.sh`, "the whole-render
mean of `|frame - luma-4 blur|`") is gone from the Spark and its input renders
(the twenty-step A/B pair, the `nvfp4-gate-*` pair) with it, so
its recorded numbers cannot be reproduced or extended. The reimplementation
used here is named, not substituted silently: mean over all 107 frames of
`|frame - gblur(sigma)|` on luma, through FFmpeg's `signalstats`. Its own
scale differs from the record's - 1.8839 against the recorded 3.1893 on the
very render that value belongs to - so the two are not comparable and the
recorded detail energies are quarantined:

| render | sigma 4 | sigma 8 |
|---|---:|---:|
| BF16 fast arm (`5b5e735c`) | 1.8839 | 2.9788 |
| FP8 tensorwise (`4611c947`) | 1.4284 | 2.4477 |
| dense BF16, `H3_DIT_SOL` unset (the §3 trap render) | 2.3985 | 3.4369 |

The ordering is the one the campaign expects (dense is the over-sharpened arm,
`lora-merge-2026-10-28.md`), and on that self-consistent instrument **the FP8
tensorwise arm carries 18-24% less high-frequency detail energy than BF16**.

## 5. The reference's per-row granularity, and cuBLASLt's answer

The reference is `fastvideo/layers/quantization/fp8_config.py` at the release's
pinned revision (`3d8ac9d14bd697a89ede8f170cbfbca012a9edcc`), read again this
unit. The channel branch is:

- activations, `_quantize_rowwise`: `x_absmax = x_2d.abs().amax(dim=-1,
  keepdim=True)`, same `(absmax/448).clamp(min=1/(448*512))` floor, then
  `(x / x_scale.to(x.dtype)).clamp(-448, 448).to(float8_e4m3fn)`, returning
  `x_scale` as `[M, 1]`;
- weights, `convert_model_to_fp8`'s channel branch: `weight.abs().amax(dim=1)`,
  the same floor, and `(weight / w_scale.to(weight.dtype).unsqueeze(1)).clamp(
  -448, 448)` - one scale per **output channel**;
- the matmul, `apply`: `scale_b = w_scale.view(1, -1)` against
  `scale_a = x_scale` `[M, 1]`, so the two scale vectors enter as an **outer
  product**, and the docstring's own words are "higher accuracy but slower".

cuBLASLt has a mode for exactly that. `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
= 3` (`cublasLt.h:944` locally, `:933` in the Spark's CUDA 13.0), set through
`CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31` / `_B_SCALE_MODE = 32` (`:1651`,
`:1656`; Spark `:1552`, `:1557`), documented as "the vectors are expected to
have M and N elements respectively, and each (i, j)-th element of product of A
and B is multiplied by i-th element of A scale and j-th element of B scale"
(`:941-945`). It does not work here, and the two failures are different:

| operands | result | meaning |
|---|---|---|
| E4M3, four DiT shapes, with and without a 64 MiB workspace, all-ones or patterned scales | `status 15` = `CUBLAS_STATUS_NOT_SUPPORTED` (`cublas_api.h:98`) | the descriptor is accepted and **no kernel serves it** |
| BF16, same attributes, same vectors | `status 7` = `CUBLAS_STATUS_INVALID_VALUE` (`cublas_api.h:93`) | the **combination is rejected** - what the header promises for an unsupported data/scale/compute triple (`cublasLt.h:1524`) |

The probe is `tests/h3/fp8_gemm_bm`'s modes 6-8 and 10, extended this unit; the
tiny 512x1024x2048 shape fails the same way as the four real ones, so it is not
size. The conclusion is that the mode belongs to the block-scaled family this
port does not run, and the per-row arm cannot go through cuBLASLt on this
architecture and CUDA version.

## 6. The route that is left, and what it costs

cuBLASLt will not apply the two vectors, but they are a **post-multiply**: the
reference's own `apply` is `out = (A_fp8 @ W_fp8^T) * scale_a * scale_b`
broadcast over `(i, j)`, and the port can do that multiply itself. Per
evaluation:

1. two extra per-row amax/scale/quantize kernels - the same shape as the
   existing tensorwise pair (`h3d_fp8_amax_kernel`, `h3d_fp8_scale_kernel`,
   `h3d_fp8_quantize_kernel`, `h3d_dit_gpu.cu:519-627`), with the reduction
   over `k` per row instead of over the whole tensor, for both the activation
   `[rows, k]` and the weight `[n, k]` (whose rows are the output channels);
2. the same GEMM with no scale pointers at all (alpha 1), as variant 0 of the
   reproducer already runs;
3. one elementwise epilogue over the output: `bf16(acc * row[i] * col[j])`.

Step 2 is measured, per call, at the four projection shapes (mode 9, median of
three, GPU empty):

| projection | shape | no-scale FP8 GEMM | tensorwise arm's own kernel |
|---|---|---:|---:|
| qkv | 18799x5376x21504 | 21.588 ms | 26.7 ms (nvjet 128x256x128) |
| fc1 | 18799x5376x28672 | 29.074 ms | |
| attention out | 18799x7168x5376 | 6.935 ms | 11.3 ms (nvjet 192x176x128) |
| fc2 | 18799x14336x5376 | 14.961 ms | |

The two wide-N projections and the two narrow-N ones average 25.3 and 10.9 ms
against the tensorwise table's 26.7 and 11.3, so **the scale vectors cost
nothing in the GEMM**: the epilogue is the whole price. It is 4.58 GB of F32
read plus 2.29 GB of BF16 written per evaluation (the four outputs at 18,799
rows), about **27 ms at this host's ~250 GB/s**, plus whatever the F32 output
costs the GEMM itself; call it under 1% of an 8.0 s evaluation, against the
3,312 ms the arm already saves.

That is the next unit: the per-row kernels, the epilogue, and
`H3_DIT_GEMM=bf16|fp8|fp8-row`, with the three-arm table the port record asks
for (ms per evaluation against detail energy and the render digest).

*Corrected 2026-10-29*: the epilogue paragraph above is 50x low - it summed
the four output matrices once, when an evaluation runs each of them once **per
block**, 50 times. The measured epilogue is **1,648 ms per evaluation** (32.96
ms per block-set, 206 GB/s, at the streaming rate), plus 231 ms for the F32
output the raw matmul must leave instead of BF16. Built and priced in
[`fp8-row-arm-2026-10-29.md`](fp8-row-arm-2026-10-29.md): the arm is exact but
costs 1,728 ms more per evaluation than the tensorwise arm, and the identity
this section claimed - "the scale vectors cost nothing in the GEMM" - does not
survive the F32 output, which more than eats the unscaled kernel's own gain.

## 7. What this unit does not establish

- **No logit- or latent-level quality comparison**, and no NLL over many
  positions: the render pair is the instrument, and it is chaotic at four
  steps. The FP8 arm is not shown to be worse or better than BF16 in any
  perceptual sense; it is shown to be a different, softer-by-detail-energy
  sample of the same shot.
- **The detail-energy scale is reimplemented**, for the reason in §4; the
  recorded 3.1893 / 3.2824 / 4.5626 cannot be reproduced on this host because
  neither the script nor its inputs survive.
- **The swiglu anomaly is unexplained** (§2).
- **The per-row arm is designed, not built.** Its arithmetic follows the
  reference's two branches, but nothing has been measured end to end, and the
  epilogue's own cost is an estimate from bandwidth, not a measurement.
