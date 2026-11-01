# The video VAE decoder, attributed, and its four bias passes removed (2026-10-29)

This unit takes the measurement `vae-options-2026-10-28.md` section 6 left open -
"the matmul and elementwise shares of FRAMES ... still rest on the 2026-10-27 nsys
run, which predates both the tensor default and the k=8/r=2 restructure" - and
lands the one change the attribution made obvious: the decoder's four block
linears no longer write a bias in a separate pass, their consumers add it on
load.

Commits: `c7bb59e1` (the fusion) and `a323cc7d` (the BF16-operand bench arm) plus
the one carrying this record. Spark `gx10-e1d2`, sm_121a, CUDA
13.0, no clock policy (busy 2.4-2.5 GHz). Job A as everywhere in this campaign:
768x768, 107 frames, seed 42, four evaluations, `H3_DIT_SOL=0.9`,
`H3_VAE_ATTENTION=tensor`, `H3_VAE_GEMM=bf16`, `H3_DIT_GEMM=fp8` (the tensorwise
route), the NVFP4 encoder.

## 1. The capture

One `nsys profile --trace=cuda` of a whole job A render (`~/decn.sh`,
`~/decn.nsys-rep` / `~/decn.sqlite`). The render printed the tensorwise FP8 arm's
own digest, `4611c9478c00ee28c159e04d9a057ee19bb08fe827aad7d79ef2f855392532aa`,
so the capture is the arm the campaign runs and not a neighbouring configuration.
Its phases: TEXT 14.511, LOAD 10.208, DENOISE 31.031, AUDIO 2.570, **FRAMES
34.511**, VIDEO 0.619 s.

The decoder is found by the two kernels only it launches
(`h3d_attention_mma_f32_kernel`, `h3d_video_qkv_rope_f32_kernel`), whose first
start and last end bound the phase. Inside that window: **33.243 s of wall,
29,518 kernels, 30.135 s of GPU busy time (90.7%)**, 3.108 s of gaps, and 88% of
the CLI's own FRAMES - the difference is the tile loop's first upload, the final
readback and the wrapper. `misc/decn/decn-split.py` reproduces every table here
from the sqlite.

## 2. The workload model, closed by the counts

`vae-options` section "The workload model this unit corrects" predicted the 768
job's plan from the code: `configured_tile_pixels` picks 304 pixels, so three
tiles an axis, a tile is 19x19 latents, `7 x 19 x 19 + 5` = **2,532 rows**,
`chunk_plan` gives 6 chunks, and 6 x 9 = **54 tile submissions** = **1,944 block
calls**. The capture's counts are the confirmation it did not have:

| what the plan predicts | count in the capture |
|---|---|
| 54 submissions | 53 `h3d_layer_norm_f32_kernel`, 53 proj_out GEMMs, 53 readbacks |
| 1,944 block calls | 1,944 rope, 1,944 attention, 1,943 swiglu, 1,943 x 4 block GEMMs |
| 2 per block | 3,886 `h3d_rms_norm_f32_kernel`, 3,886 `h3d_scale_add_f32_kernel` |

So the harness that priced one block at 88.006 ms and the job disagree for the
reason `vae-options` suspected: the job runs exactly the block count the plan
implies (1,944 x 88.006 ms = 171.1 s of matmul in a 196.9 s phase is impossible),
which makes the *harness's* F32 figure the outlier, not the block count. At the
reduced mode the same harness says 11.3 ms a block, and the capture's own four
block GEMMs sum to 18.36 s over 1,943 calls - see section 4.

## 3. The phase, per kernel

From the differenced-free single capture: `calls`, `total ms`, `avg ms` inside the
window, and what each kernel moves (its own arithmetic, from the shapes).

| kernel | grid | calls | total ms | avg ms | per call it moves |
|---|---|---:|---:|---:|---|
| `cutlass_..._gemm_128x256_16x3` (w1 2048->16384) | 80x16 | 1,943 | 9,159.77 | 4.714 | 320.9 MB, 169.9 GFLOP |
| `cutlass_..._gemm_128x128_32x3` (w2 8192->2048) | 160x2 | 1,943 | 4,520.40 | 2.327 | 170.8 MB, 85.0 GFLOP |
| `cutlass_..._gemm_128x256_16x3` (qkv 2048->6144) | 80x6 | 1,943 | 3,531.18 | 1.817 | 133.3 MB, 63.7 GFLOP |
| `h3d_attention_mma_f32_kernel` | 40x32 | 1,944 | 2,775.09 | 1.428 | 52.5 GFLOP |
| `h3d_linear_bias_kernel<float>` (w1) | 162048x1 | 1,943 | 2,587.18 | 1.332 | 331.9 MB |
| `h3d_swiglu_f32_kernel` | 32x2532 | 1,943 | 1,965.48 | 1.012 | 248.9 MB |
| `h3d_video_qkv_rope_f32_kernel` | 1x32 | 1,944 | 1,391.96 | 0.716 | 124.5 MB |
| `cutlass_..._gemm_128x128_16x5` (out 2048->2048) | 160x2 | 1,943 | 1,152.36 | 0.593 | 58.3 MB, 21.2 GFLOP |
| `h3d_linear_bias_kernel<float>` (qkv) | 60768x1 | 1,943 | 948.97 | 0.488 | 124.5 MB |
| `h3d_scale_add_f32_kernel` | 8x2532 | 3,886 | 912.95 | 0.235 | 62.2 MB |
| `h3d_rms_norm_f32_kernel` | 2532x1 | 3,886 | 624.71 | 0.161 | 41.5 MB |
| `h3d_linear_bias_kernel<float>` (out, w2) | 20256x1 | 3,886 | 486.90 | 0.125 | 41.5 MB |
| `cutlass_..._gemm_128x128_16x5` (x_embedder) | 160x3 | 53 | 47.40 | 0.894 | 20.7 MB, 0.006 GFLOP |
| `h3d_linear_bias_kernel<float>` (proj_out) | 30384x1 | 53 | 11.88 | 0.224 | 31.1 MB |
| `h3d_layer_norm_f32_kernel` | 2532x1 | 53 | 8.74 | 0.165 | 41.5 MB |
| `h3d_linear_bias_kernel<float>` (x_embedder) | 20216x1 | 53 | 6.08 | 0.115 | 20.7 MB |
| `cutlass_..._gemm_256x64_16x4` (proj_out 2048->3072) | 320x1 | 53 | 3.72 | 0.070 | 27.3 MB, 31.9 GFLOP |
| `cutlass_..._gemm_64x64_32x6` (post_quant 24->24) | 40x1 | 53 | 0.22 | 0.004 | 0.005 MB |
| `h3d_linear_bias_kernel<float>` (post_quant) | 237x1 | 53 | 0.07 | 0.001 | 0.2 MB |
| **total** | | **29,518** | **30,135.04** | | |

Groups, in time:

- the four block GEMMs **18,363.71 ms (60.9% of busy)**;
- the four block bias passes **4,023.05 ms**, plus 18.03 ms of per-tile biases;
- the F32 attention **2,775.09 ms**;
- SwiGLU 1,965.48, RoPE 1,391.96, scale-add 912.95, RMSNorm 624.71;
- everything per tile (x_embedder, proj_out, post_quant, the output layer norm)
  78.1 ms.

## 4. The GEMMs: 36 TFLOPS, and neither bandwidth- nor tensor-bound

660.3 TFLOP over the 1,943 block calls in 18.364 s is **35.96 TFLOPS**, and across
those same calls 1.33 TB moves, **72 GB/s**. Two readings:

- **Not bandwidth.** The 250 GB/s streaming rate would carry 1.33 TB in 5.3 s; the
  GEMMs take 18.4 s. Per shape the traffic rate is 68.1 (w1), 73.4 (w2), 73.4
  (qkv) and 98.3 GB/s (out) - 27% to 39% of the host's rate.
- **And the kernel class is TF32, not BF16.** cuBLASLt selects
  `cutlass_80_tensorop_s1688gemm_*` under `CUBLAS_COMPUTE_32F_FAST_16BF`, and
  `s1688` is CUTLASS's TF32 tensor-op tile (16x8x8). `vae-options` section "The
  matmul compute mode" measured the two reduced modes bit-identical in one
  process, on both hosts, through a raw cuBLASLt probe with no port code in the
  path - two names, one computation, and the one with the unambiguous name is
  TF32. 35.96 TFLOPS against an `mma.sync` TF32 peak of ~59 TFLOPS at 2.4 GHz
  (half the 118 BF16 rate at that clock) is **61% of the TF32 peak**.

That is the phase's structural finding: the decoder is executing F32-typed
activations and weights through the *TF32* path while the hardware has a BF16
path at twice the rate, and the consumer of every one of those GEMMs is a
bandwidth-bound F32 kernel at 250 GB/s that would read half the bytes if the
activations were BF16. It is a numerical change - it moves the rounding of both
operands from TF32's 10-bit mantissa to BF16's 8 - so it is priced, not taken
(section 8.3).

## 5. The elementwise kernels against their own floors

| kernel | per call | bytes | GB/s | floor at 250 GB/s |
|---|---:|---:|---:|---|
| bias, w1 (331.9 MB) | 1.332 ms | 331.9 MB | **249.2** | 1.328 ms |
| bias, qkv (124.5 MB) | 0.488 ms | 124.5 MB | **255.1** | 0.498 ms |
| bias, out + w2 (41.5 MB) | 0.125 ms | 41.5 MB | 332 (L2) | 0.166 ms |
| swiglu (249 MB) | 1.012 ms | 248.9 MB | **245.9** | 0.996 ms |
| scale-add (62 MB) | 0.235 ms | 62.2 MB | **264.7** | 0.249 ms |
| RMSNorm (41.5 MB) | 0.161 ms | 41.5 MB | **257.8** | 0.166 ms |
| layer norm (41.5 MB) | 0.165 ms | 41.5 MB | 251.5 | 0.166 ms |
| video QKV RoPE (124.5 MB) | 0.716 ms | 124.5 MB | 173.9 | 0.498 ms |

Every one of them is at the streaming rate except the RoPE kernel, which is 70%
of it: 0.218 ms a call, **0.42 s of the phase**, and it is the only elementwise
kernel with a lever left in it (it reads each QKV element twice, once for the
RMS sum and once for the rotation).

The consequence for the *bias* passes is that they cannot be made cheaper, only
removed: they exist because cuBLASLt refuses `CUBLASLT_EPILOGUE_BIAS` on these
layouts (`vae-options`, "Refuted"), and each one is a full read-modify-write of a
linear's output.

## 6. The gaps, and what they are

3.108 s of the window is GPU idle. The largest gaps - 341.0, 336.7, 333.2, 332.9,
332.9, 332.6 and then six of 100-111 ms - all sit **between the last kernel of one
tile and the first GEMM of the next**, which is where the host runs
`extract_latent_tile`, `prepare_input_rows`, `unpack_frames` and `stitch_tiles` in
Rust, and where it waits on the tile's readback. The readback itself is in the
window's copy table: **53 device-to-host copies, 1.65 GB, 1.972 s** (plus 159
device-to-device copies of 1.10 GB in 8 ms for the register-token copies, and 53
uploads of 12.8 MB). 1.65 GB is the projected output, `2,532 x 3,072 x 4 B` per
tile, read back to the host so `unpack_frames` can turn it into frames - 129x the
12.8 MB the latents cost to upload.

So the phase pays twice for the same design: 1.97 s of stream time to bring the
projected rows to the host, and up to 3.1 s of host time to unpack and stitch
them, neither of which the GPU is doing. That is the second lever, and it is a
schedule change rather than a numerical one (section 7).

## 7. The levers, ranked by measured recoverable time

| lever | class | measured size |
|---|---|---|
| the four block bias passes, fused into their consumers | schedule, bit-identical | **-3.876 s measured** (landed, section 8) |
| BF16 operands for the four block GEMMs | numerical (TF32 -> BF16 rounding) | 3.10x on the GEMMs, ~10.7 s of the phase net, against 132x the rounding (section 8.3) |
| the per-tile host work and the 1.65 GB readback | schedule, bit-identical | up to 3.1 s of idle + 1.97 s of stream time |
| the video QKV RoPE at 174 GB/s | schedule, bit-identical | 0.42 s (and the fusion already spent 0.19 s inside it) |
| the tile/chunk overlap (2.22x the rows the output needs) | the release's own plan | outside this unit: 54 submissions cover 136,728 rows for the 61,632 a 107-frame render needs |

The last row is the arithmetic the plan forces: nine 19x19-latent tiles cover
3,249 latents where the 48x48 plane has 2,304 (1.41x), and six chunks each run
seven latent frames to contribute two new ones (1.57x); 1.41 x 1.57 = 2.22. The
reference tiles and chunks the same way, so removing the overlap is a deviation
from the release, not a fix - it is named here so it is not rediscovered as a bug.

## 8. The change: four biases moved to their consumers

Each of the four consumers already reads its linear's whole output, so it takes
the bias pointer and adds it on load: `h3d_swiglu_f32` (w1),
`h3d_video_qkv_rope_f32` (qkv) and `h3d_scale_add_f32` (out and w2, before the
multiply). `h3d_vae_linear` is called with no bias for those four, and the three
kernels take a nullable bias so the video VAE encoder and the audio VAE, whose
gates are bit-exact against the reference, do not move.

The arithmetic is the reference's own: `h3_linear_bias_kernel` (`:783-791`) adds
one F32 value to one element and the consumer then reads it; with the bias in the
consumer, the same F32 add happens to the same element before the same consumer
operations. No value changes, so the render must stay byte-identical - which is
also the check that the four rewrites are right.

### 8.1 Measured, before against after

One binary per arm, four renders interleaved before/after/before/after on the same
host and environment, `~/decfuse.sh` and `~/decfuse-r{1,2}-{before,after}.*`:

| round | arm | TEXT | LOAD | DENOISE | AUDIO | FRAMES | VIDEO | MP4 sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | before (`375c801e`) | 10.744 | 9.381 | 31.170 | 2.549 | **33.624** | 0.611 | `4611c947...` |
| 2 | before (`375c801e`) | 10.989 | 10.170 | 31.617 | 2.583 | **34.588** | 0.570 | `4611c947...` |
| 1 | after (`c7bb59e1`) | 10.778 | 9.526 | 31.492 | 2.569 | **30.265** | 0.610 | `4611c947...` |
| 2 | after (`c7bb59e1`) | 11.185 | 10.249 | 31.808 | 2.603 | **30.195** | 0.594 | `4611c947...` |

FRAMES 34.106 s before against 30.230 s after, **-3.876 s, -11.4%**, with
`clocks.sm` sampled between renders at 2,405-2,411 MHz (13-16 W; the host has no
clock policy). The earlier attempt's own before render, under a different
session's cache state, was 33.998 s with the same digest. **All four renders are
byte-identical** to `4611c9478c00ee28...`, and ffmpeg's own comparison of before
against after, and of each arm against its own repeat, is
`PSNR inf / SSIM 1.000000` - which is the check the fusion's arithmetic argument
predicts.

### 8.2 The after-arm capture, kernel against kernel

The same window as section 1, one `nsys profile` of the after binary
(`~/decfuse.nsys-rep` / `.sqlite`, `~/decfuse-n.log`, whose render is
`4611c947...` too):

| kernel | before ms | after ms | delta |
|---|---:|---:|---:|
| w1 GEMM | 9,159.77 | 9,043.88 | -116 |
| w2 GEMM | 4,520.40 | 4,447.96 | -72 |
| qkv GEMM | 3,531.18 | 3,443.63 | -88 |
| out GEMM | 1,152.36 | 1,119.78 | -33 |
| **the four block bias passes** | **4,023.05** | **0** | **-4,023** |
| video QKV RoPE | 1,391.96 | 1,580.81 | +189 |
| SwiGLU | 1,965.48 | 1,933.95 | -32 |
| F32 attention | 2,775.09 | 2,826.92 | +52 |
| scale-add | 912.95 | 895.59 | -17 |
| RMSNorm | 624.71 | 673.32 | +49 |
| per-tile (embed, proj_out, post, layer norm) | 78.11 | 79.34 | +1 |
| **window / busy / kernels** | **33.243 / 30.135 s / 29,518** | **29.088 / 26.045 s / 21,746** | **-4.155 / -4.090 s** |

Three readings:

- **The four bias passes are gone, and nothing else took their place.** The only
  `h3d_linear_bias_kernel<float>` entries left are the three per-tile grids
  (proj_out, x_embedder, post_quant: 159 calls, 17.7 ms), and the four block GEMMs
  are each a few percent *faster*, which is the same weights and inputs measured in
  a later session.
- **The RoPE kernel pays 189 ms of the 4,023 ms back.** It re-reads each QKV
  element up to three times (once for the head's RMS sum, once for the rotated
  pair), so the bias add lands on each of those loads: 0.716 -> 0.813 ms a call.
  A pre-biased QKV buffer is what this change removed, so the residue is inherent
  to fusing it here; what is left would need the kernel to hold the biased values
  in registers across the reduction, which it cannot in its one-thread-per-
  dimension form.
- The +52 ms on the attention and +49 ms on the RMSNorm are cross-session moves on
  kernels this change does not touch (1.9% and 7.8%); the four renders' byte
  equality is what rules out a real regression there.

## 8.3 The BF16 operand arm, priced

`tests/h3/vae_gemm_bm 2532 3` on the Spark (`~/decfuse-bench.txt`), the same four
block shapes, three calls with the first discarded:

| mode | ms a block | note |
|---|---:|---|
| `fp32` | 89.906 | the reference path, and `vae-options`' own 88.006 reproduced |
| `tf32` | 12.599 | |
| `bf16` (`H3_VAE_GEMM=bf16`) | 11.283 | one computation with `tf32`, bit for bit |
| **BF16 operands** (probe, not an arm) | **3.634** | A and B typed BF16, output BF16 |

So at the decoder's own shapes the BF16 class is **3.10x** the FAST class it runs
today, and the arm's price is a numerical one: `bf16i` against `fp32` measures
`max_abs 1.413, rms 0.5768, rel_rms 3.63e-3` against the FAST pair's 2.75e-5 -
132x the rounding, because TF32 carries 10 mantissa bits and BF16 eight. The four
block GEMMs' 18.364 s in situ project to **5.92 s** at that ratio, less the
per-call activation rounding the probe excludes (each GEMM's A operand, 218 MB a
block, ~0.87 ms, ~1.7 s over the phase): about **10.7 s of the phase**, or 11% of
the 92.6 s render, at the cost of a rounding change on every decoder activation.

Two limits on that number, both in the probe and not in the arm: the bench's FAST
figure is 11.283 ms a block against the in-situ 9.571, so `3.634` is divided by a
denominator 17.9% above the model's own; and a first call costs 159.4 ms
(cuBLASLt's algorithm selection, and on sm_89 a JIT), which is why it is
discarded and every call printed.

**The smallest form of the arm is not available.** BF16 operands with the
decoder's own F32 accumulator - the version that would leave every elementwise
kernel and every consumer untouched - is refused on both hosts, and differently:
the workstation's sm_89 returns `CUBLAS_STATUS_INTERNAL_ERROR` and the context is
unusable afterwards (`--f32out`), while the Spark's sm_121 **segfaults inside the
call**, with the three FAST modes' own numbers already on stdout and the probe's
status line never reached. So an adopted arm would have to write BF16 output and
teach the consumers to read it, which is why it is a unit of its own rather than
a switch: BF16-resident weights, per-call activation rounding, six elementwise
kernels reading half the bytes, and a quality gate this campaign has not yet
built for anything but renders.

## 9. What this does not establish

- **What the BF16 operand arm would really buy.** Section 8.3 measures the
  GEMM ratio at the decoder's shapes (3.10x) and the rounding it costs
  (rel_rms 3.63e-3 against the FAST pair's 2.75e-5), but the phase projection
  divides an in-situ 18.364 s by a bench ratio whose own denominator is 17.9%
  above the in-situ model, and it excludes the per-call activation rounding. No
  arm exists; the F32-output form of it is refused on sm_89 and segfaults on
  sm_121.
- **Whether the BF16 rounding is visible at all.** Nothing here renders the arm,
  and this campaign's four-step renders cannot rank arms that differ numerically
  (`fp8-row-arm-2026-10-29.md` section 6); a converged sampler or a teacher-forced
  metric is what a quality decision on this arm would need.
- **That the bias fusion is the whole of section 5's recoverable time.** It is the
  4.02 s the capture attributes to the four block bias passes; the 18 ms of
  per-tile biases (x_embedder, proj_out, post_quant) stay in their own kernel
  because their consumers are the packing and the readback, not a block op.
- **Anything about the audio VAE** (2.57 s, `h3d_conv1d_kernel` and
  `h3d_conv_transpose1d_kernel` in the same capture) or the DiT: this unit's
  window is the video decoder only.
- **One geometry, one arm, one capture.** The kernel shares are of the 768x768
  107-frame job on the tensorwise FP8 route; a 512 job takes 288-pixel tiles and
  three chunks, and the scalar-attention arm (if anyone runs it) puts 80.3 s of
  scalar attention in this phase instead of 2.78.
- **That every kernel is at its floor.** The floors in section 5 are the host's
  250 GB/s streaming rate applied to each kernel's own bytes; they say nothing
  about a fused or restructured kernel's floor, which for swiglu and the norm pair
  could be lower if the elementwise chain were restructured the way the DiT's was
  (`dit-elementwise-2026-10-29.md`).
