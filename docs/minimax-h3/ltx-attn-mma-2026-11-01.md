# The dense attention, tensor-core (2026-11-01)

Phase D, third pass. `stage2-attention-cost-2026-11-01.md` located 78 % of a
refiner block in four attention kernels running at 0.8-4.4 TFLOPS where the
tower's own kernel does 61.5 on the same shape, and named the cause: the
refiner's operands are token-major and the tensor-core arms only ever read
head-major. This is the first of those four, ported and measured.

**Result: stage 2's denoise falls from 121.781 s to 94.046 s, -22.8 %, and the
picture is the same fox.** The block falls from 830.6 ms to 638.0 ms.

Arm: `H3_LTX_ATTN_MMA=1` (`3fc1567f`). Default unchanged - the scalar walk
stays the default and the one-block golden's bit-exact reference.

## 1. What was ported, and how little it was

`h3d_attention_mma_kernel` (`native/h3/h3_dit_gpu.cu:4151`) reads head-major
operands and **writes token-major output**: `output[((row * heads) + head) *
dim]`. The refiner's operands are token-major, which is why the port wrote a
separate scalar walk instead of transposing 126 MB of Q per layer. Since the
output layout already matched, the port is three index expressions
(`query`, and the two K/V staging reads) and two extents (`q_rows` and
`kv_rows` instead of one `sequence`, which the 16,128-by-1,024 text
cross-attention needs). Everything else - the 16-row query tile per warp, the
64-key staging through shared memory with `ldmatrix`, the natural-log online
softmax, the `mma` fragments - is the tower's kernel unchanged.

`h3d_ltx_attention_mma` (`:4569`) validates exactly what `h3d_ltx_attention`
validates and adds the kernel's one restriction: head dimension 128. That
leaves A2V and V2A (64 wide, 4.1 % of a block) on the scalar kernel.

## 2. The two kernels on the same operands

`tests/h3/ltx_attn_bm 16128 3`, both arms on identical BF16 operands, one
warm-up pass each:

| shape | scalar | MMA | gain |
| --- | ---: | ---: | ---: |
| square 16,128 x 16,128 x 32 heads | 2,300.967 ms, 1.85 TFLOPS | 69.480 ms, 61.34 TFLOPS | 33.1x |
| cross 16,128 x 1,024 x 32 heads | 146.462 ms, 1.85 TFLOPS | 5.169 ms, 52.35 TFLOPS | 28.3x |

The square MMA figure lands on the head-major kernel's own 69.3 ms, which says
the token-major reads cost nothing measurable - the strided rows are consumed
through `uint4` copies and per-row gathers either way.

## 3. What they compute, and how far apart

The two arms are different arithmetic classes: FMA with a row-wise walk against
tensor-core products with online softmax, so `h3_ltx_refiner_gate`'s bit gate
cannot hold both. The numbers it would report instead:

| shape | differing | rel_rms | max_abs | the output's own scale |
| --- | ---: | ---: | ---: | ---: |
| square | 65.2 % | 1.024e-4 | 4.883e-4 | 0.096 |
| cross | 54.4 % | 2.211e-4 | 9.766e-4 | 0.198 |

One to two BF16 ulps at those magnitudes (`max_abs / scale` = 5.1e-3 and
4.9e-3, against a BF16 ulp of 7.8e-3 at 0.096 and 7.8e-3 at 0.198). 0.10 % and
0.08 % of the 516,096 output rows have a 128-wide sum whose *sign* the last bits
decide; those are the only rows where a difference of this size is not a
rounding difference, and they are the rows whose sum is within one part in a
thousand of zero.

## 4. End to end, on the deliverable route

Same route, same draft (`/mnt/h3/recipe.bin`, sha256 `2cb3e8d2...`), same seed,
anchor 0.7, one switch apart, both runs with the same two profile switches:

| | step 0 | step 1 | step 2 | denoise | VAE decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| scalar (control) | 44.691 | 40.322 | 36.768 | 121.781 | 15.215 |
| `H3_LTX_ATTN_MMA=1` | 35.512 | 31.040 | 27.494 | **94.046** | 15.692 |

-9.18 / -9.28 / -9.27 s a step, which is the prediction to within 2 %:
142.2 ms of text cross-attention per block (147.86 -> 5.66 in the block's own
lap) is 6.83 s of 48 blocks, plus layer 0's dense self-attention at 2,301.9 ->
69.5 ms is 2.23 s, total 9.06 s a step.

The block split with the arm on, means over 144 block-steps, both runs with
the same two profile switches:

| group | scalar | MMA |
| --- | ---: | ---: |
| video self + text, the group | 696.1 | 506.2 |
| - `attn1` projections, norms, rotary, search | 36.1 | 35.8 |
| - `attn1` core (141 routed + 3 dense) | 469.8 | 422.6 |
| - `attn2` projections and norms | 28.5 | 28.4 |
| - **`attn2` core (dense, all 144)** | **147.9** | **5.7** |
| - the residual / per-head gate / out-projection | 13.8 | 13.7 |
| the AV pair | 65.4 | 65.6 |
| video feed-forward | 66.5 | 63.4 |
| audio, all of it | 2.7 | 2.8 |
| **block** | **830.6** | **638.0** |

The `attn1` core is a mean over 144 block-steps, so it mixes the 141 routed
instances with layer 0's three dense ones; the layer-0 instance on its own is
2,301.9 -> 69.5 ms, and the routed Sol forward is unchanged at 430.8 in both
runs. The feed-forward's 66.5 -> 63.4 is not the arm - it is the same GEMM on
a different state - and it is the size of this instrument's run-to-run spread.

At 94.046 s the recipe's stage 2 is 172.027 - 27.735 = 144.3 s, so a full run
is ~221 s against 248.5 s.

## 5. The picture

Frames at 0.5 / 1.5 / 2.5 / 3.5 / 4.5 s, control against arm:

| t | SSIM | PSNR |
| --- | ---: | ---: |
| 0.5 s | 0.9774 | 43.46 dB |
| 1.5 s | 0.9815 | 43.95 dB |
| 2.5 s | 0.9802 | 43.76 dB |
| 3.5 s | 0.9787 | 43.30 dB |
| 4.5 s | 0.9761 | 42.98 dB |

The fox is the same fox, in the same pose, at the same framing, through all five
frames - the arm renders slightly finer fur and edge detail, which is what a
different accumulation order in a three-step refinement looks like. 43 dB is not
"identical"; it is the size a 1e-4 per-attention difference reaches after 48
blocks x 3 steps of a diffusion, and the acceptance question is whether the
subject and the motion survive, not whether the pixels do.

Clips: `outputs/prof-mma.mp4` (arm) and `outputs/prof-stages.mp4` (control),
same binary, same draft, one switch apart.

## 6. What is left in the block

With the dense core done, the block is 638.0 ms and **430.8 ms of it (67.5 %)
is the Sol sparse forward** - now the single largest cost in the campaign, and
the same fix applies: it is `h3d_sol_ltx_forward_kernel`
(`native/h3/h3_dit_gpu.cu:7296`), the token-major re-expression of
`h3d_sol_forward_kernel`, which measures 30.5 ms at this shape against the
refiner's 430.8. Its 48 % exact / 52 % folded split says an `mma` dense walk
plus a folded correction is the natural shape. A2V/V2A (34.0 ms, 5.3 %) would
need head dimension 64 in the MMA kernel, and the projections (131 ms, 20.5 %)
are already at 92.5 TFLOPS.

## 7. Artifacts

- Kernel and entry: `native/h3/h3_dit_gpu.cu` (`:4370` the kernel, `:4569` the
  entry), declaration `native/h3/h3_dit_gpu.h`, binding
  `crates/ds4-sys/src/h3_dit.rs`, switch
  `crates/ds4-core/src/minimax_h3/ltx_refiner.rs` (`env_mma`).
- `tests/h3/ltx_attn_bm.cu`, `make tests/h3/ltx_attn_bm`; log
  `scratch/ltxattn.log` on the Spark.
- Run: `scratch/mma-arms.sh`, log `scratch/mma-arms.log` (4,318 STAGES lines).
