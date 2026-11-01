# Stage 2's attention is the wall, not the GEMM (2026-11-01)

Phase D, second pass. The first pass closed on a question: a routed block costs
759 ms and its projections are `414 TFLOP` by the tower's parameter count, so
11 TFLOPS - and the GEMMs, the reasoning went, must be running at a fraction of
what cuBLASLt reaches here. That is **wrong**, and this pass says so with three
measurements:

- the block's thirty-four projections run at **92.51 TFLOPS** in isolation, not
  11, and take **94.9 ms** of a block;
- the block is **830.6 ms** (mean over 48 layers x 3 steps), and **78 %** of it
  is four hand-written attention kernels;
- those kernels' H3 counterparts - the same walk, in the tower's head-major
  layout - measure **61.5 TFLOPS** at the same shape, so the gap is the port's
  layout choice on the attention, not the arithmetic this part can do.

Instruments: `tests/h3/refiner_gemm_bm` (new, `9de9929d`), `H3_BLOCK_STAGES=1`
and `H3_ATTN_PROFILE=1` (new, `1f058ea7`), `tests/h3/sol_forward_bm` rebuilt and
run at this shape. Route unchanged: release sigmas, Sol on, anchor 0.7, the
recipe's own draft, 16,128 video rows.

## 1. The projections are not the gap

`tests/h3/refiner_gemm_bm` runs the block's thirty-four projections in forward
order at 16,128 video rows, 126 audio rows and 1,024 prompt rows, each synced
and timed, after a warm-up pass. The whole set:

| arm | GEMM total | TFLOPS |
| --- | ---: | ---: |
| `CUBLAS_COMPUTE_32F` (what `h3d_linear` asks for) | 94.865 ms | 92.51 |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | 90.208 ms | 97.29 |

The compute class buys 5.2 % and nothing more, and 92.51 TFLOPS is *above* the
65-90 band the campaign had been quoting as cuBLAS's reach here. Largest shapes:

| projection | rows | shape | ms | TFLOPS |
| --- | ---: | --- | ---: | ---: |
| video FF up | 16,128 | 4096 -> 16384 | 22.317 | 97.00 |
| video FF down | 16,128 | 16384 -> 4096 | 26.157 | 82.76 |
| video self Q/K/V/out, each | 16,128 | 4096 -> 4096 | 5.28 - 5.43 | 99.7 - 102.5 |
| video text K/V, each | 1,024 | 4096 -> 4096 | 0.35 | 96 - 98 |
| A2V Q | 16,128 | 4096 -> 2048 | 2.656 | 101.88 |
| V2A K/V, each | 16,128 | 4096 -> 2048 | 2.59 - 2.83 | 95 - 105 |

So the projection path is healthy and Phase D has nothing to win there. The
`FAST_16BF` arm is priced and rejected on value, not on risk: 5 % of 95 ms is
4.7 ms of a 40 s step.

## 2. Where a block actually goes

`H3_BLOCK_STAGES=1` laps the five groups the forward has and
`H3_ATTN_PROFILE=1` laps each attention's own stages, both in one run. Mean over
48 layers x 3 steps = 144 block-steps:

| group | ms | share |
| --- | ---: | ---: |
| video self-attention (`attn1`) | 505.9 | 60.9 % |
| - projections (Q, K, V) | 26.5 | 3.2 % |
| - q/k RMS norms | 2.3 | 0.3 % |
| - rotary pair | 4.3 | 0.5 % |
| - Sol routing search | 3.0 | 0.4 % |
| - **Sol sparse forward** | **469.8** | **56.6 %** |
| video text cross-attention (`attn2`) | 176.4 | 21.2 % |
| - projections (routed Q on video rows) | 27.3 | 3.3 % |
| - q/k RMS norms | 1.2 | 0.1 % |
| - **dense attention core** | **147.9** | **17.8 %** |
| - the residual/gate/out-projection around the two | 13.8 | 1.7 % |
| the AV pair (A2V, V2A) | 65.4 | 7.9 % |
| - projections | 28.0 | 3.4 % |
| - the two attention cores | 34.0 | 4.1 % |
| video feed-forward | 66.5 | 8.0 % |
| - projections | 48.5 | 5.8 % |
| audio self + text + feed-forward, all of it | 2.7 | 0.3 % |
| **block** | **830.6** | |

The four attention cores - Sol sparse forward, the text cross-attention, A2V
and V2A - are 651.7 ms of 830.6, **78.4 %**. Everything the GEMM does, in all
six attentions and the feed-forward, is 130.9 ms, **15.8 %**.

The mean is over three steps because they are not equal: step 0's block is
915.4 ms against step 2's 751.2 ms. The difference is the first step's weight
touch (section 5), and it is inside the blocks, not around them.

## 3. The attention kernels are the outlier

Each core's rate, its own FLOPs over its measured time - dense for the dense
ones, and 48 % of dense for the routed one, which is the share of query-block /
KV-block pairs the route keeps exact at this shape (section 3's table):

| core | FLOPs | ms | TFLOPS |
| --- | ---: | ---: | ---: |
| `attn1`, Sol sparse forward (routed) | 2.05e12 | 469.8 | 4.36 |
| `attn1`, dense (layer 0 only) | 4.26e12 | 2,301.9 | 1.85 |
| `attn2`, dense | 2.70e11 | 147.9 | 1.83 |
| A2V core | 1.66e10 | 13.3 | 1.25 |
| V2A core | 1.66e10 | 20.8 | 0.80 |

Against the same shapes run through the **tower's own kernels**
(`tests/h3/sol_forward_bm 16128 32 3 <tau>`):

| kernel | shape | ms | note |
| --- | --- | ---: | --- |
| `h3d_sdpa`, MMA arm | 16,128 x 16,128 x 32 heads | **69.3** | 61.5 TFLOPS |
| `h3d_sol_forward` + search | same, tau 1.00 | 30.5 + 2.5 | 48.0 % of pairs exact |
| `h3d_sol_forward` + search | same, tau 1.25 | 30.4 + 2.5 | 47.5 % exact |
| `h3d_sol_forward` + search | same, tau 1.50 | 29.9 + 2.6 | 46.8 % exact |

Same token count, same heads, same head width, same taus. The tower's sparse
forward is **15.4x** faster than the refiner's (30.5 ms against 469.8), and the
tower's dense attention is **33x** faster than the refiner's (69.3 ms against
2,301.9 at the same 4.26e12 FLOPs).

The cause is named in the source. The refiner's operands are token-major
(`[tokens, heads, dim]`), so the port re-expressed the attention row-major
rather than transpose every operand per layer: `h3d_ltx_attention_kernel`
(`native/h3/h3_dit_gpu.cu:3222`) is the scalar walk with `H3D_ATTENTION_ROWS`
rows per warp and eight keys at a time, and `h3d_sol_ltx_forward_kernel`
(`:7296`) is the same walk as H3's in the token-major shape. The tensor-core
arm exists only for the head-major layout - `h3d_attention_mma_kernel` (`:4151`)
behind `h3d_sdpa`'s `context->attention_mma` (`:4368`), which is on by default
for the tower. The refiner never reaches it.

This also retires the layer-0 mystery from the first pass: the "1.9 s extra" was
not a dense-vs-sparse policy cost, it was 4.26e12 FLOPs through the scalar
kernel at 1.85 TFLOPS.

## 4. What the fix would buy, and what it would cost

A block is 830.6 = 651.7 of attention cores + 130.9 of projections + 48.0 of
everything else (norms, rotary, the Sol search, the AdaLN and residual stream,
the per-head gates). Leave the last two terms alone and price the cores at the
tower's rates - 30.5 for the sparse forward, 5.4 for the text cross-attention at
61.5 TFLOPS, 3.0 for the two AV cores:

    130.9 + 30.5 + 5.4 + 3.0 + 48.0  ~=  218 ms

against 830.6, i.e. 121.8 s of denoise becomes ~32 s and the whole recipe
(248.5 s) becomes ~159 s. That is an estimate, not a measurement: it assumes the
token-major MMA path reaches the head-major kernel's rate, which it will not
exactly, because the strided rows cost something in the operand loads.

The cost is arithmetic, not risk: `h3d_ltx_attention_kernel` is the kernel the
one-block golden was validated through, and the gate holds most of the block
**bit-identical** to the reference (`h3_ltx_refiner_gate.rs:15`). MMA
accumulates a different order, so an MMA arm is a new arithmetic class that
needs its own qualification - `rel_rms` and the picture, not the bit gate - the
same way the FP8 projection route and the Sol route did.

Order of value, by measured share:

1. `h3d_sol_ltx_forward` (56.6 % of a block) - a token-major MMA/blocked walk.
2. `h3d_ltx_attention` dense core (17.8 %) - a token-major MMA dense attention;
   it also covers the dense arm and layer 0.
3. the AV cores (4.1 %) - the same dense kernel at 126 keys.

There is a cheaper intermediate for (2): with 1,024 keys the dense text
cross-attention could transpose K/V (2 x 1024 x 128 x 32 = 16 MB) and use the
existing head-major MMA kernel unchanged, at a 16 MB copy per layer. That is
0.13 ms of traffic at this host's rate to remove 142 ms.

## 5. The first step's weight touch, for the record

Step 0 costs 7.9 s more than step 2 (44.69 against 36.77 s) and the extra is
spread evenly inside the blocks (+164 ms each, 915.4 against 751.2 ms). 37.13 GB
of weights over 48 blocks is 0.773 GB a block, and 0.773 GB in 164 ms is
**4.7 GB/s** - so the first step is the lazy per-block first touch of the shard
at a storage-rate read, and every later step is resident. The `weights_gib=34.58`
line is real but it is not a per-update sweep: there is no 44 s weight bill to
pay twice.

## 6. Artifacts

- `tests/h3/refiner_gemm_bm.cu` and `make tests/h3/refiner_gemm_bm`; log
  `scratch/refiner-gemm.log` on the Spark (34 lines, both classes).
- `H3_BLOCK_STAGES` / `H3_ATTN_PROFILE` in
  `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`; log `scratch/stages-arms.log`
  (4,318 STAGES lines and the step times), script `scratch/stages-arms.sh`.
- `tests/h3/sol_forward_bm 16128 32 3 1.0|1.25|1.5`; log `scratch/solbm.log`.
- Run: `outputs/prof-stages.mp4`, `STAGE2 step=0/1/2 = 44.691 / 40.322 / 36.768 s`,
  denoise 121.781 s, VAE decode 15.215 s - within 1 % of the pre-instrument
  run (`44.447 / 40.104 / 36.537`, 121.088 s), so the laps cost a lap's syncs
  and nothing else.

## 7. What this withdraws

`refiner-update-anatomy-2026-11-01.md` §4 concluded that the gap was "either
the shape/heuristic the port asks for, or the number of calls each block makes
around the GEMMs", and that the published bar "needs the GEMMs at 5x the rate
the blocks currently achieve". Both halves are withdrawn: the GEMMs are at
92.51 TFLOPS, inside cuBLAS's class and above its quoted band, and the 5x is
owed by the attention. The 414-600 TFLOP-per-update figures in that table were
GEMM-only while the time was whole-block; the right table is section 2 here.
