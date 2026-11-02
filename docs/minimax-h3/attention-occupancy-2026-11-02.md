`nsys` put 69% of a denoise step in attention and the per-call number was 906 ms
at the delivery shape, 1.6-2.0x off a stock kernel. This unit names what that
kernel is doing and buys the first part of the gap: **-23.1% per call at the
delivery shape, 47.9 -> 61.9 TFLOPS, bit-identical output**. Nothing here
changes arithmetic, precision or sparsity.

## What the kernel is, corrected

The record said the block-sparse `h3d_sol_forward` plan was "running what is in
practice a dense attention". That is wrong, and the correction matters: the
delivery route never calls it. `h3d_sdpa` (`h3_dit_gpu.cu:4321`) dispatches to
`h3d_attention_mma_kernel` when `attention_mma` is set, which is the default
(`context->attention_mma = !h3d_attention_scalar_from_env()`, `:291`).
`h3d_sol_forward` is reached only from `dit.rs:2837`, which is gated on
`H3_DIT_SOL`. So the kernel to fix is a plain dense bf16 MMA kernel, and no
sparse plan is involved in the 906 ms.

**The shape is S = 38,900, not 45k.** `attn_bm` at 38,900 measures 905.3-911.6 ms,
which is the profile's 906 ms median; at 44,992 the same benchmark measures
1,221 ms. The 44,992 figure in the study came from Rapid-H3's render, and the
profile's "S ~ 45k" carried it over. Everything below is measured at 38,900
unless it says otherwise.

## Step 0: what it was doing

`ncu` is not available for this: it fails with `ERR_NVGPUCTRPERM`, there is no
password-less sudo on the Spark, and enabling counters needs a driver
parameter, which this campaign's rules exclude. The three numbers the step asks
for come from `ptxas -v`, the device properties, and the benchmark instead.

| | value |
|---|---|
| registers / spills | 168, 0 bytes spill stores and loads |
| static shared memory | 34,816 B |
| block | 128 threads (4 warps x 16 query rows) |
| device | 48 SMs, 65,536 regs/SM, 102,400 B smem/SM, 24 MB L2 |
| **occupancy** | **2 CTAs/SM = 8 warps of 48 (16.7%)** |
| achieved | 47.5-60.0 TFLOPS, 47.9 at the delivery shape |

The kernel was **shared-memory bound, not register bound**: two CTAs take
69,632 B of the 102,400 an SM offers and a third needs 104,448. The register
file would have allowed three (3 x 168 x 128 = 64,512 of 65,536). The binding
resource was the `+8` bf16 pad in `H3D_MMA_LD`: 34,816 B instead of 32,768.

The S sweep says the loop is not simply waiting on cold K/V. TFLOPS falls with
S rather than sitting flat (57.0 at 8,192; 60.1 at 16,384; 50.5 at 32,768), and
one head's K and V at the delivery shape is 19.9 MB against a 24 MB L2, so the
decline is capacity pressure, not a fixed staging cost.

## The change

The padded row is replaced by an unpadded 128-wide row with a XOR swizzle over
its sixteen 16-byte chunks (`h3d_mma_slot`): `chunk ^ (row & 7)`. That is the
same bank rotation the pad bought, for `ldmatrix`'s access pattern - eight
consecutive rows of one chunk column land in eight distinct bank groups.

- shared memory 34,816 -> **32,768 B** per CTA
- registers 168 -> **169**, still 0 spills
- occupancy 2 -> **3 CTAs/SM = 12 warps**
- 3 x 32,768 = 98,304 B (fits 102,400); 3 x 169 x 128 = 64,896 (fits 65,536)

## Measured

`tests/h3/attn_bm`, H = 56, D = 128, non-causal, 3-5 calls per point.

| S | before | after | gain | TFLOPS after |
|---|---:|---:|---:|---:|
| 8,192 | 33.7 ms | 26.8 ms | -20.6% | 71.8 |
| 16,384 | 128.1 ms | 108.6 ms | -15.2% | 70.9 |
| 24,576 | 321.2 ms | 245.5 ms | -23.6% | 70.5 |
| 32,768 | 609.1 ms | 479.4 ms | -21.3% | 64.2 |
| **38,900** | **911.6 ms** | **700.5 ms** | **-23.1%** | **61.9** |
| 44,992 | 1,221.0 ms | 934.9 ms | -23.4% | 62.1 |

**The output is bit-identical.** The benchmark's FNV over the result tensor is
unchanged at all six shapes, before and after:

| S | out_fnv |
|---|---|
| 8,192 | f22ed4407cfca740 |
| 16,384 | 098fc30450b0c823 |
| 24,576 | 2b738cbdcdd489af |
| 32,768 | 924425d4d8705852 |
| 38,900 | 2e63ecb41f0771db |
| 44,992 | 81b9a2b4c6480b49 |

At the delivery shape the gap to the record's stock range (458-565 ms) goes from
1.6-2.0x to **1.24-1.53x**. The unit's 1.25x bar is met at the slow end of that
range and not at the fast end, so this closes part of the gap, not the unit.

## The gate

The benchmark's hash is the kernel's own output; the engine-level check is a DiT
forward through the whole model. Both were run against the committed kernel and
against the working-tree kernel, built separately (`~/ds4-test/gate-ab.sh`):

| | committed (padded) | working tree (swizzled) |
|---|---|---|
| `FORWARD video=` | cb7bbf34526896b8 | **cb7bbf34526896b8** |
| `FORWARD audio=` | e10227686629592c | **e10227686629592c** |
| `video_edge` | 1.148437500 / -1.640625000 | 1.148437500 / -1.640625000 |
| `TEXT fnv` | e389ad74da16deb8 | e389ad74da16deb8 |
| `LAYOUT positions` | b80b23d417768272 | b80b23d417768272 |

The forward is bit-identical, so this is a speed change with no numerics in it.


## What this says to do next

Double buffering cannot be bolted onto the current block shape. Two stages of
K and V is 65,536 B, which is one CTA of four warps - fewer warps than the three
CTAs reached here. It only pays combined with **8 warps x 16 rows (128 query
rows per CTA)**: 8 warps x 32 x 169 = 43,264 registers (fits), 65,536 B of
shared (fits), so 1 CTA of 8 warps with the next tile's loads overlapping the
current tile's math. That is the next experiment, and it is also the block shape
Rapid-H3's kernel uses.

The secondary lever is unchanged and still unmeasured at the delivery canvas:
`h3d_attention_mma_f32_kernel` launches 1,890 times per step at 1.17 ms each
(2.6 s/step), and `H3_DIT_GEMM=fp8` has never run at this canvas.
