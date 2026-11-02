The four-warp block left 1.24-1.53x on the table at the delivery shape. This
unit closes it: **536.8 ms per call at S = 38,900 against 703.4 ms, -23.7%,
61.7 -> 80.8 TFLOPS, output bit-identical**, and inside 1.25x of stock at both
ends of its 458-565 ms range. The block is now eight warps x 16 rows with a
three-deep `cp.async` ring in dynamic shared memory, and it is the default;
`H3_DIT_ATTENTION=four` names the old one.

## 1. Why the four-warp block could not get there

Two numbers bound it, and neither is the arithmetic.

**Traffic.** A CTA covers `H3D_MMA_ROWS = 64` query rows and reads *all* of the
head's K and V, which at S = 38,900 is 38,900 x 128 x 2 B x 2 matrices = 19.9 MB
per CTA. There are 38,900/64 = 608 CTAs per head and 56 heads, so a call reads
**677 GB**. One head's K+V is 19.9 MB against a 24 MB L2, so it is L2 traffic,
and at 703 ms that is 963 GB/s sustained. Doubling the rows a CTA covers halves
it.

**Registers.** 171 registers x 128 threads x 3 CTAs = 65,664 is already past the
65,536 an SM has, so the four-warp block runs 3 CTAs only because ptxas lands at
171 and the third CTA is the last one that fits. There is no room left for the
compiler to hold a prefetched `ldmatrix` fragment across an `mma`, because the
per-thread budget is what caps the CTA count in the first place. The accumulator
alone is `result[16][4]` = 64 registers and cannot shrink: `mma.m16n8k16` forces
16 query rows per warp and 128 head dimensions means 16 n8 slices.

An eight-warp CTA of 128 rows changes both: half the traffic, and 256 threads
x up to 255 registers = 65,280 still fits *one* CTA, so the register file stops
being a constraint and becomes headroom.

## 2. The change

`h3d_attention_mma_pipe_kernel` (`h3_dit_gpu.cu`), next to the block it
replaces:

- **8 warps x 16 rows = 128 query rows per CTA**, so `grid.x` is 304 instead of
  608 and the K/V traffic per call halves to 338 GB.
- **A three-stage ring** of K and V tiles, `cp.async.cg.shared.global` 16-byte
  copies with a per-tile `commit_group`. The prologue fills two stages; each
  iteration issues the tile `stages - 1` ahead, `cp.async.wait_group 2`, then one
  `__syncthreads()`. The load for a tile was issued two tiles earlier, so the
  barrier waits for copies that have already landed instead of for a
  load-then-sync.
- **Dynamic shared memory**, 3 x 2 x 64 x 128 bf16 = 98,304 B, because the ring
  is past the 48 KB static limit. `h3d_pipe_admit()` raises the CTA ceiling once
  per process with `cudaFuncSetAttribute` and retires the arm on failure.
- **A tail tile stages itself.** `cp.async`'s optional `src-size` operand
  zero-fills, so a key row past the end of the sequence costs `bytes = 0`
  instead of a branch, and the zeroed scores are masked exactly as before.
- **Unchanged**: the 64-key tile, the k16 `mma` order, the Q fragments, one
  maximum per tile, F32 scores/sums/accumulator, BF16 P. That is why the output
  does not move.

ptxas on the Spark: **208 registers, 0 spills, 0 bytes stack**, one CTA per SM,
8 warps. Measured, not derived — `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
in `~/ds4-test/occ.cu`:

| block | registers | smem/CTA | threads | CTAs/SM | warps/SM |
|---|---:|---:|---:|---:|---:|
| four-warp | 168 | 32,768 B static | 128 | 3 | 12 |
| pipelined | 208 | 98,304 B dynamic | 256 | 1 | 8 |

So the pipelined block wins **with a third fewer warps**, which is the useful
thing to know about it: the traffic and the ring matter more than the warp
count, and 208 registers is what leaves the second CTA out (2 x 208 x 256 =
106,496 of 65,536). Dropping to 128 registers would admit two CTAs and 16 warps;
that is the next lever if this one is ever revisited.

## 3. Measured

Same binary, same tensors, alternating arms (`tests/h3/attn_bm`, 56 heads, three
calls each):

| S | four-warp | pipelined | delta | four TFLOP/s | pipe TFLOP/s |
|---|---:|---:|---:|---:|---:|
| 8,192 | 26.6 ms | 23.2 ms | -12.9% | 72.2 | 82.9 |
| 16,384 | 108.7 ms | 94.5 ms | -13.1% | 70.8 | 81.5 |
| 24,576 | 247.0 ms | 213.5 ms | -13.5% | 70.1 | 81.1 |
| 32,768 | 477.1 ms | 379.9 ms | -20.4% | 64.5 | 81.0 |
| **38,900** | **703.4 ms** | **536.8 ms** | **-23.7%** | **61.7** | **80.8** |
| 44,992 | 940.5 ms | 721.1 ms | -23.3% | 61.7 | 80.5 |

The shape of the table is the point. The four-warp block's rate *falls* with S
(72 -> 62 TFLOPS), which is the traffic term above: at 8,192 tokens a CTA's K+V
is 4.2 MB and the whole working set is L2-resident; at 38,900 it is not. The
pipelined block is **flat at 80-83 TFLOPS across every shape**, because halving
the traffic took the capacity pressure out and the ring took the stall out.

Against the stock range in `rapid-h3-study-2026-11-01.md` (458-565 ms at this
shape) the gap goes from 1.24-1.53x to **0.95-1.17x**. The unit's 1.25x bar is
met at both ends.

## 4. Bit-identity

`attn_bm`'s FNV over the output tensor, both arms, all six shapes - these are
the same six values the previous unit recorded:

| S | out_fnv |
|---|---|
| 8,192 | f22ed4407cfca740 |
| 16,384 | 098fc30450b0c823 |
| 24,576 | 2b738cbdcdd489af |
| 32,768 | 924425d4d8705852 |
| 38,900 | 2e63ecb41f0771db |
| 44,992 | 81b9a2b4c6480b49 |

The engine gate does **not** corroborate this, and that is worth stating plainly:
`h3_dit_gate` pins the scalar arm (`attention: Attention::Scalar`,
`h3_dit_gate.rs:277`) because its goldens are the reference engine's exact
hashes, and `dit.rs:1822` turns that into `h3d_set_attention(context, 1)`, which
clears both tensor-core flags. So the `FORWARD video=cb7bbf...` line that the
previous unit quoted as an A/B of two builds was an A/B of the scalar kernel
with itself, and so was this unit's first attempt at the same check — both arms
printed the same hash because both arms ran the scalar path. The handoff's own
guardrail ("a compute-mode arm can be inert, and the timing alone will not say
so") caught this the second time it mattered.

Engine-level coverage of the tensor-core blocks has to come from a path that
sets `Attention::Tensor`: `h3d_set_attention(context, 0)` sets both flags
(`attention_mma = 1`, `attention_pipe = 1`), which is what `h3-rs` and
`tests/h3_engine_gate` do, and what this unit renders with.

## 5. The bug this found, and how it was localized

The first build was 23.9% faster and its hash differed from 96.8% of the
previous output at rounding scale (max_abs 8.1e-4, rms 1.8e-4). It was not a
race: two runs of the same arm agreed bit for bit, and the dump comparison
showed the difference spread over the tensor rather than concentrated at a
boundary - which is what a *systematically wrong tile* looks like, not a
boundary error.

Bisecting on sequence length found it in one pass: S = 64 and 128 matched (one
and two tiles, all of them filled by the prologue), and S = 192 differed - the
first length whose third tile is filled inside the loop. The look-ahead was
`base + H3D_MMA_KEYS` where it had to be `base + (stages - 1) * H3D_MMA_KEYS`:
the loop re-staged the tile the prologue had already filled and never staged the
tile two ahead, so from the third tile on the block computed against stale keys.

The lesson is the cheap one: with a ring buffer, a **sequence-length sweep**
converts "the numbers differ" into "the third tile is wrong" in three runs. An
eye on the diff would not have.

## 6. The delivery canvas

The delivery job (`outputs/turbo-delivery/run.log`: 1344x768, 121 frames, 6
steps, seed 42, crf 15 preset slow, bf16i decoder, the Turbo-v4 checkpoint)
rendered three times with one binary, alternating arms, each into its own
directory:

| | denoise (6 steps) | per step | wall | clip md5 |
|---|---:|---:|---:|---|
| four-warp, `outputs/four-attn` | 317.6 s | 52.9 s | 434 s | `922e2ce81b14...` |
| pipelined, `outputs/pipe-attn` | **270.1 s** | **45.0 s** | **386 s** | `922e2ce81b14...` |

**-15.0% of the denoise, -11.1% of the wall, and the two clips are
byte-identical** (`922e2ce81b14382b68223db184d5f540`) - which is the whole
pipeline, denoise through VAE decode and mux, not just the kernel's tensor.

The share moves as the kernel number predicts, which is the check the unit was
missing: 50 attention calls per evaluation at 0.703 s is 35.2 s of the 52.9 s
four-warp step (66%, the 69% `nsys` once measured), and at 0.537 s is 26.8 s of
the 45.0 s pipelined step (60%). Every second the kernel gave back landed in
the step and nowhere else.

**A trap this cost a render to find.** The first "pipelined" render measured
376.0 s of denoise — worse than the recorded 382.5 s within noise, and nothing
like the kernel delta. It had run `target/release/h3-rs`, which was still the
binary from the last session, built before this kernel existed: both arms of
that first A/B were the four-warp block, and the 18% spread between them was
session variance. **A render only tests a kernel if the binary was rebuilt after
the source changed** - check the two modification times before believing a
stage time, and prefer one binary with two arms over two binaries.

## 7. What is left

- The `fp8` GEMM arm at this canvas, which the study measured at 212 against 99
  TFLOPS and which has never run at the delivery shape.
- The decode, 390 s of the wall before this unit and now the largest single
  block of it.
- `h3d_attention_mma_f32_kernel`, 1,890 launches per step at 1.17 ms.
- The four-warp block stays as the named arm; if nothing needs the A/B again it
  can go the way of the SOL kernels.
