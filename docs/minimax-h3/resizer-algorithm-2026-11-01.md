# The resizer's last 4 seconds: name the kernel, then halve the operands

Unit: `feat/rapid-h3`, 2026-11-01. Follows
[`resizer-cost-2026-11-01.md`](resizer-cost-2026-11-01.md), which took the
stage from 16.5 s to 7.5 s at 32 x 34 x 58 and left two levers: F16 operands
and the direct convolution. This unit took the first, and had to do the step
before it to make the first measurable.

**Result: 7.49 s -> 3.89 s**, gate PASS on five arms, `im2col` 2.29 -> 1.16 s,
`gemm` 3.95 -> 1.49 s. The target for this unit was under 4 s.

## Step 0: the algorithm, named

`h3d_linear_lt_compute` and `h3d_linear_lt_columns` passed a NULL algorithm, so
cuBLASLt ran its heuristic on every one of the ~230 tile products and took
whatever came back. The previous unit recorded that the same source dispatched
`magma_sgemmEx_kernel` in one build and `cutlass_80_tensorop_*` in the next and
concluded the dispatch was not stable. With the choice pinned per shape and its
id printed, the honest answer is more specific: **the heuristic is stable, and
it answers the two formulations differently.**

| shape | arm | candidate | id | tile | stages | split-K | kernel class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8192x13824x512 | f32, rows | 0 | 16 | 0 | 0 | 1 | SIMT (`magma_sgemmEx`) |
| 8192x13824x512 | f32, tf32, columns | 0 | 21 | 23 | 3 | 3 | cutlass tensor op |
| 8192x13824x512 | f16, columns | 0 | 67 | 32 | 35 | 1 | half tensor op |
| 63104x512x512 | f32, rows | 0 | 16 | 0 | 0 | 1 | SIMT |

`id=16` carries no tile, no stages and no split-K: it is the fallback SIMT
product, and the heuristic chooses it for the row-major TRANSB question every
time. `id=21` and `id=67` are the tensor-core families. So the earlier
"instability" was two different questions being asked in different builds, not
a coin toss - and it was invisible because nothing named the answer.

The mechanism, for the next person:

- `cublasLtMatmulAlgoGetHeuristic` is called once per (context, shape, type,
  compute class, orientation), the answer is kept in a 64-entry table, and the
  algorithm is passed to `cublasLtMatmul` explicitly. `h3d_lt_algo_report`
  prints what each shape got.
- The preference raises `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` from
  cuBLASLt's default of **zero** to 64 MiB. The default silently excludes every
  split-K and stream-K candidate before the search begins; a candidate that
  still asks for more than the ceiling is skipped, and the next one is taken.
- `H3D_LT_ALGO=<i>` selects the i-th candidate without a rebuild,
  `H3D_LT_ALGO_TRACE=1` traces each choice, and the bench and the gate print
  the report beside the arm.

Two bugs the report found in its first two runs, both worth recording because
neither shows up as a wrong number:

1. **A candidate whose workspace did not fit ended the search.** The pointwise
   `63104x512x512` and `252416x512x512` products read `cand=-1`: their first
   candidate wants 129 MB of split-K workspace, past the ceiling, and the loop
   treated "cannot afford" as "no answer" and fell back to NULL. Skipping to
   the next candidate is the right answer.
2. **The row-major path asked the heuristic with A and B swapped.** It passed
   `(weight_layout, input_layout)` while the multiply it then ran was
   `(input, weight)`. A and B are not interchangeable to the heuristic - the
   candidate list is filtered by operand order and transpose pair - and the
   row-major shapes came back with nothing usable at all. With the order fixed
   they get `id=16`.

## Step 1: the operands, halved

The checkpoint is F16. The F32 arms were widening it, doing 27x-expanded F32
im2col, and asking the tensor cores to round F32 to TF32 on the way in.

- `H3D_F16` joins the dtype enum.
- The permuted weight is never widened: `permute_kernel` is a gather, so it
  runs on the raw half words and the upload touches no bit. This is also why
  the load drops from 1.02 s to 0.44 s.
- The zero-pad kernel is templated on its destination and is where the
  activation is rounded to F16, in the same pass that pads it - no extra
  buffer, no extra pass, and the 27x-expanded buffer is the smaller one.
- `h3d_conv3d_gemm` takes the operand class from the tensors it is handed and
  runs the `__half` im2col with `CUBLAS_COMPUTE_32F`. Accumulation and output
  stay F32, so nothing downstream of the convolution changes.
- The 1x1x1 "convolutions" are pointwise products that go through the VAE's
  F32 linear, so their weight stays F32 on this arm.

F16 the *compute type* by itself buys nothing - `CUBLAS_COMPUTE_32F_FAST_TF32`
beside F16 operands is a rounding with nothing to round, and the report naming
the arm "f16 tf32" was the tell. Asked for `CUBLAS_COMPUTE_32F` it measures the
same 3.88 s and says something true.

## The five arms

At 32x34x58 (63,104 latent voxels in, 252,416 out), relative error against the
f64 reference, gate bound 5e-4 for every arm:

| arm | run | im2col | gemm | total | relative |
| --- | --- | --- | --- | --- | --- |
| fp32 rows | 157.8 ms (gate shape) | - | - | 12.3 s | 1.9e-7 |
| fp32 columns | 16.7 ms (gate shape) | - | - | 12.3 s | 1.9e-7 |
| tf32 rows | 13.6 ms (gate shape) | - | - | 12.3 s | 6.2e-5 |
| tf32 columns | 13.0 ms (gate shape) | 2.33 s | 3.51 s | 7.11 s | 6.2e-5 |
| **fp16 columns (default)** | 90.0 ms (gate shape) | 1.16 s | 1.49 s | **3.89 s** | 6.5e-5 |

The gate-shape column is the small end-to-end pass the gate times, 8 voxels in
and 4x4 out; it is not comparable across shapes, only within a row.

Stages on the F16 arm, 3.886 s total: `blocks_in` 0.730, `blocks_out` 3.033,
`conv_out` 0.075, `conv_in` 0.003, `norm_out` 0.009, `download` 0.028,
`prep` 0.004, `resize` 0.003.

Effective throughput, for the record. Each side is 18 blocks: 12 res blocks
with an `in_layers` and an `out_layers` 3x3x3 product each, and 6 temporal
blocks with a depthwise and a pointwise pass. So the two sides hold 24
out-volume and 24 in-volume 3x3x3 products, `2 * 252416 * 13824 * 512` and
`2 * 63104 * 13824 * 512` MACs each: **107 TFLOP in all**. The GEMM phase ran
at **30.5 TFLOPS on TF32 and 72.0 TFLOPS on F16**. `n` is 512 and `k` is
13,824, and `m` is as tall as the volume, so the tensor cores are close to
their own ceiling and there is no arithmetic-intensity problem left to solve -
what remains is a 27x expansion to avoid.

## What is left

`im2col` is now 42% of the convolution phases (1.16 s of 2.78 s) and 30% of
the whole stage: the direct convolution is the lever that removes it, at the
cost of reading the A-tile from the padded volume instead of from a contiguous
expanded matrix. The pointwise linear is still the row-major F32 path and still
lands on `id=16`; it is about 1 TFLOP against the convolutions' 107, so it is
not worth a route of its own. Neither is needed to hit the target this unit was
set, and the two-stage product route in the handoff is now unblocked.

## Lesson

A measurement instrument that names its own cause is worth its code twice
over. The algorithm report found an operand-order bug in the code that
produced the report, on the second run, without anyone looking for one - and
the previous unit's conclusion ("the dispatch is not stable") was a false
alarm that a single named id would have resolved. When a vendor heuristic
chooses for you, make it say what it chose.
