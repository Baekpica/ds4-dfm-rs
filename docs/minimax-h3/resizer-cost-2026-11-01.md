# The resizer's cost, split three ways (2026-11-01)

The latent resizer cost 16.5 s for a 32 x 34 x 58 draft, and the handoff's
reading was that the im2col owned it: TF32 moved nothing, so the wall had to be
bandwidth. That reading was wrong on both counts. This is the receipt, from the
instrument that produced each number to the fix it justified.

## What each instrument said

`h3d_conv3d_timing` (cudaEvents inside `h3d_conv3d_gemm`, recorded in the work's
own stream) put 11.03 s of the 16.6 s wall inside the convolution: im2col
2.29 s, GEMM 8.62 s, bias 0.13 s. So the im2col was 21%, not 90%, and the GEMM
was the single largest item.

That left 5.5 s unaccounted, which `upscale_timed` (a lap per stage, each with a
synchronize) placed entirely inside the two block runs: `blocks_out` 12.98 s of
a 16.6 s call, against 12 of the graph's 18 convolutions.

`nsys profile` named it. Per-kernel time over two runs at 16 x 18 x 30:

| share | kernel | calls | ms each |
|---|---|---|---|
| 69.5% | `magma_sgemmEx_kernel` (SIMT SGEMM) | 300 | 7.73 |
| 18.8% | `h3d_im2col3d_kernel` | 350 | 1.79 |
| 2.2% | `cutlass_80_simt_sgemm_64x64` | 52 | 1.44 |
| 1.5% | `h3d_group_norm_partial_kernel` | 122 | 0.42 |
| 1.3% | `h3d_zero_pad3d_kernel` | 100 | 0.45 |
| 1.3% | `h3d_group_norm_apply_kernel` | 122 | 0.36 |

and `nsys stats -r cuda_api_sum` on the host side: **cudaFree 910 calls, 2.27 s;
cudaMalloc 910 calls, 1.29 s**. Those are the missing seconds: both calls
synchronize the device, and the resizer allocates and frees a 500 MB buffer five
or six times per block.

Two facts fell out that no host timer could have shown:

- The GEMM was a **SIMT** kernel. `CUBLAS_COMPUTE_32F_FAST_TF32` had been inert
  for the whole branch: the row-major + `TRANSB` heuristic answers this shape
  with `magma_sgemmEx` and ignores the compute mode, which is why the TF32 arm
  measured 16.75 s against 16.34 s and looked like proof that precision was not
  the lever.
- The im2col ran at 1.79 ms per 453 MB tile, 253 GB/s, against the 273 GB/s the
  part specifies. It was already at the memory-system floor for a 27x expansion,
  so no rewrite of it was going to help; only not doing it would.

## The three changes

**The pool, not the driver** (12.3 s, -4.3 s). `h3d_alloc` now allocates with
`cudaMallocAsync` on the context's stream and records it, `h3d_free` hands the
block back to that stream's pool, and `h3d_create` raises the default pool's
release threshold so a freed block waits for the next caller. A tensor's free
asks a small table of live streams first: a Rust struct frees its fields in
declaration order, so a context can be destroyed before the tensors made from
it, and an ordered free on a destroyed stream is a use-after-free. The first
version of this crashed at teardown for exactly that reason.

**Column-major** (7.5 s, GEMM 8.77 -> 3.97 s). `C[R, dim] = A[R, k] * W[dim, k]^T`
is byte for byte the column-major product `C'[dim, R]`, because a column-major
`[dim, R]` buffer read row-major is `[R, dim]`. Asking the transposed question -
`TRANSA` on the weight, no layout attribute at all, writing into the caller's own
output buffer - is what got a tensor-core kernel in the build that introduced it,
where the row-major arm was still dispatching `magma_sgemmEx`.

**And that dispatch is not a property of the code.** Re-measured two commits
later, with `h3d_linear_lt_compute`'s source byte-identical, the row-major TF32
arm also reaches tensor cores: `nsys` on it now shows
`cutlass_80_tensorop_*` at 50.3% and no MAGMA kernel at all, and its GEMM is
3.97 s against the 8.77 s it measured in the earlier build. The im2col did not
move (2.29 s in every run), so this is the GEMM's kernel choice and not a clock
or memory-bandwidth difference. cuBLASLt is being asked with a NULL algorithm
and answering from a heuristic whose inputs are not all in this repository.

| arm (6097f90c) | wall | im2col | GEMM |
|---|---:|---:|---:|
| fp32 rows | 12.45 s | 2.31 | 8.88 |
| tf32 rows | 7.52 s | 2.31 | 3.97 |
| tf32 columns | 7.51 s | 2.31 | 3.97 |

The consequence for the product is not "use columns", it is **stop asking a
heuristic**: `cublasLtMatmulAlgoGetHeuristic` returns the candidates, the port
should keep them, measure them once, and record which one it ran. Until then a
2x swing in the most expensive phase can happen between two builds with no code
change, which is exactly what this table shows. Columns stays the default for
the narrow reason that it is the orientation that produced the tensor-core
dispatch in the build where the row-major one did not.

**GroupNorm over the whole grid** (no measurable win; 0.42 + 0.36 ms per pass).
It launched one block per (batch, group): 32 blocks for a 252k-voxel volume, each
striding its voxels by the block width. The rewrite - partial sums per (batch,
group, part) in double, a finalize that folds mean and rstd into a
per-(batch, channel) scale and shift, and a coalesced apply that can emit the
following SiLU - is now 363 GB/s instead of 33 GB/s on the same volume, and it
retired a separate SiLU pass. It was 2.8% of device time, not the 5 s the stage
lap's residue suggested: that residue was the allocations all along. The rewrite
stays because it is strictly better and it is the kernel the reference's own
`F.silu(norm(x))` order deserves, but it is not what closed the gap.

## The four arms, measured

`tests/h3_upscale_gate` now runs every arm against the same f64 reference and
holds each to its own bound:

| arm | time (8 voxels) | relative error |
|---|---|---|
| fp32 rows | 156.9 ms | 2.216e-6 |
| fp32 columns | 24.4 ms | 2.216e-6 |
| tf32 rows | 13.6 ms | 6.053e-5 |
| tf32 columns | 13.2 ms | 6.053e-5 |

`fp32 columns` reproducing `fp32 rows` to the digit is the check on the layout
argument: if the byte reinterpretation were wrong the two would differ. The TF32
arms land 27x looser than F32 and 8x inside the 5e-4 bound, which is the
reference implementation's own precision class - it computes these convolutions
in F16, ten mantissa bits, the same as a TF32 product.

At 32 x 34 x 58 the default arm is now **7.5 s**: im2col 2.30, GEMM 3.97, bias
0.13, everything else 1.11 - from 16.5 s at the start of the session.

## What is left

The GEMM is 107 TFLOP in 3.97 s, 27 TFLOPS. Traffic per convolution is 4.4 GB
(two thirds of it the expanded columns), so at 53 GB/s it is not bandwidth-bound;
it is a TF32 kernel with a narrow N (512) and the next lever is a wider operand
class, not a wider tile. `tile` at 16384 rows would need 906 MB of scratch, which
is not a trade this port should make while the DiT shares the device.

So the next two units, in order:

1. **F16 operands** (`H3_UPSCALER_GEMM=f16`): the checkpoint is F16 already, so
   the weight needs no widening, the columns are written as half (im2col 2.30 ->
   1.15 s) and the GEMM is F16 x F16 with F32 accumulate (`CUBLAS_COMPUTE_32F`,
   exactly what the reference does). Expect ~4.4 s total. The gate needs a
   fifth arm and its own bound, and the comparison that matters is against an
   F16 reference, not the f64 one: record both.
2. **The direct convolution**, which drops the im2col entirely (step 2 of the
   handoff's plan): a CTA owning `[64 voxels x 128 output channels]` reading
   A-tiles straight out of the padded volume, because in channels-last a run of
   successive `x` voxels at fixed `(t, y)` and one tap is contiguous and
   `out_width` is 68 or 116, so a 32-64 voxel tile never crosses a row. That is
   where the last 1.15-2.30 s lives.

## The lesson this unit bought

**A compute-mode arm can be inert, and the timing alone will not say so.** The
TF32 arm was measured, reported, committed and reasoned about for a day while
cuBLASLt dispatched the same SIMT kernel either way; the arm's own number
(16.75 s against 16.34 s) even looked like evidence that arithmetic was not the
lever. The kernel's *name* in a profiler settled it in one run. When an arm moves
nothing, name the kernel before drawing a conclusion from the absence of a
difference.
