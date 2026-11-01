# The Gemma4 encoder's INT8 convrot route, built and gated (2026-10-29)

Unit: the first step of the Gemma4 port
([`gemma-connector-port-plan-2026-10-29.md`](gemma-connector-port-plan-2026-10-29.md),
section 5, unit 1) - the INT8 convrot linear the release serves Gemma-4-12B
through. The verdict the next unit needs: **the op is in
`native/h3`, every step but the rotation is bit-exact against a host
transcription of the reference, the route costs rel RMS 1.3% against the BF16
linear it replaces, and the release's own checkpoint weights are reproduced
from their BF16 source with 0-6 INT8 elements per tensor differing by one.** The
gate also settled a question the source leaves open: the *offline weight*
conversion ran with **F32** operands, not the BF16 a served activation uses.

Commit: `b234a72d`. Workstation `vizzio` (RTX 4070 SUPER,
sm_89, CUDA 13.3), Spark `gx10-e1d2` (GB10, sm_121a, CUDA 13.0, no
`nvidia-clocks.service`, so its timings are indicative and not clock-pinned).
The artifact is `/mnt/h3-artifacts/Gemma4/` on the Spark.

## 1. The reference, and what it does not say

`comfy-kitchen` 0.2.26, unpacked read-only at `/data/refs/comfy-kitchen-v0.2.26`;
the release calls it from `runtime/cache_ops/int8_policy.py:58-107`
(`DevInt8ConvRotLinear.forward`), which passes `convrot=True`, the layer's
`convrot_groupsize` and an output dtype code. Five steps, each cited where it is
implemented:

| Step | Reference |
|---|---|
| grouped Hadamard rotation of the activation | `tensor/int8_utils.py:55-70` |
| row-wise absmax and F32 scale | `backends/eager/quantization.py:857-871` |
| row-wise INT8 quantization in the operand's dtype | same lines, with `_int8_scale_for_math` `:815-817` and `_round_int8` `:820-826` |
| int8 x int8 -> int32 accumulation | `:749-781`, `:1021` |
| F32 scale epilogue, BF16 cast, bias last | `:1006-1051` |

No Python runs in this campaign, so the port plan's "eager reference" is
replaced by two things that need none: a host transcription of those five steps,
each carrying its citation, and the release's own artifact, whose INT8 weights
and F32 scales were produced by exactly this code path (section 5).

The source also leaves one thing open. `_int8_scale_for_math` casts the scale to
the operand's dtype before the division, so the divisor and the quotient are
BF16 for a BF16 operand - and the served activation *is* BF16. Whether the
offline conversion ran that way or with an F32 operand is not stated anywhere
the port can cite, so the gate measures it rather than assuming it.

## 2. What was built

`native/h3/h3_dit_gpu.{cu,h}`, beside the FP8 route and the same shape: the
kernels, then the entry points, with `h3d_linear_int8_convrot` as the chain.

- `h3d_int8_hadamard` - the sign matrix, one bit per entry, `ceil(group/32)`
  words per row;
- `h3d_int8_rotate` - the grouped rotation, one block per `(row, group)`, one
  thread per output column, F32 accumulation in index order;
- `h3d_int8_scale_rows` - the row absmax (the FP8 route's own per-row reduction,
  reused) and `clamp(absmax / 127, min=1e-30)`;
- `h3d_int8_quantize_rows` - the BF16 divisor, the BF16 quotient, `round_` (half
  to even) and the clamp;
- `h3d_linear_int8_raw` - cuBLASLt `CUBLAS_COMPUTE_32I`, row-major TRANSB, int32
  output;
- `h3d_int8_scale_epilogue` - `bf16(acc * (row_scale * column_scale))`, the
  reference's own association, then the existing `h3d_linear_bias_kernel` for
  the bias.

`struct h3d_int8_plan` carries the workspace, the way `h3d_sol_plan` does for the
search step. `H3D_I8` and `H3D_I32` join the dtype enum; the DiT's own int8 MLP
stays unported, and the header's named deviation says so.

## 3. The Hadamard matrix as sign bits

`_build_hadamard` divides the Kronecker power of `h4` by `sqrt(size)`, and
`size` is a power of 4, so every entry is exactly +/- a power of two. Storing
the signs and applying the one scale at the end is therefore not an
approximation: summing `+x` or `-x` and multiplying by `2^-k` is the reference's
own accumulation, term for term and rounding for rounding, because scaling an
F32 sum by a power of two commutes with each rounding. The harness checks the
sign bits against a host Kronecker build for sizes 4, 16, 64 and 256 - all four
pass - and the artifact gate checks the whole matrix against the release's own
weights (section 5), which is the check that a sign error could not survive.

## 4. Gate 1: the native harness

`tests/h3/int8_convrot_bm` (`make CUDA_ARCH=<arch> tests/h3/int8_convrot_bm
[rows] [k] [n] [group]`). Deterministic operands, no model; the weight is built
the way the converter would have: the source weight is rotated in groups in the
host oracle and then quantized row-wise, so the route's own claim
`(H A) . (H W) = A . W` is what the last line measures.

| Step | Result |
|---|---|
| Hadamard sign bits, sizes 4/16/64/256 | exact against the host Kronecker build |
| rotation | bit-exact against the host F32 order (0 of 3,932,160); 2 of 3,932,160 differ from the exactly rounded (double) answer by one BF16 ulp |
| row scales | 0 of 1,024 differ |
| INT8 values | 0 of 3,932,160 differ |
| int32 accumulate | 0 of 4,194,304 differ against a 64-bit host dot |
| epilogue | 0 of 4,194,304 differ, bit for bit |
| route vs the source BF16 linear | rel RMS 1.33% (1.33% at 3,840x15,360, 1.43% at 15,360x3,840, 1.22% with one row) |

The same run on the Spark returns the same deviations to four digits
(1.333/1.334/1.430/1.222%), which is the point: neither host's arithmetic is in
the result. Per call, at a 1,024-row prompt:

| Shape (rows x k x n) | rotate | scale+quantize | int8 matmul | epilogue | chain |
|---|---|---|---|---|---|
| 1024 x 3840 x 4096, sm_89 | 0.672 | 0.027 | 0.173 | 0.016 | 0.913 |
| 1024 x 3840 x 4096, sm_121 | 1.072 | 0.035 | 0.260 | 0.076 | 1.469 |
| 1024 x 3840 x 15360, sm_89 | 0.878 | 0.027 | 0.507 | 0.229 | 1.544 |
| 1024 x 3840 x 15360, sm_121 | 1.080 | 0.035 | 0.936 | 0.429 | 2.692 |
| 1024 x 15360 x 3840, sm_89 | 2.665 | 0.081 | 0.832 | 0.015 | 3.444 |
| 1024 x 15360 x 3840, sm_121 | 4.300 | 0.377 | 1.142 | 0.026 | 5.725 |

Milliseconds, median of three after a discarded first. The rotation is the
largest single piece, and by construction the only one with a summation order
to choose; a fast Walsh-Hadamard transform replaces it with four radix-4
butterfly stages at a different rounding, which is a named deviation to measure
when the encoder's own cost matters (section 7).

## 5. Gate 2: the release's own weights

`tests/h3/test_int8_convrot.rs` (portable, `make tests/h3-suite`; `H3_GEMMA_DIR`
or the first argument names the directory, exit 3 is a skip). It reads the two
released files - the BF16 Gemma-4-12B shard, 686 tensors,
and the INT8 sibling, 1,342 - and for two layers (0 and 47) and all seven
quantized linears does two things:

- **reproduce**: `_rotate_weight` then `quantize_int8_rowwise` on the BF16
  source, compared with the checkpoint's I8 bytes and F32 scales;
- **round trip**: dequantize the checkpoint, un-rotate by the same matrix
  (`H H = I`), and compare with the BF16 source. A wrong matrix leaves the two
  uncorrelated - a residual near 1, not near the measured 0.9%.

Measured on 13 tensors, ~466M INT8 values:

| Quantity | Result |
|---|---|
| INT8 values equal to the checkpoint's | all but 0-6 per tensor, |d| = 1 |
| F32 scales | equal to within one F32 ulp (worst rel 1.2e-7) |
| round trip, checkpoint vs source | rel RMS 0.88% - 1.22%, the INT8 quantum |
| round trip, this gate's own values | the same figures, to five digits |

The round trip is what makes the gate load-bearing: it exercises the Hadamard
signs, the group geometry and the rotation direction on real weights, and the
residual it reports is the quantization error the format is supposed to have.
The un-rotated reconstruction of the checkpoint's own weights matching the
source at 0.88% says the port and the release agree on the transform; the
0-6 single-step differences say the remaining disagreement is the rotation's
summation order, which the artifact does not record. Run on the Spark against
the local copies (`/mnt/h3-artifacts/Gemma4`, no mount in the path), the gate
returns the same flip counts and the same residuals to five digits, so neither
the host nor the mount is in the result.

**The convention the artifact pinned.** Four candidate conventions were measured
on each tensor, differing in whether the rotation's result is rounded to BF16
before the quantization, whether the divisor is the BF16 rounding of the scale,
and whether the stored scale is that BF16 value:

| Rotation | Divisor | Stored scale | INT8 equal | Scale |
|---|---|---|---|---|
| BF16 | BF16 | F32 | 0.911 | a BF16 ulp off (rel 3.8e-3) |
| BF16 | BF16 | BF16 | 0.911 | a BF16 ulp off |
| BF16 | F32 | F32 | 0.949 | a BF16 ulp off |
| **F32** | **F32** | **F32** | **1.0000** | **within an F32 ulp** |

So the converter quantized an **F32** model: the rotation's F32 accumulation is
not rounded to BF16, and neither the divisor nor the stored scale is. That is
`quantize_int8_rowwise` with `x.dtype == F32`, which is what loading the
checkpoint as F32 and rotating it produces. The BF16 row is kept in the gate
because it is what a *runtime activation* needs - a served activation is BF16,
so `_int8_scale_for_math`'s cast is real there, and the kernel implements that
row. The two paths differ, and the file that decides which is which is the
artifact, not the reading.

## 6. Deviations, named

- **No Python.** The plan's "bit-compare against the eager reference" is met by
  a host transcription of it (section 4, every step but the rotation bit-exact)
  plus the release's own artifact (section 5). The CUDA backend's own spread
  against eager could not be measured, because running it needs the interpreter
  the operator forbade; what replaces it is the exact-fraction and the F32-ulp
  scale agreement above.
- **The rotation's summation order.** The reference accumulates through cuBLAS
  in a tile order the artifact does not record; the port accumulates in index
  order. Measured cost: 2 of 3,932,160 rotated values differ from the exactly
  rounded answer by one BF16 ulp, 0-6 INT8 elements per tensor differ by one,
  and the route's deviation from the BF16 linear is unchanged to four digits.
- **M padding only.** `_int8_matmul_accumulate` zero-pads M, K and N to
  cuBLASLt's alignments; the port pads M exactly as the reference does and
  refuses a K or N that is not a multiple of 8, which every INT8 linear in the
  artifact satisfies (K is a multiple of the group size).
- **The sign-bit Hadamard.** A representation change, argued exact in section 3
  and checked against the artifact.
- **The group is capped at 1,024** (`H3D_INT8_GROUP_MAX`), a power of 4 and at
  most one warp-block; the artifact's `comfy_quant` blobs all say 256.

## 7. What the next unit uses

Unit 2 of the plan, the 48-layer Gemma4 text tower, calls
`h3d_linear_int8_convrot` through `ds4-sys` once; that FFI declaration and the
per-layer plan allocation belong to it. Two properties this unit establishes
that the tower depends on: the quantization is BF16-valued on the *activation*
path (section 5's BF16 row), and every weight-side number the port needs is
already in the checkpoint, so nothing rotates a weight at run time.

Open, and cheap to close when the encoder's cost matters: the rotation at
1.1-4.3 ms per projection is the largest piece of the chain, and the
fast Walsh-Hadamard form would replace 256 adds per output with four radix-4
stages at a different rounding - a named, measurable deviation, not a free one.
