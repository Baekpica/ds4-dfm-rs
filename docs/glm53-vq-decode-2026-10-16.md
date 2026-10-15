# GLM VQ direct expert decode

The retained Q4 staging profile spends 3.489 s of 8.002 s decode time
(43.61%) in routed VQ restoration and computation. Each selected matrix
is expanded to a 16 MiB BF16 buffer and then read by cuBLAS. The unchanged
width-one isolated NCU replay recorded 55.2 µs restoration and 97.8 µs
GEMV, with 164 GEMV registers and 23.1% occupancy. Those replay counters
use rotating experts and are not whole-model bandwidth measurements.

## Implementation

Adapt YoungAi `7fb26cc332d740416792204a29c7d13d2986caf5`,
`src/cuda/cuda_vq_row.inc.cu:146-280` and
`src/cuda/cuda_vq_persist.inc.cu:1-210`:

- Read 256 indices as three aligned 128-byte lines and distribute words
  through warp shuffles. The thirteenth bit uses a separate 32-byte plane.
- Load the next index block before consuming the current one.
- Keep the layer-shared E4M3 book in each persistent CTA and compute only
  selected experts. The 13-bit book occupies 64 KiB, without BF16 expansion.
- Fuse gate/up and SwiGLU. Compute down directly, then reduce experts in
  the existing ascending expert order.

The GLM arithmetic differs from YoungAi: gain is applied to each codeword
element before BF16 rounding, rather than after the dot. Gate/up retain
their BF16 boundary; clamp, BF16-rounded SiLU and BF16-rounded product
remain. FP32 dot/reduction order changes. No expert weight, codebook or
nonexpert precision is changed by this kernel.
E4M3 is the codebook storage format; dots use FP32 FMA on restored BF16
values, without FP8 or NVFP4 matrix arithmetic.

Direct decode is enabled by default for single-token calls with
aligned, uniform 13-bit E4M3 GVQ1 matrices of gate/up [2048,4096] and
down [4096,2048]. Small verification batches retain restore+GEMM. Shape,
book-sharing, alignment or shared-memory failures retain the fallback.
Graph capture remains rejected by the existing adapter. Descriptors,
BF16 intermediate rows and partial outputs are reserved in the session
workspace before execution; routing remains on the existing host path.
The additional per-session allocation is 165,056 bytes.
`DS4_GLMVQ_DECODE=0` restores the previous path for diagnostics.

For 42 MoE layers and top-8 routing, gate/up/down packed indices, high
planes and row gains total 1,723,072,512 logical bytes per decoded token.
The previous path additionally wrote and reread 16,911,433,728 bytes of
expanded BF16 weights each. These are execution-layout counts, not DRAM
counters: cache effects, codebooks, activations and other model operations
must be accounted for separately. DeepSeek throughput is not a GLM bound.

## Isolated evidence

The fixture contains eight actually selected layer-three experts, their
shared book and saved layer-zero normalized inputs. This preserves real
weights and access geometry; it is not the exact current Q4 prompt's
routing/activation assignment. The 41.5 MB payload exceeds GB10 L2;
the book and activation are hot. The baseline runs production restore,
cuBLAS and SwiGLU for the same 24 matrices. Gather/readback overhead is
excluded from both isolated timings.

At saved input row zero, 13-bit/32-warp execution measured 2.627→0.192 ms
per eight-expert layer. The matched 16-warp candidate measured 0.222 ms
(its baseline 2.653 ms). The 32-warp specialization uses 64 gate/up and
56 down registers, without stack/local spill. Dynamic shared storage is
72 KiB for gate/up and 96 KiB for down, including BF16 activations.

Four 12/13-bit × 16/32-warp cases plus the matched-input replay passed
independent FP64 per-projection bounds:
`gamma(K+32)*sum(abs(x*w)) + BF16 half-ULP`, with maximum ratio 0.831.
The 12-bit layout cases mask the real fixture's high plane; production
artifact support remains the loader's uniform 13-bit contract. Fused
SwiGLU is compared with the existing transform on the candidate's actual
gate/up values; weighted reduction matches the existing rounding/order.
Restoration/packing goldens and unchanged nonexpert gates are reused.

The isolated processes load system CUDA 13.4.92 and cuBLAS 13.8.0.4,
without Torch imports. Fixture SHA256:
`336362b24d68b8176816e5db0c6f6618671aa9ee813a6d4e63aa9d758f7cad8b`.

## Whole-workload comparison

| Fresh process | Prefill tok/s | Decode tok/s | Process wall, s |
|---|---:|---:|---:|
| Restore 1 | 121.81 | 4.02 | 218.00 |
| Direct 1 | 121.75 | 6.66 | 282.22 |
| Direct 2 | 122.36 | 6.60 | 208.72 |
| Restore 2 | 122.45 | 4.03 | 206.40 |
| Restore median | 122.130 | 4.025 | |
| Direct median | 122.055 | 6.630 | |

Median decode improves 64.72%, with prefill differing by −0.06%. Both
direct runs execute 1,344 direct calls: 42 MoE layers × 32 decode steps.
All runs use the same KeepHead Q4 artifact and native binary, 1,984 raw
code prompt tokens plus 32 forced non-EOS greedy transitions, context
2,048, chunk 512, Q4 direct/staging on and MTP/capture/VQ-MMA off.
The [evidence](evidence/glm53-vq-decode-2026-10-16.json) records build,
source, artifact, prompt and library identities.

The full prefill frontier and all 32 generated IDs agree. Prefill is
unchanged in this workload, so that frontier does not test the changed
decode arithmetic. First decode-forward time is 0.2662–0.2673 s for
restore and 0.1636–0.1642 s for direct; these are not HTTP TTFT.
Prefill plus decode totals inferred from rounded rates are approximately
24.20→21.08 s (−12.87%). Startup/loading variation dominates process
wall time; no cold-start improvement is claimed.
Observed busy clocks were 2,184–2,190 MHz under the unchanged 300–2,200 MHz
policy. Minimum system available memory was 42.02 GiB; the same guard
used a 104 GiB ceiling, 100 GiB high watermark and 12 GiB reserve.

## Changed-arithmetic text checks

Six held-out text tasks per artifact cover Korean, English, code, math,
JSON and stopping, with 106 teacher-forced target distributions and up to
96 generated tokens per case. Reuse the saved same-artifact controls;
the default-selection edit does not change the qualified kernels.

| Artifact | Paired NLL delta, nat | Case bootstrap 95% interval |
|---|---:|---:|
| Original nonexpert precision | +0.003779 | [−0.001158, +0.010535] |
| KeepHead Q4 candidate | −0.003494 | [−0.006999, +0.001016] |

All five previously passing task answers remain identical on both
artifacts. The original Korean summary retains its language and facts;
its changed wording matches the saved chunk-64 control. Both original
answers hit the 96-token cap, so this is not a complete-summary pass.
The Q4 candidate's existing Korean tool-markup/language/repetition failure
persists. The kernel introduces no additional observed task failure;
this small screen does not establish broad quality equivalence.

Adopt the direct kernel with its shape/resource fallback and kill switch.
Keep the Q4 artifact unadopted. The measured 64.72% gain belongs to the
KeepHead Q4 workload above; original-artifact speed requires its own
measurement. MTP, graph, long-context, vision, serving and broader quality
remain outside this proof.
