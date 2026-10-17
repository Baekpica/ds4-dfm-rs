# GLM VQ cooperative nonexpert Q4_K decode

This change targets execution time on the nonexpert Q4_K candidate.
It keeps routed/shared experts, weight bytes, activation precision and
the BF16 output boundary. The artifact's separate quality issues remain
[open](glm53-vq-q4k-2026-10-15.md#adoption-boundary).

## Measured target

The complete 1,984+32 workload attributed 892.08 ms of 8,305.41 ms decode
time to 6,048 small Q4 dots: 10.74%, or 189 projections/token. Their packed
weights total 3,444,572,160 bytes/token. This is logical weight traffic;
the reported payload bandwidth is not a DRAM-counter measurement.

An isolated actual KDA matrix showed latency stalls: NCU recorded 25.4
L1TEX-wait cycles/instruction interval and 0.68 eligible warps/scheduler.
The implementation adapts YoungAi `7fb26cc`'s
`src/cuda/cuda_v41_q4k.inc.cu:29-179`: 16-byte CTA-wide weight loads,
register-resident block headers and contiguous packed-word decoding.
GLM uses ordinary stream ordering and its own FP32 FMA/BF16 boundary;
ds41 PDL and state rules are not imported.

## Selection and resources

| Actual matrix [output,input] | Scalar, µs | Staging, µs | Double buffer, µs |
|---|---:|---:|---:|
| KDA Q/K/V [8192,4096] | 135.25 | 83.51 | 112.68 |
| KDA output [4096,8192] | 157.61 | 85.37 | 118.27 |
| DSA Q-B [16384,1536] | 82.97 | 62.12 | 79.65 |
| DSA KV-A [512,4096] | 8.24 | 8.70 | 9.40 |

These CUDA-event replays rotate at least 64 MiB of real weight copies.
The KDA Q replay uses saved layer-zero activations; other shapes use synthetic
BF16 inputs with the same dense access pattern. They are not in-model
routing or MTP measurements. Shape-weighted isolated times predict
25.50→15.78 ms/token; the full trace measured 27.88 ms/token before this
change. End-to-end A/B decides adoption.

The selected path stages aligned matrices larger than 4 MiB at width one.
Small matrices and widths 2–8 retain scalar dots. Resource/alignment
failures also retain scalar execution. `DS4_GLMVQ_Q4_STREAM=0` disables
staging. Double buffering remains diagnostic: it was slower on these GLM
shapes, including the reference's 4–12 KiB staging regime.

On sm_121a, staging/pipe use 51/53 registers at width one and 64 for the
experimental width-eight instance; stack/local storage is zero. KDA
staging uses 18 KiB dynamic shared memory and adds no global allocation.
The same NCU flushed-cache protocol measured 167.52→103.39 µs, L1TEX wait
25.4→8.7 cycles and eligible warps 0.68→1.12. Achieved occupancy fell from
93.2% to 61.6%; reduced stalls still improved time. NCU-instrumented CUDA
event durations are excluded from ordinary timings. GB10 executed the
pipeline tests and its SASS contains `LDGSTS.E.BYPASS.128`.

## Numerical and workload boundary

Thirty matrix/width/transport replays passed. New staged/pipelined outputs
were checked against independent FP64 affine-Q4 dots, using
`gamma(K+32)*sum(abs(x*w)) + BF16 half-ULP`. Maximum bound ratio was 0.851.
The cases include actual KDA/DSA/dense shapes, widths 1/3/8 and partial
CTA/persistent-loop tails. Forced pipe requests exceeding the device's
shared-memory allowance used staging. Lane/reduction order changes are
explicit; bit identity is not required.

## Whole-workload comparison

Four fresh processes used the same binary and KeepHead Q4 artifact in
scalar/staging/staging/scalar order. Medians improve decode by 3.23%;
both staging samples exceed both scalar samples. Prefill differs by
−0.29%; its execution path is unchanged.

| Mode / run | Prefill tok/s | Decode tok/s | Process wall, s |
|---|---:|---:|---:|
| Scalar 1 | 122.31 | 3.89 | 205.87 |
| Staging 1 | 121.78 | 4.01 | 217.25 |
| Staging 2 | 122.05 | 3.99 | 209.40 |
| Scalar 2 | 122.24 | 3.86 | 213.82 |
| Scalar median | 122.275 | 3.875 | |
| Staging median | 121.915 | 4.000 | |

This is 1,984 raw-code prompt tokens plus 32 forced non-EOS greedy
transitions, context 2,048, chunk 512, MTP/capture/VQ-MMA off. Observed
busy SM clocks were 2,184–2,190 MHz within the unchanged 300–2,200 MHz
policy. Each process owned its weights through in-process VMM; OS file
cache was inherited. Startup/loading variability dominates process wall
time, which does not improve in this sample. First decode-forward time
was 0.2632–0.2790 s; this is not HTTP TTFT.

The scalar whole profile peaked at 74.599 GiB of traced device/static
allocations, excluding driver, OS and mmap accounting. Staging adds only
per-CTA shared memory; no global allocation is added. All four runs kept
at least 42.03 GiB system-available memory under the 104/100 GiB scope
guard and 12 GiB reserve. Guard availability is not device peak memory.

Build base is `923e9b34` plus the native source digests in the
[evidence](evidence/glm53-vq-q4-stream-2026-10-16.json); benchmark SHA256 is
`819b4471987a2fc2a476ed31322f3b1ef6cf51b306d311fcdd998c736e589f35`.
CUDA 13.4 runtime/cuBLAS/cuBLASLt resolve from `/usr/local/cuda-13.4`.
KeepHead manifest is
`d16712edb7940dd070873d7aa660f7731c036605484a1bc54155e5f5c802fe47`.
The full-vocabulary prefill frontier is unchanged, as expected for a
decode-only change, and all 32 generated IDs agree. Neither observation
alone establishes quality of the changed decode arithmetic.

The six-case changed-kernel text screen completed with finite logits over
106 teacher-forced targets. Relative to the saved scalar Q4 control,
mean target NLL changes by +0.00264 nat/token; a six-case bootstrap gives
95% interval [−0.00258,+0.00599]. English, code, arithmetic, JSON and the
Korean stop-answer contracts retain the same answers. Both kernels fail
the existing Korean-summary case with unsolicited tool markup, English
text and truncation. This limited screen found no additional task failure;
its aggregate statistics do not establish broad quality equivalence.

Retain cooperative loads for eligible Q4 matrices. Artifact adoption,
MTP, graphs, 8K, vision, API serving and state reuse remain unqualified here.

## Follow-up transport screen on the retained KDA artifact

At `de0ee6a4`, the user-selected 136-matrix KDA Q4_K artifact is active;
its earlier artifact-quality rejection is superseded. The retained
[latent-attention profile](glm53-vq-latent-2026-10-16.md) attributes 22.52%
of decode to Q4 staging: 2.567 GB logical weights/token at 189.15 GB/s
payload bandwidth. This is not measured DRAM traffic.

The kernel source and all 136 selected payload hashes match the earlier
Q4 evidence. Reusing its NCU source counters identifies waits at weight
staging stores and the first input-dependent FMA. A bounded replay compares
constant-shape code, single-stage cp.async and next-block input prefetch.
Each entry below is the median of three interleaved CUDA-event observations
in microseconds, using actual layer-zero matrices and saved source inputs.

| Shape | Current | Shape only | Async | Async + prefetch |
|---|---:|---:|---:|---:|
| [8192, 4096] | 83.27 | 82.25 | 107.35 | 105.23 |
| [4096, 8192] | 83.34 | 85.80 | 107.93 | 115.82 |

Async variants regress. Shape specialization does not consistently improve
both shapes beyond observed variation. Reject this screen; keep production
unchanged. All 24 outputs match the current dot path. No new arithmetic,
whole-model loading or quality rerun was needed. The replay rotates 72 MiB
of copied weights with no intervening model kernels; it is not a serving
benchmark. Observed busy clocks are 2,184–2,190 MHz.

Each candidate uses the same 18 KiB dynamic shared memory and no spills.
Registers fall from 51 to 40 for specialization; prefetch uses 51/48.
Those reductions do not establish a speedup. The cause of the async slowdown
was not profiled further because neither async variant proceeds to adoption.
Commands, outputs, libraries and reproducer sources are in the
[screen evidence](evidence/glm53-vq-q4-refine-2026-10-16.json).
