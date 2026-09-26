# Image round 2: window-attention score cache

**Accepted: image round 2.** Three fresh HTTP pairs reduce median request
wall time by 31.01% and TTFT by 47.47%, with no repeatable decode regression.
Focused wrapper and full-model image proofs pass exactly. One OFF-only HTTP
wording variation is recorded below; all original and ON responses match.

The baseline is accepted R1 commit `22cc4e64`, including the three retained
text optimizations. This round leaves its full-attention kernel unchanged.

## Fresh HTTP A/B and decision

Original, candidate OFF and default ON each use three fresh guarded workers
in rotated order. Every worker receives one matching warm request and one
measured request, capped at 64 tokens. Model, image, prompt, serial serving
shape and resident owner are identical across arms.

| Median of three samples | Wall ms | TTFT ms | Reported prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| R1 original | 6805.385 | 4450.7 | 180.1 | 26.8 |
| Candidate OFF | 6806.614 | 4449.1 | 180.2 | 26.8 |
| Default ON | 4695.218 | 2338.1 | 345.9 | 26.9 |

ON reduces wall time by 31.01%, 31.02%, 31.08% against original and by
31.02%, 31.11%, 30.96% against same-build OFF. Median TTFT reductions are
47.47% and 47.45%, respectively. Reported media prefill improves 92.06%
versus original; it includes vision work and differs from language-only
prefill. Decode samples span 26.7–26.9 tok/s with no consistent regression.
[HTTP samples](http-ab.json), [launch receipt](http-launch-receipt.json).

Seventeen of 18 warm/measured token/content signatures match. Only
`screen-2-off/sample` changes wording, starting at zero-based token index 12.
That response still describes PROJECT ATLAS with Queued 12, Running 7 and
Failed 3. Its warm response and every original/ON response match the
reference. This OFF-control variability is retained in the evidence.
The raw runner's exactness flag is false and its `requires_review` status
is resolved by this review, the exact kernel tests and all nine complete
image-model proofs. No claim is made that all 18 HTTP responses are exact.
[Full response comparison and review](http-proofs.json).

All cached-token counts, memory census faults and governor faults are zero.
All nine HTTP workers exit cleanly; busy GPU samples are 2190 MHz within
the unchanged 300–2200 MHz range. [Guards](guards.json),
[HTTP clocks](http-clocks.json).

Adopt the scoped window score cache: gains repeat against both controls,
candidate correctness passes, and the shared-memory/register costs below
are justified by the measured benefit. No persistent allocation is added.

## Workload and selected bottleneck

A fresh cold `screen.png` request with R1 enabled measured 7542.464 ms wall,
5146.6 ms TTFT and 26.9 decode tok/s. Configuration and artifact are unchanged:
GB10, MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8 plus BF16 projector, context 8192,
native chunk 4096, continuous width 0, prefix reuse/MTP off and 64 output tokens.
The fixture SHA256 is
`c11a3a38ba18c8769a0b0785470c089d2fa7287955e033de8cb99b7ae80f67b4`.
[Shared workload contract](../attn-score-cache/workload.json).

| Fresh whole-profile region | Time | Share of request wall |
|---|---:|---:|
| 24 scalar window-attention calls | 2307.323 ms | 30.59% |
| Four retained full-attention calls | 927.160 ms | 12.29% |

Window attention is also 59.86% of the 3854.315 ms vision interval.
Language prefill is 911.960 ms and decode 2476.565 ms. There are no NVTX
stage ranges: these intervals follow native execution order and distinctive
projector/logits readbacks. They are not separately instrumented CPU or
server stage timings. [Whole trace summary](nsys-before.json).

## Detailed profiling and change

The isolated target has 3072 rows, Q32/KV8, head dimension 64, one interleaved
FP32 QKV allocation, strides 3072, offsets 0/2048/2560, window 64 and 32 sinks.
Causal/group masks are disabled. The frozen scalar reproducer is reused:
its function-body hash still matches production after R1. Synthetic inputs,
sink values and warmed cache state differ from projected image features.
[Baseline receipt](baseline-source.json), [candidate receipt](candidate-source.json).

Full NCU identifies an L2 bottleneck: 88.78% throughput with only 4.41% SM
compute throughput. The scalar kernel scans all keys three times, repeats
each QK dot and updates global output for every valid key. Output read/write
accesses alone cover 51.60 GB of cache sectors for 25.17 MB of final output.

The candidate clamps the contiguous valid interval to at most 129 keys,
caches each unscaled dot once, and accumulates each output dimension in a
register. Dot, maximum, denominator and output accumulation retain scalar
order. Sinks remain last in maximum and denominator; caching unscaled dots
preserves the multiply/subtract expression before `expf`.

| Full NCU metric | Original | Candidate |
|---|---:|---:|
| Duration | 97.158 ms | 8.354 ms |
| Grid × block | 384 × 256 | 98304 × 128 |
| Registers/thread; local spills | 40; 0 | 44; 0 |
| Dynamic shared memory/CTA | 0 | 780 B |
| Executed warp instructions | 935,582,464 | 394,968,064 |
| Global load requests | 200,775,680 | 56,553,472 |
| Global store requests | 25,292,800 | 196,608 |
| L2 read sectors | 1,410,420,174 | 223,126,345 |
| L2 write sectors | 809,369,600 | 786,432 |

L2 reads fall 84.18%; total warp instructions fall 57.78%. Registers increase
by four and each CTA adds 780 B of shared memory, with no persistent tensor
allocation. Arithmetic is not uniformly reduced: FFMA counts halve and EX2
nearly halves, while repeated per-dimension normalization increases FMUL
and reciprocal counts. These cache/request/sector counts are not off-chip
DRAM byte measurements. [Exact comparison](ncu-comparison.json),
[baseline explanation](baseline-analysis.json).

Neither window capture reports sampling overflow. Both warn that NCU did
not lock clocks; measured SM frequencies were 2.196616/2.196717 GHz. The
user-managed 300–2200 MHz range remains unchanged. Complete NCU sections:
[baseline](ncu-baseline.txt), [candidate](ncu-candidate.txt).

## Isolated timing and correctness

Three fresh process pairs alternate order, with three warmups and ten
measured iterations per process.

| Pair | Original mean ms | Candidate mean ms | Reduction |
|---|---:|---:|---:|
| 0 | 96.750380 | 8.363286 | 91.356% |
| 1 | 96.181799 | 8.360083 | 91.308% |
| 2 | 96.443631 | 8.359808 | 91.332% |
| Median | 96.443631 | 8.360083 | 91.332% |

Every repeated run has the same diagnostic hash. Separate pilot full-buffer
comparison passes; both 25165824-byte dumps have SHA256
`72584b105556ce680ad83a3906e5dda4b55f18ab8cb1c345364093bd4d9a5126`.
The repeated hashes are not a substitute for that full-buffer comparison.
[Samples and pilot hashes](isolated.json). All recorded guard exits are
clean; busy clock samples are 2190 MHz. [Guards](guards.json), [clocks](clocks.json).

The [focused test](../../../../tests/mimo2_vision_window.cu) compares actual
wrapper OFF/default/ON outputs exactly at rows 1/31/32/33/63/64/65/127/128/
129/130/3072/6144/8192. It covers window tails, sink extremes/nonfinite values,
row/column permutations, invalid input, fallback masks/layouts, input
immutability and output canaries. The small double oracle's largest error
is 1.55e-7. [Window test output](test-window.txt),
[retained R1 regression output](test-retained-full.txt).

Nine fresh Rust proof processes cover screen/photo/document across R1
original, R2 OFF and R2 ON. Every image matches all 152,576 finite
post-prefill logits, 64 generated tokens and the complete input bytes
(request, rendered prompt and prompt tokens). All guard exits are zero.
These proofs exercise native media encoding and language inference; their
timings exclude HTTP scheduling/output processing and are not speed A/B.
[Full-model results](model-proof.json),
[input/output hashes and build receipts](model-proof-receipts.json).

Dispatch requires this vision geometry, rows 1–8192, window 64 with sinks,
aliased Q/K/V and disjoint output. Other shapes retain their prior paths.
`DS4_MIMO2_VISION_WINDOW=0` disables the candidate. Standard optimization
flags are unchanged; no split compilation is used. Source:
[kernel](../../../../cuda/mimo2_media.cuh),
[wrapper](../../../../ds4_mimo2_gpu.cuh).

Raw reports, CSVs, binaries and outputs remain under
`scratch/mimo-media-20260926/ncu-window-r2/`. This snapshot records kernel
and full-model correctness plus completed HTTP A/B. Begin the next round
with a fresh whole-workload measurement of this retained baseline.
[Final decision and scope](receipt.json).
