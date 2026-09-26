# Image round 1: full-attention score cache

**Accepted: image round 1.** Three fresh HTTP pairs reduce median TTFT by
65.37% and total request wall time by 55.23%, with unchanged median decode
rate and exact outputs. The isolated full-attention kernel is about 10×
faster. Proof-harness timings are not A/B measurements. The three accepted
[text rounds](../../mimo2-2026-09-26/README.md) remain in both controls:
`284ff879`, `e06d1f30`, `a3fd2a5c`.

## Fresh HTTP A/B and decision

Original, candidate OFF and default ON each use three fresh guarded workers
in rotated order. Each worker receives one matching warm request and one
measured request, both capped at 64 tokens. The image, prompt, serial shape
and resident weight owner are identical across arms.

| Median of three samples | Wall ms | TTFT ms | Reported prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Original | 15175.239 | 12817.3 | 62.2 | 26.9 |
| Candidate OFF | 15168.630 | 12813.2 | 62.2 | 26.9 |
| Default ON | 6794.645 | 4439.1 | 180.6 | 26.9 |

ON reduces TTFT by 65.43%, 65.35%, 65.37% against the original and by
65.36%, 65.41%, 65.34% against same-build OFF. Median wall reductions are
55.23% versus original and 55.21% versus OFF. Reported prefill improves
62.2→180.6 tok/s; this media-request rate is distinct from the language-only
trace interval below. Decode samples span 26.8–26.9 tok/s, with all three
arm medians at 26.9 and no measured repeatable decode regression.

All 18 warm/measured HTTP responses have identical 64 returned token IDs,
visible content, finish reason and counts. All cached-token counts are zero;
all nine workers have clean guard exits and zero memory/governor faults.
Busy clock samples are 2190 MHz. Separate Rust screen/photo/document proofs
compare every logit and the complete input contract, as detailed below.
[HTTP samples](http-ab.json), [response proofs](http-proofs.json),
[launch receipt](http-launch-receipt.json), [clocks](http-clocks.json),
[guard records](guards.json).

Adopt the scoped score cache: the gain is consistent in every pair, both
controls agree, correctness passes, and shared-memory costs are justified
by the measured latency reduction. No persistent tensor allocation is added.

## Workload and bottleneck

GB10, `sm_121a`, MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8 plus BF16 projector.
The unchanged `screen.png` fixture has SHA256
`c11a3a38ba18c8769a0b0785470c089d2fa7287955e033de8cb99b7ae80f67b4`.
Context 8192, native chunk 4096, `--cont-width 0`, prefix reuse off, MTP off,
64-token maximum, temperature 0, and returned token IDs. Native draft width
1 denotes the ordinary path with MTP disabled. Artifact hashes, exact prompt
and observed serial serving settings are in [workload.json](workload.json).

The completed `image-r1-serial3` whole profile measured 15.912 s HTTP wall,
13.522 s reported TTFT, and 26.9 decode tok/s. This was a cold first image,
including lazy serial graph allocation. The HTTP A/B instead warms
each fresh worker once before its measured request.

| Whole-profile region | Time | Share of HTTP wall |
|---|---:|---:|
| All 28 vision attention calls | 11.638 s | 73.14% |
| Four full-attention calls, layers 0/9/18/27 | 9.331 s | 58.64% |
| Other 24 windowed attention calls | 2.307 s | 14.50% |

The trace has no NVTX stage ranges. Its 12.254 s vision interval, 0.908 s
language-prefill interval and 2.461 s decode interval are inferred from
ordered native execution and distinctive projector/logits readbacks; they
are not separately instrumented CPU or server stage timings.
[Whole-profile evidence](nsys-before.json), [clean guard exits](guards.json).

## Detailed profiling and change

The isolated reproducer uses the actual kernel at 3072 rows, Q32/KV8,
head dimension 64, one interleaved FP32 QKV allocation, strides 3072 and
offsets 0/2048/2560, full attention without sinks/causal/group masks.
Synthetic values and repeated-buffer cache state differ from projected
image features. Both NCU captures use `--set full`, kernel replay, two
warmups and one selected launch; the task owner was stopped for collection.

The original scalar path recomputes every QK dot three times and updates
the global output for every key. The candidate caches one unscaled dot per
key in shared memory, then accumulates each output dimension in a register.
Dot, max, denominator and output accumulation preserve their original
ascending order. Caching the unscaled dot preserves the multiply/subtract
expression before `expf`. Source:
[kernel](../../../../cuda/mimo2_media.cuh),
[scoped wrapper](../../../../ds4_mimo2_gpu.cuh).

| Full NCU metric | Original | Candidate v1 |
|---|---:|---:|
| Duration | 2324.215 ms | 230.400 ms |
| Grid × block | 384 × 256 | 98304 × 128 |
| Registers/thread; local spills | 40; 0 | 40; 0 |
| Dynamic shared memory/CTA | 0 | 12,552 B |
| Executed warp instructions | 10.523 billion | 5.742 billion |
| Global load requests | 4.832 billion | 1.208 billion |
| Global store requests | 604,176,384 | 196,608 |
| L2 read sectors | 34,369,698,599 | 7,497,753,801 |
| L2 write sectors | 19,333,644,288 | 786,432 |

This exchanges shared memory for repeated global traffic; it adds no
persistent tensor allocation. Shared storage reaches 33,032 B at the
8192-row limit. Arithmetic instructions are not uniformly reduced: FFMA
and EX2 thread counts halve, while per-dimension normalization increases
FMUL and reciprocal counts. Total warp instructions fall 45.44%. The new
key mapping also increases the K access footprint; removing Q/output
traffic dominates the aggregate decrease. These are cache/request/sector
measurements, not a claim about off-chip DRAM bytes.

The baseline reports a 512 MiB warp-sampling buffer overflow. Sampled-PC
stall comparisons are excluded; replay instruction/request/sector counts
and static SASS support the analysis. Source `L1TagRequests` counts tags,
not 32-byte sectors. Both profiles warn that NCU did not lock clocks;
observed SM frequencies were 2.196725/2.196037 GHz, while the user-managed
300–2200 MHz range was preserved.
[Baseline analysis](baseline-analysis.json), [counter comparison](ncu-comparison.json),
[complete baseline sections](ncu-baseline.txt),
[complete candidate sections](ncu-candidate.txt).

## Isolated A/B and correctness

Three fresh pairs alternate original/candidate order. Each process uses
two warmups and three measured iterations on unchanged inputs.

| Pair | Original mean ms | Candidate mean ms | Reduction |
|---|---:|---:|---:|
| 0 | 2325.206 | 230.684 | 90.079% |
| 1 | 2329.728 | 230.050 | 90.125% |
| 2 | 2324.984 | 229.987 | 90.108% |
| Median of process means | 2325.206 | 230.050 | 90.106% |

Every run reports the same full-output diagnostic FNV hash. Separate pilot
full-buffer comparison and the focused test establish byte parity; the
hash alone is not treated as that proof. Busy clock samples were 2190 MHz.
[Samples](isolated.json), [clocks](clocks.json), [test output](test.txt).

The [focused test](../../../../tests/mimo2_vision_attn.cu) passes full-output
byte comparison at rows 1, 31, 32, 33, 127, 128, 129, 1024, 3072, 6144 and
8192. It checks input immutability, output canaries, finite hard cases,
NaN-max/invalid-sum behavior, and a double oracle for small finite inputs
(largest observed absolute error 1.77e-7). Actual wrapper diagnostics verify
default/ON and OFF/fallback paths, including window/sink, causal/group,
head/layout and separate-QKV cases. Invalid wrapper inputs launch no work.

The complete Rust image proof also passes all nine fresh processes:
screen/photo/document × original/OFF/ON. Each compares all 152,576 finite
post-prefill logits and 64 generated tokens exactly. Request JSON, rendered
prompt and prompt-token bytes also match across arms. All guard exits are
zero. This directly exercises native media encoding and language prefill;
its direct greedy generation excludes HTTP scheduling and sampling/output
processing. [Full-model proof](model-proof.json),
[input/output hashes and build receipts](model-proof-receipts.json).

The candidate is limited to full vision attention with the geometry above,
rows 1–8192, aliased Q/K/V and disjoint output. Other shapes retain the
original path. `DS4_MIMO2_VISION_ATTN=0` disables it. Native and standalone
builds use the original optimization flags; no split compilation was used.
[Build, source and adoption receipt](receipt.json),
[original reproducer receipt](baseline-source.json).

An optional scratch variant cached normalized weights. Its single bounded
pilot measured 229.658 ms versus v1's 229.594 ms, with exact output. It was
not retained; this decision does not reject the main score-cache candidate.
[Refinement decision](normalized-decision.json).

Raw reports, full counter/SASS CSVs, binaries, outputs and guard logs stay
under `scratch/mimo-media-20260926/{image-r1-serial3,ncu-attn-r1}/`.
The Rust proof harness is under `scratch/mimo-media-20260926/media-proof/`.
Full-model proof and HTTP A/B are complete. Start the next round with a new
whole-workload profile of this retained baseline. Copied profiler and pilot
receipts preserve their earlier collection-time status; [receipt.json](receipt.json)
records the final decision.
