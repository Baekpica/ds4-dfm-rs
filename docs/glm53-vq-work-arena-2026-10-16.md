# GLM VQ workspace allocation assessment

Reject the context work-arena candidate. Four fresh-process runs show no
repeatable whole-inference gain; both allocation modes retain large prefill
variation. Candidate code and its diagnostic flag were removed. This
allocation round changes no weights, accepted kernels or runtime defaults.

## Target and candidate

The [chunk trace](glm53-vq-chunks-2026-10-16.md) records 1.643 s in 71 initial
prefill `cudaMalloc` calls. The largest VQ context calls are 617.854 ms for
288.188 MiB MMA storage, 125.688 ms for 456 MiB scratch and 64.700 ms for
32 MiB cuBLAS workspace. These explain that trace's delay; the underlying
driver/OS trigger remains unproven.

An allocation-only replay preserves the preceding 108 VMM and 14 runtime
allocations, totaling 80.181 GiB, and touches every buffer. Separate
allocations take 20.185/27.574 ms; an arena takes 15.000/16.060 ms. It does
**not** reproduce the 1.643 s delay. It lacks artifact mappings, loading
history and real cuBLAS handles, so these proxy timings do not qualify a
production improvement.

The actual-model candidate combines only five VQ context buffers: scratch,
cuBLAS workspace, routing pairs, MMA work and decode work. At chunk 1,024,
the owner is 776.376 MiB. Views retain 256-byte alignment without padding or
a larger quote. Unsupported alignment falls back to separate allocations.
Graph/KV buffers and cuBLAS-owned allocations remain separate. No arithmetic
or weight representation changes.

A small actual-CUDA test covers two live contexts, freeing one and reusing
the other, capacities 1/8/32/512/1024/2048 and unaligned fallback.
Compute Sanitizer reports zero errors and leaks. Unchanged primitive
numerical proofs were reused.

## Matched whole workload

Original artifact `c3c35fe3…` manifest, 1,984 raw code input tokens plus 32
forced non-EOS greedy tokens, context 2,048, chunk 1,024, MTP/capture off.
All four runs use the same experimental binary, native sources and loaded
libraries. Busy clocks are 2,184–2,190 MHz under the 300–2,200 MHz policy.
Processes are fresh; OS file-cache state is inherited.

| Order | Arena | Prefill tok/s | Decode tok/s | Phase sum, s |
|---:|---|---:|---:|---:|
| 1 | Off | 459.60 | 7.32 | 8.688 |
| 2 | On | 345.87 | 7.31 | 10.114 |
| 3 | On | 459.85 | 7.34 | 8.674 |
| 4 | Off | 368.17 | 7.37 | 9.731 |

Median phase sum rises from 9.210 to 9.394 s, approximately 2.00%.
These sums use rounded throughput and exclude model loading and proof-file
output. Process wall times are recorded separately: 199.0–238.3 s.
The result rejects repeatable gain, not a fixed percentage threshold.
Without another trace, individual slow runs cannot be attributed to a
particular allocation call.

All frontier logits and generated IDs match; this is bounded path parity,
not general quality. Every run processes the same 33,435 MMA work items.
Minimum system available memory is 34.390–34.576 GiB. No candidate DEVICE
peak was profiled. Guards retain 104/100 GiB max/high, 12 GiB reserve/trip,
no scope swap and a 1,800 s deadline; PSI is telemetry.

The adoption-only quality pair and additional candidate trace were skipped.
The rejected patch, build receipt, lifecycle test and all four measurements
are retained in [the evidence](evidence/glm53-vq-work-arena-2026-10-16.json).
The experimental binary includes the removed, default-off candidate and
must not be described as a clean-HEAD build.

## Remaining conversion cost

The saved chunk-1,024 trace exposes a separate target. For each DSA layer,
the graph expands latent KV with BF16 GEMM, converts its output to FP32,
then converts the same values back to BF16 for attention
(`ds4_glmvq_graph.inc:114-125`, `cuda/glm_vq_linear.cuh:92`,
`cuda/glm_vq_attention.cuh:105`).

Across 32 decode steps, the two large conversion groups take 0.568383 and
0.562375 s: 1.130758 s, or 25.82% of the 4.378622 s decode phase. Prefill
spends 0.053487 s in the corresponding pair. These are existing trace
costs, not a demonstrated optimization gain. A direct BF16 handoff is the
candidate for eliminating repeated memory traffic.

An isolated replay of the existing pair at 1,985 keys, 64 heads and
dimension 256 takes 3.167 ms after 16 warmups and 64 repeats. Its 780.534 MB
logical transfer volume implies 246.42 GB/s. NCU records 1.630/1.594 ms,
16 registers per thread and 2.197 GHz; L2 read/write sector counts match
the logical volume. The full capture exposes no DRAM byte counter, so this
is not measured DRAM bandwidth. Constant finite inputs and separate BF16
source/destination buffers omit real projection work and scratch reuse.

Subsequently, the user selected the existing 136-matrix KDA Q4_K artifact
for speed optimization despite the recorded quality differences. Its whole
workload is being profiled before selecting the next kernel change; the
25.82% contribution above belongs to the original artifact.

Qualification remains bounded to 2K eager text. This round does not qualify
8K, MTP, captured decode, vision, live tools, banks, snapshots or disk KV.
