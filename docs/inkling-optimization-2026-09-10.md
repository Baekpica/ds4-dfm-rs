# Inkling MQ85GB optimization on GB10

Continued in [rounds 13–15](inkling-optimization-2026-09-11.md) and
[rounds 16–18](inkling-optimization-2026-09-11-r16.md), September 11.

This campaign starts at merged v0.1.2 main `10658bdf` and keeps the default
aligned dense-Q8 numerical path. Results apply to the six-shard MQ85GB
artifact on one DGX Spark. They do not establish MQ89, Q8_0-main, independent
source parity, long-context serving or concurrent-request performance.

## Campaign scope

After the aligned Q8 batch candidate reached 111.53 prefill tok/s, the
campaign target expanded to another detailed `ds4-perf` investigation and
**at least three additional prefill improvements** from that state, including
a retained routed-IQ2 gain. Rounds 4–7 retain four additional improvements,
with the routed-IQ2 gain in round 4. The first three candidates are separate.
Work prioritized prefill, with decode retained as a regression gate. The
original three-decode-improvement target is deferred and has not been achieved;
this report claims no decode speedup. Rounds 8–12 extend the campaign with an
8192-token workload: rounds 10–12 retain three further prefill improvements
(BF16 panels above 4096 rows, attention grouped by KV head and dense Q8
tiles), raising the 8K chunk-512 median from 230.84 to 272.76 tok/s and the
2K median from 259.67 to 283.82 tok/s with exact logits throughout. The
chunk default remains 512.

## Protocol

The raw `speed-bench/promessi_sposi.txt` fixture supplies 2048 prefill tokens,
followed by 64 greedy decode tokens. Context allocation is 2113. Each sample
starts a fresh process after a separate warmup process; no prompt cache is
reused. One resident VMM owner holds the same base and MTP-BF16 mappings
throughout. MTP is **off** in the timed workload.

`ds4-perf scout --proof --repeats 3 --cache-policy warmup-then-fresh` records
unprofiled throughput separately from its Nsight Systems phase trace. Controls
bracket each candidate. Acceptance requires a phase gain above 1% with
nonoverlapping sample ranges, full-vocabulary/token proof, and no phase or
first-step slowdown above 3%. `first_token_sec` measures the first decode
step after prefill; it is not request time to first token.

Hardware/software: NVIDIA GB10, driver 610.43.02, CUDA 13.3, Nsight Systems
2026.1.3; CUDA build uses `sm_121a`. The worker memory guard uses a 12 GiB
limit, 10 GiB high watermark and 12 GiB host reserve. The owner remains
resident between samples. Measured calibration was 245–246 GB/s device-copy
bandwidth, 27.5 TFLOP/s FP32 SIMT and 2.83 µs launch latency; these are
calibration kernels, not model throughput or Tensor Core peak.

The benchmark prerequisite `58de146` restores Inkling frontiers by replay
outside timed phases because native snapshots are unavailable. The original
main benchmark failed before decode. Cold versus restored 64/129-token
frontiers passed full-logit and token checks. Native fixture repair `b5a4e33`
imports the owner's base ranges explicitly. Neither repair counts as a
performance improvement.

## Round 1: ordinary BF16 projections

Baseline Nsight attributed 25.976 s of 58.081 s prefill wall time to stable
BF16 projection kernels. The old schedule traversed all weight rows for each
token. Adjacent token warps now reuse a weight row, and the final scale-one
BF16 store is folded into the projection. Router logits keep FP32. Input
conversion, per-lane products, warp reductions and BF16 boundaries retain
the prior arithmetic. `DS4_INKLING_NO_LINEAR=1` restores the prior path.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Initial control | 35.36 / 35.32 / 35.33 | 18.05 / 18.10 / 18.09 |
| BF16 candidate | 56.89 / 56.75 / 56.76 | 17.95 / 17.96 / 17.94 |
| Rollback control | 35.34 / 35.29 / 35.33 | 18.11 / 18.09 / 18.12 |

Candidate medians are **56.76 prefill / 17.95 decode tok/s**. Against the
initial control, prefill throughput improves 60.7%; decode throughput is
0.8% lower. This is a prefill improvement, with no decode gain claimed.
Both initial and rollback `ds4-perf compare --regression` runs report
`Improved`, with zero logit or token differences. This retains one prefill
improvement; it does not count toward the decode goal.

The 4096×4096, 64-row component gate including conversion and final store
measured 9.150 ms → 1.375 ms. In the full-model trace, ordinary BF16 kernels
take 3.950 s and unchanged router projections 0.095 s. End-to-end prefill
wall time becomes 36.213 s. Routed/shared expert MMVQ now takes 26.357 s,
about 73% of prefill wall time, and is the next target.

All three candidate frontier arrays (200058 logits each) and 64-token streams
are byte-identical to the original control. Native 12-token full/chunk/decode
checks match logits, hidden state, all 7,225,344 KV/convolution bytes and
accepted-prefix restoration for lengths 1–9. Eight MTP cycles, 18 greedy
tokens and image/audio session regressions pass. Synthetic tests cover
projection shapes, non-BF16 inputs, aliases, bounds and the kill switch;
53 `ds4-perf` tests pass.

## Round 2: batched expert MMVQ

The round-1 trace spent 26.357 s in token-at-a-time routed/shared MMVQ.
The new path quantizes the complete activation batch once, buckets routes
by expert and executes compact four-assignment/two-output-row tiles. It
preserves the canonical Q8 activation bytes, each format's integer dot
fragments and the original four-warp up / one-warp down reduction. Invalid
routes remain zero; the final finite guard moves into the output store.
Width-one decode keeps its existing path. `DS4_INKLING_NO_MOE_BATCH=1`
restores the round-1 implementation.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Round-1 control | 56.89 / 56.75 / 56.76 | 17.95 / 17.96 / 17.94 |
| Expert batch candidate | 95.96 / 96.37 / 96.43 | 17.95 / 17.96 / 17.95 |
| Rollback control | 56.86 / 56.79 / 56.67 | 17.96 / 17.94 / 17.92 |

Candidate medians are **96.37 prefill / 17.95 decode tok/s**: 69.8% more
prefill throughput than round 1, with the same decode median. All nine
frontier and token proof pairs across these three paths have identical
hashes. Both `ds4-perf compare --regression` comparisons report `Improved`
with zero logit/token differences. This retains the second prefill
improvement; no decode gain is claimed.

The new expert kernels take 12.786 s in the full-model prefill trace;
prefill wall time is 21.465 s. BF16 projections take 3.967 s and dense Q8
3.508 s. The 64-token, 17-expert component fixture including preparation
measured IQ2_XXS up 6.632 → 4.347 ms and IQ2_XS down 4.297 → 2.220 ms.
These component fixtures do not represent full-model throughput.

Five quantization formats pass byte-exact component comparisons, including
ragged/repeated/invalid routes, maximum assignment counts, workspace bounds
and diagnostic controls. Native full/chunk/decode, accepted-prefix restore,
eight MTP cycles, image/audio and 53 `ds4-perf` checks also pass. Native
full-vocabulary output matches the rollback path exactly.

## Round 3: aligned dense Q8 batches

Round 2 still spent 3.508 s in dense Q8 projections. The two dense layers
now group up to eight adjacent tokens through the existing aligned-Q8
primitive. The aligned weight artifact, activation quantization and each
output's reduction stay unchanged. The fast path covers dense up/down
shapes at widths 2–2048; shared experts use the separate round-2 path.
`DS4_INKLING_NO_Q8_BATCH=1` restores per-row dispatch.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Round-2 control | 95.96 / 96.37 / 96.43 | 17.95 / 17.96 / 17.95 |
| Dense Q8 candidate | 112.23 / 111.53 / 111.27 | 17.96 / 17.94 / 17.96 |
| Rollback control | 96.19 / 95.80 / 95.61 | 17.94 / 17.94 / 17.93 |

Candidate medians are **111.53 prefill / 17.96 decode tok/s**: 15.7% more
prefill throughput than round 2. Both preceding and rollback comparisons
report `Improved`, with zero full-vocabulary logit or greedy-token
differences. Decode is unchanged within measurement variation. The three
original prefill improvements are retained; the expanded target still
requires **three additional prefill improvements and three decode gains**.

The 64-token dense component gate measured up 36.428 → 5.275 ms and down
19.658 → 3.117 ms, including activation preparation. Boundary widths through
2048, incomplete eight-column tiles, aliases, spans, absent artifacts and
kill switches pass. Native full/chunk/decode, accepted-prefix restore,
eight MTP cycles, image/audio and 53 `ds4-perf` checks pass with exact
logits and state.

## Investigation after round 3

A fresh `ds4-perf scout` with Nsight Systems and Compute reproduced round 3:
prefill 112.30 / 111.89 / 111.24 and decode 17.89 / 17.94 / 17.92 tok/s.
The repeatability comparison reports `Pass`; this is not another improvement.
All proof arrays and tokens remain exact. The fresh trace has 18.583 s of
prefill wall time and 67,312 kernels. Only 91.7 ms (0.49%) falls outside
the union of kernel intervals.

| Prefill path | Kernel time | Share of phase wall |
| --- | --- | --- |
| Routed IQ2_XXS up, layers 3–39 | 5.805 s | 31.24% |
| Routed IQ2_XS down, layers 3–39 | 2.606 s | 14.02% |
| Shared Q8 up | 2.145 s | 11.54% |
| Shared Q8 down | 1.077 s | 5.80% |
| Ordinary BF16 projections | 3.964 s | 21.33% |

Launch order, shapes and format specializations were checked against all
40 sparse layers and 32 prefill chunks. Routing worklists take about 28 ms
and activation conversion 20 ms. The measured priority is the routed IQ2
path, followed by repeated shared-Q8/BF16 payload loads.

The automatic Compute sample selects the first routed Q8 layer, so five
additional exact kernel targets isolate IQ2 up/down, shared Q8 up/down and
BF16 query projection. IQ2 uses 94/95 registers per thread, reaches about
35% active warps, and executes no Tensor Core instructions. Long-scoreboard
stalls account for 31.6%/39.0% of the sampled IQ2 up/down warp stalls. Shared
Q8 and BF16 also show substantial load-dependency stalls. These counters
motivate load reuse and alternate execution schedules; they do not establish
DRAM saturation. Requested L1/L2 bytes are not DRAM traffic.

Supplementary counters use one matching launch, strict application replay,
no forced cache flush and no clock lock. They are diagnostic evidence,
separate from unprofiled speed samples. A single-sample chunk-size sweep
peaks at 116.59 tok/s with 256 tokens per chunk, with exact proofs; it lacks
the repeated A/B protocol and does not count as a retained improvement.

## Round 4: warp-owned expert tiles

The round-3 trace spent 8.41 s in routed IQ2 up/down and 3.22 s in shared
Q8 through the round-2 kernel, whose four-warp CTAs decoded every IQ2
weight fragment once per column, re-read each column's activations from L2
for every row pair and merged through shared memory. One warp now owns a
tile of four (Q8 up: two) rows by eight routed columns: each fragment is
decoded once per row, the canonical Q8_1 activations are relaid into an
aligned SoA once per call, and the warp replays the original schedule
(four K quarters for up, the two-fragment chain for down) so every lane
product, warp partial, ascending merge and XOR tree is byte-identical.
Q3_K/Q4_K layers keep the round-2 kernel. `DS4_INKLING_NO_MOE_TILE=1`
restores the round-2 kernel for all formats.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Fresh round-3 control | 110.99 / 110.00 / 111.70 | 17.95 / 17.95 / 17.97 |
| Expert tile candidate | 149.51 / 149.44 / 150.07 | 17.97 / 17.98 / 17.98 |
| Rollback control | 110.80 / 110.26 / 111.53 | 17.97 / 17.97 / 18.00 |

Candidate medians are **149.51 prefill / 17.98 decode tok/s**: 34.7% more
prefill throughput than the fresh round-3 control, decode unchanged.
Both `ds4-perf compare --regression` runs report `Improved` with zero
full-vocabulary logit differences and identical frontier/token hashes.
This retains the first additional prefill improvement and the mandatory
routed IQ2 gain: in the full-model trace IQ2_XXS up falls from 4.91 to
2.29 ms per launch (5.81 s to 2.71 s) and IQ2_XS down from 2.19 to 1.50 ms
(2.61 s to 1.77 s); shared Q8 down falls from 1.08 s to 0.68 s while shared
Q8 up stays at 2.26 s because its two-byte-aligned weight loads dominate.
Prefill wall time is 13.74 s; ordinary BF16 projections (3.94 s, 28.7%)
and shared Q8 up (16.5%) are the next targets.

The fixture with production geometry (M4096, 256 experts, random routes)
measured 64 tokens: IQ2_XXS up 5.94 → 4.54 ms and IQ2_XS down 3.73 →
3.22 ms; 512 tokens: 35.4 → 12.6 ms and 16.8 → 8.2 ms; shared Q8 (2 experts,
64 tokens) up 1.98 → 1.79 ms and down 1.20 → 0.72 ms. All five formats,
ragged/repeated/invalid/random routes, malformed IDs, workspace bounds and
the kill switch are byte-exact against the per-token oracle. Native
12-token full/chunk/decode logits match the kill-switch control and round 3
exactly; accepted-prefix restore 1–9, eight MTP cycles, image/audio and
53 `ds4-perf` tests pass.

## Round 5: BF16 projection tiles

The round-4 trace spent 3.94 s (28.7%) in ordinary BF16 projections. The
round-1 kernel grouped eight token warps per weight row, but each warp still
streamed its token's activation vector from L2 for every output row, so a
64-token launch moved about 2 GB through L2 for 33.5 MB of weights. One CTA
now owns 16 weight rows and 16 tokens: the token slab is staged in shared
memory once per row tile, each lane reuses a weight vector across the 16
token accumulators, and every output keeps its lane K stripe, eight-FMA
chains, FP32 adds, XOR tree and BF16 store. Widths below 16 and K widths
not divisible by 256 (media 4800) keep the grouped kernel; decode is
unchanged. `DS4_INKLING_NO_LINEAR_TILE=1` restores the grouped kernel.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Round-4 control | 149.51 / 149.44 / 150.07 | 17.97 / 17.98 / 17.98 |
| BF16 tile candidate | 193.21 / 192.99 / 193.77 | 17.97 / 17.99 / 17.99 |
| Rollback control | 149.65 / 149.82 / 150.29 | 17.96 / 17.96 / 17.97 |

Candidate medians are **193.21 prefill / 17.99 decode tok/s**: 29.2% more
prefill throughput than round 4, decode unchanged. Both comparisons report
`Improved` with zero logit or token differences. This retains the second
additional prefill improvement. In the full-model trace the BF16 kernels
fall from 3.94 s to 0.88 s (Q 0.32 s, O 0.33 s, K/V 0.09 s each, R 0.05 s);
prefill wall time is 10.64 s, and the expert tiles (IQ2_XXS up 2.71 s,
shared Q8 up 2.25 s, IQ2_XS down 1.77 s) now hold 70% of it.

The standalone kernel probe measured 4096×4096: 64 rows 1.069 → 0.218 ms,
512 rows 9.257 → 1.595 ms; 4096×1024, 64 rows 0.259 → 0.061 ms; the MTP head
4096×32768 at 129 rows 18.46 → 3.56 ms; all byte-exact. Eight tile variants
were exact; the chosen four-warp tile uses 158 registers without spills,
whereas the eight-warp variant spilled 744 bytes and ran slower than the
grouped kernel. The native linear test covers rows 1–129, 512 and 2048,
media K=4800, the MTP head, partial aliases, bounds and the kill switch,
all exact through the API. Native 12-token logits match round 4 and the
kill-switch control; session, accepted-prefix, eight MTP cycles and
image/audio gates pass.

## Round 6: 512-token prefill chunks

The default chunk grows from 64 to 512 tokens. More assignments per expert
improve tile fill and weight reuse; the kernel arithmetic remains unchanged.
`DS4_INKLING_PREFILL_CHUNK=64` restores the previous default. Graph scratch
grows with the cap, which is still bounded by the allocated context.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| BF16-tile control | 193.21 / 192.99 / 193.77 | 17.97 / 17.99 / 17.99 |
| Chunk-512 candidate | 248.92 / 247.48 / 247.74 | 17.95 / 17.95 / 17.95 |
| Chunk-64 rollback | 193.05 / 192.68 / 192.21 | 17.95 / 17.95 / 17.95 |

The candidate median is **247.74 prefill / 17.95 decode tok/s**, a 28.2%
prefill throughput gain over the preceding candidate and 28.6% over the fresh
rollback. Both comparisons report `Improved`, checking 1,200,348 logits
with max_abs=0 and no token mismatches. This retains the third additional
prefill improvement after round 3. No decode improvement is claimed.

The prefill trace falls from 10.64 to 8.31 s and from 72,560 to 11,156 kernel
launches. Routed IQ2 up/down take 1.65/1.02 s. Shared Q8 up/down take
1.92/0.56 s; the shared-up path is now the largest single expert operation.
These timings are diagnostic; only the unprofiled samples enter comparison.

The 512/1025-token sweep matches independent cold frontiers exactly across
all 200058 logits and eight following greedy tokens. The fixture pins and
records its chunk cap; the original 64/129-token fixture remains available.
Short native full/chunk/decode and accepted-prefix state checks, eight MTP
cycles, and image/audio session checks also pass. These short state checks
and text boundary proofs do not qualify long-context or concurrent serving.

At context 2113, native session allocation exactly matches its estimate:
181,045,504 → 468,718,848 bytes for the base session, and 283,648,512 →
881,015,296 bytes with MTP. Increasing the cap therefore costs 274.35 MiB
or 569.69 MiB respectively. Both caps pass session/media checks; both MTP
runs pass eight cycles with exact target logits/KV/convolution state.

Raw chunk evidence is under `scratch/inkling-perf/r5/`; the directory retains
its original experiment number although this is the sixth retained prefill
change. Boundary and detailed topology receipts are in `resume-codex/`.

## Round 7: shared Q8 up payload reuse

Shared Q8 up took 1.92 s, the largest single expert operation after round 6.
The warp-owned tile repeatedly loaded each two-byte-aligned weight fragment
across columns. Four cooperating warps now reuse each fragment across eight
assignments and two output rows, while preserving the original K partitions,
FP32 multiply/FMA chains, ascending inter-warp merge, XOR tree and finite guard.
The path consumes canonical Q8_1 activations directly and avoids a separate
SoA transformation for these calls. It uses the existing workspace and stream.

Dispatch requires Q8_0, two experts, two used experts and at least 16 prompt
rows: only shared up meets this topology. Shared down and width-one decode
keep their previous paths. `DS4_INKLING_NO_SHARED_Q8=1` restores the round-6
shared-up tile; `DS4_INKLING_NO_MOE_TILE=1` also disables this specialization.

| Path | Prefill samples (tok/s) | Decode samples (tok/s) |
| --- | --- | --- |
| Chunk-512 control | 248.92 / 247.48 / 247.74 | 17.95 / 17.95 / 17.95 |
| Shared-up candidate | 260.31 / 259.32 / 259.25 | 17.95 / 17.95 / 17.94 |
| Shared-up rollback | 248.49 / 247.38 / 247.05 | 17.95 / 17.94 / 17.96 |

The candidate median is **259.32 prefill / 17.95 decode tok/s**, 4.7% more
prefill throughput than round 6 and 4.8% more than the fresh rollback.
Both comparisons report `Improved`, each checking 1,200,348 logits with
max_abs=0 and no token mismatches. Decode and first-decode-step medians are
unchanged. This retains the fourth additional prefill improvement after round 3.

In the full-model trace, shared Q8 up falls from 1.916 to 1.535 s across
160 calls, about 20% less kernel time. Prefill wall falls from 8.31 to 7.96 s;
kernel count falls from 11,156 to 10,996 by eliminating 160 SoA transforms.
The trace reports 64 registers per thread and 6,144 bytes of static shared memory.
These profile measurements explain the change; the table uses unprofiled TPS.

The component probe used production geometry, preallocated workspace,
alternating timing order and cached occupancy queries on both paths. At 512
tokens, the eight-column shared-up candidate took 9.496 ms versus 11.998 ms.
The same approach regressed shared down (3.553 to 5.210 ms), and a 16-column
up variant also regressed; neither is enabled. All candidates were byte-exact.

Native tests cover 15/16/17-token dispatch boundaries, production 64/512-token
shapes, repeated/invalid routes, aliases, workspace bounds and both controls.
Actual 18-token full/chunk/decode logits and accepted-prefix state 1–9 match
the rollback exactly. Context-2113 MTP session/media checks, eight MTP cycles,
the 512/1025-token cold/replay boundary proof and 53 `ds4-perf` tests pass.
Evidence and source/build hashes are in `scratch/inkling-perf/resume-codex/r7/`.
The initial model-free test launch inherited the live owner's IPC environment;
that failed invocation is retained, and the clean-environment retry passed.

Round-7 repository checks pass: `cargo fmt`, workspace Clippy, all eight C/Rust
host parity targets, serialized `cargo test --workspace --locked`, and
`cargo check --workspace --all-targets --locked`. They ran after the timed
scouts, without the live IPC environment. Logs are in
`scratch/inkling-perf/resume-codex/final-checks/`.

## Round 8: wider chunk validation

A new, fixed three-round prefill campaign starts at `450fea0`: rounds 8–10.
Round 8 tests chunk capacity; the remaining two rounds target kernels with
an explicit 8192 chunk. No further chunk sweep is included.

The supported maximum rises from 2048 to 8192. This is a maximum batch width,
so shorter prompts remain valid. Host sizing, CUDA wrappers, MMVQ assignment
bounds and `ds4-perf` validation use the same limit. **The default stays 512:**
the 8192 default candidate failed the speed gate.

The extended workload uses the same raw prompt and 64 greedy output tokens,
with 8192 input tokens and context allocation 8257. It retains the same owner,
artifacts, warmup/fresh-process policy and three repeats. The 2K regression
workload remains separate.

| Input / chunk | Prefill samples (tok/s) | Decode median (tok/s) |
| --- | --- | --- |
| 8K / 512 control | 230.99 / 230.71 / 230.61 | 14.20 |
| 8K / 8192 candidate | 217.02 / 215.90 / 214.31 | 14.21 |
| 2K / 512 control | 260.54 / 259.42 / 259.15 | 17.94 |
| 2K / 8192 candidate | 263.04 / 261.95 / 262.44 | 17.95 |

At 8K, increasing the chunk alone reduces median throughput **6.4%**;
`ds4-perf compare --regression` reports `Regressed`. The separate 2K
comparison reports `Pass`: its 1.2% median gain does not exceed the robust
gain threshold across the sample ranges. Each comparison checks 1,200,348
logits with max_abs=0 and zero token differences. An earlier 1024-chunk
pilot at 2K measured 264.37 versus 259.42 tok/s with exact proof; it is
intermediate round-8 evidence, not another round or the selected default.

The 8K trace explains the next kernel target: ordinary BF16 projections take
3.031 s across 3360 calls at chunk 512, versus 5.445 s across 210 calls at
chunk 8192. Reducing launch count alone does not offset the larger working
set. Total profiled prefill wall time rises from 35.800 to 37.953 s. These
trace times are separate from the unprofiled throughput table.

Native dense-Q8, routed-expert, BF16 and attention tests cover 8191/8192
rows, incomplete tiles and grid stride. Cold/replayed 8192/16385-token
frontiers are exact. At context 8257 and chunk 8192, base graph allocation
matches its quote of 5,576,422,656 bytes; MTP matches 11,348,081,152 bytes.
Base/MTP session, media and accepted-prefix checks pass, including short
prompts with the large configured cap.

The first MTP fixture allocated an unnecessary second 8K reference graph
and tripped the memory-pressure guard. Its failed evidence is retained.
The reference now allocates only its 18-token transcript plus verification
margin; the target remains at context 8257/chunk 8192. The bounded retry
passes exact logits, KV and convolution state. The resident owner survived.
This is fixture memory repair, not a runtime memory reduction.

Evidence is under `scratch/inkling-perf/extra-three/r8/` and `r8-wide/`.
The latter's `run.status` records the expected nonzero exit after the failed
speed gate; `wide-compare/compare.json` records `Regressed`.
`completion.json` records the completed experiment and rejected default.

## Round 9: Q3 routed-up payload reuse, not retained

At an explicit 8192 chunk, layer 40's routed Q3_K up projection consumed
0.799 s, about 2.1% of prefill wall time. Its fallback repeated weight
fragment decoding across columns and performed an unused SoA conversion.
The candidate shares decoded payloads across eight assignments, consumes
canonical Q8_1 input directly, and preserves all four-warp reductions.
Dispatch was restricted to 256 experts, six routes, 4096-by-4096 weights
and at least 2048 prompt rows. Smaller component workloads regressed.

The component probe, including activation/routing preparation, improved
8K from 789.300 to 498.036 ms and 2K from 196.062 to 125.571 ms. It remained
byte-exact. The four-column alternative was slower than the eight-column
candidate. Full-shape 2047/2048/2049/8192 tests, invalid routes, workspace
bounds, nonblocking stream and rollback tests all pass.

| 8K input / chunk 8192 | Prefill samples (tok/s) | Decode median (tok/s) |
| --- | --- | --- |
| Preceding round-8 control | 217.02 / 215.90 / 214.31 | 14.21 |
| Q3 candidate | 219.18 / 216.73 / 216.34 | 14.20 |

The full-model median improves only 0.4%, from 215.90 to 216.73 tok/s,
with overlapping sample ranges. The candidate is not retained; an isolated
kernel gain is insufficient. The same-hour comparison uses the preceding
fresh-process control, identical prompt/artifacts and an unchanged chunk.

In the full-model trace, Q3 falls from 798.942 to 498.192 ms, while total
prefill wall time falls only from 37.953 to 37.710 s. Kernel count drops
10681 to 10679 by removing the unused relayout and duplicate tile table.
The candidate uses 94 registers per thread, with zero local bytes
reported by Nsight. This confirms the local gain without establishing a
sufficient end-to-end improvement. Raw source, binary hashes and rejected
candidate evidence remain under `scratch/inkling-perf/extra-three/r9/` and
`q3-probe/`.

`ds4-perf compare` reports `Pass`, not `Improved`: 1,200,348 logits
checked with max_abs=0 and zero token differences. No additional Q3
variant or speed retest is included in the fixed campaign.

## Round 10: BF16 token panels above 4096 rows

At the explicit 8192 chunk, ordinary BF16 projections took 5.445 s across
210 calls versus 3.031 s across 3360 calls at chunk 512: each 16-token group
swept every weight row before the next group, so an 8192-row input slab no
longer stayed in L2. The candidate changes only the CUDA tile job order.
Neighbouring jobs now visit output rows within internal 512-token panels,
so each panel's input stays resident while all weight rows pass once. Every
output keeps its lane K stripe, FMA chain, XOR tree and BF16 store, so
results are byte-identical. The job count is unchanged; the final partial
panel uses its actual width, without padding or duplicate outputs.

Dispatch requires at least 4097 rows, 4096 input width and 512/1024/4096
output width, the q/k/v/r/o shapes. Inputs of 4096 rows or fewer and all
other shapes keep the original schedule, so the chunk-512 release path is
unchanged. `DS4_INKLING_NO_LINEAR_PANEL=1` restores the original order.
The configured prefill chunk is not changed by this round.

Native component timings for the 4096-by-4096 projection: 2048 rows
14.09 to 14.01 ms and 4096 rows 64.6 to 64.4 ms (unchanged schedule),
4097 rows 65.9 to 31.3 ms and 8192 rows 110.0 to 65.3 ms. At 8192 rows,
the 1024-wide output falls 32.0 to 16.4 ms and the 512-wide output 16.6 to
8.9 ms. An initial 2049-row minimum regressed 2049 rows from 13.06 to
15.09 ms and was narrowed before any full-model scout.

| 8K input / chunk 8192 | Prefill samples (tok/s) | Decode median (tok/s) |
| --- | --- | --- |
| Panel rollback | 216.70 / 212.95 / 213.43 | 14.21 |
| Panel candidate | 233.42 / 232.49 / 230.02 | 14.21 |

The median improves **8.9%** at chunk 8192, from 213.43 to 232.49 tok/s.
`ds4-perf compare` reports `Improved`, checking 1,200,348 logits with
max_abs=0 and zero token differences; decode and first-step medians are
unchanged. Against the round-8 chunk-512 control (230.71 tok/s) the 8192
chunk is now on par, not robustly faster, so **the default remains 512**.
The panel order is retained as the wide-input kernel path. In the trace,
BF16 falls from 5.451 to 3.080 s and prefill wall from 38.186 to 35.857 s
with an unchanged kernel count.

Native tests cover 2047/2048/2049/4096/4097/8192 rows, the three wide
output widths, NaN-poisoned outputs before selected and rollback calls,
partial aliases and unsupported shapes. Evidence is under
`scratch/inkling-perf/extra-three/r10/`; `user-stop-2107/` retains the
interrupted build that preceded the resumed run.

## Round 11: attention grouped by KV head

The fixed three-round campaign ended with round 10. At the user's request,
further prefill rounds continue under the same protocol: the 8K input at the
release chunk 512 is the primary comparison and the 2K workload is the
regression check. Both use the same owner, artifacts and proof.

At 8K input and chunk 512, attention took 6.405 s of the 35.8 s prefill
trace: 4.4 s in the seven global layers and 2.0 s in the 35 local layers.
The release kernel gave each (query, head) row its own CTA, so the four
query heads of a KV head re-read the same K/V rows, every element paid a
64-bit ring modulo, and each key's loads stalled before the next key's could
issue. Counters showed about 118 instructions per (head, key) pair with 57%
of issue slots stalled on memory.

One CTA now owns a (query, KV head) pair. Its four warps keep the release
key phases, but each warp scores the four query heads together, so K/V are
read once per four heads. Keys stay 64-bit while row offsets and distances
use 32-bit arithmetic, the ring slot advances without a modulo, and the next
key's K/V and biases are prefetched while the current key is scored. Per (head, key) the FMA chain, XOR tree, score,
online-softmax update and four-warp merge are unchanged, so outputs are
byte-identical. A bound of six CTAs per SM keeps 80 registers without
spills; an eight-CTA bound spilled and was slower.

Dispatch requires at least 16 rows; decode and MTP verify widths keep the
per-head kernel. `DS4_INKLING_NO_ATTN_GROUP=1` restores it at every width.

Native component timings at position 7680: local 512 rows 3.47 to 1.57 ms,
global 512 rows 97.2 to 22.4 ms and global 16 rows 2.58 to 1.27 ms. The
attention test cross-checks 1/7/15-row chunks on the per-head kernel against
16-row and full-prompt chunks on the grouped kernel, and the rollback against
the grouped baseline, all exact, with the existing FP64 probes, captured
replays, ring wrap and rejected-suffix checks.

| Input / chunk 512 | Prefill samples (tok/s) | Decode median (tok/s) |
| --- | --- | --- |
| 8K grouped rollback | 230.76 / 230.84 / 230.89 | 14.21 |
| 8K grouped candidate | 261.98 / 261.22 / 261.08 | 14.21 |
| 2K grouped rollback | 260.23 / 259.67 / 259.63 | 17.97 |
| 2K grouped candidate | 273.07 / 271.83 / 271.92 | 17.97 |

The 8K median improves **13.2%**, from 230.84 to 261.22 tok/s, and the 2K
median 4.7%, from 259.67 to 271.92 tok/s. Both comparisons report
`Improved`, each checking 1,200,348 logits with max_abs=0 and zero token
differences; decode and first-step medians are unchanged. In the 8K trace,
attention falls from 6.405 to 2.238 s and prefill wall from 35.749 to
31.638 s with an unchanged kernel count. Evidence, counters and the private
probe are under `scratch/inkling-perf/extra-three/r11/`.

## Round 12: dense Q8 prefill tiles

After round 11, the layer 0-1 dense MLP took 1.80 s of the 31.6 s prefill
trace across 4096 launches. The aligned Q8 vec kernel computed eight tokens
per launch, so every 512-token chunk streamed each up and down weight
matrix 64 times: about 438 GB per 8K prefill, which is the measured device
bandwidth for 1.8 s.

A prefill tile keeps each aligned weight row's codes and scales in
registers while eight-token groups stream through shared memory, so a
weight row is read once per call. Up (K 4096) keeps two rows per warp; down
(K 16384) keeps one row per warp and stages each group in two K slices.
Every output keeps the vec kernel's lane-per-block chain, dp4a order,
scale expression and shfl_down tree, so results are byte-identical. All
rows of a call are quantized in one launch, which yields the same Q8_1 bytes
as the eight-row launches.

Dispatch covers 1 to 8192 rows at the two Inkling widths; other shapes,
devices whose dynamic shared-memory opt-in is below the 73,728-byte down
tile and `DS4_INKLING_NO_Q8_TILE=1` keep the eight-column loop. Native timings at
512 rows: up 38.1 to 8.9 ms, down 21.6 to 9.4 ms; the batch test checks
2 to 8192 rows, the rollback, kill switches and rejected shapes exactly.

| Input / chunk 512 | Prefill samples (tok/s) | Decode median (tok/s) |
| --- | --- | --- |
| 8K tile rollback | 261.79 / 261.46 / 260.90 | 14.21 |
| 8K tile candidate | 273.50 / 272.76 / 272.06 | 14.21 |
| 2K tile rollback | 272.91 / 271.70 / 271.64 | 17.96 |
| 2K tile candidate | 284.53 / 283.82 / 283.40 | 17.96 |

The 8K median improves **4.3%**, from 261.46 to 272.76 tok/s, and the 2K
median 4.5%, from 271.70 to 283.82 tok/s. Both comparisons report
`Improved`, each checking 1,200,348 logits with max_abs=0 and zero token
differences; decode and first-step medians are unchanged. In the 8K
trace the dense MLP falls from 1.802 to 0.571 s, quantize launches from
6672 to 2640 and the kernel count from 42,976 to 34,912; prefill wall falls
from 31.564 to 30.323 s. Evidence and the private probe are under
`scratch/inkling-perf/extra-three/r12/` and `r12d/`.

With this binary, an explicit 8192 chunk measured 277.34 / 276.21 / 270.82
tok/s at 8K: a 1.3% median gain over the chunk-512 candidate with
overlapping samples. `ds4-perf compare` reports `Pass`, not `Improved`, and
the wide chunk needs 5.58 GB of graph scratch at context 8257, so **the
default remains 512**. Evidence: `scratch/inkling-perf/extra-three/r13/`.

Two review fixes follow the round-12 measurement: the grouped attention
loop keeps 64-bit `query`/`first` and steps over a 32-bit distance with a
wrap-safe guard, with a native case ending exactly at UINT32_MAX, and the
dense Q8 tile returns to the eight-column loop when the device's dynamic
shared-memory opt-in is below the down tile. The fixed binary measures
272.48 / 271.54 / 271.42 tok/s at 8K (`Pass` against the round-12
candidate, exact logits and tokens; `r13/fix2/`). A first attempt with a
fully 64-bit key loop measured 255 tok/s and was discarded (`r13/fix/`).

## Reproduction and evidence

Build with `make -j2 ds4-bench-perf ds4-perf CUDA_ARCH=sm_121` after configuring
the checkout's normal CUDA/Rust toolchain. Start a guarded full base+MTP VMM
owner and export its `DS4_CUDA_WEIGHT_IPC_MANIFEST` and scope `both`. Use the
same owner, artifact mappings and guard for both sides:

```sh
./ds4-perf scout --out scratch/inkling-candidate --collector nsys --fit \
  --calibration scratch/inkling-machine/calibration.json \
  --proof --repeats 3 --cache-policy warmup-then-fresh \
  --workload scratch/inkling-workload.json -- \
  ./ds4-bench-perf --cuda -m "$INKLING_MAIN" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2113 --gen-tokens 64
```

The workload manifest must include all six shards, prompt, IPC manifest,
MTP sidecar, tokenizer config and Jinja sidecar; the first shard's key is
`model`. Add `--env DS4_INKLING_NO_LINEAR=1` before `--` for the BF16 control,
`--env DS4_INKLING_NO_MOE_BATCH=1` for the expert-batch rollback control,
`--env DS4_INKLING_NO_Q8_BATCH=1` for the dense-Q8 rollback control, or
`--env DS4_INKLING_NO_MOE_TILE=1` for the expert-tile rollback control, or
`--env DS4_INKLING_NO_LINEAR_TILE=1` for the BF16-tile rollback control, or
`--env DS4_INKLING_NO_SHARED_Q8=1` for the shared-up rollback control, or
`--env DS4_INKLING_NO_LINEAR_PANEL=1` for the BF16-panel rollback control, or
`--env DS4_INKLING_NO_ATTN_GROUP=1` for the grouped-attention rollback control, or
`--env DS4_INKLING_NO_Q8_TILE=1` for the dense-Q8-tile rollback control.
Use `--env DS4_INKLING_PREFILL_CHUNK=64` to restore the preceding chunk cap.
See [ds4-perf](ds4-perf.md) for calibration, workload schema, memory guards
and `compare --regression`.

Raw evidence is retained under `scratch/inkling-perf/`: `baseline/`,
`round1/`, `round1-control/`, `round2/`, `round2-control/`, `round3/`,
`round3-control/`, their comparisons, binary/source hashes, memory logs and
`linear-*` / `batch-*` / `q8-*` component/state logs. The repeated investigation
is in `deep-current/`, `deep-repeatability/`, `deep-vs-control/` and
`deep-targeted-ncu/`; decomposition and exact counter commands are retained.
Rounds from four onward use a new resident owner; their fresh controls,
candidates, rollback controls and comparisons are under `r4/` and later
per-round directories with their own workload manifest.
Initial fixture failures, the invalid first workload manifest and
the corrected expert-test build typo are retained separately.

| Identity | SHA-256 |
| --- | --- |
| Corrected control executable | `6d5f2f114437dce760fe36cafcbc2496f51bd88d1941e7f59c518ebf2b40133c` |
| BF16 executable | `7c97e70f9feab2fd916dd65a4ddf9f1edf0afe7c9b7faee6bb8d1cd15b6f2f59` |
| Expert batch executable | `5dc6b64ebc3e23ef1c5ae808580e200ca0a4d33206447420787122d7ad88073f` |
| Dense Q8 executable | `65766c3d6af9490875c4738306dea3ef9f0026dc19198bab8b45b11e4bce07b8` |
| Expert tile executable | `036a9c986739d5b36166a6d759d290f6680c24873d6ff11a2b04b96ae8dbc6d9` |
| BF16 tile executable | `82c0f12a2c77bf25781fafa6d75e49a975176cf4a0f873f2715f6e6f0c7166fc` |
| Chunk-512 executable | `27c8f157dcd0475754b48602e2ee2f9f49e30dea30457e96c9f3b416165de121` |
| Shared Q8 up executable | `66aac1ebaba2326865e9a80cbee2ae552e4b37d546cc9c8a69a1f6e7f62b3594` |
| BF16 panel executable | `fb87ed77ea6cf92aa206cb3393ced758cf904f79d682b9e32561245e58495c36` |
| Grouped attention executable | `cfffb1723f4d24d650fe4edd7c2a2b9b3946c7a6b87c351be1eab1683455274b` |
| Dense Q8 tile executable | `d2c9dfca3e25f2b75ce3cde57aab92e948f6cbfa15357ef8ce3893eeb9631bb6` |
| Review-fix executable | `0ab5bb013f1c503f22e27971d336b1e30fa470ba827af82f0bd53a7c62628ab5` |
| Prompt | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Baseline–round-3 IPC manifest | `4f9e46dce133c5a14bf85f3ecd71e0437b27bbcaad38a1c679d4aebd3b5a8de8` |
| Round-4–7 IPC manifest | `43b795a0d21d293af31ca3fdca0a30402ee464a58279e7b9c04f41432d0583e7` |
| Frontier proof JSON | `34867789bafce5999ea77da41112db7e77f866aea4aef234f0b4b85510dd2587` |
| Token proof JSON | `867d71cc5e5221be944f879c86eed3f024dd602aaacfa17a463e23a034c1b8ed` |
