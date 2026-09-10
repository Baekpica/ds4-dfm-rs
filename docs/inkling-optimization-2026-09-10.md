# Inkling MQ85GB optimization on GB10

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
this report claims no decode speedup.

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

Final repository checks pass: `cargo fmt`, workspace Clippy, all eight C/Rust
host parity targets, serialized `cargo test --workspace --locked`, and
`cargo check --workspace --all-targets --locked`. They ran after the timed
scouts, without the live IPC environment. Logs are in
`scratch/inkling-perf/resume-codex/final-checks/`.

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
`--env DS4_INKLING_NO_SHARED_Q8=1` for the shared-up rollback control.
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
| Prompt | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Baseline–round-3 IPC manifest | `4f9e46dce133c5a14bf85f3ecd71e0437b27bbcaad38a1c679d4aebd3b5a8de8` |
| Round-4–7 IPC manifest | `43b795a0d21d293af31ca3fdca0a30402ee464a58279e7b9c04f41432d0583e7` |
| Frontier proof JSON | `34867789bafce5999ea77da41112db7e77f866aea4aef234f0b4b85510dd2587` |
| Token proof JSON | `867d71cc5e5221be944f879c86eed3f024dd602aaacfa17a463e23a034c1b8ed` |
