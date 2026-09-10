# Inkling MQ85GB optimization on GB10

This campaign starts at merged v0.1.2 main `10658bdf` and keeps the default
aligned dense-Q8 numerical path. Results apply to the six-shard MQ85GB
artifact on one DGX Spark. They do not establish MQ89, Q8_0-main, independent
source parity, long-context serving or concurrent-request performance.

## Expanded target

After the aligned Q8 batch candidate reached 111.53 prefill tok/s, the
campaign target expanded to another detailed `ds4-perf` investigation and
**at least three additional prefill improvements** from that state. The
original three decode improvements remain required. A retained improvement
to the routed IQ2 kernels is also mandatory. The first three
prefill candidates do not count toward this additional target. Publication
waits for the complete campaign and its correctness/regression checks.

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
or `--env DS4_INKLING_NO_Q8_BATCH=1` for the dense-Q8 rollback control.
See [ds4-perf](ds4-perf.md) for calibration, workload schema, memory guards
and `compare --regression`.

Raw evidence is retained under `scratch/inkling-perf/`: `baseline/`,
`round1/`, `round1-control/`, `round2/`, `round2-control/`, `round3/`,
`round3-control/`, their comparisons, binary/source hashes, memory logs and
`linear-*` / `batch-*` / `q8-*` component/state logs. The repeated investigation
is in `deep-current/`, `deep-repeatability/`, `deep-vs-control/` and
`deep-targeted-ncu/`; decomposition and exact counter commands are retained.
Initial fixture failures, the invalid first workload manifest and
the corrected expert-test build typo are retained separately.

| Identity | SHA-256 |
| --- | --- |
| Corrected control executable | `6d5f2f114437dce760fe36cafcbc2496f51bd88d1941e7f59c518ebf2b40133c` |
| BF16 executable | `7c97e70f9feab2fd916dd65a4ddf9f1edf0afe7c9b7faee6bb8d1cd15b6f2f59` |
| Expert batch executable | `5dc6b64ebc3e23ef1c5ae808580e200ca0a4d33206447420787122d7ad88073f` |
| Dense Q8 executable | `65766c3d6af9490875c4738306dea3ef9f0026dc19198bab8b45b11e4bce07b8` |
| Prompt | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Shared IPC manifest | `4f9e46dce133c5a14bf85f3ecd71e0437b27bbcaad38a1c679d4aebd3b5a8de8` |
| Frontier proof JSON | `34867789bafce5999ea77da41112db7e77f866aea4aef234f0b4b85510dd2587` |
| Token proof JSON | `867d71cc5e5221be944f879c86eed3f024dd602aaacfa17a463e23a034c1b8ed` |
