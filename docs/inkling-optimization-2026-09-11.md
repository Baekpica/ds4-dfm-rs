# Inkling MQ85GB prefill on GB10, 2026-09-11

Continuation of [rounds 1–12](inkling-optimization-2026-09-10.md), starting
from merged runtime `49895b8` (the tree of reviewed round-12 `4c32556`).
Continued in [rounds 16–18](inkling-optimization-2026-09-11-r16.md).
All measurements use MQ85GB on one DGX Spark / GB10, CUDA 13.3.73,
driver 610.43.02, compiled for `sm_121a` through `CUDA_ARCH=sm_121`.

## Three retained rounds

These cumulative comparisons use the final binary with all three new
switches set and with defaults. The controls restore the prior merged
runtime's retained paths, using fresh runs with the same fixture and
artifacts. Their medians are separate from the per-round controls below.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 271.21 → 312.88 | +15.36% | 14.21 → 14.22 | Improved |
| 2,048 | 283.41 → 327.86 | +15.68% | 17.96 → 17.97 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`, no bad values)
and has zero generated-token mismatches. Three throughput samples per side:

- 8,192 control: 272.26 / 271.21 / 271.15; default: 313.03 / 312.88 / 312.29 tok/s.
- 2,048 control: 284.42 / 283.41 / 283.11; default: 329.42 / 327.86 / 327.59 tok/s.

Retained commits: R13 [`4081119`](https://github.com/Baekpica/ds4-dfm-rs/commit/4081119a9efa718188a03eeb6ab86552ed51a053), R14 [`5509610`](https://github.com/Baekpica/ds4-dfm-rs/commit/550961016d4b8f32149f6bd96c9eb3d02a20accc), R15 [`f7881de`](https://github.com/Baekpica/ds4-dfm-rs/commit/f7881de7ef83596ae9c39958001c4c5ac6cdba05).

## Protocol

- `speed-bench/promessi_sposi.txt`, 8192/2048 input tokens, 64 greedy output
  tokens, MTP off, default prefill chunk 512, context allocation input+65.
- Three unprofiled fresh processes per side, each preceded by a separate
  warmup, same resident base+MTP VMM owner and default aligned Q8 artifacts.
- `ds4-perf scout --proof --repeats 3 --cache-policy warmup-then-fresh`,
  Nsight Systems and fit collected separately from speed samples. All six
  main shards, MTP, prompt, template/tokenizer files and IPC manifest hashed
  before and after. Guard max/high 12/10 GiB, host reserve 12 GiB.
- Each retained round compares one new diagnostic switch with the default
  on the same binary. Full-vocabulary logits and generated IDs must match;
  prefill, decode and first-token latency pass `compare --regression`.
  Sample envelopes describe observed variation, not confidence intervals.

Temperature/clock samples are retained in `thermals.csv`; no clock or
power policy was changed for this campaign.

## Round 13: shared Q8 up weights retained across columns

The entry trace spent 6.195 s (20.4% of 30.415 s) in shared Q8 up.
The existing eight-column CTAs reloaded the weight rows for every input
window. Each new CTA owns 16 output rows and keeps their weight fragments
in registers while eight routed inputs at a time pass through shared
memory. Every warp replays the original four K partitions and four FMA
steps, ascending partial merge, XOR tree and nonfinite guard.

The path requires two shared experts, Q8_0, K4096 and at least 64 prompt
rows. Misaligned Q8_1 input or insufficient shared memory falls back;
`DS4_INKLING_NO_SHARED_TILE=1` restores the prior shared-up kernel.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 271.54 → 299.53 | +10.31% | 14.20 → 14.22 | Improved |
| 2,048 | 283.10 → 313.14 | +10.61% | 17.96 → 17.95 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`, no bad values)
and has zero generated-token mismatches. Three throughput samples per side:

- 8,192 control: 271.70 / 271.54 / 271.04; default: 300.86 / 299.53 / 298.48 tok/s.
- 2,048 control: 284.31 / 283.10 / 283.01; default: 315.26 / 313.14 / 312.94 tok/s.

The same-binary 8K traces show shared up 6.165 → 3.392 s and prefill wall
30.396 → 27.581 s, with 34,912 → 34,272 kernel calls. The tile removes one
worklist-kernel launch per shared-up call. It uses 255 registers per thread
and no local memory in the trace. The isolated production-shape probe at
512 tokens measured 9.054 to 5.279 ms; it is separate from model throughput.

Native checks cover ragged 126-row matrices, 63/64/65 and 8192 tokens,
repeated/invalid routes, poisoned destinations, workspace bounds, a
nonblocking stream and a four-byte-aligned input subview that must fall back.
The 65-token full forward matches one-token decode and chunks 2/3/7 exactly
in logits, hidden state, KV and convolution; accepted-prefix restores 1–9
also match. The unchanged narrow paths remain the decode/MTP controls.

Benchmark binary SHA-256:
`fd41cd67179f3a11446eced27d2f1ea379550eb66663bf7c0a80a7f0bb2f1465`.

## Round 14: shared Q8 down weights retained across columns

After round 13, shared down still used the general warp tile: each input
window reloaded weights and required an activation relayout and worklist.
The new down specialization retains K2048 weights and streams eight inputs
through the same shared slab. It preserves the original one-warp, eight-step
FMA chain and removes both preparatory kernel launches.

The threshold is 128 assignment rows, or 64 prompt tokens with two shared
experts. Alignment/shared-memory fallback remains available;
`DS4_INKLING_NO_SHARED_DOWN_TILE=1` selects the prior path. Shared up keeps
its round-13 implementation and narrow decode/verification is unchanged.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 299.54 → 306.98 | +2.48% | 14.21 → 14.20 | Improved |
| 2,048 | 313.60 → 322.14 | +2.72% | 17.94 → 17.96 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`, no bad values)
and has zero generated-token mismatches. Three throughput samples per side:

- 8,192 control: 300.51 / 299.54 / 298.95; default: 308.76 / 306.85 / 306.98 tok/s.
- 2,048 control: 314.97 / 313.60 / 313.19; default: 323.05 / 321.93 / 322.14 tok/s.

Shared down in the same-binary 8K trace: 2.232 → 1.508 s
(8.1% of control prefill);
prefill wall 27.561 → 26.900 s,
kernel calls 34,272 → 32,992.
The new kernels use 128 registers per thread and 0 local bytes in the trace.

The isolated 512-token down probe measured 3.448 → 2.355 ms with exact
outputs. Native batch checks extend the shared-up cases to shared down,
including 63/64/65 and 8192 tokens, 126 output rows, repeated/invalid routes,
nonblocking streams and four-byte input alignment. Full 65-token forward,
chunks 2/3/7 and accepted-prefix 1–9 all match logits, hidden, KV and
convolution state exactly.

Benchmark binary SHA-256: `6e809e6b7f0656c0b35ee52d22583eb5e3835d54dddbc0067f3968b47c8de9fc`.

## Round 15: Q4_K expert prefill coverage

The three Q4_K projections in layers 40–41 still fell back to four-column
MMVQ after building an unused activation relayout and eight-column worklist.
The new eight-column tile loads each payload fragment and unpacks its scales
once, then reuses them across inputs. Up keeps four ordered warp partitions
and two K steps; down keeps one partition and four steps. Lane dot products, ordered
sums, XOR reduction and nonfinite handling match the retained oracle.

The path starts at 3072 assignments, or 512 prompt tokens with six routed
experts. `DS4_INKLING_NO_Q4_TILE=1` restores the old path. Smaller widths keep
their dispatch; the isolated 64-token up candidate regressed and was not
enabled. Shared Q8 and IQ2 are unchanged in this round.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 307.82 → 312.88 | +1.64% | 14.21 → 14.22 | Improved |
| 2,048 | 322.32 → 327.86 | +1.72% | 17.96 → 17.97 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`, no bad values)
and has zero generated-token mismatches. Three throughput samples per side:

- 8,192 control: 308.22 / 307.82 / 307.59; default: 313.03 / 312.88 / 312.29 tok/s.
- 2,048 control: 323.24 / 322.32 / 321.56; default: 329.42 / 327.86 / 327.59 tok/s.

Q4_K expert kernels in the same-binary 8K trace: 1.273 → 0.832 s
(4.7% of control prefill);
prefill wall 26.835 → 26.437 s,
kernel calls 32,992 → 32,896.
The new kernels use 72/110 registers per thread and 0 local bytes in the trace.

Isolated 512-token probes measured up 43.865 → 28.179 ms and down
22.747 → 12.930 ms. Native tests cover 511/512/513 and 8192 tokens, ragged
126-row matrices, random/repeated/invalid routing, full 4096-row matrices,
nonblocking streams and four-byte Q8_1 input alignment. The full-model
forward gate uses 515 tokens to enter the new path, with exact logits,
hidden, KV and convolution versus single-token decode and chunks 2/3/7.
The committed KV/convolution snapshot checks 93,327,360 bytes exactly;
accepted-prefix 1–9 checks also pass.

Benchmark binary SHA-256: `ba07a8f4c0e22f30daeb0cb8063f5e96ce200c2341fe306aaf604f46944d2bb6`.
## Rejected probes

Six IQ2 tile/occupancy variants failed to establish a gain at the 512-token
fixture. Constraining four-row tiles to six or eight resident CTAs spilled;
two-row/eight-column tiles avoided spills but did not improve the measured
up/down pair. None was integrated or counted as a retained round.

## Final validation

- Native batch and full-forward gates pass for every round, including the
  515-token Q4_K threshold check and exact committed KV/convolution state.
- `tests/test_inkling_session` passes with the shared MQ85GB + eight-layer
  MTP-BF16 owner: lifecycle, accepted greedy tokens and committed target state,
  image/audio identity and invalid-input state. This short-context regression
  does not measure MTP throughput or repeat the historical HTTP checks.
- `cargo fmt`, workspace clippy, all eight host parity targets, serialized
  workspace tests, `make test-ple-formats` and all-target workspace check pass.
- The four default Rust hosts are relinked; CLI/bench/agent `--help` and
  server `--version` startup checks succeed.
  No native host/control-plane boundary or weight payload changes.

## Evidence

Raw commands, hashes, guards, proofs, traces and comparisons are retained
under `scratch/inkling-perf/next-three/`, including `baseline/`, `r13/`,
`r14/`, `r15/`, `cumulative/`, `final-checks/`, `final-session/`,
`final-hosts/`, `iq2/`, `shared/`, `q4/` and `thermals.csv`. The workload manifests
are the unchanged `scratch/inkling-perf/extra-three/workload-8k.json` and
`scratch/inkling-perf/r4/workload.json`. These are MQ85GB text-prefill
measurements, not source-model, long-context, media-performance or MTP
throughput qualification. [HTTP/media scope](inkling-small.md) remains
separate.
