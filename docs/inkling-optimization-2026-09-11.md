# Inkling MQ85GB prefill on GB10, 2026-09-11

Continuation of [rounds 1–12](inkling-optimization-2026-09-10.md), starting
from merged runtime `49895b8` (the tree of reviewed round-12 `4c32556`).
All measurements use MQ85GB on one DGX Spark / GB10, CUDA 13.3.73,
driver 610.43.02, compiled for `sm_121a` through `CUDA_ARCH=sm_121`.

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

## Rejected probes

Six IQ2 tile/occupancy variants failed to establish a gain at the 512-token
fixture. Constraining four-row tiles to six or eight resident CTAs spilled;
two-row/eight-column tiles avoided spills but did not improve the measured
up/down pair. None was integrated or counted as a retained round.

## Evidence

Raw commands, hashes, guards, proofs, traces and comparisons are retained
under `scratch/inkling-perf/next-three/`, including `baseline/`, `r13/`,
`iq2/`, `shared/`, `q4/` and `thermals.csv`. The workload manifests are the
unchanged `scratch/inkling-perf/extra-three/workload-8k.json` and
`scratch/inkling-perf/r4/workload.json`. These are MQ85GB text-prefill
measurements, not source-model, long-context, media-performance or MTP
throughput qualification. [HTTP/media scope](inkling-small.md) remains
separate.
