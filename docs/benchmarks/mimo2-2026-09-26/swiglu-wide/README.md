# SwiGLU/Q8 wider blocks — rejected

Round 6 is **rejected**: the isolated gain did not produce consistent
end-to-end improvement. After rejecting round 5, fresh whole-workload
measurement showed 6.71528 s prefill. SwiGLU/Q8 took 254.651712 ms in
94 calls, 3.792% of prefill. [Whole-workload evidence](nsys-before.json).

Baseline full NCU showed long-scoreboard latency and few eligible warps.
The experiment groups more of each 2048-value assignment into one CTA:
128 → 256 → 512 threads reduces CTAs from 131072 → 65536 → 32768.
The kernel body, per-thread arithmetic, 8-lane amax groups and D4 layout
stay unchanged. No extra tensor allocation is needed.

## Isolated result

| Fresh-process sample | Frozen original ms | New 128 ms | New 256 ms | New 512 ms |
|---|---:|---:|---:|---:|
| 0 | 2.690174 | 2.692871 | 2.691133 | 2.674070 |
| 1 | 2.715579 | 2.688034 | 2.696691 | 2.678764 |
| 2 | 2.722384 | 2.725925 | 2.678778 | 2.673309 |
| Median | 2.715579 | 2.692871 | 2.691133 | 2.674070 |

512 threads wins every pair against all three controls. Its median latency
falls 1.529% against the original and 0.698% against same-build 128.
Each process uses 32768 assignments, width 2048, 16 warmups and 100 timed
launches; four-arm order rotates between samples.
[All samples, guards and output hashes](isolated.json).

All four complete 72 MiB D4 dumps compare byte-exact, including 2,097,152
FP32 scales and 67,108,864 quantized bytes. All 12 guard/payload exits are
zero. Synthetic gate/up tensors and deterministic expert-major routing
(102–161 assignments per expert) differ from model values and residency.
The [receipt](receipt.json) records source, binaries and profiler guards.
Busy [clock samples](clocks.json) were 2190 MHz; sampling is one second.

## Full NCU: 128 → 512

| Counter | Baseline | Candidate |
|---|---:|---:|
| Profiled duration | 2.712448 ms | 2.673760 ms |
| Executed instructions | 55,050,240 | unchanged |
| Global load/store sectors | 17,301,504 / 2,883,584 | unchanged |
| Registers/thread; spills | 23; 0 | unchanged |
| Static/dynamic shared bytes | 0 / 0 | unchanged |
| Eligible warps/scheduler | 0.0511 | 0.0641 |
| Achieved occupancy | 93.24% | 81.79% |
| Long-scoreboard ratio per issue | 225.97 | 197.52 |

Eligible warps improve despite lower achieved occupancy. Total L2 sectors
fall about 0.43%; the driver-selected shared-memory carveout changes from
32 to 8 KiB, with unchanged per-block driver overhead. The results
support a modest scheduling/cache benefit but do not isolate its cause.
See [exact counters](counters.json), full
[baseline](baseline-details.txt) / [candidate](candidate-details.txt) details
and their raw CSV exports.

## Prior whole-workload estimate

If the isolated gain transfers exactly and other costs remain unchanged,
the measured 3.792% contribution predicts only **0.058%** prefill throughput
improvement against original, or **0.0265%** against same-build 128.
These were [estimates](whole-estimate.json), not measured model gains.

## Actual model A/B and decision

| Median tok/s | Retained original | New OFF | Default ON |
|---|---:|---:|---:|
| Prefill | 1219.21 | 1218.30 | 1219.23 |
| Decode | 24.45 | 24.42 | 24.40 |

Each arm ran three fresh measured processes, each preceded by a fresh
warmup process, with rotated order and the same resident weight owner.
The [workload](workload.json) pins the four-shard artifact, prompt and
8192-input/128-output protocol. [All measured samples](actual-result.json).

ON is only 0.00164% above original by median. It loses two of three pairs
against same-build OFF, and has mixed pairwise results against original.
Reject because the gain is not consistent, not because it falls below a
fixed percentage floor. Decode ranges overlap; no causal decode penalty
is assigned. The [experimental patch](rejected-wide.patch.gz) was reverted;
the round-3 kernel baseline remains.

All nine measured samples and nine warmups have exactly matching complete
152,576-logit arrays and 128-token streams. All 18 guard/payload exits are
zero. [Full proof records](actual-proof.json). Busy model clock samples
were 2184–2190 MHz; [clock evidence](actual-clocks.json).

Original/OFF/default-ON also match every Down-output byte at the 32- and
8192-token dispatch boundaries, with finite outputs and all six guard
exits zero. These [boundary checks](boundary-proof.json) exercise routing,
D4 production and its consumer. Their timings are correctness-only;
original boundary checks overlapped CPU compilation. The
[final receipt](receipt.json) records build commands, binaries, patch,
boundary test and actual A/B evidence.
