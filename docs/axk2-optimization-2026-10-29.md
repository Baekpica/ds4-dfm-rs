# A.X-K2 prefill/decode campaign — 2026-10-29

Optimization rounds start from `daa0901f861eb19d3d4301123b90666fe032533e`
on `feat/perf-ax-k2`. This record covers the 62-part
[Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)
release, revision `a0c3b02e9b45dc67ef8dd1d48eb48034c93d6607`;
`release.json` SHA256
`3b4da5db6502611afe3ea06d83ef7cb77c2c205a9f3e93c75a2293579735e14c`.

One GB10, 48 SM, CUDA 13.4.92 / `sm_121a`, driver 615.71.10.
Boot `f5d9ee6b-7f0f-443c-b629-4f39b4519994`; the enabled clock service
keeps 300–2200 MHz. Observed busy clocks accompany each round. Actual native
libraries: system libcuda 615.71.10, cuBLAS/cuBLASLt 13.8.0.4 and
CUDA runtime 13.4.92. One model process at a time, CPU affinity 5–9,15–19
(D1's additional whole-binary comparison pins CPU 9),
page cache cleared between processes, 2 GiB host-memory reserve and trip.
No MTP, no drafter, no CPU weight offload.

## Round protocol

Fresh processes alternate off/on with the same executable, artifact and
prompt. A 2,048-token warmup/reset allocates lazy workspace, followed by a
cold 8,192-token prefill and 64 forced greedy tokens in P1 (256 in P2/P3).
This differs from the
published incremental 2K–64K sweep; its numbers are not interchangeable.
Full-vocabulary frontier logits and generated token IDs are retained.

```sh
DS4_AXK2_FLASH_BF16=0 ./ds4-bench --cuda -m /mnt/p31/data/models/A.X-K2-VQ-GGUF \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 8192 \
  --warmup-prefill 2048 --gen-tokens 64 --dump-frontier-logits-dir proof-off
# Repeat with DS4_AXK2_FLASH_BF16=1 in a fresh process.
```

Prompt SHA256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
PDL is enabled for rates and disabled only for Nsight attribution. Kernel
sums use launch correlation IDs inside `ds4.prefill` / `ds4.decode` NVTX
ranges. Isolated NCU uses `--clock-control none`, never the full model.

## P1 — direct BF16 flash output (adopted)

Baseline prefill: 15.883 s wall / 15.842 s kernels. Flash attention is
3.624 s (22.9% of kernel time); `ax_bf16_rows` is 0.553 s (3.5%), including
both query packing and latent-output packing. The 48 MiB latent copy
alone reaches 240–252 GB/s on finite random values: remove the traffic.

Single-split prefill rows >=48 write the value GEMM's BF16 operand directly.
This removes the F32 intermediate write/read and its conversion launch.
The existing consumer rounded at exactly this boundary. Class 5.1: storage
and epilogue fusion, no new arithmetic or allocation; decode and short rows
keep their F32 path. `DS4_AXK2_FLASH_BF16=0` retains the old path.

`tests/test_axk2_flash` checks dense/sparse selection, masked fillers,
causal and partial tiles, row counts 1/48/49/64/255/256 and allocation
canaries. Direct BF16 matches CPU rounding of the original F32 output.
All 2,296 original F32 SASS instructions match after normalizing constant
bank relocation offsets. Synthetic 256-row pipeline timing (flash plus
conversion) is 2.429–2.467 ms off versus 2.030–2.075 ms on. Synthetic
selection/cache distribution differs from a real model; only whole-model
measurements decide adoption.

| Fresh process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 516.23 | 15.41 |
| 2 | on | 533.92 | 15.34 |
| 3 | on | 533.66 | 15.33 |
| 4 | off | 514.55 | 15.46 |
| 5 | off | 513.54 | 15.35 |
| 6 | on | 534.63 | 15.22 |
| Median | off | 514.55 | 15.41 |
| Median | on | 533.92 | 15.33 |

Prefill improves **3.76%**, with nonoverlapping repeat ranges. All six
processes exit 0, all 163,840 logits are finite, and full logits and all
64 generated tokens agree exactly. Busy SM clocks are 2,184–2,190 MHz;
minimum available host memory is 15.608 GiB, full PSI avg10 stays zero.

The short decode median is 0.52% lower. A fresh **512-token** follow-up
pair (on then off, otherwise identical) reverses that direction:

| Mode | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| on | 534.68 | 15.29 |
| off | 516.76 | 15.05 |

Both exit 0 with identical finite logits and all 512 tokens. Prefill again
improves 3.47%; decode differs by +1.59%. Clock policy, busy SM clocks,
CPU affinity (the ten 3.9 GHz cores), model, tokens and PSI match. The
remaining process-to-process decode variation is unresolved; neither this
pair nor the short median establishes a decode gain or a consistent loss.
The unchanged decode kernel's SASS and reversed sign support retaining the
repeatable prefill gain with no identified resource/numerical cost. **This
counts only as one prefill round.** It is not sustained serving qualification.

The on Nsight trace gives 15.375 s prefill wall versus 15.883 s before.
Copies fall from 3,904 to 1,952 calls, 552.6 to 128.5 ms; flash falls from
3,624.2 to 3,488.5 ms. The value cuBLAS kernel now starts before flash ends:
its 790.5 ms raw duration contains 535.6 ms of overlap, leaving 254.9 ms
beyond earlier kernels. Kernel interval union falls 15.840 to 15.332 s.
Do not sum those overlapping durations as extra compute. cuBLAS supports
its own [dependent launches](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-0-update-1);
`DS4_GLMVQ_PDL=0` controls this project's launch wrapper, not cuBLAS.
The profiled decode wall is 1,061.1 versus 1,065.1 ms for 16 tokens.

Measured binary SHA256 (source above plus this round's native diff):

- `ds4-bench`: `3fe81cb61dcde3b7337cc0d58fbf2da4baabe84e8920fdb6821c5c1d3b81d0ea`
- `ds4-bench-perf`: `4cfdcd9d1527adfaa37937a7e19c9416cde795ab49d9e41145c3fc232ce671ad`

Raw receipts, logs, clocks, source digests, full logits/tokens and profiles
remain in ignored `scratch/axk2-perf/`: `p1-{0-0,1-1,2-1,3-0,4-0,5-1}`,
`p1-long-{0-1,1-0}`, `baseline-trace`, `p1-trace`, `p1-timeline.txt` and
`p1-fixture.log`. The final publication sweep and further rounds are pending.

## P2 — cooperative sparse-key metadata (adopted)

Retained P1 attention is 22.69% of prefill wall. Its 256-row NCU probe
spends 64.7% of long-scoreboard samples on the selected-key/exponent chain;
its 73 GFLOP compute floor is 0.677 ms versus 2.096 ms observed. Four
warp lanes now gather four independent keys and exponents together before
the warp copies their latent rows. Class 5.1 scheduling; floating-point
operations and allocations are unchanged. F32 decode stays serial.
`DS4_AXK2_FLASH_KEYS=0` restores P1; this path requires P1's BF16 output.

All six fixture cases pass against CPU-rounded F32, including canaries.
The retained native F32 path keeps all 2,296 SASS instructions. The isolated
BF16 serial times are 2.046923/2.015478/2.043101 ms; cooperative times are
1.985931/1.947278/1.973226 ms (3.42% median reduction). Full NCU reports
2.096 → 2.038 ms, 166.22 → 148.78 million executed instructions, and
400.5 → 400.0 MB of L2-to-crossbar traffic. Both use 255 registers and an
80-byte stack. No additional allocation or floating-point work pays for
the gain. Synthetic selections differ from the full model.

The full-model protocol extends generation to **256 tokens** to reduce
short timing noise. Both arms use 8K cold prefill after the same 2K reset;
P1 stays enabled and only `DS4_AXK2_FLASH_KEYS` changes.

| Fresh process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 534.11 | 15.37 |
| 2 | on | 540.63 | 15.36 |
| 3 | on | 539.42 | 15.51 |
| 4 | off | 534.19 | 15.14 |
| 5 | off | 532.90 | 15.44 |
| 6 | on | 540.46 | 15.37 |
| Median | off | 534.11 | 15.37 |
| Median | on | 540.46 | 15.37 |

Prefill gains **1.19%**, with nonoverlapping repeat ranges and positive
paired gains in all three pairs. Decode medians agree; retain its observed
15.14–15.51 tok/s process spread, without claiming a decode improvement.
All six processes exit 0 with identical finite full logits and all 256
tokens. Busy SM clocks are 2,184–2,190 MHz, CPU frequency is 3.9 GHz,
full memory PSI stays zero, and minimum available memory is 15.687 GiB.
Timed prefill/decode samples show no power/thermal cap. One idle startup
sample reports a power cap at 208 MHz and 0% GPU utilization, 179 seconds
before the frontier dump. NVML does not expose a GPU memory clock here.

The new whole-workload trace reduces prefill wall 15.375 → 15.185 s and
flash 3,488.5 → 3,318.3 ms (4.88%). Kernel interval union is 15.144 s;
512.8 ms of overlapping durations must not be added as extra wall time.
The profiled decode wall is 1,057.2 ms for 16 tokens. These are attribution
measurements, separate from the unprofiled rates above.

Measured `ds4-bench` SHA256:
`7e90986eccb52ba1fbc5b115bee824a126436ceb7b65ac487990075e81e4c26b`;
`ds4-bench-perf`:
`a2bbbd79dcf09bc7dfc5522f75365e08fd4b51498a50d6235b2650d61ed4896f`.
Source is `430b711f` plus this round's native diff. Raw receipts are
`scratch/axk2-perf/p2-*`, including clocks/host telemetry, full outputs,
fixture timing, source/SASS receipts and the new trace. Prefill now has
two adopted rounds; decode rounds and the final publication sweep remain.

## P3 — stage sparse attention metadata (adopted)

Retained P2 flash costs 3.318 s, 21.85% of prefill wall. The isolated
256-row kernel spends 13.46% of all stall samples on key validation and
the dependent exponent lookup; its 73 GFLOP compute floor remains
0.677 ms at 107.8 TFLOPS. A small kernel now gathers each selected
`(key, exponent)` before flash. The packed records occupy the unused tail
of the existing F32-capacity output tensor: 16,384 metadata bytes per row
fit within its 65,536 unused bytes. There is no additional allocation.

This costs one launch per sparse slice and 8 MiB additional logical
metadata traffic per 256-row pipeline. Class 5.1 scheduling: latent values,
accumulation and rounding are unchanged. The measured benefit covers
that staging cost. `DS4_AXK2_FLASH_META=0` restores P2; disabling either
`FLASH_KEYS` or `FLASH_BF16` also bypasses staging. Dense and F32 decode
paths retain their previous schedules.

The six fixture shapes pass CPU-rounded F32 comparison and allocation
canaries, with an added all-masked sparse row. The full pipeline, including
staging, runs 1.843493/1.852742/1.860086 ms versus P2's
1.976979/1.986605/1.990307 ms: 6.74% lower median latency. Full NCU gives
2.038 → 1.810 ms flash plus 14.272 µs staging. Metadata falls to 6.21%
of stall samples. Registers/stack remain 255/80 bytes. Executed
instructions fall 148.78 → 143.06 million, plus 0.52 million in staging;
L2-to-crossbar read traffic is 400.05 → 392.46 + 2.52 MB, including reduced
local loads. This cache-interface metric excludes staging writes and is
not DRAM traffic.

SASS review caught and removed an incidental store-order change in the
fallback. Final native kernels retain all 2,152 cooperative BF16 and 2,296
F32 instructions; the staged kernel matches its isolated probe's 2,088.
Synthetic selections/cache states differ from the model. Full-model
staging costs 32.5 µs per call, above the isolated probe's 14.3 µs.

The same six fresh 8K/256 processes vary only `DS4_AXK2_FLASH_META`:

| Fresh process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 538.15 | 15.35 |
| 2 | on | 547.12 | 15.33 |
| 3 | on | 546.75 | 15.37 |
| 4 | off | 541.45 | 15.38 |
| 5 | off | 538.22 | 15.29 |
| 6 | on | 545.94 | 15.49 |
| Median | off | 538.22 | 15.35 |
| Median | on | 546.75 | 15.37 |

Prefill gains **1.58%** by medians; individual pairs gain 1.67%, 0.98%
and 1.43%. Repeat ranges do not overlap. Decode ranges overlap, so its
0.13% median difference is not counted as a decode optimization. All six
processes exit 0 with identical finite full logits and all 256 tokens.
Busy SM clocks are 2,184–2,190 MHz, CPU frequency is 3.9 GHz, full PSI
stays zero, and minimum available memory is 15.662 GiB. Three idle startup
samples flag a power cap at 208 MHz/0% utilization, 178–180 seconds before
the frontier; timed phases show no power/thermal cap.

The retained whole-workload trace gives 15.185 → 15.064 s prefill wall.
Flash falls 3,318.3 → 3,138.1 ms (5.43%); 1,464 staging launches add
47.6 ms, leaving a net 132.6 ms attention-pipeline saving. Kernel interval
union falls 15.144 → 15.020 s; 481.9 ms of overlap is not extra wall time.
Decode wall is 1,058.9 ms for 16 tokens. These profiled wall times and the
unprofiled rates above are distinct measurements.

Measured `ds4-bench` SHA256:
`7d031010a67727184ce3ca939e6dc847f3eafa5349208e91c5a62312297386c3`;
`ds4-bench-perf`:
`68b5413e56116cde7bbb519be55d3cf907865cb4bfbdc505b2052aa43e3b0e56`.
Source is `7872f58f` plus this round's native diff. Binary/source hashes
were checked after all runs. Raw receipts are `scratch/axk2-perf/p3-*`.
Prefill now has three adopted rounds; decode rounds and the final
publication sweep remain.

## D1 — Q8 decode staging and input loads (adopted)

Retained P3's Q8 indexer projections cost 81.37 ms per 16 decoded tokens,
7.69% of the 1,058.1 ms decode wall. The 128×7,168 projection takes
15.41 µs and the 8,192×1,536 projection 67.96 µs per layer. At 235 GB/s,
their weight-only floors are 4.15 and 56.89 µs. The small projection has
only 16 CTAs; the large projection is nearer its bandwidth floor.

The new path stages weights with `cp.async.cg` and exposes 16 independent
input loads in the dot loop. The large projection uses one warp per CTA
instead of eight; an explicit 8,192×1,536 shape predicate limits that
mapping. Class 5.1 scheduling: sequential FMAs, reductions, rounding and
weight bytes remain unchanged. No additional global allocation is needed;
large-projection dynamic shared memory falls 13,056 → 1,632 bytes per CTA.
`DS4_AXK2_Q8_STREAM=0` restores the original path. Multirow MMQ is unchanged.

The isolated probe uses the artifact's exact Q8 payloads and rotates at
least 96 MiB of copies, with finite synthetic BF16 input. It retains the
model's shapes and compiled instructions, but not its actual input/cache
history. Median timings over three repeats are 16.800 → 11.680 µs small
and 70.944 → 63.776 µs large. One initial small baseline is 19.968 µs;
the remaining two are 16.768/16.800 µs. These raw samples remain recorded.
The large optimized path reaches about 89% of its weight-only ceiling.

Full NCU with native fast-math flags reports 34.336 → 17.312 µs small
(cold input, unlike steady-state timing) and 73.312 → 66.240 µs large.
The final probe and model both configure 64 KiB shared memory for the new
kernels; the original large kernel configures 100 KiB. An earlier probe
forced the new large kernel to 100 KiB; those receipts are retained but
are superseded by `q8-native-large-default`. Registers change 40 → 39/44,
with no spills. Executed warp instructions fall 358,400 → 320,896 small
and 6,012,928 → 5,275,648 large. The latter's L2 read-interface rate falls
because fewer bytes cross that interface; it is not a DRAM bandwidth loss.

`tests/test_axk2_q8` passes CPU per-lane FMA/XOR-reference and output-canary
checks for zero input, partial output groups, and both model shapes.
The original 280 SASS instructions and the new probe's 400 instructions
match their native counterparts after constant-bank relocation is masked.
Unchanged Q4 one-row staging and expert GU kernels also retain their SASS.

The first, async-only candidate was not adopted: six fresh 8K/512 runs
had decode off 15.37/15.39/15.38 and on 15.23/15.42/15.43 tok/s. Its mean
did not improve. The revised candidate's broad-affinity measurements were:

| Process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 546.71 | 15.08 |
| 2 | on | 547.24 | 15.17 |
| 3 | on | 544.93 | 15.51 |
| 4 | off | 546.35 | 15.37 |
| 5 | off | 546.47 | 14.99 |
| 6 | on | 542.36 | 15.40 |

All three pairs improve decode, but the 2.12% median gain is inflated by
unresolved absolute variability. Observed GPU clocks, temperature, PSI,
CPU frequency and faults do not explain the slower runs. We therefore
added a prespecified four-run old/new/new/old comparison pinned to CPU 9,
using the saved P3 binary and the new binary, both with the same P1–P3
paths. CPU pinning is a control, not a proven explanation of the earlier
variation. The unchanged 8K prefill follows a 2K warmup/reset, then 512
forced greedy tokens:

| Process | Binary | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | P3 | 545.80 | 15.37 |
| 2 | D1 | 548.25 | 15.46 |
| 3 | D1 | 546.07 | 15.53 |
| 4 | P3 | 542.78 | 15.36 |
| Median | P3 | 544.29 | 15.365 |
| Median | D1 | 547.16 | 15.495 |

Decode improves **0.85%**, with nonoverlapping repeat ranges and paired
gains of 0.59% and 1.11%. Prefill's unchanged path shows +0.53% variation;
this counts only as one decode round. All four processes exit zero with
identical finite 163,840 logits and all 512 tokens. The earlier two series
also pass full output comparison. Busy SM clocks stay 2,184–2,190 MHz,
full memory PSI stays zero, and minimum available memory is 15.842 GiB.
Raw CPPC counters are also recorded; they estimate delivered performance,
not calibrated physical clock frequency, using the
[Linux CPPC formula](https://kernel.org/doc/html/v6.6/admin-guide/acpi/cppc_sysfs.html).

The broad-affinity candidate trace reduces Q8 81.37 → 73.32 ms per 16
tokens: small 15.41 → 10.47 µs, large 67.96 → 64.65 µs. Total profiled
decode is 1,061.1 ms, slightly above 1,058.1 ms: the unchanged Q4 q_b
projection is slower in that trace. Thus attribution confirms the target
saving, while the unprofiled fixed-core repeats establish adoption. A new
whole-workload trace on the retained baseline starts the next round.

Measured source: `7fcce49d` plus this round's native diff. SHA256:

- `ds4-bench`: `0ba68b8e0144a29aefda94ea5ab0f5842fe6dcafc6407d6365faf52b9d2b6022`
- `ds4-bench-perf`: `dd3051f804b52cb5896e8785e7c8f01334281995d12c8a8052df7f73b731cffb`

Receipts: ignored `scratch/axk2-perf/d1-*`, `d1b-*`, `d1c-*`,
`q8-native-*`, `q8-default-*` and the exact payload SHA256s in
`q8-fixture/receipt.json`. Prefill has three adopted rounds and decode one;
final publication and two more decode adoptions remain.

## D2 — compact expert gate/up loop (adopted)

The fresh retained-D1 trace, now pinned to CPU 9, records 15,000.6 ms
prefill and 1,060.3 ms decode wall per 16 tokens. Expert gate/up consumes
195.36 ms (18.43% of decode wall). Eight selected experts stream
33.10/36.77 MB for the 9/10-bit gate/up pair, giving weight-only floors
of 140.8/156.4 µs at 235 GB/s, versus 203.5 µs average in the model.
The 32-warp kernel caps registers at 64; its fully expanded inner loop
spills live values to local memory. Previously tested smaller warp counts
did not provide a useful isolated gain.

The v2 compact schedule expands two codewords at a time. It retains the
same FMA sequence, reduction, gain placement, tail block and activation.
Class 5.1 scheduling. Only A.X-K2 selects it;
`DS4_AXK2_GU_COMPACT=0` restores the original schedule. The generic default
and all other families remain unchanged. No global allocation, shared
memory requirement, weight bytes or floating-point work is added.

The actual-payload probe replays the ascending expert selections captured
at the 8K frontier, with synthetic BF16 input and a 64 MiB cache flush.
Three baseline/new medians (µs) are:

| Bits | Original | Compact |
|---|---|---|
| 9 | 162.592 / 162.592 / 160.608 | 159.552 / 159.056 / 159.552 |
| 10 | 183.136 / 183.136 / 188.304 | 178.016 / 179.008 / 178.832 |

Median latency falls 1.87%/2.35%. A separate rotation through three whole
GGUF copies preserves the gain: 162.208 → 160.096 µs for 9 bits and
187.216 → 178.016 µs for 10 bits. Each part is 1,591,391,872 or
1,767,565,696 bytes; three copies hold more selected weights than L2.
These probes are faster than the full-model kernel, so their cache/input
history does not completely reproduce the model. They do not establish
whole-model gains on their own. Expansion by four was slower; expansion
by one was roughly even and was not selected.

The cost is additional loop instructions. Full NCU's cold-cache replay
records 39.96 → 43.15 million executed warp instructions (+8.0%), while
local-read sectors fall 1,196,032 → 409,600 and local-write sectors
989,184 → 399,360. Registers stay at 64 and the 9-bit stack falls
56 → 24 bytes. Configured shared memory stays 32 KiB. NCU replay latency
increases 172.800 → 181.312 µs, opposite the unprofiled probe result;
this diagnostic regime difference is retained explicitly. The complete
unprofiled workload below decides whether the tradeoff is useful.

`tests/test_axk2_decode` checks both bit widths and both gain modes,
independent sampled CPU FMA/XOR dots, complete gate/up/activation parity,
and allocation canaries. All pass. All **38 existing GU/down kernels**,
including the MiniMax kernels using the shared dot helper, retain their
SASS after relocation normalization. The new native 2,160-instruction
9-bit kernel matches the probe. The unchanged Q4 one-row staging kernel
also retains all 464 instructions.

Six fresh processes use CPU 9, 8K cold prefill after a 2K warmup/reset,
and 512 forced greedy tokens. Only `DS4_AXK2_GU_COMPACT` changes:

| Process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 546.19 | 15.49 |
| 2 | on | 546.01 | 15.64 |
| 3 | on | 545.49 | 15.70 |
| 4 | off | 547.45 | 15.48 |
| 5 | off | 542.38 | 15.56 |
| 6 | on | 545.77 | 15.66 |
| Median | off | 546.19 | 15.49 |
| Median | on | 545.77 | 15.66 |

Decode improves **1.10%**, with paired gains 0.97%, 1.42% and 0.64%.
The absolute ranges, 15.48–15.56 off and 15.64–15.70 on, do not overlap.
The extra integer instructions are worthwhile in this workload: total
latency falls, local traffic falls, and memory requirements do not grow.
Prefill's unchanged path differs by −0.08%, within its repeat range.

All six runs exit zero with identical finite 163,840 logits and all 512
tokens; the baseline also matches D1's retained outputs. Minimum available
memory is 15.881 GiB and full PSI stays zero. Busy SM clocks are
2,184–2,190 MHz. Decode temperatures span 52–59 °C with no observed
power/thermal cap. CPPC mean delivered-performance readings on CPU 9
span 151.797–151.813, against nominal 150, with no meaningful arm drift.
Source and executable hashes remain unchanged throughout the series.

Measured source: `bc230fd5` plus this round's native diff. SHA256:

- `ds4-bench`: `3970ef73dfbbb08a4e8a966ef5cd1ce38bed254a39fdaa0c52d51c7073579588`
- `ds4-bench-perf`: `afc9b121afe0c8f3bb3fcf813b634e50bcbf6f6e79363e415c9fd5d242f01039`

Receipts are in ignored `scratch/axk2-perf/d2-*`, `gu-unroll-*`,
`gu-rotate-*` and `gu9-unroll2.*`. The next retained-baseline trace will
also record the compact kernel's in-model contribution. Prefill has three
adoptions and decode two; one more decode adoption and publication remain.
