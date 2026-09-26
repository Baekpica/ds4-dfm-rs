# MiMo prefill optimization — 2026-09-26

GB10, CUDA 13.3, sm_121a; MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8.
Fixture: `speed-bench/promessi_sposi.txt`, 8192 prompt tokens, 128 greedy
output tokens, 4096-token prefill chunks, MTP/DFlash off. Each sample uses
its own process after a separate warmup process, sharing one VMM owner.
GPU clock range stays 300–2200 MHz; busy clocks are recorded per round.
[Workload and artifact hashes](sum-residual/workload.json).

Three rounds are accepted below. Three additional rounds were completed
and rejected; the retained inference code remains round 3. The next
campaign targets image input, followed by audio and video.

## Routed sum and residual fusion

**Accepted: round 1 of 3.** The final build improves prefill in all three
fresh pairs, with exact 152,576 logits and all 128 tokens in every run.

| Final build | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Original median | 1192.96 | 24.44 |
| Fused median | 1199.38 | 24.38 |
| Change | +0.54% | −0.25% |

Prefill pair gains: +0.49%, +0.74%, +0.51%. The first two final pairs
overlapped CPU SASS analysis; the last pair was uncontended and decode
improved 24.44→24.47. All samples remain recorded. Decode ranges overlap;
there is no claim of a decode speedup. Retain the consistent prefill gain
below 1% because resource costs decrease and no repeatable decode penalty
was established. [Final samples and proofs](sum-residual/final-ab.json),
[build receipt](sum-residual/receipt.json).

Fresh whole-workload nsys measured 6.90556 s prefill. Expert sum consumed
226.85 ms in 94 calls; each was followed by a separate residual add.
The earlier adjacent-call trace measured 300.51 ms combined (4.37% of
prefill). Full isolated NCU matched production geometry: 4096 rows,
4096 channels, eight route slots, grid 65536, block 256.

NCU showed 2.39 ms sum plus 0.777 ms add, no spills, 0.23% L2 hit rate,
and dominant long-scoreboard waits. Fusion removes a 64 MiB temporary
write and its 64 MiB reread per call. The dense-FFN scratch allocation
remains; no additional VRAM or arithmetic is introduced.

| Isolated path | Median ms | Change |
|---|---:|---:|
| Separate sum + in-place add | 3.175731 | — |
| Scalar fusion | 2.694980 | −15.14% |
| float4 fusion | 2.924834 | −7.90% |

Three fresh processes per path, rotated order, 16 warmups and 100 measured
iterations. All outputs after 116 accumulated passes were byte-exact.
Scalar fusion was selected. It uses 28 registers and no spills; full NCU
executed instructions fell from 60,817,408 across two kernels to 30,932,992.
See [isolated samples](sum-residual/isolated.json),
[baseline NCU](sum-residual/ncu-baseline.txt), and
[fused NCU](sum-residual/ncu-fused.txt).

### Numerical and dispatch contract

Sum slots 0…7 from zero with separate FP32 multiply/add rounding, then add
the residual. Do not seed the route accumulation with the residual or
contract the operations into FMA. The CUDA path covers 32–8192 rows;
narrow decode retains the existing sum/add path. `DS4_MIMO2_SUM_RESIDUAL=0`
restores the separate path. Unsupported dispatch returns before GPU work;
execution errors abort instead of replaying the fallback.

[Kernel tests](../../../tests/mimo2_sum_residual.cu) compare byte parity
at 1, 31, 32, 33 and 4096 rows. [Wrapper tests](../../../tests/mimo2_sum_dispatch.cu)
cover default/diagnostic dispatch, invalid buffers, alignment, refusal
without mutation, and 32/33-row parity. Build either with:

```sh
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_121a \
  tests/mimo2_sum_residual.cu -o /tmp/mimo2-sum-test
/tmp/mimo2-sum-test
```

### Build control

The initial candidate used `--split-compile 8`. SASS comparison found seven
changed existing attention/router kernels, including new local loads/stores
in HMMA prefill. This candidate's OFF control was slower than the original.
Its three-arm A/B remains [recorded](sum-residual/candidate-ab.json), but is
not the final build qualification.

The final build uses the original compiler flags. All 36 existing project
kernels executed by the fixture have byte-identical SASS to the original
binary. The three external cuBLAS kernels were outside this comparison.
See the [per-kernel comparison](sum-residual/codegen.json).

Raw reports, binary snapshots, full logits, guards, scripts and rejected
patches remain under `scratch/mimo-prefill-20260926/`; this directory keeps
compact measurements and proof receipts. Dated evidence qualifies this
artifact/workload, not all serving modes.

## Attention scale and residual fusion

**Accepted: round 2 of 3.** The retained round-1 binary is the original
control. Three fresh samples per arm use rotated order and a separate
warmup process before each sample, with the same resident weight owner.

| Build / path | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Round-1 original median | 1199.67 | 24.46 |
| New build, fusion OFF median | 1199.12 | 24.47 |
| New build, default ON median | 1206.87 | 24.48 |
| ON vs original | +0.60% | +0.08% |

Prefill gains against the original are +0.50%, +0.61%, +0.71%; against
the same-build OFF control they are +0.71%, +0.65%, +0.64%. All nine
samples have identical full 152,576 logits and 128 tokens, and clean
guard exits. Decode ranges overlap with no measured regression. Retain
the consistent gain below 1% because memory traffic decreases without
additional VRAM or arithmetic. See [samples](attn-residual/ab.json),
[proof hashes and exits](attn-residual/proofs.json),
[build receipt](attn-residual/receipt.json), and
[workload hashes](attn-residual/workload.json).

The fresh [whole-workload trace](attn-residual/nsys-before.json) on round 1
measured 6.82443 s prefill. The 96 adjacent scale/add pairs consumed
125.386 ms (1.837%). Full isolated NCU preserved their production geometry:
4096 rows × 4096 channels, grid 65536, block 256. The separate kernels took
500.29 µs and 778.02 µs, with long-scoreboard waits of 205.2 and 312.2 cycles.

| Isolated path | Median ms | Change |
|---|---:|---:|
| Separate scale + in-place add | 1.302458 | — |
| Scalar fusion | 0.763819 | −41.36% |

These are unprofiled timings: three fresh processes per arm in alternating
order, 32 warmups and 100 measured iterations. Both inputs are restored
before each iteration outside CUDA-event timing. Synthetic values and
restore-driven cache state differ from the full graph; end-to-end A/B
decides adoption. [Isolated results](attn-residual/isolated.json),
[baseline source receipt](attn-residual/source-receipt.json).

All three kernels use 16 registers and have no spills. NCU executed
instructions decrease from 19,922,944 across the two original kernels to
11,010,048. Fusion removes the projection's 64 MiB scaled write and 64 MiB
reread per call; allocations and arithmetic are unchanged.
[Baseline NCU](attn-residual/ncu-baseline.txt),
[fused NCU](attn-residual/ncu-fused.txt).

The numerical contract is a separately rounded FP32 multiplication by
`0.707f`, followed by a separately rounded residual addition; no FMA.
The projection has no later consumer before overwrite. CUDA rows 32–8192
use the fused path by default; `DS4_MIMO2_ATTN_RESIDUAL=0` restores scale
then add. Narrow decode keeps its old calls. The
[focused test](../../../tests/mimo2_attn_residual.cu) covers 1, 31, 32, 33
and 4096 rows, signed zero/subnormal inputs, default/explicit/fallback
dispatch, invalid buffers and alignment, and refusal without mutation.
[Test output](attn-residual/parity.txt).

Busy model-run clocks were 2184–2190 MHz (median 2190), preserving the
user's configured range. [Clock summary](attn-residual/clocks.json) includes
the sampling limits of the shorter isolated runs. Raw reports and all
warmup/sample proofs remain in `scratch/mimo-prefill-20260926/`.

## Gate/Up bounded scheduling

**Accepted: round 3 of the initial 3.** The retained round-2 binary is the
original control. The new binary uses the original compiler flags and a
full MMQ rebuild. Three fresh samples per arm use rotated order, each
after a separate warmup process with the same weight owner.

| Build / path | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Round-2 original median | 1206.39 | 24.43 |
| New build, scheduling OFF median | 1206.57 | 24.44 |
| New build, default ON median | 1217.76 | 24.44 |
| ON vs original | +0.94% | +0.04% |

Prefill pair gains are +0.65%, +0.86%, +0.99% against the original and
+0.85%, +0.85%, +0.93% against same-build OFF. All nine samples have
identical full 152,576 logits and 128 tokens, with clean guard exits.
Decode ranges overlap; no decode speedup is claimed. See
[samples](gateup-schedule/ab.json), [proofs](gateup-schedule/proofs.json),
[build receipt](gateup-schedule/receipt.json), and
[workload](gateup-schedule/workload.json).

The fresh [round-2 whole trace](gateup-schedule/nsys-before.json) measured
6.79655 s prefill. IQ2 Gate/Up consumed 2.21873 s in 94 calls (32.64%).
Full isolated NCU preserves M=2048, K=4096, 256 experts, top-k 8 and
4096 input tokens: grid (16, 768, 2), block (32, 8, 1).

A full-warp boundary after each adjacent-k32-pair N fragment limits
compiler scheduling and temporary lifetimes. Each accumulator retains its
original multiplication and accumulation order. The measured tradeoff is:

| Full NCU counter | Change |
|---|---:|
| Executed instructions | +2.29% |
| Shared-load instructions | +19.70% |
| Shared-load wavefronts | +0.639% |
| Register-spill instructions; local load/store sectors | −52.94% |
| Total L2 sectors | −18.767% |
| Tensor INT8 operations; global load/store sectors | Unchanged |

Both paths allocate 128 registers per thread and 37,744 bytes static
shared memory per block, with no dynamic shared memory or additional
tensor/workspace buffers. Shared-load bank conflicts change +0.036%.
These counters do not establish unchanged scalar arithmetic or a measured
DRAM-byte total. [Baseline NCU](gateup-schedule/ncu-baseline.txt),
[bounded NCU](gateup-schedule/ncu-bounded.txt).

The earlier rejection over-weighted the instruction increases. The
[historical reassessment](gateup-schedule/historical-reassessment.json)
accounts for the lower spill and total L2 traffic. That reassessment only
justified retrying; the fresh profiling, exact proofs and three-arm A/B
above establish adoption on round 2. The extra instructions remain a
reported cost, outweighed here by consistent gains and lower traffic.

| Unprofiled isolated entry | Median ms | Change vs original |
|---|---:|---:|
| Original | 24.335621 | — |
| New build, OFF | 24.337925 | +0.009% |
| New build, default ON | 23.887592 | −1.84% |

Each arm uses three fresh processes, 16 warmups and 100 timed iterations.
Full gate and up outputs (67,108,864 FP32 values each) are byte-exact
across sample-0 dumps; later timing repeats do not dump outputs. Synthetic
weights/routing preserve geometry, not model values or full-graph cache
state. NCU uses kernel replay with cache flushing; the older baseline
fixture has one warmup and the candidate has 16. Its 23.192→22.715 ms
profiled time is diagnostic; adoption uses unprofiled A/B.
[Isolated results and counter units](gateup-schedule/isolated-result.json).

Default dispatch requires DGX Spark, an IQ2_XXS pair with M=2048, K=4096,
256 experts, top-k 8, and 2048–65536 routed assignments (256–8192 tokens).
`DS4_MIMO2_GATEUP_BOUNDED=0` restores the original schedule. Other shapes,
architectures and narrow decode retain that schedule. The switch is cached
on first use; change it between processes. The
[production-entry harness](../../../tests/mimo2_gateup_schedule.cu) checks
the 255/256-token boundary and same-shape top-k 6 fallback.
[Boundary parity](gateup-schedule/boundaries.txt) and a separate
[kernel-dispatch trace](gateup-schedule/dispatch.json) confirm top-k 6
executes the original specialization with the switch unset.

Busy model clocks were 2184–2190 MHz (median 2190); all 24 busy isolated
clock samples were 2190 MHz. The user's 300–2200 MHz range is preserved.
[Clock summary](gateup-schedule/clocks.json). Raw traces, dumps, guards and
scripts remain under `scratch/mimo-prefill-20260926/`.

## Additional round 4: Down activation async copy

**Rejected.** A fresh whole trace attributed 21.45% of prefill to IQ2_XS
Down. Full isolated NCU motivated activation-only async copies while
preserving the weight loader, MMA order and allocation size. Three rotated
original/OFF/ON pairs were byte-exact, but ON lost every pair: median
latency increased 0.69% against original and 1.00% against same-build OFF.
Spills and L2 traffic increased. The candidate was reverted before model
qualification; no end-to-end gain is claimed.
[Decision, measurements and archived patch](down-async/README.md).

## Additional round 5: RoPE row mapping

**Rejected.** After a fresh whole trace, full NCU of both KV-head shapes
showed memory-dependency stalls. A 2D row grid removed about 44% of executed
instructions while preserving arithmetic and memory requests. Full outputs
remained exact, but all six isolated pairs regressed: median latency rose
1.63% and 2.24% for KV heads 4 and 8. Restore the original mapping; fewer
instructions alone did not improve this workload.
[Decision, measurements and archived patch](rope-rows/README.md).

## Additional round 6: SwiGLU/Q8 wider blocks

**Rejected after model A/B.** Fresh nsys attributed 3.792% of prefill to
SwiGLU/Q8. Full NCU and a bounded block-size comparison selected 512 threads
for model testing: isolated latency improved 0.70–1.53% without additional
VRAM or arithmetic. Actual prefill medians were 1219.21 / 1218.30 / 1219.23
tok/s for original / OFF / ON. ON lost two of three pairs against OFF; its
+0.00164% median against original does not establish a consistent gain.

All nine measured and nine warmup processes produced identical complete
logits and tokens with clean guards. Decode medians were 24.45 / 24.42 /
24.40 tok/s; ranges overlap and no causal decode regression is claimed.
The candidate was reverted.
[Full decision and evidence](swiglu-wide/README.md).
