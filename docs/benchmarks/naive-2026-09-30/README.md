# Naive GB10 optimization rounds

Retained rounds from the September 30 request: prefill **2/3**, decode **3/3**.
Rejected probes do not count. Integration and long-context serving remain
separate gates; see [the family contract](../../naive-n05-flash.md).

## Protocol

GB10, driver 615.71.09, CUDA 13.3, clock lock 300–2200 MHz; observed
2190–2197 MHz, active samples 2190 MHz. Same resident VMM owner for all runs:
MQ87 main plus DSpark weights, no additive Q8 repack. Workers load only the
main model for these ordinary-decode measurements.

The pinned four-shard MQ87 artifact uses revision
`65b235a48e87acaf7b806b96870378c9c68a654c`. Workload: 8192 tokens from
`speed-bench/promessi_sposi.txt`, 32 greedy decode tokens, chunk 2048.
Each side has three fresh processes, each preceded by a separate warmup.
`ds4-perf scout` hashes all shards, prompt and consumed owner manifest
before/after each side; both artifacts are complete with no warnings.
Nsight throughput is excluded from the unprofiled comparison.

## D1: reuse one-row attention scores

The initial whole workload spends 51.21% of prefill GPU time and 53.75%
of decode GPU time in attention. Isolated full-counter NCU identifies
serial global-load dependencies and a decode grid of only 16 CTAs on 48 SMs.
The reproducer preserves DSA layout, 64/4 GQA, key/value widths 192/128,
8192 history and ascending scattered top-2048 IDs. Its BF16 operands and
selected IDs are synthetic; routing and cache warmth differ from the model.

One-row attention stores its rounded BF16 scores on chip, avoiding the
second QK pass. Dot, online softmax and ascending V sum keep their original
arithmetic order. DSA uses 16 KiB shared memory per CTA; SWA uses 1 KiB.
There is no additional persistent allocation. Wide prefill stays unchanged:
the DSA tile reduces occupancy and failed the isolated speed check.
`DS4_NAIVE_DECODE_SCORES=0` restores the original path.

| Unprofiled tok/s | Baseline samples | Candidate samples | Mean change |
| --- | --- | --- | ---: |
| Prefill | 454.86, 455.84, 455.77 | 455.21, 455.53, 455.35 | −0.028% |
| Decode | 11.29, 11.29, 11.28 | 13.40, 13.40, 13.40 | +18.724% |

`compare --regression` returns `Improved`: 915456 checked logit values,
zero absolute/relative difference, token mismatch and argmax mismatch.
All three full-vocabulary frontier and 32-token proof files have identical
baseline/candidate SHA-256. The actual-weight 43-token regression also
passes: width 1/7 hidden states and logits, committed target/indexer/draft
rows, and rejected-draft independence are byte-exact. Independent SWA/DSA
equations and device-backtrace CUDA memcheck pass with zero errors.

NCU cold-cache DSA time falls 4.123264→2.723776 ms. Global L1 read sectors
fall 4458048→2885184, and long-scoreboard stalls fall 16.726→12.230 cycles
per issue. Warm isolated DSA time falls about 2.437→1.529 ms. NCU uses
`--clock-control none --cache-control all`; replay overhead is excluded.
The whole-workload Nsys attention sum falls 1.470203→1.024660 s in decode;
decode wall time falls 2.848922→2.402018 s. Prefill wall time remains
17.976745→17.979104 s, consistent with the unprofiled noise envelope.

Raw evidence: ignored `scratch/naive/round-d1-{base,candidate,compare}`,
`round-d1-state.log`, `attention-decode-{base,cache}.ncu-rep`, and
`attention-cache-{parity,memcheck}.log`. Feature baseline: `a3ab40ad`.

## Uncounted probes

Wide DSA score caching: three warm original samples 3.677/3.677/3.666 ms
versus 4.864/4.896/4.877 ms. Reject despite exact outputs.
One-warp CTAs: one initial 2.438→2.419 ms probe. Insufficient evidence;
the production path retains four warps per CTA.

## P1: reuse wide SWA scores

After D1, a fresh whole-workload capture still spends 51.207% of prefill
GPU time in attention. SWA takes 1.727902 s, 9.628% of that phase's GPU
kernel sum. Its faithful isolated shape is 2048 queries, 64/8 GQA,
192/128 key/value widths, window 128 and ring capacity 2175. Synthetic
operands retain the geometry and layout, rather than actual routed values.
Full-counter NCU shows 75.44% LSU utilization and 96.21% occupancy.

The 1-KiB score tile retains theoretical occupancy, unlike wide DSA.
Enable it above seven rows; keep bounded verification and DSA unchanged.
`DS4_NAIVE_SWA_PREFILL_SCORES=0` restores the prior path. No allocation
is added, and the second QK MACs and loads are removed.

| Unprofiled tok/s | Baseline samples | Candidate samples | Mean change |
| --- | --- | --- | ---: |
| Prefill | 455.29, 455.53, 455.92 | 466.43, 466.53, 466.59 | +2.401% |
| Decode | 13.37, 13.40, 13.39 | 13.38, 13.40, 13.40 | +0.050% |

Both scouts are complete with no warnings; `compare --regression` returns
`Improved`. All full-logit and 32-token hashes match the baseline. The
actual-weight width/cache regression and independent SWA/DSA fixtures
pass, including a varied-exponent BF16 fixture. These are exact comparisons
to the retained implementation, not proof of full-model reference parity.
Cold NCU falls 11.03→8.19 ms with 40 registers/thread and 100% theoretical
occupancy; achieved occupancy is 96.21→98.54%. Three warm isolated
samples fall 11.163/11.140/11.153→8.261/8.283/8.295 ms.
Whole Nsys prefill wall time is 17.558307 s after the change; decode stays
2.401494 s. Raw evidence: `scratch/naive/round-p1-{current,base,candidate,compare}`,
`round-p1-state.log`, `attention-varied-check.log`, and
`attention-swa-prefill-{base,cache}.ncu-rep`. Retained baseline: `a12914b8`.

## D2: fill idle SMs during DSA decode

After P1, a fresh whole capture measures decode attention at 1.023999 s,
44.735% of GPU kernel time. The one-row DSA grid still has only 16 CTAs
on 48 SMs. Full-counter NCU measures 2.73 ms, 8.31% achieved occupancy,
40 registers and 16.38 KiB shared memory per CTA.

Use one CTA per head with four parallel key walks. Keep the complete XOR
dot tree, BF16 boundaries, serial online denominator and ascending V FMA.
The exact `exp(0)` term is elided. The shared score/probability tile is
4.10 KiB; no persistent allocation changes. Wider DSA is excluded: its
isolated probe is slower. `DS4_NAIVE_DSA_DECODE_TILE=0` restores D1;
disabling `DS4_NAIVE_DECODE_SCORES` still restores the original path.

| Unprofiled tok/s | Baseline samples | Candidate samples | Mean change |
| --- | --- | --- | ---: |
| Prefill | 466.73, 466.35, 466.94 | 466.22, 466.62, 466.50 | −0.049% |
| Decode | 13.36, 13.40, 13.39 | 15.54, 15.51, 15.54 | +16.040% |

Both scouts are complete without warnings; `compare --regression` returns
`Improved`. All 915456 logit values, argmaxes, tokens and proof hashes match.
Actual-weight width/target/indexer/draft state regression, independent
equations, early/future/padded IDs and varied BF16 fixtures pass. CUDA
memcheck reports zero errors. Observed clocks are 2184–2197 MHz; samples
above 40 W record 2184 or 2190 MHz within the user clock range.

Cold NCU falls 2.73→1.61 ms and warm isolated samples fall
1.529350/1.530883/1.529587→0.906118/0.906963/0.906915 ms. Theoretical
occupancy rises 41.67→100%; achieved occupancy rises 8.31→11.05%.
More CTAs and Q/ID requests increase global L1 read sectors
2885184→3430656 and executed warp instructions 22554624→27543424.
The latency gain pays this work cost; allocation equality does not imply
equal memory traffic or instructions.

Whole Nsys attention falls 1.023830→0.695414 s and decode wall time falls
2.401404→2.072915 s; prefill stays 17.554341→17.545052 s. Raw evidence:
`scratch/naive/round-d2-{current,base,candidate,compare}`,
`round-d2-state.log`, `round-d2-{primitive-parity,memcheck}.log`,
`round-d2-dsa-{base,tile}.ncu-rep` and `round-d2-telemetry.jsonl`.
Retained baseline: `782aa821`.

## P2 / D3: index the full DSA history directly

After D2, a fresh whole capture measures prefill attention at 8.743242 s,
49.955% of GPU time, and decode DSA at 0.434528 s, 22.203%. Full-counter
NCU source correlation finds 8913152 warp instructions at V address
calculation and 2621696 at QK address calculation, out of 27543424.
Both repeatedly compute `key % capacity` for a history that never wraps.

The forward guard enforces `pos + n <= context`; causal IDs satisfy
`key <= pos < capacity`. Direct indexing is therefore the same address.
Use the explicit full-history policy for DSA, retaining modulo for SWA.
Arithmetic, buffers, KV layout and allocations stay unchanged.
`DS4_NAIVE_DSA_DIRECT=0` restores the previous DSA address calculation.
This one change improves both phases, counted as P2 and D3 after separate
target profiles and a matched whole-workload proof.

| Unprofiled tok/s | Baseline samples | Candidate samples | Mean change |
| --- | --- | --- | ---: |
| Prefill | 466.73, 466.41, 467.04 | 491.26, 491.50, 491.35 | +5.280% |
| Decode | 15.49, 15.53, 15.54 | 17.44, 17.45, 17.39 | +12.285% |

Both scouts are complete without warnings; `compare --regression` returns
`Improved`. All 915456 logit values, argmaxes, tokens and proof hashes match.
Actual-weight 48-layer/target/indexer/draft state regression, independent
equations, early/future/padded IDs and device memcheck pass. Observed clocks
are 2190–2197 MHz; samples above 40 W are 2190 MHz.

At isolated capacity 8192, cold prefill falls 4.24→3.21 ms with unchanged
40 registers/shared memory; decode falls 1.62→0.82960 ms with unchanged
38 registers/4.10 KiB shared memory. Warm prefill samples fall
3.667739/3.678352/3.687741→3.225776/3.211278/3.212208 ms; decode falls
0.906320/0.908197/0.906827→0.496331/0.495195/0.494669 ms. The benchmark
reserves capacity 8225 for 8192 input plus generation; all selected IDs
are inside both capacities. Follow-up bounded 8225 captures preserve that
actual capacity as well as the existing synthetic-value/routing limits.
At capacity 8225, cold prefill is 4.34→3.21 ms and decode is
1.61→0.82390 ms. Global L1 read sectors stay identical; executed warp
instructions fall 824535040→698650624 in prefill and
27543424→15877504 in decode. Registers and shared memory stay unchanged.

Whole Nsys prefill attention falls 8.737404→7.852506 s and wall time falls
17.538436→16.650192 s. Decode DSA falls 0.435217→0.207441 s and wall time
falls 2.070530→1.845149 s; SWA stays 0.259411→0.259325 s. Raw evidence:
`scratch/naive/round-p2d3-{base,candidate,compare}`, `round-d3-current`,
`round-p2d3-{state,memcheck}.log`, `round-{p2,d3}-direct-*.ncu-rep`,
`round-{p2,d3}-cap8225-*.ncu-rep` and `round-p2d3-telemetry.jsonl`.
Retained baseline: `dde65cdf`.
