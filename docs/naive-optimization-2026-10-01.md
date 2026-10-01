# Naive additional optimization rounds, 2026-10-01

Base: `2aedeb43` (main after PR70). Earlier P4/D3 are excluded from these
additional rounds. Retained so far: **prefill 0, decode 1**.

## Protocol

GB10, driver 615.71.09, CUDA 13.3, `sm_121a`, clock range 300–2200 MHz;
observed clocks 2190–2197 MHz. MQ87 source/artifact pins are unchanged from
[the family contract](naive-n05-flash.md). One resident VMM owner shares the
main weights; no serving worker or competing GPU workload runs during samples.

Primary workload: `promessi_sposi.txt`, 8192 input tokens, prefill chunk 2048,
32 greedy generated tokens. Each side has three fresh unprofiled processes,
each preceded by a separate warmup; Nsight uses another warmed fresh process.
`ds4-perf scout --proof --workload` verifies all four model shards before and
after each side. Both scouts are complete with no warnings. Other retained
controls are fixed to one. These results qualify this workload, not 1M or
DSpark acceleration.

## D4: SWA decode unit exponent

Whole baseline: SWA attention takes 260.393 ms, 15.04% of decode kernel time;
decode wall time is 1.845375 s. The serial online softmax evaluates two
exponents per key, although one is exactly one for finite operands.

Full-counter isolated profiling preserves the 64/8-head geometry, 128-key
window and capacity 2175 with synthetic BF16 operands. Removing the unit
exponent shortens the recurrence dependency. QK, FMA/add order, BF16 boundaries,
sink and ascending V accumulation remain byte-exact; exceptional values use
the original equation. Dispatch applies only to cached SWA at width one.
`DS4_NAIVE_SWA_DECODE_UNIT=0` restores the previous path.

| Measurement | Off | On |
| --- | ---: | ---: |
| Prefill samples, tok/s | 506.14 / 506.04 / 505.72 | 505.78 / 506.17 / 506.18 |
| Decode samples, tok/s | 17.42 / 17.46 / 17.39 | 18.02 / 17.96 / 18.00 |
| Mean prefill, tok/s | 505.9667 | 506.0433 |
| Mean decode, tok/s | 17.4233 | 17.9933 (+3.27%) |
| Nsight SWA decode total, ms | 260.393 | 204.138 |
| Nsight decode wall, ms | 1845.375 | 1794.225 |
| Warm isolated kernel, us | 105.386 | 92.941 |
| NCU cold-cache kernel, us | 263.392 | 223.104 |
| Dynamic warp instructions | 1,338,240 | 1,431,808 |

Registers stay 40/thread and allocated shared memory stays 2 KiB/CTA.
Instruction count increases about 7% through control flow; exponent work
decreases. No allocation or intermediate tensor is added. Wide SWA regressed
about 6.2% in isolated warm runs, so prefill and wider verification keep the
previous recurrence. DSA remains unchanged.

Correctness: 131,453 recurrence cases / 1,074,253 intermediate F32 states
match an independently compiled source equation, including every finite BF16
code and Inf/NaN payloads. Wrapped/varied attention fixtures and capacity-2175
memcheck pass. Actual-weight 43- and 2053-token tests compare all valid prefix
and same-width committed target/index/draft KV, checkpoint tokens, 48 hidden
rows and full logits. Width-seven/rejected-value checks compare the accepted
frontier; retired/future draft slots are excluded. Captured decode is not
tested by this eager-family proof.

`ds4-perf compare` reports Improved: zero token/argmax mismatches and zero
logit error across 915,456 checked values. Its decode time envelope is
−3.50% to −2.78%; prefill is −0.091% to +0.071%. `cargo test -p ds4-perf`
passes. Guarded build/model runs exit zero; minimum host availability is
25.56 GiB and campaign peak memory PSI full avg10 is 0.76, including hashing.

Local evidence: `scratch/naive/perf-2026-10-01/unit-{off,on}/scout.json`,
`unit-compare/compare.json`, `unit-state-{43,2053}.log`, `unit-softmax.log`,
`unit-cap2175-{warm,memcheck}.log`, `swa-dec-2175-{off,on}.ncu-rep`,
`unit-source.{json,patch}`, and `unit-model.guard.jsonl`.
