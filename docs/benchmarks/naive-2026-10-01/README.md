# Naive GB10 additional optimization rounds, 2026-10-01

Base: `2aedeb43` (main after PR70). Earlier P4/D3 are excluded from these
additional rounds. Final retained: **prefill 4, decode 3**.

Latest matched 8K arm: `index2-on`, **518.85 prefill / 18.85 ordinary decode
tok/s**. Retained native source: `ee28ed26`; benchmark SHA-256:
`db5cbe0433acd561f6fbaa818d1312e0fcc69058e49464fcec6021db9e47a8b2`.
Documentation layout follows main `4ea49fbb` (PR71).

## Protocol

GB10, driver 615.71.09, CUDA 13.3, `sm_121a`, clock range 300–2200 MHz;
observed clocks 2184–2197 MHz. MQ87 source/artifact pins are unchanged from
[the family contract](../../naive-n05-flash.md). One resident VMM owner shares the
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

## P5/D5: consecutive SWA ring addresses

Fresh retained-D4 measurement: prefill/decode wall 16.186014 / 1.794196 s;
cached SWA totals 1.275177 / 0.203818 s (7.88% / 11.36% of wall).
Each consecutive key repeated unsigned remainder for both K/V passes.
Compute the initial slot once, increment/wrap, and restart before the V pass.
Addresses, causal order, arithmetic and BF16 boundaries remain unchanged.
Dispatch covers cached SWA width one or greater than seven; other paths keep
their previous behavior. `DS4_NAIVE_SWA_RING_WALK=0` restores remainder.

| Measurement | Off | On |
| --- | ---: | ---: |
| Prefill samples, tok/s | 506.19 / 505.42 / 505.87 | 512.51 / 512.64 / 512.92 |
| Decode samples, tok/s | 17.96 / 17.93 / 17.96 | 18.15 / 18.17 / 18.10 |
| Mean prefill, tok/s | 505.8267 | 512.6900 (+1.36%) |
| Mean decode, tok/s | 17.9500 | 18.1400 (+1.06%) |
| Nsight SWA prefill total, ms | 1276.848 | 1067.507 |
| Nsight SWA decode total, ms | 203.672 | 189.025 |
| Nsight prefill/decode wall, ms | 16191.658 / 1791.186 | 15983.106 / 1784.918 |
| Warm isolated prefill, ms | 8.294753 | 6.893195 |
| Warm isolated Unit decode, us | 92.959 | 80.323 |
| NCU cold prefill/decode, us | 8191.136 / 229.280 | 6820.032 / 219.104 |

Isolation uses synthetic resident BF16 operands, capacity 2175 and three
alternating fresh process pairs; NCU captures one cold-cache launch per path.
Dynamic warp instructions fall 16.4% / 13.0%. Registers stay 40/thread,
allocated shared memory 2 KiB/CTA; memory requests/sectors and spill metrics
match. No allocation, tensor or arithmetic work is added.

Forty edge cases match output bytes, including early windows and wrap;
tiny-capacity and maximum-position fixtures are address stress only.
Early/wrapped memcheck, primitives and Rust perf tests pass. Actual-weight
43/2053/2181-token proofs pass; 2181 crosses the physical 2175-row ring.
State-proof scope is the same as D4. Both full scouts are complete without
warnings; exact comparison checks 915,456 logits with zero error and zero
token/argmax mismatches. Time envelopes: prefill −1.46% to −1.23%, decode
−1.32% to −0.77%. Observed active clocks are 2184–2197 MHz; mean sampled
active clocks differ by 1.02 MHz. Guard minimum availability is 25.33 GiB;
peak memory PSI full avg10 is 0.83, including hashing.

Evidence: `ring-{off,on}/scout.json`, `ring-compare/compare.json`,
`ring-state-{43,2053,2181}.log`, `ring-source.{json,patch}`,
`ring-prototype/{warm,edges,memcheck-early,memcheck-wrap}.log`, its four
NCU reports, and `ring-model.guard.jsonl`. The preceding retained diagnosis
uses official normalization with prior full-hash witnesses and current
metadata checks; adoption scouts freshly hash all shards.

## P6/D6: stable warp router

Fresh retained-Ring measurement: prefill/decode wall 15.984742 / 1.782391 s;
router totals 47.884 / 74.477 ms (0.300% / 4.178% of wall). A 256-thread
CTA computes sigmoid probabilities, then one thread scans 256 scores eight
times. NCU records only 1.74 active lanes per issued warp instruction and
3,056 / 6,259,101 shared-load wavefronts at widths one / 2048.

One warp now computes eight experts per lane and selects stable maxima with
shuffles. The original sigmoid and bias expressions, lower-ID ties, serial
selection-rank probability sum, normalization and numeric-ID sort remain
exact. Nonfinite operands use the original serial semantics, including its
duplicate selections. Dispatch covers width one or above seven;
`DS4_NAIVE_ROUTER_WARP=0` restores the untouched original kernel.

| Measurement | Off | On |
| --- | ---: | ---: |
| Prefill samples, tok/s | 512.58 / 512.40 / 512.64 | 514.31 / 513.88 / 514.52 |
| Decode samples, tok/s | 18.20 / 18.11 / 18.04 | 18.77 / 18.84 / 18.94 |
| Mean prefill, tok/s | 512.5400 | 514.2367 (+0.33%) |
| Mean decode, tok/s | 18.1167 | 18.8500 (+4.05%) |
| Nsight router prefill/decode total, ms | 47.863 / 74.545 | 3.309 / 6.556 |
| Nsight prefill/decode wall, ms | 15970.108 / 1776.722 | 15929.722 / 1708.669 |
| Warm isolated width 1 / 2048, us | 49.4944 / 225.9205 | 4.1797 / 16.5915 |
| NCU cold width 1 / 2048, us | 51.776 / 257.568 | 6.656 / 20.480 |

Three alternating fresh isolated pairs use resident synthetic F32 logits and
bias; no captured model operands are claimed. Block reduction measured
4.9813 / 52.0213 us and was slower than warp. Warp instructions fall
9,996→1,450 / 20,526,816→3,024,608; active lanes rise to 28.56 / 28.06 and
finite-row shared loads vanish. Registers/thread rise 30→40, allocated shared
memory remains 3 KiB and spills remain zero. Global load requests increase
one per CTA, 56→57 / 131,535→133,583. Finite checks and shuffles add work;
sigmoid count and normalization arithmetic stay unchanged. No inference
workspace or launch is added.

Prototype: 120 parity runs and four memcheck/synccheck gates pass.
Production: 40 old/new raw ID/weight-byte cases, finite operand-byte checks,
memcheck and primitives pass; cases cover biased ties, signed zero, tiny
probability sums and Inf/NaN. Actual-weight 43/2053/2181 proofs and Rust perf
tests pass. Both full scouts complete without warnings. Exact comparison
checks 915,456 logits with zero error and zero token/argmax mismatches.
Time envelopes: prefill −0.412% to −0.241%, decode −4.752% to −3.037%.
Active clocks are 2190–2197 MHz with equal sampled means; minimum available
memory is 25.40 GiB and peak PSI full avg10 is 0.80, including hashing.

Evidence: `router-{off,on}/scout.json`, `router-compare/compare.json`,
`router-state-{43,2053,2181}.log`, `router-source.{json,patch}`,
`router-production-parity.log`, `router-production-memcheck.log`,
`router-prototype/{resource-compact.json,parity.log}`, six NCU reports,
warm-pair logs and `router-model.guard.jsonl`. Eager/8K qualification limits
remain those above. Benchmark SHA-256:
`0025ef24e0fc7216b8b26ca15827f3efca0a0f82edcfc7110ad75ba2059a2fc1`.

## P7: ordered MoE sum and residual

Adopted for prefill; counts through P7: **prefill 3, decode 3**.
This adds no decode round.

Fresh retained-router diagnosis: 188 MoE sum/residual pairs take 302.373248 ms,
1.8968% of prefill wall. The pair writes an F32 intermediate for the residual update. Fusion removes
that write/read and one launch, retaining
ascending expert order and every BF16 down/product/partial-sum/residual
boundary. Dense layer zero and widths one through seven keep the old pair.
`DS4_NAIVE_SUM_ADD=0` restores it. F32 extents must be aligned and disjoint.

| Measurement | Off | On |
| --- | ---: | ---: |
| Prefill samples, tok/s | 514.46 / 514.77 / 514.28 | 515.99 / 516.28 / 516.05 |
| Decode samples, tok/s | 18.92 / 18.87 / 18.89 | 18.92 / 18.92 / 18.91 |
| Mean prefill, tok/s | 514.5033 | 516.1067 (+0.31163%) |
| Mean decode, tok/s | 18.8933 | 18.9167 |
| Nsight MoE sum/residual total, ms | 302.082784 | 257.159808 |
| Nsight prefill/decode wall, ms | 15930.590 / 1710.227 | 15898.263 / 1712.579 |
| Warm isolated width-2048 pair, ms | 1.616283 | 1.383939 |
| Matched NCU width-2048 pair, ms | 1.610336 | 1.378720 |

Three alternating fresh isolated pairs use synthetic resident operands;
full-counter NCU uses cache-control none. Actual routed values and preceding
down-kernel cache state are not captured. Combined warp instructions fall
33,816,576→29,097,984; global load requests/sectors fall 5.56%/8.33%, stores
fall 50%. Fused registers/thread are 27, versus sum 26 and residual 16;
static shared memory is zero, allocated shared memory stays 1 KiB/CTA and
spills stay zero. Source arithmetic is unchanged. One 32 MiB intermediate
write/read per 2048-row layer disappears; fallback scratch remains allocated.

Prototype and production parity matrices, exceptional GPU-byte fixtures,
memchecks, primitives, actual-state 43/2053/2181 and Rust perf tests pass.
Both scouts complete without warnings; 915,456 checked logits match
exactly, with zero token/argmax mismatches. Proof scope remains eager/8K.

Automatic comparison: **Pass**, not Improved. Engineering
adopts the repeatable prefill gain: its time envelope is −0.3874% to −0.2364%,
while decode is −0.2643% to +0.0529%, with no meaningful regression. No fixed
percentage floor is imposed. Both arms observe 2190–2197 MHz and equal sampled
means, 2193.2766 MHz (47 samples each). Minimum host availability is
25.5179 GiB; peak memory PSI full avg10 is 0.67, including hashing.

Evidence: `sum-{off,on}/scout.json`, `sum-compare/compare.json`,
`sum-state-{43,2053,2181}.log`, `sum-production-suite.json`,
`sum-prototype/candidate-results/{receipt,resource-compact}.json`,
`sum-source.{json,patch}` (11 frozen source pins), and `sum-model.guard.jsonl`.
Benchmark SHA-256:
`69cf89f7b64fb1029e6acebf65ed6242942e4c16de0595b335f7a5ec9157a234`.

## P8: paired index-score query reuse

Fresh retained-P7 measurement puts index scores at 662.634 ms, 4.176% of
prefill wall time (15.866838 s); decode uses 6.087 ms. Four warps per CTA
reconstruct one key each and reload query/head weights for every key.
Full-counter isolation at histories 2112/8192, rows 32, packed queries and
live contiguous causal positions finds LSU/MIO pressure: 9,943,120 load
requests at 8192, 97.87% L1 hits and no explicit shared/spill accesses.
This supports reducing repeated instructions, rather than staging cached data.
Synthetic operands and untimed score/ID readbacks differ from model producers.

Two keys per warp share query/head-weight loads. One/Two template paths select
partial or full pairs before the head loop; scales are loaded once per key.
Each key retains its RN reconstruction, four RN FMAs, XOR 16/8/4/2/1,
fmax, signed RN weight product and ascending 16-head RN accumulation.
Dispatch requires Warp layout, 32 rows and history above 2048.
`DS4_NAIVE_INDEX_U2=0` restores one key per warp. Planar, narrow/decode,
all-IDs, top-k order, cache representation and allocation stay unchanged.

| Measurement | Off | On |
| --- | ---: | ---: |
| Prefill samples, tok/s | 516.23 / 515.88 / 516.07 | 518.28 / 519.08 / 519.18 |
| Decode samples, tok/s | 18.82 / 18.81 / 18.87 | 18.82 / 18.84 / 18.88 |
| Mean prefill, tok/s | 516.0600 | 518.8467 (+0.5400%) |
| Mean decode, tok/s | 18.8333 | 18.8467 (noise; no decode credit) |
| Nsight prefill index total, ms / calls | 662.158048 / 1728 | 564.822848 / 1728 |
| Nsight prefill wall, ms | 15879.330960 | 15760.095712 |
| Nsight decode wall, ms | 1705.109312 | 1705.729152 |
| Warm isolated H2112, us, 3-pair mean | 156.896 | 134.555 |
| Warm isolated H8192, us, 3-pair mean | 604.030667 | 516.591333 |
| NCU H8192, cache none / all, us | 628.800 / 619.136 | 540.672 / 540.032 |
| Warp instructions | 94,991,120 | 85,959,472 |
| Global load requests / sectors | 9,943,120 / 72,738,640 | 5,625,936 / 37,025,616 |

Registers remain 44/thread, explicit shared memory zero and driver shared
allocation 1 KiB/CTA; no stack or spills in ptxas, no new device allocation.
Warp FFMA/FMUL/FADD/SHFL counts match the original. Predicated-thread FADD
work rises 0.515%; other examined per-key arithmetic matches. An earlier
conditional-second-key prototype improved kernel time 9–13%, but added 77.95%
warp instructions through reconvergence and reread second scales four times.
The retained alternative removes those costs and improves warm time 14.24–14.48%.
Occupancy counters above device limits are artifacts, not physical occupancy.

Correctness: 27 actual-header fixtures check all GPU score and stable-ID bytes
before and after timing, with ties, signed zero, FP8/scale/query edge values,
odd histories, causal boundaries and guarded narrow fallbacks. Finite CPU
checks sample 2648 values per standard fixture; exceptional values use GPU
bytes. Three memchecks, primitive tests and `cargo test -p ds4-perf` pass.
Actual-weight states 43/2053/2181 pass; 2181 exercises four full scoring tiles
and real SWA wrap. Valid-state proof scope follows the earlier rounds.

Both scouts are complete with warnings empty. Compare reports **Pass**, zero
error across 915,456 logits and zero token/argmax mismatches. Adopt the useful
prefill gain: time envelope −0.6356% to −0.3955%, beyond observed variation;
decode −0.3708% to +0.2657% is noise. No additional decode round is credited.
Sampled clocks are 2190–2197 MHz, 47 samples per arm. Minimum sampled host
availability is 30.6316 GiB; host-wide PSI full avg10 peaks at 0.83/0.14.
Telemetry spans warmup, loading, profiling and idle at 10-second intervals.

Local evidence: `scratch/naive/perf-2026-10-01/index2-{off,on}/scout.json`,
`index2-compare/compare.json`, `index2-evidence.{json,md}`, `index2-state-*.log`,
`index2-production-suite.json`, `index2-source.{json,patch}` (12 frozen pins),
`index2-build.guard.jsonl` and `index-u2-unroll-prototype/candidate-results/`.
Benchmark SHA-256:
`db5cbe0433acd561f6fbaa818d1312e0fcc69058e49464fcec6021db9e47a8b2`.

## Source and benchmark pins

The table pins each adopted round's native source and measured benchmark
bytes. The last binary was built from `6e0ae4d9` plus `index2-source.patch`; all 12
frozen source pins match `ee28ed26`. Documentation-only main sync adds no
performance credit.

| Additional round | Source commit | Benchmark SHA-256 | Prefill / decode credit |
| --- | --- | --- | ---: |
| D4: SWA unit exponent | [c7ea8da2](https://github.com/Baekpica/ds4-dfm-rs/commit/c7ea8da2d067533cb60e7361487aeeaf1e54c876) | `2778f4d11f8815f29f8963dcc6ac2adc0f55193cc9bf1512425d8c5799bc31ab` | 0 / 1 |
| P5/D5: SWA ring addresses | [255633e1](https://github.com/Baekpica/ds4-dfm-rs/commit/255633e1b999219cfacd93cf95bef5c8e910533b) | `4a8b6a3458b67afb62dcd6f93b1f4b70e79821999d2a847e63082b774683192a` | 1 / 1 |
| P6/D6: stable warp router | [6a427d5f](https://github.com/Baekpica/ds4-dfm-rs/commit/6a427d5f66fc85f502ee39a2dae67ef1e7657a8c) | `0025ef24e0fc7216b8b26ca15827f3efca0a0f82edcfc7110ad75ba2059a2fc1` | 1 / 1 |
| P7: ordered MoE sum/residual | [6e0ae4d9](https://github.com/Baekpica/ds4-dfm-rs/commit/6e0ae4d903679aa91a780ede42ac6a636e93e8d4) | `69cf89f7b64fb1029e6acebf65ed6242942e4c16de0595b335f7a5ec9157a234` | 1 / 0 |
| P8: paired index queries | [ee28ed26](https://github.com/Baekpica/ds4-dfm-rs/commit/ee28ed26dfcb52f5933fc7555679762afd27c9e2) | `db5cbe0433acd561f6fbaa818d1312e0fcc69058e49464fcec6021db9e47a8b2` | 1 / 0 |

Additional credit: P4/D3. Historical P4/D3 before `2aedeb43` is excluded.
Binary receipts come from `index2-evidence.json` and each retained arm's
`*.binary.sha256`; its pending label predates the adopted P8 commit above.

## Final validation

The required fresh retained-path measurement completed with proof and no scout
warnings: prefill 519.35 / decode 18.85 tok/s, wall 15.769120 / 1.702718 s;
prefill index 564.985888 ms. This single diagnostic is not the headline pair.
`retained-index2/{status,measurement}.json` and `retained-index2/scout/scout.json`
record completion and unchanged source/binary pins.

All 20 sequential host-check stages pass: fmt, clippy, eight Rust/C host
parities, Naive memory/state/main/draft binding, serialized workspace tests,
PLE formats, Bonsai/Qwen references, workspace targets, native server/CLI
(including perf-nvtx), and the model-free live-runner tests.
`final-host-u2/result.json` records completion and unchanged source during
execution. Final local Markdown links, anchors, tables and fences pass.
Independent native review has no blockers; the two documentation findings
(P7 count and stale live-endpoint wording) are corrected.

Qualification remains ordinary eager 8K inference. Captured long-context,
DSpark acceleration, Qwen acceleration and new serving limits are not added.

## Prospective Qwen and shared-path ideas

The [September 30 source review](../naive-2026-09-30/upstream-review.md)
records the upstream claims and ds4 equivalents. These ideas are unmeasured
follow-ups, not adopted rounds or Qwen speed claims.

| Pinned primary implementation | Candidate and required evidence |
| --- | --- |
| [Exact pivot and ordered top-k](https://github.com/jschmied/qwen38-flash-next-gb10/blob/e0ef69d4f5575dad00d34e05479eaf4c6547bace/patches/kernel-det/persistent_topk.cuh) | Avoid repeated candidate sorting/materialization at large histories; preserve masks, ties and selected IDs, then measure each family's real history lengths. |
| [PLE deduplication/staging](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/src/vllm_ple_mmap.py) | Deduplicate repeated row descriptors and expand on GPU if duplicate rates justify it. Existing ds4 page reuse/batched leases remain; measure map/lease work, faults and warm gathers. |
| [Reduced proposal head](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/src/patch_mtp_draft_vocab.py) | Qwen already has a compact proposal head. Other drafters need head profiling, proposal/confidence parity, multilingual acceptance and total decode gates. Naive DSpark acceleration remains unqualified. |
| [GB10 fast-path gate](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/Dockerfile) | Audit real device limits, launch attributes and fallback counts; enable only fitting shapes with scoped capability predicates and whole-model gains. |

Shared quantized-input reuse or producer/consumer fusion also needs each
consumer's layout/arithmetic and invalidation on actual input writes. Existing
fused Gate/Up and Naive SwiGLU already remove some work. Qwen requires its own
whole profile, detailed target evidence and matched fresh-process A/B.
