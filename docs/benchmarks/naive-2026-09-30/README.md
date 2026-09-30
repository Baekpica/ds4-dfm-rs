# Naive GB10 optimization rounds

Retained rounds from the September 30 request: prefill **0/3**, decode **1/3**.
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
