# P4: pack F32 indexer queries

After P3, a fresh 32K whole-workload capture attributes 15.474392 s,
19.295% of prefill GPU time, to the signed indexer. Top-k takes only 1.303%.
The faithful isolated score target uses 32 queries, 16 heads of width 128,
32K E4M3 history, original F32 scales, signed weights and causal masks.
Values and cache residency are synthetic; geometry and arithmetic match.

Full-counter NCU identifies an LSU instruction bottleneck: 99.27% L1/TEX
throughput and 90,135,376 global load requests. Pack the four dimensions
consumed by each lane during the existing query round-trip. A vector load
replaces four scalar loads while preserving the four F32 FMAs, XOR tree,
serial signed-head sum, stable ties and selected IDs. No tensor or KV
allocation changes. `DS4_NAIVE_INDEX_PACK=0` restores planar queries.

| Unprofiled tok/s | Baseline samples | Candidate samples | Mean change |
| --- | --- | --- | ---: |
| 8K prefill | 497.83, 497.94, 498.43 | 506.32, 506.07, 506.09 | +1.625% |
| 8K decode | 17.39, 17.44, 17.45 | 17.39, 17.48, 17.46 | +0.096% |
| 32K prefill | 408.13, 407.79, 407.67 | 431.46, 431.17, 431.19 | +5.740% |
| 32K decode | 17.16, 17.17, 17.19 | 17.18, 17.15, 17.24 | +0.097% |

Both shapes use three fresh processes per side with separate warmups,
book input, chunk 2048 and 32 greedy output tokens. Complete scouts have
no warnings and hash all model shards before/after. Every shape checks
915456 logit values with zero difference, token mismatch and argmax mismatch.
Full-vocabulary frontier and token hashes match across all samples.
Receipts: [paired evidence](index-pack-evidence.json).

The 8K automatic comparison returns `Improved`. The 32K comparison stays
`Inconclusive`: first-decode median 93.0→93.8 ms, with overlapping sample
ranges and a 3.156% upper envelope exceeding its 3% screening threshold.
The threshold and original verdict are unchanged. Retain the reproducible
prefill gain; ordinary decode envelopes overlap at both shapes. This counts
as a fourth prefill round, not a fourth decode gain.

Cold NCU score time falls 3.37→2.42 ms. Global load requests fall
90,135,376→39,827,536; read sectors remain 291,366,736. Executed warp
instructions fall 426,581,456→380,465,936. Registers rise 42→44 but both
allocate 48 per thread; achieved occupancy is 74.19→73.39%.
Three warm 32K/32-query samples fall 3.360–3.363→2.416–2.418 ms.
The isolated 1M/32-query samples fall 107.639–107.707→77.479–77.525 ms;
this is not full 1M serving or end-to-end throughput.

Packing costs more producer work: store sectors rise 8192→32768 and warp
instructions 216292→230048 for 512 head rows. Registers rise 17→20, both
allocated as 24. The complete 32K query-producer sum rises
0.012662→0.013098 s. Including that cost, whole Nsys prefill wall time falls
80.347604→76.014088 s; indexer time falls 15.474971→11.143397 s. Decode wall
time is 1.879945→1.873257 s. Profiled timings are separate from the table.

Independent placement/score fixtures pass byte-exactly. Bounded memcheck
passes 32769 history / 32 queries and 1M history / one query with zero errors.
The earlier oversized sanitizer probe was terminated and is not a pass.
An actual-weight 2053-token fixture activates DSA selection and verifies
scalar/packed, width 1/7 and changed rejected proposals: all 48 hidden states,
full logits, target KV, index codes/scales and draft state remain byte-exact.

Observed clocks are 2184–2197 MHz, with active samples 2184/2190 MHz inside
the user lock 300–2200. Benchmark executable SHA-256:
`369a2464f1a5c484d30c5c655699eee9b5465e0dc69227d88c4790d2aabc4931`.
Baseline: `88c1277b` plus the candidate implementation with its switch off.
Raw evidence: ignored `scratch/naive/round-p4-{8k,32k}-pair-*`,
`index-{32k,query}-{base,pack}*.ncu-rep`, `index-pack-warm.log`,
`index-pack*-memcheck.log` and `index-state.log`.
