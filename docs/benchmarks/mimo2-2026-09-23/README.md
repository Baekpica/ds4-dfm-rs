# MiMo GB10 measurements

Artifact: `Baekpica/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF`,
`MQ-IQ2-XXS-XS-Q8-MM-BF16`. Prompt: `speed-bench/promessi_sposi.txt`.
One GB10; user-managed 300–2200 MHz clock range preserved.

`baseline-{1,2,3}.csv` contains the original PR #56 baseline:
2,048-token incremental prefill and 128 greedy tokens at each frontier
through 65,536 tokens, one warm session per fresh process. MTP is off.
The receipt identifies the source and executable.

Earlier rounds use different workloads; compare off/on within each row:

| Round | Workload | Prefill tok/s, median of three | Decode tok/s |
| --- | --- | --- | --- |
| P1: share full-attention KV | 2K steps through 64K, 16 generated; shown at 64K | 61.40 → 71.09 | 1.69 → 1.70 |
| P2: earlier tile dispatch | 2K steps through 16K, 8 generated; shown at 16K | 218.54 → 263.87 | 5.53 → 5.54 |
| P3: L2 eviction hints | Cold 32K, 8 generated | 255.89 → 275.32 | 3.07 → 3.07 |

[Original round metadata](original-rounds.json) records adopted commits,
CSV hashes, medians and observed busy clocks. Each round directory contains
the six raw CSVs and its receipt. These cold and shorter-generation results
are separate from the 128-token card sweep.

P4 tensor-core full attention is still a candidate. Its three cold 32K
pairs measured 275.27 → 442.05 prefill tok/s (+60.59%), with decode at
3.15 tok/s on both paths. [Raw CSVs and receipt](p4/receipt.json) identify
the binary, source hashes, controls and observed clocks. These results
are not the incremental 64K curve.

The arithmetic changes summation order. The 32K open-ended continuation
differs in 112 of 128 token IDs; it is not an equivalent continuation.
Five answerable 8K tasks retained their expected meaning, and both 65K
retrieval/code tasks produced identical correct answers. The
[8K review](p4/quality-8k-review.json) and [65K evidence](p4/quality-65k.json)
include scope limits and numerical differences. The strengthened full-history
FP64 and updated-KV graph tests passed. Architecture checks and promotion
to the default path remain.

P5 replaces per-half KV staging with aligned asynchronous copies. Each
cold frontier uses three fresh-process A/B pairs and 128 generated tokens:

| Frontier | Prefill tok/s, median | Change | Decode tok/s, median |
| --- | --- | --- | --- |
| 32K | 441.04 → 961.93 | +118.10% | 3.15 → 3.15 |
| 64K | 223.44 → 835.39 | +273.88% | 1.70 → 1.69 |

All six pairs produced exactly equal 152,576-entry logits and 128 generated
token IDs. The cold 64K/32K prefill ratio increased from 50.66% to 86.85%.
This is separate from the original incremental 64K/2K retention metric.
[P5 evidence](p5/receipt.json) records the controls, clocks and CSVs.
The strengthened [primitive checks](attention-check.json) passed: actual
32K history, ragged tiles, sink handling, SWA ring masks and updated-KV
captured/eager replay. The sampled full-attention FP64 error was at most
8.65e-5, versus 9.76e-5 for the original walking path. Architecture checks
and promotion to the default path remain.

The [P5 phase profile](p5/phase-profile.json) contains one 78.43-second
`ds4.prefill` range. Global attention accounts for 27.4% of GPU kernel time;
SWA accounts for 7.9%. The instrumented run generates eight tokens after
capture and is excluded from throughput comparisons.

`../plot-mimo2.py` validates all 32 incremental frontiers, 128 generated
tokens per frontier and three CSVs per series. It writes median/min–max
curves and a JSON summary with CSV hashes and 64K/2K retention. The final
comparison graph awaits the completed optimization campaign.
