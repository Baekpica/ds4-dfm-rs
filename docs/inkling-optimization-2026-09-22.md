# Inkling MQ85GB on GB10, 2026-09-22

Continuation of [rounds 25–27](inkling-optimization-2026-09-12-r25.md).
MTP is off. Prompt is `speed-bench/promessi_sposi.txt`. The host is one
DGX Spark / GB10. `nvidia-smi -lgc` needs root here, so the SM clock was
observed rather than locked: about 2411–2418 MHz while a prefill was
running. Comparisons below are same-hour, same binary, interleaved.

## Round 28: prefill chunk 2048

The 2K trace's largest kernels are still the IQ2 slabs, the BF16 linear
tile, and the shared Q8 pipe. A cp.async double buffer on the linear tile
was faster in a 2048-row probe and lost at the 1024-row chunk the runtime
actually launches (2K end to end 490.6 tok/s either way). Widening the
chunk is the dispatch change that makes those tiles fatter.

`DS4_INKLING_PREFILL_CHUNK=1024` restores the previous cap. 4096 on the
8K shape was slower (373.95 tok/s, one sample).

Cold 8192-token prefill, 64 greedy tokens, three interleaved pairs.
Medians:

| Chunk | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1024 | 469.14 | 13.53 |
| 2048 | 476.86 | 13.54 |

Samples: 1024 = 469.55 / 469.14 / 468.55; 2048 = 476.95 / 476.86 / 476.33.
Gain +1.65%. Decode stays inside 0.05 tok/s.

One 2048-token pair: 490.39 → 496.78 tok/s prefill, decode 16.83 → 16.85.

The 8192-token frontier (200058 logits) is identical across the two caps
(`max_abs=0`, same argmax 298). The eight greedy tokens match:
`298 11 2415 7898 6510 11 537 12102`.

Context 8257 scratch, MTP off: 973,649,152 bytes at cap 1024 and
1,631,188,224 bytes at cap 2048. `tests/test_inkling_session --memory-quotes`
accepts the new default.
