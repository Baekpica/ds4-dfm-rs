# Inkling MQ85GB on GB10, 2026-09-22

Continuation of [rounds 25–27](inkling-optimization-2026-09-12-r25.md).
MTP is off. Prompt is `speed-bench/promessi_sposi.txt`. The host is one
DGX Spark / GB10. The round tables below were first measured with the
SM clock unlocked (about 2411–2418 MHz). After `nvidia-smi -lgc 300,2200`,
the final binary was remeasured with SM at 2190–2197 MHz and no sample
above 2200.

Adopted here: prefill chunk 2048, then the router logit tile. Locked-clock
cold 8192-token prefill, 64 greedy tokens, three interleaved pairs:

| Path | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| warp | 445.01 | 13.06 |
| tile | 452.58 | 13.07 |

Samples: warp = 445.01 / 444.93 / 445.19;
tile = 453.50 / 452.58 / 452.56.
Gain +1.70%. Decode stays inside 0.08 tok/s. The earlier unlocked
medians (486.39 prefill, 13.55 decode) are not this qualification.
Contexts above 8192, including 64K, were not measured.

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

## Round 29: router logit tile

The MoE gate is a 4096-by-258 BF16 GEMM. 258 is not a multiple of
the 16-row tile, so prefill kept the one-warp-per-output kernel
(about 2.3 ms, 160 launches on the chunk-2048 8K trace). The tile
now covers the first 256 rows and stores the raw FP32 sum; the last
two rows use the same warp reduction. Decode is one token, below
the tile width, and stays on the warp kernel.
`DS4_INKLING_NO_LOGIT_TILE` restores that kernel for every width.

Cold 8192-token prefill, 64 greedy tokens, three interleaved pairs.
Medians:

| Path | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| warp | 479.16 | 13.55 |
| tile | 486.39 | 13.55 |

Samples: warp = 480.86 / 479.16 / 477.81;
tile = 487.04 / 486.39 / 486.23.
Gain +1.51%. Decode stays inside 0.07 tok/s.

The 8192-token frontier (200058 logits) is identical (`max_abs=0`,
same argmax 298). The eight greedy tokens match:
`298 11 2415 7898 6510 11 537 12102`.

## Rejected on the way

Same binary, interleaved, cold 8192 unless noted.

- BF16 linear `cp.async` ring at the 1024-row chunk: 2K prefill
  490.75/490.58/490.65 vs 490.28/489.76. The 2048-row probe was faster
  and the production chunk was not.
- Linear panel schedule at 2048 rows (k=m=4096): 13.47 ms vs 14.37 ms
  for the old order. The panel still wins at 4096 and 8192 rows.
- IQ2 slab L2 prefetch of the next row-group: 8K prefill
  480.19/479.05 vs 479.81/478.01. Decode flat, slightly slower.
- BF16 linear two-stage `cp.async` at the same 32 KB footprint,
  measured after the 2048 chunk: 2048-row k=m=4096 was 14.13 ms vs
  13.98 ms, and the 4097-row panel was 31.3 ms vs 28.3 ms. Exact,
  slower on the shapes prefill actually launches.
