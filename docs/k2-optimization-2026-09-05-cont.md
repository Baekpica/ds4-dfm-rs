# K2 MQ87: continuation campaign (6+6 new accepts)

Base: `2c56fc9` on `feature/k2-horizon-375b-serving`.
The earlier six-round write-up is historical and is not counted here.

## Protocol

GB10 / sm_121a, CUDA 13.3, Rust `ds4-bench`, MQ87 four-shard GGUF.
Raw `speed-bench/promessi_sposi.txt`, first 8192 tokens, 64 greedy decode
tokens, allocated context 8257. Fixture SHA256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
Default memgov, no MTP or prefix reuse. Fresh process per cell.
`nsys` → bottleneck kernel → owner/worker down → `ncu` → change → A/B.
Kill every model PID and run `/usr/local/bin/clear_cache` between cells.
After each retained round, update this file and the HF model card
`Baekpica/K2-Horizon-375B-A23B-Mixed-Quant-GGUF` (Qwen card style:
date, engine SHA, corpus, token counts, cold/warm, command/env).

## Scoreboard

Same command and corpus every cell. Campaign baseline is HEAD `2c56fc9`
(nsys 8K+64, no owner): **286.55 prefill / 5.52 decode tok/s**.
Each finished round reports both rates and the delta against this
baseline (and against the last accepted round of that workload).

| Round | Prefill tok/s | vs baseline | Decode tok/s | vs baseline | Verdict |
|---|---:|---:|---:|---:|---|
| baseline `2c56fc9` | 286.55 | — | 5.52 | — | locked |
| Prefill 1 kill switch | 286.91 | +0.13% | 5.51 | -0.2% | same path |
| Prefill 1 assign-major `096fc9c` | **295.04** | **+3.0%** | 5.47 | -0.9% | retained |
| Prefill 2 IQ2_XS worklist | 272.03 / 272.14 | **-5.1%** | 5.51 | -0.2% | rejected |
| Prefill 3 IQ1_M slot-loop `569d13c` | **302.94** | **+5.7%** | 5.50 | -0.4% | retained |
| Prefill 4 IQ1_M row-tile | 290.18 / 295.28 | **-4.2%** | 5.50 | -0.4% | rejected |
| Prefill 5 chunk 1024 | 307.65 | **+7.4%** | 5.51 | -0.2% | rejected (drift) |
| Decode 1 GQA tile-6 | 271.21 | — | **4.13** | **-25%** | rejected |
| Decode 1 GQA tile-2 | 270.88 | — | **6.91** | **+25.2%** | retained |
| Prefill 6 kill switch | 305.11 / 305.04 | +6.5% | 7.14 | +29.3% | same path as D1 |
| Prefill 6 IQ1_M MMQ tile `7d756a9` | **444.50 / 444.65** | **+55.1%** | 7.14 | +29.3% | retained (numeric contract below) |

Prefill 1 vs last accepted (the locked baseline): **+8.49 tok/s**.
Prefill 3 vs last accepted (`096fc9c` 295.04): **+7.90 tok/s**.
Decode 1 (tile-2) vs locked baseline 5.52: **+1.39 tok/s**.
Prefill 6 vs last accepted (`569d13c` 302.94): **+141.56 tok/s (+46.7%)**;
same-binary on/off 305.11 → 444.50: **+45.7%**. Decode is unchanged
against its own kill switch (7.14 / 7.14); the +29.3% column is the D1
kernel plus this compile, not a Prefill 6 claim.
Decode IDs and all 250624 frontier logits are bit-identical across
the kill-switch, Prefill 1, Prefill 3, and Decode 1 cells.

## Rounds

| Round | Hypothesis | Measured result | Verdict |
|---|---|---|---|
| Prefill 1 | Assign-major IQ1_M MMVQ, 3-D grid `(M, tokens, used)`, ncols=1 4-warp walk. `DS4_MMQ_IQ1M_PREFILL=0` is the per-token loop. | Synth 17/257/8192 bit-identical (8192×8 = 65536). 8K A/B 286.91 → 295.04 prefill, logits/IDs exact. | Retained |
| Prefill 2 | Reuse compact worklists for raw IQ2_XS down (4.3% of HEAD GPU time). | Real-weight worklist bit-identical; NT4096 17.9 → 8.3 ms. Same-binary on/off 262.86 → 272.03, logits/IDs exact, but both cells sit below the locked 286.55 baseline. Repeat on 272.14. Patch reverted. | Rejected |
| Prefill 3 | Walk top-k slots in one `(M, tokens)` IQ1_M block so the Q8_1 row is reused. `DS4_MMQ_IQ1M_SLOT_LOOP=0` restores the 3-D grid. | Synth 17/257/8192 bit-identical (3-D vs slot-loop). 8K A/B 294.38 → 302.94 prefill, logits/IDs exact vs P1. | Retained |
| Prefill 4 | Tile 4 IQ1_M output rows per block (`DS4_MMQ_IQ1M_ROW_TILE`). | Synth bit-identical; same-binary tile1 295.28 / tile4 290.18, both below last accepted 302.94. Same-TU kernel body change, same class as P2. Reverted. | Rejected |
| Prefill 5 | 1024-token EXAONE prefill chunks vs default 512. | 271.5 → 307.65 on this binary, but rel RMS 0.0636 and 56/64 greedy IDs differ. Reverted. | Rejected |
| Decode 1 | Share each KV row across GQA query heads. Same per-head softmax order as `exaone_attn_head`. `DS4_EXAONE_ATTN_GQA=0` is one block per head. | Tile-6: bit-identical, 5.51 → 4.13 (spill). Tile-2: bit-identical, 5.51 → 6.91. | Retained (tile-2) |
| Prefill 6 | IQ1_M MMQ tile: upstream has no `load_tiles_iq1_m`, so the 8 edge-layer gate/up tensors ran the decode MMVQ per assignment (`ds4_iq1_m_moe_assign_kernel`, 28.4% of the post-D1 8K+64 trace, 41 ms per launch vs 3.8 ms for the IQ1_S worklist MMQ). New tile on the per-16 Q3_K/IQ2_XS layout (int8 = 8·(grid+delta) ∈ {±1,±7,±9}, `x_df` = d·(2s+1)/8, exact) on the compact worklist from 256 routed rows. `DS4_MMQ_IQ1M_WORKLIST=0` restores the assign-major MMVQ. | Real layer-3 gate (K 6144, M 1792, 192 experts): nt=512 52.98 → 5.68 ms, nt=257 26.96 → 4.65 ms, rel RMS 1.9e-4 vs the MMVQ tier (Q8_1 activation rounding bound, see below). 8K A/B same binary 305.11 / 305.04 → 444.50 / 444.65 prefill, decode 7.14 both. Frontier rel RMS 0.0624 vs D1, argmax same, 54/64 greedy IDs differ from token 6. | Retained under the numeric contract below |

## Prefill 6 numeric contract

Prefill 6 is the first retained round that changes arithmetic rather
than scheduling, so the bit-identity gate does not apply to it. What
was measured:

- Kernel parity: the tile and the assign-major MMVQ compute the same
  integer dot (the ±7/±9 encoding is exact). What differs is the
  activation tier: MMVQ keeps the Q8_1 scale as fp16 in `block_q8_1.ds`
  and rounds `x/(amax/127)`, MMQ keeps it as fp32 and rounds
  `x·(127/amax)`; fp32 accumulation order differs too. On random rows,
  synthetic valid blocks (all scale/delta patterns, checked and
  unchecked tiles, one and two K iterations) and the real layer-3 gate
  tensor agree at rel RMS 1.80e-4 … 1.97e-4, 1-cos ≤ 2e-8. On rows whose
  32-value blocks are exact Q8_1 points with scale 2^-8 (both producers
  emit identical q and scale), the same shapes agree at rel RMS 9.5e-8
  (synthetic) and 2.6e-7 (real, K 6144), 1-cos ≤ 1e-12: fp32 order only.
  So the 1.9e-4 is the MMVQ tier's fp16 scale plus producer rounding,
  not the tile. The IQ1_S/IQ2_XXS/IQ2_XS layers already run the MMQ
  tier in prefill (`tests/test_exaone_kernels`, "IQ1_M tile" and
  "IQ1_M tile q8-exact" rows).
- Fixture drift: frontier rel RMS 0.0624, KL(D1‖P6) 2.2e-2 nats,
  argmax 199017 in both (p 0.174 vs 0.169), top-10 overlap 9/10,
  top-50 46/50; greedy IDs diverge at token 6 (54/64 differ). Both
  64-token continuations are coherent Italian.
- Comparator: the rejected Prefill 5 (1024-token chunks, no kernel
  change, pure fp32 reorder) has the same signature on the same
  fixture: rel RMS 0.0636, KL 2.5e-2, top-10 10/10, divergence at
  token 6, 56/64 differ. This 1.75-bit MoE flips near-tie top-8 routes
  under any fp perturbation, and 8,192 tokens of KV carry the flips to
  the frontier, so rel RMS ≤ 1e-4 is reachable only by scheduling-only
  rounds.
- Short-context drift (same binary, on vs off, one process with
  incremental frontiers at 512 / 1024 / 1536 / 2048 and 16 greedy tokens
  each): rel RMS 7.2e-2 / 1.04e-1 / 6.6e-2 / 8.6e-2, KL 3.3e-2 / 3.1e-2 /
  9.1e-3 / 1.7e-3, same argmax at all four, top-10 overlap 8 / 9 / 8 / 10,
  greedy IDs differing 10 / 2 / 0 / 1 of 16. The drift is at its 8K level
  after one 512-token chunk and does not grow with context: a per-forward
  routing-flip effect, not KV accumulation.
- Determinism: two Prefill 6 processes are byte-identical in logits
  and IDs; the kill switch is byte-identical to D1.
- Harness: the first full `tests/test_exaone_kernels <gguf>` pass after
  this round reported `invalid argument` on later IQ1_S/IQ2_XXS cells
  (129-row ragged tile, one-token vec). Bisect: the synthetic IQ1_M
  tests register their `posix_memalign` map with CUDA (pinned) and then
  freed it, so later `malloc` result buffers could straddle the pinned
  range and fail `cudaMemcpy` (the "patchwork" mode noted in
  `cuda_model_range_ptr`). The P1 synthetic test had the same pattern;
  the second registered map made it hit. Both maps now live until exit;
  compute-sanitizer showed no kernel fault (real-weight section all
  green under the sanitizer).

Contract: IQ1_M gate/up prefill runs on the MMQ tier (MMQ Q8_1 producer,
MMA accumulation), the same tier as the other routed IQ types. Decode
and shapes below 256 routed rows keep the MMVQ tier.

Post-P6 nsys (8K+64, 443.28 / 7.11 under the profiler): total kernel
time 41.68 s → 28.72 s. IQ1_M is the worklist kernel at 288 × 3.50 ms
(3.5%) instead of the assign kernel at 288 × 41.06 ms (28.4%). Ranking:
IQ1_S worklist 19.8% (3.16 ms avg), `exaone_attn_decode_gqa_kernel`
16.7% (1.23 ms), IQ2_XXS worklist 15.0% (4.79 ms), q8 aligned dense vec
8.9%, Q8_0 MMQ 7.2%, IQ2_XS rectangular 5.9% (11.8 ms avg), HMMA prefill
attention 4.2%, IQ1_M worklist 3.5%. Evidence:
`scratch/k2-opt-20260905-cont/{p6-off,p6-on,p6-on2,p6-off2,p6ctx-off,p6ctx-on,p6-nsys,p6}/`.
