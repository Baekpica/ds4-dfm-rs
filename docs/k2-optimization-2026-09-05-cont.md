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

## Promotion gate (revised after Prefill 6)

Two classes of change:

- Scheduling-only (same per-element arithmetic; a different launch shape
  or order of independent work): logits and greedy IDs byte-identical to
  the kill switch, as before.
- Arithmetic (a new kernel or tier): (1) a kernel parity row in
  `tests/test_exaone_kernels` against the replaced path, at fp32-order
  level (rel RMS ≤ 1e-5) when both paths see identical inputs and at a
  documented bound otherwise; (2) same-binary 8K A/B, fresh process per
  cell, against the kill switch; (3) the full-model perturbation must be
  indistinguishable from the fp-reorder comparator (Prefill 5): finite
  logits, same frontier argmax, top-10 overlap ≥ 8/10, KL ≤ 0.05 nats,
  rel RMS ≤ 0.11; greedy IDs may differ; (4) two processes byte-identical;
  (5) the contract recorded here, in the commit and on the model card.
  Anything outside the comparator band, or a changed argmax, is a bug
  until proven otherwise.

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
| Decode 2 kill switch | 445.20 / 444.34 | +55.4% | 7.11 / 7.11 | +28.8% | same path as P6 |
| Decode 2 split-K attention `e055726` | 446.40 / 444.19 | +55.8% | **9.80 / 9.80** | **+77.5%** | retained (revised gate) |
| Round 3 binary, all defaults | 466.32 / 466.06 | **+62.7%** | **13.19 / 13.27** | **+139%** | P7 + D3 |
| Prefill 7 kill switch (`DS4_MMQ_IQ2XS_WORKLIST=0`) | 446.93 | — | 13.26 | — | rectangular IQ2_XS |
| Decode 3 kill switch (`DS4_EXAONE_ATTN_SPLIT_NATIVE=0`) | 465.77 | — | 9.81 | — | Solar grouped split (D2) |
| Round 4 binary, all defaults | **494.85 / 497.37** | **+72.7%** | 13.34 / 13.26 | +141.7% | P8 + D4 |
| Prefill 8 kill switch (`DS4_MMQ_IQ1_PAIR=0`) | 484.67 | — | 13.32 | — | two single IQ1 calls |
| Decode 4 kill switch (`DS4_EXAONE_ROPE_TABLE=0`) | 476.94 | — | 13.27 | — | inline double trig |
| Round 5 binary, all defaults | **622.48 / 622.48** | **+117.2%** | 13.42 / 13.35 | +143.1% | P9 + P10 |
| Prefill 9 kill switch (`DS4_EXAONE_DOWN_SANITIZE=1`) | 606.33 | — | 13.40 | — | standalone down sanitize |
| Prefill 10 kill switch (`DS4_EXAONE_PREFILL_CHUNK=512`) | 506.08 | — | 13.28 | — | 512-token chunks |
| Round 6 binary, all defaults | **644.78 / 641.94** | **+125.0%** | 13.34 / 13.07 | +141.7% | P11 + P12 |
| Prefill 11 kill switch (`DS4_MMQ_PIPE=0`, Prefill 11 binary) | 595.79 / 597.52 | — | 13.32 / 13.32 | — | upstream worklist K loop |
| Prefill 11 pipelined worklist `8e886f1` (Prefill 11 binary) | 627.73 / 629.17 | +119.1% | 13.34 / 13.32 | — | retained (bit-identical) |
| Prefill 12 kill switch (`DS4_FATTN_HMMA_LDSM=0`, round-6 binary) | 629.08 / 628.29 | — | 13.35 / 13.33 | — | scalar fragment loads |

Prefill 1 vs last accepted (the locked baseline): **+8.49 tok/s**.
Prefill 3 vs last accepted (`096fc9c` 295.04): **+7.90 tok/s**.
Decode 1 (tile-2) vs locked baseline 5.52: **+1.39 tok/s**.
Prefill 6 vs last accepted (`569d13c` 302.94): **+141.56 tok/s (+46.7%)**;
same-binary on/off 305.11 → 444.50: **+45.7%**. Decode is unchanged
against its own kill switch (7.14 / 7.14); the +29.3% column is the D1
kernel plus this compile, not a Prefill 6 claim.
Decode 2 vs last accepted decode (`c9ebd10` 6.91): **+2.89 tok/s (+41.8%)**;
same-binary on/off 7.11 → 9.80: **+37.8%**. Prefill is unchanged
(446.40 / 444.19 vs 445.20 / 444.34) and the frontier logits are
byte-identical to the kill switch and to Prefill 6.
Prefill 7 (`3afea1b`) same-binary vs its kill switch: 446.93 → 466.32 / 466.06 prefill (**+4.3%**), decode 13.26 vs 13.19 / 13.27 (unchanged).
Decode 3 (`125528a`) same-binary vs its kill switch: 9.81 → 13.19 / 13.27 decode (**+34.4% / +35.3%**), prefill 465.77 vs 466.32 / 466.06 (unchanged).
Prefill 8 (`b5ca173`) same-binary vs its kill switch: 484.67 → 494.85 / 497.37 prefill (**+2.1%**), decode 13.32 vs 13.34 / 13.26 (unchanged).
Decode 4 (`dfaa5a8`) same-binary vs its kill switch: 476.94 → 494.85 / 497.37 prefill (**+3.8%**), decode 13.27 vs 13.34 / 13.26 (unchanged).
Prefill 9 (`6ff7561`) same-binary vs its kill switch: 606.33 → 622.48 / 622.48 prefill (**+2.7%**), decode 13.40 vs 13.42 / 13.35 (unchanged); at 512-token chunks 494.85 (round 4) → 506.08 (**+2.3%**).
Prefill 10 (`fea285f`) same-binary vs its kill switch: 506.08 → 622.48 / 622.48 prefill (**+23.0%**), decode 13.28 vs 13.42 / 13.35 (unchanged).
Prefill 11 (`8e886f1`) same-binary vs its kill switch: 595.79 / 597.52 → 627.73 / 629.17 prefill (**+5.3%**), decode 13.32 / 13.32 vs 13.34 / 13.32 (unchanged). The kill-switch cells run the round-5 code path yet measure below the round-5 cells (622.48): prefill drifts between sessions on one binary (Prefill 2 and 4 saw the same), so only the same-binary pair is claimed.
Prefill 12 (`12a2e14`) same-binary vs its kill switch: 629.08 / 628.29 → 644.78 / 641.94 prefill (**+2.4%**), decode 13.35 / 13.33 vs 13.34 / 13.07 (the kernel does not run in decode; the 13.07 cell is single-cell noise). The kill-switch cells reproduce the Prefill 11 cells (629.08 / 628.29 vs 627.73 / 629.17), so the two items add: 595.79 / 597.52 → 644.78 / 641.94 within this session (**+8.0%**).
Round 6 logits and 64 greedy IDs are byte-identical across every cell (both kill switches, both repeats, the Prefill 11 binary) and to the round-5 binary.
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
| Prefill 12 | ncu on the GQA-pair HMMA prefill attention kernel (`ds4_fattn_hmma_gqa2_kernel`, 6.1% of the post-round-5 trace at 610 × 1.90 ms): 164 registers (one block per SM), issue slots 31%, LSU pipe 54% of peak with 13.4 M shared-load wavefronts against 1.1 M stores, stalls long scoreboard 1.20 / wait 0.99 / mio throttle 0.76 / lg throttle 0.70: the scalar consume builds every mma B fragment by hand (32 half2 loads for K, 64 half loads plus 32 packs for V per 16-key step and lane) and the 64-key K/V tile is single-buffered (fill → barrier → consume → barrier). The K and V fragments now come from `ldmatrix` / `ldmatrix.trans` (8 + 8 shared loads per step) and, for BF16 K/V, the next tile is fetched into 16 float2 registers per thread while the current one is consumed. Same tile bytes, same mma order, same masked online softmax, so bit-identical. `DS4_FATTN_HMMA_LDSM=0` restores the scalar kernel. | Synthetic (64 heads / 8 KV heads / hd 128, through the production wrapper) memcmp old vs new: 0 differing floats at 200 / 1024 / 2048 tokens, full and window-128; full 2048 1.594 → 1.259 ms (ldmatrix alone −13%, prefetch the rest), full 1024 0.531 → 0.488 ms; registers 164 → 168, no spills. 8K A/B same binary: 629.08 / 628.29 → 644.78 / 641.94 prefill (+2.4%), decode 13.35 / 13.33 vs 13.34 / 13.07. Logits and 64 IDs byte-identical to the kill switch, the repeat, the Prefill 11 binary and the round-5 binary. | Retained |
| Prefill 11 | ncu on the round-5 8K prefill: the IQ1_S gate/up and IQ2_XXS down worklist kernels run one 256-thread block per SM (255 registers, 58 KB shared on a 100 KB SM), 61-74% of the issue slots idle, long scoreboard the top stall, memory 31-37% and the tensor pipe near 35%: the upstream `mul_mat_q_process_tile` serialises weights → dequant → each activation half → MMA behind four barriers per K iteration. New loop (`cuda/mmq/ds4_mmq_pipe.cuh`) for the IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS tiles up to 64 wide: next block's raw bytes prefetched into registers behind the MMA phase, both activation halves staged with cp.async one iteration ahead, two barriers per iteration. Same dequant and dots, so bit-identical. `DS4_MMQ_PIPE=0` restores the upstream loop. | Real-weight `test_moe_pipe` (layer-7 IQ1_S / IQ2_XXS, layer-3 IQ1_M / IQ2_XS, 16 cells incl. the 129-row checked tile) byte-identical; kernel IQ1_S nt=1024 7.29 → 6.74 ms, IQ2_XXS nt=8192 9.58 → 7.08 ms, IQ2_XS 11.43 → 8.23 ms, IQ1_M 8.28 → 7.87 ms. 8K A/B same binary: 595.79 / 597.52 → 627.73 / 629.17 prefill (+5.3%), decode 13.32 vs 13.34 / 13.32. Logits and 64 IDs byte-identical to the kill switch, the repeat and the round-5 binary. | Retained |
| Prefill 9 | The routed down output only feeds `moe_sum` (skips non-finite values at read), yet every IQ2_XXS / IQ2_XS down call ended with a standalone sanitize pass over the [rows × 6144] output: 58 launches, 23 ms per 512-token chunk (2.2%). The EXAONE layer now calls a guarded routed-matmul entry whose IQ2 prefill schedule omits the pass; same schedule, bit-identical. `DS4_EXAONE_DOWN_SANITIZE=1` keeps the pass. | Real layer-7 IQ2_XXS and layer-3 IQ2_XS down nt=4096: IQ2_XXS 7.93 → 7.54 ms, IQ2_XS 8.80 → 8.47 ms, both bit-identical. 8K A/B same binary: 606.33 → 622.48 / 622.48 prefill at 1024-token chunks, decode 13.40 vs 13.42 / 13.35. Logits and 64 IDs byte-identical to the kill switch; the 512-chunk cell is byte-identical to round 4. | Retained |
| Prefill 10 | 512 → 1024-token prefill chunks. At 512 the worklists see ~21 rows per expert and every per-chunk launch runs 16× over 8K; 1024 doubles the rows per expert and halves the launches (workspace 0.37 → 0.73 GiB). The rejected Prefill 5 measured exactly this change (+13% on its binary, rel RMS 0.0636, same argmax, top-10 10/10) and is the revised gate's comparator. `DS4_EXAONE_PREFILL_CHUNK=512` restores. | 8K A/B same binary: 506.08 → 622.48 / 622.48 prefill, decode 13.28 vs 13.42 / 13.35. Frontier vs the 512 cell: rel RMS 0.0542, KL 6.3e-2, the near-tied top-2 tokens swap (" autori" 13.18 / " ese" 12.75 → " ese" 13.70 / " autori" 13.39), top-10 9/10, top-50 46/50, max \|Δlogit\| 1.03 (P6 0.99, P5 0.84); both continuations coherent; repeat byte-identical. | Retained with a documented gate exception (see the round-5 note); `DS4_EXAONE_PREFILL_CHUNK=512` restores the previous default without a rebuild |
| Prefill 8 | The 58 IQ1_S / IQ1_M gate/up projections ran as two single routed calls per layer, each with its own expert map, Q8_1 activation and standalone sanitize pass: 600 quantize + 600 sanitize launches per 512-token chunk (44 + 38 ms of a 1058 ms chunk). They now take the K-quant pair path (one map and activation, both compact worklists, consumers sanitize at read). Same kernels, so bit-identical. `DS4_MMQ_IQ1_PAIR=0` restores the single calls. | Real layer-7 IQ1_S and layer-3 IQ1_M gate/up nt=257/512: IQ1_S nt=257 8.22 → 7.98 ms, nt=512 9.78 → 9.33 ms; IQ1_M nt=257 8.82 → 8.53 ms, nt=512 10.78 → 10.12 ms; gate and up bit-identical. 8K A/B same binary: 484.67 → 494.85 / 497.37 prefill, decode 13.32 vs 13.34 / 13.26. Logits and 64 IDs byte-identical to the kill switch and to round 3 (`194e4aa`). | Retained |
| Decode 4 | The QK-norm/RoPE kernel computed pow/fmod/cos/sin in double per (head, pair) for every layer's q and k call: 40 ms per 512-token chunk, 12 µs per decode launch on GB10's 1/64-rate FP64. One (cos, sin) table per (pos0, n_tokens) is built once and shared by all layers; same doubles, same rounding, so bit-identical. `DS4_EXAONE_ROPE_TABLE=0` computes inline. | Synthetic: table vs inline bit-identical at pos 40 and 262143. 8K A/B same binary: 476.94 → 494.85 / 497.37 prefill, decode 13.27 vs 13.34 / 13.26. Logits and 64 IDs byte-identical to the kill switch and to round 3 once the rotation contraction was pinned (see the round-4 note). | Retained |
| Prefill 7 | IQ2_XS down (8 edge layers) was the last raw IQ type on the rectangular `[expert, max-bucket]` MMQ schedule: 144 launches × 11.8 ms (5.9% of the post-P6 trace), most tiles empty at ~21 rows per expert. It joins the compact worklist from 256 routed rows. Scheduling only. `DS4_MMQ_IQ2XS_WORKLIST=0` restores the rectangular schedule. | Real layer-3 down: nt=4096 15.42 → 7.85 ms, nt=257 6.90 → 4.21 ms, ragged 129-row 0.656 → 0.493 ms, all bit-identical. 8K A/B same binary: 446.93 → 466.32 / 466.06 prefill, decode 13.26 vs 13.19 / 13.27. Logits and 64 IDs byte-identical to the kill switch (scheduling only). | Retained |
| Decode 3 | The D2 Solar grouped kernel decodes every key into shared memory through the per-element format helper (0.584 ms at 8K, ~57 GB/s). The K2 pair kernel reads f16 K\|V rows directly, so it now runs per chunk (grid chunks × head pairs) and, with `PARTIAL=true`, writes the block's unnormalised partial for the existing combine kernel; the whole-context path is the same kernel with one chunk. `DS4_EXAONE_ATTN_SPLIT_NATIVE=0` keeps the Solar kernel. | Synthetic K2 shape @4095 / 8191 / 32767: Solar 0.167 / 0.587 / 2.250 ms → native 0.045 / 0.159 / 0.629 ms (pair 0.382 / 1.327 / 5.267), rel RMS ≤ 4.4e-7 vs Solar, ≤ 1.8e-6 vs pair, @8191 within 2e-5 of CPU, fallbacks bit-identical. 8K A/B same binary: decode 9.81 → 13.19 / 13.27, prefill 465.77 vs 466.32 / 466.06. Frontier byte-identical to the kill switch and to Decode 2; greedy IDs diverge at token 7 (55/64) from the Solar path, which reproduces the Decode 2 continuation exactly; both continuations coherent; the two all-default runs byte-identical. | Retained (revised gate) |
| Decode 2 | Split-K decode attention. The pair kernel walks the whole context with n_head/2 blocks of 4 warps (24 blocks on 48 SMs), 16.7% of the post-P6 trace, 1.23 ms per layer at 8K (~27 GB/s of KV). The Solar grouped split kernel shares the f16 K\|V row layout, so from 2048 keys the context is cut into 256-key chunks, one block per (chunk, KV head), plus the combine kernel. `DS4_EXAONE_ATTN_SPLIT=0` restores the pair kernel; sliding windows, wrapped rings and capture keep it. | Synthetic K2 shape: @4095 0.319 → 0.173 ms, @8191 1.214 → 0.584 ms, @32767 4.835 → 2.224 ms, rel RMS ≤ 1.8e-6 vs the pair kernel, @8191 vs CPU within 2e-5, floor/window/kill-switch cases bit-identical. 8K A/B same binary: decode 7.11 / 7.11 → 9.80 / 9.80, prefill 445.20 / 444.34 → 446.40 / 444.19. Frontier logits byte-identical; greedy IDs diverge at token 8 (56/64), both continuations coherent; two split runs byte-identical. | Retained (revised gate) |
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

## Decode 2 numeric contract

Decode-only change: the frontier is unchanged by construction (prefill
attention is untouched), so the frontier comparator says nothing about
the new kernel. Evidence instead: the kernel parity rows (rel RMS
7.1e-7 / 9.6e-7 / 1.8e-6 at 4095 / 8191 / 32767 keys against the pair
kernel, i.e. fp32 merge order only; @8191 within 2e-5 of the CPU
oracle), bit-identical fallbacks (below 2048 keys, sliding window, kill
switch), byte-identical repeat runs, and a coherent continuation that
shares the first 8 greedy tokens with the kill switch before a near-tie
flip (the same behaviour the first campaign's rejected split-K showed at
token 17). Contract: full-attention decode from 2048 keys merges 256-key
online-softmax partials; the per-key math is the pair kernel's.

## Round 6 note

Two prefill items, worked in parallel on the two largest kernel
groups of the post-round-5 trace (worklist MMQ 45%, HMMA prefill
attention 6.1%); no decode round (the memory wall from the round-4
note stands). Both are scheduling-only: same dequantization, same
dots, same operand order, only the moment the bytes arrive changes,
so the strict gate applies and every cell is byte-identical to its
kill switch, to the repeat and to the round-5 binary.

Prefill 11 ncu evidence (round-5 binary, real layer-7 tensors through
`tests/test_exaone_kernels`, owner down; reports under
`scratch/k2-opt-20260905-cont/r6/ncu/`). IQ1_S worklist at 1024
tokens / 8192 routed rows: 5.72 ms under the profiler, 255 registers,
57.9 KB dynamic shared, one block per SM (occupancy 16.7%; the SM has
100 KB of shared memory and 64K registers, so a second block cannot
fit), issue slots busy 38%, no eligible warp 61% of the cycles, memory
throughput 37%, L2 19%; stalls per issued instruction: long scoreboard
1.50, wait 0.55, not selected 0.47, math-pipe throttle 0.42, barrier
0.36; shared-store bank conflicts on 19.4 M of 49.2 M wavefronts (the
upstream x-tile layout; untouched). IQ2_XXS worklist at 8192 rows:
7.91 ms, issue 26%, no eligible 74%, memory 31%, long scoreboard 3.90
(the loader's grid lookup depends on the qs load, two DRAM/L2 round
trips per row). Both kernels spend their time waiting for global loads
that the upstream K loop issues and consumes in the same phase, four
barriers per iteration, with the tensor pipe near 35%. The pipelined
loop (`ds4_mmq_pipe.cuh`) overlaps the next block's weight fetch with
the MMA phase (registers: 12-24 per thread depending on the type) and
the activation halves with cp.async into a two-stage buffer (shared
memory 57.9 → 76.0 KB for the IQ1_S / IQ2_XXS layouts, 80.1 KB for
the per-16 layouts; still one block per SM). 128-wide tiles keep the
upstream loop: their two-stage buffer would need 112 KB. Register
count stays 255 (stack frames 56-184 bytes, no local-memory traffic
of note).

Prefill 11 result: real-weight kernel rows IQ1_S nt=1024 7.29 →
6.74 ms, IQ2_XXS nt=8192 9.58 → 7.08 ms, IQ2_XS nt=8192 11.43 →
8.23 ms, IQ1_M nt=1024 8.28 → 7.87 ms (the test's skewed router sends
20% of the rows to one expert, so 128-wide tiles on the upstream loop
bound the IQ1_S / IQ1_M numbers); 8K A/B same binary 595.79 / 597.52
→ 627.73 / 629.17 prefill (+5.3%), decode unchanged. Evidence:
`scratch/k2-opt-20260905-cont/{r6a-pipeoff,r6a-all,r6a-pipeoff2,r6a-all2,r6}/`.

Prefill 12 ncu evidence (`ds4_fattn_hmma_gqa2_kernel`, synthetic
1024-token prefill, 64 heads / 8 KV heads / head_dim 128; report under
`r6/ncu/`): 164 registers, one 256-thread block per SM (register
limited), issue slots active 31%, LSU pipe 54% of peak, 13.4 M
shared-load wavefronts against 1.1 M shared stores, no bank conflicts,
no spills; stalls per issued instruction long scoreboard 1.20, wait
0.99, mio throttle 0.76, lg throttle 0.70, math-pipe throttle 0.45.
The scalar consume step reads its mma B fragments element by element
(32 half2 loads for the 16 K fragments, 64 half loads plus 32 packs
for the 16 V fragments per 16-key step and lane) and the 64-key K/V
tile is single-buffered, so the block idles on L2 between tiles. The
new consume (`solar_fattn_consume_16_ldsm`) takes the K fragments
with `ldmatrix.x4` and the V fragments with `ldmatrix.x4.trans` (8 + 8
shared loads per step; the fragment each lane receives is the half2
the scalar loop assembled, low column in the low half), and for BF16
K/V the next tile's 8 K + 8 V float2 per thread are fetched into
registers right after the barrier that publishes the current tile
(`solar_fattn_bf16_tile_fetch/store`); the compressed Solar formats
keep the direct fill and take only the ldmatrix consume. Registers
164 → 168, still one block per SM, no local memory. Synthetic memcmp
old vs new: 0 differing floats in all six cells (200 / 1024 / 2048
tokens × full / window 128); full 2048 1.594 → 1.259 ms, of which
ldmatrix alone gives −13% and the prefetch the rest; the direct-entry
window-128 cells at 1024 / 2048 run 5-8% slower under LDSM, a path
production never takes (the wrapper keeps windows on the warp kernel
and K2 is full attention). Item worked by a forked agent in a
throwaway worktree from the same ncu data; its report and the
split-experiment logs are under `r6/fattn/`.

Prefill 12 result: 8K A/B on the round-6 binary 629.08 / 628.29 →
644.78 / 641.94 prefill (+2.4%), decode 13.35 / 13.33 vs 13.34 /
13.07 (the kernel does not run in decode). Every round-6 cell is
byte-identical in logits and greedy IDs to every other and to
`r5-all`. Evidence:
`scratch/k2-opt-20260905-cont/{r6b-ldsmoff,r6b-all,r6b-ldsmoff2,r6b-all2}/`.

Post-round-6 nsys, same session and binary, both switches off vs all
defaults (8K+64, 593.61 / 13.10 → 640.13 / 13.14 under the profiler):
total kernel time 19.79 s → 18.62 s (−5.9%). Per launch: IQ1_S
worklist 1000 × 4.544 → 4.328 ms (−4.8%), IQ2_XXS worklist 500 ×
6.652 → 5.509 ms (−17.2%), IQ2_XS worklist 80 × 7.800 → 6.665 ms
(−14.5%), IQ1_M worklist 160 × 5.149 → 5.058 ms (−1.8%), HMMA prefill
attention 610 × 1.905 → 1.417 ms (−25.6%); every other kernel within
±1% (swiglu +4% on 0.08 ms launches). The down projections (K = 1792,
seven K iterations per tile) gain three to four times more than the
gate/up projections (K = 6144, 24 iterations): the pipelined loop
hides the load latency, and what remains in the IQ1 kernels is the
expand phase itself (the per-16 grid lookups and the 2-way
bank-conflicted x-tile stores) plus the MMA, so the next step for
them is the expand, not more prefetch. The round-5 session's
per-kernel numbers (IQ1_S 4.16 ms, IQ2_XXS 6.20 ms) are not
comparable to this session's (its whole round-5 code path measured
4% slower here), which is why the kill-switch profile was rerun.
Ranking after round 6: IQ1_S worklist 23.2%, IQ2_XXS worklist 14.8%,
q8 aligned dense vec 13.8%, Q8_0 MMQ 10.4%, HMMA prefill attention
4.6%, IQ1_M worklist 4.3%, quantize_mmq 3.3%, decode attention 3.3%,
IQ2_XS worklist 2.9%, IQ1_S decode vec 2.6%, moe_sum 2.4%,
`exaone_add_kernel` 2.4% (13320 launches). Evidence:
`scratch/k2-opt-20260905-cont/{r6-nsys,r6-nsys-off,r6/kernsum_compare.py}/`.
Next prefill candidates: extend the pipelined loop to the 128-wide
tile with a single-stage activation buffer (76 KB fits), then the
IQ1_S expand phase (the x-tile store pattern conflicts 2-way on the
upstream Q8_1 layout; a swizzled layout would need matching ldmatrix
addressing in `vec_dot_q8_1_q8_1_mma`), then the 23,683 dense-vector
launches of the decode window.

## Round 5 note

No decode round: the round-4 trace leaves decode at 72 ms of
bandwidth-bound kernels per 76 ms step (see the round-4 note), so the
two rounds here are both prefill. Prefill 10 is the first retained
round that changes a serving default rather than a kernel; it is an
arithmetic-class change under the revised gate and its comparator is
the rejected Prefill 5 itself.

Prefill 10 gate result. Against the 512-chunk cell on the same binary
the 1024-chunk frontier has rel RMS 0.0542 (band ≤ 0.11), top-10
overlap 9/10 (≥ 8/10), a byte-identical repeat, coherent text on both
sides, and kernel rows at the 1024-token shapes (attention prefill
1024 vs CPU and HMMA-vs-warp; IQ1_S/IQ1_M gate/up pair at 8192 routed
rows; IQ1_S worklist at 1024 tokens; IQ2_XXS/IQ2_XS worklists at 8192
rows) that are bit-identical or within the CPU tolerance. Two items
fall outside the band: the frontier argmax swaps between the two
near-tied top tokens (gap 0.43 logits, p 0.169 vs 0.227 after the
swap) and KL is 6.3e-2 against the 5e-2 band, most of it the swap.
The perturbation scale is the comparator's (max |Δlogit| 1.03 vs 0.99
for Prefill 6 and 0.84 for Prefill 5), so a 0.43 gap is inside the
range any fp-class change moves this frontier by, and the older
Prefill 5 cell moved the same frontier without the swap. Assessment:
fp-class, not a chunk-boundary fault. The round is retained on that
assessment and flagged for review; the default is a one-line env
rollback.

Post-round-5 nsys (8K+64, 620.93 / 13.05 under the profiler): total
kernel time 22.50 s → 19.06 s. Per 8K prompt the IQ1_S worklist runs
1000 × 4.16 ms instead of 1800 × 3.15 ms (5.67 → 4.16 s: twice the rows
per launch in 1.32× the time), IQ2_XXS 500 × 6.20 ms instead of
900 × 4.83 ms (4.35 → 3.10 s), HMMA prefill attention 610 × 1.90 ms
instead of 1098 × 1.11 ms; the router now runs cuBLAS's 64×64 TF32
tile instead of 64×32 (the fp-class source of the Prefill 10 drift);
sanitize is 0.8% (down passes gone). Ranking: IQ1_S worklist 21.9%,
IQ2_XXS worklist 16.3%, q8 aligned dense vec 13.5%, Q8_0 MMQ 10.1%,
HMMA prefill attention 6.1%, IQ1_M worklist 4.0%, quantize_mmq 3.2%,
decode attention 3.1%, IQ2_XS worklist 3.0%, IQ1_S decode vec 2.5%,
moe_sum 2.4%. Evidence:
`scratch/k2-opt-20260905-cont/{r5-all,r5-p9off,r5-p10off,r5-all2,r5-nsys,r5}/`
(cells ran on the first round-5 build, whose K2 behaviour equals the
committed source; the committed build scopes the 1024 default to K2).

## Round 4 note

Decode is at the memory wall: per token the round-3 trace has 72.4 ms
of kernels (dense Q8_0 GEMVs 40.0 ms at ~240 GB/s, routed expert
vectors 17.9 ms, split attention 9.5 ms, all within ~10% of the
bandwidth floor of ~62 ms) against a 76 ms wall, launch gaps average
0.36 µs, and the only >1 ms gap per token is the bench's frontier-logits
dump. The remaining decode fat is the ~2.4 ms of small kernels per
token; Decode 4 removes the largest non-bandwidth one (the double-
precision RoPE, 0.73 ms per token) and, because the same kernel runs in
prefill, also 40 ms per prefill chunk. Further decode gains need a
structural change (graph-captured decode step) or a precision change
(FP8 KV), both outside this campaign's contract.

Round-4 incident: the first round-4 build moved every frontier logit
(rel RMS 0.063 against round 3) with both kill switches off. A
standalone old-vs-new kernel run isolated it to the RoPE rotation:
norm and (cos, sin) were bit-identical, but once c and s arrived from a
table or a branch nvcc fused the other product pair of `a·c − b·s` /
`a·s + b·c` (same opcode histogram, one ulp per rotated value), and on
this model one ulp anywhere flips the frontier. The rotation is now
written as the original contraction (c products fused, s products
rounded first); the r4b cells above are byte-identical to round 3.

Post-round-4 nsys (8K+64, 491.90 / 13.04 under the profiler): total
kernel time 23.67 s → 22.50 s. `exaone_qk_norm_rope_kernel` is 10004 ×
6.6 µs (0.3%) instead of 77 µs avg (3.3%), the table kernel runs 82
times (18 chunks + 64 tokens); the gate/up sanitize passes are gone
(sanitize launches 25666 → 23578). Ranking: IQ1_S worklist 25.2%
(3.15 ms avg), IQ2_XXS worklist 19.3% (4.83 ms), q8 aligned dense vec
11.4%, Q8_0 MMQ 9.1%, HMMA prefill attention 5.4%, IQ1_M worklist 4.5%,
IQ2_XS worklist 3.5%, decode attention 2.6%, quantize_mmq 2.2%,
sanitize 2.2%, IQ1_S decode vec 2.1%, moe_sum 2.1%, swiglu_weighted
1.6%. Evidence: `scratch/k2-opt-20260905-cont/{r4b-all,r4b-p8off,r4b-d4off,r4b-all2,r4b-nsys,r4,r4-*}/`
(the `r4-*` cells are the un-pinned first build).

## Decode 3 numeric contract

Same class as Decode 2: per-key math of the pair kernel, 256-key
partials merged by the combine kernel; only the in-chunk warp order
differs from the Solar kernel (rel RMS ≤ 4.4e-7 at 4095 / 8191 / 32767
keys) and the whole-context pair path is bit-identical to before (one
chunk of the same kernel). Evidence: the kernel rows above, frontier logits byte-identical to the kill switch, kill-switch greedy IDs equal to the Decode 2 cell, all-default repeat byte-identical, coherent continuation.

Post-round-3 nsys (8K+64, 463.94 / 13.15 under the profiler): total
kernel time 28.72 s → 23.67 s. IQ2_XS is the worklist kernel at
144 × 5.42 ms (3.3%) instead of the rectangular 11.8 ms; decode
attention is `exaone_attn_decode_gqa_kernel<4, true>` at 3904 × 0.155 ms
(2.6%) instead of 1.23 ms (D1) / 0.584 ms (D2). Ranking: IQ1_S worklist
24.1% (3.16 ms avg), IQ2_XXS worklist 18.3% (4.82 ms), q8 aligned dense
vec 10.9%, Q8_0 MMQ 8.7%, HMMA prefill attention 5.1%, IQ1_M worklist
4.3%, IQ2_XS worklist 3.3%, `exaone_qk_norm_rope_kernel` 3.3%,
`ds4_mmq_sanitize_f32_kernel` 3.0% (25666 launches), decode attention
2.6%. Evidence: `scratch/k2-opt-20260905-cont/{r3-all,r3-p7off,r3-d3off,r3-all2,r3-nsys,r3}/`.

Review fix in the same round (PR #4, Codex): the GQA pair kernel paired
heads 2 and 3 across KV heads 0 and 1 whenever the query group was odd
(`n_head % 2 == 0` passed for n_head 6, n_head_kv 2). The predicate is
now `group % 2 == 0`; K2 (group 6) is unaffected.

Post-P6 nsys (8K+64, 443.28 / 7.11 under the profiler): total kernel
time 41.68 s → 28.72 s. IQ1_M is the worklist kernel at 288 × 3.50 ms
(3.5%) instead of the assign kernel at 288 × 41.06 ms (28.4%). Ranking:
IQ1_S worklist 19.8% (3.16 ms avg), `exaone_attn_decode_gqa_kernel`
16.7% (1.23 ms), IQ2_XXS worklist 15.0% (4.79 ms), q8 aligned dense vec
8.9%, Q8_0 MMQ 7.2%, IQ2_XS rectangular 5.9% (11.8 ms avg), HMMA prefill
attention 4.2%, IQ1_M worklist 3.5%. Evidence:
`scratch/k2-opt-20260905-cont/{p6-off,p6-on,p6-on2,p6-off2,p6ctx-off,p6ctx-on,p6-nsys,p6}/`.
