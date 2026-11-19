# Hy4-flash MXQ on DGX Spark

The mixed-quant release at Hub revision `94cdace56aaa0afc463311e31a2bbc3e613dcce2` ran on one GB10 with CUDA 13.5, driver 620.70.06 and the 300–2,200 MHz clock policy. Busy SM clock was 2,190 MHz. The host-memory guard kept a 2 GiB reserve and trip floor. Each comparison used fresh model processes. Prefill rates below cover newly appended tokens. Generation has its own timer.

## Tiled MLA prefill

At the 4K frontier the scalar SWA and sparse MLA kernels took 46.6% of the 2K append. A model-free replay used 512 query rows at position 3,584, the release head counts, learned sinks, rotary layout and sparse gather width. On 64 heads, scalar and tiled medians were 12.59 and 1.10 ms. On 32 heads they were 31.27 and 3.30 ms. Nsight Compute profiled the isolated kernels. The tiled kernel shares the BF16 matrix instructions and online softmax with Ling-3.1 and maps each physical key through Hy4's window or selected group.

With 512-row chunks and 32 greedy output tokens, 4K incremental prefill improved from 309.68 to 551.58 tok/s. The four measured frontiers spanned 303.70–367.88 and 534.46–569.49 tok/s respectively. Plain decode ranged from 13.87–14.18 and 13.98–14.23 tok/s. Each file is a single sweep. [Scalar CSV](benchmarks/hy4-flash-mxq-2026-11-19/scalar-512.csv) · [Tiled CSV](benchmarks/hy4-flash-mxq-2026-11-19/flash-512.csv).

The fast path rounds absorbed queries and softmax weights to BF16 for MMA, accumulates in FP32, and shares the scalar cache. On two deterministic random head geometries its relative RMS output difference was 0.00248 (32 heads) and 0.00247 (64 heads). The reference remains available with `DS4_HY4_FLASH=0`.

## IndexPool on long prefills

The scalar index scorer occupied 12.1% of the cold 32K prefill and 20.0% of the next 2,048-token append. The tiled kernel evaluates the 16 index heads with BF16 matrix instructions and FP32 accumulation, preserves Hy4's group visibility, and merges each 1,024-group tile into its best 511. Decode and small appends use the scalar scorer. On an isolated 2,048-row append at 32K the two implementations selected the same 1,046,527 out of 1,046,528 group memberships. Excluding the first timing, their per-layer medians were 58.42 and 5.50 ms. Nsight Compute captured only the isolated kernel.

Fresh processes with 2,048-row chunks recorded 741.18 and 843.98 tok/s on the cold 32K prompt, then 586.11 and 725.73 tok/s on the next append. Eight plain output tokens ran at 12.89/13.93 tok/s with the scalar scorer and 12.88/13.79 with the tiled one. These are one pair of runs. [Scalar index](benchmarks/hy4-flash-mxq-2026-11-19/index-off-short.csv) · [Tiled index](benchmarks/hy4-flash-mxq-2026-11-19/index-short.csv).

At 512-row chunks, the tiled scorer changed mean NLL by +0.0180, +0.0236 and −0.0336 across 32 teacher-forced positions each at 8K, 32K and 64K. Top-1 agreement with the scalar scorer was 27/32, 32/32 and 31/32. The comparison used the same FP8 cache, mixed trunk and novel. Set `DS4_HY4_INDEX_MMA=0` to use the scalar score accumulation for all widths.

The first combined run used the Q8_0 DFly2 at depth two, 2,048-row chunks and 128 greedy output tokens per frontier, with 32 incremental frontiers from 2K to 64K. The arithmetic means were 730.16 prefill and 15.75 decode tok/s. Prefill ranged from 717.09 to 744.22. Decode varied from 12.36 to 21.30 with draft acceptance. [Individual frontiers](benchmarks/hy4-flash-mxq-2026-11-19/sweep-spec.csv).

## Batch the DFly2 target output head

A 4K DFly2 trace attributed 13.2% of decode wall time to one-row BF16 projections and 11.3% to small three-row cuBLAS kernels. Three calls to the 120,832 × 3,072 BF16 head took a median 12.72 ms in isolation. One three-row cuBLAS call took 3.15 ms. The target now computes its head over the verify block and reads all logits once. It borrows the existing expert buffer after the final layer and preserves each row's DFly2 seed. `DS4_HY4_BATCH_HEAD=0` selects the one-row path.

Four fresh 2K–8K runs used the same novel, 2,048-row chunks, 64 greedy outputs and Q8_0 draft depth two. The per-run means were 17.16–17.38 tok/s with one-row heads and 17.97–18.03 with the batched head. All four frontiers improved in both comparisons. Prefill means spanned 717.05–719.60 and 717.16–721.96 tok/s. One-row heads took 35/28/30/32 verify calls from 2K to 8K. Batched heads took 35/28/31/33 in both runs. The target dot uses the same BF16 operands and FP32 output with a different reduction order. [One-row A](benchmarks/hy4-flash-mxq-2026-11-19/head-0-a.csv) · [One-row B](benchmarks/hy4-flash-mxq-2026-11-19/head-0-b.csv) · [Batched A](benchmarks/hy4-flash-mxq-2026-11-19/head-1-a.csv) · [Batched B](benchmarks/hy4-flash-mxq-2026-11-19/head-1-b.csv).

We also tested direct narrow BF16 dots throughout the model. They shortened a verify step from 118–120 to 110–112 ms, but needed more verify calls and lost generated tokens/s at three of four frontiers. The retained change batches only the output head. Splitting the three-row routed gate into individual MMVQs improved the isolated gate slightly and failed to produce a consistent generation gain.

## Overlap Engram reads with prefill

At 4K, the host spent 434 ms before the first GPU kernel in a 2,864 ms append. Four isolated 2,048-token batches of actual Engram records took 373–390 ms to read. With a fresh GPU frontier, the host now reads the four tables at layers 1, 11, 21 and 31 so the SSD works while earlier layers execute. An unfinished preceding chunk can cover a bulk read of all four tables. Short inputs keep the bulk path. The same aligned page buffer and GPU staging area serve both schedules. Set `DS4_HY4_ENGRAM_LAYER_READS=0` to force bulk reads.

Seven fresh processes used 2,048-token chunks, a cold 32K prompt, a 2,048-token append and eight plain output tokens at each frontier. Reading each table at its layer raised append throughput, but slowed cold prefill in both original pairs. The final schedule recovered the cold rate. The first two rows below give the full range across their runs.

| Schedule | Runs | Cold 32K tok/s | 2K append at 32K tok/s |
|---|---:|---:|---:|
| Bulk | 3 | 828.73–831.33 | 717.20–726.25 |
| Per-layer throughout | 2 | 824.94–827.52 | 813.49–813.72 |
| Final schedule | 2 | 832.49–833.45 | 816.80–817.00 |

On the final binary, the 2K append improved 13.9% against its bulk control. The complete frontier logits and eight greedy IDs matched at both lengths across all seven runs. The eight-token append decode rates spanned 13.61–13.95 tok/s on the final schedule and 13.75–13.91 on bulk. [Raw comparisons with binary identifiers](benchmarks/hy4-flash-mxq-2026-11-19/engram-reads.csv).

## Routed experts on the prefill path

At the 32K frontier the routed MMQ took 37.6 % of a 2K append (912 ms for 41 layers of gate, up and down). Streaming each layer's 2.36 GB of expert weights once costs ~420 ms at 230 GB/s. Each row below is two fresh-process pairs at the 32K frontier with 2,048-row chunks and the previous rows applied.

| Change | Class | Cold 32K tok/s | 2K append tok/s | Switch |
|---|---|---:|---:|---|
| Q4_K/Q5_K experts on the 64-wide pipelined worklist tile | bit-identical | 834.7 -> 877.1 | 819.4 -> 853.4 | `DS4_HY4_PIPE64=0` |
| Clamped SwiGLU straight into the Q5_K down's Q8 operand; `moe_sum` zeroes non-finite rows, so the down has no scrub | bit-identical | 878.4 -> 917.5 | 858.3 -> 897.3 | `DS4_HY4_SWIGLU_Q8=0` |
| FP16 producer-warp GEMM over 128 x 64 tiles of one expert (`cuda/hy4_moe.cuh`) | value change | 921.2 -> 976.4 | 896.6 -> 948.9 | `DS4_HY4_FP16_MOE=0` |

Decode stayed at 13.4-13.9 tok/s in every pair; decode and verify widths keep MMQ. The bit-identical rows reproduced both frontier logits and the 16 greedy tokens byte for byte.

On layer 10's weights with skewed routes (max 564, median 39 rows per expert), the FP16 GEMM took gate and up 14.2 -> 11.0 ms and the down 9.6 -> 7.5 ms; on layer 3's Q4_K experts gate and up took 12.8 -> 9.2 ms. Against an F64 reference on sampled rows its relative error is 8e-4 for gate and down, where MMQ's Q8_1 activations give 1.2e-2 and 2.1e-2. Each tensor's scales are checked once against the folded FP16 range, and a tensor outside it keeps MMQ. Over the 1,664 teacher-forced positions below, NLL went from 3.00223 (MMQ) to 2.97472 against the BF16 teacher's 2.97612, and top-1 agreement with the teacher from 89.12 % to 90.38 %. Paired with MMQ the change is -0.0275 NLL (95 % document-bootstrap interval -0.0916 to +0.0189).

The attention projections stay on the aligned Q8_0 D2R tier: with `DS4_MMQ_DENSE_D2R=0` both measured frontiers lost 2.3-2.9 %.

## BF16 operands and the default chunk

Every multi-row norm converted its F32 rows to BF16 once per consuming GEMM (q_a, kv_a, the indexer key, routed down and the shared gate and up), and the tiled MLA kernel wrote F32 rows that the v_b projection converted again (1.68 ms per 64-head layer at 2,048 rows). Each norm now keeps one exact BF16 copy, and prefill attention stores its rounded BF16 rows directly into the projection's operand. Two fresh processes at the 32K frontier measured 976.4 -> 1,030.0 tok/s cold and 948.9 -> 1,000.9 on the 2K append, with byte-identical frontier logits and greedy tokens.

The prefill chunk defaults to 2,048 rows in the host and the native graph (it was 512; every measurement here sets 2,048). When a long context leaves no room for the workspace, the planner halves it down to 512. Before loading, one 1M history quoted 117.18 of 117.83 GiB with 2,048 rows, and 117.21 of 117.80 with the Q8_0 DFly2 at 1,024 rows. After loading, free memory was 1.4 GiB (plain) and 2.2 GiB (DFly2) below the priced weights, so the opened plan refused that history at the default 4 GB floor even at 512 rows (DFly2: 116.75 GiB needed, 115.62 available). A 2 GB floor admitted it at 1,024 rows, but after the history was allocated 2.96 GiB stayed free and memory pressure (PSI full ~16 %) stalled a 3.4K-token request. Without the aligned Q8 projection copies (`DS4_CUDA_NO_Q8_ALIGNED=1`, 4.13 GiB) the 1M plan opens at the default floor; the release launcher serves that way.

## Fusions and halved traffic

Each row is two fresh processes at the 32K frontier against the row's own control (the switch, or the previous binary where a row has none). Only the first row changes values.

| Change | Class | Cold 32K tok/s | 2K append tok/s | Switch |
|---|---|---:|---:|---|
| Residual, iHC mix, combine and RMSNorm in one CTA per row, from 16 rows | value change | 1,031.7 -> 1,074.6 | 1,001.5 -> 1,036.0 | `DS4_HY4_HC_FUSED=0` |
| Output gate writes the attn_output Q8 operand | bit-identical | 1,074.6 -> 1,081.4 | 1,036.0 -> 1,046.1 | `DS4_CUDA_NO_NORM_Q8EMIT=1` |
| Gate, up and SwiGLU in one tile; the down operand is written by its epilogue | bit-identical | 1,077.3 -> 1,156.1 | 1,038.8 -> 1,119.3 | `DS4_HY4_GLU_FUSED=0` |
| Routed down rows stored as BF16, the expert sum's read boundary | bit-identical | 1,150.5 -> 1,177.3 | 1,109.0 -> 1,136.8 | `DS4_HY4_DOWN_BF16=0` |
| Engram reads in one full-queue round; next-tile L2 prefetch in the expert GEMM | bit-identical | 1,177.3 -> 1,195.3 | 1,136.8 -> 1,158.7 | |
| k_b writes BF16 q_abs for the tiled MLA kernel | bit-identical | 1,195.3 -> 1,217.2 | 1,158.7 -> 1,179.0 | |

The fused iHC step computes the mix with exact F32 dots instead of prefill's TF32 GEMM. Over the 1,664 teacher-forced positions it moved NLL by +0.0105 against the FP16-expert run (95 % document bootstrap -0.0124 to +0.0333) with 92.7 % top-1 agreement, inside the tiled-attention switch's +0.0526 and 91.1 %. On a decode row the one-CTA kernel lost 3 %, so decode keeps the split launches.

The fused tile alternates 16-row blocks of gate and up over the same 64 outputs, so a lane holds both values of an output. Model-free on real weights, gate + up + SwiGLU took 12.3 -> 9.0 ms on a Q5_K layer and 10.8 -> 7.85 ms on a Q4_K layer. Prefetching the next work item's contiguous tile one piece per stage, instead of each row's super-blocks, took those to 8.35 and 7.33 ms and the down 5.27 -> 5.19 ms.

The first Engram layer stalls the GPU after layer 0. Each 2,048-token table is ~32K random 98-byte records. Waiting for each 512-read batch took 112 ms per table on the production rotation, one full-queue round 100 ms, and the stall fell 55.8 -> 48.9 ms. The SSD (Phison E27T, DRAM-less) bounds the rest: reading the same table again right away takes 45 ms, but its mapping cache does not hold four 16.5 GB tables, and four rings on four threads took 104 ms.

The 2K-64K DFly2 sweep after the BF16 down rows averaged 1,140.38 prefill (1,085.91-1,164.64) and 15.53 decode tok/s. Decode fell from 16.64 only through acceptance: 1.743 emitted tokens per verify against 1.844, with the verify step at 113.5 against 113.2 ms. The value changes above moved the greedy continuations. [Raw sweep](benchmarks/hy4-flash-mxq-2026-11-19/sweep-ef797b9a.csv).

The 32K append is now ~1.77 s per 2,048 rows: routed experts 34 %, MLA attention 15 %, the Q8_0 attention projections 15 % (q_b and attn_gate on D2R, attn_output on MMQ), the fused iHC step 7 % and the head projections 7 %. Every consumer of q rounds it to BF16 (rope, the k_b operand, the tiled kernel's rope fragments), so q_b's D2R epilogue could write BF16 and retire the 1.0 ms F32-to-BF16 pass of each 64-head layer.

## Decode

A plain token reads 13.45 GB of weights: 10.43 GB of dense tensors and the head (attn_output and attn_gate in Q8_0 alone are 3.90 GB) and 3.02 GB of its 16 routed experts. At 240 GB/s that bounds plain decode at 17.8 tok/s. At the 32K frontier a token took 72.8 ms: the Q8_0 and BF16 projections ran near that rate, but the scalar decode attention took 8.5 ms for ~33 MB of cache, one CTA per head walking keys one warp at a time.

Decode and verify rows now take the prefill MLA tile, with each row's 512 window or 2,048 selected keys split over the SMs and a combine that writes the v_b operand. The k_b projection writes their q_abs as BF16, as it does for prefill. Draft rows moved to the same tile in the verify-step round below.

| Mode at 32K, 64 tokens, two fresh processes | Scalar attention | Tiled, split keys |
|---|---:|---:|
| Plain decode | 13.80, 13.83 tok/s | 15.34, 15.42 tok/s |
| DFly2 depth 2 | 16.14, 16.21 tok/s | 17.24, 17.20 tok/s |

The first DFly2 pair emitted 1.829 tokens per verify in both runs, at 113.3 and 106.1 ms per verify. Over the 1,664 teacher-forced positions (1,560 of them decode steps) NLL moved -0.0311 against the scalar build (95 % document bootstrap -0.0702 to +0.0002) with 92.1 % top-1 agreement, to 2.95417 against the teacher's 2.97612. `DS4_HY4_FLASH_DECODE=0` restores the scalar kernel for target rows.

A DFly2 verify step at 32K left the GPU idle for ~11 % of its wall time. About a dozen `cudaMallocAsync` calls per step took 0.3-0.7 ms in driver ioctls, because the default pool releases its pages at every synchronization. Keeping 64 MB of freed pool pages measured +0.6 % plain and +0.2 % DFly2 over two pairs each, so that global setting was not adopted. Plain decode now sits at ~87 % of its 17.8 tok/s weight-stream bound. A verify step reads the dense weights once and the three rows' expert union (its routed MMQ moved ~7.3 GB, 2.4 tokens' worth), ~17.8 GB before the drafter, so at 1.83 tokens per verify even a bandwidth-bound DFly2 step would stay under ~25 tok/s.

### Verify and draft steps

At 32K and draft depth three a DFly2 step took ~120 ms. Traces counted the four rows' routed MMQ at 48.2 ms, the aligned Q8_0 projections at 19.5 ms (~227 GB/s), the BF16 projections at 17.3 ms on cuBLAS (15.8 ms after the change below) and the drafter at 17.6 ms, 9.3 ms of it in three passes over the 742 MB BF16 output head. Counting kernels alone, the step looked 29 % idle; counting graph executions, copies and memsets, it was 11 % idle.

| Change | Class | ms per verify, 32K, depth 3, 64 tokens, two pairs |
|---|---|---:|
| BF16 projections of 2–4 rows on a warp-per-row kernel that loads each weight once for every row, in the one-row order, instead of cuBLAS (sm80 wmma, 16×16 tiles, one warp per CTA) | value change: a verify row equals its decode row | below |
| Engram tables read before their layers at decode, as at prefill | bit-identical | 122.05 -> 119.90 |
| Draft rows on the tiled MLA kernel: the committed row's selection plus their own draft keys | draft-only | 121.70 -> 119.90 |
| Draft tokens ranked through a Q8_0 copy of the output head (+0.37 GiB); the verify keeps BF16 | draft-only | 119.15 -> 114.75 |
| Routed experts of 2–4 verify rows on the prefill's FP16 tile (producer warps dequantize each weight once) instead of MMQ | value change | below |

With four rows and eight rotating weight copies, a model-free harness streamed each BF16 projection shape at 233–258 GB/s against cuBLAS' 217–235, 14.51 -> 13.57 ms per verify step. At 8K and depth two, with equal text, a verify took 107.4 -> 106.7 ms. At 32K the kernel moved the 37th greedy token, and the new continuation accepted 3.20 instead of 2.37 tokens per verify. The other rows therefore compare switches that keep the target tokens: `DS4_HY4_ENGRAM_LAYER_READS=0`, `DS4_HY4_FLASH_DRAFT=0` and `DS4_HY4_DRAFT_HEAD_Q8=0`; `DS4_HY4_BF16_ROWS=0` restores cuBLAS. The staged reads also took plain decode from 15.39 to 15.52 tok/s. [A/B receipts](benchmarks/hy4-flash-mxq-2026-11-19/decode-round2.csv).

MMQ's per-k32 scale work dominates when an expert sees one to three rows. At depth two and 32K, traced with graphs off, the three rows' routed experts took 38.9 ms per step on MMQ and 30.5 ms on the FP16 tile, and a step 102.0 -> 93.3 ms. Two bench pairs on a probe build with a temporary floor switch (`DS4_HY4_FP16_MOE_MIN`, not released) measured 102.25 -> 93.85 ms per verify on different text. Through the teacher-forced DFly2 trials, which keep every token, the whole gate ran 231.4 -> 225.4 s. One decode row ran even (15.50 vs 15.55 tok/s) and keeps the vec kernels; `DS4_HY4_FP16_MOE=0` restores MMQ.

Over the 1,664 teacher-forced positions the DFly2 verify rows of the final build averaged **2.98602 NLL** against the teacher's 2.97612 (+0.00990, 95 % document bootstrap −0.02767 to +0.04847), with 89.12 % top-1 agreement. Against the release build's verify rows the change was +0.02347 (−0.01467 to +0.05892) with 91.83 % agreement, inside the band of the accepted tiled-attention switches (−0.0311 and +0.0526 at 91–92 %). The FP16 verify experts then moved it by +0.00033 (−0.03320 to +0.03679) with 92.19 % agreement, to 2.98634.

Depth two stays the release setting. Over nine frontiers from 2K to 64K with 128 tokens each, depths two, three and four decoded 19.54, 18.12 and 17.91 tok/s at 1.85, 1.92 and 2.06 tokens per verify; one 32K continuation that accepted 3.20 tokens at depth three was the exception. With the FP16 verify experts, depths two and three decoded 19.24 and 19.07 tok/s at 1.76 and 1.90 tokens per verify; their steps were ~3 and ~6 ms shorter, and the nine continuations changed with the arithmetic. [Depth screen](benchmarks/hy4-flash-mxq-2026-11-19/draft-depth-round2.csv). Over the 2K–64K sweep the round took the mean verify step from 106.7 to 92.5 ms (97.9 before the FP16 verify experts). Decode rose from 17.40 to 19.54 tok/s before them and measured 19.49 after: their continuations accepted 1.78 instead of 1.86 tokens per verify. On fixed teacher text the FP16 experts kept acceptance, 1.9179 against 1.9218 accepted tokens per trial (paired −0.0039, 95 % document bootstrap −0.0196 to +0.0104), so the sweep's acceptance change belongs to its new continuations.

## MQ-Q5-Q8: dense projections at Q8_0

A plain token read 13.45 GB, 4.85 GB of it in BF16 dense projections, the BF16 head and F32 routers. [`MQ-Q5-Q8-SSD-ENGRAM-FP8`](https://huggingface.co/Baekpica/Hy4-flash-Mixed-Quant-GGUF/tree/9a32b58fc7cc32d3e0488ddb916efd05cdf33cd3/MQ-Q5-Q8-SSD-ENGRAM-FP8) derives from the release: those projections and the head become Q8_0 (release quantizer), and the routers, BF16 values widened to F32, return to BF16 exactly. The trunk shrinks by 2.12 GiB and a token reads 11.17 GB, bounding plain decode at 21.5 tok/s.

| Tensors | Count | MQ-Q5 GB | MQ-Q5-Q8 GB |
|---|---:|---:|---:|
| Output head | 1 | 0.742 | 0.394 |
| `attn_k_b`, `attn_v_b` | 84 | 1.057 | 0.562 |
| Shared expert gate, up, down | 123 | 1.161 | 0.618 |
| `ffn_routed_down`, `ffn_routed_up` | 82 | 0.774 | 0.412 |
| `attn_q_a`, `attn_kv_a_mqa` | 84 | 0.413 | 0.219 |
| Leading dense FFN, `engram_kv` | 7 | 0.444 | 0.235 |
| Routers, F32 → BF16 | 41 | 0.258 | 0.129 |

- The BF16 router keeps the F32 router's arithmetic: one to four rows in the F32 GEMV's order, prefill rows through the same SGEMM on widened weights. Routing is bit-identical (`test_hy4_cuda`).
- Q8_0 `k_b`/`v_b`: decode and verify rows stage 64 weight rows per CTA; prefill widens them for the batched GEMM.
- Drafts read the Q8_0 head directly; the 0.37 GiB copy is not built.
- The aligned-copy predicate admits 254 of the new matrices: 4.13 → 5.76 GiB, priced by the plan.
- MQ-Q5 is unchanged: both teacher-forced gates at every position, and three sweep frontiers' logits and DFly2 tokens, match the release build exactly.

New Q8_0 matrices below 2,048 outputs (`q_a`, `kv_a`, shared gate/up, `routed_down`) missed the D2R kernel and ran MMQ, ~1.6x the cuBLAS time with an activation quantize per GEMM. Prefill rows of those shapes now widen the codes to BF16 per call and keep MQ-Q5's cuBLAS GEMMs; D2R shapes stay on D2R. `DS4_HY4_Q8_WIDE=0` restores MMQ, `=all` widens D2R shapes too.

| 32K, 16 plain tokens, fresh processes | Cold 32K prefill | 2K append | Plain decode |
|---|---:|---:|---:|
| MQ-Q5 | 1,224.44 / 1,223.33 | 1,176.10 / 1,186.05 | 14.87 / 14.97 |
| MQ-Q5-Q8, MMQ | 1,181.23 / 1,187.65 | 1,144.23 / 1,153.89 | 16.77 / 16.97 |
| MQ-Q5-Q8, every shape widened | 1,201.79 / 1,203.67 | 1,166.14 / 1,166.23 | 17.04 / 16.96 |
| MQ-Q5-Q8, released | 1,212.27 / 1,210.71 | 1,173.95 / 1,169.28 | 17.05 / 16.91 |

An earlier widening build measured 1,458 tok/s cold and NLL 9.98: its weight and input scratch aliased (`cuda_tmp_alloc` returns one buffer to every caller). [A/B receipts](benchmarks/hy4-flash-mxq-2026-11-19/q8-dense.csv).

| 1,664 teacher-forced positions | NLL | Against MQ-Q5 (95 % document bootstrap) | Top-1 with MQ-Q5 |
|---|---:|---:|---:|
| Plain | 2.96341 | +0.00924 (−0.03446 to +0.05450) | 90.69 % |
| DFly2 verify rows, depth 2 | 2.96763 | −0.01872 (−0.06005 to +0.02266) | 91.05 % |

On fixed text the drafts kept their acceptance: 1.9159 against 1.9179 accepted tokens per depth-two trial. Around the 8K, 32K and 64K frontiers of the novel, 32 consecutive targets each moved NLL by −0.057, +0.046 and +0.002 with 27, 31 and 31 top-1 agreements, inside the FP8/BF16 cache band below. [Documents](benchmarks/hy4-flash-mxq-2026-11-19/q8-teacher-documents.csv) · [Long frontiers](benchmarks/hy4-flash-mxq-2026-11-19/q8-long-frontiers.csv).

Over the 2K–64K sweep at depth two the verify step fell from 92.5 to 88.2 ms at equal acceptance (1.775 against 1.778 tokens), and decode rose from 19.49 to 20.50 tok/s; a repeat measured 20.66. Prefill measured 1,171.74 and 1,174.32 against 1,187.23. [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_q8_f54892fd.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_q8_f54892fd.meta.txt). With the `k_b`/`v_b` staging below, the same 32 continuations took 86.2 ms per verify: **1,170.10 / 20.97 tok/s**. [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_q8_cafaccab.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_q8_cafaccab.meta.txt).

### What the saved bytes buy

- **Depth stays two.** On fixed text a third draft added 0.229 accepted tokens per trial (+12.0 %, 95 % +0.148 to +0.314) while the step grew from 88.2 to 99.5 ms (+12.8 %). The depth-three sweep decoded 20.59 tok/s; its continuations repeated more (2K: 57 % repeated 8-grams, 3.20 tokens per verify).
- **Aligned copies only speed prefill.** At 32K they took cold prefill from 1,158.13 / 1,163.53 to 1,208.80 / 1,176.69 tok/s; plain decode measured 17.61 / 17.30 tok/s with them and 17.84 / 17.80 without, DFly2 steps 86.4 / 87.4 and 87.8 / 87.4 ms.
- **Banks.** At the 4 GB floor without aligned copies the plan fits three 256K banks (two for MQ-Q5) and eight 128K banks (six); with them, three 128K banks (two). One 1M history leaves 2.67 GB more than MQ-Q5.
- **`k_b`/`v_b` at verify widths.** The first Q8_0 kernel staged 64 rows with 4-byte loads and reloaded each input block per row: a depth-two verify step streamed their 0.56 GB in 4.81 ms, about as long as cuBLAS took for the BF16 pair (4.6 ms). 16-byte copies and one input load per block for a warp's eight rows took 3.31 ms (170 GB/s), bit-identical; verify steps at 32K went from 87.1 to 85.7 ms, plain tokens from 56.6 to 55.9 ms.
- **Next decode levers.** With graphs off, a depth-two verify step at 32K took 84.4 ms against MQ-Q5's 93.3: routed experts 29.7 ms, aligned Q8_0 projections 24.4 ms (~200 GB/s), and raw MMVQ 6.4 ms for `kv_a` (under the repack's 2 MiB floor) and the 1,536-input `routed_up` and shared down (the aligned kernel takes 1,024-multiple inputs).
- **Open levers.** Decode reads the aligned copies yet ran even without them, and prefill widens the raw codes of the narrow shapes, so their 0.70 GiB of copies buy nothing measured. A GEMM that dequantizes Q8_0 in producer warps would remove the widening traffic behind the 1.3 % prefill gap.

## Teacher-forced quality

Each of 26 held-out documents contributed 64 next-token predictions. Lengths ranged from 512 to 4,096. Intervals resample whole documents with replacement (3,000 draws). Positive paired NLL means the model assigned less probability to the reference continuation.

| Runtime | Mean NLL | BF16 teacher NLL | Top-1 agreement with teacher |
|---|---:|---:|---:|
| Scalar, 512-row chunks | 2.96789 | 2.97612 | 90.08% |
| Tiled, 2,048-row chunks | 3.01298 | 2.97612 | 89.24% |

At 2,048-row chunks, tiled minus scalar averaged +0.05261 NLL, with a 95% document-bootstrap interval of −0.00063 to +0.11240 and 91.05% top-1 agreement. Changing the scalar chunk from 512 to 2,048 gave −0.00752 and 98.68% agreement. The tiled arithmetic changes more positions than the chunk reference. The largest increase on a single tool-use document was +0.601 NLL across its 64 measured positions.

A separate FP8/BF16 latent-KV comparison used the same mixed weights, 512-row tiled prefill, and 32 consecutive teacher-forced targets in each region of the public-domain benchmark novel.

| Frontier | FP8 KV NLL | BF16 KV NLL | Top-1 agreement |
|---:|---:|---:|---:|
| 8,192 | 2.1362 | 1.9128 | 25/32 |
| 32,768 | 1.2636 | 1.2295 | 30/32 |
| 65,536 | 1.9755 | 1.9960 | 31/32 |

These three regions represent one document. The BF16 reference changes the latent KV precision on the same mixed trunk.


## Target width and end-to-end quality

With tiled attention, tiled IndexPool and the batched output head, the plain target averaged **3.00223 NLL** against the BF16 teacher's **2.97612**, with 89.12% top-1 agreement. The paired difference was +0.02611 (95% document-bootstrap interval −0.00587 to +0.06053).

The depth-two Q8_0 DFly2 trial kept the supplied reference token. Its verified target rows averaged **3.02228 NLL** across the same positions, with 88.88% top-1 agreement with the teacher. The paired difference from the teacher was +0.04616 (+0.00502 to +0.09124). Against the plain target, it was +0.02005 (−0.01280 to +0.05298), with 92.19% top-1 agreement. Of the 1,664 positions, 1,534 used a three-row trial, 26 used a two-row trial at the context boundary and 104 used prompt synchronization. Each of the 26 documents contributed 64 targets. [Document aggregates](benchmarks/hy4-flash-mxq-2026-11-19/teacher-documents.csv).

## Released configuration

The released binary's 2K–64K sweep averaged **1,187.23 prefill and 19.49 decode tok/s** (89.7 % and 67.4 % of the Qwen3.8 reference). It used 2,048-row chunks, Q8_0 DFly2 depth two and 128 generated tokens at each of 32 frontiers, with 1.78 tokens per verification. The 64K frontier measured 1,162.73 / 19.83 tok/s. [Curve and Qwen comparison](performance.md#hy4-flash-mxq) · [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_322c6ac1.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_322c6ac1.meta.txt). Before the FP16 verify experts the sweep averaged 1,181.66 / 19.54 tok/s. [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_e1f1f736.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_e1f1f736.meta.txt). Before the verify and draft step round the same sweep averaged 1,186.46 / 17.40 tok/s (1.83 tokens per verification). [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_022f4be9.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_022f4be9.meta.txt).

The sweep before this round's prefill and decode work averaged 816.45 / 16.64 tok/s. [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_af33f376.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_af33f376.meta.txt).

The shorter draft-depth comparison used four 2K–8K frontiers, 64 generated tokens each and a fresh process per depth. It records the arithmetic mean of per-frontier rates and the total emitted tokens divided by verification calls. Text and acceptance vary with width. [Per-frontier rates and acceptance](benchmarks/hy4-flash-mxq-2026-11-19/draft-depth.csv).

| Draft proposals | Decode tok/s | Tokens per verification |
|---:|---:|---:|
| 1 | 16.72 | 1.62 |
| 2 | 17.99 | 2.02 |
| 3 | 17.64 | 2.17 |
| 5 | 14.28 | 2.10 |
