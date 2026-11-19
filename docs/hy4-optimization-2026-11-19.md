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

The prefill chunk defaults to 2,048 rows in the host and the native graph (it was 512; every measurement here sets 2,048). When a long context leaves no room for the workspace, the planner halves it down to 512. On GB10 the 1M plans keep 2,048 rows with the aligned Q8 copies: the quote is 125.8 GB of 127.1 GB available, or 126.8 GB with the Q8_0 DFly2.

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

Decode and verify rows now take the prefill MLA tile, with each row's 512 window or 2,048 selected keys split over the SMs and a combine that writes the v_b operand. The k_b projection writes their q_abs as BF16, as it does for prefill. Draft rows keep the scalar kernel.

| Mode at 32K, 64 tokens, two fresh processes | Scalar attention | Tiled, split keys |
|---|---:|---:|
| Plain decode | 13.80, 13.83 tok/s | 15.34, 15.42 tok/s |
| DFly2 depth 2 | 16.14, 16.21 tok/s | 17.24, 17.20 tok/s |

The first DFly2 pair emitted 1.829 tokens per verify in both runs, at 113.3 and 106.1 ms per verify. Over the 1,664 teacher-forced positions (1,560 of them decode steps) NLL moved -0.0311 against the scalar build (95 % document bootstrap -0.0702 to +0.0002) with 92.1 % top-1 agreement, to 2.95417 against the teacher's 2.97612. `DS4_HY4_FLASH_DECODE=0` restores the scalar kernel for target rows.

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

One 2K–64K sweep after scheduling the SSD reads and batching the target head averaged 816.45 prefill and 16.64 decode tok/s. It used 2,048-row chunks, Q8_0 DFly2 depth two and 128 generated tokens at each of 32 frontiers. The 64K frontier measured 804.21 / 14.68 tok/s. [Curve and Qwen comparison](performance.md#hy4-flash-mxq) · [Raw sweep](../speed-bench/gb10_spark_hy4_mxq_af33f376.csv) · [Binary and artifact](../speed-bench/gb10_spark_hy4_mxq_af33f376.meta.txt).

The shorter draft-depth comparison used four 2K–8K frontiers, 64 generated tokens each and a fresh process per depth. It records the arithmetic mean of per-frontier rates and the total emitted tokens divided by verification calls. Text and acceptance vary with width. [Per-frontier rates and acceptance](benchmarks/hy4-flash-mxq-2026-11-19/draft-depth.csv).

| Draft proposals | Decode tok/s | Tokens per verification |
|---:|---:|---:|
| 1 | 16.72 | 1.62 |
| 2 | 17.99 | 2.02 |
| 3 | 17.64 | 2.17 |
| 5 | 14.28 | 2.10 |
