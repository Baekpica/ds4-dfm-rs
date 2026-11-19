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
