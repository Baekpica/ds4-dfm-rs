# Hy4-flash MXQ on DGX Spark

The mixed-quant release at Hub revision `94cdace56aaa0afc463311e31a2bbc3e613dcce2` ran on one GB10 with CUDA 13.5, driver 620.70.06 and the 300–2,200 MHz clock policy. Busy SM clock was 2,190 MHz. The host-memory guard kept a 2 GiB reserve and trip floor. Each comparison used fresh model processes. Prefill rates below cover newly appended tokens. Generation has its own timer.

## Tiled MLA prefill

At the 4K frontier the scalar SWA and sparse MLA kernels took 46.6% of the 2K append. A model-free replay used 512 query rows at position 3,584, the release head counts, learned sinks, rotary layout and sparse gather width. On 64 heads, scalar and tiled medians were 12.59 and 1.10 ms. On 32 heads they were 31.27 and 3.30 ms. Nsight Compute profiled the isolated kernels. The tiled kernel shares the BF16 matrix instructions and online softmax with Ling-3.1 and maps each physical key through Hy4's window or selected group.

With 512-row chunks and 32 greedy output tokens, 4K incremental prefill improved from 309.68 to 551.58 tok/s. The four measured frontiers spanned 303.70–367.88 and 534.46–569.49 tok/s respectively. Plain decode ranged from 13.87–14.18 and 13.98–14.23 tok/s. Each file is a single sweep. [Scalar CSV](benchmarks/hy4-flash-mxq-2026-11-19/scalar-512.csv) · [Tiled CSV](benchmarks/hy4-flash-mxq-2026-11-19/flash-512.csv).

The fast path rounds absorbed queries and softmax weights to BF16 for MMA, accumulates in FP32, and shares the scalar cache. On two deterministic random head geometries its relative RMS output difference was 0.00248 (32 heads) and 0.00247 (64 heads). The reference remains available with `DS4_HY4_FLASH=0`.

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
