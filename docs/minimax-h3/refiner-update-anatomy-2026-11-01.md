# One routed update, priced (2026-11-01)

Phase D, first pass. Two instruments (`H3_STAGE2_PROFILE`, `H3_BLOCK_PROFILE`,
`5cbd36d0`) on the deliverable route: release sigmas, Sol on, anchor 0.7, the
recipe's own draft, 16,128 rows.

**This also corrects a number the campaign has been reasoning from.** Weight
streaming was never the wall - see section 3.

## 1. The update, split

`STAGE2 blocks=48 weights_gib=34.58 rows=16128` from the loader, and one step's
four phases:

| phase | seconds | share |
| --- | ---: | ---: |
| prepare (edges, adaln, patchify) | 0.717 | 1.6 % |
| 48 blocks | 43.703 | 98.3 % |
| head | 0.007 | 0.02 % |
| sampler (noise lerp, blend, Euler) | 0.000 | 0.00 % |
| step | 44.447 | |

So there is nothing to win outside the stack: the sampler and the head are
rounding error, and the edges are 1.6 %.

## 2. Inside the stack: uniform, with one outlier

All 48 blocks timed individually (`H3_BLOCK_PROFILE=1`); the syncs make each
block's own number the useful one and inflate the sum to 36.4 s:

| block | ms |
| --- | ---: |
| layer 0, the one dense attention (`routed=false`) | 2,690 |
| layers 1-47, routed | 687 - 913 |
| average over all 48 | 759 |

Layer 0 is 3.5x a routed block because its video self-attention is dense over
16,128 tokens; the difference, about 1.9 s, is that attention. Every routed
block lands in the same band, so the cost is the stack's own arithmetic spread
evenly, not a pathological layer, a cold cache or one bad kernel.

That same 1.9 s per block is why the dense arm is 131 s: with `--sol` off all
48 attentions are dense, `48 x 1.9 s` on top of the routed 44 s.

## 3. The correction: streaming the weights costs 0.15 s, not 136 s

`reference-latency-2026-10-31.md` §5 and §7 reasoned that 3 x 34.6 GiB of
weights in 24.71 s is "4.5 TB/s against this host's measured 241-246 GB/s", and
concluded the published stage-2 number could not be this artifact on this part.
The arithmetic is off by a factor of 1000:

| quantity | value |
| --- | ---: |
| the shard | 34.58 GiB = 37.13 GB |
| one update's weight sweep at 246 GB/s | **0.151 s** (0.34 % of 44.447 s) |
| three updates, i.e. 111.4 GB, over 24.71 s | 4.51 **GB**/s, not TB/s |

Every update reads the whole shard, and that read is a third of one percent of
the update's cost. Nothing in stage 2 is stream-bound on this host.

## 4. What actually binds, and what the published bar implies

The stack is compute-bound. Per update the 48 blocks apply the video tower's
projections to 16,128 rows: about 414 TFLOP by the tower's own parameter count
(qkv 67 M, cross-attn 67 M, FF 134 M per block, 48 blocks) and 600 TFLOP if the
whole 18.6 G-parameter shard is counted per token. Measured:

| arm | update | TFLOP | TFLOPS |
| --- | ---: | ---: | ---: |
| ours, routed | 44.447 s | 414 - 600 | 9.3 - 13.5 |
| ours, dense | 131.2 s | same | 3.2 - 4.6 |
| published, `24.71 s / 3` | 8.24 s | 414 - 600 | 50 - 73 |

Against this part's envelope (`docs/gb10-hardware.md`): BF16 peak 128.4 TFLOPS,
cuBLAS BF16 GEMMs 65-90 TFLOPS. The published 24.71 s therefore sits **inside**
what cuBLAS reaches here, so the bar is not impossible and the earlier
conclusion is withdrawn: what is 5x off is our achieved rate, not the physics.

The blocks' GEMMs go through `h3d_linear_lt`, which is cuBLASLt, so the gap is
not a naive matmul - it is either the shape/heuristic the port asks for, or the
number of calls each block makes around the GEMMs (per-head projections,
transposes, the elementwise stream between them). That is the next measurement:
the exact block shapes at 16,128 rows through the same call, to say whether the
GEMM alone reaches the 65-90 band.

## 5. Where this leaves the phase

- Nothing to win in prepare / head / sample.
- The deliverable route's cost is 48 uniform blocks; the only asymmetric block
  is layer 0, and it is dense by the release's own policy.
- The published bar is achievable in principle: it needs the GEMMs at 5x the
  rate the blocks currently achieve, i.e. cuBLAS's own envelope, not a smaller
  model and not a different part.
- Dense attention is 1.9 s per layer at 16,128 tokens, about 4 % of peak on that
  term; it is off the deliverable route but it is where a further 3x of the
  *dense* arm would come from.

Artifacts: `outputs/prof-phases.mp4`, `outputs/prof-blocks.mp4`, logs
`scratch/profile-arms.log`, script `scratch/profile-arms.sh`, both on the Spark.
