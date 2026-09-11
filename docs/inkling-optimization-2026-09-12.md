# Inkling MQ85GB prefill on GB10, 2026-09-12 (rounds 19–21)

Continuation of [rounds 16–18](inkling-optimization-2026-09-11-r16.md). Starting
tree is merged main `6baf01f` after PR #30. All measurements use MQ85GB on
one DGX Spark / GB10, CUDA 13.3.73, driver 610.43.02, `sm_121a` via
`CUDA_ARCH=sm_121`. MTP is off in the timed path. Prefill chunk default is
1024 throughout.

These three rounds are new. Rounds 1–18 do not count toward this campaign.

## Protocol

- `speed-bench/promessi_sposi.txt`, 8192/2048 input tokens, 64 greedy output
  tokens, MTP off, context allocation input+65.
- Three unprofiled fresh processes per side, each preceded by a separate
  warmup, same resident base+MTP VMM owner (`--repack-iq2-aligned`, PID 133642).
- `ds4-perf scout --proof --repeats 3 --cache-policy warmup-then-fresh`.
  All six main shards, MTP, prompt, template/tokenizer files and IPC
  manifest hashed. Guard max/high 12/10 GiB, host reserve 12 GiB.
- Each retained round compares one new diagnostic switch with the default
  on the same binary. Full-vocabulary logits and generated IDs must match;
  prefill, decode and first-token latency pass `compare --regression` with
  verdict `Improved` (>1% nonoverlapping prefill, decode/first-token not
  slowed >3%).
- Kernel-level claims come from a model-free probe at the production shape
  (1024 tokens, M=4096, 256 routed / 2 shared experts, skewed top-6 routing)
  under `scratch/inkling-perf/r19-r21/`, timed as the median of five
  three-repeat samples, byte-exact against the release entry point.

## Entry topology

The round-18 8K candidate trace (25.27 s prefill wall, chunk 1024):

| Kernel | Share | Wall | Launches |
|---|---:|---:|---:|
| `inkling_tile_kernel` (IQ2_XXS up, IQ2_XS down) | 40.2% | 9.97 s | 592 |
| `inkling_shared_tile_kernel` (shared Q8 up/down) | 22.3% | 5.55 s | 656 |
| `inkling_linear_tile_kernel` (BF16 q/k/v/r/o) | 11.8% | 2.94 s | 1680 |
| `inkling_attention_group_kernel` | 9.3% | 2.30 s | 336 |
| Q4_K tile, Q3_K four-column, dense Q8 | 9.2% | 2.29 s | 64 |

Isolated 1024-token production shapes: shared Q8 up 10.61 ms, shared Q8
down 4.68 ms, IQ2_XXS up 20.97 ms, IQ2_XS down 13.51 ms.

## Round 19: shared Q8 tiles stage an SoA slab with half deltas

The round-13/14 resident-weight kernels held every block delta as a float,
which put the up variant at 255 registers and one CTA (eight warps) per SM.
ncu on the 1024-token shape: issue slots busy 29%, short-scoreboard 26%,
wait 20%, long-scoreboard 19%, 2.00 active warps per scheduler, and one
third extra shared-memory wavefronts from the 9-word block stride (lanes
`g` and `g+4` collide). The canonical 36-byte slab also costs three LDS per
(column, block) and a `half2` load shared by four lanes.

The new slab keeps the same byte count but is split while staging: eight
`qs` rows of `int2` pairs followed by the `half2` scales. One 8-byte LDS
feeds both `dp4a`, the eight blocks of a K step land in distinct banks, and
weights keep their `half` deltas packed as `half2` until each (row, block)
use. The conversion equals the load-time float, so every lane product, FMA
chain, ascending partial merge, XOR tree and nonfinite guard is unchanged.
The register footprint drops to 128 (up) and 64 (down), giving two and four
CTAs per SM. `DS4_INKLING_NO_SHARED_SOA=1` restores the canonical-row slab
and float deltas.

Isolated 1024-token shapes (exact): up 10.70 → 6.62 ms, down 4.70 → 3.51 ms.
Staging variants that copied `int4` rows with runtime division or hoisted
sixteen loads spilled under the 128-register cap and ran at 8–9 ms; one
36-byte block per thread pass (nine words in flight) is the retained form.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 325.41 → 350.02 | +7.56% | 13.78 → 13.77 | Improved |
| 2,048 | 341.19 → 368.86 | +8.11% | 17.26 → 17.26 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 327.10 / 325.41 / 324.51; default: 351.44 / 349.64 / 350.02 tok/s.
- 2,048 control: 343.16 / 341.19 / 340.51; default: 370.43 / 368.86 / 367.82 tok/s.

First-token latency stays within 0.3%. The same-binary 8K traces show
shared Q8 tiles 5.57 → 3.72 s and prefill wall 25.41 → 23.66 s. Commit
`f57cf26`. Benchmark binary SHA-256:
`eacc9d4d9af54eeebc1dff1ccdd1446264b650cd04211346552eaaf6d55dceec`.

## Round 20: branch-free IQ2 expert tiles with arithmetic signs

The IQ2_XXS up and IQ2_XS down tiles (40% of the round-19 trace) ran at
128 registers and four CTAs per SM with issue slots busy 48% and
long-scoreboard stalls at 43%. SASS showed why: each column's activation
loads sit behind a `source < 0` branch and a runtime `/ used`, so the
compiler issues a column's two 16-byte loads, waits, runs its 32 `dp4a`,
then starts the next column; sign application spent about twenty ALU
instructions per fragment emulating `__vcmpne4` / `__vsub4`.

The lean kernel hoists the activation row index out of the K steps, loads
all eight columns unconditionally (a padded column reads row 0 and its
store stays guarded), and expands the 7-bit sign code arithmetically:
`t = (nibble * 0x204081) & 0x01010101`, `s = t * 0xFF`,
`v = (grid ^ s) + t`, which equals `__vsub4(grid ^ s, s)` because every
grid byte is at least 8. The integer dots, `sumi * ls / 8`, the
`d * xs` product, the FMA chains, ordered merges and XOR tree are
unchanged. The launch bound targets three CTAs per SM, which hid more load
latency than four at 128 registers or two at 182 in every probe variant.
Padded columns cost real dots, so the lean kernel starts at 256 prompt
tokens (1536 assignments): an ungated first measurement improved prefill by
the same margin but slowed decode 1.2% because the width-one tiles pay for
seven padded columns per expert. Decode and verify widths keep the branched
kernel. `DS4_INKLING_NO_IQ2_LEAN=1` restores it everywhere.

Isolated shapes (exact), lean vs branched:

| Tokens | IQ2_XXS up (ms) | IQ2_XS down (ms) |
|---:|---:|---:|
| 8 | 0.90 vs 0.84 | 0.69 vs 0.53 |
| 128 | 4.91 vs 5.45 | 3.65 vs 3.29 |
| 256 | 6.46 vs 7.69 | 4.47 vs 4.71 |
| 512 | 9.89 vs 11.92 | 6.19 vs 7.59 |
| 1024 | 18.52 vs 20.94 | 11.14 vs 13.88 |

Instruction-only variants (hoist or arithmetic signs with the branch kept)
gained under 1%; shared-memory grid/sign tables lost occupancy and ran
slower; five or six CTAs per SM spilled.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 351.68 → 378.57 | +7.65% | 13.78 → 13.78 | Improved |
| 2,048 | 369.61 → 400.37 | +8.32% | 17.27 → 17.28 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 353.60 / 351.68 / 351.18; default: 380.27 / 378.57 / 377.80 tok/s.
- 2,048 control: 369.61 / 369.61 / 369.46; default: 401.92 / 400.37 / 398.86 tok/s.

First-token latency stays within 0.3%. The same-binary 8K traces show the
IQ2 tiles 9.95 → 8.37 s and prefill wall 23.49 → 21.91 s. The ungated first
measurement (`scratch/inkling-perf/r19-r21/r20-ungated/`) reached
352.47 → 378.75 tok/s at 8K with decode 13.78 → 13.62 (-1.2%), which the
width gate removes (its binary SHA-256
`e0b60e11c4d020e2f500a6b380ce336f5fc3026025a5c341ac590a9122886d9b`).
Commit `5b0d40a`. Benchmark binary SHA-256:
`74e30865a8ef70be0ca3ea8e6bd3843272a4287fa8d31e0dd6c5961bea07c1f8`.
In the retained run the source-file hash check flagged only
`cuda/mmq/inkling_mmvq.cuh`, which round 21 edited after the round-20
binaries were built; every binary and object hash matched.

## Round 21: Q3_K expert up decodes each fragment once

Layer 40 keeps its routed up projection in Q3_K. That call still fell back
to the four-column MMVQ kernel: 3.6% of the round-20 trace in eight launches
of 99 ms, six times the IQ2_XXS up cost for the same shape, because
`vec_dot_q3_K_q8_1` re-unpacks the 3-bit values, high-bit mask and 6-bit
scales for every column. The earlier Q3 probe (rounds 16–18) only shared
the payload loads and measured 3%.

The new tile unpacks a row fragment once into four signed `int` words and
four scales, then reuses it for eight routed columns; each column loads its
four Q8 words at a 32-byte stride and its four scales as one 16-byte load.
Per (row, column, fragment) the `dp4a` products, the `sc` multiply, the
four-term `d8` FMA chain, the `d3` FMA into the K-partition sum, the
ascending warp merge and the XOR tree are the MMVQ sequence, so outputs are
byte-identical; a variant that rounded the inner products separately did
not match and was discarded. Four rows per warp take 255 registers (two
CTAs per SM) and still beat every two-row or register-capped variant.
The path starts at 256 prompt tokens; `DS4_INKLING_NO_Q3_TILE=1` restores
the four-column kernel.

Isolated shapes (exact): 1024 tokens 96.90 → 38.52 ms, 512 tokens
50.29 → 21.95 ms, 256 tokens 26.06 → 14.58 ms.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 378.72 → 387.73 | +2.38% | 13.79 → 13.79 | Improved |
| 2,048 | 399.98 → 409.97 | +2.50% | 17.27 → 17.28 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 380.02 / 378.72 / 377.74; default: 388.64 / 387.73 / 387.38 tok/s.
- 2,048 control: 401.68 / 399.52 / 399.98; default: 411.26 / 409.97 / 408.97 tok/s.

First-token latency stays within 0.5%. The same-binary 8K traces show the
layer-40 up call 0.79 s (eight `inkling_mmvq_kernel` launches) leaving the
top five, and prefill wall 21.89 → 21.39 s. Native: `tests/test_inkling_batch`
covers 255/256/257/8192 tokens at 124 rows (tile) and 126 rows (four-column
fallback), random/repeated/invalid routes, the full 4096-row shape and the
kill switch. Commit `da09a2e`. Benchmark binary SHA-256:
`4bb58a77cdcb594c7d2ff5d317565bcc41d2e1e803d74308911194380d0a93f9`.

## Cumulative

Same-binary default versus all three new switches off
(`DS4_INKLING_NO_SHARED_SOA=1`, `DS4_INKLING_NO_IQ2_LEAN=1`,
`DS4_INKLING_NO_Q3_TILE=1`) on the round-21 binary.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 326.21 → 387.67 | +18.84% | 13.77 → 13.79 | Improved |
| 2,048 | 342.29 → 410.11 | +19.81% | 17.26 → 17.28 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 328.35 / 326.21 / 326.16; default: 389.40 / 387.67 / 386.74 tok/s.
- 2,048 control: 343.65 / 341.85 / 342.29; default: 412.21 / 410.11 / 409.01 tok/s.

First-token latency stays within 0.6%. The same-binary 8K trace falls from
25.28 s to 21.38 s prefill wall: IQ2 tiles 9.96 → 8.34 s, shared Q8 tiles
5.55 → 3.67 s, the layer-40 Q3_K up call 0.79 s → below the top five.
Against the round-18 report's 328.16 tok/s (8K) and 342.31 tok/s (2K), the
default path now runs 387.67 and 410.11 tok/s.

Two additional unprofiled `ds4-bench-perf` launches on the default path:

- 2,048: 413.41 / 411.15 prefill tok/s, 17.29 / 17.29 decode tok/s.
- 8,192: 389.76 / 387.79 prefill tok/s, 13.78 / 13.78 decode tok/s.

## Rejected probes

All probes are model-free 1024-token production shapes under
`scratch/inkling-perf/r19-r21/`, byte-exact unless noted.

- IQ2 tile instruction diet alone (`iq2/`): hoisting the row index or
  spreading signs arithmetically while keeping the column branch gained
  under 1% (20.94 → 20.71 ms up); the compiler spent the freed registers
  and occupancy fell from four to three CTAs with no scheduling benefit.
- IQ2 grid/sign tables in shared memory: 26.76 ms at two CTAs per SM with
  the branch, 18.27 ms branch-free at three CTAs, never better than the
  arithmetic form; five or six CTAs per SM spilled 260–760 bytes and ran
  1.4–5× slower.
- Shared Q8 slab staging as coalesced `int4` copies (runtime division per
  vector, or sixteen hoisted loads per column) spilled under the
  128-register cap and ran at 8.2–9.0 ms against 6.6 ms for the retained
  one-block-per-pass staging; R=1 × 16 warps (10.82 ms) and R=2 × 12 warps
  (7.10 ms) lost to R=2 × 8 warps at two CTAs.
- BF16 linear tile launch bounds (`linear/`): every variant (three to five
  CTAs per SM via a 2-step slab, eight warps, eight-step slab, two-row
  tiles) was equal or slower than the release 16×16 tile (3.17 ms for the
  4096×4096 shape; 4.7–60 ms for the capped variants, which spill).
- Q3_K float form with separately rounded inner products mismatched the
  MMVQ oracle (15.4M of 25.2M outputs), confirming that the retained tile
  must keep the `fmad` contraction of the source; two-row tiles and
  register caps (3–6 CTAs) ran 1.2–10× slower than the four-row tile.

## Final validation

- `tests/test_inkling_batch` passes on every round binary, including the
  new kill-switch parity blocks (`DS4_INKLING_NO_SHARED_SOA`,
  `DS4_INKLING_NO_IQ2_LEAN`, `DS4_INKLING_NO_Q3_TILE`) and the Q3_K
  threshold edges.
- 515-token `test_inkling_forward`: `max_abs=0`, committed KV/convolution
  exact, chunks 2/3/7 exact, on every round binary.
- `tests/test_inkling_session --memory-quotes` on the round-21 binary:
  campaign context 8257 at cap 8192 quotes 5,576,422,656 bytes (MTP off)
  and 11,348,081,152 bytes (MTP on).
- `tests/test_inkling_session` with the shared MQ85GB + eight-layer
  MTP-BF16 owner: lazy allocation, no-op/extend/decode/reset/rewind parity
  at context 32 (estimate 164,265,472 bytes = measured), native MTP eight
  cycles with 18 greedy tokens and exact target logits/KV/convolution,
  image and audio identity/invalid-input state.
- `cargo fmt --all -- --check`, workspace clippy, `cargo test -p ds4-perf`
  and all-target workspace check pass.
- Retained commits: R19 `f57cf26`, R20 `5b0d40a`, R21 `da09a2e`.

## Evidence

Raw commands, hashes, guards, proofs, traces, ncu reports and comparisons
are retained under `scratch/inkling-perf/r19-r21/`: `prof/` (timing and
ncu of the four entry kernels), `shared/`, `iq2/`, `linear/`, `q3/`
(variant probes), `r19/`, `r20/` (retained gated run), `r20-ungated/`,
`r21/`, `cumulative/` and `final-checks/`. The workload manifests are the
unchanged `scratch/inkling-perf/r16-r18/workload-{8k,2k}.json`. These are
MQ85GB text-prefill measurements, not source-model, long-context,
media-performance or MTP throughput qualification.
