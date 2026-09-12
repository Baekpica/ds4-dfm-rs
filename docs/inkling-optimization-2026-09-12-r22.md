# Inkling MQ85GB prefill on GB10, 2026-09-12 (rounds 22–24)

Continuation of [rounds 19–21](inkling-optimization-2026-09-12.md). Starting
tree is merged main `c0fc829` after PR #31 (measured as its pre-squash
commit `aaeffe5`, identical content). All measurements
use MQ85GB on one DGX Spark / GB10, CUDA 13.3.73, driver 610.43.02, `sm_121a`
via `CUDA_ARCH=sm_121`. MTP is off in the timed path. Prefill chunk default is
1024 throughout.

These three rounds are new. Rounds 1–21 do not count toward this campaign.

## Protocol

- `speed-bench/promessi_sposi.txt`, 8192/2048 input tokens, 64 greedy output
  tokens, MTP off, context allocation input+65.
- Three unprofiled fresh processes per side, each preceded by a separate
  warmup, same resident base+MTP VMM owner. The owner was restarted for this
  campaign (PID 236445, manifest `/tmp/ds4-inkling-r22.MXV11I/weights.ipc`)
  and the workload manifests rehashed; model, MTP, prompt and template hashes
  are unchanged.
- `ds4-perf scout --proof --repeats 3 --cache-policy warmup-then-fresh`.
  All six main shards, MTP, prompt, template/tokenizer files and IPC
  manifest hashed. Guard max/high 12/10 GiB, host reserve 12 GiB.
- Each retained round compares one new diagnostic switch with the default
  on the same binary. Full-vocabulary logits and generated IDs must match;
  prefill, decode and first-token latency pass `compare --regression` with
  verdict `Improved` (>1% nonoverlapping prefill, decode/first-token not
  slowed >3%).
- Kernel-level claims come from model-free probes at production shapes under
  `scratch/inkling-perf/r22-r24/`, timed as the median of five three-repeat
  samples, byte-exact against the release entry point.

## Entry topology

The round-21 8K candidate trace (21.38 s prefill wall, chunk 1024):

| Kernel | Share | Wall | Launches |
|---|---:|---:|---:|
| `inkling_tile_kernel` (lean IQ2_XXS up, IQ2_XS down) | 39.8% | 8.34 s | 592 |
| `inkling_shared_tile_kernel` (SoA slab) | 17.5% | 3.67 s | 656 |
| `inkling_linear_tile_kernel` (BF16 q/k/v/r/o) | 14.5% | 3.03 s | 1680 |
| `inkling_attention_group_kernel` | 11.1% | 2.32 s | 336 |
| Q4_K tile, dense Q8, Q3_K tile | ~10% | | |

## Round 22: grouped attention reduces each head in one lane group

The grouped prefill kernel scores four query heads per (query, KV head)
CTA. Per key, every lane ran the full 32-lane XOR butterfly for all four
heads (20 shuffles and 20 adds), then computed the score, running maximum,
two exponentials and the softmax sum for all four heads redundantly, before
the V update that needs alpha/beta for every head. SASS: 1171 instructions
with 40 `SHFL`, 78 `FADD` and 38 `MUFU` per loop body against 122 `FFMA`.

The transposed butterfly keeps every head's tree: the offset-16 step sends
the two values the partner keeps and adds the partner's copy of the two
kept ones, so lanes 0–15 hold heads 0/1 and lanes 16–31 heads 2/3; the
offset-8 step does the same for one head per 8-lane group; steps 4/2/1,
the score, `fmaxf`, both `__expf` and the sum update then run once per
group. The pairs (lane, lane ^ delta) are the release pairs in the release
order, so each head's sum is the release float. Alpha and beta are broadcast
from the group's first lane for the V accumulation, whose FMUL/FFMA
sequence is unchanged; only the bias of the owned head is loaded. Per key
this removes 14 shuffles, 14 adds and three heads' softmax scalars.
`DS4_INKLING_NO_ATTN_TRANSPOSE=1` restores the all-lane reduction.

Isolated 1024-row chunk after 3072 committed keys (exact): global extent
19.29 → 14.98 ms, local window 3.19 → 3.03 ms. Reading the current chunk's
K/V from a bf16 staging copy instead of f32 with per-read rounding gained
12% on the local shape but lost 16% on the global one and was not retained.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 387.56 → 394.46 | +1.78% | 13.71 → 13.72 | Improved |
| 2,048 | 408.79 → 411.98 | +0.78% | 17.17 → 17.18 | Pass |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 389.68 / 387.56 / 386.65; default: 396.75 / 394.20 / 394.46 tok/s.
- 2,048 control: 411.74 / 408.79 / 407.95; default: 414.95 / 411.77 / 411.98 tok/s.

The 8K trace shows grouped attention 2.33 → 1.96 s (-16%) and prefill wall
21.42 → 20.99 s. At 2K attention is 6% of the trace, so the round clears
the 1% envelope only at 8K; the 2K comparison is retained as a no-regression
check (`Pass`, +0.78%, first-token and decode within 0.2%). Native:
`tests/test_inkling_attention` covers both extents with chunks 1–8192,
captured replay, rejected suffixes, ring wrap and the new kill switch.
Commit `d1343fa`. Benchmark binary SHA-256:
`0cc20158bcf4b76ebd2cb8519f43753c257690ee90062f7299a37e2b946965a5`.

## Round 23: resident Q8 tiles run the K loop once per column

The round-19 SoA slab kernel sits at the 128-register cap of two CTAs per
SM and still spilled 144 bytes: ncu on the 1024-token up shape showed
2.4 GB of local-memory loads per launch (short-scoreboard 24%,
long-scoreboard 20%). The live set is the resident weight payload (64
registers), packed deltas (16), and the partial and merged sums of eight
columns (32).

The column layout keeps the same slab and the same resident weights but
runs the K loop once per column, so only one column's partial and merged
sums stay live, and stages the activation scale as a float once per block
instead of converting the `half2` in every warp. Per (row, column, block)
the `dp4a` pair, `scale = dw * xs`, FMA chain, ascending partition merge,
XOR tree and nonfinite guard are unchanged, so outputs are byte-identical.
Spills fall to 24 bytes (up) and 56 bytes (down).
`DS4_INKLING_NO_SHARED_COLUMN=1` restores the one-loop kernel;
`DS4_INKLING_NO_SHARED_SOA=1` still restores the canonical-row slab.

Isolated 1024-token shapes (exact): up 6.68 → 5.96 ms, down 3.49 → 3.05 ms.
Two- and four-column passes sat between the two; three-row tiles and
higher block caps spilled and lost.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 394.00 → 403.13 | +2.32% | 13.72 → 13.72 | Improved |
| 2,048 | 412.40 → 421.27 | +2.15% | 17.17 → 17.17 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 395.56 / 394.00 / 393.85; default: 405.35 / 403.13 / 402.47 tok/s.
- 2,048 control: 414.59 / 412.40 / 410.65; default: 424.50 / 421.27 / 420.16 tok/s.

First-token latency stays within 0.6%. The 8K trace shows the shared tiles
3.65 → 3.25 s and prefill wall 21.01 → 20.59 s. Native:
`tests/test_inkling_batch` adds the column switch to every Q8 tile case.
Commit `10f49d0`. Benchmark binary SHA-256:
`5cb3b863cdeae092f51d9498f1240e18c6dc19315e7c0439374a7965eb88b2b3`.

## Round 24: two-key attention iterations and lean Q4_K tiles

Two remaining per-key and per-column loops kept the release structure
after rounds 20–23. This round changes both; each has its own switch and
the comparison turns both off.

**Grouped attention, two keys per warp iteration** (`DS4_INKLING_NO_ATTN_PAIR=1`
restores one key). The transposed kernel is latency-bound on one key's
dot → butterfly → score → exponentials chain per warp. Loading and scoring
keys i and i+4 together gives the scheduler two independent chains; the
softmax scalars and V updates are still applied in key order, so every
running maximum, sum and accumulator sees the release sequence. Isolated
1024-row chunk after 3072 keys (exact): global 14.99 → 13.64 ms, local
3.05 → 2.46 ms. Three keys were no better and four spilled.

**Q4_K expert tiles, lean four-row form** (`DS4_INKLING_NO_Q4_LEAN=1`
restores the two-row branched tile). The round-15 tile still skipped
padded columns with a branch and derived the activation row inside the K
steps, and owned two rows per warp. The lean tile hoists the row, loads
every column (row 0 for padding, stores guarded) and owns four rows; down
caps registers for three CTAs per SM, up keeps the compiler's 255 at two.
`vec_dot_q4_K_q8_1_impl_vmmq` and the ordered sums are unchanged. Isolated
1024-token shapes (exact): up 56.5 → 39.0 ms, down 26.7 → 19.1 ms. A
decode-once rewrite of the Q4_K dot (nibbles, scales, `dm` and the
per-column `u` sums shared across rows) matched only with
`fmaf(dm.x, sumf_d, -(dm.y * sumf_m))` and gained 3% over the lean tile, not
enough to keep. Setting `__launch_bounds__` with an explicit minimum of one
block let the compiler take 168 registers for the control kernel (110 in
round 15); the control keeps the bound unset so the comparison is against
the round-15 code.

A first measurement of this round returned `Pass` at both widths (8K
402.57 → 407.82 tok/s, envelope -1.70..-0.50%; retained under
`scratch/inkling-perf/r22-r24/r24/pass-first/`) with the attention trace
almost unchanged (1.96 → 1.93 s). The tracked kernel indexed its per-head
bias pointer array with the lane group (`rel[head_lane]`), which put the
four pointers in local memory (32 stack bytes, also present in round 22);
the model-free probe had computed one scalar pointer. With a scalar
`relh` the stack is zero for every mode and the tracked entry reproduces
the probe: global 14.96 → 13.59 ms, local 2.71 → 2.40 ms (MODE 1 → MODE 2).
The retained comparison below uses that binary.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 403.97 → 412.81 | +2.19% | 13.72 → 13.71 | Improved |
| 2,048 | 423.91 → 430.67 | +1.59% | 17.18 → 17.17 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 405.89 / 403.97 / 403.26; default: 414.19 / 412.81 / 411.82 tok/s.
- 2,048 control: 425.52 / 423.91 / 421.86; default: 433.76 / 430.51 / 430.67 tok/s.

First-token latency stays within 0.3%. The 8K trace shows grouped
attention 1.88 → 1.69 s, the Q4_K tiles 0.80 → 0.70 s and prefill wall
20.51 → 20.05 s. Native: `tests/test_inkling_attention` adds the pair
switch to both extents; `tests/test_inkling_batch` adds the Q4_K lean
switch and 124-row four-row tiles beside the 126-row fallback. Commit
`0a9d2e2`. Benchmark binary SHA-256:
`b8417c38c6c3424f1bb386be701d16c134377d7153cf2a5fec51be45c5f2ec23`.

## Cumulative

Same-binary default versus all new switches off
(`DS4_INKLING_NO_ATTN_TRANSPOSE=1`, `DS4_INKLING_NO_ATTN_PAIR=1`,
`DS4_INKLING_NO_SHARED_COLUMN=1`, `DS4_INKLING_NO_Q4_LEAN=1`) on the
round-24 binary.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 388.13 → 413.41 | +6.51% | 13.71 → 13.72 | Improved |
| 2,048 | 410.18 → 430.61 | +4.98% | 17.17 → 17.17 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 388.33 / 388.13 / 386.45; default: 414.41 / 413.41 / 412.55 tok/s.
- 2,048 control: 412.32 / 410.18 / 409.26; default: 432.66 / 430.52 / 430.61 tok/s.

First-token latency stays within 0.5%. The same-binary 8K trace falls from
21.41 s to 20.10 s prefill wall: grouped attention 2.32 → 1.69 s, shared Q8
tiles 3.67 → 3.24 s, Q4_K tiles 0.80 → 0.71 s. Against the round-21 report's
387.67 tok/s (8K) and 410.11 tok/s (2K), the default path now runs 413.41
and 430.61 tok/s; against the round-18 report (328.16 / 342.31) the two
campaigns together give +26.0% and +25.8%.

Two additional unprofiled `ds4-bench-perf` launches on the default path:

- 2,048: 434.57 / 432.78 prefill tok/s, 17.18 / 17.19 decode tok/s.
- 8,192: 415.24 / 413.86 prefill tok/s, 13.72 / 13.72 decode tok/s.

## Final validation

- `tests/test_inkling_batch` passes on every round binary, including the
  new kill-switch parity blocks (`DS4_INKLING_NO_SHARED_COLUMN`,
  `DS4_INKLING_NO_Q4_LEAN`) and the 124-row four-row Q4_K tiles.
- `tests/test_inkling_attention` passes with both extents, chunks 1–8192,
  captured replay, rejected suffixes, ring wrap and the
  `DS4_INKLING_NO_ATTN_TRANSPOSE` / `DS4_INKLING_NO_ATTN_PAIR` parity runs.
- 515-token `test_inkling_forward`: `max_abs=0`, committed KV/convolution
  exact, chunks 2/3/7 exact, on every round binary.
- `tests/test_inkling_session --memory-quotes` on the round-24 binary:
  campaign context 8257 at the default cap 1024 quotes 973,649,152 bytes
  (MTP off) and 1,790,212,608 bytes (MTP on); at cap 8192, 5,576,422,656
  and 11,348,081,152 bytes. All unchanged from round 21.
- `tests/test_inkling_session` with the restarted MQ85GB + eight-layer
  MTP-BF16 owner: lazy allocation, no-op/extend/decode/reset/rewind parity
  at context 32 (estimate = measured 164,265,472 bytes), native MTP eight
  cycles with 18 greedy tokens and exact target logits/KV/convolution,
  image and audio identity/invalid-input state.
- `cargo fmt --all -- --check`, workspace clippy, `cargo test -p ds4-perf`
  and all-target workspace check pass.
- Retained commits: R22 `d1343fa`, R23 `10f49d0`, R24 `0a9d2e2`.

## Rejected probes

All probes are model-free production shapes under
`scratch/inkling-perf/r22-r24/`, byte-exact unless noted.

- IQ2 tile instruction diet on the lean kernel (`r19-r21/iq2/` rebuilt):
  the integer `sumi * ls / 8` as `truncf(float * 0.125f)` and a 16-byte
  L1 sign-mask table both ran 2–4% slower than round 20 (18.12 ms up);
  the kernel is no longer instruction-bound.
- BF16 linear tile with an fp32 token slab (`linear/`): converting each
  BF16 pair once per CTA instead of once per warp removed 25% of the
  instructions but doubled the shared-memory reads; every slab/bound
  variant was 15–80% slower than the release tile (3.17 ms for 4096×4096).
- Attention current-chunk K/V read from a bf16 staging copy (`attn/` AT 2):
  -12% on the local window, +16% on the global shape; not retained.
- Attention generic KEYS loop with predicated fetches: slower than the
  hand-written two-key form (14.82 vs 13.64 ms global); four keys spilled
  (27 ms).
- Shared Q8 tile two- and four-column passes (5.78–6.16 ms up) sat between
  the one-loop and per-column forms; three rows per warp and higher block
  caps spilled and lost.
- Q4_K decode-once dot (`q4/`): only forms with
  `fmaf(dm.x, sumf_d, -(dm.y * sumf_m))` matched the MMVQ oracle; the gain
  over the lean tile was 3% (37.5 vs 38.7 ms up) and it was not kept.

## Evidence

Raw commands, hashes, guards, proofs, traces, ncu reports and comparisons
are retained under `scratch/inkling-perf/r22-r24/`: `ncu-*.txt` (entry
profiles), `attn/`, `shared/`, `linear/`, `q4/` (variant probes), `r22/`,
`r23/`, `r24/`, `cumulative/` and `final-checks/`. The workload manifests
`workload-{8k,2k}.json` rehash the round-19 files with the restarted
owner's IPC manifest. These are MQ85GB text-prefill measurements, not
source-model, long-context, media-performance or MTP throughput
qualification.
