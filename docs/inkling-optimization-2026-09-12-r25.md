# Inkling MQ85GB prefill on GB10, 2026-09-12 (round 25)

Continuation of [rounds 22–24](inkling-optimization-2026-09-12-r22.md).
Starting tree is merged main `d1c5061` after PR #32. All measurements use
MQ85GB on one DGX Spark / GB10, CUDA 13.3.73, driver 610.43.02, `sm_121a`
via `CUDA_ARCH=sm_121`. MTP is off in the timed path. Prefill chunk default
is 1024 throughout. The owner was restarted for this campaign and the
workload manifests rehashed; model, MTP, prompt and template hashes are
unchanged.

The brief was the two largest remaining kernels: the lean IQ2 expert tile
(42.5% of the 8K trace) and the resident shared Q8 tile (16.6%), then, on
the user's instruction, an attention round that may change the numerical
contract the way the Qwen3.8 fused kernel did. Rounds 25 and 26 are exact;
round 27 is the first Inkling kernel whose outputs are not byte-identical
to the reference sequence, and its section states the new contract.

## Protocol

As in rounds 22–24: `speed-bench/promessi_sposi.txt`, 8192/2048 input
tokens, 64 greedy output tokens, MTP off; `ds4-perf scout --proof
--repeats 3 --cache-policy warmup-then-fresh` with three unprofiled fresh
processes per side; `compare --regression` must report `Improved` with
1,200,348 logits `max_abs=0` and zero token mismatches for rounds 25 and
26. Round 27 changes the attention contract and is judged under the
relative-RMS contract it introduces (`--logit-rel-rms 0.12`, described in
that section). Kernel-level claims come from model-free probes at
production shapes under `scratch/inkling-perf/r25-r27/`, byte-exact
against the release entry except where round 27 says otherwise.

## Entry topology

The round-24 8K candidate trace (20.05 s prefill wall):

| Kernel | Share | Wall |
|---|---:|---:|
| `inkling_tile_kernel` (lean IQ2_XXS up, IQ2_XS down) | 42.5% | 8.32 s |
| `inkling_shared_tile_kernel` (column layout) | 16.6% | 3.25 s |
| `inkling_linear_tile_kernel` | 15.4% | 3.02 s |
| `inkling_attention_group_kernel` | 8.6% | 1.69 s |

ncu on the 1024-token shapes: shared up (2 CTAs, 128 registers) issues
56% of slots with short-scoreboard 24.5%, wait 18%, long-scoreboard 13%
and barrier 6%; shared down (4 CTAs, 64 registers) runs the L1/LDS pipe at
86% with short-scoreboard 38%. The lean IQ2 up tile (3 CTAs, 168
registers) issues 56% with long-scoreboard 24%, wait 18% and
short-scoreboard 15%.

## Round 25: resident Q8 tiles stream columns through a cp.async ring

The column kernel of round 23 still stages every eight-column slab with
plain loads between two barriers, so the load latency is exposed once per
slab and the down tile, at four CTAs per SM, saturates the shared-memory
pipe reading 8 bytes of activation per (row, column, block-quarter) for
only two rows.

The pipeline keeps the resident weights and the per-column K loop but
takes the activations from a float-scale SoA (one relayout per call, the
same bytes as the IQ2 relayout) and streams them through a four-stage ring
of column buffers that `cp.async` fills two columns ahead, so staging
overlaps compute, costs no registers, and the ring needs 18 KB instead of
a 37 KB slab. Up keeps two rows per warp at two CTAs per SM; down owns four
rows per four-warp CTA at four CTAs, halving shared-memory bytes per
output. Per (row, column, block) the `dp4a` pair, `dw * xs`, FMA chain,
ascending partition merge, XOR tree and nonfinite guard are the release
sequence, so outputs are byte-identical.
`DS4_INKLING_NO_SHARED_PIPE=1` restores the round-23 staged-slab kernels.

Isolated 1024-token shapes (exact): up 5.95 → 4.96 ms, down 3.09 → 2.44 ms.
Rings of three, six and eight stages were within 2% of four; twelve warps
per CTA (168 registers) and three rows per warp spilled and lost; the
two-row pipelined down tile at four CTAs was 4% slower than round 23.

## IQ2 expert tile: rejected probes

The lean IQ2 tile of round 20 (three CTAs per SM at 168 registers) was the
other target. Every exact restructure below was measured on the 1024-token
IQ2_XXS up shape (`scratch/inkling-perf/r19-r21/iq2/`, rebuilt) against the
lean kernel's 18.3 ms:

- Compile-time K and assignment stride plus a validity bitmask instead of
  eight source registers (address math by shifts): 19.5 ms at three CTAs;
  the compiler took the freed registers without issuing loads earlier.
- The same with the next K step's raw fragments prefetched during the
  current step: 19.9 ms at three CTAs, 21.0 ms unbounded (two CTAs),
  24.1 ms capped at four CTAs (spills).
- Earlier in this campaign series: a float-truncation form of the integer
  `sumi * ls / 8`, an L1 sign-mask table, shared-memory grid/sign tables,
  and five or six CTAs per SM, all slower.

The kernel issues 56% of slots with long-scoreboard 24%, wait 18% and
short-scoreboard 15% stalls; its per-output arithmetic (eight `dp4a`, the
integer scale and truncating division, one conversion, one product and one
FMA per fragment) is fixed by the MMVQ contract, and occupancy is bounded
by the decoded four-row fragment set plus the 4×8 accumulators. Further
gains need either a different numerical contract (tensor-core integer dots
with a different float accumulation order) or a per-tile shared-memory
activation slab, which costs occupancy; the latter is probed below.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 410.45 → 417.76 | +1.78% | 13.68 → 13.68 | Improved |
| 2,048 | 427.67 → 436.87 | +2.15% | 17.12 → 17.09 | Pass |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 411.73 / 410.45 / 409.76; default: 420.06 / 417.40 / 417.76 tok/s.
- 2,048 control: 431.33 / 427.67 / 427.04; default: 439.65 / 436.87 / 435.56 tok/s.

The 2K comparison is `Pass` only because one control sample (431.33) puts
the envelope's upper end at -0.97%; decode and first-token move within
0.5%, the latter by the extra relayout launch. Commit `30037de`.
Benchmark binary SHA-256:
`9cfdb11bb33ed193bbf0a41007d0afa8c0f17ad5245394f4c754a1f534580fcc`.

## Attention: what carries over from Qwen3.8 and what does not

Qwen3.8's long-context fix fused the QSA slot walk into one kernel per
(row, KV head) with 32-token key tiles in shared memory and register
accumulation, accepting fp32 reordering (fixture max 1.3e-7); the other
families use tensor-core flash attention. Inkling's grouped kernel already
walks the KV prefix once per (query, KV head) with the four query heads
sharing each key row, and its contract is byte-exactness against the
per-head kernel. The transferable exact idea is a query tile that reads
each key row once per several queries. A probe (`r22-r24/attn/`,
`at_qtile_kernel`) showed two things: (1) on the 1024-row global shape the
two-query tile was slower than the two-key kernel (17.75 vs 13.70 ms) even
though it halves K/V reads, so the kernel is not bound by K/V traffic but
by the per-key score, exponential and V chains; (2) local layers cannot
tile consecutive queries exactly, because each query's window start
decides which warp scores which key, so the tile would have to hold
queries four apart. Exact query tiling is therefore rejected; the
remaining attention lever is a different numerical contract (tensor-core
dots with a different accumulation order), which round 27 adopts on the
user's instruction.

## Round 26: IQ2 expert tiles stage each tile's activations once per CTA

The lean IQ2 kernel assigns one (tile, row group) job per warp and every
job reads its eight columns' activation fragments from L1: two 16-byte
loads and a scale per (column, K step), 96 loads per job, 115 GB of L1
traffic per 1024-token up launch, with long-scoreboard the largest stall.
The four warps of a CTA usually hold consecutive row groups of the same
tile, so the same fragments are fetched four times per CTA and 1024 times
per tile.

The slab kernel gives each CTA one eight-column tile: it stages the
tile's activation rows once in shared memory (8 × 4 KB int8 plus 8 × 512 B
scales for up, half that for down; a padded column reads row 0) and its
four warps sweep the tile's row groups, reading fragments from shared
memory. Weight decode, the `dp4a` dots, the truncating scale, the
per-fragment FMA, the ordered partition merge and the XOR tree are the lean
kernel's, so outputs are byte-identical. Up runs two CTAs per SM (36 KB
slabs, 128 registers), down four (18 KB). The path applies with the lean
gate (256 prompt tokens) on aligned SoA weights; `DS4_INKLING_NO_IQ2_SLAB=1`
keeps the per-warp lean tiles.

Isolated 1024-token shapes (exact): IQ2_XXS up 18.07 → 14.76 ms, IQ2_XS
down 10.81 → 7.30 ms. Note that the register-diet and prefetch variants
above had failed on the same kernel: what mattered was not the load
count per job but the L1 traffic and its latency, which shared memory
removes at the cost of occupancy.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 417.34 → 463.31 | +9.92% | 13.68 → 13.68 | Improved |
| 2,048 | 436.58 → 486.85 | +10.33% | 17.12 → 17.12 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 419.82 / 417.34 / 417.24; default: 465.02 / 463.31 / 462.58 tok/s
  (envelope -10.27..-9.24%).
- 2,048 control: 438.53 / 436.58 / 435.71; default: 489.27 / 486.85 / 485.59 tok/s
  (envelope -10.95..-9.69%).

The 8K candidate trace: `inkling_tile_slab_kernel` 6.50 s (37.1%, was
8.44 s as the lean tile), linear tile 3.06 s, shared pipe 2.84 s, grouped
attention 1.70 s (9.7%); prefill wall 19.89 → 17.96 s. Commit `2377a4d`.
Benchmark binary SHA-256:
`846463015d37b63484ac3fb152f854bbd5b80aa5b5aeb938173a5727a6c3b00a`.

## Round 27: prefill attention on the tensor cores (new numerical contract)

With the expert tiles reduced, grouped attention is 9.7% of the 8K trace
(1.70 s over 336 launches), and the exact analysis above leaves no exact
lever. The user asked for a Qwen-style contract change if it pays, so this
round replaces the prefill attention arithmetic.

**Kernel** (`cuda/mmq/inkling_attention.cuh`,
`ds4_mmq_inkling_prefill_attn_hmma`, dispatched from
`ds4_gpu_inkling_attention` for widths of 16 rows and more). One CTA of
eight warps owns a 64-query tile of two query heads that share a KV head
(grid: query tiles × 16 head pairs). The current chunk's K/V rows are first
rounded to bf16 into a sticky device copy in the cache row layout (the
value the KV store commits), so every key is a bf16 row that `cp.async`
streams into a two-stage shared-memory ring of 64-key tiles (2 × 34 KB,
opt-in dynamic shared memory) one tile ahead of compute. Scores are bf16
`m16n8k16` MMAs of the bf16-rounded Q against the tile with fp32
accumulation (bf16 × bf16 products are exact in fp32); the relative bias
is read from the prepared table, bf16-rounded and added in fp32 with the
1/128 scale; the softmax runs online once per 64-key tile. Probabilities
enter the PV MMAs as a bf16 hi/lo pair (`hi = bf16(w)`, `lo = bf16(w - hi)`,
two MMAs per fragment), so the weights carry ~16 mantissa bits; the fp32
output is divided by the row sum and rounded to bf16 as before. Key tiles
start at multiples of 64 in absolute key position and every masked key
contributes an exact zero, so a query's result does not depend on the
prefill chunk boundary. 240 registers, one CTA per SM. The path is
opt-in: `DS4_INKLING_ATTN_HMMA=1` selects it, `DS4_INKLING_NO_ATTN_GROUP=1`
bypasses it, and the default stays the byte-exact grouped kernel for the
reason given under *Contract*. Decode and MTP verify widths (below 16
rows) keep the exact per-head kernel either way.

**Contract.** The exact kernels' outputs were byte-identical to the
reference sequence. The tensor-core path differs from them only by
summation order (MMA accumulation, per-tile instead of per-key softmax)
and the bf16-pair probability rounding (relative 2^-16 per weight).
`tests/test_inkling_attention` now keeps two baselines: the exact one
(rollbacks and sub-16-row chunks reproduce it byte-for-byte, as before)
and the tensor-core one, which every 16+ row chunking (16, 63, 257, 700,
8192 rows) reproduces byte-for-byte and which is checked against the FP64
softmax at the same probe positions with the exact kernels' bound (half a
bf16 ulp + 2e-6) plus 2^-16 · max|V|. On the 8,201-token fixture 0.12%
(local) and 0.62% (global) of the 33.6 M bf16 outputs move, each by at most
one bf16 ulp plus that slack, and the maximum error against FP64 is the
same as the exact kernels' (1.2e-4 local, 3.0e-5 global on the probe).

At the model level the change is not small, and that is a property of the
model rather than of this kernel: `tests/test_inkling_forward` (515-token
MQ85GB prompt) measures the tensor-core prefill against the exact prefill
at relative RMS 0.127, max 3.11 on the last-position logits, same greedy
token, 58% of the committed KV/convolution bytes changed. A control
experiment replaced the exact grouped kernel's final `__fdiv_rn(value,
total)` by `value * __frcp_rn(total)` (a one-ulp-class reordering of a
single rounding) and measured the same prefill-versus-decode deviation:
relative RMS 0.063 at 16 tokens and 0.105 at 515 tokens, max 1.2 / 2.2.
Bf16 residuals and top-6-of-256 routing amplify any reordering to that
level, so a logit tolerance cannot separate this kernel from the exact one
better than the model's own chaos floor. The forward test therefore keeps
its byte-exact prefill/decode/chunk parity on the default path and bounds
the opt-in tensor-core prefill at relative RMS 0.5 with the same greedy
token. Because the amplified deviation also changes 64-token greedy
continuations (see the measurement), the path is shipped opt-in rather
than as the default: making it the default is a one-line polarity change
once the project decides to give up byte-exact prefill/decode parity for
this model.

**Probes** (`scratch/inkling-perf/r25-r27/attn/`, 1024-row chunk, times in
ms; release = the grouped two-key kernel):

| Shape | Release | half MMA, 16-key steps | half MMA, 64-key steps | bf16 MMA, cp.async ring, single bf16 P | bf16 MMA, cp.async ring, bf16-pair P (retained) |
|---|---:|---:|---:|---:|---:|
| local, 3072 committed | 2.39 | 0.80 | 0.82 | 0.90 | 0.92 |
| global, 3072 committed | 13.6 | 2.36 | 2.29 | 1.92 | 2.11 |
| global, 7168 committed | 33.5 | 4.93 | 4.75 | 3.14 | 3.65 |

Rejected on the way: half-precision K/V/P (25% of outputs move and the
bf16→half conversion has a range caveat), single bf16 probabilities (70%
move), register-staged K/V prefetch and a 16-key softmax step (the
scheduler sank the loads under the 255-register cap), mask-free interior
tiles and a one-tile-ahead bias pipeline (spills), a lazy output rescale
(no gain), four heads per CTA and two CTAs per SM (spills). ncu on the
first variant: 22% issue-active with long-scoreboard the top stall at
eight warps per SM; the retained kernel's remaining cost is the fp32
softmax plus the bias loads (~10% at depth, 45% on local layers, where the
4-byte bias table read per score is now the largest term).

**Measurement contract.** `ds4-perf compare` gains `--logit-rel-rms X`
(docs/ds4-perf.md): the per-logit `atol/rtol` and whole-sequence check of
the exact contract is replaced, when X is positive, by a bound on each
frontier's relative RMS deviation plus argmax equality, with greedy
sequences allowed to diverge and still reported. The bound is X at 1,024
tokens and grows with `log2(ctx)/10` (1.1X at 2K, 1.3X at 8K, 1.6X at
64K): reordering noise grows with the attended length while the model's
amplification saturates, and the one-ulp control measured 0.063 at 16
tokens and 0.105 at 515 tokens, log-linear in length. This round uses
X = 0.12, the control's floor at 1K rounded up, so the 8K bound is 0.156
and the 2K bound 0.132; relaxing `atol` instead would have needed
`atol ≈ 2` (max logit difference 1.70 at 8K) and would still fail on the
sequence check. Exact rounds keep the default contract.

**Measurement** (opt-in path on versus the default, three unprofiled fresh
processes per side, `--logit-rel-rms 0.12`):

| Input | Prefill default → opt-in (tok/s) | Gain | Decode (tok/s) | Relative RMS (bound) | Argmax | Greedy 64 | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| 8,192 | 464.26 → 501.87 | +8.10% | 13.68 → 13.68 | 0.114 (0.156) | 3/3 equal | 3/3 diverge (from token 2) | Improved |
| 2,048 | 487.23 → 504.38 | +3.52% | 17.12 → 17.13 | 0.250 (0.132) | 3/3 differ | 3/3 diverge (from token 0) | Incorrect |

Three throughput samples per side:

- 8,192 default: 465.98 / 464.26 / 463.52; opt-in: 503.53 / 501.87 / 500.55 tok/s
  (envelope -7.95..-6.91%).
- 2,048 default: 490.33 / 487.23 / 485.59; opt-in: 507.75 / 504.38 / 504.27 tok/s
  (envelope -4.36..-2.76%).

At 8K the frontier keeps its argmax (298, margin 4.06 over the runner-up
in the default logits) and the greedy continuation diverges from the
third generated token. At 2K the default frontier is a three-way near tie
(196349 at 12.495, 272 at 12.470, 1320 at 12.412) and the opt-in path
ranks 1320 first (12.846), so the argmax check fails on a 0.025-logit
margin; its relative RMS, 0.250, is also 1.9× the 2K bound. The forward
test on the 515-token fixture measured 0.127; the control floor of 0.12
was fixed from two prompts, and the opt-in deviations across prompts
(0.10–0.25) spread wider than that, so the 2K result is as much a
statement about how tightly the floor can be estimated from two points as
about the kernel. Under the requested contract the round stands as: 8K
Improved, 2K Incorrect, opt-in only; the default path is byte-identical
to round 26 (`DS4_INKLING_ATTN_HMMA` unset changes no kernel).

Under the exact contract the same 8K run reads: 599,457 of 1,200,348
logits beyond `atol/rtol 1e-4`, max difference 1.70, three of three
greedy sequences differ (recorded in `r27-default-on/`, measured before
the switch polarity was flipped, 464.07 → 501.10 tok/s).

Native gates in the same run: `tests/test_inkling_batch`,
`tests/test_inkling_forward` (byte-exact prefill/decode/chunk parity on
the default path; opt-in prefill relative RMS 0.127, same greedy token)
and the ds4-perf unit tests (19, including the relaxed-contract test)
pass. Benchmark binary SHA-256:
`32fcaaf6708a6295f5aa1f6fcf49af2c3eaecdb55d8f50796e6290a77d085524`.

## Cumulative

Round-26 binary (`2377a4d`, the same build as round 27 with the opt-in
switch unset), both retained exact switches off versus the default:

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 410.68 → 464.09 | +13.0% | 13.69 → 13.68 | Improved |
| 2,048 | 428.63 → 486.86 | +13.6% | 17.13 → 17.13 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 412.15 / 410.68 / 409.67; default: 466.68 / 464.02 / 464.09 tok/s.
- 2,048 control: 430.92 / 428.63 / 427.88; default: 490.32 / 486.86 / 486.10 tok/s.

First-token latency stays within 0.2%. Against the round-24 report's
413.41 tok/s (8K) and 430.61 tok/s (2K), the default path now runs 464.09
and 486.86 tok/s (+12.3% / +13.1%); against the round-18 report (328.16 /
342.31) the three campaigns together give +41.4% and +42.2%, and against
the campaign's round-0 baseline (230.7 tok/s at 8K) +101%. With the opt-in
tensor-core attention the 8K figure is 501.87 tok/s.

Two additional unprofiled `ds4-bench-perf` launches on the default path:

- 2,048: 491.13 / 488.75 prefill tok/s, 17.13 / 17.13 decode tok/s.
- 8,192: 467.30 / 465.14 prefill tok/s, 13.68 / 13.68 decode tok/s.

## Final validation

- `tests/test_inkling_batch` passes on every round binary, including the
  new kill-switch parity blocks (`DS4_INKLING_NO_SHARED_PIPE`,
  `DS4_INKLING_NO_IQ2_SLAB`) and the aligned IQ2 slab cases.
- `tests/test_inkling_attention` passes with both extents: the exact
  baseline is reproduced byte-for-byte by chunks 1–8192 on the default
  path and by the `DS4_INKLING_NO_ATTN_GROUP` /
  `DS4_INKLING_NO_ATTN_TRANSPOSE` / `DS4_INKLING_NO_ATTN_PAIR` rollbacks;
  the opt-in tensor-core baseline is reproduced byte-for-byte by every
  16+ row chunking, stays within its FP64 bound at the probe positions and
  within one bf16 ulp plus slack of the exact baseline everywhere (0.12%
  of local and 0.62% of global outputs move); captured replay, rejected
  suffixes, ring wrap and the 16-row window ending at `UINT32_MAX` pass on
  all three kernels. Timing at 7,680 committed keys, 512 rows: local
  597 / 1,185 / 3,473 µs and global 2,053 / 17,806 / 97,313 µs for
  tensor-core / grouped / per-head.
- 515-token `test_inkling_forward`: `max_abs=0`, committed KV/convolution
  exact, chunks 2/3/7 exact on the default path; opt-in prefill relative
  RMS 0.127 with the same greedy token.
- `tests/test_inkling_session --memory-quotes` on the final binary:
  campaign context 8257 at the default cap 1024 quotes 973,649,152 bytes
  (MTP off) and 1,790,212,608 bytes (MTP on); at cap 8192, 5,576,422,656
  and 11,348,081,152 bytes. Unchanged from round 24 (the attention staging
  copy is a sticky scratch allocation outside the session quote).
- `tests/test_inkling_session` with the restarted MQ85GB + eight-layer
  MTP-BF16 owner: lazy allocation, no-op/extend/decode/reset/rewind parity
  at context 32 (estimate = measured 164,265,472 bytes), native MTP eight
  cycles with 18 greedy tokens and exact target logits/KV/convolution,
  image and audio identity/invalid-input state.
- `cargo fmt --all -- --check`, workspace clippy, `cargo test -p ds4-perf`
  (including the relaxed-contract unit test) and all-target workspace
  check pass.
- Retained commits: R25 `30037de`, R26 `2377a4d`, ds4-perf contract
  `c7ce810`, R27 (opt-in) `9a6de80`.

## Rejected probes

All probes are model-free production shapes under
`scratch/inkling-perf/r25-r27/` (rebuilt `r19-r21/iq2/` for the IQ2
diet), byte-exact unless noted.

- IQ2 lean-tile register diet and next-step prefetch (`iq2-build*.log`,
  `pk_diet_kernel`): 19.5–24.1 ms against 18.3 ms; see the IQ2 section.
- Shared Q8 ring depths of three, six and eight stages (within 2% of
  four), twelve warps per CTA (168 registers, spills) and three rows per
  warp (spills); the two-row pipelined down tile at four CTAs (4% slower
  than round 23).
- Attention query tiles of two, four and eight consecutive queries
  (`r22-r24/attn/`, `at_qtile_kernel`): exact only for the global extent
  and slower there (17.75 vs 13.70 ms); local windows need queries four
  apart.
- Tensor-core attention variants (`attn/`, not exact): half-precision
  K/V/P, single-bf16 probabilities, register-staged K/V prefetch, 16-key
  softmax steps, mask-free interior tiles, one-tile-ahead bias loads, lazy
  output rescale, four heads per CTA, two CTAs per SM; the table in the
  round-27 section gives the timings.

## Evidence

Raw commands, hashes, guards, proofs, traces, ncu reports and comparisons
are retained under `scratch/inkling-perf/r25-r27/`: `ncu-*.txt` (entry
and probe profiles), `shared/`, `attn/` (variant probes, `run*.txt`),
`r25/`, `r26/`, `r27/` (opt-in measurement), `r27-default-on/` (the same
kernel measured as the default under the exact contract before the
polarity flip), `cumulative/` and `final-checks/`. The workload manifests
`workload-{8k,2k}.json` rehash the round-19 files with the restarted
owner's IPC manifest. These are MQ85GB text-prefill measurements, not
source-model, long-context, media-performance or MTP throughput
qualification; the round-27 contract is qualified only by the
attention fixture, the 515-token forward bound and the 8K/2K frontiers
reported above.
