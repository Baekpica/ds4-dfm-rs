# Prefill and decode optimization playbook

What the 2026-09 optimization campaigns on this engine taught, written down
so the next campaign (another family, another host) starts from the method
and the failure modes instead of rediscovering them.  Everything below was
measured on one DGX Spark (GB10, `sm_121`, 128 GB unified memory, ~240 GB/s
LPDDR5X, ~90K random 4 KiB IOPS from the NVMe) with mixed-quant MoE models
served from a resident VMM weight owner.  The numbers are quoted so the
weight of each lesson is visible; they do not transfer to other hardware,
only the reasoning does.

Campaigns this distills (all in `docs/` and `scratch/`):

| campaign | model | result |
|---|---|---|
| 2026-09-01 / 09-02 prefill (9 + 5 rounds) | Qwen3.8-Flash-Next Q5 + SSD-PLE | 8K prompt 497 -> 604 -> 911 tok/s |
| 2026-09-02 / 09-03 decode (8 rounds) | same | 93 -> 52 ms per MTP step, 18 -> 31 tok/s |
| 2026-09-04 long context (5 rounds) | same | cold 64K 530 -> 1371 tok/s, cold 196K 1052 -> 1288 |
| 2026-09-05 K2-Horizon-375B MQ87 (12 prefill + 4 decode rounds) | K2 IQ1/IQ2 experts | 8K prefill 287 -> 645 tok/s (+125 %), decode 5.5 -> 13.3 tok/s |
| 2026-09-06 prefill (3 rounds) | Qwen | cold 8K 1215 -> 1362 tok/s (+12 %), 64K +4 % |
| 2026-09-06-r4 (D2R K=2560 + HC q8 + o_proj K=6144) | Qwen | cold 8K 1353 -> 1432 tok/s, 64K 1431 -> 1555 |

The one-line version: **almost every large win came from removing work the
graph did not need (idle time, fallbacks, repeated transformations, redundant
round trips), and the kernel-tuning rounds came last and paid least.**  This
is the order the engine's `AGENT.md` prescribes and the campaigns confirmed
it round after round.

---

## 1. Method: one round

A round is one change, measured against the previously adopted binary under
identical conditions, kept only if the gain is stable and the numerics are
accounted for.  Rounds that skipped a step produced the wrong conclusion
every time they did.

1. **Profile the whole request, not a kernel.**  `nsys --trace=cuda` on the
   production shape (an 8K cold prefill, a 64K cold prefill, a decode window
   of ~10 tokens).  Export to sqlite and reduce it yourself: per chunk, GPU
   busy vs idle, the *gap before each kernel* (idle is attributed to the
   kernel that was waiting), then kernel totals with launch counts and
   average duration.  `scratch/qwen-prefill-opt-20260906/{nsys_chunks.py,
   kern.py}` are the two scripts; they run in seconds.  Rank gaps and kernels
   together.  A 1.06 s gap before the PLE gather kernel out of a 6.6 s
   prefill was the biggest single item of the last Qwen campaign and no
   kernel view shows it.
2. **Attribute before designing.**  Launch counts and average durations tell
   you whether a bucket is one slow kernel or many small ones (334 dense MMQ
   launches per chunk, 514 quantize launches, 387 sanitize passes).  For a
   kernel, `ncu` on a model-free probe (the fixtures have `DS4_*_PROFILE_*`
   modes that launch the production shape on synthetic weights) gives the
   stall reasons; on this host `ncu` runs only with the weight owner down.
3. **Change one thing.**  Two changes in one round cost a full re-measure
   when one of them regresses (the 09-06 round 3 folded a SwiGLU epilogue and
   a fused sum together; the epilogue lost 0.7 % and hid the sum's +1 %
   until a kill-switch run separated them).
4. **Prove the numerics on a fixture first** (section 5), then run the
   full-model gate.  Fixtures are fast (seconds) and catch layout bugs the
   full model only shows as an illegal address at boot.
5. **A/B on fresh processes, same hour, medians of three.**  One fresh
   worker per run against the resident owner, cold KV directory, the same
   corpus slice, `thinking` disabled, non-streaming so `usage`/`timings` are
   present.  Throughput drifts 1-4 % between hours on one binary (thermal,
   page cache, owner state), so the baseline is re-run in the same session
   whenever a difference is under ~2 %.  Never build while timed runs are
   on; the compile steals the cores the page workers use.
6. **Keep a kill switch** for every new execution path (`DS4_*=0/1`, read
   once per process).  It is the attribution tool: same binary, path on vs
   off, and it is what turns "the text changed" into "the text changed
   because of X" (section 5.4).
7. **Commit with the evidence**: what was slow and by how much, why, the
   contract (bit-identical / fp32 reorder / value parity), the A/B numbers
   with the shape and corpus, and the `Tests:` that actually ran.

What counts as the metric matters as much as the protocol:

- Prefill: tokens per second of a **cold** prefill of a fixed shape.  A
  repeated prompt on a warm worker is not cold; the SSD-PLE page cache and
  the disk-KV cache both hide the cost you are trying to measure.  Say which
  corpus: a prompt whose n-grams hit the page cache and one whose n-grams
  miss it differ by 30 % on the same binary.
- Long context: the single-shot `--ctx-start N --ctx-max N` shape (one cold
  N-token prefill, the production TTFT of a long request) and the
  incremental sweep (2K steps on a warm session) answer different questions;
  the sweep cannot exercise the next-chunk lookahead because the next chunk
  is not known.
- Decode: **milliseconds per step**, not tokens per second, whenever
  speculative decoding is on.  tok/s = tokens-per-step / ms-per-step, and
  tokens-per-step moves with the draft acceptance rate, which moves with the
  text, which moves with any fp32 reorder (a near-tie token flips ~25 tokens
  into the standard prompt).  93 -> 52 ms per step is a kernel result;
  18 -> 31 tok/s is that plus acceptance luck.

---

## 2. Prefill: where the time went and how it was removed

Ordered as `AGENT.md` orders it: unexpected fallback paths, repeated
transformations, redundant memory traffic, routing and indexing overhead,
launch fragmentation, fast-path utilization, and only then kernel tuning.

### 2.1 GPU idle time is the first target

If the profile shows idle, no kernel work matters until it is gone.

- **Find the wait, then the bug behind it.**  The 09-04 trace put the GPU
  idle 3.8-12 s per 8K chunk before the PLE gather kernel.  The obvious
  reading ("SSD is slow") was wrong: the page cache's victim scan stopped at
  the *first* evictable way instead of the least recently used one, so a
  prefetch's pages were evicted by the next request for the same set and a
  chunk re-read a third of its pages at gather time.  True LRU + 16 ways
  took the 64K wait from 59.5 s to 3.8 s (530 -> 1163 tok/s).  More I/O
  workers had made it *worse* (64 workers: 339 tok/s) because faster
  completion produced more READY victims for the broken scan.  A knob that
  makes things worse when turned up is a bug, not a limit.
- **Overlap I/O with compute at the granularity the pipeline allows.**  The
  next chunk's pages are queued right after this chunk's rows are gathered
  (layer 1), so 47 layers of compute hide the reads (the "lookahead").
  Cache sizing follows from the overlap: one chunk's working set per
  in-flight stream, two for two alternating banks.
- **The first chunk has nothing to hide behind.**  Opening every prompt
  with a short chunk (2,048 rows) exposes only its own reads and hides the
  full-size chunk behind it: cold 8K +9.1 %.  Two boundaries: a short
  *trailing* chunk is a net loss (a 256-row chunk ran its 48 layers at low
  occupancy and hid few reads: -7 % on a 2,304-token prompt), so the split
  applies only when the trailing chunk is at least as long as the opening
  one; and on prompts whose pages are already cached the opening chunk costs
  its own inefficiency (~1.7 %).  Chunk-size changes also change GEMM shapes
  (cuBLAS algorithm choice) and therefore near-tie tokens; that is expected.
- **Queue depth is not the lever when the pipeline is.**  A Linux-AIO
  rewrite of the page workers (depth 32) lost 14 % on the cold-PLE prompt
  although the raw device scales with depth: deeper queues made the
  head-of-line rows the gather waits on arrive later.  Fix ordering
  (consumption order, targeted wake-ups) before adding concurrency.

### 2.2 Fallback paths: the biggest single wins

A quantization format or a shape that the fast tier does not admit falls to
a generic kernel, and the fallback is usually 2-10x slower than the tier it
missed.  Read the engine's "engaged path" log lines and grep the dispatch
predicates before touching any kernel.

- **Rectangular schedules over ragged work.**  MoE prefill routes ~5-20 rows
  per expert; a `[expert, max-bucket]` grid launches mostly empty tiles.  A
  compact worklist (one persistent kernel, grid = SM count, tiles of
  8/16/32/64/128 columns chosen per expert bucket inside the kernel) is the
  right shape: K2 IQ2_XXS down 12.9 -> 7.0 ms, IQ1_S gate/up 9.8 -> 5.0 ms,
  IQ1_M gate/up from a per-token MMVQ loop to an MMQ tile +55 % end to end.
- **Shapes that are "almost" the tile.**  An expert-down with K = 640 is not
  a multiple of the 256-wide K-quant super-block, so the recipe stores a
  512-column main + 128-column tail and the engine ran the tail as a separate
  F32 accumulate.  Folding the tail into the MMQ K loop as one extra
  half-iteration (+14 %) and then running the dense K = 640 shared-expert
  down through the same fused-tail entry as a one-expert MoE (+17 %) were
  the two largest prefill rounds of the 09-02 campaign.  When a dense
  operation can be expressed as a routed one with an identity map, the
  routed tier's kernels are usually the better ones.
- **Every projection on the quantized-activation tier.**  PLE key/value and
  QSA k/v pairs ran on Q8_0 DP4A kernels while every other projection used
  the Q8_1-activation MMQ tier; moving them (+11 %) changed numerics within
  the MMQ tolerance, so this is an arithmetic round with a contract, not a
  scheduling round.
- **Chunk size is a dispatch parameter.**  K2 at 512-token chunks saw ~21
  rows per expert and 16x the per-chunk launches of an 8K prefill; 1,024
  doubled the rows per expert (+20 %).  It also changed the greedy IDs (56
  of 64), which is why it was rejected once and adopted only with the
  revised gate (section 5.3).
- **Tiers that exist but are never entered.**  Qwen's dense Q8 projections
  missed the aligned D2R tier because the weight owner's repack rule
  admitted only `dims[0] % 1024 == 0` (K=2560 qkv/z/q).  Admitting the
  D2R prefill contract moved them: cold 8K +2.8 %, 64K +5.1 %.  o_proj
  (K=6144) already had the artifact via `%1024` and still missed the
  dispatch cap of 4096 (kept for DeepSeek K=8192, 0.81x vs mmq); raising
  the cap to 6144 only: +1.3 % / +2.5 %.  The "engaged path" log line
  for a tier is worth checking against the tensor list once per family.

### 2.3 Repeated transformations

The same activation quantized, converted or permuted more than once is pure
waste and usually bit-identical to remove.

- Pair kernels that share one expert map and one Q8 activation for gate/up
  (K2 +4 %, Qwen +11 % in combination with the worklist).
- Quantize once per **token**, not once per assignment slot: the routed
  gate/up quantize gathered every token row top-k times (0.84 GB per 8K
  layer).  Quantize the compact rows and scatter the 144-byte Q8 blocks into
  the expert-sorted layout the tiles stream (+1 %; see 2.7 for the variant
  that lost).
- Hyper-connection rows converted to BF16 once by the norm instead of per
  consuming GEMM (+3.3 %), later the norm storing only BF16 rows plus one
  scale per lane, with F32 recomputed from the hyper input where needed.
- The Qwen HC mix is one F32 tensor consumed by qkv+z or q+index; emitting
  its Q8 once through the existing norm-q8 registry retired the per-GEMM
  quantize (bit-identical, +0.8 % / +0.9 %).  The D2R preq entry had to
  accept the same K % 128 as the launch; a leftover %1024 gate kept the
  published buffer unused.
- Counting-sort expert id maps instead of the warp-scan builder
  (bit-identical, +7 % with the sanitize drop).
- Trigonometry tables: K2's QK-norm/RoPE kernel computed `pow/fmod/cos/sin`
  in double per (head, pair) per launch; on GB10 FP64 runs at 1/64 rate.  One
  (cos, sin) table per chunk position: 40 ms per 512-token chunk gone,
  12 us per decode launch gone.

### 2.4 Redundant memory traffic

At ~240 GB/s every [rows x hidden] F32 round trip at 8K rows costs ~0.35 ms
per read or write; a MoE layer had a dozen of them.  Count bytes per layer
before and after; the trace's kernel time for bandwidth-bound kernels is
bytes / 240 GB/s within 10 %.

- **Sanitize passes.**  A standalone non-finite pass over every MMQ output
  is a full read + write; when every consumer already zeroes non-finite
  values at read (`moe_sum`, the weighted SwiGLU), the pass is redundant.
  K2 Prefill 9 removed 58 launches per chunk; after D2R retired its own
  outputs Qwen still carries 303 dense ones on a 6,144-row chunk (0.8 %).
- **Pack copies.**  The expert-down read its 512-column input from a packed
  copy of the 640-wide SwiGLU rows although the same entry already read the
  128-column tail in place with a row stride.  Reading both halves in place
  removed 378 MB per layer (+0.8 %).  Whenever a kernel takes a stride,
  strided reads are free; a pack kernel is a smell.
- **Fused epilogues on elementwise chains** (safe class): `moe_sum` +
  shared-expert sigmoid gate + residual add in one kernel (two round trips
  per layer, +1 %); residual + next norm in one kernel; SwiGLU producing the
  quantized down input directly where the layout allows.
- **Keep F32 intermediates out of memory.**  The hyper-connection mix used
  to write [rows x 10240] F32 logits from a cuBLAS GEMM and re-read them in
  a mix kernel (~3 GB per sub-layer); forming the mix-up logits on the
  tensor cores inside the mix kernel and applying the sigmoid in the
  epilogue cut it to ~1.7 GB (+7 %).  The 1.6 GB per-bank QSA score scratch
  went the same way: a fused attention kernel with an online softmax never
  writes scores (104 + 64 -> 67 ms per layer).
- **Recurrent state in registers.**  The GDN recurrence did two loads and a
  store per state element per token; keeping 32 rows per thread in registers
  for the whole chunk halved the kernel (33 -> 17 ms), with `fmaf` written
  explicitly so the contraction matched the old code bit for bit.

### 2.5 Launch fragmentation

Per-launch cost on this host is ~5-10 us of GPU gap plus host time; a Qwen
layer had ~62 launches at decode and an 8K prefill chunk ~4,000.  Pairs
(2.3), persistent worklists (2.2) and fused epilogues (2.4) all cut launches
as a side effect; the rounds that targeted launches directly:

- One launch per layer for the routed vec tier instead of one per token
  (Qwen R15, -7.6 % per decode step) — a per-token split introduced for
  row-stability had silently split the assignment-major down call too.
- Paired 320-row and 4-row BF16 projections (R12), warp-parallel router
  selection replacing a serial kernel (R13).
- Decode attention over the whole context in one launch was the wrong
  direction: split-K across the context (section 3.4) adds a combine launch
  and still wins 78 % because the single launch used 1 of 48 SMs.

### 2.6 Kernel tuning, last

Only after the graph is clean.  The signals that led to each tuning round:

- **Shuffle-bound reductions.**  The QSA block scorer reduced every lane's
  4-dim partials with five shuffles per head (20 shuffles per 16 FMAs, ~12 %
  of FMA peak).  Eight lanes per block holding 16-dim query slices in
  registers, a transpose tree for the four head sums, and a 132-float tile
  stride for conflict-free `LDS.128`: 12.0 -> 4.2 ms.
- **Block barriers in a per-token loop.**  The GDN recurrence paid four
  `__syncthreads` per token across four warps; one warp owning eight
  columns with xor-shuffle cross-group sums, a warp-private shared Q/K copy,
  and the next token's operands prepared one iteration ahead: 16.6 ->
  7.1 ms (fp32 reorder; fixture 1.5e-8).
- **Long-scoreboard stalls in a K loop.**  ncu on the worklist MMQ kernels:
  255 registers, one block per SM, 61-74 % issue slots idle, the dominant
  stall the global loads issued at the top of each K iteration and consumed
  immediately.  Prefetching the next block's raw bytes into registers after
  expanding the current tile, and staging both activation halves with
  `cp.async` one iteration ahead (4 -> 2 barriers per iteration): +5.3 %,
  bit-identical, for tiles up to 64 wide (a 128-wide two-stage buffer no
  longer fits the 100 KB SM).
- **Fragment loads.**  `ldmatrix` / `ldmatrix.trans` for the HMMA attention
  fragments plus a BF16 next-tile register prefetch: +2.4 %.
- **Occupancy over reuse for small per-thread state.**  GQA decode attention
  sharing one KV row across six query heads spilled registers and lost
  25 %; the same idea over two heads gained 25 %.  Check the register count
  after every "reuse across heads/rows" change.

### 2.7 What did not work, and the rule each one taught

| attempt | result | rule |
|---|---|---|
| per-column token map inside the MMQ tile so the y tile reads the token-compact activation directly | gate/up kernel +22 %, prefill -3 %, despite 0.6 GB less DRAM traffic per layer | the tile's contiguous 18 KB y loads are the hot loop; DRAM bytes saved outside the loop do not pay for an indirection inside it |
| weighted SwiGLU applied in the up launch's write back (reads the stored gate at the same index) | up launch 4.49 -> 5.84 ms, more than the 0.095 s pass it removed | an MMQ write back may *store* scattered, it must not *read* scattered; fuse elementwise work where the reads are coalesced |
| Linux AIO page workers, depth 32 | cold-PLE prompt -14 % | deeper queues without consumption-order scheduling delay the rows the consumer waits on |
| QSA reduce over 6-head groups; generic Q8 MMVQ with 8 warps; IQ1_M 4-row tile | no stable gain or slower | occupancy and register pressure; measure before assuming reuse wins |
| balanced halves for prompts between one and two opening chunks | -4 % vs the split, -8 % vs one chunk | a chunk under ~2K rows runs the MoE tiles at low occupancy; below that size the I/O overlap is worth less than the efficiency lost |
| batched IQ1_M MMVQ across tokens (K2 Prefill 1) | isolated parity passed, first token changed at the frontier | an isolated-kernel parity test is not a graph gate; the frontier logits and greedy IDs are |

---

## 3. Decode: a different workload

Decode at width 1-2 is bandwidth- and launch-bound, not compute-bound.  The
Qwen budget at the end of the campaign, per ~52 ms MTP step: dense GEMVs at
~240 GB/s (the 248,320-row LM head alone 2.7 ms per call), the two-row
routed-expert traffic, ~62 launches per layer, host gaps ~2 ms per pass.
Prefill kernels are the wrong kernels here and vice versa; dispatch on width.

### 3.1 Width-dependent dispatch

- Decode widths (1..8 rows, including the two-row verify pass and the
  two-bank lane) belong on vector kernels (`mmvq`, warp-per-row aligned Q8),
  not on MMQ tiles: moving top-10 routing from ten-row MMQ tiles to the vec
  tier was -6 % per step (Qwen R9); K2's aligned Q8 dense-vec path serves
  every dense projection at width 1.
- The vec tier's assignment cap is a policy enum, not a magic number
  (`DS4_ROUTED_VEC_MAX_ROWS`), and the split between "one launch per
  assignment" and "one launch per token" must be checked per call site
  (R15).
- Tiny GEMVs deserve their own kernels: a warp per output row with 16-byte
  BF16x8 loads replaced a naive per-thread kernel on the hyper-connection
  path (1,067 launches per step, -8 %).

### 3.2 Row stability is a contract, not an optimization

Speculative decoding verifies a draft by running [committed, draft] as one
two-row pass and compares its row-0 arithmetic with what a one-row pass
would have produced.  That only works if every kernel's per-row result is
independent of the batch width:

- widths 2..8 use the same aligned Q8 NC kernel as width 1 (the raw `mmvq`
  partitions by column count and differs at 2e-7);
- multi-expert rows on the routed vec tier launch one token at a time;
- the F32 router uses a row-stable GEMV, not cuBLAS (9e-4 apart).

The gate is `tests/test_qwen4exp_verify`: 48 accept/reject steps, logits
and every recurrent state bit-exact against the serial oracle.  Every decode
round since has had to keep it, and it has caught a real regression (the
per-token split in R14 that R15 undid was found by measurement, but the
state parity of R14 itself came from this test).  For MTP on compressed-KV
families the contract is token-level at width > 1 (`AGENT.md`, "MTP /
compressed-KV decode rules"): the committed cache values are inherently
fp-noisy across widths, the committed token stream and frontier counts are
not.

### 3.3 Speculative decoding: never verify twice

Before R14 the drafter's token was verified by a *second* one-row target
pass, so MTP never saved a pass and an auto-quenched run was 5 % faster than
the speculative one.  The fix is protocol, not kernels: run the committed
token and the draft as one two-row pass, checkpoint the recurrent frontiers
(GDN state, PLE conv state) inside the same kernels after row 0 (a template
instantiation, because a branch in the hot loop cost 3.5 % prefill), and
roll back a rejection by pointer swap plus a hash re-derivation; position-
indexed caches only reset their lengths.  67.5 -> 56.7 ms per step, -16 %.
Measure the speculative path with the quench heuristic in mind: under nsys
the per-launch overhead trips it after 12 cycles.

### 3.4 Attention at decode

- **Split the context across blocks.**  One block per head over a 32K
  context used 1 of 48 SMs; split-K across context chunks with a combine
  kernel: K2 decode +78 % (the largest decode round anywhere in these
  campaigns).  Qwen's sparse-attention reduce got the same treatment at
  rows <= 8 (128-slot chunks + combine, -11 % per step).
- **Share KV across GQA heads only as far as registers allow** (2.6).
- **Run the decode-attention pair kernel per chunk**, not once over the
  whole context, when the context is split (K2 Decode 3).

### 3.5 The rest of the step

- Two-bank lanes: run the two decode-ready banks as one two-row operation
  wherever the arithmetic is bank-independent (embedding, hyper-connection,
  output projection, GDN recurrent grid, PLE gather, Q5 tail, routed main);
  each was +1-9 % on a two-request microbenchmark and each has a fixture
  that shows the two-row result bit-identical to two one-row calls.
- Host gaps: ~2 ms per pass is launch issue and synchronization; captured
  graphs help only if every captured kernel reads live substrate state
  (`AGENT.md`, "CUDA captured-decode rules") — by-value arguments replay
  stale.
- The LM head (248K rows) is 2.7 ms per call at bandwidth; the only lever
  there is the artifact (aligned Q8) and not calling it twice.

---

## 4. Memory: the constraint that shapes everything on a unified-memory host

- One resident model, one owner, one worker.  The owner holds the weights
  (80 GiB here) through VMM/IPC; workers import them and add their graphs
  (24-30 GiB for a 196K-262K two-bank Qwen worker).  Every measurement
  worker is a fresh process against the same owner, so the A/B cost is
  seconds, not the 5-minute weight load.
- Additive derived artifacts (aligned Q8 repacks) live in the owner and are
  cheap (0.94 GiB for 160 tensors); artifacts that *replace* raw residency
  (IQ2/Q2K SoA) change the memory plan.
- Score/scratch buffers are sized to the width that uses them: the QSA
  score scratch shrank from prefill-width to decode-width once prefill went
  fused (graph 9.86 -> 8.33 GiB at 64K), which is what made the 262K
  two-bank worker fit.
- Bounded caches for SSD sidecars are sized from the overlap policy (one
  in-flight chunk per stream), not from free memory; oversizing hides
  eviction bugs instead of fixing them.

---

## 5. Numerics: the contract per class of change

Correctness before speed is only useful if "correct" is defined per change.
Three classes were used; every commit names its class.

### 5.1 Bit-identical (scheduling, fusion of elementwise chains, layout)

Same quantized values, same tiles, same dot-product order, same expression
in the fused kernel: the fixture compares byte for byte (`memcmp`), and the
full-model gate (logits + greedy IDs) is exact.  Things that break
bit-identity silently:

- **Contraction.**  `a + b * c` in one kernel is an FMA; the same math in
  two kernels is a rounded product then an add.  Write `__fmul_rn` /
  `__fadd_rn` (or `fmaf` explicitly, if the old code was contracted) in the
  fused kernel and prove it on the fixture.
- **Different translation units.**  `expf` and division under
  `--use_fast_math` are the same intrinsic only if both TUs use the same
  flags; the Makefile does, but check when the epilogue moves from
  `ds4_cuda.cu` into the vendored MMQ TU.
- **Batch width.**  `mmvq` at N=1 and N=2 partition by column count and
  differ at 2e-7; MMQ tile widths do not change the per-element order.
- **Chunk size / row count.**  cuBLAS picks algorithms per shape; the
  opening chunk changed the greedy continuation of a near-tie prompt while
  every kernel stayed bit-identical (the kill switch reproduced the old
  text).  This is not drift, but it is a shape change to document.

### 5.2 fp32 reorder (accumulation order changes, fused attention)

Fixture tolerance against the CPU/F64 reference of the order of 1e-7 (max
abs) to 1e-8, self-parity of the chunked vs single-shot path bit-exact, the
two-row verify gate exact.  Examples: the fused QSA attention (1.3e-7), the
warp-independent GDN loop (1.5e-8), the block scorer (identical top-512
selection).

### 5.3 Value parity (a different quantization path or tier)

For rounds that change what gets computed (Q8_0 DP4A -> Q8_1 MMQ
activations, D2R vs mmq fold order, chunk-size changes on K2): a kernel
parity row against F32 at the tier's known tolerance (~2.5e-3 rel RMS),
the same-binary kill-switch A/B, the comparator band on the frontier logits
(same argmax, top-10 overlap >= 8/10, KL <= 0.05, rel RMS <= 0.11), a
byte-identical repeat of the new path, and the contract written into the
doc.  K2's chunk-size round was rejected under the byte-identical gate and
accepted under this one with its drift measured (rel RMS 0.0636).

### 5.4 Reading "the text changed"

A greedy continuation that differs after N tokens is a *symptom* with three
causes: a bug, a class-5.3 change, or a near-tie token under a class-5.1/5.2
change.  Tell them apart with the kill switch on the same binary: if the
old text comes back with the switch off and the fixture is bit-identical,
it is the shape (5.1); if the fixture shows the tier tolerance, it is 5.3
and needs the band; if neither, it is a bug.  Do not compare texts across
sessions or artifacts, and never quote tok/s from a run whose text differs
without saying so.

### 5.5 Fixtures: make them look like production

- The Qwen MoE fixture routed 1,200 rows over 8 experts (150 rows each:
  128-wide tiles plus a 32-wide tail) and passed a kernel that faulted in
  production, where 512 experts see ~5 rows each and the 8- and 16-wide
  tiles carry the work.  Add the production routing regime (a few rows per
  expert) to every MoE fixture.
- Model-free fixtures register weights as host memory; a kernel that
  re-reads a weight tile per row block runs 2.5x slower there than in the
  served model.  Cache the fixture weights on the device
  (`ds4_gpu_cache_model_range`) before timing.
- Include a non-finite input in every fixture for a kernel that guards at
  read; the guard is part of the contract.
- A fixture that passes is not a full-model gate.  Boot the real model
  (the 256-token prewarm chunk found the narrow-tile bug in seconds).

---

## 6. Host and tooling pitfalls that cost hours

- `pgrep -f`/`pkill -f` with a pattern that appears in the calling shell's
  own command line matches the shell.  Keep patterns out of inline heredocs;
  check the PID's `ps` line before acting on it.
- The engine's instance lock refuses a second `ds4` process: stop the
  production worker before any `ds4-bench` run against the same owner.
- tmux starts commands from the server's environment; per-run engine
  switches ride in the command string (`EXTRA_ENV`), and a switch that
  silently did not reach the worker re-measures the baseline.
- Unix socket paths are limited to ~108 bytes; a manifest under a long
  scratch path makes the weight owner exit at "broker socket path is too
  long".  Use `/tmp`.
- Never grep a worker log for "failed" as a completion signal (engine
  lines contain "failed 0"); wait on a marker you print yourself.
- `nsys` traces of the first chunk include the bench's warm-up; identify
  chunks by the kernel that runs once per layer (the GDN recurrence, 36
  launches per chunk) rather than by time.
- `ncu` with the resident owner is a documented OOM on this host.
- After `git checkout` of a file, `make` rebuilds its object even if the
  bytes are unchanged; a 13-minute MMQ compile can be the price of a
  cosmetic revert, and a build started while `make` is still evaluating an
  earlier goal picks up edits made in between.  Do not edit sources while
  a build runs; keep patches as scripts in scratch and apply them to a
  clean tree.
- A compile that runs during a timed run perturbs it; run traces (which are
  compared by kernel time) during builds, timed A/Bs never.

---

## 7. Checklist per round

1. Trace the production shape; rank gaps and kernels with launch counts.
2. Name the class of the change (5.1 / 5.2 / 5.3) before writing it.
3. Fixture with the production regime; bit-exact or a stated tolerance.
4. Full-model boot (prewarm chunk) and the family's resident gates.
5. Same-hour A/B, fresh workers, medians of three, both shapes (short
   prompt for TTFT, long prompt for the depth term), decode unchanged.
6. Kill switch on the same binary; attribute any text change.
7. Commit: bottleneck and its share, why the old path was inefficient, the
   contract, the numbers with shape and corpus, `Tests:`.
8. Update the doc with what was rejected and why; the next campaign starts
   there.
