# Repository instructions

`ds4-dfm-rs` is the independent Rust-host continuation of DwarfStar DFM.
The default `ds4`, `ds4-server` and `ds4-bench` are Rust hosts; ds4 is a
serving engine, and agents or frameworks connect over HTTP.
CUDA/MMQ/VMM and hardware-specific kernels remain native. C executables
with `-c` suffixes are deprecated: they keep the C sampler and the old
sampling policy and no longer track the Rust hosts.

Start with [README.md](README.md), the [documentation index](docs/README.md),
and the [architecture](docs/rust-migration/ARCHITECTURE.md).
The [v0.1.0 gate ledger](docs/releases/v0.1.0.md) records verified production
qualification and workload limits; the [v0.1.1 ledger](docs/releases/v0.1.1.md)
records the performance workflow. Dated reports establish only their
recorded commit, artifact and workload; old PIDs and handoffs are not live state.
The [v0.1.2 ledger](docs/releases/v0.1.2.md) records shared Jinja qualification.
New families use [official input templates](docs/chat-templates.md); keep
tokenizer/media processing and output protocol work outside input grammar.

## Required reading by task

Read these before starting the task; a brief handed to another agent must
name them too.

| Task | Read first |
|---|---|
| Any change that alters rounding, precision or accumulation order | [Goals](#goals) (mixed-quant acceptance), [Numerical tolerance](#gb10-dgx-spark-bottleneck-know-how), playbook §5 change classes. Bit identity is not a requirement (VQ weights are approximate already): gate on final output against a noise floor measured on this host from an existing configuration change |
| Performance optimization | [Optimization guidelines](#inference-performance-optimization-guidelines) and phase order, [GB10 know-how](#gb10-dgx-spark-bottleneck-know-how), [playbook](docs/prefill-decode-optimization-playbook.md), [gb10-hardware.md](docs/gb10-hardware.md), [ds4-perf.md](docs/ds4-perf.md), the family doc and its latest `docs/<family>-optimization-*.md` |
| Builds, probes, profiling, or a brief that names tools | Check the live toolchain first (`nvcc --version`, `readlink -f /usr/local/cuda`, `nvidia-smi`); call tools as `/usr/local/cuda/bin/{nvcc,nsys,ncu}` (the Makefile's `CUDA_HOME`), never a versioned path copied from notes |
| Anything that loads a model (bench, server, resident tests, nsys) | [Safety](#safety), [host-memory-guard.md](docs/host-memory-guard.md) (resident-model runs: 2 GiB reserve/trip). One GPU job at a time; never ncu on a full-model process |
| Model family serving or a new family | [serving contract](#new-model-family-serving-contract), [model families](docs/ds4-dfm-model-families.md), [sampled speculation](docs/speculative-sampling.md) for MTP and drafters |
| Commits | [Common Rules](#common-rules), [CONTRIBUTING.md](CONTRIBUTING.md) |

Parallel analysis agents: one kernel or scope per agent; read-only (no GPU
runs, builds or repository edits while a model or measurement runs; notes
under their own scratch directory); the brief carries the documents above,
the commit and artifact measured, current numbers, profile paths, and what
was already done or rejected. Each report gives the kernel's floor against
GB10 peaks, measured fraction, causes ranked by recoverable time with
evidence, each change's numerical class, and the cheapest measurement that
confirms it.

## DeepSeek V4.1 (ds41) — the port's reference is /data/YoungAi

The `ds41` family is a port of the DeepSeek V4.1 Flash engine kept at
`/data/YoungAi` (READ-ONLY reference, pinned at commit
`7fb26cc332d740416792204a29c7d13d2986caf5`). Every ported function mirrors
that engine with file:line citations, every deviation is named and measured
against it, and `/data/YoungAi` is the source of truth for ds41 behavior —
not the upstream ds4 tree, not any summary, comment, or memory note. Read
the engine source (and its own records: `bug.md`, `fable5.md`, `speed.md`,
`mtp*.md`, `log.md`) before changing, explaining, or planning anything the
port covers. Never modify `/data/YoungAi`; a behavior question is answered
by its code, and when the port and the engine disagree, the engine wins and
the port is fixed.

The DSML carriage (prompt bytes and the completion shape) has its own
reference: the official V4.1 encoding at `tests/ds41_ref/oracle/encoding.py`,
vendored byte-identical from `deepseek-ai/DeepSeek-V4.1-Flash` (MIT) and
pinned by sha256 in `tests/ds41_ref/ref_drive.py`. Every carriage deviation
from it is named and measured; the record is
`docs/ds41-dsml-reference-matrix.md` and `docs/ds41-dsml-function-map.md`.

## The ds41 DSML carriage executes nothing

The server renders the prompt, samples, parses the DSML out of the sampled
bytes, and frames it as OpenAI tool calls. It never executes a tool, never
resolves a client-supplied path, and never fetches a URL. Whoever calls it
owns the tool loop.

- The client-visible contract is the OpenAI wire alone (chat completions,
  completions, messages, responses). No request may require a field, a
  header, or a call sequence that only one client emits, and no
  caller-specific branch enters the server.
- No tool implementation enters `crates/ds4-server`: no filesystem read or
  write, no process spawn, and no network fetch reachable from a request
  body. Tool handling stops at framing, carrying tool results back into the
  transcript, and the on-disk replay store of (call id, exact DSML bytes)
  (`crates/ds4-server/src/tool_memory.rs`).
- Pixels arrive base64-only (`image_url`, Anthropic image sources and
  `video_url` must be base64). The oracle's path and URL image forms belong
  to the training-side encoder and are not served.
- Server filesystem writes stay limited to operator-configured state: the
  SSD/disk KV directory (which also carries the replay store), the file named
  by `--expect-plan`, and the opt-in `--trace` file (full prompts and outputs
  in plain text).
- The one piece of conversation text the server authors is the malformed-call
  retry suffix in `crates/ds4-server/src/retry.rs`; it is prompt-side repair,
  not a tool.

## New model-family serving contract

- Keep serving commands, flags, and client-visible behavior common across
  families. A new MoE family must implement every applicable serving option:
  context length, `max_seqs`, continuous banks, partial prefix reuse, session
  snapshots, SSD/disk KV directory and capacity, and MTP enablement and draft
  controls when the model provides MTP. Integrate a separate drafter such as
  DFlash at initial support when the artifact provides one; if both embedded
  MTP and the sidecar exist, expose and verify both modes. Parsing a flag
  alone is not support.
- Implement family-specific state, admission, and kernels behind that common
  surface. Report requested, effective, and qualified limits; reject forced
  unsupported settings and qualify each combination with live serving gates.
  Consult original upstream ds4 for serving flag conventions, without assuming
  that it implements the new model family.
- Keep per-kernel diagnostic knobs and kill switches family-specific. Select
  optimizations from each family's measured prefill and decode bottlenecks.
- Context length defaults to the model's native maximum (for example
  `max_position_embeddings`). A quantized or VQ artifact changes weights, not
  the context contract: size scratch so it never scales with rows x keys
  instead of capping context to fit it.
- Price what grows with context before fixed weights: Ling-3.1's BF16 latent
  KV (2.53 GiB per 262K bank) forced serial 512-row prefill; FP8 rows admit
  2,048-row banks.
- A serial lane must resume at the common prefix, not only on extension:
  agent clients re-render the last answer, so an extend-only sync prefilled
  every MiMo turn from zero (cached_tokens 0 even for an identical resend).
  Positional and SWA state rewinds while the ring, counted from the newest
  row written, holds the window, never below a restored snapshot (it holds
  one window); drafter windows re-run below the prefix. The host asks native
  for the kept rows: a host mirror lost the high-water mark across rewinds.
  Gate with an identical resend and a re-rendered continuation.
- Recurrent-state (KDA, linear-attention) payloads restore only a prompt that
  extends the saved history: test disk restore with the next turn; re-sending
  a turn always misses.
- Send only valid UTF-8. Models sample a character's lead bytes and then ASCII
  (GLM, DeepSeek and Beam all did in Korean), and openai-python clients such as
  Hermes decode every SSE line strictly, so one such delta ends the turn.
  `json_escape_bytes` writes U+FFFD; streams hold back a character cut between
  tokens only while the next token can still complete it.

## Inference Performance Optimization Guidelines

When optimizing inference performance, prioritize end-to-end execution-path efficiency over isolated kernel micro-optimizations.

- **GB10 clock prerequisite.** Keep the reference host at 300–2200 MHz for stable absolute performance and to reduce the DGX Spark trip risk reported by the operator. The operator-enabled `nvidia-clocks.service` applies persistence mode and this range automatically at boot. Do not ask the user to set clocks on each run or reboot when the service is enabled and successful; verify its status/current-boot journal read-only and record the boot identity and observed busy SM clocks. Do not stop/restart the service (`ExecStop` resets clocks), change/reset its policy, or let a profiler override it. Investigate a missing/failed service or conflicting observations before requesting user action. Missing historical clock metadata does not prove a different policy. See the [clock policy](docs/gb10-hardware.md#clock-policy-and-measurement-stability). (Codex learning)
- **Absolute stability before adoption.** Evaluate absolute prefill/decode rates and their repeat ranges alongside paired gains/losses. If absolute levels drift materially, explain clock/throttling, thermal, memory/PSI, cache, competing-work or speculative-acceptance differences before deciding. Preserve raw samples and disclose unresolved variability; a favorable percentage or smoothed graph cannot establish stability. (Codex learning)

- **Never chase bit identity for speed.** Do not restrict, redesign, delay
  or reject a speed change to keep outputs bit-identical, and do not spend
  effort making a faster path bit-identical. A change that moves rounding or
  accumulation order is judged by its output against a noise floor measured
  on this host from an existing configuration change (prefill: chunk-size
  frontier KL; decode: teacher-forced NLL and top-1 against an accepted
  arithmetic switch). Bit identity is only a free convenience of pure
  scheduling changes. A gate's fixed tie or threshold is not a noise floor:
  when an order change crosses one, compare against the reference before
  calling it a bug.

Follow this evidence-driven sequence for every optimization round, regardless of tool:

1. Measure the complete representative workload with `ds4-perf`, Nsight Systems, or another suitable tool. Establish wall time and separate prefill/decode costs. (Codex learning)
2. Identify a specific bottleneck and quantify its contribution to end-to-end latency. The target may be a kernel, a group of operations, dispatch, data movement, synchronization, or host work. (Codex learning)
3. Profile that target separately in detail: use NCU for GPU kernels or the appropriate profiler for other regions. Prefer a faithful isolated reproducer; use a bounded in-model capture when isolation would change the relevant behavior. Preserve relevant shapes, layouts, routing, cache state, and execution regime, and disclose differences. Narrow the execution scope, not diagnostic context: collect the metrics needed to explain the bottleneck, including broad counter coverage when useful, without repeatedly initializing the whole model when a smaller reproducer suffices. (Codex learning)
4. Derive the optimization hypothesis from those measurements, then implement a scoped change with an explicit numerical contract. Explore alternative explanations across execution structure, thread/tile mapping, memory access and reuse, fusion, numerical representation, and compiler output. Let evidence select the technique; neither this list nor the profiling procedure prescribes an implementation. (Codex learning)
5. Compare optimization off/on under matched conditions using fresh processes and correctness checks. Measure both the target and end-to-end prefill/decode performance; adopt only a reproducible useful gain without unacceptable correctness or performance regressions. After either adoption or rejection, restart whole-workload measurement on the retained baseline, identify its current bottleneck, and narrow detailed profiling again before choosing the next change. (Codex learning)

Do not impose a fixed percentage floor on adoption. A gain below 1% is worth retaining when repeated comparable A/B runs consistently improve performance beyond observed noise, preserve correctness, and show no meaningful regression elsewhere. Prefer such gains when they require no additional VRAM, arithmetic work, or other material resource cost. Check the relevant costs rather than assuming that unchanged tensor shapes imply unchanged work. If costs increase, report the tradeoff and judge it against the measured benefit; neither a small percentage nor one counter alone decides adoption. An automatic tool threshold is a screening heuristic, not the final engineering decision. (Codex learning)

**Adoption rule.** A free-lunch change with a consistent repeated gain is adopted however small; a change with a cost (memory, numerics, a regression elsewhere) is weighed against its gain or left to the user with both numbers; a rejection names the cost that outweighs the gain; when targets conflict, ask the user which to secure first. Cases: [playbook, Adoption](docs/prefill-decode-optimization-playbook.md#adoption).

**Fast-path coverage** is a common opportunity: maximize the fraction of the inference graph executed by the runtime's best hardware-appropriate primitives where measurements support it. The objective is verified end-to-end improvement; no implementation direction is mandatory.

Before optimizing an already-fast kernel, search for:

- operations unexpectedly falling back to slower kernels;
- irregular tensor shapes that can be decomposed into native fast-path tile or iteration sizes;
- repeated quantization, format conversion, normalization, permutation, or activation materialization;
- redundant intermediate writes and subsequent accumulation passes;
- routing, sorting, indexing, or expert-map construction whose complexity grows with tokens or experts;
- separate kernels that can legally share an accumulator, worklist, quantized activation, routing map, or intermediate representation;
- dense operations that can reuse an optimized routed primitive through an identity or single-expert mapping;
- prefill operations incorrectly using decode-oriented kernels, or vice versa.

Prefer eliminating a slow execution path over making an already-fast path incrementally faster.

Treat prefill and decode as distinct workloads. Use explicit width- or shape-dependent dispatch where their optimal execution strategies differ.

Model-specific specialization is acceptable and encouraged when it produces measurable gains, but isolate specialization behind explicit capability predicates based on tensor shape, quantization format, topology, and workload width. Generalize the dispatch policy rather than forcing specialized kernels into a generic implementation.

For every optimization:

1. Record the specific bottleneck and its measured contribution to end-to-end latency.
2. State why the existing path is inefficient.
3. Preserve or explicitly define the numerical contract.
4. Add focused parity or regression tests.
5. Keep a fallback or kill switch for new execution paths where practical.
6. Benchmark with fresh processes and comparable workloads.
7. Report both kernel-level improvement and end-to-end prefill/decode impact.
8. Verify that improving prefill does not regress decode, and vice versa.

### Phase order: optimize what makes the next phase cheaper to measure

Within a family campaign, order the phases so each finished phase speeds up the measurement of the next:

1. **Input paths exist first.** Every input the family accepts (text, and image/video through a vision sidecar) must work end to end before any phase is optimized; the functional implementation is a prerequisite, not a later phase.
2. **Prefill before decode.** Decode measurements on a realistic workload start with a prefill (long prompts, sweeps, cold frontiers); a faster prefill shortens every later run.
3. **TTFT next.** Time to first token also depends on functional reuse (disk KV, partial reuse, cache hits); the compute share of it is optimized here, the rest belongs to those features.
4. **Plain decode before MTP.** The non-speculative decode step is the baseline every MTP comparison divides by, and a verify pass is built from the same kernels; optimize it first, then the drafter, verify widths and acceptance.
5. **Text before vision performance.** Optimization rounds run on text workloads; once the text path is done, optimize the vision encoder and media prefill. Do not feed media into every text round.

Measure each phase against the theoretical upper bound of its dominant work (tensor-core peak for compute-bound kernels, memory bandwidth for streaming ones), stated per kernel class, and aim near 85-90% of it. The usable floors set the priority: once a context segment's (8K, 64K, 128K, 256K, 1M ...) prefill floor is above ~1,000 tok/s, further prefill work on it drops in priority (1,200-1,300 tok/s is comfortable); decode must reach at least 20 tok/s, with MTP where a segment's plain-decode bandwidth bound is below that.

Use this order as an investigation heuristic, not a fixed optimization sequence; measured contribution and detailed profiling determine priority:

unexpected fallback paths → repeated transformations → redundant memory traffic → routing/indexing overhead → launch fragmentation → fast-path utilization → individual optimized-kernel tuning.

The guiding principle is:

**Measure the whole workload, explain the specific bottleneck, and validate the change end to end.** Consider execution-path topology as well as individual kernels.

## GB10 (DGX Spark) bottleneck know-how

GB10 is the engine's optimization target. Its limits (SMs, clocks, `mma.sync`
peaks per type, DRAM/L2/shared-memory/NVMe numbers, missing `wgmma`/`tcgen05`)
and how they were measured live in [docs/gb10-hardware.md](docs/gb10-hardware.md);
start every roofline there and add new hardware facts there, not here.

The notes below were measured on GB10, sm_121, CUDA 13.4, busy clock
2,184–2,190 MHz under the 300–2,200 MHz policy. Background and history:
[GLM VQ](docs/glm53-vq.md).

- **Attribute first.** Export the Nsight trace to SQLite and sum kernels inside
  the `ds4.prefill` / `ds4.decode` NVTX ranges; divide by tokens. For each
  decode kernel compute bytes moved / time: anything far below ~235 GB/s is a
  target, anything near it only shrinks with fewer bytes.
- **Decode is bytes per token.** Streaming kernels reach ~250 GB/s (BF16
  dot), 233–237 (FP8 dot), 216 (Q4_K stream). Below ~5 µs a kernel is
  latency-bound: launch, tail and exposed DRAM latency dominate.
- **Never leave a serial loop on one thread.** A one-thread top-8 over 288
  experts cost 56 µs per layer; a one-thread 20-iteration Sinkhorn 25 µs. Use
  warp argmax / one lane per element with the same operation order.
- **Batch loads in single-block kernels.** One load per dependent add exposes
  ~1 µs per iteration: a one-block 16K RMS took 19 µs, 6.8 µs with 16 loads in
  flight per thread and the same accumulation order.
- **No host syncs on the hot path.** Any `ds4_gpu_tensor_read/write`
  (`cudaMemcpy`) drains the stream; per-layer routing readback plus an item
  upload cost ~1.3 ms per decoded token. Keep routing and tables on device.
- **Fuse conversion chains.** Convert → cuBLAS GEMV → convert is three launches
  and ran at 147–178 GB/s; one direct kernel reading the F32 row reached 250.
- **Shared memory caps tiling.** 99 KiB opt-in per block (101,376 B). A 64 KiB
  E4M3 codebook leaves ~35 KiB for staging, so one CTA per SM; stage sizes and
  pipeline depth must be budgeted against it (BT=64 BF16 staging does not fit).
- **Tensor-core VQ is fragment-bound, not FLOP-bound.** Rebuilding BF16 weights
  per fragment cost 23 issued instructions per MMA (28% tensor use). E4M3
  codewords feed `mma.m16n8k32.e4m3` unconverted; promote every 128 K to FP32
  (tensor-core accumulation truncates). Choose the k order freely when A and B
  agree so each lane loads whole codewords. Next limits: MIO queue, barriers,
  operand-register MOVs (`"+f"` constraints), and the weight stream DRAM floor.
- **L2 is small for MoE streams.** Bulk-prefetching each row's 384-byte chunks
  helps; prefetching the next work unit early made the FP8 kernel slower.
- **Noise.** With the model's page cache evicted, the first prefill chunk's
  lazy workspace allocation can stall (PSI spike) and cost 4–7% of prefill.
  Record load time and PSI; prefer repeated fast-load runs.
- **Numerical tolerance.** Use an existing configuration change as the noise
  reference (GLM 8K frontier, 512 vs 1024 chunks: logit RMS 0.23, KL 2.3e-7).
  Pure scheduling changes come out bit-identical for free, which makes their
  A/B trivial; never pay speed or effort for bit identity elsewhere.
- **Noise grows with context.** Ling-3.1's 1,024- vs 2,048-row chunk KL is
  0.0002 at 32K and 0.08 at 128K (argmax flip). Gate KV and weight formats
  against it at the same long frontiers, plus teacher-forced NLL after
  32K–128K prompts; short receipts cannot see cache precision.
- **Simulate a cache format first.** A writer switch storing
  quantize→dequantize values in the old layout measured Ling-3.1's FP8
  latents before any reader changed. Pick scales that keep values exact in
  the old type (E4M3 × 2^k is a BF16): real storage then matched the
  simulation bit for bit and every BF16 kernel stayed.
- **MTP verify widths are their own regime.** A 2-4 row forward falls off
  every one-row direct kernel onto convert + cuBLAS unless each kernel takes
  rows; that fallback was 15 of 70 ms per GLM one-draft step. Give direct
  kernels a token loop that reuses each weight load.
- **Expert overlap decides draft depth.** Consecutive GLM tokens share few of
  their top-8 experts (4 rows: ~26 unique of 32), so routed-expert bytes grow
  almost linearly with verify rows; an expert-union kernel saved under 10%.
  Prefer one or two drafts; measure the expert union before deeper drafts.
- **Drafts need the prompt.** A draft layer whose KV starts at decode drafts
  blind at first (GLM one-draft acceptance 0.63 → 0.97 within one run).
  Fill its KV during prefill. Check generated text for loops: a repeating
  bench continuation accepts ~100% and overstates MTP gains. Sliding-window
  draft blocks see only their window: MiMo's (128 positions) gained 5-8%
  acceptance from the prompt and verify rows.
- **Peaks.** Per SM and clock `mma.sync` does 1,024 F16/BF16 FLOP, 2,048
  FP8/INT8 and 4,096 NVFP4: ~108 / 215 TFLOPS at the 2.19 GHz cap, 128 / 255
  / 509 at ~2.5 GHz; cuBLAS BF16 65–90 TFLOPS. Streaming reads 231–246 GB/s
  (spec 273). Details: [docs/gb10-hardware.md](docs/gb10-hardware.md).
- **Store BF16-rounded rows as BF16.** F32 buffers of values already rounded
  at a BF16 boundary doubled GLM residual and expert-output traffic; BF16
  storage is lossless (prefill 1042 → 1121 tok/s).
- **Name predicates need HF spellings.** The Q8 aligned repack skipped
  `token_embd` but copied Ling-3.1 v2's `model.word_embeddings` (0.64 GiB of
  an ~8 GiB margin). Read the repack log when a tensor type changes.
- **Dependent launch.** A GLM token runs ~1,300 kernels; PDL with weights
  staged before the wait gave +1.8% decode, bit-identical. CUPTI durations
  then include the wait: attribute with PDL off.
- **Keep copies inside the dependent-launch chain as kernels.** A driver
  `cudaMemcpy` D2D between two PDL launches hung the GLM VQ MTP prefill in
  ~60% of CLI runs (GPU idle, host spinning in the next copy; never under
  nsys or gdb). A copy kernel that waits like its neighbours fixed it.
- **Small copies and gathers lose.** TMA copies of 128-byte rows and
  register-prefetched indices made the FP8 prefill kernel 31–50% slower;
  tensor-core decode experts lost to SIMT (97 vs 212 GB/s, per-lane gathers).
- **Multi-token SIMT GEMV is load-latency bound.** At three tokens the Q4_K
  dot stalls on each token's input loads (long scoreboard, 40% issue); the
  vendored MMQ/MMVQ stay at 150–190 GB/s. A BF16 MMA over the exact 4-bit
  integers runs 2–8 tokens at the one-token cost; stage 2,304-byte row
  segments (576-byte ones lost DRAM locality).
- **Diff the SASS of paths a refactor keeps.** Adding an epilogue template
  to the verify Q4_K kernel moved its row address math into the load loop:
  identical results, MTP decode 0.9% slower. Compare `cuobjdump -sass` of
  the old and new kernels (constant-bank offsets masked) before an A/B.
- **Apply the VQ gain after the dot.** Rounding gain·codeword to BF16 per
  weight was half the expert dot's instructions; one FP32 multiply after
  the dot made 2–4-row verify experts 10–13% faster (one row is DRAM-bound).
- **Read ptxas spills before tuning.** Sixteen warps of 128 registers
  spilled the GLM flash kernel's O and Q (34 GB of local traffic per 8K
  call, L2 at 84%); eight warps of 255 registers held them: 32.3 → 18.2 ms,
  bit-identical. 16-byte cp.async copies 64 bytes apart fetch each 32-byte
  sector twice; copy rows as contiguous 512-byte warp spans.
- **Mixed-input GEMMs: dequantize in producer warps.** HMMA saturates with
  four warps per SM on an MMA-dense stream. Consumers that convert 4-bit
  weights between MMAs, synchronized per stage, alternate conversion and
  MMA bursts: 62% tensor pipe. Producer warps writing dequantized A
  fragments (full/empty mbarriers, persistent CTAs) leave consumers LDS +
  HMMA: 82–84 TFLOPS of the 108 FP16 peak, against MMQ's 61 TOPS of INT8,
  whose per-k32 scale epilogue was 74% of its instructions.
- **Replay real routes in MoE reproducers.** Beam's 2,048-row prefill
  chunks send a median 1,011 pairs to the busiest expert (count CV
  0.6–3.4). A uniform-random harness put a warp-specialized gate at 4.6 ms
  against 5.4 ms in the model; replaying routes dumped from one prefill
  (temporary hook) reproduced the model. With all-128-token units the same
  kernel reached 81% of the FP8 peak, on real routes 55%: skewed routes
  leave many units latency-bound on their index stream.
- **Attribute NCU stalls to source lines.** Export
  `ncu --page source --print-source sass --csv`, map each address minus
  the kernel's first address through `nvdisasm -g` line tables, and sum
  stall reasons per line (the CUDA-source CSV view carried no metrics).
  Split mbarrier waits by the barrier offset in `SYNCS.PHASECHK` (full vs
  empty) to see which role starves.
- **Keep dependent loads off unit boundaries.** Per-stage `map[]` loads
  and divisions in producers and `map → scale` chains in epilogues were
  the largest stall of a warp-specialized VQ gate. A tiny kernel writing
  each pair's token row and scale, metadata loaded one unit ahead, and the
  unit record passed to consumers inside the ring (a 16-byte cp.async in
  stage 0) took it from 5.0 to 4.0 ms, bit-identical.
- **ptxas reorders `mma.sync`.** `asm volatile` orders only NVVM; with two
  A fragments alternating across MMAs ptxas rebuilt one A register quad
  with four MOVs per QMMA. For VQ codewords (8 bytes = k 4q..4q+3,
  16+4q..16+4q+3) compute tokens × weights^T: a codeword is exactly an
  m16n8k32 B fragment (one LDS.64, no MOVs) and tokens are A via
  ldmatrix.x4 once their rows are stored in the matching k order. Results
  stayed bit-identical: the k32 reduction is symmetric in A and B.
- **Runtime indices into struct arrays go to the stack.** `item.main[m]`
  with a runtime `m` gave an 80-byte stack frame and local loads in a copy
  loop (13.7 → 10.8 ms after a ternary select). Read ptxas stack sizes,
  not only spill counts.
- **L2 holds CTAs × (current + prefetched) blocks.** 48 CTAs prefetching
  240 KB next-unit blocks beside a 10.5 MB activation overflowed the 24 MB
  L2 (1.65 GB DRAM for a 1.3 GB minimum); 120 KB blocks fit. Pace the
  next-unit L2 prefetch one piece per stage: front-loading it lost 5–10%,
  none lost 45%. A global tile-major unit order lost 20% (activation reuse
  distance); tile-major over one expert's items saved 14% on a 64-token
  fused kernel (concurrent CTAs share each block).
- **mbarrier waits need a suspend hint.** `try_wait` spin loops were 19% of
  a kernel's issued instructions; the suspend-time operand (CUTLASS uses
  0x989680) parks the warp in hardware.
- **Raw tiles may not fit beside a decoded double buffer.** FP8 K/V
  attention tiles staged raw (2 × 17 KB) plus decoded halves (2 × 35 KB)
  needed 103 KB; producers loading the next tile into registers (LDG.128)
  and decoding into the halves fit in 68 KB and made Beam's 64K-frontier
  prefill attention 1.49× faster, byte-identical. When two 16-byte stores
  per lane share a bank group, alternate which one each lane issues first.
- **Every included header belongs on the object's rule.** A new .cuh
  pulled in by `ds4_beam_gpu.cuh` but missing from the `ds4_cuda.o`
  dependency line left the object stale, and an unchanged binary was
  measured as a variant. Check the object's timestamp after each edit.
- **Verify rows meet in L2 when every row walks the same order.** One
  launch with a CTA group per verify row (each stages only its row) and
  one task order across groups streamed each shared expert about once:
  MiniMax's 4-row expert verify ran 1.3-1.6x faster per layer. An
  input-union kernel did not fit (three 6,144-wide rows beside a 66 KiB
  book exceed 99 KiB). One row keeps the expert-major order.
- **L1 bank conflicts also hit `cp.async.ca`.** Rows 1,152 B (9 x 128 B)
  apart put all 32 lanes of an 8-byte copy on one bank pair (32
  wavefronts); rotating each lane's piece order spread them over three
  and cut MiniMax's R13 expert gate 6%, bit-identical.
- **Probe an estimate before building on it.** A model of the expert
  kernel projected 25-35% from a k-outer loop with per-row activation
  scales; the harness ran it even, and loading the next stage's codewords
  between a stage's MMA halves ran 11% slower (consumers holding two of
  three ring slots starve the producers).
- **Frontier logits are chaotic in a 60-layer MoE.** Two accepted MiniMax
  attention paths (F32 vs BF16 P) differ by logit rel-RMS 0.15-0.76 and KL
  up to 14 at single frontiers, greedy tokens within a few tokens. Gate
  value changes on paired NLL and top-1 over many assistant positions
  (`tests/minimax_quality`), not on single-position logits.
- **systemd-oomd acts on pressure, not on available memory.** A +1.6 GiB
  pinned pool left 3.6 GiB available, yet a 60 s cold prefill drove the
  user slice's memory pressure to 58% (> 50% for 20 s) and oomd killed
  the run. Budget pinned and device allocations against reclaim pressure.
- **Drop caches before a model launch.** A previous run's page cache and
  the driver's freed-page pool stalled a 107 GiB A.X-K2 server at the
  guard's MemoryHigh (PSI 45%); the same launch after `clear_cache` came
  up at PSI 0.
- **Select a large top-k by radix, not by sorting.** A decode row's
  top-2,048 of 8K keys took ~0.2 ms per layer in a chunk-sort-and-merge
  tree; one block of four 8-bit digit passes (per-warp histograms), then a
  counting and a writing pass, took 11 µs, and 2,048 prefill rows of 8K
  keys 0.44 ms against a block sort's 1.9 ms. Ties go by index.
- **Decode Q4_K by block, not by element.** A column pass (`W^T x`) that
  decoded one Q4_K value per thread and row ran at 25 GB/s; lanes decoding
  whole 256-blocks (`glmvq_q4_acc`'s layout) with partial rows summed in
  warp order ran 4.5x faster.
- **Give a one-row kernel more than one SM.** A gated RMSNorm with two
  rank-16 BF16 projections (458 KB) in one block took 42 µs per row;
  recomputing the norm in 16 blocks for `W_down` and splitting `W_up` over
  28 took 13 µs, bit-identical.
- **TF32 is exact for BF16 values.** An F32 GEMM whose operands are
  BF16-rounded (norm rows, widened BF16 weights) runs on TF32 tensor cores
  with exact products: Ling-3.1's router 1.43 → 0.66 ms per 4,096 rows
  (BF16 rows on BF16 MMA: 0.36 ms).
- **Hold a recurrent state column in registers across verify rows.**
  Ling-3.1's KDA verify re-read and re-wrote its F32 state per row through
  L2: three rows 127 → 97 µs per layer, MTP decode +4.5%.
  One decode row at 255 registers lost 1.5%: dispatch by rows.
- **Group decode launches only where PDL cannot overlap them.** One token's
  five KDA projections in one stream launch: decode +3.3%; the same grouping
  on q4m at verify widths: 7% in a harness, 0% in the model.
- **Probe a decode access pattern's read-only ceiling.** 240-byte rows per
  warp capped a read-only probe at 191 GB/s; the expert down kernel ran
  within 10% of it, so no prefetch variant could win.
- **GQA decode attention is an MMA.** MiMo's 16 query heads per KV head are
  the m16 rows of an f16 MMA over one row of FP8 codes; choosing the k order
  per lane (dims 48t..48t+47 of a key, values 16g..16g+15) makes fragments
  contiguous. Lanes loading their own 16-byte segments capped a read-only
  probe at 150 GB/s; a block staging whole rows with contiguous cp.async
  reached 182-191 (64K-256K), 4.4x the per-key shuffle walk.
- **Widen an FP8 KV tile once per head pair.** Eight warps over two query
  heads of one KV head, the next tile's codes in registers during the MMAs:
  MiMo's prefill attention 8.9 -> 68-74 TFLOPS. A second f16 component of Q
  and P bought nothing against E4M3 keys; against exact f16 keys it did.
  E4M3 Q (one scale per row and head) makes the stored keys MMA fragments
  without widening: 69 -> 95 TFLOPS at 64K. Its error on random rows was
  100x f16 Q's, yet NLL pairs after 32K and 131K prompts stayed inside the
  accepted switches' band: gate operand formats on the model.
- **Run ncu with `--clock-control none`.** Its default locks the base
  clock for the profile (2.40 GHz on GB10, above the 2,200 MHz policy).
- **Host sampling is decode time once drafts verify sampled rows.** Sorting
  MiMo's 152,576 logits for top-p took 13 ms per draw on a Grace core, and a
  sampled MTP2 step draws up to four times: 127 ms against 79 greedy.
  Sorting the head first (tokens ≥ 1e-3 of the top probability, the full
  order's prefix) took the step to 83 ms: sampled code 18.4 → 32.7 tok/s.
  Time a sampled request whenever drafting or the sampler changes.

# Agent Notes

Support is limited to explicit model-family/artifact contracts listed in
[README.md](README.md#supported-model-families). Keep the Rust host and native
backend small and direct; this is not a generic GGUF runner.

## Goals

- Keep production inference on the validated GPU execution paths. DGX Spark
  CUDA is the release reference; inherited Metal support needs separate checks.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- For mixed-quant artifacts, accept explained arithmetic differences when they
  do not materially affect output quality. Assess logits, representative answers
  and relevant cache state; bit identity or a fixed relative-RMS threshold alone
  is not the acceptance criterion. KV, position and media-layout errors remain
  correctness failures.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.
- Derive an artifact revision with the converter's quantizer only after it
  reproduces the published bytes: Ling-3.1 v1's Q4_K needed
  `-ffp-contract=off` (aarch64 FMA contraction moved 3 of 244 blocks).

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Keep new host/control-plane code in Rust; do not add a C++ host layer.
  Existing CUDA/MMQ implementation stays native.

## Safety

- The DGX Spark reference host has a user-managed GPU clock range of
  300–2200 MHz (reconfirmed 2026-10-27). Preserve it for benchmarks and
  restored serving; follow the systemd clock-service and stability rules above.
- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not load independent huge model copies concurrently. Use the intended
  weight owner and bounded workers; inspect ownership and memory first.
- Preserve unrelated servers and resident owners. Stop only processes owned by
  the task. See [host-memory-guard.md](docs/host-memory-guard.md) for the guard's
  admission rules and documented limits.
- Prefer short GPU smoke tests for build verification
  (Metal on macOS, CUDA on Linux).

## Layout

- `crates/ds4-cli`: production CLI and benchmark.
- `crates/ds4-server`: HTTP, admission, scheduling, streaming and tools.
- `crates/ds4-core`: model/session ownership, host catalog and tokenizer.
- `crates/ds4-kv`, `ds4-dist`: KV policy and distributed runtime.
- `crates/ds4-sys` + `native/bridge`: narrow native inference boundary.
- `crates/ds4-perf`: standalone inspect/scout/compare/optimize workflow.
- `crates/ds4-perf-gpu`: optional profiling-only CUDA calibration/CUPTI helper.
- `ds4.c`: native engine, GPU state/execution and retained compatibility helpers.
- `ds4_cli.c`, `ds4_server.c`, `ds4_bench.c`: deprecated C host oracles.
- `ds4_ling3vl_*.inc` / `ds4_ling3vl_gpu.cuh`: Ling-3.0-flash-VL graph,
  bank runtime, image spans and vision tower. Its KDA, latent-attention and
  routed-MoE stages reuse the Solar/GLM, Motif and Step primitives; only the
  grouped router, the M-RoPE, the fused-row latent RMSNorm and the BF16 MLA
  absorb pair are its own.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu`: CUDA backend. Single TU; mirrors `ds4_metal.m`'s role on
  NVIDIA. CUDA env vars and dispatcher behavior are documented in
  `misc/cuda-env-vars.md`; CUDA MTP specifics in `misc/cuda-mtp/README.md`.
- `cuda/mmq/`: vendored llama.cpp ggml-cuda matmul kernels + ds4-side adapter.
  See `cuda/mmq/VENDOR.md` for the upstream pin and re-sync procedure.
- `tools/ds4_weight_server.cu`: optional CUDA weight-server sidecar for
  multi-process testing. See `misc/proof-harness/README.md`.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material. A few
  reference docs are force-added (`cuda-env-vars.md`, `cuda-mtp/`,
  `proof-harness/`, `ANTHROPIC_LIVE_CONTINUATION.md`, `RESPONSE_API.md`).

## CUDA captured-decode rules

- Captured decode kernels that consume `pos0`, `n_comp`, `n_index_comp`,
  `raw_start` / `raw_row`, `n_raw`, selected-row counts, or scratch pointers
  MUST read live substrate state (`g_decode_dev` / `g_layer_dev[il]`) or be
  keyed by regime into the graph cache. By-value kernel arguments are baked at
  graph queue time and replay stale. Reference: 7c4b84d, a1cff19, 8fb3c54.
- Long-context captured-vs-eager parity (essay prompt, n=1024, FP32, every
  enabled overlay) is a release gate, not a smoke test. See `make proof-cuda-long`.
- Optimization commits land with a correctness proof AND a speed proof. The
  proof harness records both: `tests/ds4_proof.py --scenario ...`. Skipping the
  correctness proof on the grounds that "we already had it before" is how the
  pos0 regression slipped past three previous commits.

## MTP / compressed-KV decode rules

- Drafts verify against the request's sampler at every temperature; never gate drafting on temperature, thinking or a forced control prefix. A trial draws its verify rows with `spec_sample_rows` (sample-and-match, every draw spent; only an uncut prefix's last draw is the next token); a forced position ends the prefix before commit through the host's `limit`: [sampled speculation](docs/speculative-sampling.md). Greedy-only verification and whole-request prefix gates once left MTP off for every thinking request and every agent turn with tool results.
- `--mtp-mode on` removes `DS4_MTP_SPEC_DISABLE` from the environment: an A/B that sets the switch beside the flag drafts in both runs (MiMo timed the same twice). An off run drops the drafting flags and passes `--mtp-mode off`.
- Lossless = verified tokens AND committed compressed-KV == the accepted-prefix transition. Not bit-identity (cross-width MoE order differs), not tokens alone (carried state sets future logits).
- Emit compressor state in the verify forward, roll back rejects; never re-emit in a second pass (a commit-reforward emit was ~100% wrong).
- Gate on ‖Δcompressed-cache‖ and long context, not short-prompt tokens — short context hides cache corruption behind raw/windowed KV.
- FP vs structural is element-wise magnitude (rel ≈1e-6 vs ≈1); a late token flip is not FP.
- A request finishing inside an accepted draft prefix lands only the rows before its last token: ordinary decode leaves that token pending, and the server keys a retired bank by every generated token but the last. Probe reuse across `max_tokens` so finishes hit each draft position (Ling-3.1 MTP: 2 of 10 follow-ups missed).
- Size GPU-readback buffers to the tensor and fail loud — undersized returns a constant → false "identical".
- At D>=1 the batched verify (width 1+D) is NOT bit-identical to a width-1 decode (cross-width MoE-order FP, same as N=1 vs N=8). At **N=1** the committed cache IS invariant to rejected-draft VALUES (width-matched, no cross-row terms) — validate the rollback there expecting bit-identical cache; comparing D>=1 vs width-1 mode-0 conflates inherent FP with real bugs.
- At **N>1** the committed cache is NOT value-invariant to rejected drafts, and that is NOT a bug: the in-forward verify is itself draft-VALUE-dependent via cross-row MoE-reduction FP — holding cur's input identical, varying a causally-masked rejected draft shifts the verify logits by FP while the argmax (committed token) is unchanged, and that FP rides an expert flip into the committed KV (deep-layer ‖Δ‖ ≈ O(10) while the final logit moves ≈0.03 and the token does not flip). So N>1 losslessness is TOKEN-LEVEL: gate the committed-row FRONTIER (counts) + token stream invariance to rejected-draft values; treat committed-cache VALUES as inherent-FP-noisy (informational), and add a byte-exact restore self-check (read back each restored lane vs its checkpoint source) to guard the capture/restore plumbing independently. Don't chase the N>1 cache-value diff as a rollback bug — bisect: the divergence onset is a cur-row (M=1) logit FP in the forward, not the restore (which is byte-exact).

## Testing

Use the [contribution checks](CONTRIBUTING.md): Rust host parity and serialized
workspace tests are model-free; CUDA/family/proof gates need their exact
fixtures. On Linux, `make` prints help: select `make cuda-spark`,
`make cuda-generic`, or an explicit CUDA architecture for native validation.
Use live server tests only when intentionally testing the API surface.

For ds4-perf, follow [the workflow and evidence contract](docs/ds4-perf.md).
Run `cargo test -p ds4-perf`. Build optional instrumentation with
`make ds4-bench-perf` after the CUDA build. Use NVIDIA's official Rust `nvtx`
SDK directly in `ds4-cli` under `perf-nvtx`; keep its default features disabled
and enable only `std` unless a measured need justifies more. Use `LocalRange`
for the measured `ds4.prefill` / `ds4.decode` operations. Do not add custom NVTX
FFI, route it through `ds4-core`/`ds4-sys`/the bridge, or annotate each token or
kernel. Ordinary inference builds remain independent of this optional SDK.

Multi-process testing (proof harness, multi-profile sweeps, MTP correctness
work that loads base + MTP gguf into the same device) goes through
`ds4_weight_server`. See `misc/proof-harness/README.md`. Single-process
runs hit the same prefill ceiling without a sidecar via the in-process
VMM arena, which is on by default.

## Common Rules
- When writing something intended for human consumption, (comment, commit message, reply to prompt) use as few words as possible. Pick every word meticulously to reduce the volume to a strict minimum. Be down to the point. Less is more.

- Avoid superlatives and praise. Stop telling me I am absolutely right. Give me the cold hard truth.

- Avoid magic numbers and strings by extracting recurring or meaningful values into descriptive constants (const) or enums. Keep self-explanatory, one-off values inline to avoid clutter. If a value comes from a spec (e.g. HTTP 200 OK), use a constant regardless.

- Reduce code indentation. Avoid Arrow Anti-Pattern. Leverage early return and continue.

- Keep function names short. Less than 30 characters.

- Use enums instead of booleans for function parameters.

- Let the reader of the code breathe. Add empty lines between logical blocks of code.

- Add a small, to the point, comment to explain *what* the block does and *why*. Use examples when possible. Propose ASCII drawings to explain complete systems.

- Treat member visibility changes as a breaking design shift. Keep all fields and functions private unless external access is strictly required by the design. Prompt the user for explicit approval before changing any access modifier from private to internal or public.

- Program to levels of abstraction. Lower-level mechanics (e.g., raw hardware I/O, sector parsing, direct socket streams) must be encapsulated in a dedicated driver/abstraction layer. Expose clean, high-level APIs to the rest of the application so calling code works with domain concepts, not raw implementation details.

- Don't touch blocks of code unrelated to the feature you implement. e.g. Don't add comments to a block of code if you did not create it or modify it. As much as possible try to minimize the number of changed lines when implementing a feature.

- Preserve the inference boundary: host → ds4-core → ds4-sys → native backend.
  Host-only libraries such as the official NVTX SDK are side dependencies;
  they do not belong in the inference ABI. Keep raw device handles out of
  application code and follow [FFI_CONTRACT.md](docs/rust-migration/FFI_CONTRACT.md).

- Always use {}, even on a one-line "if" statement.

When you write a commit message, follow these 7 rules:
Rule 1: Separate the subject line from the body with a single blank line.
Rule 2: Limit the subject line to 50 characters (72 is the absolute hard limit).
Rule 3: Capitalize the first letter of the subject line.
Rule 4: Do not end the subject line with a period.
Rule 5: Use the imperative mood in the subject line (e.g., "Fix bug," "Add feature,"
        not "Fixed" or "Adds"). Test formula: It must complete the sentence: "If applied,
        this commit will [your subject line here]".
Rule 6: Wrap the body text manually at 72 characters to prevent Git formatting issues.
Rule 7: Use the body to explain what and why vs. how. Assume the code explains the how;
        the message must explain the context and reasoning.

- If the prompt indicates that a bug is being fixed, don't write the fix right away. First write the test. Observe it failing. Then write the fix. And observe the test passing.
