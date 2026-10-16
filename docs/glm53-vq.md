# GLM-5.3-Flash VQ: concept, execution and optimization

GLM-5.3-Flash VQ (family `Glm53Vq`) stores the routed experts of the
official FP8 GLM-5.3-Flash as vector-quantized indices and keeps the other
tensors in their source formats, except where a selection replaces them
with Q4_K. This page explains the format, how the CUDA backend executes it
on DGX Spark (GB10), what each optimization changed and what still limits
speed. Dated reports in this directory keep their own recorded workloads.

The format concept follows the YoungAi technical notes for DeepSeek V4.1
Flash ([TECHNICAL.md §4.1/§4.4](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/TECHNICAL.md)).
GLM is a separate family: its recipe, arithmetic and qualification are its
own, and DeepSeek throughput is a reference point, not a GLM guarantee.

## The format

```
w[r, 8j : 8j+8]  ≈  g_r · C[idx[r, j]]      C: 8192 × 8 E4M3, one per layer
bits per weight = 13/8 = 1.625, plus one F16 gain per output row
```

- **Vectors, not scalars.** Eight consecutive input-axis weights of a row
  share one index. Joint coding beats scalar rate–distortion at the same
  budget and needs only one gain per row; there are no per-group scales.
- **One codebook per layer.** All 288 experts and the gate/up/down matrices
  of a layer share one K=8192 codebook (64 KiB as E4M3). Experts are
  statistically alike, so sharing costs little and lets a persistent CTA keep
  the whole book in shared memory for every expert it touches.
- **E4M3 codewords.** Halves lookup bytes against F16 (+0.1% relative MSE in
  the recipe screen). E4M3 converts exactly to F16/BF16, and it is also the
  native FP8 tensor-core operand type.
- **12 + 1 bit-plane layout.** A row's 12-bit stream packs 256 indices into
  exactly 384 bytes (three 128-byte lines); the 13th bit lives in a separate
  1-bit plane. Loads that do not fill whole 128-byte lines lose roughly a third
  of GB10 bandwidth, so the main stream keeps 12-bit geometry.
- **Row gain.** Output-row RMS, F16. The decoder multiplies it in FP32.

The artifact (`FORMAT.md`, GVQ1 v1) is a set of independently readable GGUF
parts. Every tensor start is 128-byte aligned. The loader verifies all
file, tensor and payload SHA256 values before use; workers verify parts in
parallel (about 20 s for the 74 GiB artifact on Spark).

| Component | Storage (Q4_K v3 artifact) | Bytes read per decoded token |
|---|---|---:|
| Routed experts, 42 layers × top-8 | GVQ1, 1.625 bit | 1.72 GB |
| KDA Q/K/V/O, 34 layers | Q4_K | 2.57 GB |
| DSA q_a/q_b/kv_a/o, shared experts, dense FFN | Q4_K | 1.46 GB |
| Output head | Q4_K | 0.36 GB |
| KDA gates, indexer, router, mHC maps | source BF16 | 0.56 GB |
| Latent attention (KV_B, cache at 8K) | BF16 | 0.55 GB |
| KDA recurrent state (read + write) | F32 | 0.27 GB |

About 7.5 GB per token (9.6 GB with the source-precision projections and
head). At the measured ~235–250 GB/s streaming ceiling that is a 30–32 ms
wall; decode measures 40 ms. The DeepSeek V4.1 VQ engine reads about 6.3 GB
per token with the same Q4_K choice for its nonexpert weights.

## Execution on GB10

### Decode (one row)

- **Experts.** One persistent CTA per SM loads the layer book once and
  streams the eight selected experts' rows: 384-byte index blocks read as
  three full lines, codes distributed with warp shuffles, the 13th bit from
  its plane, gate/up and the clamped SwiGLU fused, down computed directly and
  experts reduced in ascending expert order. Routing stays on the GPU: a
  route kernel places selected experts from per-layer device tables, so
  decode no longer syncs with the host per MoE layer. The row gain multiplies
  the FP32 dot of the exact E4M3 codewords once, as in prefill; rounding
  gain·codeword to BF16 per weight was half the dot's instructions
  (`DS4_GLMVQ_GAIN_DOT=0` restores it).
- **Non-expert projections.** Q4_K uses a cooperative stream kernel, FP8 a
  direct dot that rounds weights at the source BF16 boundary, BF16 a direct
  dot that reads the F32 activation row in place (the output head reaches
  250 GB/s). Small matrices split a row across warps. Shared-expert and
  dense gate/up run as one tensor-core pass that writes the clamped SwiGLU
  (a tile holds 8 gate rows over the same 8 up rows): at one token for the
  2,048-row shared expert (57 → 49 µs per pair) and at every verify width
  (58 → 53 µs at three tokens). `DS4_GLMVQ_GATE_UP=0` restores the two
  projections and the SwiGLU kernel.
- **mHC.** One kernel computes the 4-stream RMS and the 24 F32 mapping dots;
  the Sinkhorn split runs one lane per element and also applies the
  following RMS norm.
- **Attention.** KDA layers run the recurrence; DSA layers use the absorbed
  latent attention (`(Q·W_k) Cᵀ`, `(P·C) W_vᵀ`) over the selected keys:
  direct GEMVs for both absorptions and a single-pass flash kernel whose
  keys split across 48 CTAs, joined by a combine kernel.
- **Dependent launch.** A token runs about 1,300 short kernels. Decode and
  verify widths launch them with programmatic dependent launch: each kernel
  lets its successor start, which stages its constant weights (or prefetches
  them to L2) and then waits for its inputs. `DS4_GLMVQ_PDL=0` turns it off.

### Prefill (4096-row chunks)

- **Experts on FP8 tensor cores.** The source model was trained and served
  with FP8 weights and dynamic 1×128 activation scaling. Codewords are E4M3,
  so `mma.m16n8k32.e4m3` takes them unconverted as the A operand; activations
  are quantized once per token and 128 channels; every 128-K block promotes
  its tensor-core partial sum to FP32 with its activation scale; the row gain
  multiplies the FP32 dot. The k32 reduction order is free as long as A and
  B agree, so lane q takes whole codeword 4k+q: one 8-byte book read fills two
  A registers and B is eight source-order bytes. Items carry up to 64 tokens
  of one expert (two cp.async stages beside the 64 KiB book), so a weight tile
  is decoded about once per chunk; the reduce adds the shared expert. Gate
  and down outputs are stored as BF16 (their rounding boundary), and the up
  epilogue writes down's E4M3 rows and 1×128 scales itself.
- **KDA.** One CTA per (64-token chunk, head) reads the raw q/k/v/forget rows
  once, reproduces the recurrence's inputs exactly and writes an operand record
  (q e^G, (k e^(G_last-G))ᵀ, −W, causal q·k in F16; U in F32). One CTA per head
  then walks the chunks with the state in F32 registers, runs every product on
  F16 tensor cores (value-major, so state and V' accumulators repack in
  registers) and applies the gated output norm. The shared Solar kernels kept
  F32 planes and four passes (~2 GB per layer at 2,048 rows): 12.2 → 4.7 ms.
  The forget and gate projections stay BF16 rows for these kernels.
  Appends under 64 rows keep the recurrence; `DS4_GLMVQ_KDA_MMA=0` restores
  the Solar chunks.
- **Attention.** One flash kernel per query row computes `softmax(q Cᵀ) C`
  over the latent rows with BF16 tensor cores, without expanding K/V. Index
  scores are BF16 MMAs folded over the 32 heads in registers, so no per-key
  buffer exists and the context follows the model's native 1M.
- **mHC.** Each expand writes the new four-stream residual and accumulates the
  next half's RMS and 24 mapping dots in the same pass; the split collapses.
  The residual holds BF16-rounded values, so prefill stores it as BF16
  (32 KB per row instead of 64).
- **Projections.** Q4_K uses the vendored MMQ path with F32 outputs; matrices
  with one input (q/k/v, gate/up, q_a/kv_a) share one Q8_1 quantization.
  FP8/BF16 restore to BF16 and use cuBLAS, sharing one BF16 input copy.

### MTP (one or two drafts)

```
draft:  h(s-1), token s ──MTP layer──▶ d1 ──▶ d2 …     (layer 45, own KV)
verify: [token s, d1, …] as one n-row forward ──▶ n argmaxes
accept: keep k rows; KDA state ◀── journal slot k, index tail ◀── replay
```

- **One verify forward.** The n candidates run as one forward, so each
  weight is read once. Every kernel has a 1–4 row path that reuses each
  weight load and keeps the one-row arithmetic: verify rows equal decode
  rows bit for bit except Q4_K and the attention split count.
- **Verify-width Q4_K on tensor cores.** One BF16 `m16n8k16` MMA covers 16
  rows and 8 tokens: A holds the exact 4-bit integers, B the inputs (BF16
  values already), so products are exact and only the F32 order differs.
  Each 32-element sub-block keeps its own partial scaled by d·sc; an MMA
  with A = m adds the min term. Stages of 16 rows × 16 blocks are whole
  lines and long DRAM runs. A forward's Q4_K at three tokens: 26.3 → 21.6 ms,
  about the one-token cost. `DS4_GLMVQ_Q4_MMA=0` restores the SIMT stream.
- **Rollback.** KDA writes its state after each row into a journal slot;
  index tails are replayed from stashed rows; latent, pool and draft-layer
  rows past the frontier are rewritten before any read.
- **Draft context.** Prefill fills the draft layer's KV from the prompt;
  without it early drafts see only decoded tokens.
- **Depth.** Routed-expert bytes grow almost linearly with verify rows
  (consecutive tokens share few experts), so one or two drafts win. Serve
  with `--cuda --mtp-draft 2 --temp 0` (two drafts); `DS4_GLMVQ_MTP_DRAFT=N`
  caps drafts per verify for measurement.

## Numerical contract

Each optimization keeps the source rounding boundaries unless it states a
change. Accepted changes and their measured effect at the 8K frontier
(7,936 prompt tokens, all 154,880 logits) are listed below. The reference
noise band is an earlier prefill chunk-size change (512 vs 1024): logit
RMS 0.23, KL 2.3e-7.

| Change | Arithmetic | Frontier logits | 128 greedy ids |
|---|---|---|---|
| Router, RMS, Sinkhorn, device routing | identical order | bit-identical | identical |
| Direct BF16/F32 projections, mHC mix | FP32 accumulation order | ≤ 1 BF16 ulp, relRMS 2.3e-6 | identical |
| Chunked KDA prefill | FP32 order of the recurrence | RMS 0.20, KL 2.2e-7 | identical |
| FP8 expert prefill | E4M3 activations, gain after the dot | RMS 0.20, KL 1.1e-7 | identical |
| Flash DSA attention, absorption GEMVs | FP32 softmax/accumulation order | within noise band | identical |
| MTP verify kernels | one-row order per row | not applicable | identical at 8K; near-tie flip on prose |
| Tensor-core verify Q4_K | exact products, F32 sub-block order | not applicable | prose and code equal MTP off (128 ids) |
| Expert gain after the dot (decode, verify) | FP32 dot × gain, no per-weight BF16 rounding (relRMS 3e-3 vs it) | not applicable | 512 decode ids and MTP code ids identical |
| Fused Q4_K gate/up/SwiGLU | verify: bit-identical; one-token shared expert: tensor-core F32 order | not applicable | 512 decode ids and MTP code ids identical |
| Prefill mHC mix in one pass, narrow RMS | F32 order | RMS 0.20, KL 2.1e-7 | identical |
| F32 Q4_K prefill outputs | no BF16 boundary after MMQ | RMS 0.24, KL 1.2e-7 | identical |
| Expand fused with the next mix | F32 order through Sinkhorn | RMS 0.32, KL 1.8e-6 | identical |
| Tensor-core KDA chunks | F16 operands (k F32, V' in two halves) | RMS 0.25, KL 7.9e-7 | identical |
| BF16 storage of BF16-rounded rows (residual, expert gate/down, KDA gates) | identical values | bit-identical | identical |
| Fused down quantization, scan staging, dependent launch | identical order | bit-identical | identical |

On the text-quality cases the tensor-core KDA moves the paired NLL by
+0.008 ± 0.006 (standard error) nat/token against the F32 recurrence, with
the same mean KL as the F32 chunks (0.0056 vs 0.0054): within the model's
chaos floor, where any F32 reordering already reaches that KL.

Position, length, routing-table, cache and rollback state must stay exact;
floating-point differences are judged by output quality, not bit identity.

## Optimization log, 2026-10-16

Workload: 7,936 input + 128 forced greedy tokens, context 8,192, MTP and
capture off, fresh process, busy SM clock 2,184–2,190 MHz. From the Q4_K v3
row on, one untimed 1,024-token prefill warms the process first (a cold
first chunk paid lazy allocation and page-cache stalls).

| Step | Prefill tok/s | Decode tok/s | Main mechanism |
|---|---:|---:|---|
| Start (`6c143772`), chunk 1,024 | 422.2 | 16.59 | — |
| Decode overhead kernels | 424.6 | 18.25 | serial top-8/Sinkhorn removed, batched loads, no host route sync |
| Direct BF16 projections | 429.5 | 20.33 | one kernel per projection, mHC fused |
| Chunked KDA prefill | 506.5 | 20.20 | delta-rule chunks instead of 1024 serial steps |
| FP8 expert prefill | 566.3 | 20.01 | mma.e4m3 on codewords |
| Q4_K v3 artifact, warm-up | 594.4 | 23.47 | 316 nonexpert matrices Q4_K |
| Chunk 2,048 | 625.5 | 23.07 | fewer per-chunk weight passes |
| Flash DSA attention | 688.4 | 23.39 | no K/V expansion, one pass |
| Absorption GEMVs | 691.6 | 24.29 | direct q·W_k, low·W_v |
| Fused mHC mix, Q4 epilogue, narrow RMS | 759.9 | 24.00 | one pass per row |
| 64-token expert items | 794.5 | 23.69 | weight tiles decoded once |
| 4,096-row chunks, lean attention buffers | 823.3 | 24.41 | workspace 12.0 → 7.1 GiB |
| F32 Q4_K prefill outputs | 861.4 | 24.40 | no output pass |
| Fused index scores | 874.3 | 23.42 | no per-key dot buffer |
| Shared expert in the reduce | 881.8 | 24.51 | two passes per layer removed |
| Expand fused with the next mix | 900.7 | 23.93 | residual read once |
| Shared prepared inputs | 920.4 | 24.43 | one Q8_1/BF16 copy per input |
| Tensor-core KDA chunks | 1042.3 | 24.49 | one operand record, F16 MMA scan |
| BF16 mHC residuals | 1081.3 | 23.72 | half the residual traffic |
| BF16 expert gate/down outputs | 1120.9 | 23.81 | 400 MB less per layer at 2,048 rows |
| Down input quantized in the up epilogue | 1132.8 | 23.78 | one pass removed |
| BF16 KDA gate rows | 1145.7 | 24.33 | no widening pass |
| Dependent decode launch | 1153.1 | 24.68 | kernels stage weights early |

Decode varies ±1.5% between fresh runs here; the prefill rows are the
measured trend. From the Q4_K v3 row on, the warm-up prefill matches the
chunk size. Rows from the BF16 residuals on were measured on 2026-10-17
(two-run mean for that row). The dependent launch was judged on 512-token
decodes at 2K, interleaved fresh runs: 24.55/24.78 → 25.09/25.12 tok/s.

MTP, same artifact. The 8K bench continuation is a repeating loop (draft
acceptance near 100% once the draft layer sees the prompt), so MTP gains
are reported on natural 3K-token prompts with 128 tokens (`60d0b689`; the
October 16 build in parentheses):

| Prompt | MTP off | 1 draft | 2 drafts | Tokens per verify (1 / 2) |
|---|---:|---:|---:|---|
| Prose (technical notes) | 25.20 (24.53) | 30.42 (30.42) | 32.92 (30.83) | 1.83 / 2.46 (1.80 / 2.33) |
| Code (C source) | 24.87 (24.04) | 33.06 (32.46) | 35.24 (35.48) | 1.97 / 2.72 (1.97 / 2.72) |

The tensor-core verify Q4_K then shortened a two-draft cycle from 76.8 to
72.7 ms on code (35.5 → 37.5 tok/s, same text) and from 75.3 to 71.8 ms on
prose (32.7 → 37.1 tok/s; its 128 tokens now equal the MTP-off decode, which
the SIMT verify had left at token 27 on a near-tie, so acceptance moved to
2.67 tokens per verify). Moving the expert gain after the dot took the code
cycle to 69.0 ms (39.5 tok/s, same tokens); one-row decode stayed at
24.8–25.3 tok/s with 512 identical tokens. The fused gate/up took the code
cycle to 68.9 ms (two interleaved pairs, 39.51 → 39.58 tok/s); its 0.3 ms
per decoded token stays inside the ±1.5% fresh-run spread.

## Measured performance

The [2K–64K README sweep](benchmarks/glm53-vq-spark-64k-2026-10-17/README.md)
(2,048-token incremental prefill, 128 tokens, three fresh processes) records
1,030.9 prefill and 30.04 decode tok/s with two MTP drafts (1,022.6 / 24.82
without), falling 5.6% in prefill from 4K to 64K. The
[October 16 sweep](benchmarks/glm53-vq-spark-64k-2026-10-16/README.md) recorded
826.7 / 31.41 (24.07 without); its continuations accepted more drafts (2.38
against 2.26 tokens per verify), so compare per-token speed on the MTP-off line.

## Remaining bottlenecks

- **Decode bytes and Q4_K rate.** Q4_K (4.4 GB per token) is half the decode
  time: one-token GEMVs stream at 160–264 GB/s by shape, routed experts at
  ~210 GB/s. At ~250 GB/s the 7.5 GB floor is ~30 ms; decode measures 40 ms.
- **Decode dispatch.** About 1,300 kernels per token. Dependent launch hides
  part of the gaps; CUDA graphs with device-resident positions remain (the
  DeepSeek VQ engine measured 43.9 → 32.5 ms per token).
- **Verify width.** A two-draft cycle (three verify rows) takes ~69 ms.
  Q4_K now costs about one token's pass; each verify row still adds nearly
  a full routed-expert pass at ~175 GB/s.
- **Prefill experts.** The FP8 kernel runs at ~25% of the FP8 tensor peak and
  is bound by shared-memory instruction pressure and barriers; at 2,048-token
  chunks the weight stream (~1.5 GB per layer) also sets a DRAM floor.

Measured and not kept, so they need a new idea before another attempt:

| Variant | Result |
|---|---|
| Tensor-core decode experts (E4M3 activations, `m16n8k32`) | 97 GB/s at one row vs 212 SIMT: per-lane index gathers, register spills |
| FP8 prefill experts: TMA copies of 128-byte token rows | bit-identical, 31% slower |
| FP8 prefill experts: 8 warps, register-prefetched indices | bit-identical, 50% slower (barriers, latency) |
| KDA local kernel: persistent CTAs / L2 prefetch of the next item | 11% / 25% slower |
| Selected-pool sort before flash attention | +0.7% prefill; logit RMS 0.79, KL 2.4e-6 (beyond the noise band) |
| Q4_K decode split capped at 18 KiB staging | 187 → 238 GB/s isolated on 4096×16384; decode −0.75% in the model |
| Vendored MMQ / MMVQ at verify widths | flat 150–190 GB/s, no better than the SIMT kernel |
| Verify down without mid-row groups (contiguous tasks per CTA, L1 reads) | bit-identical, 3–25% slower |
| Persistent grid / 576-byte stages for the verify Q4_K MMA | 1–2% / 6% slower (short strided row segments) |
