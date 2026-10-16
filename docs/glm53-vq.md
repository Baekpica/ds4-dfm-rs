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
wall; decode measures 41 ms. The DeepSeek V4.1 VQ engine reads about 6.3 GB
per token with the same Q4_K choice for its nonexpert weights.

## Execution on GB10

### Decode (one row)

- **Experts.** One persistent CTA per SM loads the layer book once and
  streams the eight selected experts' rows: 384-byte index blocks read as
  three full lines, codes distributed with warp shuffles, the 13th bit from
  its plane, gate/up and the clamped SwiGLU fused, down computed directly and
  experts reduced in ascending expert order. Routing stays on the GPU: a
  route kernel places selected experts from per-layer device tables, so
  decode no longer syncs with the host per MoE layer.
- **Non-expert projections.** Q4_K uses a cooperative stream kernel, FP8 a
  direct dot that rounds weights at the source BF16 boundary, BF16 a direct
  dot that reads the F32 activation row in place (the output head reaches
  250 GB/s). Small matrices split a row across warps.
- **mHC.** One kernel computes the 4-stream RMS and the 24 F32 mapping dots;
  the Sinkhorn split runs one lane per element and also applies the
  following RMS norm.
- **Attention.** KDA layers run the recurrence; DSA layers use the absorbed
  latent attention (`(Q·W_k) Cᵀ`, `(P·C) W_vᵀ`) over the selected keys:
  direct GEMVs for both absorptions and a single-pass flash kernel whose
  keys split across 48 CTAs, joined by a combine kernel.

### Prefill (2048-token chunks)

- **Experts on FP8 tensor cores.** The source model was trained and served
  with FP8 weights and dynamic 1×128 activation scaling. Codewords are E4M3,
  so `mma.m16n8k32.e4m3` takes them unconverted as the A operand; activations
  are quantized once per token and 128 channels; every 128-K block promotes
  its tensor-core partial sum to FP32 with its activation scale; the row gain
  multiplies the FP32 dot. The k32 reduction order is free as long as A and
  B agree, so lane q takes whole codeword 4k+q: one 8-byte book read fills two
  A registers and B is eight source-order bytes. Three cp.async stages fit
  beside the 64 KiB book in the 99 KiB opt-in shared-memory limit.
- **KDA.** A token-parallel prep reproduces the recurrence's q/k/v, BF16 beta
  and decay argument exactly; the delta-rule chunk (UT transform) kernels
  shared with Solar/GLM run on them. Appends under 64 rows keep the recurrence.
- **Attention.** One flash kernel per query row computes `softmax(q Cᵀ) C`
  over the latent rows with BF16 tensor cores, without expanding K/V.
- **Projections.** Q4_K uses the vendored MMQ path; FP8/BF16 restore to BF16
  and use cuBLAS.

### MTP (one or two drafts)

```
draft:  h(s-1), token s ──MTP layer──▶ d1 ──▶ d2 …     (layer 45, own KV)
verify: [token s, d1, …] as one n-row forward ──▶ n argmaxes
accept: keep k rows; KDA state ◀── journal slot k, index tail ◀── replay
```

- **One verify forward.** The n candidates run as one forward, so each
  weight is read once. Every kernel has a 1–4 row path that reuses each
  weight load and keeps the one-row arithmetic: verify rows equal decode
  rows bit for bit except the smallest Q4_K matrices and the attention
  split count.
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

MTP, same artifact. The 8K bench continuation is a repeating loop (draft
acceptance near 100% once the draft layer sees the prompt), so MTP gains
are reported on natural 3K-token prompts with 128 tokens:

| Prompt | MTP off | 1 draft | 2 drafts | Tokens per verify (1 / 2) |
|---|---:|---:|---:|---|
| Prose (technical notes) | 24.53 | 30.42 | 30.83 | 1.80 / 2.33 |
| Code (C source) | 24.04 | 32.46 | 35.48 | 1.97 / 2.72 |

## Remaining bottlenecks

- **Decode bytes and Q4_K rate.** Q4_K streams at ~211 GB/s and is half the
  decode time; routed experts run at ~190–200 GB/s.
- **Decode dispatch.** About 1,900 kernels per token; CUDA graphs with
  device-resident positions and programmatic dependent launch are the next
  structural step (the DeepSeek VQ engine measured 43.9 → 32.5 ms per token).
- **Verify width.** Each MTP row adds nearly a full routed-expert pass;
  multi-token Q4_K also slows past two tokens (L1/L2 input traffic).
- **Prefill experts.** The FP8 kernel runs near 60 TFLOPS with hot weights and
  is bound by shared-memory instruction pressure and barriers; at 2,048-token
  chunks the weight stream (~1.5 GB per layer) also sets a DRAM floor.
