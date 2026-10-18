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

## R13+8 residual stage (container v2)

The 1.625-bit experts were the main source of quality loss. Container v2
adds an optional 8-bit residual stage to selected layers:

```
w[r, 8j : 8j+8]  ≈  g_r · (C1[i1[r, j]] + C2[i2[r, j]])
C2: 256 × 8 E4M3 per layer; i2: one byte per vector (GVR8 tensor)
bits per weight = 13/8 + 8/8 = 2.625
```

- **Format.** `<matrix>.weight_residual` (type `0x47565238`, header `GVR1`)
  holds row-major index bytes: 512 per gate/up row, 256 per down row, whole
  128-byte lines. `glm_vq.layer.L.codebook.1` is the book (ID `4L+1`);
  `glm_vq.residuals` links each GVQ1 matrix to its residual. The parent's
  row gain scales the sum; the two E4M3 words add exactly in FP32. A layer
  is all-or-none, and only its part declares `glm_vq.format_version` 2.
- **Why 8 bits.** Layer 23, weight relative MSE: R13 0.1438; +8 bits 0.0471
  (0.0412 with joint beam assignment); +11 bits 0.0297. Per extra GiB the
  8-bit stage removes the most error, and its 2 KiB book fits beside the
  64 KiB layer book in every GB10 kernel (101,376 B opt-in): decode
  gate/up 76 KiB, decode down and 4-row verify 100,352 B, prefill
  98,560 B. Wider residual books would need a different staging scheme.
- **Decode and verify.** `RES` kernel variants load each block's 256 index
  bytes as two words per lane (`r0`, `r1`) beside the 12-bit words and take
  byte `k*32+lane` with one shuffle per k-step; the 2 KiB book is staged in
  shared memory after the layer book. Arithmetic: GAIN_DOT adds `C1+C2` per
  element before the FP32 dot and multiplies the gain once; GAIN_WEIGHT
  rounds `gain*(C1+C2)` to BF16 as the restore does. Verify rows keep the
  one-row order (bit-identical per row).
- **Prefill.** The FP8 tensor-core kernel stages 16 residual bytes per row
  per 128-K stage and issues a second `m16n8k32.e4m3` MMA per k32 step with
  the residual codewords as A and the same activation fragment as B, into
  the same partial sum. Tiles carry 64 tokens like R13 layers: staging the
  residual bytes as well would need 107,008 B, so each lane loads word q of
  its rows' 16 residual bytes one stage ahead into a register and the quad
  transposes them (98,816 B). The staged 32-token tiles (98,560 B,
  `DS4_GLMVQ_MMA8_T64=0`) were MIO-bound: twice the codebook gathers and B
  loads per token (NCU: 2.34x shared wavefronts). Both widths agree bit for
  bit; a 4K-token gate launch takes 8.0 ms (R13), 10.8 ms (64-token R13+8)
  and 15.3 ms (32-token R13+8). The BF16 register MMA has no residual stage:
  with `DS4_GLMVQ_MMA8=0` those layers take restore+GEMM.
- **Restore.** The scalar restore adds the residual; the vector restore
  declines R13+8 matrices so the scalar path keeps them exact.
- **Validation.** Rust checks GVR1 headers, offsets, checksums, padding,
  book links, parent shapes and versions (`residual_stage_is_validated`);
  `make test-glm-vq-res` checks restore bits, decode and verify against
  restore+GEMM and the FP8 prefill MMA against it on random E4M3 data.
- **Layer choice.** Measured end to end: upgrading one layer at a time
  (others R13) and measuring held-out KL against the original model ranks
  the bottom layers first (5: 0.066, 8, 6, 14, 3, then 11, 9, 15, 10, 16,
  7, 4, 12, 13), then 17, 41, 44, 27 and 22 (0.021–0.011). The release
  upgrades these 19 layers (77% of the summed single-layer gains); the
  previous revision (tag `r13p8-l3-16`) upgraded 3–16. A local routed-FFN
  screen normalized by each layer's FFN output energy chose 4, 5, 9 and
  25-40 instead and lost end to end (KL 0.498 vs 0.390-0.437 for
  bottom-heavy choices): FFN output energy spans three orders of magnitude
  across layers, so normalized local error does not rank layer impact.
- **Memory.** Nineteen R13+8 layers add 0.844 GiB each (16.0 GiB). At the
  native 1M context with two MTP drafts `ds4-server --check-config` quotes
  116.2 GiB (weights 88.0, bank 12.8, scratch 10.7, MTP 0.7, floor 4) of
  118.3 GiB available on GB10: one 1M bank with 4,096-row prefill chunks
  (the R13 release fits two). The five layers added to 3–16 were sized to
  keep those chunks.
- **Measured on GB10, 19 vs 14 layers.** Same binary: teacher-forced
  held-out PPL 4.042 → 3.895, lower on all six cases of the model card's
  set (5,089 targets); served answers to a long English context with a
  short Korean question loop on 0 of 10 seeds instead of 4; fast path vs
  `DS4_GLMVQ_DECODE=0 DS4_GLMVQ_MMA=0` KL 0.0035. Speed: 1,101.3 / 30.93
  vs 1,119.6 / 32.02 in the
  [2K–64K MTP 2 sweep](benchmarks/glm53-vq-spark-64k-19-layers-2026-10-19/README.md).
- **Measured on GB10, layers 3–16 vs R13.** `tests/glm_vq_quality` (6 cases,
  106 target tokens): mean target NLL 0.313 → 0.285, and the Korean summary
  R13 garbles is answered; fast path vs `DS4_GLMVQ_DECODE=0
  DS4_GLMVQ_MMA=0` KL 0.0038 (R13: 0.0035). Greedy "대한민국의 수도는…"
  loops under R13 (a repeated clause with `--think`; English reasoning, then
  "Namwon…" repetition with `--nothink`) and ends normally under R13+8 in
  both modes. Speed: 1,113.7 / 31.81 vs R13 1,175.4 / 35.00 in the
  [2K–64K MTP 2 sweep](benchmarks/glm53-vq-spark-64k-r13p8-2026-10-18/README.md).

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
  following RMS norm. Each thread loads its 16 columns of the four streams
  and their norm weights before warp 0's Sinkhorn, whose iterations then
  hide that latency (6.3 vs 11.6 µs per one-row call, bit-identical;
  `DS4_GLMVQ_SPLIT_PRELOAD=0` keeps the generic loop).
- **Attention.** KDA layers run the recurrence; DSA layers use the absorbed
  latent attention (`(Q·W_k) Cᵀ`, `(P·C) W_vᵀ`) over the selected keys:
  direct GEMVs for both absorptions and a single-pass flash kernel whose
  keys split across 48 CTAs, joined by a combine kernel.
- **Dependent launch.** A token runs about 1,300 short kernels. Decode and
  verify widths launch them with programmatic dependent launch: each kernel
  lets its successor start, which stages its constant weights (or prefetches
  them to L2) and then waits for its inputs. Device copies inside the chain
  are kernels too: a driver `cudaMemcpy` between two dependent launches hung
  two-draft prompt processing in most CLI runs. `DS4_GLMVQ_PDL=0` turns it off.

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
  The forget and gate projections stay BF16 rows for these kernels, and so do
  q/k/v when q4p computes them (the rounding decode and verify already keep).
  Appends under 64 rows keep the recurrence; `DS4_GLMVQ_KDA_MMA=0` restores
  the Solar chunks.
- **Attention.** One flash kernel per query row computes `softmax(q Cᵀ) C`
  over the latent rows with BF16 tensor cores, without expanding K/V. Eight
  warps (4 head groups × 2 latent halves) of 255 registers hold O and Q:
  sixteen warps of 128 registers spilled them (34 GB of local traffic per
  3,840-row call). Index scores are BF16 MMAs folded over the 32 heads in
  registers, so no per-key buffer exists and the context follows the
  model's native 1M.
- **mHC.** Each expand writes the new four-stream residual and accumulates the
  next half's RMS and 24 mapping dots in the same pass, with cp.async staging
  the next W tile and rows behind the current ones; the split collapses.
  The residual holds BF16-rounded values, so prefill stores it as BF16
  (32 KB per row instead of 64).
- **Projections.** Q4_K from 256 tokens runs on FP16 tensor cores
  (`cuda/glm_vq_q4p.cuh`): producer warps dequantize each weight once,
  `fp16(fp16(q·S) − M)` with `S = fp16(64·d·sc)`, `M = fp16(64·dmin·m)`, into
  the A fragment order; consumer warps run a plain FP16 GEMM on the FP16
  input (scaled by 2^-e only past 2^15). The vendored MMQ spent 74% of its
  instructions on per-k32 scales (29% tensor pipe); this reaches 82–84
  TFLOPS (8192 × 4096 × 4096: 4.48 → 3.37 ms) and is 30× closer to an FP64
  dot. Below 256 tokens, and with `DS4_GLMVQ_Q4_PREFILL=0`, MMQ keeps F32
  outputs and one Q8_1 pass per shared input (q/k/v, gate/up, q_a/kv_a).
  FP8/BF16 restore to BF16 and use cuBLAS, sharing one BF16 input copy.
  DSA q leaves q4p as BF16 rows; the DSA and KDA heads and the SwiGLU rows
  enter it as BF16 rows, prepared directly.

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
| Fused down quantization, scan staging, dependent launch, mHC split preload | identical order | bit-identical | identical |
| Eight-warp flash attention, staged expand-mix, up-epilogue gate loads, BF16 q4p input rows | identical order | bit-identical | identical |
| FP16 Q4_K prefill (q4p) | FP16 activations and dequantized weights instead of Q8_1 | RMS 0.26, KL 1.0e-6 | identical |
| BF16 KDA q/k/v rows from q4p | decode's BF16 output rounding | RMS 0.23, KL 9.7e-7 | identical (512 decode ids too) |

On the long-prompt cases (four prompts, 76 teacher-forced targets) q4p sits
closer than MMQ to a BF16 restore of the same Q4_K weights (mean KL 0.0024
vs 0.0045, paired NLL −0.0007 vs −0.012 nat/token), every argmax and answer
unchanged; a 4096 → 1024 chunk change alone reaches KL 0.0034.

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
| Eight-warp flash attention | 1205.4 | 25.54 | no register spills |
| FP16 Q4_K prefill | 1274.6 | 25.31 | weights dequantized once, FP16 MMA |
| BF16 KDA q/k/v rows | 1300.2 | 25.29 | half the q/k/v traffic |
| Staged expand-mix loads | 1300.9 | 25.58 | loads one W tile ahead (1290.4 in the same pairs) |
| Up-epilogue gate loads | 1296.6 | 24.93 | up kernel 791 → 751 ms in the model; end to end within noise (1298.5) |
| BF16 rows into q4p | 1343.8 | 25.40 | no F32 copies of BF16 rows (1298.8 in the same pairs, without the row above) |

Decode varies ±1.5% between fresh runs here; the prefill rows are the
measured trend. The flash, FP16 Q4_K and q/k/v rows are means of two interleaved fresh-process
pairs, against merged `main` (`c5384b5e`, 1148.6), against
`DS4_GLMVQ_Q4_PREFILL=0` (1200.9) and against F32 q/k/v (1281.6); a 2K
decode pair of 512 tokens stayed at 25.8–25.9 tok/s. From the Q4_K v3 row on, the warm-up prefill matches the
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

The [2K–64K README sweep after #88](benchmarks/glm53-vq-spark-64k-prefill-2026-10-17/README.md)
(`3dd964ba`; 2,048-token incremental prefill, 128 tokens, three fresh
processes) records 1,163.5 prefill and 35.08 decode tok/s with two MTP drafts
(2.32 tokens per verify), falling 7.1% in prefill from 4K to 64K; a whole 8K
prompt prefills at 1,343.8 tok/s. The [sweep at merged `main` before it](benchmarks/glm53-vq-spark-64k-main-2026-10-17/README.md)
(`c5384b5e`) recorded 1,042.2 / 35.12 (2.34 tokens per verify, 5.8% fall). The
[earlier October 17 sweep](benchmarks/glm53-vq-spark-64k-2026-10-17/README.md)
(`60d0b689`) recorded 1,030.9 / 30.04 (1,022.6 / 24.82 without MTP, 2.26
tokens per verify) and the [October 16 sweep](benchmarks/glm53-vq-spark-64k-2026-10-16/README.md)
826.7 / 31.41 (24.07 without, 2.38 tokens per verify); acceptance differs
with the continuations, so compare verify cycles (75.3 → 66.7 ms since
`60d0b689`) or MTP-off decode (25.9 tok/s at 2K on the merged code).

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
- **Prefill experts.** The FP8 kernel runs at ~30% of the FP8 tensor peak
  (tensor pipe 34%) and is 37% of an 8K prefill. Each 128-K stage needs
  ~2,900 shared-memory wavefronts (B fragments 35%, codeword gathers at ~3×
  their conflict-free cost 27%) against ~2,050 tensor cycles, but one barrier
  per stage aligns all 16 warps: gathers, then MMAs (stall samples: MIO queue
  29%, tensor pipe 18%, barrier 14%). Overlapping them needs a ring deeper
  than the 64 KiB book leaves room for, or fewer bytes per MMA.
- **Prefill input passes.** q4p still converts the shared F32 norm rows to
  FP16 in a pass of its own (81 ms per 8K prefill); the norm kernel could
  write them.

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
| FP8 prefill experts: producer warp + mbarrier ring, MT = 2 | bit-identical, 1–8% slower: a 288-thread block allocates registers as 384 (168 per thread, MT = 2 needs ~242), and the 64 KiB book leaves no room for a third stage |
| q4p: 4 stages / raw rows two stages ahead | 3–5% / 2% slower than 3 stages, one ahead |
| FP8 prefill experts: one n8 tile at a time (one live partial) | bit-identical, 4% slower |
| FP8 prefill experts: codeword halves stored apart (LDS.32 into the A registers, no MOVs) / copy addresses fixed per item | bit-identical, unchanged |
