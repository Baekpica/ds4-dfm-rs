# MiniMax-H3 VQ recipe — the FL2VA DiT at 13-bit GVQ1, everything else exact

[Documentation index](README.md) | [Repository README](../README.md) |
[Port plan](minimax-h3-port-plan.md) | [VQ calibration](vq-calibration.md) |
[GLM VQ](glm53-vq.md) | [Beam VQ](beam-vq.md)

Status: **proposed, nothing executed.** Branch `feature/minimax-h3`, base
`600e465e`. Every artifact number is read from the checkpoint's own index or
shard headers; every recipe number is quoted from a document, with its path.
The benchmark this design follows is the operator's own MiniMax-family release,
`Baekpica/MiniMax-M3.1-Flash-VQ-GGUF` at `74e5a9fd`; a local copy of its
recipe, format, budget and quality evidence is at
`/data/minimax-m3.1-vq-recipe/`.

## 1. What is being converted

| Input | Value | Source |
|---|---|---|
| Artifact | `MiniMaxAI/MiniMax-H3`, revision `42ed227ee7df40d41602854ae760620d6eb651fe` | HF model API |
| Partition | `fl2va` (tasks `t2va`, `fl2va`) | `FL2VA/model_index.json`, `_minimax_h3` |
| Download subset | `FL2VA/` = 144.05 GB, 81 files | the listing; plan section 3.3 |
| DiT | 13 shards, 535 tensors, 66,280,430,144 B = 33.14B params, **BF16** (45 of 54 tensors in shard 1; F32 for `rope.inv_freq`, `time_embedder.proj_in`, `audio_patch_proj`) | index `metadata.total_size`, shard-1 header |
| Text encoder | Qwen3-VL, 1058 tensors, 66,714,780,128 B | index |
| Video VAE | 10.42 GB; audio VAE 0.605 GB | listing |

DiT geometry (`FL2VA/transformer/config.json`): hidden 5376, 50 layers,
2 token-refiner layers, 56 heads × 128, FFN 14336, latents 24, audio latents
32, patch [1,2,2], `adaln_out_features` 96768.

Measured per-block shapes (shard-1 header):

| Tensor | Shape | Params | Share |
|---|---:|---:|---:|
| `blocks.N.attn.qkv_proj` | [21504, 5376] | 115.6M | 18% |
| `blocks.N.attn.out_proj` | [5376, 7168] | 38.5M | 6% |
| `blocks.N.mlp.fc1` | [28672, 5376] | 154.1M | 24% |
| `blocks.N.mlp.fc2` | [5376, 14336] | 77.1M | 12% |
| `blocks.N.adaln_proj.linear` | [96768, 2688] | 260.1M | **39%** |
| norms, biases, `rope.inv_freq`, `time_embedder`, `audio_patch_proj` [5376, 32] | — | ~0.6B | keep native |

Every quantizable matrix has a column count that is a multiple of 64
(5376 = 84·64, 7168 = 112·64, 14336 = 224·64, 21504 = 336·64, 2688 = 42·64),
which is what every container in this family requires. `audio_patch_proj` is
[5376, 32] and is not quantizable in these formats.

## 2. The precedent that governs the design

`Baekpica/MiniMax-M3.1-Flash-VQ-GGUF`, revision `74e5a9fd`, recipe id
`minimax-m31f-vq8-r13x57-r8x20-q4k-nvfp4idx-ssdkv-embssd-v1`. A 428B
multimodal MoE shrunk from a 236 GB checkpoint to 115.5 GB of GGUF parts, of
which 111.3 GB stay resident on one Spark, with 1,048,576 tokens of context.

### 2.1 Its expert settings, verbatim from `recipe.json`

| Setting | Value |
|---|---|
| Method | "YoungAi/Beam GVQ1 **v1**: 8-dim vectors along the input axis, one E4M3 codebook per layer (ID 4·L) shared by all 128 experts and w1/w3/w2, F16 row gains" |
| Index width | **13 bits** (K = 8192): 12-bit stream + 1-bit plane, on all 57 MoE layers |
| Residual stage | **R13+8**: second E4M3 book, K = 256, ID 4·L+1, one `GVR8` index byte per 8 weights, on 20 layers |
| Residual placement | "the layers with the largest calibration-measured end-to-end gain per byte" — chosen by a 57-single-layer-fork sweep, not by intuition |
| Calibration | `mode: imatrix`, `pool: stratified`, `col_power 0.75`, `col_floor 0.02`, `col_shrink 0`, `gain_refit: ls`, `init: equidistant`, `lloyd_iterations: 32`, `shrink_tokens 256`, train rows w1 40 / w2 80 / w3 40 per expert |
| Non-experts | Q4_K with per-column calibration importance (attention, shared experts, dense FFN 0–2, lm_head); Q8_0 for the indexer and drafter; F32 router; every norm native |
| Bits per weight | 1.63 on the 37 plain layers, 2.63 on the 20 residual layers, **1.98 average** |

### 2.2 Its residency contract — this is the part the operator asked for

The release is a set of independent GGUF parts, each declaring a `residency`
in `release.json`:

| Role | Part | Residency |
|---|---|---|
| trunk | 57 expert layers + `body-nonexpert.gguf` | `resident` (device) |
| embedding | `embed-tokens-bf16.gguf`, BF16, 2.29 GiB | **`ssd-rows`**: one 12,288-byte row read per token, never uploaded whole |
| vision | `vision-bf16.gguf`, original BF16, 1.61 GiB | **`mmap-on-demand`**: paged in only while an image or video is encoded |
| drafter | `dspark-q8_0.gguf` | `resident` |
| K/V of a session | NVFP4, 33.75 GiB at 1M tokens | **on SSD**, read block by block as the MSA indexer selects them; index keys stay resident in 4.22 GiB |

The design sentence worth quoting, because it is the whole idea: "The memory
this frees goes into expert precision rather than a large K/V cache." The
bulky, quality-critical or cold things are streamed from SSD at source
precision; the resident budget is spent on bits for the part that runs every
step.

### 2.3 What it measured, and what it admits

Teacher-forced over 60,431 held-out positions (12,191 in assistant turns), the
release bytes against the original:

| Arm | KL, assistant turns | Same top-1 | KL, all positions |
|---|---:|---:|---:|
| 13-bit on all 57 layers | 0.223 | — | 2.17 |
| + R13+8 on the bottom 20 MoE layers (3–22) | 0.171 | 87.2% | 1.71 |
| + R13+8 on the measured 20 layers (shipped) | **0.150** | **88.2%** | 1.98 |
| *floor: the original against itself, batched differently* | *0.003* | — | *0.565* |

Two methodological facts matter more than the numbers:

1. **A noise floor was measured first.** The original run twice with different
   GEMM batch sizes disagrees at KL 0.565 over all positions (top-1 84%); on
   assistant turns the same floor is 0.003. A gate written against the
   all-positions KL would therefore have been meaningless — 88% of the
   "error" there is the model's own chaos on positions it was never trained to
   predict.
2. **Weight-space error does not predict end-to-end quality.** The release's
   own `weight-error-by-layer.json` reports per-layer cosine 0.912–0.924 at
   13 bits — lower than a 12-bit dense bake's 0.984 — and still lands at KL
   0.150. The importance-weighted assignment and the least-squares gains do
   the work that raw reconstruction error does not capture.

And the release states its own limit in the card's first paragraph: "Lossy at
about 2.0 bits per routed-expert weight … Expect failures the source model
rarely shows: reasoning that loops, text that breaks mid-character or switches
language, weaker answers. Treat it as quantization research." That is the
operator's own verdict on a far lower bit rate than H3 needs, and it is the
honest frame for this document as well.

### 2.4 Kernel budget, which decides whether a rung is implementable

From the same card, against the GB10's 101,376 B per block:

| Stage | Codebooks | Decode gate/up | Decode down | Prefill MMA |
|---|---:|---:|---:|---:|
| 13-bit | 64 KiB | 77,824 B | 90,112 B | 96,768 B |
| 13-bit + R8 | 66 KiB | 79,872 B | 92,160 B | 98,816 B |

A 13-bit rung with the residual stage lands at 98,816 B of 101,376 B. There is
no room for a third stage at K = 8192; that is why the residual is a second
book of K = 256 and not more index bits.

## 3. What this means for H3 — and why H3 can afford more bits

The M3.1 bit budget is a **fit** budget: 428B parameters cannot be held in
BF16 on a 121 GiB host, so the experts had to reach ~2 bpw. H3 has no such
constraint. Its DiT is 33.14B parameters and 61.73 GiB in BF16, so it already
fits, phase by phase, with room to spare (plan section 4.4). The pressure on
H3 is *load time* (96.6 s measured for 66.28 GB) and *warm residency*, not fit.

That inverts the trade: H3 should buy **fewer bytes only where they are cheap**
and spend the savings on exactness everywhere else. Concretely:

| Tier | DiT | Text encoder | VAEs | Whole stack resident | Note |
|---|---|---:|---:|---:|---|
| BF16, as released | 61.73 GiB | 62.13 GiB (BF16) | 10.26 GiB | 124.1 GiB — does not fit | phases must be torn down; every render pays ~97 s for the DiT alone |
| **Chosen: 13-bit DiT, rest exact** | **6.75 GiB** | **62.13 GiB (BF16)** | **10.26 GiB** | **79.1 GiB — fits in 107 with 28 GiB spare** | one lossy element: the DiT weights; the prompt encoder and both decoders stay bit-exact |
| 13-bit + R8 residual on measured blocks | 7.7–9.4 GiB | 62.13 GiB | 10.26 GiB | 80–82 GiB — still fits | spends 1–2.6 GiB on the blocks a sweep ranks worst |
| 9/10-bit DiT (GVQ v2) | 4.7–5.2 GiB | 62.13 GiB | 10.26 GiB | 77 GiB | not needed for fit; only if the render gate says 13-bit is wasteful |

So the rung is **GVQ1 v1, dim 8, K = 8192, 13-bit (12-bit stream + 1-bit
plane), E4M3 codebook, F16 row gains, one book per matrix** (H3 is dense: the
shared book exists to serve 128 experts per layer, and H3 has one matrix per
class per block, so each gets its own book), fitted with `imatrix` weighting,
`gain_refit ls`, `col_power 0.75`, 32 Lloyd iterations. Arithmetic:
33.14B params at 13/8 bits = 53.9 GB… no — 32.5B quantizable params × 1.625
bits/weight / 8 = **6.60 GB**, plus 7.97M rows × 2 B of gains = 15.9 MB, so
**~6.6 GB / 6.15 GiB for the codes and 6.75 GiB with part overhead**. Against
66.28 GB of BF16 that is **10.0× fewer bytes**, and the DiT load goes from
96.6 s to ~10 s.

Why 13 bits and not 9/10: at 13 bits the format's own kernel budget still fits
the 99 KiB shared-memory limit (section 2.4), the M3.1 release proves the path
end to end at this width, and 9/10-bit is where the measured dense ladder loses
six to twelve cosine points. H3 does not need to go there.

The residual stage is kept as a **measured option**, not a default: the plan's
gate sweeps the 52 DiT blocks one at a time (the M3.1 method) and adds a GVR8
stage only to the blocks whose render metric improves most per byte. Budget:
+1 bit/weight on all 52 blocks = +2.6 GB; on the worst 20 = +1.0 GB.

## 4. Container and format

Use the M3.1 / Beam / GLM container unchanged: GGUF v3 with the GVQ1 v1 matrix
payload (magic `GVQ1`, 128-byte header, `[4sHBBIIIIIIQQQQ32s32s]`), one
`[K, 8]` E4M3FN codebook tensor per matrix (type `0x47564342`), F16 row gains
inside the payload, and the residual stage as `GVR8` (type `0x47565238`).

The header fields that matter, from `FORMAT.md` (local copy
`/data/minimax-m3.1-vq-recipe/FORMAT.md`): version 1, index bits 13 (12 is
also allowed), vector dimension 8, rows R, columns C, K = 8192, codebook
precision 1 = E4M3FN, codebook ID, flags 0, gain offset 128, low-12-bit offset
`align128(128 + 2R)`, high-1-bit offset `align128(low12 + R·C·3/16)`, total
bytes, and the two SHA-256 values (codebook, and every byte after the header).

```
W[r, 8j + d] = float32(gain[r]) * float32(codebook[index[r, j]][d])
```

The bit packing is not the GVQ v2 layout this repository's `vq-calibration.md`
describes: for adjacent indices `a, b` the low stream holds
`[a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255]`, and bit 12 of
index `j` is bit `j % 8` of byte `j // 8`. A converter that writes the v2
byte-major layout under a v1 magic would be rejected by the strict reader, so
this is a P4 gate item, not a detail.

Residual, container version 2: `W = gain[r] · (C1[i1][d] + C2[i2][d])`, the row
gain shared by both stages, the sum exact in F32, stage-2 target
`w/g − C1[i1]` assigned by column-weighted nearest against the stored book,
rounding `FP32 → F16 → E4M3FN`.

Per-shard parts, not one container: the DiT's 13 source shards become 13
independent GGUF parts each under 4 GiB of payload, listed in a `release.json`
with `role`, `residency`, `required`, `bytes`, `sha256`, `layer`, `variant`,
`container_version`, `index_bits`, `residual` — the M3.1 release shape. This
also solves the `DQVD` 4 GiB u32-offset ceiling without inventing a new
container.

## 5. What stays exact, and what is streamed

| Component | Treatment | Residency |
|---|---|---|
| DiT weights | GVQ1 v1 13-bit, per-matrix books | `resident` |
| DiT norms, biases, `rope.inv_freq`, `time_embedder`, `audio_patch_proj` | source BF16/F32 bytes | `resident` |
| Text encoder | **source BF16, unquantized** | `resident` in the warm daemon; `mmap-on-demand` when the process is cold and the job is a single render |
| Video VAE | **source BF16, unquantized** | `resident`; freed between jobs if the daemon needs the headroom |
| Audio VAE | **source BF16, unquantized** | `resident` |
| Anything that must be bit-identical | BF16 everywhere, plus the reference's own block streaming | see below |

The exact fallback is measured, not assumed: the reference's `--ssd-streaming`
keeps two DiT blocks resident and streams the next from SSD, and it is
documented as exact — 1.63 GB peak against 27.06 GB, **identical SHA-256 across
all runs and SSIM 1.0 against the resident baseline**, at +37.6%
load-plus-denoise on GB10 (`docs/GB10_PROFILE.md`). That is the operator's "SSD
stream and keep the quality" requirement, already proven inside the reference
engine. For the port it is the T0 path: stream BF16 blocks exactly and accept
the time, or quantize and accept a measured floor. Both stay implemented; the
CLI chooses.

## 6. Calibration

Step 1 is **nocal** (Lloyd on the weights alone, `X = NULL`) so the pipeline
works before any activation plumbing exists, and so the first rung is
reproducible by anyone from the checkpoint.

Step 2 is **imatrix**, which is what the M3.1 release actually ships and what
its numbers were produced with: per-token activation statistics at each
matrix's input, collected by the C reference on the Spark (there is no
PyTorch there — `import torch`, `transformers` and `safetensors` all fail), at
the shapes of the frozen prompt set, reduced per matrix, then fed to the
assignment and the gains with `col_power 0.75`, `col_floor 0.02`, stratified
pool and `gain_refit ls`.

Step 3, only if a block fails the render gate with imatrix weighting in place:
the GVR8 residual on that block (section 3), and the sweep that places it is
the M3.1 one — one block at a time, ranked by measured end-to-end gain.

## 7. The render gate

Metrics in increasing cost: weight-level (max abs, rel RMS, cosine — recorded
but **not** used as an accept criterion, per section 2.3), latent-level
(per-step latent relative error against the reference run), and render-level
(full-video SSIM, PSNR, LPIPS against the BF16 golden, per prompt).

The set: 8 prompts × 2 shapes (512×512 / 22 frames / 20 steps, and 1024×576 /
107 frames / 50 steps), pinned seeds, rendered by the reference binary so the
only variable is the weights.

Acceptance, mirroring the M3.1 method rather than inventing one:

1. measure the **floor** first — the reference against itself at two different
   batch sizes, exactly as the M3.1 card does;
2. accept a rung when every prompt meets the operator's SSIM floor and no
   prompt shows a documented composition, anatomy or colour regression, with
   the margin to the floor reported, not just pass/fail;
3. per block and per class (adaln / attn / mlp), so a failing class can be
   split out instead of costing the rung;
4. report the number of prompts whose content changed qualitatively, the way
   the M3.1 card reports "same top-1 88.2%" — a rate, not a verdict.

Proposed default floor: SSIM ≥ 0.99 full-video, which is stricter than the two
kernel-change precedents this tree accepted (0.999303 and 0.984985). The floor
is the operator's; plan section 10 asks for it.

## 8. What the port must implement, per rung

| Rung | Decode path | Reuse | New work |
|---|---|---|---|
| 13-bit | E4M3 codeword as the `mma.m16n8k32.e4m3` B fragment, 12-bit low stream + 1-bit plane decode, gain after the dot | `cuda/glm_vq_mma.cuh`, `glm_vq_mma8.cuh`, `glm_vq_gemv.cuh`, `glm_vq_decode.cuh` | a per-matrix book loader (dense), the 13-bit unpack, dense M×N tiling |
| 13+8 | the same plus a second 8-bit lookup per 8 weights, summed in F32 | `cuda/glm_vq_*` residual path (GLM-5.3 R13+8) | book-ID plumbing per block |

Documented constraints to respect, all from `AGENTS.md`'s GB10 section and the
M3.1 kernel table: 101,376 B per block (so 13+8 lands at 98,816 B); a codeword
is exactly one B fragment, one `LDS.64`, no MOVs; apply the gain after the dot;
promote every 128 K to F32; and read ptxas spills before tuning.

## 9. Cost and stop-loss

Encode: CPU-only, per shard, one Lloyd solve per matrix, 32 iterations,
7.97M rows; the M3.1 bake's own shape (4.18M calibration tokens, 89,729 source
tensors) is the scale reference. Gate cost: 16 renders per rung, the 1024×576
leg ≈ 19 min each on GB10, so ≈ 4.2 h per rung for that leg.

Stop-loss, fixed now: if 13-bit fails the render gate and imatrix weighting
does not close the gap, add the residual to the worst blocks (section 6 step 3)
before lowering any other component's precision; if that still fails, ship the
exact block-streaming path (section 5) and report VQ as not viable for this
model. Do not quantize the text encoder or the VAEs to rescue a DiT rung — that
is the opposite of what the precedent does, and the memory is already
sufficient.
