# MiniMax-H3 FL2VA — the DiT conversion recipe

[Documentation index](README.md) | [Repository README](../README.md) |
[Cheat sheet](minimax-h3-cheatsheet.md) | [Port plan](minimax-h3-port-plan.md)

Status: **recipe, nothing executed.** Branch `feature/minimax-h3`, base
`600e465e`. Every shape, dtype and byte count below was read from the
checkpoint's own index and shard headers over range requests; the constants
(`H3_*`) were read from the reference's `h3_host.h`. Nothing here is inferred.

Source: `MiniMaxAI/MiniMax-H3` at revision
`42ed227ee7df40d41602854ae760620d6eb651fe`, partition `fl2va`, component
`FL2VA/transformer/` (13 shards, 535 tensors, 66,280,430,144 bytes, BF16).

## 1. The DiT, tensor by tensor

535 tensors, in four groups. "VQ" marks the matrices this recipe quantizes;
"native" marks what stays exactly as shipped.

### 1.1 The 50 blocks (500 tensors, 32.27B params)

| Tensor | Shape | dtype | Params | Bytes | Decision |
|---|---:|---|---:|---:|---|
| `blocks.N.attn.qkv_proj.weight` | [21504, 5376] | BF16 | 115.6M | 231.2 MB | VQ |
| `blocks.N.attn.out_proj.weight` | [5376, 7168] | BF16 | 38.5M | 77.1 MB | VQ |
| `blocks.N.mlp.fc1.weight` | [28672, 5376] | BF16 | 154.1M | 308.3 MB | VQ (fused gate+up) |
| `blocks.N.mlp.fc2.weight` | [5376, 14336] | BF16 | 77.1M | 154.1 MB | VQ |
| `blocks.N.adaln_proj.linear.weight` | [96768, 2688] | BF16 | 260.1M | 520.2 MB | VQ (39% of the model) |
| `blocks.N.adaln_proj.linear.bias` | [96768] | BF16 | 96,768 | 193 KB | native |
| `blocks.N.attn.q_norm.weight` | [128] | BF16 | 128 | 256 B | native |
| `blocks.N.attn.k_norm.weight` | [128] | BF16 | 128 | 256 B | native |
| `blocks.N.norm1.weight` | [5376] | BF16 | 5,376 | 10.8 KB | native |
| `blocks.N.norm2.weight` | [5376] | BF16 | 5,376 | 10.8 KB | native |

Per block: 645,464,064 params, 1.2909 GB. Fifty blocks: 32,273,203,200 params.

### 1.2 The 2 token-refiner blocks (16 tensors, 0.77B params)

Identical matrix set without `adaln_proj` (both norms, no modulation):
`attn.qkv_proj` [21504, 5376], `attn.out_proj` [5376, 7168], `mlp.fc1`
[28672, 5376], `mlp.fc2` [5376, 14336], plus `q_norm` [128], `k_norm` [128],
`norm1` [5376], `norm2` [5376] per block. 385,351,680 params per block,
770,703,360 for the pair.

### 1.3 The head and the conditioning path (13 tensors)

| Tensor | Shape | dtype | Params | Decision |
|---|---:|---|---:|---|
| `final_layer.adaln_proj.linear.weight` | [10752, 2688] | BF16 | 28.9M | VQ |
| `final_layer.adaln_proj.linear.bias` | [10752] | BF16 | 10,752 | native |
| `final_layer.video_out.weight` | [96, 5376] | F32 | 516k | native (output head) |
| `final_layer.video_out.bias` | [96] | F32 | 96 | native |
| `final_layer.audio_out.weight` | [32, 5376] | F32 | 172k | native (output head) |
| `final_layer.audio_out.bias` | [32] | F32 | 32 | native |
| `final_layer.norm.weight` | [5376] | BF16 | 5,376 | native |
| `condition_proj.weight` | [5376, 5120] | BF16 | 27.5M | VQ |
| `condition_proj.bias` | [5376] | BF16 | 5,376 | native |
| `video_patch_proj.weight` | [5376, 96] | F32 | 516k | native (input, tiny, F32 path) |
| `video_patch_proj.bias` | [5376] | F32 | 5,376 | native |
| `audio_patch_proj.weight` | [5376, 32] | F32 | 172k | native (C = 32) |
| `audio_patch_proj.bias` | [5376] | F32 | 5,376 | native |

### 1.4 The time embedder and RoPE (6 tensors)

`time_embedder.proj_in` [5376, 256] F32 (1.376M params, 5.5 MB),
`time_embedder.proj_out` [2688, 5376] F32 (14.45M params, 57.8 MB),
their biases, and `rope.inv_freq` [16] F32. All native. `time_embedder.proj_out`
is the second largest native tensor and is F32 by design, not by accident.

### 1.5 The totals

| Group | Tensors | Params | BF16 bytes |
|---|---:|---:|---:|
| VQ matrices (260 of them) | 260 | 33,100,198,656 | 66.20 GB |
| Native | 275 | ~40M | ~80 MB |
| **DiT** | **535** | **33.14B** | **66.28 GB** |

Every VQ matrix has a column count divisible by 8 (all of them by 64), which is
what the payload needs: indices per row = C/8, and the 13-bit layout's byte
counts (`R·C·3/16` for the main stream, `R·C/64` for the plane) are integral.

## 2. The recipe applied to this DiT

**Format.** GVQ1 version 1, written per matrix: 128-byte header
(`<4sHBBIIIIIIQQQQ32s32s`), then the F16 row gains, the 12-bit main stream and
the 1-bit plane, with the E4M3 codebook in its own tensor. Value rule:

```
W[r, 8j + d] = float32(gain[r]) * float32(codebook[index[r, j]][d])
```

**Parameters for H3.**

| Parameter | Value | Why this value here |
|---|---|---|
| Vector dimension | 8 | one index codes 8 consecutive weights, so the same index width costs half what dim 4 costs; and an E4M3 codeword is then exactly one `m16n8k32` B fragment the MMA consumes without conversion |
| Codebook size K | 8192 (13 bits) | the widest book whose E4M3 form is 64 KiB, which leaves ~35 KiB of staging inside GB10's 99 KiB of shared memory |
| Codebook precision | E4M3FN | 1 byte per codeword: 64 KiB at dim 8 / K 8192, where F16 would be 128 KiB and not fit — and it is the form the GB10 MMA path wants |
| Index bits | 13, as a 12-bit stream plus a 1-bit plane | GVQ1 v1's layout, byte-identical with and without the residual stage; 14 bits would need a 128 KiB book |
| Residual stage | GVR8, K = 256, a second E4M3 book, one index byte per 8 weights | +1 bit per 8 weights on the blocks a per-block sweep ranks worst; the way to buy accuracy once the book stops fitting |
| Codebook scope | one per matrix (260 books) | H3 is dense: every matrix is large, so nothing is gained by sharing, and a per-matrix book removes cross-tensor interference |
| Row gains | one F16 per row, 8,115,968 rows | the format's only scale; 16.2 MB total, shared by a residual stage if present |
| Lloyd | 32 iterations, `init: equidistant` | deterministic by construction: no random source, same input yields the same bytes |
| Weighting | `imatrix` from H3's own activations | the assignment and gains are fitted against the statistics of the tokens this DiT will actually see |
| `col_power` / `col_floor` | 0.75 / 0.02 | damps the importance profile without letting a single column dominate |
| Gain refit | `ls` (least squares) against the stored indices | the gains are refit *after* the assignment, which is where most of the error is recovered |
| Pool | stratified, ≥ 1000 vectors per codeword | a 4× thinner pool cost 0.96% in the reference's measurement |
| Bits per class | **uniform** across all 260 matrices | the method's own negative result: dynamic bits per layer, expert or matrix was measured and uniform won |
| Residual stage | GVR8, K = 256, 8 bits, on chosen blocks only | +1 bit per weight where a per-block sweep says it pays; the only departure from uniform, and it is measured per block |

**Byte accounts** (arithmetic from section 1.5; 33.100B quantizable params,
8,115,968 rows, 260 matrices). The rate is `index bits / dim`, because one index
codes one vector of `dim` weights — so at equal index bits a dim-8 matrix costs
*half* what the same matrix costs at dim 4, and dim 4 only reaches a high bits
per weight by carrying more index bits.

| Rung | dim | K | Bits | bpw | Codes | Books (per matrix) | Artifact | GiB | Book in smem |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **A — the MiniMax-M3.1 shape** | **8** | **8192** | **13** | **1.625** | 6.723 GB | E4M3 64 KiB × 260 = 17.2 MB | **6.76 GB** | **6.29** | 64 KiB, fits |
| B | 8 | 4096 | 12 | 1.500 | 6.206 GB | E4M3 32 KiB × 260 = 8.6 MB | 6.23 GB | 5.80 | 32 KiB, fits |
| **A+8 — the recommended rung** | 8 | 8192 | 13 | 1.625 | 6.723 GB | + GVR8 K=256 2 KiB per chosen block | **7.0–7.3 GB** | 6.5–6.8 | 64 KiB + 2 KiB, fits |
| C — the dense bake's measured shape | 4 | 4096 | 12 | 3.000 | 12.413 GB | F16 32 KiB × 260 = 8.6 MB | 12.44 GB | 11.6 | 32 KiB, fits |
| D | 4 | 8192 | 13 | 3.250 | 13.447 GB | F16 64 KiB × 260 = 17.2 MB | 13.49 GB | 12.6 | 64 KiB, fits |
| E | 4 | 16384 | 14 | 3.500 | 14.481 GB | F16 128 KiB × 260 | — | — | **does not fit** |
| F | 8 | 16384 | 14 | 1.750 | 7.241 GB | E4M3 128 KiB × 260 | — | — | **does not fit** |
| + GVR8 residual | — | — | +1 bit per 8 weights, chosen blocks | +0.125 avg | +0.2 GB on 20 blocks, +0.54 GB on all 52 | +2 KiB per book | — | — | 2 KiB |
| BF16, as shipped | — | — | 16 | 16.0 | 66.20 GB | — | 66.28 GB | 61.73 | — |

**Rung A+8 is the one to build first**: the 13-bit GVQ1 v1 shape the operator's
MiniMax-M3.1 release ships, with its R13+8 residual stage on the blocks a
per-block sweep ranks worst. Three reasons, in order of weight:

1. **It is the only shape with published end-to-end evidence at exactly this
   width.** 428B parameters at 13 bits with an 8-bit residual on 20 measured
   MoE layers landed at KL 0.150 on assistant turns against a noise floor of
   0.003, with 88.2% identical next tokens, and the residual cut the per-layer
   weight error about threefold (relative MSE 0.150–0.175 to 0.051–0.068). That
   is a measurement on a MiniMax model, not an extrapolation.
2. **The book has to fit 99 KiB of shared memory, and the residual is how you
   buy bits once the book stops fitting.** A 14-bit book is 128 KiB at dim 8 and
   128 KiB at dim 4 — neither fits, and 128 KiB of F16 at dim 4 is what rung E
   costs. A second 256-entry stage costs 2 KiB per block and adds its bit only
   where a measurement says it pays.
3. **The E4M3 codeword is the form the GB10 MMA path consumes without
   conversion** — one codeword is exactly an `m16n8k32` B fragment, one
   `LDS.64`, no MOVs — whereas the F16 codewords rung C needs must be
   converted before the MMA. That is a decode-path cost on every token, in the
   opposite direction to the size.

Rung C stays in the ladder because it is the shape with the *dense* bake's
measured quality curve, and because a matrix at 3.0 bits per weight is a
genuinely different point on the rate–distortion curve: nearly twice the bits
per weight of rung A. If rung A+8 clears the render floor, C is unnecessary; if
it fails, C is the fallback that spends 5.7 GB more for more bits, before any
of the 14-bit shapes that cannot fit.

The rungs are not different formats: every one is the same payload with a
different header, a different book and optionally a second stage, so adding a
rung is a header change and an encoder argument, not new software.

**Where the bits are.** The four classes are not equivalent, and the plan's
gate reports them separately so a failing class can be split out:

| Class | Params | Share | Note |
|---|---:|---:|---|
| `adaln_proj.linear` (50) | 13.006B | 39.3% | drives every block's modulation; the largest single decision, and the one to watch |
| `mlp.fc1` (50) | 7.707B | 23.3% | fused gate+up, so its error reaches both branches |
| `attn.qkv_proj` (50) | 5.780B | 17.5% | |
| `mlp.fc2` (50) | 3.854B | 11.6% | |
| `attn.out_proj` (50) | 1.927B | 5.8% | |
| refiner + head + `condition_proj` | 0.826B | 2.5% | |

At 13 bits the adaln class alone is 2.64 GB of the 6.75 GB artifact, and the
attention class is 1.56 GB. Bits per class stay uniform — the method measured
dynamic bits per layer, expert and matrix and uniform won — so the levers when
a class misbehaves are the residual stage on that class's blocks and, if that
fails, leaving the class exact. Removing `adaln_proj` from quantization
entirely would take the artifact to 4.11 GB while keeping 39% of the DiT's
parameters bit-exact; that is the fallback if the render gate puts the failure
there, not a rung to start from.

## 3. The artifact this produces

**Parts, one per source shard.** Thirteen GGUF files, each holding the codebooks
and payloads of the matrices that came from that shard, each under 4 GiB of
payload, each independently readable:

```
layer-01-13  FL2VA DiT shard 1  (matrices of blocks 0..3 + patch/condition/time)
...
part-00-12   ...
```

Names, roles and the residency field follow the release shape the operator
already publishes: a `release.json` entry point listing
`{path, role, residency, required, bytes, sha256, layer, variant,
container_version, index_bits, residual}` per part, a `tensor-manifest.json`
giving every stored tensor as `{name, part, stored_shape, type, type_id,
payload_bytes, payload_sha256}` plus a `source_mapping` row for every one of the
535 source tensors exactly once, and a `recipe.json` recording the parameters of
section 2 verbatim. The native tensors ride in the same parts under their
original names, at their source dtype, so a reader has one file set, not two.

**Metadata per part** (self-describing, so a wrong reader fails loudly instead
of decoding garbage): `general.architecture` `minimax_h3_vq`,
`general.alignment` 128, and `minimax_h3_vq.{format_version, variant,
source_revision, recipe_id, part_role, codebooks, layer, index_bits,
weighting, residuals, residency}`.

**Container choice, and the ceiling that forces it.** The `DQVD` container's
offsets are u32 and its writer refuses more than 4 GiB, which a 6.76 GB DiT
exceeds. Per-shard parts solve that without inventing a container version: each
part is under the ceiling, and each carries its own hashes. The final artifact
for the runtime can be reassembled into one GGUF (its offsets are u64) or read
as the part set.

## 4. The residual stage, if the gate asks for it

`GVR8`: a second E4M3 codebook of K = 256 per chosen block, one index byte per
8-weight vector, the row gain shared with the first stage:

```
W[r, 8j + d] = float32(gain[r]) * (float32(C1[i1[r,j]][d]) + float32(C2[i2[r,j]][d]))
```

Stage 2 targets `w/g − C1[i1]` using the *stored* (least-squares) gain and the
*stored* stage-1 indices, so the two stages never disagree about what stage 1
already did. Placement is measured, not chosen: fork each block one at a time
against the 13-bit baseline, rank by the render metric's gain per added byte, and
arm the blocks above the significance threshold. The container version goes to
2 only for parts that carry residuals, and the link is all-or-none per part.

## 5. What stays exactly as shipped

275 tensors, about 80 MB: 210 normalization weights (all `norm1`, `norm2`,
`q_norm`, `k_norm`, `token_refiner.final_norm`, `final_layer.norm`), every bias
(including the 9.68 MB of `adaln_proj` biases), `rope.inv_freq`, the F32 time
embedder pair (63 MB), the F32 video and audio patch projections, and the F32
output heads. These are copied byte for byte into the artifact and are never
recomputed.

The text encoder (Qwen3-VL, 14 shards, 66.71 GB) and both VAEs (11.03 GB) are
also out of scope for this recipe: they stay BF16, exact. See the plan's
section 4.4 for why H3 can afford that and this model's M3.1 sibling could not.

## 6. Steps

### 6.0 Step 0 — get the checkpoint onto the Spark

The converter reads a directory of safetensors shards, so what has to be on the
host is the mode partition and nothing else. The card says the same thing in its
own words: the repository "hosts the original checkpoint (`FL2VA/`, `Ref2VA/`)
and the diffusers format side by side, so scope the download to what your
framework needs".

```bash
# on the Spark, as the model owner
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "FL2VA/*" --include "model_index.json" --include "README.md" \
  --local-dir /mnt/h3/MiniMax-H3
```

Run 2026-10-24: 85 files, about twenty-five minutes, 135 GiB on disk. Three
things worth knowing before repeating it: the `hf` CLI needs **no login** for
this repository (it is public and not gated; the token warning it prints only
concerns rate limits), `--revision` is pinned so a re-run reproduces the same
bytes, and `hf download` resumes, so an interrupted run is restarted with the
same command rather than re-fetched.

The layout it produces, which is what the converter and the reference both
expect:

```
/mnt/h3/MiniMax-H3/
├── model_index.json
└── FL2VA/                    144.05 GB, 81 files
    ├── model_index.json        partition, tasks t2va/fl2va, sigma_shift_scales
    ├── transformer/            13 shards + config + index   <- the converter's input
    ├── text_encoder/           14 shards + config           <- the render gate's input
    ├── video_vae/source/       + the model's own .py sources
    ├── audio_vae/              + the model's own dac_*.py sources
    ├── processor/  tokenizer/  required by the reference, not by the converter
```

Verify, without a script and without Python:

```bash
ls /mnt/h3/MiniMax-H3/FL2VA/transformer/*.safetensors  | wc -l   # 13
ls /mnt/h3/MiniMax-H3/FL2VA/text_encoder/*.safetensors | wc -l   # 14
find /mnt/h3/MiniMax-H3 -name '*.incomplete' | wc -l             # 0
du -sh /mnt/h3/MiniMax-H3                                        # 135G
grep -o '"total_size":[0-9]*' \
  /mnt/h3/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json  # 66280430144
```

That last number is the DiT's own account of itself: 66,280,430,144 bytes over
535 tensors, which is the figure every byte calculation in this recipe is
checked against.

Space: `/mnt/h3` is 469 GB, and the checkpoint leaves ~310 GB free, which covers
every rung in section 2 several times over. The full repository would need
498 GB and does not fit — the `Ref2VA` partition and the diffusers-layout root
are the same weights under different names (section 8.2 of the plan), and
neither is read by anything in this recipe.

One prerequisite that belongs to the render gate rather than to the conversion:
the C reference needs `libicu-dev`, which is not installed on the Spark.

### 6.1 Inventory
   with dtype, shape, data offsets and shard, and reconcile its total against
   `metadata.total_size` = 66,280,430,144 B. Gate: exact match, zero unknown
   tensors, and the class table of section 1.5 reproduced.
### 6.2 Plan

Emit the per-matrix table: name, shape, rows, columns, indices per
   row, codebook bytes, main-stream bytes, plane bytes, gain bytes, total
   bytes; print the per-class subtotals and the artifact total. Gate: the
   arithmetic reproduces 6.756 GB at 13 bits before any encoding happens.
### 6.3 Statistics (imatrix)

The reference is instrumented to dump the input
   activation of each VQ matrix over the frozen prompt set (see the plan
   section 5.1 for the fixture method and its ≥ 99.996% bit-match check),
   then reduce to per-matrix column statistics. Gate: every one of the 260
   matrices has statistics, and the fixture reproduces the reference's own
   block output bit for bit on the check rows.
### 6.4 Encode

Per shard, per matrix: row-RMS normalize, 32 Lloyd iterations
   with importance-weighted assignment, least-squares gain refit, final
   assignment against codebook and gains rounded to their stored precision
   (F16 gains, E4M3 codewords). Gate: determinism — re-encode a shard and
   compare bytes.
### 6.5 Assemble

Write the parts, the codebook tensors, the gains, the
   streams, the manifest and the `release.json`. Gate: every payload hash,
   every byte account, and every source tensor accounted for exactly once.
### 6.6 Verify

Read each part back with a strict reader implementing the
   header rules, decode every payload, and compare against the source tensor.
   Gate: max abs, relative RMS and cosine per matrix and per class; the reader
   must reject a mutated payload, a bad hash, a non-canonical offset and a
   truncated part (test each).
### 6.7 Render

The gate in the plan section 5: eight prompts, two shapes,
   the reference binary, SSIM/PSNR/LPIPS per prompt against the BF16 golden,
   per class, with the floor measured first.

## 7. Cost

Encoding is CPU-only and per matrix: 260 matrices, 33.1B parameters, 8.12M rows,
32 Lloyd iterations. The dominant cost is the assignment step: every row is
compared against 8192 codewords per iteration, i.e. ~8.12M × 8192 × 8 × 32
≈ 1.7 × 10^13 operations, which is a few hours on 20 cores with the codebook
streamed through shared memory and the running argmin in registers (never
materializing the distance matrix). Statistics collection and the render gate
are the other two costs; the plan's section 7 gives their measured scale.

Nothing in this recipe touches the source checkpoint, and every step is
re-runnable from the pinned revision.

## 8. Sources and provenance

Which part of this document comes from where. The distinction matters: the
**inventory and geometry are H3's own artifact**, read first-hand; the
**parameters are the published recipe of this toolchain**, not something
derived for H3; and the **byte accounts, token tables and tier totals are
arithmetic** over both, marked [A] where they appear elsewhere in the plan.

### 8.1 H3's own artifact (read first-hand)

| What it gave | Source | Revision |
|---|---|---|
| 280 files, sizes, LFS digests, the `fl2va` partition being self-contained | HF model API `https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3?blobs=true` | `42ed227ee7df40d41602854ae760620d6eb651fe`, last modified 2026-08-13 |
| 535 DiT tensors, class counts, all 18 non-block names, `metadata.total_size` 66,280,430,144 | `FL2VA/transformer/model.safetensors.index.json` | same revision |
| Per-block shapes and dtypes (45 BF16 + 9 F32 in shard 1), patch/condition/time tensors | `FL2VA/transformer/model-00001-of-00013.safetensors` header, read over HTTP range requests | same |
| `final_layer` shapes (adaln [10752, 2688], video_out [96, 5376] F32, audio_out [32, 5376] F32) | `model-00013-of-00013.safetensors` header | same |
| Geometry: hidden 5376, 50 layers, refiner 2, 56 heads × 128, FFN 14336, latents 24, audio latents 32, patch [1,2,2], `adaln_out_features` 96768 | `FL2VA/transformer/config.json` | same |
| Partition, tasks, `sigma_shift_scales`, `schema_version` | `FL2VA/model_index.json` | same |
| Text encoder: Qwen3-VL, 64 layers, hidden 5120, 64 q / 8 kv heads, vision 27 blocks at hidden 1152 | `FL2VA/text_encoder/config.json` | same |
| VAE contract: 24 latent channels with means and stds, clip 17, token drop 3, tile 256 / overlap 64 | `FL2VA/video_vae/config.json` | same |
| Audio VAE: 32 latent channels, 32 kHz, stereo | `FL2VA/audio_vae/config.json` | same |
| Text encoder inventory: 1058 tensors, 66,714,780,128 bytes | `FL2VA/text_encoder/model.safetensors.index.json` | same |
| The VAEs' reference implementation, used as the port's specification | `FL2VA/video_vae/klvae.py`, `vae_cnn.py`, `vae_vit.py`, `minimax_h3_video_vae.py`; `FL2VA/audio_vae/dac_*.py` | same |

### 8.2 The reference engine

| What it gave | Source | Revision |
|---|---|---|
| Validated platform (ARM64, GB10, driver 595.84, CUDA 13.0), CLI surface, the `5 + 17n` frame rule, the canvas list and the `768 × 1344` pixel ceiling, the `8 × 8` token grid at 256², `--token-reduction` behaviour | `/data/h3.c` `README.md` | `eae5d54e3a0419064b64171dcaebcfd92e9f3f63` (`JLFN/h3.c`, origin `https://github.com/JLFN/h3.c.git`, branch `main`, 2026-08-31) |
| Load 96.563 s, denoise 2.715 s, SSD streaming 51.829 / 84.798 s at 1.630 GB peak with identical SHA-256, VAE 1112.960 → 247.737 s, denoise 815 s, wall 33:36.8 → 18:56, latent `32×36×64×24` | `/data/h3.c` `docs/GB10_PROFILE.md` | same |
| Required files inside the mode partition; Ref2VA optional | `/data/h3.c` `h3.c:428-433` and `h3.c:438-451`; `Makefile` test targets | same |
| `H3_CANVAS_MULTIPLE` 32, `H3_MAX_PIXELS` 768×1344, `H3_FPS` 24, `H3_AUDIO_LATENT_FPS` 40, `H3_VAE_SPATIAL_RATIO` 16 | `/data/h3.c` `h3_host.h:7-11` | same |
| `latent_t = ((frames-5)/17)*5 + 2`, `h3_temporal`, `h3_latent_canvas`, `h3_adapt_canvas` | `/data/h3.c` `h3_host.c:25-70` | same |
| The 22-frame decoder chunk requirement | `/data/h3.c` `h3.c:867-869` | same |

### 8.3 The method and the formats

| What it gave | Source | Revision |
|---|---|---|
| The 8-dim vector-quantization method with one codebook per layer, F16 row gains, GPU Lloyd with equidistant init and double atomics, the final assignment against stored precision, the 12 + 1 bit-plane layout and its reason (128-byte line fills), E4M3 codewords at +0.12% error, codebook sharing at +0.15% average error, the ≥ 1000-vectors-per-codeword pool, the convex error curve and spreading bits thin, "zero corpus" | `/data/YoungAi` `docs/TECHNICAL.md` §4.1 | `28944791d64c55b14e83ce0f5326ca5ebaac226e` (branch `feat/quality-inspection`, 2026-10-21). The dense bake cites the encoder at `a4e6d3177f310cabf3e8d5d214d437a7068ddf23` |
| The multiplicative-gain repair (反修): per-expert output-channel gains + router bias, Gauss–Seidel with a ridge on mean channel energy, the FP4 lattice-direct solve, the ≥ 0.5% held-out mount gate, ~40 MB and ~1 h per domain | `/data/YoungAi` `docs/TECHNICAL.md` §4.2 | same |
| Its measured quality tables (top-1 71.80 → 74.38 etc., KL −17 to −30%) | `/data/YoungAi` `docs/TECHNICAL.md` §5.1 | same |
| The rejected families: additive low-rank, bias/mean correction, codebook re-solve, dynamic bits per row, entropy coding | `/data/YoungAi` `docs/TECHNICAL.md` §6 | same |
| The DQZ2 runtime sidecar and its collapse to one per-token scalar (`F = shared_out + λ(x)·Σ ge·y_e`), five op types, `--zchain <dir>`, absent file = today's path | `/data/YoungAi` `ds4_zchain.h` | same |
| Version-3 payload layout (`DQVQ` / `DQV3`, one book per layer in the blob header, E4M3 only, 13-bit as 12-bit stream + 1-bit plane, version whitelist as a hard stop) | `/data/YoungAi` `vq_fmt.h` | same |
| The GVQ1 v2 container (dim 8, 9/10-bit, E4M3 books, byte accounts, weighting modes, gain refit) and the A.X-K2 release settings (9 bits per 8 weights, 10 on seven layers; `imatrix`, `ls`, `split3`, `col-power 0.75`) | `/data/ds4-dfm-rs` `docs/vq-calibration.md` | `600e465e` (= `origin/main` = `dev`) |
| The GB10 kernel facts used to size the rung: 101,376 B per block with 99,344 B opt-in, an E4M3 codeword being exactly an `m16n8k32` B fragment (one `LDS.64`), gain applied after the dot, promote every 128 K to F32, cuBLAS BF16 65–90 TFLOPS against ~108 TFLOPS F16/BF16 and ~215 TOPS FP8/INT8, "attribute first" | `/data/ds4-dfm-rs` `AGENTS.md`, GB10 section; kernel inventory in `cuda/` (`glm_vq_*.cuh`, `ds41_vq_*.cuh`) | `600e465e` |
| The VQ/GB10 budget analysis the plan's byte accounting follows | `/data/ds4-dfm-rs` `docs/axk2-vq-gb10-fit.md` | branch `research/vq-calibration-lit`, unmerged at the time it was read |
| The plan's document structure and gate discipline | `/data/ds4-dfm-rs` `docs/deepseek41-port-plan.md` | `600e465e` |

### 8.4 The MiniMax-family VQ release (the operator's own precedent)

| What it gave | Source | Revision |
|---|---|---|
| The recipe under which a MiniMax model was actually published: GVQ1 v1, dim 8, K 8192, 13-bit, per-layer E4M3 book, F16 gains, R13+8 residual on 20 measured layers, `imatrix` / `stratified` / `col_power 0.75` / `col_floor 0.02` / `gain_refit ls` / equidistant init / 32 Lloyd iterations | `Baekpica/MiniMax-M3.1-Flash-VQ-GGUF`, `RECIPE.md` and `recipe.json`; local copy `/data/minimax-m3.1-vq-recipe/` | `74e5a9fd12fb57a5c1babb559536cf10733939ea`, last modified 2026-10-23 |
| The GVQ1 header field by field, the low-12 stream packing, the high-1 plane bit order, the two-stage residual value rule, the GGUF type ids, the part roles and `residency` values, the metadata keys, the rejection list | the same release, `FORMAT.md` | same |
| Measured fidelity (KL 0.150 assistant turns, same top-1 88.2%, the floor of 0.003), the self-declared lossiness warning, the residency table, the SSD read behaviour, the kernel budget table against 101,376 B | the same release, `README.md` | same |
| The 117.2 GiB available basis and the row-by-row memory budget with its `basis` column (exact byte layout vs estimate) | the same release, `memory-budget.json` | same |
| The alternative arms and the noise floor | the same release, `quality/release-r13only/result.json`, `quality/release-bottom/result.json`, `quality/noise-floor.json`, `quality/r8-sweep/summary.json`, `quality/kv/kl.json` | same |

### 8.5 The dense VQ bake

| What it gave | Source |
|---|---|
| The per-matrix codebook precedent on a dense model: dim 4 × K 4096 f16, 3.0 bits/weight, corpus-free Lloyd, byte-verified container, 14-way sharded encode, and the measured weight/quality ladder (cosine 0.984–0.999 at 3.0 bpw against 0.914–0.950 at 1.5) that the plan uses to justify starting at the quality end | `/data/embeddinggemma2/RECIPE.md` sections 2, 5 and 5b |
| The requirement that the deliverable carry no Python | project rule 14 in `~/.opengrok/project-work-rules.md` |

### 8.6 Environment measurements (this session, on the Spark)

| What it gave | Source |
|---|---|
| GB10 at compute capability 12.1, driver 580.178.04, CUDA 13.0 `V13.0.88`, `compute_121` buildable, gcc 13.3.0, rustc 1.98.1, ffmpeg 6.1.1, `hf` 1.28.0, 121 GiB total / 111 free, `/` 547 G free, `/mnt/models` 393 G free, `libicu-dev` missing, no `docker`, no PyTorch, HF reachable | read-only probe scripts run in the operator's ssh tmux pane, output `~/h3probe.out` and `~/h3probe2.out` on the Spark |
| The two block devices (931 GiB NVMe, 953 GiB USB) | `sysfs` over the gvfs sftp mount |
| `/mnt/h3` exists, is a filesystem root created 2026-10-24 09:44, size unconfirmed | the same mount |

### 8.7 What is not from any source

The byte accounts of section 2, the class shares of section 2, the token and
FLOP tables and the tier totals in the [cheat sheet](minimax-h3-cheatsheet.md),
and the projected load times at each rung are **arithmetic over the sources
above**, not measurements. They are labelled [A] wherever they appear. The only
measured numbers in these documents are the ones quoted with a source, plus
whatever P2 replaces the estimates with once the reference is built on the
Spark.

## 9. The converter is mostly already written

The dense bake at `/data/embeddinggemma2` ships a working converter, and the
H3 job is a smaller change to it than writing anything new. Its include set is
the whole dependency list:

```c
#include "vq_qc.h"          /* vq_encode_full, vq_pack, vq_unpack_dequant */
#include "quants.h"         /* ds4q_quantize_chunk (q4_K) */
#include "ds4_st41.h"       /* v41_st_*: multi-shard safetensors index */
#include "ds4_float.h"      /* ds4_bf16_to_f32 */
#include "ds4_quantfmt.h"   /* ds4_deq_q4_K, the gold decoder */
```
`tools/eg2_pack.c:213-217`

The load-bearing find is the third line. `ds4_st41.h`
(`/data/YoungAi/src/common/`) is a multi-shard safetensors index:
`v41_st_open(S, dir)` scans every `*.safetensors` in a directory and builds
name → (shard, offset, dtype, shape) with a binary search
(`ds4_st41.h:132-163`), and `v41_st_data()` **mmaps one shard at a time, on
demand** (`ds4_st41.h:169`). That is H3's `FL2VA/transformer/` layout exactly —
13 shards plus `model.safetensors.index.json` — so the converter can read the
checkpoint in place, matrix by matrix, without staging 66 GB anywhere and
without a new reader.

### 9.1 Reuse map

| Artifact | What it is | For H3 |
|---|---|---|
| `tools/eg2_pack.c` (64 KB) | the whole driver: `plan`, `encode`, `verify`, `merge`, `check`; the DQVD container writer (magic, version, 64-byte header, 48-byte slots, 16-byte alignment); the slot table; `build_plan`, `plan_report`, `plan_tsv`, `layout`, `write_header`, `write_slot_table`, `emit_payload`, `payload_metrics`, `merge_run`, `check_container`; and the flags `--cold cls,…`, `--limit N`, `--only SUBSTR`, `--shard i/N`, `--profile` | copy to `tools/h3_pack.c` and change four things: `classify()`, `name_is_kept_matrix()`, the `dim_of`/`nc_of` rung table, and `recipe_string()` |
| `tools/eg2_vq.c` | standalone encoder with its own metrics and a dequant oracle; CLI `tensor <hf-dir> <name> [dim nc]` | reuse as-is to measure a single H3 matrix across rungs before committing to one |
| `tools/ds4_gguf_write.h` (7.8 KB) | a complete GGUF v3 writer in one header: `ds4gw_begin/put/kv_u32/kv_str/kv_arr_*/tensor/header/write/end` | reuse verbatim for the artifact wrap |
| `tools/eg2_artifact_check.sh` | the four-sided artifact check: writer manifest against the reader's own inventory, container payloads against GGUF bytes, verbatim tensors against the checkpoint's bytes, and the payload magic | adapt the five paths and the name list |
| `tools/eg2_gguf.c` (69 KB) | the GGUF re-wrap of the DQVD container, with a manifest mode | adapt: H3's metadata keys and native-tensor list |
| `/data/YoungAi/src/common/ds4_st41.h` | multi-shard safetensors index, mmap per shard | verbatim |
| `/data/YoungAi/src/common/ds4_float.h`, `ds4_quantfmt.h`, `ds4_sha256.h` | BF16 conversion, the gold decoders, hashing | verbatim |
| `/data/YoungAi/gguf-tools/quants.c` + `quants.h` | the reference block quantizers (`ds4q_quantize_chunk`) | only if a native tensor ever needs re-encoding; H3's natives are copied verbatim |
| `/data/YoungAi/vq_qc.h` | the VQ encoder itself: `vq_encode_full`, `vq_pack`, `vq_unpack_dequant` | verbatim; note `g_vq_dim`/`g_vq_nc` are extern and zero by default, which is the documented setting for a base with no sidecar path |
| `bin/eg2_pack_cpu` | the CPU build the bake actually ran | build H3's the same way: `cc -O3` of the driver against those objects, no CUDA in the encode path |

### 9.2 What H3 has to add

Four edits in the driver, one table, one command-line change:

1. `classify()` (`eg2_pack.c:329`) returns H3's classes — `adaln`, `fc1`,
   `qkv`, `fc2`, `out`, `refiner`, `head`, `native` — from the tensor name.
2. `name_is_kept_matrix()` (`eg2_pack.c:312`) lists the 275 native tensors of
   section 5; everything else that is a rank-2 matrix with
   `cols % 8 == 0` and not `audio_patch_proj` is VQ.
3. The rung table replaces `dim_of`/`nc_of` (`eg2_pack.c:362-363`), which today
   choose per class from a cold mask. For H3 the rung is uniform across all 260
   matrices and selected on the command line — rung C is `dim 4, nc 4096,
   bits 13, F16 codewords` — with the residual stage as a per-block flag.
4. `recipe_string()` (`eg2_pack.c:378`) records the H3 recipe id, and the
   native policy string lists what was left exact.

The sharding that the 4 GiB container ceiling needs is not new work: `--shard
i/N` and the `merge` subcommand exist, and H3 has exactly 13 natural shards.
`--only SUBSTR` and `--limit N` are the tool's own experiment path and are what
the rung ladder is explored with, one matrix at a time, before a full encode.

### 9.3 What is genuinely new work

1. The statistics-to-imatrix step for H3 (section 6 step 3), which the dense
   bake ran corpus-free and which needs the reference's activation dump here.
2. **The E4M3 codebook path and the residual stage in the writer.** The
   transferred converter writes F16 codebooks in the DQVQ payload layout
   (`vq_qc_bytes.h:20`) — that is rung C, and it is what has been built and
   exercised so far. Rung A needs the GVQ1 v1 13-bit layout with an E4M3 book;
   rung A+8 needs the 256-entry second stage and the two-stage value rule. Both
   are format work in the writer and the matching reader, and both are
   prerequisites for the recommended rung rather than optional extras.
   **Done, in `tools/h3_vq/bake/h3_gvq.{c,h}`** (branch `feature/minimax-h3`):
   the GVQ1 v1 writer and strict reader, the GVR1 residual payload, and the
   encoder with the parameter set of section 2. Gate: `bin/h3_gvq selftest`
   passes — the byte account, the 13-bit round trip, the value rule against an
   independent oracle, byte-equal payloads across thread counts, a stage-1
   payload unchanged by the residual stage, 20 GVQ1 and 5 GVR8 rejections, and
   all three padding gaps exercised. The `imatrix` column profile is an
   argument the statistics step (section 6 step 3) will fill; with no
   statistics the encoder runs the reference's own zero-corpus mode.
3. The per-block residual sweep (the M3.1 method): convert one block at a time
   to the residual stage, re-run the held-out measurement, take the blocks with
   the largest drop. Its counterintuitive result is worth carrying — the middle
   of the stack won, and the rule of thumb that protects the bottom layers
   measured *worse* on the same byte budget (KL 0.171 against 0.150). For H3
   the analogue is per DiT block, measured rather than assumed.
4. The `pq`/render gate harness: the reference binary, the frozen prompt set,
   the SSIM/PSNR/LPIPS measurement and the per-class breakdown. The bake's
   analogue (`eg2_retrieval_score.sh`, `eg2_phase5_gate.sh`) is a scoring
   harness over a frozen input set, so the shape is there to copy, but the
   metrics are H3's.
5. The runtime decode kernel in the port, as listed in the plan's P5 and P7.

That is the whole difference between "a new converter" and "a converted
converter": a classify table, a name list, a rung table, a recipe string, two
format additions (the E4M3 book and the residual), and the measurement harness.

## 10. Running it anywhere: the encode contract

The branch carries a complete CPU encoder, and it is the reference a GPU
implementation has to match. Nothing else is needed: no CUDA, no Rust, no GPU,
no prior build of anything.

```
tools/h3_vq/build.sh                                    # cc -O2 -lm -lpthread
tools/h3_vq/bin/h3_gvq selftest                         # the format gate, ~2 s
tools/h3_vq/bin/h3_gvq encode <FL2VA/transformer> <tensor> <prefix> \
    --iters 32 --residual
tools/h3_vq/bin/h3_gvq check <prefix> --src <FL2VA/transformer> --name <tensor>
```

Input: the five paths `h3.c:428-436` requires (section 8.1). The encoder takes
any 2-D matrix with `cols % 64 == 0` in BF16, F16 or F32 through the
safetensors index, so the 13 DiT shards are used exactly as published. Nothing
is pre-converted and no plan file is needed for a single matrix.

The parameters are section 2's: dim 8, K 8192, E4M3 book, 32 Lloyd rounds, a
pool of 1000 vectors per codeword (capped at K x 1000 = 8,192,000 vectors per
matrix), equidistant init, least-squares gain refit rounded to F16, and the
GVR8 residual with 32 rounds of its own. `selftest` proves the format
independently of any checkpoint; the encode commands prove the data path.

### 10.0 What a conversion-only host needs

Converting is standalone: no serving code, no CUDA, no port, no GPU when the CPU
path is used. An H100 host needs only:

- `FL2VA/transformer` from the checkpoint: 13 shards, 66.28 GB, plus
  `config.json` and `model.safetensors.index.json`. All 260 VQ matrices live
  there. The text-encoder shards and the VAEs are only needed for the render
  gate, not for the conversion, so a conversion host does not need the full
  144 GB partition (section 8.1, `h3.c:428-436`).
- ~25 GB of output: 22.3 GB of payloads plus the books, and the reports.
- Disk headroom for the checkpoint's page cache is not required; the tool mmaps
  one matrix at a time and widens it to F32 (1.04 GB for the largest class).

Two routes, and the choice is about iteration rather than feasibility:

| Route | Work | Full-model time |
|---|---|---|
| CPU as-is | none, today's binary | ~50 h on 20 cores; it scales with cores, so a 64-core x86 host lands near 15 h [A over M] |
| Device kernel | the kernel in 10.2 | minutes to low tens of minutes [E] |

### 10.1 The measured baseline

Spark GB10, 20 cores, 2026-10-24, one real matrix, the recipe's own parameters:

```
blocks.0.adaln_proj.linear.weight   96768 x 2688
260,112,384 weights   32,514,048 vectors   pool 8,192,000 (1000.0/codeword)
PROFILE pool=0.2 lloyd=431.8 book=0.003 gain=98.6 final=103.2 residual=45.9 \
        write=0.3 metrics=3.2  total=679.7 s
BYTES   gvq=53,028,992 (1.6310 bpw)  gvr=32,514,176 (1.0000 bpw)
        book=65,536  rbook=2,048  total=85,610,752 (2.6330 bpw)
QUALITY stage1 cos=0.935474 relL2=35.34%   two-stage cos=0.979842 relL2=19.99%
```

`gain` covers the provisional full pass plus the least-squares refit; `final`
is the second full pass against the stored gain. [M]

The rate: 32 rounds x 8,192,000 vectors x 8192 codewords = 2.147e12 codeword
comparisons in 431.8 s, i.e. **4.97 G comparisons/s** and 13.5 s per round.
11.3 minutes for a 260M-weight matrix end to end. [M]

Over the plan's own 260 matrices (4.13754e9 vectors; 5.777e14 + 1.699e13 =
5.95e14 comparisons, from `h3-plan.tsv` by the same arithmetic the tool
performs) that is 33.2 h of Lloyd, and at this matrix's ratio of other stages
to Lloyd (251 s against 432 s, 0.58x) about **50 h** for the full rung A+8
encode on 20 cores. [A over M]

For scale: the M3.1 release's routed experts measured rel MSE 0.150-0.175 at
13 bits and 0.051-0.068 with R13+8 (section 8.4); this matrix lands at rel MSE
0.125 (stage 1) and 0.040 (two-stage), better on both, with no imatrix
weighting yet. `relL2 = sqrt(rel MSE)`, which is why the printed relL2 looks
larger than the precedent's MSE figures.

### 10.2 What a GPU kernel has to do

One comparison is an 8-wide weighted squared distance: about 9 FLOPs with the
column weights folded into the operands. The assignment is therefore a
tall-skinny weighted GEMM `[vectors x 8] x [8 x K]` followed by an argmin over
K. Three properties of this rung decide the design:

- **The book fits on chip.** 64 KiB in E4M3 per matrix, against 228 KiB of
  shared memory per H100 SM (99 KiB opt-in on the Spark). Stage it once per
  block and every comparison reads shared memory. This is the reason section 2
  chose E4M3 at K 8192 and why a 14-bit book does not fit.
- **Vector traffic is small.** 4.14e9 vectors x 32 B = 132 GB for the whole
  model, 40 s at H100 bandwidth. The kernel is not HBM bound.
- **The argmin does not tensor-core.** 5.95e14 compares plus index selects is
  ~70 s of H100 lane throughput on its own. Expect it, not the GEMM, to set the
  floor. [E]

The GEMM half is 9.5e15 FLOPs: 142 s at the H100's 67 TFLOP/s FP32 peak, ~19 s
at its 495 TFLOP/s TF32 peak, and 2-5 minutes at a realistic 30% of either with
the epilogue overlapped. That puts an H100 with a purpose-written kernel in the
single-digit to mid-teens minutes, a 200-600x gain over the measured CPU path;
the same kernel on the Spark's own GB10 should land at 10-40 minutes, which is
the cheaper experiment and needs no new host. [E]

### 10.3 What the payload must satisfy, whatever produced it

A GPU encoder does not have to reproduce these bytes. E4M3 codewords and F16
gains are exact, but the Lloyd trajectory and tie-breaks belong to the
implementation. It must produce a payload that the strict reader accepts and
the value rule reproduces:

1. `check <prefix> --src ... --name ...` passes: magic, version, index bits,
   dim, K, precision, flags, all four canonical offsets, both checksums, the
   gain rules, zero padding in every gap, and the codebook's own SHA-256.
2. Its own output is byte-identical across repeated runs at one configuration.
   The CPU asserts this across thread counts, which is the strongest form
   available to it.
3. With the residual armed, the stage-1 `.gvq` is byte-identical to the same
   encode without it: the second stage must not disturb stage 1.
4. Quality is reported the same way, from `check`, as stage-1 and two-stage
   cosine and relL2 against the source.

Report the PROFILE and QUALITY lines back. Sizes plus a rate in codeword
comparisons per second are what make two implementations comparable; a
vectors-per-second figure mixes the 8192-codeword and 256-codeword passes and
means nothing.

### 10.4 The whole model, in one loop

The encoder takes one matrix per process; the loop is the missing driver, not
new format work.

```
ST=/path/to/MiniMax-H3/FL2VA/transformer
tools/h3_vq/bin/h3_pack_cpu plan "$ST" > h3-plan.tsv     # 260 vq rows
mkdir -p out
awk -F'\t' 'NR>1 && $3=="vq" {print $1}' h3-plan.tsv |
while read -r n; do
  tools/h3_vq/bin/h3_gvq encode "$ST" "$n" "out/$(printf '%s' "$n" | tr / _)" \
      --iters 32 --residual || break
done
```

Each matrix writes `<prefix>.gvq`, `.book`, `.gvr`, `.rbook` and a `.report`.
260 matrices at ~85.6 MB is 22.3 GB including the books and the residual,
against 6.76 GB for the payloads alone: the books are per matrix by design
(section 2), which is the price of a 13-bit index at this dim.
