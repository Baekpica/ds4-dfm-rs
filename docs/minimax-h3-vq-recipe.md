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
| Vector dimension | 8 | codes 8 consecutive weights of a row jointly, needs no per-32 scale, and one codeword is exactly an `m16n8k32` B fragment |
| Codebook size K | 8192 | 13 bits: 12-bit stream + 1-bit plane, the widest index that still fits the 99 KiB shared-memory budget with the staging a dense GEMM needs |
| Codebook precision | E4M3FN | halves the lookup bytes against F16 for +0.12% error in the reference's measurement |
| Index bits | 13 | the quality end of the ladder; H3 does not need 9 or 10 bits because it fits in memory at BF16 already |
| Codebook scope | one per matrix (260 books) | H3 is dense: every matrix is large, so nothing is gained by sharing, and a per-matrix book removes cross-tensor interference |
| Row gains | one F16 per row, 8,115,968 rows | the format's only scale; 16.2 MB total |
| Lloyd | 32 iterations, `init: equidistant` | deterministic by construction: no random source, same input yields the same bytes |
| Weighting | `imatrix` from H3's own activations | the assignment and gains are fitted against the statistics of the tokens this DiT will actually see |
| `col_power` / `col_floor` | 0.75 / 0.02 | damps the importance profile without letting a single column dominate |
| Gain refit | `ls` (least squares) against the stored indices | the gains are refit *after* the assignment, which is where most of the error is recovered |
| Pool | stratified, ≥ 1000 vectors per codeword | a 4× thinner pool cost 0.96% in the reference's measurement |
| Residual stage | GVR8, K = 256, 8 bits, on chosen blocks only | +1 bit per weight on ~40% of the parameters instead of +1 bit everywhere |

**Byte accounts** (arithmetic from section 1.5; 33.100B quantizable params):

| Rung | Bits/weight | Codes | Gains | Books | Total | GiB |
|---|---:|---:|---:|---:|---:|---:|
| 12-bit (1.5) | 1.50 | 6.206 GB | 16.2 MB | 17.2 MB | 6.239 GB | 5.81 |
| **13-bit (1.625), chosen** | **1.625** | **6.723 GB** | **16.2 MB** | **17.2 MB** | **6.756 GB** | **6.29** |
| 14-bit (1.75) | 1.75 | 7.241 GB | 16.2 MB | 17.2 MB | 7.274 GB | 6.77 |
| 13-bit + GVR8 on 20 blocks | — | +0.5–0.8 GB | — | +0.5 MB | ~7.3–7.6 GB | 6.8–7.1 |
| 13-bit + GVR8 on all 52 blocks | — | +4.138 GB | — | +1.3 MB | 10.894 GB | 10.15 |
| BF16 (as shipped) | 16.0 | 66.20 GB | — | — | 66.28 GB | 61.73 |

Against BF16 the chosen rung is **9.81× fewer bytes**; the DiT's load goes from
the measured 96.563 s to roughly 10 s at the same I/O rate. Measured evidence
for the 13-bit configuration itself, and for what its error looks like at the
output, is in the [plan](minimax-h3-port-plan.md) section 5 and the
[cheat sheet](minimax-h3-cheatsheet.md) section 6.

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

At 13 bits the adaln class alone is 2.64 GB of the 6.76 GB artifact, and the
attention class is 1.56 GB. Removing `adaln_proj` from quantization would take
the artifact to 4.12 GB but keep 61% of the DiT's parameters exact — a
legitimate rung, and the first thing to try if the adaln class is what fails
the render gate.

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

1. **Inventory.** Read all 13 shard headers, emit the manifest of 535 tensors
   with dtype, shape, data offsets and shard, and reconcile its total against
   `metadata.total_size` = 66,280,430,144 B. Gate: exact match, zero unknown
   tensors, and the class table of section 1.5 reproduced.
2. **Plan.** Emit the per-matrix table: name, shape, rows, columns, indices per
   row, codebook bytes, main-stream bytes, plane bytes, gain bytes, total
   bytes; print the per-class subtotals and the artifact total. Gate: the
   arithmetic reproduces 6.756 GB at 13 bits before any encoding happens.
3. **Statistics (imatrix).** Instrument the reference to dump the input
   activation of each VQ matrix over the frozen prompt set (see the plan
   section 5.1 for the fixture method and its ≥ 99.996% bit-match check),
   then reduce to per-matrix column statistics. Gate: every one of the 260
   matrices has statistics, and the fixture reproduces the reference's own
   block output bit for bit on the check rows.
4. **Encode.** Per shard, per matrix: row-RMS normalize, 32 Lloyd iterations
   with importance-weighted assignment, least-squares gain refit, final
   assignment against codebook and gains rounded to their stored precision
   (F16 gains, E4M3 codewords). Gate: determinism — re-encode a shard and
   compare bytes.
5. **Assemble.** Write the parts, the codebook tensors, the gains, the
   streams, the manifest and the `release.json`. Gate: every payload hash,
   every byte account, and every source tensor accounted for exactly once.
6. **Verify.** Read each part back with a strict reader implementing the
   header rules, decode every payload, and compare against the source tensor.
   Gate: max abs, relative RMS and cosine per matrix and per class; the reader
   must reject a mutated payload, a bad hash, a non-canonical offset and a
   truncated part (test each).
7. **Render.** The gate in the plan section 5: eight prompts, two shapes,
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
