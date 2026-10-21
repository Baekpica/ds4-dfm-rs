# VQ calibration: an importance matrix for shared-codebook GVQ

`tools/vq_calib/` turns a calibration corpus into per-expert activation
statistics and uses them in the Lloyd codebook training, the vector
assignment and the row gains of a GVQ (YoungAi-style 8-dimensional, one
E4M3 codebook per MoE layer) conversion. It is the VQ counterpart of
llama.cpp's importance matrix, which llama.cpp makes mandatory for its 1–2
bit types (`src/llama-quant.cpp` `tensor_requires_imatrix`). Every earlier
GVQ release here (DeepSeek V4.1, GLM-5.3, Ling-3.1, Beam-1.0) trained its
codebooks on the weights alone; YoungAi's body VQ is corpus-free by design
("全程零语料, 码本 Lloyd 解在权重自身") and only its GR/RB sidecars see data.

The first user is the A.X-K2 VQ release
([Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)),
whose routed experts must fit 9 bits per 8 weights (10 on seven MoE
layers, 16, 20, …, 40) to serve 262,144 tokens on one DGX Spark: ~1.14 BPW,
below IQ1_S.

## Pipeline

| Step | Tool | Output |
|---|---|---|
| 1. Calibration set | `build_calib.py` (seeded, sources pinned by revision) | `calib.safetensors` (`input_ids`, `offsets`), manifest; disjoint `heldout.safetensors` |
| 2. Reference forward check | `selftest_ref.py` | the family's layer forward vs transformers' decoder layer on the same dequantized weights |
| 3. Statistics | `run_collect.sh` → `collect_stats.py` (one data-parallel rank per GPU) + `reduce_stats.py` | `stats/layer-LL.safetensors` (FP64 sums), `stats/head.safetensors`, held-out dumps |
| 4. Choose the weighting | `measure_layer.py` | held-out routed-output error per variant and layer |
| 5. Experts | `run_recipe.py` → `convert_layer.py` (`--mode`, `--gain-refit`, `--books`) | `layer-LL-experts.gguf` (GVQ v2, 9/10-bit) + receipts |
| 6. Non-experts | `convert_nonexpert.py --imatrix-dir stats` | `body-nonexpert.gguf` (Q4_K with per-column importance, Q8_0, source tensors), `embed-tokens-bf16.gguf` |
| 7. Release | `assemble_release.py`, `axk2_vq_format.py --release`, `publish_release.py` | `release.json`, `tensor-manifest.json`, hashes |
| 8. End-to-end check | `eval_e2e.py` | teacher-forced KL, next-token agreement, NLL per held-out case; goldens for the runtime |

The collector streams the FP8 (or BF16) source one layer at a time: each GPU
holds its shard of the calibration tokens' residual stream, every rank reads
layer L+1 into pinned memory while it runs layer L, and the expert weights
stay FP8 on the device and are dequantized per expert. On 7 RTX PRO 6000
(96 GB) 3.92M calibration tokens through A.X-K2's 61 layers took 7.5–9
minutes per rank.

## Statistics

Per MoE layer and routed expert e (sums over the tokens routed to e):
`count`, Σw, Σw² of the final route weight; Σx² and Σw²x² of the gate/up
input (the FFN-normalized hidden state); Σa² and Σw²a² of the down input
a = silu(g)·u; Σw²(silu'(g)·u)² and Σw²silu(g)² per intermediate channel (how
an error in a gate or up row reaches the layer output). Per layer, Σx² of the
input of every non-expert projection. llama.cpp's MoE imatrix is the
`count`-normalized Σx² / Σa² subset (no routing weight; each expert normalized
by its own count) — enough when every row is quantized independently, but a
codebook shared by 256 experts and three projections also has to know how
often and how strongly each expert is routed.

## Weighting

For vector i (expert e, projection p, row r, columns 8j..8j+7) with stored
row gain g_r and normalized v = w / g_r, the layer-output error is
approximated diagonally as Σ_d ω_{i,d} (v_d − c_d)². The assignment keeps
only the column factor (the row factor cancels in the argmin); the centroid
update is the per-dimension ω-weighted mean.

| Mode | ω | Data |
|---|---|---|
| `none` | 1 (earlier releases) | — |
| `row` | g_r² (row-energy centroids; YoungAi's unmeasured candidate) | — |
| `imatrix` | g_r² · h̄_{e,p}[k], h̄ = per-tensor mean-normalized Σx² (gate/up) or Σa² (down) | counts |
| `imatrix_freq` | g_r² · Σw²x² or Σw²a² (expert frequency and route weight kept; projection types balanced) | routes |
| `output` | down: g_r²·Σw²a²; gate/up: g_r²·‖D[:,r]‖²·(Σw²s_r²/Σw²)·Σw²x² | routes + sensitivities |

`--gain-refit ls` replaces the RMS row gain by the importance-weighted least
squares gain Σ_k h_k w_k ĉ_k / Σ_k h_k ĉ_k² after assignment and re-assigns
once (still one F16 per row). `--books split3` trains one book each for gate,
up and down (IDs 4L, 4L+3, 4L+2) at the same K. Rare experts' column profiles
are shrunk toward the layer mean with a pseudo-count of 256 tokens, and every
column weight is floored at 2% of its tensor's mean.

## Container (GVQ v2)

Nine and ten index bits use a byte-major version 2 of the GVQ1 matrix
(`axk2_vq_format.py` is the reference encoder, decoder and strict reader).
128-byte header `<4sHBBIIIIIIQQQQ32s32s`: `GVQ1`, version 2, bits 9 or 10,
dimension 8, rows R, columns C, K = 2^bits, codebook precision 1 (E4M3FN),
codebook ID, flags 0, gain offset 128, main offset `align128(128 + 2R)`, plane
offset `align128(main + R·C/8)`, total `align128(plane + (R·C/8)·(bits − 8)/8)`,
SHA-256 of the codebook bytes, SHA-256 of every byte after the header.

- Gains: R × F16, finite and non-negative (a zero row has gain +0).
- Main stream: one byte per index, its low 8 bits, row-major vector order
  (index n = r·(C/8) + j).
- Plane: the high bits. 9-bit: bit 8 of index n is bit n % 8 of byte n / 8.
  10-bit: bits 8–9 of index n are the 2-bit field at bit 2·(n % 4) of byte
  n / 4. Least significant first.
- C is a multiple of 64, so rows are whole bytes and never padded: a 7,168-wide
  row is 896 index bytes plus 112 (9-bit) or 224 (10-bit) plane bytes, a
  2,048-wide row 256 plus 32 or 64.
- `W[r, 8j + d] = f32(gain[r]) · f32(book[index[r, j]][d])`.
- Books are `[K, 8]` E4M3FN tensors; a layer part holds one to three of them:
  ID 4L (gate, or all three projections when shared), 4L+3 (up) and 4L+2
  (down); 4L+1 is reserved for a residual stage. Each matrix names its book in
  the header.

## Measurements (A.X-K2)

The release, [Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF),
uses `--mode imatrix --gain-refit ls --books split3 --col-power 0.75` with
10-bit codebooks on MoE layers 16, 20, 24, 28, 32, 36 and 40.

Measured on Runpod (RTX PRO 6000 Blackwell, sm_120, CUDA 13.2, torch 2.13)
against the FP8 original run by `axk2_ref.py`, teacher-forced, KL(teacher ‖
converted) in nats per token. Other hardware, CUDA builds and kernels move these
numbers slightly; compare them statistically, not digit by digit.

**Per layer** (`measure_layer.py`, the held-out tokens' own FFN inputs; routed
output relative error ‖Δy‖²/‖y‖²; layer 1 at 10 bits, the others at 9):

| Layer | Unweighted | Row energy + LS gains | imatrix, p 0.75, LS gains, split3 | imatrix_freq, same | output, p 1.0, same |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.284 | 0.259 | 0.120 | 0.125 | 0.110 |
| 8 | 0.357 | 0.356 | 0.276 | 0.286 | 0.313 |
| 15 | 0.397 | 0.398 | 0.298 | 0.294 | 0.292 |
| 30 | 0.487 | 0.487 | 0.479 | 0.479 | 0.478 |
| 41 | 0.516 | 0.516 | 0.477 | 0.477 | 0.476 |
| 60 | 0.652 | 0.604 | 0.209 | 0.126 | 0.175 |
| mean change, 7 layers (global / token mean) | — | −2.3% / −1.3% | −26.9% / −23.9% | −28.2% / −21.8% | −27.0% / −21.3% |

The weighted variants raise the plain weight error (0.264 → 0.34) and lower
the importance-weighted one (0.199 → 0.162): they spend precision where the
activations are.

**End to end** (22 held-out cases, 53,142 tokens; experts converted, the
rest FP8; 10-bit on layers 1–7):

| Codebook training | KL | Top-1 | NLL (FP8 1.130) |
|---|---:|---:|---:|
| Unweighted (earlier releases) | 0.859 | 69.5% | 1.792 |
| Row energy + LS gains | 0.858 | 69.4% | 1.790 |
| imatrix, p 0.75, LS gains, split3 | **0.788** | **71.0%** | **1.720** |
| imatrix_freq, p 0.75, LS gains, split3 | 0.791 | 70.6% | 1.725 |
| output, LS gains, split3 | 0.799 | 70.7% | 1.733 |

The −24% per-layer gain becomes −8% KL end to end: the 9-bit middle layers
improve only 2–10% and the residual stream still drifts (FFN-input cosine
0.987 at layer 15, 0.68 at 41, 0.64 at 60). Where the 10-bit layers go matters
once the weighting has made the bottom layers accurate: at the same bytes,
10-bit on 1–7 gives KL 0.788, on 15–21 0.774, spread over 16, 20, …, 40
0.756. On a second, independent held-out set (78 cases, 188,578 tokens, used
for no choice) the release bytes give KL 0.753 (1–7) vs **0.727** (spread),
better in all 19 domains and 73 of 78 cases. The Q4_K/Q8_0 non-expert body
alone costs KL 0.046.

### Second use: Beam-1.0 (12/13-bit, drop-in)

[Baekpica/Beam-1.0-501B-A23B-VQ-GGUF](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF)
was re-converted with the same weighting (imatrix, column power 0.75,
importance-weighted LS gains), keeping everything the existing `beam_vq`
runtime pins: one shared book per layer, the published 13-bit layers 3–10,
12-bit elsewhere, GVQ1 v1 and the recipe id. The runtime loads the new parts
unchanged. Statistics took 4.5 minutes on 8 GPUs for 3.91M calibration tokens
through the 1 TB BF16 source; conversion took 14–60 s per layer. Against the
first, weights-only build (teacher-forced, RTX PRO 6000):

| Held-out | First build KL / top-1 | Calibrated KL / top-1 |
|---|---:|---:|
| The 7 published cases (5,491 tokens) | 0.291 / 83.3% (NLL 1.456) | **0.144 / 87.7%** (NLL 1.340) |
| 22 cases, 54,017 tokens | 0.492 / 80.4% | **0.278 / 85.8%** |

The calibrated build has lower KL in all 18 domains of the 22-case set, and
the hidden-state cosine with the BF16 original at layer 40 rises from 0.939
to 0.973. When the first build was published it measured 0.295 / 82.9% on
H100 on the 7 cases. Beam's adapter scripts (its reference forward, collector
and GVQ1 v1 converter) are shipped in that release's `tools/`, not in this
directory.

## Pitfalls found on the way

- **transformers' group-limited router is wrong for checkpoints whose biased
  scores go negative.** `AXK2TopkRouter` (inherited from DeepSeek-V3's HF
  port) fills the masked groups with 0.0; A.X-K2's layers 24, 41 and 48 have
  every `e_score_correction_bias` below −1.3, so the masked experts win the
  top-8 for every token (40 of 256 experts used), and layers 34–37, 42, 49,
  51, 52, 56, 57, 60 are partly affected. Megatron, vLLM `grouped_topk` and
  llama.cpp `build_moe_ffn` use −inf. On the held-out set the FP8 teacher's
  NLL is 1.130 (PPL 3.10) with −inf and 1.348 (PPL 3.85) with 0.0. The
  collector uses −inf.
- **Rare experts:** with 3.9M tokens the rarest expert of a layer still sees
  thousands of tokens, but per-expert column profiles of experts with tens of
  tokens are noise; shrink them.
- **Zero-weight dimensions:** an unfloored importance profile lets a VQ
  dimension take any codeword value; floor the column weights.

## Adding another family

The weighted Lloyd, the GVQ v2 container and the release tools are family
independent. A new family needs its own `<family>_ref.py` layer forward with
the `Stats` hooks (see `axk2_ref.py`), checked against transformers by a
`selftest`, and a non-expert policy in `convert_nonexpert.py`.
