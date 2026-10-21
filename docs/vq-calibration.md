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
whose routed experts must fit 9 bits per 8 weights (10 on the bottom seven
MoE layers) to serve 262,144 tokens on one DGX Spark: ~1.14 BPW, below
IQ1_S.

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

## Measurements (A.X-K2)

RESULTS_PLACEHOLDER

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
