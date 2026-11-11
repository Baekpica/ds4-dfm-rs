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

### Third use: DeepSeek-V4.1-Flash-1017 (12/13-bit, drop-in)

[Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF)
patches the 40 routed-expert DQVL v3 blobs of the weights-only
[VQ release](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF)
(YoungAi `vq8sh14-q4k-mtpnative`: one E4M3 book per layer shared by 384 experts and
gate/up/down, K=8192 on layers 0–13, K=4096 on 14–39) in place, with the A.X-K2
weighting (`imatrix`, col_power 0.75, shrink 256, floor 0.02, LS gains, 32 Lloyd
iterations) and nothing else changed: no search, by design. Every other byte of the
base (header, q4_K skeleton, MTP towers) is the VQ release's, so ds4-dfm-rs's
`ds41` path loads it unchanged (`ds41_inspect`: bind 1000/1000).

The family tools are `ds41_ref.py` (the official `inference/model.py` prefill path in
plain PyTorch: mHC Sinkhorn, Engram, compressed sparse attention with the two-level
indexer, kernels emulated), `run_ds41.py` (collector and teacher-forced passes),
`convert_layer_ds41.py` (FP4 e2m1 experts, DQVL v3 through `ds41_vq_format.py`, which
round-trips the published blobs byte for byte and reproduces YoungAi's codebooks bit
for bit), `assemble_ds41_ivq.py`, `score_ds41.py`, `score_regions_ds41.py` and
`measure_ds41.py`. Calibration: 3,653,893 tokens rendered with the model's own
`encoding.py` (`build_calib_ds41.py`); every expert of every layer was routed.

Measured on Runpod (8× A100 80GB, sm_80, CUDA 13.2, torch 2.13), 26 held-out cases,
58,803 tokens:

| MoE layers | Per token (median), VQ → iVQ | Change | Layers improved | Energy-weighted, VQ → iVQ |
|---|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → **0.215** | **−10%** | 14 of 14 | 0.220 → 0.164 (−26%) |
| 14–39 (12-bit) | 0.348 → **0.327** | **−6%** | 26 of 26 | 0.345 → 0.075 (−78%) |
| All 40 | 0.309 → **0.288** | **−7%** | 40 of 40 | 0.302 → 0.106 (−65%) |

| | Floor | VQ | iVQ |
|---|---:|---:|---:|
| KL, all positions | 0.026 | 0.369 | **0.350** (−4.9%) |
| KL, generated positions | 0.010 | 0.295 | 0.291 (−1.4%) |
| Same top-1, all positions | 96.2% | 82.41% | 82.47% |

Every layer improves for the typical token, but only by 7%, and end to end KL falls
4.9% with next-token agreement unchanged. Chat, reasoning, tool calls and code improve by 3–21% KL; Chinese
and Japanese web text gets worse end to end although iVQ's per-layer routed error is
lower on those cases too.

The reason is visible in the per-token data: DeepSeek-V4.1's routed experts have extreme
massive activations. From layer 10 on, 1% of the held-out tokens carry 94–100% of the
routed-output energy (the largest up to 1.1e7 times the median token); on layers 20–25
the 26 sequence-initial BOS tokens alone carry ~95%. `Σx²`/`Σa²` statistics therefore
describe those tokens, and the weighted fit protects them: the energy-weighted per-layer
error falls 65%, the typical token's 7%. A.X-K2 has no BOS and showed no such gap
(−26.9% energy-weighted vs −23.9% token mean).

That explains the gap between the two measures, not the small per-token gain. Two
follow-up conversions were measured the same way:

- `--stats-norm token`: the collector also sums `x2n`/`a2n`, each token's `x²`/`a²`
  divided by its own mean square, so every routed token weighs the same.
- The same with calibration mix v2 (`build_calib_ds41.py --mix v2`): 3.75M tokens, no
  domain above 8%, English 34%, Chinese 16%, Korean 13%, Japanese 8%, seven more
  languages.

| Per token (median), mean of 40 layers | VQ | imatrix (release) | + per-token stats | + mix v2 |
|---|---:|---:|---:|---:|
| Routed-expert error | 0.309 | 0.288 (−7.0%) | 0.288 (−7.0%) | 0.288 (−6.8%) |
| KL, all positions | 0.369 | 0.350 | 0.349 | 0.352 |

Neither lever moves the result. The Chinese and Japanese web regression stays in all
three variants, even though their per-layer errors on those cases fall alike. Expert
columns are not flat here: column-importance CV is 0.23–0.39 for gate/up and
0.73–1.50 for down, against A.X-K2's 0.13 and 0.30.

The remaining suspect is the diagonal approximation. The importance-weighted weight
error falls by 22% and the per-token output error by 7%, while on A.X-K2 the two
moved together. That points to correlated input channels, which only a
covariance-aware fit (GPTQ-style error feedback) or a fit on the true layer-output
loss would account for.

Two properties of this model matter for any comparison:

- **Not batch-invariant.** BF16 GEMM results depend on the row count (a 64-row and a
  3,072-row GEMM differ by up to 0.2%), and near-tied routes amplify that. Measure the
  reference's own floor (the same model, batched differently) before reading a
  difference; here it is KL 0.010 on generated positions.
- **Prompt positions are untrained.** The original assigns parts of its system prompt
  and tool schemas probabilities near 1e-8 (e.g. `$PARAM` → `ETER`), so a quantized
  model can have lower NLL there. `score_regions_ds41.py` reports generated positions
  (raw text and assistant turns) separately.

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
- **Massive activations:** when a few tokens (BOS, delimiters) carry most of a
  layer's activation energy, as in DeepSeek-V4.1, energy-weighted per-layer errors
  mostly measure those tokens. Report the per-token error too (`measure_ds41.py`
  records both). Weighting every token equally in the statistics
  (`--stats-norm token`) did not change the per-token gain there.

## Adding another family

The weighted Lloyd, the GVQ v2 container and the release tools are family
independent. A new family needs its own `<family>_ref.py` layer forward with
the `Stats` hooks (see `axk2_ref.py`), checked against transformers by a
`selftest`, and a non-expert policy in `convert_nonexpert.py`.

### Fourth use: hVQ, error feedback through the input correlation (DeepSeek-V4.1-Flash-1017)

[Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF)
tests the suspect the iVQ measurements left: the diagonal approximation. iVQ weighs column
k of an expert matrix by D_k (the importance above) and ignores how input channels move
together. hVQ uses the whole second moment of an expert's input, GPTQ / GPTVQ style: the
8-column blocks the format fixes are quantized in order, and each block's error is fed
forward into the targets of the columns not yet quantized.

Only the indices and the row gains change. The codebooks are the iVQ release's bytes; K,
offsets, sizes and the recipe id are unchanged, and every byte outside the 40
routed-expert blobs is the VQ base's, so ds4-dfm-rs's `ds41` path loads the file as it
loads VQ and iVQ (`ds41_inspect`: bind 1000/1000).

**Definition.** For expert e and a projection with input second moment S (5120² for
gate and up, which share an input and one factorization; 2304² for down):

    R   = correlation of S                                   (unit diagonal)
    H_t = D^1/2 ((1 - t) I + t R) D^1/2  + damp · mean(D) · I
    H_t = V V^T,  V upper triangular                          (Cholesky of the reversed matrix)

D is the iVQ release's column weights, unchanged (`imatrix`, col_power 0.75, shrink 256,
floor 0.02, from the release's statistics). A row's output error is
`(w − q) H_t (w − q)^T = ‖(w − q) V‖² = Σ_B ‖e_B‖²` with
`e_B = (w_B − q_B) V_BB + Σ_{A<B} (w_A − q_A) V_AB`: block B's term depends on blocks
A ≤ B only, so the blocks are chosen in ascending order, each to minimize its own term:

    target   w~_B = w_B + r_B V_BB^-1,     r_B = Σ_{A<B} (w_A − q_A) V_AB
    codeword c = argmin (w~_B / g − c) M_B (w~_B / g − c)^T,     M_B = V_BB V_BB^T

This is GPTQ's update written with the factor of H instead of the factor of H⁻¹: one
Cholesky per expert input, no inverse (`tests/test_ds41_hvq.py` drives the textbook
inverse-factor form in FP64 with the kernel's choices: every choice is its optimum). The
row gain g is fixed during a pass and refit afterwards by least squares under H_t,
`g = w H_t ĉ^T / ĉ H_t ĉ^T`; as in iVQ there are two passes (RMS gains, refit, again).

**t = 0 is iVQ.** H_0 = D has no feedback, M_B is the column weights, and the kernels add
the off-diagonal terms as exact zeros to iVQ's own fused sums. `convert_layer_ds41_hvq.py
--t 0` matches the published layer-14 iVQ blob byte for byte, and matches
`convert_layer_ds41.py` run on the same host. Synthetic tests cover both bit widths.

**Statistics.** Per expert, over the tokens routed to it, `Σ x x^T / mean(x²)` and
`Σ a a^T / mean(a²)`: every token divided by its own mean square, as `x2n` / `a2n`,
because raw sums describe the BOS and other massive tokens (above). Same calibration set
(3,653,893 tokens) and teacher stream as iVQ, dealt into the same 8 shards; the diagonal
statistics collected alongside agree with the iVQ release's to count_l1: maximum relative difference 0.0035; x2_profile_rel: maximum relative difference 0.147; a2_profile_rel: maximum relative difference 0.148. A layer's moments
are 45.1 GiB (48.4 GB) in FP32 (40 layers: 1.9 TB), so `run_hvq_ds41.py` keeps them on the device
only: collect a layer, convert it, drop them. Each chunk's Gram matrix is computed in FP64
and added to an FP32 sum: in the archived pilot, against FP64 sums the correlations of twelve probed experts
(29 to 497,852 routed tokens) differ by at most 2.1e-7 (all-FP32: 2.0e-6). The printed
summary is retained; the raw precision-probe files were lost with the previous pod.

The inputs are far from uncorrelated. The participation ratio `C² / ‖R‖_F²` of an expert's
input correlation, C for uncorrelated channels, is 19.3–135.3 for gate/up (mean layer median 77.1, 5120 channels); 95.6–1009.1 for down (mean layer median 531.6, 2304 channels).

**Rare experts** (the rarest of a layer saw 4 tokens) are shrunk before the
correlation is taken, with a pseudo-count of tokens: gate/up toward the layer mean per
token (experts share the hidden space), down toward the identity (an expert's channels are
its own). No input matrix needed a further diagonal lift.

**Choice of t, damping and shrink.** The pilot ran in the previous pod; its raw
files were lost, and the table survives in the archived session output. The full
build and final measurements below were run on the replacement pod.

Chosen once, on layers 3, 11, 19 and 27, by the median
over the 58,803 held-out tokens of the routed-output error (the original's inputs and
routes, `sweep_hvq_ds41.py`), and applied to all 40 layers:

Pilot executed in the previous pod; raw sweep JSONL and probe files were lost. The table and precision summaries below are recovered verbatim from the archived session output, not rerun on this pod.

| t | damping | shrink (tokens) | Layer 3 (13-bit) | Layer 11 (13-bit) | Layer 19 (12-bit) | Layer 27 (12-bit) | Mean ratio to iVQ | Target energy, last quarter of the blocks (gate/up) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VQ (weights only) | | | 0.2231 | 0.2612 | 0.2897 | 0.3683 | 1.091 | |
| iVQ (t = 0) | | | 0.1936 | 0.2466 | 0.2589 | 0.3561 | 1.000 | 1.00 |
| 0.25 | 0 | 256 | 0.1094 | 0.1899 | 0.1688 | 0.3032 | 0.7097 | 1.02 |
| 0.5 | 0 | 256 | 0.0922 | 0.1746 | 0.1496 | 0.2908 | 0.6448 | 1.05 |
| 0.75 | 0 | 256 | 0.0831 | 0.1666 | 0.1389 | 0.2853 | 0.6107 | 1.09 |
| 0.9 | 0 | 256 | 0.0799 | 0.1649 | 0.1352 | 0.2857 | 0.6014 | 1.13 |
| 1 | 0 | 256 | 0.0794 | 0.1653 | 0.1345 | 0.2883 | 0.6024 | 1.19 |
| 0.75 | 0.01 | 256 | 0.0834 | 0.1670 | 0.1391 | 0.2854 | 0.6116 | 1.09 |
| 1 | 0.01 | 256 | 0.0794 | 0.1652 | 0.1345 | 0.2878 | 0.6020 | 1.18 |
| 0.75 | 0.1 | 256 | 0.0860 | 0.1687 | 0.1419 | 0.2858 | 0.6198 | 1.08 |
| 0.75 | 0 | 4096 | 0.0837 | 0.1670 | 0.1397 | 0.2854 | 0.6127 | 1.09 |
| **1** | **0** | **4096** | **0.0793** | **0.1647** | **0.1345** | **0.2863** | **0.6003** | 1.17 |


The error falls with t up to the full correlation on layers 3–19 and changes little from
0.75 on layer 27. Damping did not produce the lowest ratio in this grid: at t = 1 and
shrink 256, damping 0.01 changes the mean ratio from 0.6024 to 0.6020; at t = 0.75,
both tested damping values increase it. The pseudo-count also has a small effect.
Feedback does push the later columns' targets away
from the weights the codebook was fitted to (the archived pilot reported a mean squared norm in the last quarter of
the block order of 1.17 times the weights' at t = 1), but that costs
less than the feedback gains. The setting with the lowest mean ratio to iVQ, t = 1, no
damping, pseudo-count 4096, was used; t = 0.9 and t = 1 with pseudo-count 256 are within
0.4% of it.

**Measurements.** Runpod, 1× H200 (sm_90, CUDA 13.2, torch 2.13), the reference forward,
the 26 held-out cases (58,803 tokens). VQ and iVQ were measured again in the same runs,
so their numbers differ slightly from the A100 ones above.

| MoE layers | Per token (median): VQ → iVQ → hVQ | hVQ vs iVQ | Layers hVQ < iVQ | Energy-weighted: VQ → iVQ → hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → 0.215 → 0.115 | -46.4% | 14 of 14 | 0.220 → 0.164 → 0.074 | -55.0% |
| 14–39 (12-bit) | 0.347 → 0.327 → 0.228 | -30.2% | 26 of 26 | 0.345 → 0.075 → 0.014 | -81.7% |
| All 40 | 0.309 → 0.288 → 0.189 | -34.5% | 40 of 40 | 0.301 → 0.106 → 0.035 | -67.3% |

| vs the original | Floor | VQ | iVQ | hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| Routed-expert error per layer, per token (median, mean of 40 layers) | — | 0.309 | 0.288 | 0.189 | -34.5% |
| Routed-expert set overlap with the original (mean of layers 1–39) | — | 75.2% | 76.5% | 82.2% | +5.7 pt |
| Final hidden state cosine | — | 0.778 | 0.798 | 0.869 | +0.071 |
| KL, all positions | 0.037 | 0.370 | 0.352 | 0.273 | -22.3% |
| KL, generated positions | 0.012 | 0.293 | 0.290 | 0.241 | -17.0% |
| KL, prompt positions | 0.111 | 0.601 | 0.538 | 0.371 | -31.1% |
| Same top-1, all positions | 95.6% | 82.42% | 82.41% | 84.67% | +2.26 pt |
| Same top-1, generated positions | 96.3% | 82.39% | 82.22% | 84.39% | +2.17 pt |
| NLL, all positions (original 1.260) | 1.259 | 1.236 | 1.252 | 1.387 | +0.14 |
| NLL, generated positions (original 1.235) | 1.235 | 1.334 | 1.347 | 1.359 | +0.01 |

| Held-out text | Tokens | KL: VQ → iVQ → hVQ | hVQ vs iVQ | Same top-1: VQ → iVQ → hVQ | NLL: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4,096 | 0.497 → 0.482 → 0.419 | -13% | 78.6% → 78.5% → 80.4% | 1.12 → 1.43 → 1.41 → 1.42 |
| Web en / ko / zh / ja / es | 10,346 | 0.377 → 0.403 → 0.350 | -13% | 75.3% → 74.4% → 76.8% | 1.90 → 2.05 → 2.11 → 2.11 |
| Math web | 1,476 | 0.217 → 0.204 → 0.138 | -32% | 92.5% → 93.8% → 95.4% | 0.09 → 0.29 → 0.28 → 0.22 |
| Code files (Python, C) | 4,094 | 0.109 → 0.103 → 0.087 | -15% | 96.7% → 96.4% → 97.0% | 0.10 → 0.17 → 0.16 → 0.15 |
| Chat ko / zh / ja | 6,047 | 0.264 → 0.237 → 0.134 | -44% | 84.4% → 85.1% → 89.2% | 1.11 → 1.10 → 1.09 → 1.15 |
| Reasoning (think) | 6,142 | 0.157 → 0.139 → 0.084 | -40% | 88.2% → 88.4% → 91.6% | 0.87 → 0.81 → 0.83 → 0.88 |
| Instruction following | 1,517 | 0.323 → 0.333 → 0.326 | -2% | 79.8% → 80.5% → 81.9% | 0.98 → 0.99 → 1.01 → 1.08 |
| Tool calls / agents (DSML) | 2,659 | 1.330 → 1.015 → 0.563 | -45% | 69.8% → 71.3% → 76.5% | 2.37 → 1.26 → 1.42 → 2.61 |
| SWE agentless | 1,923 | 1.059 → 1.024 → 0.800 | -22% | 71.5% → 72.2% → 75.0% | 2.25 → 1.37 → 1.36 → 2.47 |
| Competitive programming (think) | 4,095 | 0.397 → 0.391 → 0.195 | -50% | 79.7% → 80.5% → 85.7% | 1.21 → 1.03 → 1.01 → 1.22 |
| Long 8K (web, finance) | 16,382 | 0.292 → 0.290 → 0.275 | -5% | 84.7% → 84.3% → 84.8% | 1.23 → 1.29 → 1.29 → 1.35 |

The direct routed-output error falls in all 40 layers under both summaries. The mean
per-layer median falls from 0.2876 to 0.1885 (34.5%); the mean energy-weighted ratio
falls from 0.1061 to 0.0347 (67.3%). The median reduction is largest in layers 0–2
(79.1%, 61.6% and 60.6%) and smallest in layers 26, 30 and 27 (18.7%, 18.9% and
19.6%). The large energy-weighted gain gives more influence to the massive routed
outputs; the median describes a typical token within each layer. The error tails
also fall, although an average 2.2% of tokens per layer have higher error than iVQ.
Plain weight error rises 21.5% and the diagonal-weighted error rises 32.0%, while
the directly measured routed outputs become closer to the original.

The final prediction distribution also moves closer to the original: generated-position
KL falls 17.0%, top-1 agreement rises 2.17 percentage points, final hidden-state cosine
rises from 0.7985 to 0.8692, and mean route-set overlap rises from 76.55% to 82.23%.
All-position KL is lower in 18 of 19 domains; long finance rises from 0.1606 to
0.1659. Generated-position KL is lower in 17 of the 18 domains with generated
targets; instruction-following chat rises slightly, from 0.3263 to 0.3280.
The long-finance case has no generated targets. The original's batching floor is
0.0368 KL over all positions and 0.0120 over generated positions; this is a
measurement of numerical variation, not a statistical confidence interval.

NLL on the actual held-out text worsens: 1.3466 → 1.3589 at generated positions
(+0.9%), and 1.2517 → 1.3869 over all positions (+10.8%). The largest all-position
NLL increases are in tool/agent and SWE cases, while raw math and code improve.
KL measures agreement with the original distribution; NLL scores the particular
target tokens, so the two need not move together. These results support improved
fidelity to the original, with a measured NLL tradeoff. They are teacher-forced
measurements of routed-expert substitutions: the non-expert body is the original
checkpoint, so the GGUF's q4_K body and MTP are not evaluated by these scores.
The same held-out set also selected the pilot setting. No quality threshold was
used to accept or reject the build.

**Cost.** One GPU and a 250 GB host: the residual streams of the 8 shards (139.4 GiB) rotate
through one pinned arena, the Engram rows of every shard are gathered once, and each layer
takes K=8192: mean collect 86.5 s, fit 104.3 s, pack/verify/write 28.6 s; K=4096: mean collect 90.1 s, fit 72.5 s, pack/verify/write 23.2 s (packing overlaps the following layer). The 40 layers took 2.00 h (7184.4 s); peak device memory 95.98 GiB collecting and 98.26 GiB converting; peak host RSS 177.70 GiB.
