# The Gemma4 text tower: port map from the release's own sources (2026-10-29)

Unit 2-3's source read, first-hand and cited. Two references, both read this
session and both needed: the model's own config, which the checkpoint carries as
metadata, and the forward that consumes it, which is HuggingFace's
`gemma4`/`gemma4_unified` implementation at the pin
(`requirements/stage2-observed.txt:7` = `transformers==5.14.1`, matching the
plan's `< 5.15`). The wheel is unpacked read-only at
`/data/refs/transformers-5.14.1` (fetched from PyPI, never installed).

Nothing here is inferred from a summary: every line below names the source it
was read from, and the checkpoint's own tensor list is quoted where it confirms
or corrects the reading. Two corrections to
[`gemma-connector-port-plan-2026-10-29.md`](gemma-connector-port-plan-2026-10-29.md)
come out of it (four norms per layer, not three; the tokenizer's "weights" are
the attention mask).

## 1. The config, as the checkpoint states it

`__metadata__["gemma_config"]` in
the released Gemma-4-12B INT8 convrot shard
(`gemma_assets.py:109-118` reads the same key), under `text_config`:

| Field | Value |
|---|---|
| `hidden_size` / `intermediate_size` | 3840 / 15360 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 16 |
| `head_dim` / `global_head_dim` | 256 / 512 |
| `num_key_value_heads` / `num_global_key_value_heads` | 8 / 1 |
| `attention_k_eq_v` | true |
| `layer_types` | 48 entries, `full_attention` at 5, 11, 17, 23, 29, 35, 41, 47 (40 sliding) |
| `sliding_window` | 1024 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `rms_norm_eps` | 1e-6 |
| `rope_parameters.sliding_attention` | `rope_type` default, `rope_theta` 1e4 |
| `rope_parameters.full_attention` | `rope_type` proportional, `partial_rotary_factor` 0.25, `rope_theta` 1e6 |
| `max_position_embeddings` / `vocab_size` | 262144 / 262144 |
| `tie_word_embeddings` | true |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | 0 / 2 / 1 |
| `enable_moe_block`, `hidden_size_per_layer_input`, `num_kv_shared_layers`, `use_double_wide_mlp` | false, 0, 0, false - the MoE, PLE and kv-sharing branches of the forward are dead here |
| `use_bidirectional_attention` | `"vision"` - so the text path is causal |

## 2. The tensor list, and what it confirms

From the checkpoint header (1342 tensors; the INT8 file's own `model.*` names are
already the flattened HF layout, `gemma_assets.py:314-331`):

- `model.embed_tokens.weight` BF16 `[262144, 3840]`; `model.norm.weight` BF16 `[3840]`.
- Per layer, the four norms and the scalar:
  `input_layernorm.weight`, `post_attention_layernorm.weight`,
  `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight` (all
  BF16 `[3840]`) and `layer_scalar` BF16 `[1]`. **Four** norms, not three: the
  forward wears `input -> attn -> post_attention -> residual -> pre_feedforward
  -> mlp -> post_feedforward -> residual`
  (`modeling_gemma4.py:1415-1450`).
- Sliding layer (0): `q_proj` I8 `[4096, 3840]` (16 x 256), `k_proj` `[2048, 3840]`
  (8 x 256), `v_proj` `[2048, 3840]`, `o_proj` `[3840, 4096]`, `q_norm`/`k_norm`
  BF16 `[256]`, no `v_norm` (it is a weightless RMSNorm,
  `modeling_gemma4.py:1220`).
- Full layer (5): `q_proj` `[8192, 3840]` (16 x 512), `k_proj` `[512, 3840]`
  (1 x 512), **no `v_proj`**, `o_proj` `[3840, 8192]`, `q_norm`/`k_norm` `[512]`.
  That is `attention_k_eq_v` and `num_global_key_value_heads=1`
  (`modeling_gemma4.py:1194-1231`), and it is why the eight full layers are
  exactly the eight layers with no `v_proj`.
- Every layer: `mlp.gate_proj` / `up_proj` I8 `[15360, 3840]`, `down_proj`
  I8 `[3840, 15360]`, each with a F32 `[out, 1]` `weight_scale` and a
  `comfy_quant` blob naming group 256 - the unit-1 op, 328 instances.
- The extractor's own tensors: `text_embedding_projection.video_aggregate_embed`
  BF16 `[4096, 188160]` and `.audio_aggregate_embed` `[2048, 188160]`, with
  biases. 188160 = **49 x 3840**, which is how many hidden states the tower must
  hand over (section 4).

## 3. The layer graph

From `modeling_gemma4.py`:

- Embedding: `embed_tokens(input_ids) * embed_scale` with
  `embed_scale = hidden_size ** 0.5` cast to the weight's dtype
  (`:1465-1477`, `:1608-1610`) - so `bf16(sqrt(3840))`, and the reference's own
  comment records that the cast is deliberate.
- RMSNorm (`:197-212`): computed in **F32** (`hidden_states.float()`), as
  `x * pow(mean(x^2) + eps, -0.5)` - `pow`, explicitly not `rsqrt` - then
  `* weight.float()`, then cast back to the input dtype.
- Attention (`:1183-1296`), per layer type:
  - `q = q_norm(q_proj(x))` then RoPE; `k = k_norm(k_proj(x))` then RoPE;
    `v = v_norm(v_proj(x))` **without** RoPE, or for a `k_eq_v` layer
    `v = v_norm(k_proj(x))` - the raw projection, not the normalised/rotated key.
  - `head_dim` is 512 on the full layers and 256 on the sliding ones; the KV head
    count is 1 and 8 respectively, repeated by `num_key_value_groups`
    (`eager_attention_forward`, `:841-842`).
  - `scaling = 1.0`, not `head_dim ** -0.5` (`:1200`; the forward passes it
    explicitly, so the eager default never applies).
  - `scores = (q @ k^T) * 1.0`, then `+ mask`, then softmax **in F32** and a cast
    back to the query dtype (`:845-855`), then `@ v`.
- MLP (`:1074-1090`): `down_proj(gelu_pytorch_tanh(gate_proj(x)) * up_proj(x))`.
- Every layer ends with `hidden_states *= layer_scalar`, a buffer of ones
  (`:1389`, `:1462`) - a multiply that changes nothing in value, and the port
  keeps it so a checkpoint that carries a non-one scalar stays coherent.
- The final `model.norm` is applied after the loop (`:1736`).

## 4. Position, mask, and the output contract

- `position_ids = arange(sequence_length)` (`:1689-1691`) - the left padding does
  NOT shift positions; token 0 of the padded window is position 0.
- RoPE (`:1093-1181`, `modeling_rope_utils.py:187-252`):
  - sliding: `dim = head_dim = 256`, `base = 1e4`, 128 frequencies, `emb =
    cat(freqs, freqs)` = 256.
  - full: `head_dim = global_head_dim = 512`,
    `rope_angles = int(0.25 * 512 // 2) = 64` rotated frequencies followed by
    `192` **zeros**, so 256 frequencies and 192 of them are unrotated.
  - both cast `cos`/`sin` to the input dtype (`:1180`), which is BF16 here.
- Masks: `create_causal_mask` for the full layers and
  `create_sliding_window_causal_mask` for the sliding ones (`:1694-1708`).
  Their function is `kv_idx <= q_idx` (`masking_utils.py:76-80`) AND, for sliding,
  `kv_idx > q_idx - sliding_window` (`:92-100`), AND the padding mask on the key
  axis (`:168-179`), then `where(keep, 0.0, finfo(dtype).min)` (`:608-610`) -
  and that `dtype` is `inputs_embeds.dtype`, BF16
  (`masking_utils.py:942` builds it from `inputs_embeds`, and `:990` passes it
  to the mask interface). So the fill is `finfo(bfloat16).min`
  (-3.3895313892515355e+38), a **finite** value, added to the BF16 scores in
  BF16: an earlier version of this line called it an F32 `-inf` mask and that
  was wrong. Two consequences the port depends on: the add rounds to BF16 once
  (torch computes it in float and stores BF16), and a fully masked row - every
  left-padding row - softmaxes to uniform rather than to NaN, because
  `x - max` is `fill - fill = 0` for every entry.
- The tower must return the **49** hidden states the extractor stacks:
  `FeatureExtractorV2.forward` (`feature_extractor.py:115-129`) takes
  `torch.stack(hidden_states, dim=-1)` and its aggregate embeds are
  `[out, 49 * 3840]`. `enabled_encoder`'s `encode` asks the inner model for
  `output_hidden_states=True` (`encoders/base_encoder.py:67`) and returns the
  tuple with the mask, one pair per prompt.
- The extractor itself is unit 4; what unit 2 owes it is the 49 states in order
  (embedding first, then layers 0..47), BF16, at `[1, 1024, 3840]` each.

## 5. The tokenizer, and what unit 3 needs

`tokenizer.py:30-63` with `TOKENIZER_MAX_LENGTH = 1024`
(`gemma_assets.py:162`):

1. strip the text;
2. tokenize with truncation to 1024;
3. prepend `bos_token_id` (2) unless the tokenizer already emitted it first, and
   re-truncate to 1024;
4. left-pad to 1024 with `pad_token_id`;
5. return `[(token_id, attn), ...]` - and that second element, which the plan
   record calls a per-token weight, is the **attention mask**:
   `encoders/base_encoder.py:56-66` builds `attention_mask` from exactly it.

The assets are all inside the checkpoint, as U8 tensors: `tokenizer_json`
32,169,626 B (the HF `tokenizers` JSON) plus `hf_asset__tokenizer_config.json`
2,749 B, `hf_asset__processor_config.json` 1,382 B,
`hf_asset__chat_template.jinja` 17,466 B and
`hf_asset__generation_config.json` 255 B. So unit 3 needs no external file, and
`build_gemma_hf_tokenizer` (`gemma_assets.py:241-250`) is the sidecar-to-tokenizer
recipe: `PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_buffer(...))`
with the config's kwargs minus `_TOKENIZER_CONFIG_SKIP`.

Two properties of the tokenizer are load-bearing and must be checked against the
JSON rather than assumed: the model is a byte-level BPE with a 262144 vocabulary,
and the `tokenizer_config.json` sidecar carries the normalization and the added
tokens the wrapper relies on.

## 6. The two ATen formulas, read at the pinned torch

Two expressions in section 3 belong to PyTorch, not to the model file, so they
were read from torch 2.13.0's own sources (`requirements/stage2-observed.txt:7`
pins `torch==2.13.0+cu132`), fetched read-only into
`/data/refs/torch-2.13.0-aten`:

- **`pow(mean + eps, -0.5)` is `rsqrtf`.** `Gemma4RMSNorm._norm` calls
  `torch.pow(mean_squared, -0.5)` and its own comment says `pow` was chosen over
  `sqrt`/`rsqrt`; `aten/src/ATen/native/cuda/PowKernel.cu:170-179` shows why that
  does not matter, `pow_tensor_scalar_kernel` dispatches the exponent `-0.5` to
  `rsqrt_kernel_cuda`, which is `c10::cuda::compat::rsqrt` =
  `::rsqrtf` for a float (`aten/src/ATen/native/cuda/UnaryOpsKernel.cu:96`,
  `c10/cuda/CUDAMathCompat.h:122-123`). The port's own
  `h3d_rms_norm_kernel` therefore already uses the reference's primitive, and
  the two differ only in the *reduction order* of `mean(x^2)` - torch's own
  tree against the port's - which is a last-bit difference to name and measure,
  not to chase. The one real gap is the shape of the call:
  `h3d_rms_norm` requires a weight, and Gemma's `v_norm` is built
  `with_scale=False` (`modeling_gemma4.py:1220`), so the tower needs the weight
  to be optional.
- **`gelu_pytorch_tanh`** is `GeluCUDAKernelImpl` with `GeluType::Tanh`
  (`aten/src/ATen/native/cuda/ActivationGeluKernel.cu:21-31`), in the operand's
  opmath type (float for BF16):
  `kBeta = M_SQRT2 * M_2_SQRTPI * 0.5`, `kKappa = 0.044715`,
  `x_cube = x * x * x`, `inner = kBeta * (x + kKappa * x_cube)`,
  `y = 0.5 * x * (1 + tanhf(inner))`, one cast back to BF16. The constants are
  that expression, compiled the same way - not `0.7978845608028654` written out
  by hand.

## 7. What this changes in the port plan- The plan's unit 2 line says "48 layers, three norms per layer": it is **four**
  (section 2).
- The plan's unit 3 line says "per-token weights, not a plain id list": the
  second element is the attention mask (section 5), which is a simpler contract -
  the mask is what the encoder forwards.
- The tower's gate (section 3 item 3 of the handoff) is the operator-produced
  golden; the 49 hidden states and the tokenizer dump are what it must carry.
- Nothing in the tower needs the vision or audio towers in the same checkpoint
  (`vision_model.*`, `multi_modal_projector.*`, `audio_projector.*`), and nothing
  needs the generation path: `enhance_t2v`/`enhance_i2v` and the sampling
  parameters are prompt-enhancement, not conditioning.
