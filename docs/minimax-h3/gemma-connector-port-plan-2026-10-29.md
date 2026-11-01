# The Gemma4 text encoder and the AV connector — port map and numeric reference (2026-10-29)

Operator decision this day: port the Gemma4 encoder and the connector into the
Rust host and the native CUDA backend, rather than build the fixed prompt cache
once with the release's Python. This record is the port's source read: what the
artifact holds, what the reference math is, where each piece lives, and how each
unit is gated. It states no math that was not read first-hand this session.

## 1. Scope, and what this port does and does not buy

The transformer is conditioned on Gemma-4-12B features, post-connector. The release
computes them once offline and serves two bf16 tensors
(`runtime/prompt_cache.py:37`, `CONTEXT_SHAPES` = video `(1,1024,4096)`, audio
`(1,1024,2048)`); at request time neither Gemma nor the connector is loaded
(`runtime/prompt_cache.py:100-113`, `stage2_gemma_loaded`/`gemma_encode_calls` 0).
Porting the encoder therefore buys **a Python-free build of the conditioning**,
and per-request prompts if we want them later. It does not change any
request-time figure, and the encoder cannot be resident: measured below, its
weights are 14.32 GiB against the release's 2.8 GiB of free pool
(`configs/default.json` recipe; the release page records 116.9 GiB resident of
119.68 GiB).

## 2. The artifact, measured

File (recipe file, `runtime/prompt_cache.py:18-19` pins name and bytes):
the released Gemma-4-12B INT8 convrot shard,
15,372,969,374 B, sha256 `6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f`,
now at `/mnt/h3-artifacts/Gemma4/` on the Spark (hashed on this workstation
before the move). A bf16 sibling (26,263,858,182 B) sits beside it; the recipe
does not name it and the release states a bf16-derived cache is not
interchangeable (`docs/setup.md:127-129`).

Contents (safetensors header, read over gvfs):

- 1342 tensors: 328 I8 weights, 328 F32 `weight_scale` (shape `[out, 1]`, i.e.
  per-output-row), 353 BF16, 333 U8.
- Of the U8, 328 are `*.comfy_quant` metadata blobs and 5 are embedded HF
  assets: `chat_template.jinja`, `generation_config.json`,
  `processor_config.json`, `tokenizer_config.json`.
- Each blob is JSON: `{"format": "int8_tensorwise", "convrot": true,
  "convrot_groupsize": 256}` (72 bytes, read from
  `model.layers.0.mlp.down_proj.comfy_quant`). `runtime/cache_ops/int8_header.py:512-540`
  reads the same key, defaulting to 256.
- Text tower: 48 layers, hidden 3840, intermediate 15360, `head_dim` 256 with
  `global_head_dim` 512, `layer_types` mixing `sliding_attention` and
  `full_attention`, `attention_k_eq_v` true, `gelu_pytorch_tanh`,
  `rms_norm_eps` 1e-6; four norms per layer
  (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `post_feedforward_layernorm`), all BF16 [3840]. Config from the file's own
  `__metadata__["gemma_config"]`
  (`architectures: ["Gemma4UnifiedForConditionalGeneration"]`, `model_type`
  `gemma4_unified`, `gemma_version` `gemma4-12b-…-v1`).
- Caption projection: `text_embedding_projection.video_aggregate_embed.weight`
  [4096, 188160] and `...audio_aggregate_embed.weight` [2048, 188160], BF16 with
  biases; 188160 = 49 x 3840.
- Connector (inside the transformer checkpoint, `model.diffusion_model.` prefix):
  `video_embeddings_connector.transformer_1d_blocks.0..7`, 16 tensors and
  403,005,504 B each; `audio_embeddings_connector.*` 16 tensors and
  100,839,488 B each; `learnable_registers` [128, 4096] and [128, 2048].

## 3. The INT8 convrot math, as the reference implements it

Read from `comfy-kitchen` 0.2.26 (wheel `comfy_kitchen-0.2.26-cp312-abi3-manylinux_2_24_{aarch64,x86_64}`,
`requirements/offline-context.txt` names the API). The CUDA backend is a compiled
`_C.abi3.so`; the **eager** and **triton** backends ship as Python source, and the
backend is selectable in-process (`comfy_kitchen/__init__.py:764-793`
`set_backend_priority` / `enable_backend` / `disable_backend` / `list_backends`).
That is the handle for gating our port on hardware without the extension.

- **Rotation matrix** (`tensor/int8_utils.py:11-45`): a normalized *regular*
  Hadamard, base `h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, Kronecker
  powered until `size` (must be a power of 4), then divided by `sqrt(size)`.
  For group size 256 that is three Kronecker steps (4 -> 16 -> 64 -> 256) and a
  divide by 16.
- **Activation rotation** (`tensor/int8_utils.py:55-70`): reshape the last dim
  into groups of `group_size`, multiply each group by `H`. Weights are rotated
  offline by `H^T` (`:44-52`), and `h4` is symmetric, so the two agree.
- **Row-wise activation quantization** (`backends/eager/quantization.py:857-871`):
  `scale = clamp(max|x| over the row / 127.0, min=1e-30)` in float32, then the
  division is done in the activation's dtype (`_int8_scale_for_math` casts the
  scale to `x.dtype`), rounded with `round_` (half-to-even) and clamped to
  `[-128, 127]`, cast to int8. For a bf16 activation the divide-and-round is
  therefore a bf16 computation, not fp32.
- **Accumulation** (`:749-780`): int8 x int8 to int32 (`torch.int8_mm`, else
  `torch._int_mm` = cuBLASLt; zero-padded on M/K/N to the required alignments).
- **Epilogue** (`:965-1035`): `result.float() * (x_scale_row * weight_scale_col)`
  in fp32 chunks, cast to `out_dtype` (bf16), then add bias in that dtype.
- The call site is `runtime/cache_ops/int8_policy.py:97-104`:
  `torch.ops.comfy_kitchen.int8_linear(value, weight, weight_scale, bias,
  out_dtype_code, convrot=True, convrot_groupsize, None)`, wrapped by
  `DevInt8ConvRotLinear` (`:58-107`), whose weights are I8 `[out, in]` with a
  per-row F32 scale and an optional bias.

## 4. The driver chain the port must reproduce

The release's `text_encoders/gemma/` package:

| Step | File | What the port must mirror |
|---|---|---|
| Entry | `encoders/base_encoder.py:30-72` | the encoder's `encode(prompts)` returns `[(hidden_states, mask)]`; system-prompt and generation defaults at `:76-92`, `:170-190` |
| Assets | `gemma_assets.py:59-153`, `:230-292` | the embedded sidecars become an HF config, tokenizer and processor; `resolve_gemma_weight_paths` |
| Weights | `encoders/encoder_configurator.py:231-481` | key ops for gemma3 / gemma4-unified / gemma4-dense, rotary buffers, model construction |
| Tokenizer | `tokenizer.py:11-63` | `tokenize_with_weights` — the second element is the attention mask, not a per-token weight (`base_encoder.py:56-66`) |
| Features | `feature_extractor.py:12-128` | `norm_and_concat_per_token_rms`, `_rescale_norm`, `FeatureExtractorV1/V2` with the aggregate embed |
| Connector | `embeddings_connector.py:15-256` | eight `_BasicTransformerBlock1D`, `_replace_padded_with_learnable_registers`, the AV wiring |
| Processor | `embeddings_processor.py:10-117` | `process_hidden_states`, `convert_to_additive_mask`, right-pad ordering, binary mask |
| Consumer | `runtime/prompt_cache.py:24-70` | the payload contract the cache validates |

## 5. Units, each with its gate

1. **The int8 convrot kernel** in `native/h3`: grouped Hadamard rotation,
   row-wise int8 quantization in the activation dtype, int8 accumulate, fp32
   epilogue, bias. Gate: bit-compare against the eager reference on fixed inputs
   (deterministic, no model), then against the CUDA backend on the same inputs
   to quantify the backend spread.
2. **The Gemma4 text tower**: 48 layers, four norms, sliding and full attention,
   `attention_k_eq_v`, `gelu_pytorch_tanh`, HF-side rotary. Gate: hidden states
   against the pinned encoder run on the fixed prompt.
3. **The tokenizer and processor**: tokenize_with_weights plus the embedded chat
   template and processor config. Gate: identical id/weight sequences against the
   pinned tokenizer for the fixed prompt and a few probes.
4. **The feature extractor and connector**: per-token RMS, aggregate projection,
   eight 1D blocks, learnable registers. Gate: the two contexts against the
   pinned `process_hidden_states` output.
5. **The cache payload**: emit and load the two contexts in the contract's shape
   and dtype so the path in `crates/ds4-core/src/minimax_h3/` can consume
   them; gate on `validate_payload`'s rules (`prompt_cache.py:44-70`).

## 6. Open items and risks

- **The int8 connector file is absent**: the connector's INT8 weights live only in
  the INT8 convrot transformer shard (`…-transformer-comfy-int8-convrot.safetensors`),
  21,504,034,224 B (not on either host, searched this session). Unit 4 can be
  gated on the BF16 connector weights we do have; that is a named deviation, not
  the recipe, and the delivered cache must come from the recipe's INT8 weights.
- **Where the reference runs**: `runtime/cache_builder.py:105-107` refuses
  anything but a single SM121 GPU, so the *delivered* cache is Spark-only. The
  op itself does not carry that gate, and the eager backend is forceable, so the
  development gates can run on the workstation (x86_64 wheel available; the
  workstation already has torch 2.13.0 and transformers 5.14.1, which satisfies
  the pin's `< 5.15`). Backend and device differences are a named deviation to
  quantify, not a free assumption.
- **Tokenizer fidelity** is the likeliest silent failure: the second element of
  each `tokenize_with_weights` pair is the attention mask, and the mask is what
  `feature_extractor` uses to zero the padded positions and to weight its
  per-token statistics (`feature_extractor.py:30-66`), so a tokenizer that
  disagrees on where the real tokens end produces a plausible but wrong context.
  Corrected 2026-10-29: the pair is `(id, mask)`, not `(id, weight)`.
- **Numeric contract**: this port's gate is against pinned Python, not against
  bit identity; the int8 kernel itself can be
  held to bit equality with the eager reference because it is deterministic.

## 7. Reproduction

Header and blob reads: a 20-line Python reader over the gvfs mount (the pattern
used throughout this session) — 8-byte little-endian header length, JSON header,
`data_offsets` relative to `8 + header_length`. Reference sources:
the pinned release's `text_encoders/gemma/` package and the
comfy-kitchen wheel unpacked from
`python3 -m pip download comfy-kitchen==0.2.26 --no-deps -d <dir>` (both
`manylinux_2_28_aarch64` and `manylinux_2_28_x86_64` exist).
