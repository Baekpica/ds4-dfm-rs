# Unit 4: the AV connector and the feature extractor

The conditioning's second half. Unit 2b ends at the tower's 49 hidden states and
unit 3 at the ids; this unit turns those states into the two bf16 contexts the
release serves - video `(1, 1024, 4096)` and audio `(1, 1024, 2048)`
(`runtime/prompt_cache.py:29-30`) - with the released feature extractor and its
two 8-block 1D connectors.

Everything below was read first-hand from the released `text_encoders/gemma/`
package (the pin the release runs,
`d1511477` = v1.2.0) and from the checkpoints' own headers. No GPU job was run
for the reading; the runs are in sections 5 and 6.

10→## 1. What the graph is, and the geometry the checkpoint fixes

`EmbeddingsProcessor.process_hidden_states` (`embeddings_processor.py:96-117`):

    stack the 49 states            torch.stack(states, dim=-1)      :120
    per-(token, layer) RMS         norm_and_concat_per_token_rms     :48-64
    reshape to D*L                 normed.reshape(B, T, D * L)       :62
    two aggregate embeds           _rescale_norm then Linear         :121-128
    additive mask                  convert_to_additive_mask          :15-19
    right-pad permutation          _compute_right_pad_order           :23-38
    video connector                8 x _BasicTransformerBlock1D       :153-191
    audio connector                the same at width 2048
    final weightless RMS           rms_norm(hidden_states)            :189

The geometry is not the constructor's defaults; it is the transformer shard's
own `__metadata__.config.transformer` (read from the bf16 dev file's header):
`connector_num_attention_heads` 32 with `connector_attention_head_dim` 128 for
video and `audio_connector_attention_head_dim` 64 for audio,
`connector_num_layers` 8, `connector_positional_embedding_max_pos` `[4096]`,
`connector_num_learnable_registers` 128, `connector_apply_gated_attention` true,
`rope_type` "split", `frequencies_precision` "float64". `connector_ff_bias` is
absent, so `embeddings_connector.py:214`'s default `True` stands - which is what
the checkpoint's own `ff.net.0.proj.bias` and `ff.net.2.bias` tensors show.

30→Three details are the reference's and are reproduced rather than simplified:

- **The extractor's output order is `d * 49 + l`, not `l * 3840 + d`.**
  `normed.reshape(B, T, D * L)` (`:62`) flattens the stacked `[B, T, D, L]` in C
  order, so a hidden dimension's 49 layers are contiguous. The aggregate embeds
  are `[4096, 188160]` and `[2048, 188160]` and read that order. A port that
  wrote layer-major would be right in shape and wrong in value.
- **The connector's attention takes no mask.** `_replace_padded_with_learnable_
  registers` returns `zeros_like(additive_attention_mask)`
  (`embeddings_connector.py:151`), i.e. every key valid, so the eight blocks run
  dense. Only the *register fill* is mask-dependent.
- **The per-(token, layer) RMS squares in BF16 before it accumulates.** `:60`
  computes `encoded_text ** 2` as its own tensor, so the sum reads BF16 squares;
  the mean, the `1e-6` add and the `rsqrt` then each round again.

40→## 2. The artifact the unit does not have

The recipe names two INT8 files (`runtime/prompt_cache.py:16-28`). The Gemma one
is present. The connector one is **not**: the manifest pins
the INT8 convrot transformer shard
(`diffusion_models/…-transformer-comfy-int8-convrot.safetensors`),
21,504,034,224 B, sha256 `2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b`
(`configs/checkpoints.json`, key `offline_connector`), and `cache_builder.py:112-114`
rejects any connector whose name or byte count disagrees.

The Spark's checkpoint directory holds a second INT8 convrot transformer shard,
whose byte count is exactly the recipe's 21,504,034,224 but whose file name
carries a different variant tag than the pin. It is not the pinned artifact and
it is not used here.
Whether the two files' connector tensors coincide (the second file's LoRA deltas
target the transformer blocks and the edges, not `*_embeddings_connector.*`) is
not established, and the recipe's own builder would refuse the file by name.

Unit 4 therefore runs on **BF16 connector weights** (the release's own
BF16 Gemma-4-12B shard and its BF16 transformer shard). That is a named
deviation, not the recipe: `docs/setup.md:127-129` states a BF16-derived cache
is not
interchangeable with the INT8 one. The INT8 convrot route itself is already
built and gated (`h3d_linear_int8_convrot`, units 1 and 2b), so swapping the
operand source is a loader change, not a graph change.

## 3. The tensors, checked against the shards

`Processor::open` reads 262 tensors and 6,344,455,168 B, each checked against
its checkpoint descriptor (dtype and every dimension) before it reaches the
device:

| source | shard | tensors |
|---|---|---:|
| `text_embedding_projection.{video,audio}_aggregate_embed.{weight,bias}` | the Gemma checkpoint | 4 |
| `model.diffusion_model.{video,audio}_embeddings_connector.learnable_registers` | the transformer | 2 |
| per connector block: `attn1.{q_norm,k_norm}.weight`, `attn1.{to_q,to_k,to_v,to_out.0}.{weight,bias}`, `attn1.to_gate_logits.{weight,bias}`, `ff.net.0.proj.{weight,bias}`, `ff.net.2.{weight,bias}` | the transformer | 8 x 16 x 2 = 256 |

Per connector: 8 blocks, and the six int8 linears the recipe counts
(`connector_int8_linears` 96 = 8 x 6 x 2) are the attention's four projections
and the feed-forward's two. `q_norm`, `k_norm`, `to_gate_logits` and
`learnable_registers` stay BF16 in the recipe too.

## 4. The six kernels

The connector's norms, linears, SDPA, GELU and residual adds are the generic
`h3d_rms_norm`, `h3d_linear`, `h3d_sdpa`, `h3d_gelu_tanh` and `h3d_add`. Six
entries are new (`native/h3/h3_dit_gpu.{h,cu}`), all in the header's AV
connector section:

| entry | reference |
|---|---|
| `h3d_scale_constant` | `_rescale_norm`, `feature_extractor.py:68-69` |
| `h3d_connector_stack_scale` | `norm_and_concat_per_token_rms`, `:48-64` |
| `h3d_connector_interleave` | `normed.reshape(B, T, D * L)`, `:62` |
| `h3d_gather_rows` | `_apply_right_pad_order`, `embeddings_processor.py:40-42` |
| `h3d_register_fill` | `_replace_padded_with_learnable_registers`, `embeddings_connector.py:139-151` |
| `h3d_text_rope` | `apply_split_rotary_emb`, `rope.py:42-85` |
| `h3d_per_head_gate` | `PytorchGatedAttention`, `ops.py:94-106` |

The RoPE tables are built on the host rather than on the device: the reference's
grid is `generate_freq_grid_np` (`rope.py:86-108`) - float64 `theta **
linspace(0, 1, inner_dim / 2)` times `pi/2`, cast to float32 - and the angle is
`grid[g] * (2 * row / 4096 - 1)` (`:159`, `:136-143`). `rope_tables` in
`gemma_connector.rs` is that computation, and the gate shares it.

`h3d_sdpa` was hard-wired to `head_dim == 128` (the video VAE's width). The
connector needs 64 as well, so `h3d_attention_tiled_kernel` is now templated on
`HEAD_DIM` and `h3d_sdpa` dispatches 64 or 128. The change is a generalisation,
not a rewrite, and it was verified as one: `cuobjdump -sass` of the compiled
`<128>` instantiation is byte-for-byte the 4,146 instruction lines the
non-template kernel produced before the change.

## 5. The gate, and what it proves

`tests/h3_gemma_connector_gate` (unit 4's gate, `--probe` / default / `--prompt`).

**Op probes, bit-exact on both hosts.** Eight kernels against host
transcriptions of the same expression, the transcriptions reproducing the
device's reduction order (256 partials, the same halving tree):

    PROBE stack_scale    values=392      differ=0  verdict=residual pass
    PROBE interleave     values=200704   differ=0  verdict=exact    pass
    PROBE gather_rows    values=4096     differ=0  verdict=exact    pass
    PROBE register_fill  values=4096     differ=0  verdict=exact    pass
    PROBE scale_constant values=4096     differ=0  verdict=exact    pass
    PROBE rope_video     values=32768    differ=0  verdict=exact    pass
    PROBE rope_audio     values=16384    differ=0  verdict=exact    pass
    PROBE per_head_gate  values=32768    differ=0  verdict=residual pass

The two marked residual end in a libm call on the device (`rsqrtf`, `expf`) where
the transcription uses the host's own; both came back with zero differing
elements at these inputs, so the residue is below one bit here rather than merely
under a ceiling.

**The graph runs end to end on both hosts.** Workstation (sm_89), 49 synthetic
states at 128 rows, seed 1: `tensors=262 bytes=6344455168 load=53.410 run=0.088`,
`GEMMA-CONNECTOR-OK`. The Spark (sm_121), the same run on its own disk:
`load=1.199 run=0.231`. Two runs at seed 1 give identical stage hashes on each
host, and seed 7 gives different ones.

**The full path runs on the Spark.** `--prompt 'A golden retriever running on a
beach at sunset.'`: the checkpoint's own tokenizer (1024 wide, bos 2, 1013
padding), the tower's 48 layers, then the extractor and both connectors -
`rows=1024 padding=1013 load=1.216 run=0.403`, every stage hashed,
`GEMMA-CONNECTOR-OK`. This is the first run of the whole conditioning chain in
this port.

## 6. The cross-host finding: the BF16 linear is not bit-stable

The two hosts do **not** agree on the BF16 arm, and where they disagree is
exactly where a `h3d_linear` call sits:

| stage | FNV sm_89 | FNV sm_121 |
|---|---|---|
| `scales` | `f2231100c4002dff` | `f2231100c4002dff` |
| `normed` | `9629e928b5d6488f` | `9629e928b5d6488f` |
| `video_features` | `3fd024002cd20995` | `45af76b34d01d99b` |
| `audio_features` | `4a88cdca0a6595fa` | `9446ac95a33e3478` |
| `video` | `4cff505ef3e46d68` | `45d39e59a21fcab7` |
| `audio` | `b9d09900a73cd156` | `dc250578ea895bc1` |

The two stages above the first linear are **byte-identical** - they are this
port's own kernels, whose order is fixed. Everything downstream differs. Both
hosts' dumps at 128 rows were compared element by element (`relRMS` is over all
elements; `absRMS` is the difference's own RMS, against the output's):

| stage | values | differ | relRMS | absRMS | outRMS | 1 ulp | ~2 ulp | >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `video_features` | 524288 | 2.66% | 5.82e-04 | 1.06e-03 | 1.83 | 12560 | 253 | 197 |
| `audio_features` | 262144 | 2.71% | 5.87e-04 | 8.23e-04 | 1.40 | 6334 | 176 | 129 |
| `video` | 524288 | 26.30% | 5.53e-03 | 5.53e-03 | 1.00 | 72844 | 10231 | 13767 |
| `audio` | 262144 | 47.36% | 2.89e-03 | 2.89e-03 | 1.00 | 75665 | 8947 | 6651 |

The two contexts' output RMS is exactly 1 by construction - the connector's
closing weightless RMS norm. Most differing elements are a single BF16 ulp
(0.39-0.78% of the value, eight mantissa bits); the >5% tail is at outputs near
zero, where a fixed absolute difference is a large fraction. At the extractor
the absolute difference is ~1e-3 on outputs of RMS ~1.8, which is a relative
5.8e-4.

`scale_constant` is covered bit-exactly by the probe on both hosts, so the
divergence enters at the GEMM: `h3d_linear` dispatches to cuBLASLt, whose
accumulation order is architecture-specific, and the BF16 output rounding is
what makes an accumulator difference visible. It is larger than one reordering
of a single rounding predicts, so the two architectures do not merely reorder
the same sum.

Consequence, and the reason this is recorded rather than smoothed over: **the
BF16 arm is not a cross-host bit-stable gate, and the delivered cache must come
from the INT8 convrot route** - which unit 2b measured as bit-identical across
these two hosts, and which the recipe requires anyway.

## 7. What is still owed

- **The value gate.** This unit's gate is a wiring proof: op probes and a
  deterministic graph, not a comparison against the reference. The operator's
  golden (section 4c of the handoff) is what turns it into a value gate, and
  `--dump DIR` now writes one raw BF16 file per stage for exactly that
  comparison.
- **The INT8 connector arm**, blocked on the recipe's file (section 2).
- Unit 5, the cache payload (`prompt_cache.py:44-70`), which validates shape,
  dtype, finiteness and the prompt+recipe fingerprint but **not** values.

## 8. Reproduction

    make tests/h3_gemma_connector_gate CUDA_ARCH=sm_89      # or sm_121
    tests/h3_gemma_connector_gate --probe
    tests/h3_gemma_connector_gate --gemma <int8 gemma>.safetensors \
        --connector <bf16 dev transformer>.safetensors --rows 128 --seed 1 \
        --dump /tmp/dumpw
    tests/h3_gemma_connector_gate ... --prompt 'A golden retriever running on a beach at sunset.'

`rm -f native/h3/*.o` before every gate build. The commits are `9dc12b9e`
(the graph, the kernels, the gate), `3bc28fb4` (one host buffer in the loader)
and `31323152` (`--dump`); the Spark ran `c10.bundle` and `c11.bundle` from
`d6eacf8a`.
