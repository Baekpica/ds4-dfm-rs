# Gemma-4 26B A4B: the reference semantics the engine must reproduce

This file is the evidence record for the Gemma-4 VQ engine. Every kernel in
`ds4_gemma4_gpu.cuh` and `cuda/gemma4_*.cuh` cites a line here, and every line
here traces to the pinned reference source or to the checkpoint itself. It was
written before any kernel, because nothing in this tree or on disk stated these
semantics: the checkpoint ships no modeling code, and the artifact carries
weights only.

## 1. Pinned source of truth

The checkpoint repository (`google/gemma-4-26B-A4B-it`) carries `config.json`,
`generation_config.json`, `tokenizer*` and safetensors. It carries no
`modeling_gemma4.py`, so the reference implementation is Hugging Face
Transformers, pinned here at one commit:

    repo     huggingface/transformers
    commit   cc1451df3b5b218295bc16803569256c53c9d6e2
    date     2026-11-02T18:40:40Z
    subject  [Improvement] Formalize KV sharing (#47290)

Files read at that commit, with sha256 (fetched 2026-11-02, kept under
`/data/tmp/gemma4-ref/` for the duration of the port):

    modeling_gemma4.py                       257dc8a7dc36c6d331351e9f0d8f0903bbe69749d2b9327c5cc9942bec0deafe
    configuration_gemma4.py                   961665805532efd0829aa5665c6b9d2c12ac43a0ff91b75eadca421d08ad8ce0
    modeling_rope_utils.py                    ddec71d6c9343af437f656812177fe5c98f6057fd5240c5cbf208520c2872745
    masking_utils.py                          a18fd6bcba9b5870e3a3b147fa3e7d38c1c934282d056cdefbad3a1965fa28fd
    activations.py                            5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665
    modeling_gemma4_assistant.py              7d12d0922085afa6e60cbf05bb3afdca8f91a027895bcf42bf40479f4e1a953b
    configuration_gemma4_assistant.py         a40e52323b882d844628d8799eaa1b1c7633851867216b327ac32a06b83141d9
    generation/candidate_generator.py         0fa7538233fb3b738d1b840e0f28fefaa93506a41b7d4fb6da2853a08454734f

The trunk checkpoint is `google/gemma-4-26B-A4B-it` at revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (recorded in the artifact's
`release.json` as `source_revision`); the drafter is
`google/gemma-4-26B-A4B-it-assistant` at `6e5aaaf4c42b98394530b8fda2e95cadd65c151c`.

## 2. Configuration that is actually in force

Read from `/data/models/gemma-4-26B-A4B-it-hf/config.json` (the same file is
copied into the artifact), `text_config`:

    hidden_activation          gelu_pytorch_tanh        <- not silu
    final_logit_softcapping    30.0
    attention_k_eq_v           true                     <- full layers have no v_proj
    num_kv_shared_layers       0
    enable_moe_block           true
    num_experts / top_k        128 / 8
    moe_intermediate_size      704
    intermediate_size          2112                     <- dense FFN
    hidden_size                2816
    num_hidden_layers          30
    num_attention_heads        16
    num_key_value_heads        8                        <- sliding layers
    head_dim                   256                      <- sliding layers
    global_head_dim            512                      <- full layers
    num_global_key_value_heads 2                        <- full layers
    sliding_window             1024
    layer_types                sliding x5, full, ... (full at 5, 11, 17, 23, 29)
    rope_parameters            sliding: default, theta 1e4
                               full:    proportional, partial_rotary_factor 0.25, theta 1e6
    hidden_size_per_layer_input 0                       <- PLE disabled
    rms_norm_eps               1e-6
    tie_word_embeddings        true
    vocab_size                 262144
    max_position_embeddings    262144
    use_bidirectional_attention "vision"                <- text stays causal; see 3.9

`gemma4_text/configuration_gemma4.py:277-296` derives the per-layer config:
every `full_attention` layer overrides `head_dim` to `global_head_dim` and
`num_key_value_heads` to `num_global_key_value_heads`. That is where the two
attention geometries in `ds4_gemma4_plan.h` come from.

The absence of PLE tensors in the checkpoint (`model.embed_tokens_per_layer`,
`per_layer_model_projection`, `per_layer_projection_norm`) matches
`hidden_size_per_layer_input = 0`; per-layer inputs are therefore not part of
the port.

## 3. The forward pass, exactly

### 3.1 Embedding and its rounding

`gemma4_text/modeling_gemma4.py:1557-1561, 1428-1441`. The embedding table is a
`Gemma4TextScaledWordEmbedding`: a lookup multiplied by `embed_scale`, which is
constructed as the Python float `hidden_size**0.5` = 53.065996645686 and stored
in a *tensor* of the weight dtype (`nn.Buffer(torch.tensor(embed_scale))`).
Multiplied as a tensor, the scalar is bf16 for a bf16 checkpoint:

    bf16(53.065996645686) = 53.0    (spacing 0.25 at exponent 5; 212.26 -> 212)

The engine multiplies the dequantized embedding row by 53.0f, not by the exact
root. This is a measurable difference (0.12 %) and is deliberate.

### 3.2 RMSNorm

`modeling_gemma4.py:204-223`. Computed in float32:

    y = x * pow(mean(x^2) + eps, -0.5)          # float32, eps = 1e-6
    y = y * weight.float()                       # when with_scale
    out = y.to(x.dtype)                          # one rounding to bf16

There is no `(1 + weight)` reparametrisation: the stored weight multiplies
directly. `with_scale=False` occurs on `v_norm` and on the router norm.

Measured invariant, not to be relied on: in all 30 layers the `q_norm` and
`k_norm` weight vectors are constant across their 256 or 512 dimensions (layer 0
q 1.8779296875, k 1.494140625; layer 5 q 1.8759765625). The engine still reads
them as vectors, as the reference does, so a checkpoint that varies them needs
no code change.

### 3.3 Attention projections and order

`modeling_gemma4.py:1150-1262`. Per layer:

    q = q_proj(x)                     # [rows, 16 * head_dim]
    q = q_norm(q)                     # RMSNorm over head_dim, per head, with scale
    q = rope(q, cos, sin)
    k = k_proj(x)                     # [rows, kv_heads * head_dim]
    k = k_norm(k)
    k = rope(k, cos, sin)
    v = v_proj(x), or v = k_proj(x)   # attention_k_eq_v: v_proj is None on full layers
    v = v_norm(v)                     # RMSNorm with_scale=False, no rope
    attn = softmax(q @ k^T * 1.0, fp32) @ v
    out = o_proj(attn)

The attention scale is **1.0**, not `head_dim**-0.5`
(`modeling_gemma4.py:1167`: `self.scaling = 1.0`). The q/k norms replace it.

On `attention_k_eq_v` layers the *un-normed* key projection is also the value
input, so `v = v_norm(k_proj(x))` and `k = rope(k_norm(k_proj(x)))`: the same
weights, different norms, in that order. `ds4_gemma4_bind.inc` binds
`v = k` on exactly those layers (5, 11, 17, 23, 29).

RoPE is applied to q and k only, in the model dtype: `x * cos + rotate_half(x) * sin`
with `cos`/`sin` cast to the input dtype first (`modeling_gemma4.py:778-798`).

### 3.4 Masking

Text is causal: `is_causal = config.use_bidirectional_attention != "all"`
(`modeling_gemma4.py:1169`), and the checkpoint sets `"vision"`, so text
attention is ordinary causal.

The sliding layers add `kv_idx > q_idx - 1024`
(`masking_utils.py:88-97`), so the window is the query plus the 1023 preceding
positions. Combined with causality the visible set is `[q-1023, q]`.

The full layers see the whole causal prefix.

### 3.5 RoPE, both geometries

`modeling_gemma4.py:1078-1148` and `modeling_rope_utils.py:193-265`.

Sliding layers (`rope_type: default`, `head_dim` 256, theta 1e4):

    inv_freq[i] = 1 / 10000^(2i/256),  i = 0..127
    freqs       = pos * inv_freq
    cos = cos(cat(freqs, freqs)), sin = sin(cat(freqs, freqs))   # length 256

Full layers (`rope_type: proportional`, `head_dim` 512, theta 1e6,
`partial_rotary_factor` 0.25):

    rope_angles = int(0.25 * 512 // 2) = 64
    inv_freq[i] = 1 / 1000000^(2i/512),  i = 0..63
    inv_freq[i] = 0,                      i = 64..255
    cos = cos(cat(freqs, freqs)), sin = sin(...)                  # length 512

`rotate_half` is the half-split convention: out[:256] = -x2*sin + x1*cos,
out[256:] = x1*sin + x2*cos. Because the frequency vector is zero beyond index
63, only dimensions 0..63 and 256..319 rotate on full layers (128 of 512). The
note in that file says proportional RoPE always returns an encoding the size of
`head_dim`, which is why cos/sin are 512 long rather than 128.

Concrete values for the golden test: sliding `inv_freq[0..2]` =
(1.0, 0.930572040929699, 0.8659643233600653); full `inv_freq[63]` =
0.033376246942920386, `inv_freq[64]` = 0.0.

### 3.6 The decoder block

`modeling_gemma4.py:1368-1427`, in this order and with no other residual
adds:

    residual = x
    h = input_layernorm(x)
    h = self_attn(h)                       # 3.3
    h = post_attention_layernorm(h)
    x = residual + h

    residual = x
    h = pre_feedforward_layernorm(x)
    d = mlp(h)                             # dense, 3.7
    d = post_feedforward_layernorm_1(d)

    w, idx = router(residual)              # NOTE: the pre-FFN residual, not h
    e = pre_feedforward_layernorm_2(residual)
    e = experts(e, idx, w)                 # 3.7
    e = post_feedforward_layernorm_2(e)

    h = d + e
    h = post_feedforward_layernorm(h)
    x = residual + h

    x = x * layer_scalar                   # last operation of the block

`layer_scalar` is a bf16 buffer of shape [1], not a parameter, and the
checkpoint's 30 values are not ones (min 1.390625, max 1.829102, mean 1.761979
over layers 0..29; layer 0 = 1.390625, layer 29 = 1.5703125). Omitting it, or
applying it before the residual add, changes every hidden row.

### 3.7 Dense FFN and experts

Dense (`modeling_gemma4.py:1059-1075`):

    down( act(gate(x)) * up(x) ),   act = gelu(approximate="tanh")

The activation is `gelu_pytorch_tanh` from the config, which is
`torch.nn.functional.gelu(x, approximate="tanh")`
(`activations.py:31-51`), i.e. the tanh approximation
`0.5x(1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, not SiLU. It is the same
activation inside every expert.

Experts (`modeling_gemma4.py:1263-1301`): each expert is a packed
`gate_up_proj` of shape `[128, 2*704, 2816]`, split with `.chunk(2, -1)` — gate
first, up second — then `down_proj` of shape `[128, 2816, 704]`. Per selected
expert:

    gate, up = W_gate_up[e] @ x
    y = act(gate) * up
    y = W_down[e] @ y
    y = y * top_k_weights[t, pos]          # fp32 weight promotes the product
    out[t] += y                            # rounded to bf16 at the add

The artifact stores the pair split into `experts.<e>.gate_proj` and
`experts.<e>.up_proj`, in that order, which
`ds4_gemma4_bind.inc` relies on.

### 3.8 The router

`modeling_gemma4.py:1302-1337`, in this order:

    s = rms_norm_no_scale(x)               # eps 1e-6, float32, no weight
    s = s * router_scale * hidden_size**-0.5
    logits = proj(s)                       # [rows, 128]
    p = softmax(logits, dim=-1, dtype=float32)
    w, idx = topk(p, 8)
    w = w / sum(w)                         # renormalise the selected eight
    w = w * per_expert_scale[idx]          # AFTER renormalisation, not before

`router_scale` is a per-hidden-dimension vector of length 2816 and
`per_expert_scale` a per-expert vector of length 128; both live in
`model/.../router.{scale,per_expert_scale}` in the checkpoint and are bound as
`mlp.gate.scale` and `mlp.gate.per_expert` in the artifact. Layer 0 samples:
`router.scale[0..2]` = (2.98828125, 3.005859375, 3.00390625),
`per_expert_scale[0..2]` = (1.87109375, 1.876953125, 1.8740234375).

Rounding order matters here and is not symmetric. `s * self.scale` multiplies
two bf16 tensors, so the product is rounded to bf16; `* hidden_size**-0.5`
then multiplies a bf16 tensor by a Python float, which torch computes in
float32 from the unrounded constant and rounds once to bf16. So
`hidden_size**-0.5` is *not* pre-rounded to bf16 the way `embed_scale` is.
`w * per_expert_scale[idx]` is float32 times bf16, which promotes to float32
and keeps the weights full precision until the expert output is rounded.

### 3.9 Final norm, head and softcap

`modeling_gemma4.py:1687` and `1778-1856`. After layer 29:

    x = model.norm(x)                      # RMSNorm with weight, 3.2
    logits = x @ embed_tokens^T            # tied head, no separate lm_head weight
    logits = tanh(logits / 30.0) * 30.0    # final_logit_softcapping

The head is the embedding table (`_tied_weights_keys`; the checkpoint has no
`lm_head.*` key, checked over all 1013 names), so the 262144x2816 table is read
twice per token: once for the input row, once for the logits. The softcap is
applied to the full vocabulary row, after the head, as three elementwise ops on
a bf16 tensor: divide by 30.0, tanh, multiply by 30.0. Each op computes in
float32 and rounds to bf16, so the engine applies the same three roundings.

`use_bidirectional_attention: "vision"` affects only image tokens in the
multimodal wrapper; no vision tensor exists in the VQ artifact, and the
text-only engine treats every token as causal.

## 4. The drafter (gemma4_assistant)

Reference: `modeling_gemma4_assistant.py` at the same pin, driven by
`generation/candidate_generator.py:1211-1360`
(`SinglePositionMultiTokenCandidateGenerator`).

Config of `google/gemma-4-26B-A4B-it-assistant`
(`modeling_gemma4_assistant/configuration_gemma4_assistant.py:44-140` and the
checkpoint's `config.json`):

    backbone_hidden_size     2816
    hidden_size              1024
    num_hidden_layers        4          (layer 3 is full_attention)
    num_attention_heads      16
    head_dim                 256 / 512  (sliding / full)
    intermediate_size        8192       (dense MLP, no MoE)
    use_ordered_embeddings   false      <- the centroid mask path is NOT used
    num_centroids            2048       (present in the config, unused)
    centroid_intermediate_top_k 32      (unused)
    num_kv_shared_layers     4          (every drafter layer consumes trunk KV)
    final_logit_softcapping  absent

Because `use_ordered_embeddings` is false, `self.masked_embedding` is never
built and the drafter's logits are a plain `lm_head` over its own tied
262144x1024 embedding (`modeling_gemma4_assistant.py:137-172`). The centroid
routing described in the class docstring therefore does not run for this
checkpoint; an engine that implements it would be implementing a path the
released drafter does not take.

The draft step (`candidate_generator.py:1329-1360`):

    for each of up to num_assistant_tokens steps:
        inputs = cat[ embed(last_token_id), last_hidden_state ]   # 2816 + 2816
        h = pre_projection(inputs)                                # [1024]
        h = drafter_layers(h, ky=trunk shared KV, position=pos)    # q_len = 1
        logits = lm_head(h)                                       # 262144
        last_token_id = argmax(logits)
        last_hidden_state = post_projection(h)                    # [2816]

Three details decide the numbers:

1. `position_ids` is a constant `input_ids.shape[1] - 1` for every step of the
   invoice round, i.e. the index of the last committed token
   (`candidate_generator.py:1327`). The assistant never advances the position.
2. The KV is the trunk's: `shared_kv_states[layer_type]`, truncated to the
   current committed length (`candidate_generator.py:1316-1319`), and the
   mask is bidirectional over that prefix
   (`modeling_gemma4_assistant.py:196-240`), which with `q_len == 1` is the
   same as causal.
3. `last_hidden_state` fed back is the *post-projected* trunk-width state, and
   the embedding half comes from the trunk's embedding table, not the
   drafter's.

### Which trunk layers must publish KV

The trunk's `config.json` has `num_kv_shared_layers = 0`, and the assistant
comment in `configuration_gemma4.py:246-250` states that "the last non-consumer
layer of each type stores its KV states, even if no layer reuses them:
assistant models (e.g. Gemma4Assistant) read them from the `shared_kv_states`
returned by the target model".

The pre-#47290 implementation states the same rule as an explicit formula
(`git show 4888db4e:src/transformers/models/gemma4/modeling_gemma4.py`,
`Gemma4TextAttention.__init__`):

    first = num_hidden_layers - num_kv_shared_layers          # 30 for this trunk
    store = not shared and layer_idx == len(prev) - 1 - prev[::-1].index(type)

which evaluates to layer 28 (last sliding) and layer 29 (last full). The engine
publishes exactly those two layers' K and V, post-norm and post-rope, at full
length, whenever drafting is enabled.

Note for the record: at the pinned HEAD, `kv_sharing_modes` returns
`[None] * num_hidden_layers` when `num_kv_shared_layers == 0`, so the store
branch never runs for a 0-sharing trunk — the code and the released configs
disagree after #47290. The port follows the documented rule and the previous
implementation, not the regression, and this is the one place where the
reference is self-inconsistent.

## 5. What the engine must therefore not do

- Do not use SiLU anywhere. The activation is GELU, tanh approximation.
- Do not scale attention by 1/sqrt(head_dim). The scale is 1.0.
- Do not apply the router's per-expert scale before renormalising the top-8.
- Do not route from the normed FFN input. The router reads the post-attention
  residual.
- Do not apply `layer_scalar` at the block start or per branch; it multiplies
  the block output.
- Do not tie the drafter's head to the trunk's embedding, or implement the
  centroid path: the released assistant uses its own 262144x1024 table and
  `use_ordered_embeddings = false`.
- Do not advance `position_ids` inside a draft round.
- Do not forget the vocabulary softcap at 30 or the bf16-rounded
  `sqrt(hidden_size)` embedding scale.

## 6. Measured differences worth knowing

- `embed_scale`: exact root 53.065996645686 vs the reference's bf16 53.0.
- `layer_scalar` range: 1.390625 .. 1.829102, so a block output is scaled by
  up to 83 %.
- q/k norm weights are per-head vectors of length `head_dim`; measured constant
  across all 30 layers (section 3.2), which is an invariant of this checkpoint,
  not of the architecture.
- `router.scale` layer 0 spans 2.9765625 .. 3.013671875 (20 distinct values over
  2816 entries) and `per_expert_scale` 1.870117 .. 1.877930 (9 distinct over
  128); both are real per-index vectors, so indexing them by dimension and by
  expert is not optional.

## 7. How this document is refreshed

The reference is pinned; a re-read means fetching the same files at the same
commit and diffing against the hashes in section 1. If upstream moves and the
semantics change, the files listed here are the ones to diff, in this order:
`modeling_gemma4.py`, `modeling_rope_utils.py`, then the assistant pair.
