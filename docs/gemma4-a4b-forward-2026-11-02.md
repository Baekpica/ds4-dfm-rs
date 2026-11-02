# Gemma-4-26B-A4B: the CPU forward and the OpenAI host (2026-11-02)

The fifth unit on this family. Registration made it selectable, decode proved
the values and the fused dot gave the shape a forward can afford. This one
ties them into a working model: a single-token forward over the release, the
checkpoint's own tokenizer, and a standalone OpenAI-compatible host that
answers a chat completion.

The gate is end to end and was run:

    prompt: "What is the capital of Sweden? Answer in one word."
    25 prompt tokens, 2 completion tokens, greedy
    reply:  "Stockholm"

through `POST /v1/chat/completions` (buffered), through the same route with
`"stream": true` (the three SSE deltas "Hello", " there", "!" for a greeting
prompt), and through the CLI example on the same question.

## What landed

- `crates/ds4-core/src/gemma4_a4b/weights.rs` — the streaming matrix.
  `Matrix::open` maps the `RQVQ` file read-only, re-uses the header check the
  decode unit already wrote (`decode::header`), parses every payload once, and
  then either walks every row (a projection) or gathers named rows (a routed
  expert's 1408 gate/up rows or 2816 down rows). Row walks split across
  `available_parallelism()` scoped threads, each with its own payload handles.
- `crates/ds4-core/src/gemma4_a4b/forward.rs` — the graph: embedding with the
  bf16-downcast scale, the two attention geometries, the RoPE split, per-head
  q/k norms, the value norm without a weight, the `k_eq_v` full-attention
  value, the sliding-window ring, the dense MLP, the router and the top-8
  experts, the four post-norms, the layer scalar, the final norm and the
  softcapped logits.
- `crates/ds4-core/src/gemma4_a4b/tokenizer.rs` — the checkpoint's BPE from
  `tokenizer.json` (byte fallback, fused unk), the official Gemma-4 chat
  template, and an incremental decoder that does not split a UTF-8 character
  across stream deltas.
- `crates/ds4-server/src/bin/gemma4-server-rs.rs` — `GET /v1/models`,
  `POST /v1/chat/completions` (buffered and SSE), `POST /v1/completions`
  (buffered and SSE), `GET /health`.
- `crates/ds4-core/src/vq.rs` — `codebook_f32` and `dot_row_codebook`, so a
  row walk converts each codeword once instead of once per use, and
  `add_row_values` for a one-row embedding lookup.

## Source of truth, read first-hand

- `/data/refs/transformers-5.14.1/transformers/models/gemma4/modeling_gemma4.py`
  — `Gemma4TextDecoderLayer.forward` (:1405-1463), `Gemma4TextAttention.forward`
  (:1235-1297), the router (:1339-1374), the experts (:1300-1337), the scaled
  embedding (:1474-1477), the model forward (:1653-1742), the proportional RoPE
  (`modeling_rope_utils.py:187-252`).
- `/data/llama.cpp/src/models/gemma4.cpp:155-441` — the same graph in GGML,
  including `f_attention_scale = 1.0` (:6) and the router's
  `rms_norm(attn_out) / sqrt(n_embd) * ffn_gate_inp_s` (:308-313).
- `/data/llama.cpp/models/templates/google-gemma-4-31B-it.jinja` — the chat
  template: `<bos>`, a `system` turn, `assistant` rendered as `model`, and a
  generation prompt of `<|turn>model\n` that closes an empty thought channel
  when thinking is off.

Three facts the graph turns on, each now code with a citation in place:

- a full-attention layer has no `v_proj`, and its value is the **raw** key
  projection before either norm (`:1258-1263`);
- the routed block reads the post-attention residual twice — the router
  normalises it without a weight and scales by `router.scale / sqrt(hidden)`,
  the experts normalise it with `pre_feedforward_layernorm_2`;
- the embedding scale is downcast to the checkpoint's dtype, so it is
  `bf16(sqrt(2816))` = 53, not 53.0659 (`:1605-1606` records the same rounding
  for the 31B sibling).

## Measured

One token at a time, f32, on 28 cores:

| | per prompt token | per generated token |
|---|---|---|
| first build | 3.0-3.6 s | 3.2-3.6 s |
| after the f32-codebook hoist | 1.2-1.4 s | 1.3 s |

The full chat request above took 83 s (25 prompt + 2 generated tokens) on the
first build. A 25-token prompt costs about 32 s after the hoist. The lm_head
alone is 262,144 x 2816 VQ values per step and is the single largest term.

This is a correctness path, not a performance claim: nothing was profiled
beyond the per-step wall time, and the artifact is read through the page
cache (9.0 GB) on a host with 31 GB of RAM.

## What this does not do

- No batched prefill: a prompt is evaluated one token at a time, so prompt
  cost is linear in tokens with the full per-token cost each time.
- No device path. `serving_caps` still refuses this family on every native
  lane (`Support::None`), which is correct: the separate host is not the
  native engine.
- No tools, images or audio. The template's tool blocks and the vision
  sidecar are not rendered or bound.
- Thinking is off and not exposed as a request field; the template supports
  enabling it.
- No logits-level reference comparison. The checkpoint is 49 GB and this host
  has 31 GB of RAM, so a bf16 reference forward of the source model could not
  be run here. Correctness is therefore judged by the produced text (which is
  on-topic and exact for the question asked), not by a logits diff. The next
  unit that can afford a 128 GB host should capture reference logits for one
  prompt and turn this into a numeric gate.
- The forward is f32 while the reference runs bf16, so the two would not agree
  bit for bit even with a reference available; the VQ weights are themselves
  an approximation of the checkpoint (`manifest.txt` records the per-tensor
  relL2).

## What the template should carry next

Two gaps showed up again, and one is new:

1. The release still does not say which payload layout it holds, so the host
   carries the tools' `DQV3` reader. The previous unit's note stands.
2. The artifact carries no tokenizer, so the host takes `tokenizer.json` from
   the source checkpoint as a second input. A `vq-release/1` artifact that
   ships its tokenizer next to `vq-release.json` would make the host
   single-argument.
3. The release states no chat template. `tokenizer.json` does not carry one
   either (this checkpoint's `tokenizer_config.json` has `chat_template:
   null`), so the template had to be taken from the family's published jinja.
   A `chat_template` field in the release, or the template file shipped with
   it, is the missing piece for a family that a host can serve unmodified.
