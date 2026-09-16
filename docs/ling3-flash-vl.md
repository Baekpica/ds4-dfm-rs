# Ling-3.0-flash-VL on DGX Spark

Ling-3.0-flash-VL is the eleventh family in `ds4-dfm-rs` and the fourth with
image input. It was sized deliberately against the
[Qwen3.8 Flash Next MQ-Q5](qwen38-ple-fp8.md) artifact so the same machine can
serve either one with the same operator surface.

The artifact is `Baekpica/Ling-3.0-flash-VL-Mixed-Quant-GGUF`, variant
`MQ-Q5-KDA-VIT-BF16`: three language shards plus one BF16 mmproj.

| File | Bytes | SHA-256 |
|---|---:|---|
| `Ling-3.0-flash-VL-MQ-Q5-KDA-VIT-BF16-00001-of-00003.gguf` | 29,750,367,328 | `6902566540c981273666d1df30ad88814a4f47e6d4fefbb3e50a4b4c49d84010` |
| `Ling-3.0-flash-VL-MQ-Q5-KDA-VIT-BF16-00002-of-00003.gguf` | 29,984,497,152 | `5c989b2ebe7af8f1d4c6415145ff659c9717e418d5ff642d94228fef83b8018c` |
| `Ling-3.0-flash-VL-MQ-Q5-KDA-VIT-BF16-00003-of-00003.gguf` | 23,492,477,536 | `20f983e7c3fa888f9a6f4e4784029bc9bbe23b17b22467cd3ac94d42ae5c2f66` |
| `mmproj-Ling-3.0-flash-VL-BF16.gguf` | 878,174,368 | `2a8033028450df21c86ece76d526488d62f6b5065dc239787f572a2e46ede79e` |

917 language tensors, 83,220,797,312 bytes of tensor payload at 5.3512
effective bits per weight; 334 vision tensors. Nothing else is accepted: the
host checks every tensor's name, rank, dimensions and quantization type, the
three-shard split identity, and the metadata listed below.

## Architecture

The GGUF architecture is `bailingmoe3`. 42 blocks, width 2560, vocabulary
157,184, `rms_norm_eps` 1e-6, native context 131,072.

### Hybrid attention

`layer_group_size` is 6, so a block is full attention when `(bid + 1) % 6 == 0`.
Blocks 5, 11, 17, 23, 29, 35 and 41 are MLA; the other 35 are recurrent KDA.
The GGUF carries this as a per-block `attention.head_count_kv` array of 1 and 0.

**KDA blocks.** 32 heads of 128. `attn_q`, `attn_k` and `attn_v` each project
2560 → 4096 and pass through a causal depthwise conv1d of width 4 followed by
SiLU; `q` and `k` are then L2-normalized. `ssm_f_a` is the decay projection and
`ssm_g_a` the output gate, both full rank because `no_kda_lora` is set — there
is no `_b` factor as there is on GLM 5.3. The decay is
`kda_gate_lower_bound * sigmoid(ssm_a * (ssm_f_a·x + ssm_dt))` with
`kda_gate_lower_bound` = -5.0, and `ssm_a` is stored already exponentiated in
this artifact rather than as `A_log`. The gated delta rule then runs over a
128×128 recurrent tile per head, and `ssm_norm` applies a per-head RMSNorm
before the sigmoid output gate.

**MLA blocks.** 32 heads, `qk_nope_head_dim` 128 + `qk_rope_head_dim` 64 =
192, `v_head_dim` 128, `kv_lora_rank` 512. `attn_k_b` is absorbed into Q so
attention scores against the stored 512-wide latent directly, and `attn_v_b`
expands the latent output back to 128 per head. `attn_gate` is a head-wise
sigmoid applied before the output projection.

### Routing

512 routed experts of width 768, top-8, plus one always-active shared expert.
The router is sigmoid with a selection-only correction bias
(`exp_probs_b.bias`), grouped into 8 groups of 64 of which the best 4 are kept
— ranked by the sum of each group's two best biased scores. The emitted
weights are the *unbiased* sigmoid probabilities, renormalized over the
selected eight and scaled by `routed_scaling_factor` 2.5.

SwiGLU clamps apply near the top of the stack and the two schedules differ:
the routed clamp is 4 for blocks 35–41, while the shared-expert clamp is 5 for
blocks 34–39 and 7 for blocks 40–41. A clamp bounds the SiLU output above and
the up projection on both sides.

### Positions

The VL wrapper uses M-RoPE with `mrope_section` `[8, 12, 12]`: the 32 rotary
pairs split contiguously into a temporal, a height and a width section. Pairs
are adjacent — `(2j, 2j+1)` — matching `rope_interleave`, not the half-offset
NeoX layout. `rope_theta` is 6e6.

Text rows set all three axes to the same index, so text reduces exactly to 1-D
RoPE. An image span of merged grid `H × W` places `t = base`, `h = base + y`,
`w = base + x`, and the rows after it resume at `base + max(H, W)` — the span
costs the larger of its two axes, not its token count.

### Vision

The tower is a Qwen3-VL ViT: 27 blocks of 1152 with 16 heads, FFN 4304, fused
QKV, LayerNorm with bias, `gelu_pytorch_tanh`, patch 16, 2×2 spatial merge and
a 2304-entry learned position table sampled bilinearly on its 48×48 grid.
There are no deepstack taps. It is structurally the same encoder Qwen3.8 Flash
Next already runs here, so it reuses those operators; three things differ:

- the weights are BF16/F32 rather than Q8_0;
- the Conv3D patch embedding is split along the temporal axis into two
  tensors, and a still image feeds both halves the same pixels;
- `disable_merger_proj` moves the projection out of the merger, so the merger
  is a norm plus the top-level `linear_proj` pair, emitted as `mm.0` and
  `mm.2` with a GELU between them.

Preprocessing is the factor-32 smart resize with CLIP normalization
(`image_mean` `[0.48145466, 0.4578275, 0.40821073]`, `image_std`
`[0.26862954, 0.26130258, 0.27577711]`) and this processor's own pixel budget:
`min_pixels` 4096, `max_pixels` 4,194,304.

## What the runtime reuses

Only four operators are family-specific: the grouped sigmoid router, the
M-RoPE, and the BF16 MLA absorb pair. Everything else is an existing path.

| Stage | Source |
|---|---|
| KDA recurrence, chunked prefill and banked decode | Solar/GLM kernels, third variant |
| Latent MLA attention | Motif-3 `latent_attention_bf16` |
| Routed expert GEMMs, expert sum, SwiGLU clamp, head-wise gate | Step 3.7 |
| ViT blocks, merger GELU/bias | Qwen3.8 Flash Next |
| Batched prefill workspace | shared `exaone_batch_ws` |

The SwiGLU clamp needed no new kernel: Step's operator already clamps the SiLU
output above and the up projection on both sides, which is exactly this
family's form.

## State and memory

The per-layer state is not uniform, and that is the interesting property.

- Each of the 35 KDA blocks owns a fixed 128×128 recurrent tile per head plus
  three convolution rings. That is 76.6 MiB for the whole model and it does
  not grow with context.
- Each of the 7 MLA blocks owns a linear BF16 latent cache of 576 values per
  token: 8,064 bytes per token for the model.

At 65,536 tokens that is 504 MiB of latent cache against 76.6 MiB of recurrent
state. A full 131,072-token context costs about 1 GiB of KV.

A recurrent block's history *is* its state, so a rewind is not a truncation:
both hosts drop the whole checkpoint and the next sync replays the prefix.

## Serving

```sh
MODEL_DIR=/path/to/Ling-3.0-flash-VL-Mixed-Quant-GGUF/MQ-Q5-KDA-VIT-BF16

./ds4-server --cuda \
  -m "$MODEL_DIR/Ling-3.0-flash-VL-MQ-Q5-KDA-VIT-BF16-00001-of-00003.gguf" \
  --vision "$MODEL_DIR/mmproj-Ling-3.0-flash-VL-BF16.gguf" \
  --model-id Ling-3.0-flash-VL \
  -c 65536 --max-seqs 2 \
  --kv-disk-dir /var/lib/ds4/ling-kv \
  --host 127.0.0.1 --port 8000
```

The operator surface is the one in the [serving contract](serving-contract.md):

| Option | Ling |
|---|---|
| `--max-seqs` | persistent banks, no opt-in switch |
| `--kv-disk-dir` | session and per-bank payloads, `LNG3` layout |
| prefix reuse | exact fork, and partial fork from the recurrent checkpoint pool |
| image input | PNG/JPEG data URIs, user messages, at most four per request |
| MTP | none — this architecture has no NextN predictor |

`DS4_LING3VL_PREFILL_CHUNK` pins the prefill chunk (default 4096, max 4096).
`2048` restores the previous width.
`DS4_LING3VL_NO_MLA_HMMA=1` restores Motif HG MLA on prefill.
`DS4_LING3VL_NO_BF16_REUSE=1` reconverts RMSNorm rows on every BF16 GEMM.
`DS4_LING3VL_NO_BF16_VEC=1` restores cuBLAS for n=1 BF16.
`DS4_LING3VL_NO_BF16_PAIR=1` keeps two n=1 BF16 GEMVs.
`DS4_LING3VL_NO_GEMV_XREG=1` restores the streaming n=1 warp GEMV.
`DS4_MMQ_Q5_PAIR=0` restores two n=1 Q5_K routed gate/up GEMVs.
`DS4_MMQ_VEC_SANITIZE=1` keeps the decode mmvq finite-scrub pass.
`DS4_SERVER_FORK_PARTIAL=0` drops the checkpoint pool, leaving exact fork only.

### CUDA campaign (GB10)

Workload: 8192+64, `speed-bench/promessi_sposi.txt`, SM 2184–2197 MHz.
Baseline `ds4-bench`: prefill 1142 tok/s, decode 19.54 tok/s.
After the counted rounds: prefill 1889 / 1893 tok/s, decode 24.65 tok/s.

Counted `ds4-perf compare --regression` Improved:

| Round | Change | Kill | Result |
|---|---|---|---|
| P1 | Prefill MLA dots3 HMMA | `DS4_LING3VL_NO_MLA_HMMA=1` | +50.7% prefill |
| P2 | Prefill chunk 4096 | env `2048` | +7.8% prefill |
| P3 | Pack RMSNorm once for BF16 GEMM | `DS4_LING3VL_NO_BF16_REUSE=1` | +2.7% prefill, exact logits |
| D1 | n=1 BF16 warp GEMV | `DS4_LING3VL_NO_BF16_VEC=1` | +25.3% decode |

Kept, not Improved (extrema envelope missed or flat): pair GEMV,
register-cached GEMV (`NO_GEMV_XREG`), n=1 Q5_K gate/up pair,
skip decode mmvq finite-scrub. Decode after D1 is still ~70% BF16 GEMV
plus mmvq; further inner-loop and fused-mmvq attempts did not clear 1%.

Tests: `tests/test_ling3vl_mla.cu`, `tests/test_ling3vl_matmul.c`,
`tests/test_ling3vl_q5pair.c`.

Chat input runs the official Bailing V3 Jinja template that ships in the GGUF;
the legacy token builder refuses this family rather than approximating it. The
generated tool envelope is GLM's `<tool_call>` / `<arg_key>` / `<arg_value>`
XML, and thinking is `<think>` / `</think>`.

### Partial reuse

Partial reuse costs one constant snapshot rather than a per-token window. The
latent caches are append-only, so a fork copies rows `[0, pos)` straight from
the source bank; only the recurrent state has no rewind and needs the
checkpoint slab, which reserves 32 slots of 76.6 MiB and maps them on demand.

A fork needs somewhere to land. A held source bank can still be forked, but
only into a free or evictable one, so with every bank holding a live
conversation the probe is refused and the turn prefills cold. `--max-seqs` is
the reuse budget as much as the concurrency budget; this is the shared
bank-budget refusal, not a family limit.

## Scope

Qualified on one DGX Spark with CUDA, measured at 8,192 context with two
banks: cold prefill 1,461 tok/s over 1,072 tokens, decode 19.8 tok/s, a
second turn reusing 1,123 of 1,146 prompt tokens by fork, and a disk record
restoring 1,153 of 1,173 into an empty bank after a restart.

Not release gates and not implied: Metal, ROCm, CPU inference, distributed
slices, DSpark sidecars, directional steering, speculative decoding, video
input, context above what was measured, and the full 131,072-token context.
