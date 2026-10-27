# A.X-K2 VQ: format, execution and serving

A.X-K2 VQ (family `Axk2Vq`, architecture `axk2_vq`) is a GVQ1 conversion of
[skt/A.X-K2](https://huggingface.co/skt/A.X-K2/tree/d0805cfd1fc647b4078c0d9f232c80d34ee0c893):
[Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)
(recipe `axk2-vq8-r10x7-r9x53-wlloyd-q4k-q8idx-embssd-v1`). This page
describes how the CUDA backend runs it on one DGX Spark (GB10), how serving
fits its native 262,144-token context beside 98.04 GiB of resident weights,
and what was measured. The GVQ1 kernels are GLM VQ's ([GLM-5.3 Flash
VQ](glm53-vq.md)), extended to the release's byte-major 9/10-bit indices; the
FP8 latent rows and their flash tile are Ling-3.1's
([Ling-3.1-flash VQ](ling31-flash-vq.md)); the router kernel is Beam's.

## Model and format

```
embed ─▶ [ gnorm ─▶ MLA + DSA ─▶ +h ─▶ gnorm ─▶ shared + top-8 of 256 ─▶ +h ] × 61 ─▶ rms ─▶ head
          (layer 0: dense SwiGLU FFN after a plain RMSNorm)
```

| Part | Shape | Storage |
|---|---|---|
| Attention | 61 MLA layers, 64 heads: q LoRA 1,536, KV LoRA 512 + RoPE 64, key 192 (128 + 64), value 128. `q_b` reads `[q_a_layernorm(q_a), q_a]` and writes 192 query and 128 sigmoid output-gate values per head | Q4_K `q_a`/`q_b`/`kv_a`/`kv_b`/`o`, BF16 norms |
| DSA indexer | 64 heads × 128, top-2,048 keys per query; LayerNorm-with-bias keys, split-half RoPE on dims 0–63, F32 head weights × 64^-½, score Σ_h w_h·relu(q_h·k/√128); dense below 2,049 keys | Q8_0 `wq_b`/`wk`, BF16 `k_norm`/`weights_proj` |
| Norms | gated RMSNorm (rank-16 SiLU gate) on every input norm and the MoE post norms; layer 0's post norm plain | BF16 |
| Dense FFN | layer 0, 18,432 wide | Q4_K |
| Routed experts | layers 1–60, 256 × top-8 + one shared, FFN 2,048; 8 groups, top-4 groups (−inf masks), sigmoid + bias, × 2.5 | GVQ1 v2: K1024 10-bit books (layers 16, 20, …, 40), K512 9-bit (others); separate gate, up and down books |
| Router | 256 × 7,168 per layer | BF16, F32 bias |
| RoPE | YaRN θ 1e6, factor 2 over 131,072 (β 32/1); softmax scale 192^-½·(1 + 0.1 ln 2)² | — |
| Embedding | 163,840 × 7,168 | BF16 `embedding` part, read by the host one row per token |
| Head | 163,840 × 7,168 | Q4_K |

The release is a directory: `release.json` names 62 independent GGUF parts
(60 expert parts, `body-nonexpert.gguf`, `embed-tokens-bf16.gguf`;
107,625,262,720 bytes). The host checks the pinned recipe and source revision,
every part's SHA-256 (read with `O_DIRECT`), the GVQ1 v2 headers and the
quantized scales, then binds the HF tensor names. Engine open copies the
98.04 GiB trunk to the device and keeps the embedding part mapped: each token
reads one 14,336-byte row from the page cache over the SSD.

GVQ1 version 2 stores an index's low byte byte-major (256 bytes per 2,048
inputs) and its high bits in a plane (one bit per index for 9-bit, two for
10-bit); version 1 kernels and their SASS paths stay unchanged
(`if constexpr` on the bit width).

## Execution on GB10

**Decode (one row).** Each Q4_K projection streams through GLM VQ's direct
kernels on its own (split by height); the indexer's Q8_0 `wq_b` and `wk` read
their input row from shared memory.
Routing stays on the device: F32 router logits, then Beam's group-limited
top-8 kernel and the × 2.5 scale. The GVQ decode kernels stream the eight
selected experts with the projection's book in shared memory (gate and up
books plus the BF16 input row; the down book plus eight mid rows). Attention
is absorbed MLA: `q_abs = q_nope · W_UK` reads Q4_K `kv_b` directly,
Ling-3.1's FP8 latent flash tile walks every key (at most 2,048) or the
indexer's 2,048 selections, and `W_UV` plus the sigmoid gate close the head.
The indexer stores every key, scores every causal key on BF16 MMAs and keeps
the 2,048 best by a one-block radix select (`cuda/axk2_topk.cuh`; ties by
index). The gated norms split over 16 + 28 blocks.

**Prefill (2,048-row chunks).** Q4_K projections run the FP16 tensor-core
q4p kernel from 256 rows (MMQ below) and Q8_0 the MMQ kernel. Routed experts
run `glmvq_mma8_kernel` on the E4M3 codewords after host grouping (rows ≥ 2),
each projection with its own book. Attention runs in 256-row slices:
cuBLAS batched absorb/value GEMMs over a BF16 restore of `kv_b` made once per
layer and chunk, and the flash tile with each row's own selection. Index scores run in row
slices within 128 MiB. The next chunk's embedding rows are read ahead.
`DS4_AXK2_PREFILL_CHUNK` selects up to 4,096 rows.

**Numerics.** Kernels round where the source has a BF16 tensor (transformers
`models/axk2`, the handoff's `axk2_ref.py`): gated RMSNorm as
`bf16(w · bf16(x · rsqrt))` times `sigmoid(bf16(W_up silu(bf16(W_down y))))`,
SwiGLU's `silu(g)` and `silu(g) · u` in BF16, routed outputs summed in F32 and
rounded once, BF16 cos/sin. KV rows are FP8: a 512-byte E4M3 latent with one
power-of-two exponent per row and 64 BF16 rope values (641 bytes), and a
128-byte E4M3 index key with its exponent (129 bytes), per token and layer.
`E4M3 × 2^k` is exact in BF16, so the BF16 kernels read them unchanged.

## Output protocol

The chat template is rendered from the release's `chat_template.jinja`
(byte-identical to Python `apply_chat_template` and `tokenizers` on the
[fixture cases](../tests/fixtures/axk2/chat-cases.json)). A generation prompt
ends `<|im_start|>assistant\n<think>` with thinking on and
`<|im_start|>assistant\n</think>` with thinking off; history drops earlier
turns' reasoning. Requests think unless they turn it off (the server's common
default, while the template's own default is off): `think: false` or
`reasoning_effort: "none"` select the non-think prompt, and other effort
levels do not change it. Tool calls are Hermes JSON, `<tool_call>\n{"name": …,
"arguments": …}\n</tool_call>`, parsed by the shared Hermes parser; tool
results render as `<tool_response>` turns. `<|im_end|>` ends a reply, and
`<|endoftext|>` or `<|im_start|>` stop it too. Requests keep their own
sampler (the model's defaults are temperature 0.9, top-p 0.95), and the
malformed-call retry is off, as for EXAONE's envelope.

## Serving

The serving surface is Beam's and MiniMax's: persistent banks, partial
prefix reuse, disk KV and session snapshots, present and not yet
release-qualified ([capabilities](serving-capabilities.md)); the release has
no drafter. The native context default is `max_position_embeddings` =
262,144.

**Memory.** The device holds 98.19 GiB of trunk weights and their records (the
2.19 GiB embedding part stays mapped). One graph at 2,048 rows costs 1.62 GiB
of workspace at 4K and 1.81 GiB from 16K on (the index-score slice reaches its
128 MiB budget, and as much is reserved for the `DS4_AXK2_TOPK=0` tree's
scratch), plus 28 MiB of host embedding staging. Every layer keeps 770 bytes
per token (a 641-byte latent row and a 129-byte index key): 11.47 GiB per bank
at 262,144. Further banks share graph 0's scratch and add only their rows
(2.87 GiB at 65,536). Snapshots (`AXK1`) are position-indexed, so they load
into a graph of any chunk size.

| Command (plus `-m` and the release directory) | Plan / what ran |
|---|---|
| default (262,144 context) | one bank at 512-row chunks: after the weights load 18.56 GiB are free, and a 2,048-row graph beside the bank and the 6 GiB headroom (4 GiB floor + 2 GiB burst) would need 19.28; rows halve before the context (0.34 GiB spare) |
| `--mem-floor-gb 2` | one 262,144-token bank at 2,048-row chunks (1.29 GiB spare) |
| `-c 65536` | two banks at 2,048-row chunks (4.98 GiB spare) |

The Rust plan prices the graph before the weights load (117.35 of 117.58 GiB
at the default with `--check-config`) and halves the chunk when that quote
overflows; the native fit then halves again against the memory actually free.

**Gates** (`a4a7659b` plus the chunk fit, greedy, non-think unless noted):

| Gate | Result |
|---|---|
| 244,122-token prompt, two needles at 15% and 85% (default plan, 512-row chunks) | both codes, in order; 1,494 s (163 tok/s including the reply), 5.9–6.5 GiB free throughout |
| Three turns on a 4.2K-token system prompt | cached 4,203 / 4,223, then 4,228 / 4,245 |
| Thinking turns with `--prefix-reuse partial` (history drops the reasoning) | cached 4,181 / 4,192, 4,191 / 4,211, 4,210 / 4,229 |
| Restart with the same flags, turns 4–5 | 4,260 / 4,278 from disk, then 4,280 / 4,301 |
| `-c 65536`, two different chats at once | both complete; each reply equals its run alone |

Disk records persist when a bank is evicted (the 244K prompt: 11.47 GB) or at
shutdown. Their identity follows the serving configuration: a restart with
another floor and reuse mode missed them, the same flags restored them.

## Performance

One GB10 (`sm_121a`, CUDA 13.4, driver 615.71.10, SM clock 2,190–2,197
MHz), the [2K–64K sweep](benchmarks/axk2-vq-spark-64k-2026-10-27/README.md):
three fresh `ds4-bench` processes, 2,048-token steps over the cached prefix,
128 forced greedy tokens per frontier, `AXK1` snapshots restored between
frontiers.

| Frontier | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 2K | 581.4 | 15.59 |
| 8K | 469.6 | 15.44 |
| 32K | 361.2 | 14.88 |
| 64K | 272.1 | 14.40 |
| 2K–64K, median of run means | 370.2 | 14.91 |

Where a cold 8K prefill's GPU time goes (nsys, PDL off): routed experts on
`glmvq_mma8_kernel` 30%, the sparse flash tile 23%, q4p projections 14%, index
scores 5%, the prefill gated norm 4%, BF16 copies for cuBLAS 3.5%, the q split
2%. Past 2,048 keys each row scores every causal key, so prefill falls with
the frontier; decode falls 8%.

**Decode round.** An 8K token went from 87 to 66 ms (nsys) before the sweep,
outputs within the quality bands below: DeepSeek's indexer top-k sorted
4,096-key chunks and merged them (~0.2 ms per layer for one row); a one-block
radix select (`cuda/axk2_topk.cuh`) takes 11 µs and is exact
(`make test-axk2-topk`). The W_UK absorption decoded one Q4_K value per thread
and row (25 GB/s); whole blocks per lane run 4.5× faster. The gated norm's
two rank-16 projections ran in one block per row (42 µs); split over 16 and
28 blocks it takes 13 µs, bit-identical. q_a/kv_a and the shared expert's
gate/up stream as single matrices (the shared launches ran at 114–147 GB/s).
Prefill restores `kv_b` to BF16 once per layer and chunk instead of per
256-row slice.

## Quality

- **Reference parity.** The handoff's goldens are RTX PRO 6000 (sm_120) runs
  of its reference over 22 held-out cases (53,142 tokens). The same reference
  arithmetic rerun on this GB10 over the release bytes (`axk2_ref.py` with the
  release's embedding part) matches them at 94.3–99.5% top-1 per case. The
  engine is judged against that rerun with Beam's bands, fixed beforehand:
  NLL within 1%, top-1 at most 2 points below the rerun's own agreement with
  the goldens. `tests/axk2_quality` passes over 26,866 positions: NLL
  1.98665 against 1.98767 (−0.0010 ± 0.0012 nats per token), top-1 95.88%
  against the goldens' student (the rerun: 96.69%), 95.87% against the rerun
  and 68.4% against the FP8 teacher. The build before the decode round below
  measured +0.00014 ± 0.0012 and 95.89%.
- **Paths.** Cases up to 2,048 tokens prefill 64 rows and decode the rest; the
  five longer ones (2,938–8,192 tokens) prefill all but 256 rows, so prefill
  and decode both run the sparse selection. Per case the NLL difference stays
  within ±1%, except the finance tail (256 positions): −3.9%, and −13% before
  the decode round, the engine agreeing with the teacher there more often than
  both references (82.9% and 82.1% against 80.2%). BF16 index keys
  (`DS4_AXK2_INDEX_KEYS=bf16`) left it at −11%, so FP8 key rounding does not
  explain it. Decoded row by row instead, the finance case's 8,129 positions
  sat at −0.13% (its tail −1.9%): the prefill-built cache moves that tail,
  through the prefill's E4M3 expert and FP16 projection activations, not the
  sparse selection.

## Diagnostic switches

| Variable | Effect |
|---|---|
| `DS4_AXK2_PREFILL_CHUNK` | prefill rows per chunk (default 2,048; at most 4,096) |
| `DS4_AXK2_INDEX_KEYS=bf16` | BF16 indexer keys (256 bytes each) instead of FP8, a rounding diagnostic |
| `DS4_AXK2_TOPK=0` | DeepSeek's indexer top-k kernels (chunk sort and merge tree) instead of the radix select |
| `DS4_AXK2_GAIN_DOT=0` | decode experts round gain × codeword per weight (the restore's arithmetic) instead of one multiply after the dot |
| `DS4_AXK2_TRACE=DIR`, `DS4_AXK2_TRACE_POS` | per-stage rows of one forward |

## Limits and next levers

- Decode is plain: the release has no drafter. An 8K token takes ~66 ms
  against ~46 ms for its ~11.6 GB of weights at 250 GB/s: Q4_K GEMVs run at
  ~220 GB/s (48% of the token), the routed experts at 180–207 GB/s, the
  indexer's Q8_0 GEMVs at ~165 GB/s, and ~10 ms go to short kernels (no
  decode graph yet). Past 64K the indexer grows: every key's 129-byte row is
  scored per layer and token (2.0 GB per token at 262K) and the one-block
  top-k takes ~146 µs per layer there.
- Prefill falls with context because index scores cost 16K FLOP per key and
  row per layer (BF16 MMAs at ~37 TFLOPS); E4M3 query scores or a faster
  tile are the long-context lever. At 8K the levers by share are the routed
  experts on `glmvq_mma8_kernel` (30%, ~34% of the E4M3 peak; Beam's and
  MiniMax's warp-specialized kernels took theirs 1.2–1.6× further), the
  sparse flash tile (23%, ~35% of the BF16 peak: each row gathers its own
  2,048 keys), the q4p projections (14%, ~76% of peak) and the elementwise
  passes around attention (q split, BF16 copies for cuBLAS, the prefill gated
  norm: ~10% together).
- Partial reuse is implemented (every prefix of a bank is exact) but not
  qualified: `--prefix-reuse auto` resolves to exact. Thinking turns need
  `partial`, because the template drops earlier turns' reasoning from history.
- The release's own loss stays: 68% top-1 against the FP8 teacher on the
  held-out set (see the [model card](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)).
