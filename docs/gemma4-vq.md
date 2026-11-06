# Gemma-4 26B A4B VQ: format, execution and serving

Gemma-4 26B A4B VQ (family `Gemma4Vq`, architecture `gemma4_vq`) is a GVQ1
conversion of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52)
and its assistant drafter (`google/gemma-4-26B-A4B-it-assistant`, revision
`6e5aaaf4`):
[Jlfn/gemma-4-26b-a4b-vq-gguf](https://huggingface.co/Jlfn/gemma-4-26b-a4b-vq-gguf)
(recipe `gemma-4-26b-a4b-family-r8`). The port is jlFN's
([Baekpica/ds4-dfm-rs#141](https://github.com/Baekpica/ds4-dfm-rs/pull/141)).
This page describes how the CUDA backend runs it on one DGX Spark (GB10) and
on a 12 GiB card, how serving fits it, and what was measured; the rounds
behind the kernels are in the
[optimization log](gemma4-vq-optimization-2026-11-05.md). The expert kernels
are GLM VQ's GVQ1 runtime ([GLM-5.3 Flash VQ](glm53-vq.md)) with the R13+8
residual stage; the E4M3 full-layer rows follow Beam and MiMo.

## Model and format

```
embed × √2816 ─▶ [ rms ─▶ GQA (sliding 1,024 or full) ─▶ +h
                   rms ─▶ dense FFN 2,112 ─▶ +h
                   rms ─▶ top-8 of 128 experts (FF 704) ─▶ +h
                   × layer_scalar ] × 30 ─▶ rms ─▶ tied head ─▶ 30·tanh(x/30)
```

| Part | Shape | Storage |
|---|---|---|
| Attention | 25 sliding layers: 16 query and 8 KV heads × 256, window 1,024, RoPE θ 1e4 on all 128 pairs. 5 full layers (5, 11, 17, 23, 29): 16 query and 2 KV heads × 512, proportional RoPE θ 1e6 on 64 of 256 pairs, `attention_k_eq_v` (V is the key projection). Per-head q/k RMSNorm; no attention sink | Q4_K q/k/v/o, BF16 norms |
| Dense FFN | 2,112 wide, tanh-GELU, in every layer beside the experts | Q4_K gate/up, Q8_0 down |
| Routed experts | 128 × top-8, FF 704, tanh-GELU; F32 router weights | GVQ1 R13+8: a 13-bit index per 8 weights into an 8,192-word E4M3 book (gate/up share one per layer, down one), plus an 8-bit GVR8 index into a 256-word residual book; 2.64 bits per weight |
| Router | 128 × 2,816 with a per-dimension and a per-expert scale | BF16 |
| Norms and scalars | seven RMSNorms per layer (eps 1e-6) and one layer scalar | BF16 |
| Embedding and head | 262,144 × 2,816, tied | BF16 `embedding` part, device-mapped: the gather and the head read one table |
| Drafter (`mtp` part) | 4 layers at hidden 1,024 (3 sliding, 1 full), FF 8,192, own 262,144 × 1,024 head, 5,632→1,024 and 1,024→2,816 projections; it reads the trunk's KV of layers 28 and 29 | Q4_K, BF16 norms |

The release is a directory: `release.json` names three GGUF parts with their
roles, SHA-256 and source revisions (trunk 8,588,461,824 bytes, embedding
1,476,395,648, mtp 621,995,136; 23,805 tensors). The trunk part carries the
tokenizer and the official chat template; `config.json` is the checkpoint's
own. The host checks the recipe, the revisions, every part's digest and every
GVQ1 header, then binds the HF tensor names. The residual stage lowers the
converter's expert weight error from 39.29% to 23.38% mean relL2 at the cost
of one byte per 8 weights.

## Execution

**Decode (one row).** Q4_K projections run the shared direct kernels and
the Q8_0 dense down the batched warp-8 kernel. The family's router
projection, a parallel top-8 and the persistent layer-book MoE kernel
(`cuda/gemma4_moe_persist.cuh`: persistent CTAs walk every routed expert of
the layer with its book in shared memory, one launch per pass instead of 48
launches per layer) keep routing on the device. Attention is split-K: each KV head's visible keys split into 64-key
spans, one CTA each, joined by a combine kernel; the full layers run it on
tensor cores (`cuda/gemma4_attn_split_mma.cuh`). The tied head is the
family's own BF16 GEMV, with the 30·tanh cap.

**Prefill (512-row chunks; `--native-chunk` up to 8,192).** The Q4_K
projections and the expert routes are grouped over the chunk, so each expert
matrix is decoded once per chunk (`cuda/gemma_vq_prefill.cuh`). Attention
runs on tensor cores from 16 rows (`cuda/gemma4_attn_mma.cuh`: one CTA per KV
head and row tile, 32-key tiles, F32 online softmax, P in BF16); the 2–8-row
verify blocks of the drafter keep the tile kernel.

**KV.** A sliding layer keeps `window + max(chunk, 7)` rows (at most the
context) written at `position % rows`, so a chunk never overwrites a slot one of its earlier
rows still reads; a full layer keeps the context. BF16 rows are 8,192 bytes
per token on a sliding layer and 4,096 on a full one: 5.29 GiB per bank at
262,144 positions and 512-row chunks. `DS4_GEMMA4_FULL_KV=fp8` stores the full layers as one
E4M3 row per token with an F16 scale per KV head (2,056 bytes; 2.80 GiB per
bank) and runs the same tensor-core kernels on F16 tiles. A serial rewind or
a bank fork at k is exact while the ring, counted from the newest row
written, still holds the 1,024 positions under k, and never below a restored
snapshot (`gemma4_ring_keeps`, `make test-gemma4-ring`); otherwise the
prompt re-prefills.

**Numerics.** Every kernel rounds where the reference does (transformers
`cc1451df`, transcribed in `ds4_gemma4_ref.h` and gated by
`tests/test_gemma4_primitives`). The grouped prefill and the tensor-core
attention change the summation order; `DS4_GEMMA4_EXACT_PREFILL=1` restores
the one-row path, byte-identical across chunks 1–22. The tensor-core
attention measures 1.4e-3 max error against the reference on synthetic
cases and relL2 4e-4 against the tile kernel on the model.

**12 GiB cards.** A discrete card has no pageable memory access, so the
engine copies the 9.37 GiB trunk and embedding to the device. Three opt-in
placements trade VRAM for speed: `DS4_GEMMA4_MOE_CACHE_MIB=N` keeps the
experts host-cold and fills an N MiB slot pool from the page cache as the
router asks (`DS4_GEMMA4_MOE_MISS=cpu` computes a decode miss on the host
instead of evicting); `DS4_GEMMA4_MOE_CPU_LAYERS=N` and `DS4_GEMMA4_CPU_MOE=1`
run the first N or all layers' experts on an x86-64 AVX2 host
(`cuda/gemma4_cpu_moe.cuh`). `--ssd-streaming` streams expert records from
the artifact through the same slot pool.

## Output protocol

The chat template is the official one the trunk part carries under
`tokenizer.chat_template` (the checkpoint's `chat_template.jinja` plus one
trailing blank line), rendered by the shared Jinja engine; Gemma has no
imperative renderer. `tests/fixtures/gemma4/server-vectors.json` holds the
reference renders it must reproduce. Reasoning is the
`<|channel>thought\n … <channel|>` channel, which the template opens only
when thinking is requested: a request thinks when it asks (any
`reasoning_effort` but `none`, or Anthropic `thinking`), and an omitted
effort does not. The channel never reaches `content` in any of the three
stream protocols. Tool calls are `<|tool_call>call:NAME{args}<tool_call|>`
with the template's argument syntax (bare keys, `<|"|>`-quoted strings);
`crates/ds4-server/src/tools/gemma4.rs` turns them into JSON arguments. A
call that does not parse is returned as text: the shared malformed-call
retry writes DeepSeek markup, so Gemma skips it. `<turn|>`, `<eos>` and
`<|tool_response>` stop a reply (`generation_config.json` `eos_token_id`
[1, 106, 50]), and requests that omit sampling take its temperature 1.0,
top-k 64 and top-p 0.95.

The tokenizer is the release's SentencePiece-style BPE (normalizer
`Replace(" " → U+2581)`, byte fallback, no pre-split): `tok/gemma4.rs` merges
as tokenizers' `merge_all` does and prints pieces as UTF-8, so a re-rendered
prompt reproduces its bytes and the disk KV text key matches.

## Serving

The serving surface is the sibling VQ families': persistent banks, partial
prefix reuse, disk KV and session snapshots, qualified by `ds4
--serving-check`, and the drafter on the serial lane, present but not
qualified ([capabilities](serving-capabilities.md)). The native context
default is `max_position_embeddings` = 262,144. Each command is the common
`ds4-server` one; only the model path and the card's limits change.

GB10, native context, the chunk the sweep below measured:

```bash
./ds4-server --cuda -m gemma-4-26b-a4b-vq-gguf --native-chunk 2048
```

The same with disk KV and the drafter (opt-in: draft 6 lost to plain decode
at 13K, see the limits):

```bash
./ds4-server --cuda -m gemma-4-26b-a4b-vq-gguf --native-chunk 2048 --kv-disk-dir /var/lib/ds4/kv --mtp gemma-4-26b-a4b-vq-gguf/gemma4-vq8-mtp-Q4_K.gguf --mtp-mode on --mtp-draft 2
```

A 12 GiB card at 64K, with E4M3 full-layer rows and the shared session
headroom off (the family keeps its own 384 MiB floor):

```bash
DS4_GEMMA4_FULL_KV=fp8 DS4_SESSION_GRAPH_HEADROOM_MB=0 ./ds4-server --cuda -m gemma-4-26b-a4b-vq-gguf -c 65536
```

**Memory.** On a 12 GiB RTX 4070 SUPER these settings booted with 10.12 GiB
in use and 11.45 GiB with the session graph (833 MiB free); 32,768 and
49,152 used 11.11 and 11.28 GiB, and 98,304 is refused by the plan. The plan
prices the weights, the slot pool, the KV and the graph workspace; the
engine then checks the session graph against the memory actually free and
the server refuses to listen, naming the terms, when it would leave less
than 384 MiB:

```
ds4: Gemma-4 session does not fit: ctx=8192 chunk=4096 need=2.61 GiB
     (KV 1.05 GiB + workspace 1596.5 MiB: activation 594.3 + linear 32.1
     + moe 970.1) + margin 384 MiB > free 2823.75 MiB (deficit 237.1 MiB)
```

**Checklist.** `ds4 --serving-check` on the R13+8 artifact, ctx 2,048,
prompt "Hello" (sm_89, 2026-11-03):

| Category | Item | Declared | Result | Detail |
|---|---|---|---|---|
| reuse | prefix / rewind | qualified | works | kept 21, top-1 17/17, exact |
| reuse | continue | qualified | works | kept 38, top-1 17/17, exact |
| disk / snapshot | session | qualified | works | kept 38, top-1 17/17, exact |
| banks | warm / fork / partial / disk | qualified | works | kept 21, tokens 16/16 |
| banks | width | qualified | works | tokens 16/16 + 16/16 |

With a drafter bound the checklist also runs the MTP gate
(`serving_check/gemma4_mtp.rs`): greedy streams equal with drafting on and
off, the state after an accepted block against a one-row replay, prefix
reuse, snapshot restore, a stop inside a block and sample-and-match at
temperature 0.8; `tests/gemma4_mtp_gate` (`make tests/gemma4_mtp_gate`)
runs it alone.

## Performance

The [2K–64K sweep](../speed-bench/gb10_spark_gemma4_1a8c5199.csv) (server
binary `1a8c5199`, R13+8 parts of the release's byte sizes, 2,048-row
chunks, 128 generated tokens per frontier) ran on a GB10 **outside the clock
policy**: busy SM clocks were 2,437–2,450 MHz, above the 300–2,200 MHz range
every other family's GB10 row is measured in, and the record carries no
power or thermal samples. Its absolute rates are therefore not comparable
with those rows; only ratios measured in one session are.

| Frontier | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 2K | 1,061.5 | 35.7 |
| 8K | 1,077.9 | 34.9 |
| 16K | 1,044.3 | 33.7 |
| 32K | 929.2 | 31.8 |
| 63K | 833.9 | 28.1 |
| mean of 31 frontiers | 957.2 | 31.75 |

Each row prefills the next ~2,048 tokens over the cached prefix. The 65,536
frontier is missing: the byte-estimated prompt landed a few tokens over the
booted context and was refused.

On the 12 GiB card (ctx 8,192–65,536, default chunk, a cold server per
arm, 32 greedy tokens) the device-resident lane decodes 51.8–54.8 tok/s and
prefills a 31,909-token prompt in 26.4 s; a 4 GiB slot pool with host misses
decodes 26.9–28.5 tok/s, and the all-CPU expert lane 17–19 on a host shared
with other jobs
([R8 and R9](gemma4-vq-optimization-2026-11-05.md#r8-expert-placement-on-a-12-gib-card)).

## Quality

The reference gate on the GB10 compared post-softcap full-vocabulary
distributions at the last position against transformers `cc1451df` (F32
reference, same ids, thinking off):

| Probe | ids | Reference argmax | Port argmax | Cosine | KL ref‖port | Top-10 |
|---|---:|---:|---:|---:|---:|---|
| A short, R13+8 | 28 | 818 | 818 | 0.9906 | 0.000548 | — |
| B long, R13+8 | 1,243 | 1509 | 1509 | 0.9901 | 0.419508 | — |
| A short, 13-bit | 28 | 818 | 818 | 0.9879 | 0.002371 | 6/10 |
| B long, 13-bit | 1,243 | 1509 | 1509 | 0.9873 | 0.003730 | 8/10 |
| C window edge, 13-bit | 1,024 | 1509 | 1509 | 0.9801 | 0.00038 | 8/10 |
| D first eviction, 13-bit | 1,025 | 1509 | 1509 | 0.9800 | 0.00038 | 8/10 |

B's R13+8 KL is one token's 0.6-nat deficit at the sampled position, not a
tail widening; the residual is the encoding's error. No perplexity,
teacher-forced NLL or long-context frontier was compared: the family has no
NLL gate yet. The tensor-core attention keeps the eight design greedy
tokens of the 256-token gate prompt (KL against the tile kernel 8.6e-5 on
BF16 rows, 1.7e-3 on E4M3 rows) and diverges on a longer prompt at a near
tie (frontier KL 0.026 at 256 tokens, 0.10 at 2,048), routing propagation
through 30 MoE layers.

## Diagnostic switches

Kernel paths the family enables by default; `=0` restores the reference
path the gates compare against:

| Variable | `=0` restores |
|---|---|
| `DS4_GEMMA4_ATTN_FAST`, `DS4_GEMMA4_ATTN_PREFILL` | the one-warp-per-head oracle attention (decode, prefill) |
| `DS4_GEMMA4_ATTN_MMA` | the tile prefill attention instead of tensor cores |
| `DS4_GEMMA4_ATTN_SPLIT`, `DS4_GEMMA4_ATTN_SPLIT_MMA` | the single-CTA decode attention; the scalar split kernel |
| `DS4_GEMMA4_MOE_PERSIST` | the per-expert decode loop |
| `DS4_GEMMA4_VQ_GEMV`, `DS4_GEMMA4_VQ_PREFILL` | BF16 restore plus cuBLAS for the expert projections |
| `DS4_GEMMA4_ROUTER_DOT`, `DS4_GEMMA4_ROUTER_PAR` | the shared router linear; the serial top-8 scan |
| `DS4_GEMMA4_HEAD_GEMV`, `DS4_GEMMA4_Q8_BATCH`, `DS4_GEMMA4_FUSE_ROPE` | the shared head linear; the shared Q8_0 dispatcher; separate norm and RoPE kernels |

Values and opt-ins:

| Variable | Effect |
|---|---|
| `DS4_GEMMA4_PREFILL_CHUNK` | prefill rows per forward, 1–8,192 (default 512; `--native-chunk` sets it) |
| `DS4_GEMMA4_EXACT_PREFILL=1` | one-row calls per route, the chunk-exactness gate |
| `DS4_GEMMA4_FULL_KV=fp8` | E4M3 full-layer rows |
| `DS4_GEMMA4_ATTN_SPLIT_SPAN` | keys per split-K CTA, 16–4,096 (default 64) |
| `DS4_GEMMA4_MOE_CACHE_MIB`, `DS4_GEMMA4_MOE_MISS=cpu` | the expert slot pool and its miss policy |
| `DS4_GEMMA4_MOE_CPU_LAYERS`, `DS4_GEMMA4_CPU_MOE=1`, `DS4_GEMMA4_CPU_MOE_THREADS` | host experts for the first N or all layers (AVX2), pool width (default 20) |
| `DS4_GEMMA4_MOE_STAGING_MIB`, `DS4_GEMMA4_MOE_STAGING_PREAD=1`, `DS4_GEMMA4_MOE_STAGING_QUEUED=0`, `DS4_GEMMA4_MOE_WARM=0` | the slot pool's pinned ring (default 256 MiB), its pread source, the blocking fill, no expert warm at open |
| `DS4_GEMMA4_MOE_FUSED=1`, `DS4_GEMMA4_Q4_DECODE=1` | unadopted fused gate/up and Q4_K decode kernels |
| `DS4_GEMMA4_VQ_GEMV_VARIANT`, `DS4_GEMMA4_VQ_GEMV_THREADS` | GEMV mapping (0 auto, 1 warp, 2 block) and width (128, 256, 512) |
| `DS4_GEMMA4_MOE_TRACE=1`, `DS4_GEMMA4_MOE_STAGING_TRACE=1` | routing histogram at exit; the mapping's read rate |
| `DS4_GEMMA4_VQ_VERIFY`, `DS4_GEMMA4_MOE_PERSIST_VERIFY`, `DS4_GEMMA4_Q4_DECODE_VERIFY`, `DS4_GEMMA4_ATTN_CMP` | compare a fast path with its reference in the running model |
| `DS4_GEMMA4_DEBUG=1`, `DS4_GEMMA4_HASH_LAYER=N`, `DS4_GEMMA4_HASH_LAYERS=all` | stage probes and row hashes (each a host readback) |

Set diagnostic switches before starting a fresh process.

## Limits and next levers

- The drafter is present, not qualified: draft 6 lost to plain decode at a
  13K prompt (rounds keep 1–2 tokens), its 2–8-row verify blocks stay on
  the tile attention kernel, and its exact-path byte gate fails because the
  exact switch does not cover every batched piece of the verify forward.
  The final build's plain decode has no matched MTP run.
- The qualified context is unset: servers booted at 262,144, but no prompt
  past 64K was served or measured.
- With `DS4_GEMMA4_MOE_CPU_LAYERS` between 1 and 29, or a slot pool, at ctx
  8,192, a 7,555-token prompt was refused as exceeding the context while
  N = 0 and N = 30 accepted it: the session context is right-sized below the
  advertised one on those lanes.
- Disk KV parks the whole cache, sliding rings included (989 MB at 65,536),
  writes it twice (a staged native payload copied into the KVC file, both
  fsynced; 1.10 s per checkpoint), and a request that restored a checkpoint
  stores nothing deeper, so a growing chat restores one turn behind.
- The tokenizer splits every GGUF control and user-defined token, where
  tokenizers splits only the release's 23 added tokens: raw text that spells
  another control token can encode differently.
- No teacher-forced NLL corpus or long-context quality gate exists for the
  family; the reference gate above covers single positions.
