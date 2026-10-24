# MiniMax-M3.1 VQ: format, execution and serving

MiniMax-M3.1-Flash VQ (family `MinimaxVq`, architecture `minimax_m3_vq`) is a
GVQ1 conversion of
[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e):
[Baekpica/MiniMax-M3.1-Flash-VQ-GGUF](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF)
(`74e5a9fd`, recipe `minimax-m31f-vq8-r13x57-r8x20-q4k-nvfp4idx-ssdkv-embssd-v1`).
This page describes how the CUDA backend runs it on one DGX Spark (GB10), how
its native 1,048,576-token context fits beside 101.66 GiB of resident
weights, and what was measured. The GVQ1 formats and the 8-bit residual stage
are GLM VQ's ([GLM-5.3 Flash VQ](glm53-vq.md)); MSA, the router, SwiGLU-OAI
and the output protocol are MiniMax's own.

## Model and format

```
embed ─▶ [ grms ─▶ MSA ─▶ +h ─▶ grms ─▶ dense (0–2) | shared + top-4 MoE (3–59) ─▶ +h ] × 60 ─▶ grms ─▶ head
grms(x) = x · rsqrt(mean x² + 1e-6) · (1 + w)
```

| Part | Shape | Storage |
|---|---|---|
| Attention | 60 layers, 64 query / 4 KV heads × 128; per-head Gemma q/k norms, NeoX RoPE on dims 0–63 (θ 10⁷) | Q4_K q/k/v/o, BF16 norms |
| MSA indexer | per layer 4 index-query heads × 128 and one shared 128-wide index key, own norms and RoPE; block 128, top 16 blocks per KV head with the own block forced; every visible block below 17 | Q8_0 index_q/index_k |
| Dense FFN | layers 0–2, 12,288 wide, SwiGLU-OAI | Q4_K |
| Routed experts | layers 3–59, 128 × top-4, FFN 3,072; sigmoid + correction bias for selection, unbiased weights renormalized × 2.0; one shared expert (3,072) | GVQ1 K8192 13-bit on all 57 layers; GVR8 8-bit residual on layers 16, 18–32, 35, 40, 43, 53 (R13+8) |
| Router | 128 × 6,144 per layer | F32 |
| Embedding | 200,064 × 6,144 | BF16 `embedding` part, read by the host one row per token |
| Head | 200,064 × 6,144 (rows past 200,061 masked) | Q4_K |
| DSpark drafter | 5 SWA layers, taps 3/17/31/45/59, block 7, Markov rank 256 | Q8_0 + BF16 `drafter` part (mapped only when drafting) |
| Vision tower | 32 CLIP layers + projector and 2×2 patch merge | BF16 `vision` part (never in the device inventory) |

SwiGLU-OAI is `min(g, 7) · σ(1.702 · min(g, 7)) · (clamp(u, −7, 7) + 1)` in
F32, rounded once. The routed sum is F32 in ascending expert order plus the
shared expert, rounded once.

The release is a directory: `release.json` names 61 GGUF parts (57 expert
parts, `body-nonexpert.gguf`, `embed-tokens-bf16.gguf`, `dspark-q8_0.gguf`,
the vision part; 115,526,325,376 bytes). Parts carry the recipe id written
before the residual stage existed; `release.json` names the final one and the
host accepts exactly that pair. The host checks every part's SHA-256 (read
with `O_DIRECT`, Q4_K and Q8_0 scales scanned), the residual plan against
the published layer list, then binds the HF tensor names. Engine open copies
the trunk to the device and keeps the embedding part mapped (one 12,288-byte
row per token from the page cache over the SSD).

## Execution on GB10

**Decode (one row).** Routing stays on the device: F32 router logits, then
top-4 by `sigmoid + bias` in one warp per row (ties to the lower id). The
routed experts run MiniMax's own top-4 decode kernels: the 64 KiB layer book
staged in shared memory, a persistent grid, one gain multiply after the dot;
the residual stage adds GVR8 codewords from a 2 KiB book, down's 1,024-column
tail included. q/k/v Q4_K projections of a token run as one launch; the Q8_0
indexer projections take the generic aligned Q8_0 path.

**Prefill (4,096-row chunks).** Q4_K projections run the q4p FP16 tensor-core
kernel; routed experts run GLM's `glmvq_mma8_kernel` on E4M3 codewords after
host grouping (rows ≥ 5), with the SwiGLU-OAI clamp mode. RoPE rows are
computed once per forward in double, and the next chunk's embedding rows are
read ahead. Every chunk streams each routed expert's weights once, so the
chunk size sets the expert bytes per token: 4,096 rows read half of 2,048's
(8K prefill 938 -> 1,044 tok/s) and the 1M plan still fits (1.86 GiB of
scratch); 8,192 rows measured +4% at 16K for 1.72 GiB more and are not
offered. `DS4_MINIMAX_PREFILL_CHUNK` selects up to 4,096 rows.

**MSA.** Each layer stores K and V as NVFP4 (vLLM's encoder: per 16 values
an E4M3 scale of `max(amax / 6, 2⁻⁹)`, E2M1 codes rounded to nearest even,
global scale 1), 72 bytes per 128-value vector, in pages of one (block, KV
head): 18,432 bytes, 73,728 per block and layer. The index key is one more
NVFP4 vector per token and layer. Block scores are the max over a block's
visible keys of `index_q · index_key`; the own block always wins; the top 16
per (row, KV head) are sorted ascending. Prefill rows score on BF16 tensor
cores (32 rows by up to 32 blocks per CTA, each block's keys decoded once to
BF16, exactly); up to eight rows keep a SIMT scorer that reads each block
once. Scoring grows with the context: the SIMT tiled scorer was 26% of a
32K prefill (690 tok/s), the tensor-core one 0.4% of an 8K prefill (32K:
929 tok/s). Attention reads only those blocks,
on BF16 tensor cores (each block's NVFP4 page decoded once into shared memory
as BF16, which is exact; S = QKᵀ in F32, an F32 online softmax, O += PV with
P in BF16; scale 128^-½). A prefill CTA takes eight rows and one KV head and
walks the union of their blocks. Up to eight rows (decode, DSpark verify)
split by block instead: one CTA per (KV head, selected block, row), partials
combined per head, so every row runs the one-row arithmetic.
`DS4_MINIMAX_ATTN_SIMT=1` keeps the F32 SIMT kernels.

**Numerics.** The engine follows the handoff's `tools/minimax_ref.py`, which
matches transformers 5.20.2 layer by layer: Gemma norms rounded once,
partial RoPE with float64 cos/sin, BF16 residual stream (stored as
BF16-rounded F32), F32 router logits.

## K/V tiers

MSA keeps every position on every layer: 72 bytes of index key plus 576 of
K/V per token and layer, 38,880 bytes per token in all, 37.97 GiB at 1M.
MSA's selection saves reads, not storage. The indexer needs only the index
keys to decide what attention reads, so:

- **Index keys stay resident** in every tier (4.22 GiB at 1M). Scoring and
  selection never touch the SSD.
- **K/V pages stay resident up to 131,072 tokens** (4.5 GB per bank,
  `DS4_MINIMAX_KV_RESIDENT_CTX` moves the threshold; 0 sends every context to
  the SSD tier).
- **Past it, K/V pages live in an unlinked `O_DIRECT` file** per bank,
  `[layer][block]` (36.2 GB at 1M, reserved with `fallocate` at bank
  creation), in `DS4_KV_OFFLOAD_DIR` (default `~/.cache/ds4/kv-offload`).
  A pinned pool the GPU reads in place holds what a forward touches:
  - at forward start, every layer's written blocks are pinned (a block that
    already holds tokens is read back first);
  - per layer, once the indexer has selected, the selected blocks missing
    from the pool are read into least-recently-used slots, file-contiguous
    runs in one `preadv` on eight I/O threads, dirty victims written back
    first.

  The pool holds one layer's every block plus a forward's pins (8,192 +
  1,980 slots, 0.70 GiB at 1M), so a prefill chunk's union of selections
  always fits without splitting. Until anything is evicted, the per-layer
  resolve (and its device sync) is skipped: short conversations run as in the
  resident tier.

A slot table steers the K/V store and attention kernels; the resident tier
passes none. Both tiers run the same kernels on the same bytes, so logits do
not depend on the tier (64K frontier logits, greedy tokens and cycles
byte-identical, below). Snapshots (`MMX1`: tokens, logits, index keys, whole
pages) have one layout across tiers; the SSD tier moves pages through its
file, and a cross-bank fork copies file ranges.

GB10's NVMe (`O_DIRECT`, measured; see [gb10-hardware.md](gb10-hardware.md)): 6.3 GB/s sequential, 6.2 GB/s for
72 KB random reads at eight in flight (84K IOPS, ~95 µs per wave), 512-byte
`STATX_DIOALIGN` offsets.

## DSpark drafting

The release's `drafter` part is MiniMax's DSpark head: five sliding-window
(4,096) layers of the trunk's shape with a dense 12,288 SwiGLU-OAI, a fused
context (`fc` over the residual after layers 3, 17, 31, 45 and 59, then
`hidden_norm`) and a rank-256 Markov head. Every target forward stores its
rows' taps; the context's keys (k_norm + RoPE) and values go into one BF16
ring per draft layer that holds the last 4,096 positions.

A trial embeds `[anchor, 200058 × 6]` at the anchor's position p and runs
the five layers (bidirectional inside the block, the window as context).
DSpark samples from the anchor, as llama.cpp's `sample_from_anchor` default
for DFlash-lineage exports and the DS41 port do: block row i proposes
position p + i + 1, plus the Markov bias `markov_w2 · markov_w1[token i]`,
and its argmax is token i + 1, chained on the device. (Reading rows 1..6
instead, the in-place DFlash layout, cost a third of the accepted tokens.)
The target verifies the anchor and d proposals in one forward and keeps the
longest agreeing prefix plus its own next token.

Verify widths (up to eight rows) run every row on the one-token kernels:
SIMT Q4_K (one launch per input group), F32 GEMVs for the router and the
Q8_0 indexer, routed experts row by row and block-split attention. A verify
row's logits therefore equal a decode of the same token bit for bit
(`DS4_MINIMAX_VERIFY_CHECK=1` re-runs each verified row as a decode and
reports the difference), and greedy DSpark output equals plain decoding.

DSpark runs on the serial lane (`--mtp-mode on --mtp-draft N`, N ≤ 7,
greedy only); its cost is one draft pass (about 13 ms: 1.77 GB of Q8_0
layers, the 0.69 GB head and one Markov GEMV per proposal) plus the verify
forward, whose routed experts grow with every row. Two or three proposals
are the useful range on GB10.

## Output protocol

The chat template is rendered from the original `chat_template.jinja`.
Thinking is adaptive: a reply opens with `<mm:think>…</mm:think>` or a bare
`</mm:think>`, so the think span starts in the generation; thinking off
prefills `</mm:think>` and renders effort `low` (there is no "none"). A
request without an effort renders the port's default `low`. History keeps
supplied reasoning on every assistant turn, so continuation keys the exact
prefix. Tool calls are the template's `to_xml` written back: every tag
carries the `]<]minimax[>[` namespace token,

```
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="get_weather">]<]minimax[>[<city>Seoul]<]minimax[>[</city>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>
```

dicts nest by key and lists by `<item>`. Top-level values follow the declared
schema type; nested leaves stay text unless they are JSON scalars. Streams
hold the namespace token back until the next token shows whether a tool
block opens. Requests keep their own sampler (the model's defaults are
temperature 1.0, top-p 0.95).

## Diagnostic switches

| Variable | Effect |
|---|---|
| `DS4_MINIMAX_PREFILL_CHUNK` | prefill rows per chunk (default and maximum 4,096) |
| `DS4_MINIMAX_KV_RESIDENT_CTX` | largest context with resident K/V pages (default 131,072; 0 = SSD tier always) |
| `DS4_KV_OFFLOAD_DIR` | directory of the SSD tier's unlinked files |
| `DS4_MINIMAX_KV_POOL_SLOTS` | a smaller SSD-tier pool, to make tests evict and read back |
| `DS4_MINIMAX_KV_TRACE` | print the SSD tier's hits, misses, bytes and I/O time per forward |
| `DS4_MINIMAX_TRACE=DIR` | per-layer tensor dumps of one forward (`DS4_MINIMAX_TRACE_POS`) |
| `DS4_MINIMAX_ATTN_SIMT=1` | F32 SIMT attention at every width instead of the tensor-core kernels |
| `DS4_MINIMAX_SCORE_TILED=1` | the tiled SIMT block scorer at every width |
| `DS4_MINIMAX_VERIFY_CHECK=1` | re-run each DSpark verify row as a decode and print the logit difference |
| `DS4_MINIMAX_DSPARK_MARKOV=0` | DSpark proposals without the Markov bias |
| `DS4_MINIMAX_DSPARK_NOCTX=1` | DSpark block attends to itself only (no context window) |
| `DS4_MINIMAX_DSPARK_TAP=entry\|norm` | tap the residual entering the tap layers, or the normed output |
