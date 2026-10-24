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
tail included. q/k/v Q4_K projections of a token run as one launch; the
router and the Q8_0 indexer projections run row-stable F32 GEMVs (one warp
per output, every row in the same lane and step order), so up to eight rows
compute exactly what one row does. These GEMVs and the block scorer take the
row count as a template argument (`mm_rows`): with a runtime count their
unrolled per-row loops issued all eight predicated rows for a decode row.
At decode and verify widths every MiniMax kernel launches with programmatic
dependent launch like the Q4_K and expert kernels, so those stage their
weights while the short kernels before them run (`DS4_GLMVQ_PDL=0` turns it
off).

**Prefill (4,096-row chunks).** Q4_K projections run the q4p FP16 tensor-core
kernel; the residual kernel beside each norm and a row-wise SwiGLU-OAI write
q4p's prepared FP16 input themselves when q4p takes the consumer. Routed
experts (rows ≥ 9, after host grouping) run `cuda/minimax_mma8w.cuh`:
tokens × weights^T on E4M3 codewords (a codeword is an m16n8k32 B fragment),
four producer warps filling a two-stage ring for four consumer warps of 32
rows, one activation scale per (row, 128 k) as `glmvq_mma8_kernel` (which
it equals bit for bit; `DS4_MINIMAX_MMA_WS=0` keeps it). The indexer's two
Q8_0 projections share one Q8_1 quantization. RoPE rows are computed once
per forward in double, and the next chunk's embedding rows are read
ahead. Every chunk streams each routed expert's weights once, so the
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
once. The top 16 come from one pass over the scores (each lane keeps its
best fifteen in registers, fifteen warp rounds merge the lanes): fifteen
argmax passes took 12 µs per layer at 124K. Scoring grows with the context: the SIMT tiled scorer was 26% of a
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
    first;
  - a prefill chunk long against the history (rows >= blocks / 4: 2,048 rows
    at 1M) reads the layer's every block instead. Those reads need no
    selection, so they start as soon as the previous layer's attention has
    run (an event, not a device sync) and overlap that layer's MoE and this
    layer's projections and block scores. Selective resolves of a 4,096-row
    chunk had read ~65% of the blocks anyway, with the GPU idle: ~4 s of a
    ~14.5 s chunk at 940K;
  - selective reads (decode, verify) go out as one Linux AIO submission with
    polled completions instead of the I/O threads: GB10's cores idle in
    LPI-3 (433 µs exit latency) between layers, and each pooled read paid a
    worker's wake-up and its completion's. Eight random 72 KiB reads of a
    freshly written file took 0.94 ms on the pool, 0.24 ms through AIO (one
    read: ~0.1 ms). A 256K decode settles at ~350 misses per token (~6 per
    layer, ~13 ms of reads) and went 10.5 -> 14.7 tok/s.

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

A verify of four rows costs ~2.4 decodes (8K: ~117 ms against 49.6): the
routed experts run row by row (73 ms) and the Q4_K GEMVs at four tokens
take 1.5x one token's time. So DSpark only pays while a trial keeps more
than ~2.6 tokens, which raw prose often does not: the 2K-64K sweep decoded
10.7-26.5 tok/s per frontier against ~19 plain. The session therefore
gates drafting on measured rates (`crates/ds4-core/src/minimax_mtp.rs`):
EWMAs of each trial's kept tokens per second and of plain decodes; after
two trials it probes plain decoding once, and while drafting runs slower
it decodes plainly for 16 tokens before measuring again. The rates survive
snapshot restores. Greedy output is the same either way
(`DS4_MINIMAX_DSPARK_GATE=0` always drafts). In the 2K–64K sweep the gate
moved the mean decode from 18.65 to 19.56 tok/s (three runs each); a
frontier-perfect choice would reach ~21. A batched verify over the rows' expert union (each row's one-row
arithmetic) read fewer bytes but ran no faster: beside the 66 KiB books
shared memory holds two 6,144-wide input rows, and the rest from L1 left
it latency-bound (21.1 against 24.5 tok/s row by row).

## Performance

One GB10 (`sm_121a`, CUDA 13.4, driver 615.71.10, SM clock 2.41-2.59 GHz
under load on 2026-10-24; limits in [gb10-hardware.md](gb10-hardware.md)),
model `74e5a9fd`, `ds4-bench`, cold frontiers, greedy decode, MTP off.
`promessi_sposi` is `speed-bench/promessi_sposi.txt`; the code corpus is
6.5 MB of source files concatenated.

| Frontier | Prompt | Prefill | Decode | Build, notes |
|---|---|---:|---:|---|
| 64 | promessi_sposi | 177 tok/s (0.36 s) | 18.2 tok/s | `85086da5`, 3 runs |
| 8K | promessi_sposi | 1,125-1,132 tok/s | 20.2 tok/s | `152fa899`, 128 tokens, mean of 3 (18.3 at `d7479ef9`) |
| 64K resident | code corpus | 1,045 tok/s | 19.2 tok/s | `a0153343`, 64 tokens |
| 124K resident | code corpus | 999 tok/s | 19.0 tok/s | `152fa899`, 64 tokens (14.6 at `d1eff63e`) |
| 256K SSD tier | code corpus | 884 tok/s | 14.7 tok/s | `a0153343`, 64 tokens (8.8 at `14d39b71`; 753 prefill before streamed reads) |
| 1,048,319 SSD tier | code corpus | 580 tok/s (1,807 s) | 5.5 tok/s | `6fd141ff`, 256 tokens decoded (before the decode round below); 422 (2,485 s) before streamed reads; peak RSS 110.7 GiB |

Prefill went 874 -> 1,126 tok/s at 8K in the 2026-10-24 round:
tensor-core block scoring (the SIMT scorer was 26% of a 32K prefill and
grows with the context), 4,096-row chunks (each chunk streams every touched
expert's weights once), warp-specialized experts, one Q8_1 quantization for
the indexer pair, q4p inputs written by their producers. Past 128K the
streamed SSD reads took 256K from 753 to 892 tok/s.

Where an 8K prefill's GPU time goes (nsys, `85086da5`): routed experts
48.6%, q4p projections 24.4%, attention 6.7%, the residual kernel 6.3%,
the MoE reduce 2.4%, q/k norms 2.0%, the F32 router 1.6%, the rest under
1.5% each. Against per-class floors (E4M3 / FP16 MMA peaks, the DRAM rate)
the experts run at ~31% of their MMA floor (0.60 s per 4,096-row chunk), q4p
at ~65% of the FP16 peak, attention at ~46%, the elementwise passes at ~73%
of their DRAM floor: the prefill as a whole sits at ~40-50% of the sum of
those floors. The expert kernel is bounded by its shared-memory operand
path, not by MMA latency (one warp per scheduler saturates the E4M3 pipe
with two to four accumulator chains); its comment in
`cuda/minimax_mma8w.cuh` records the measured limits and the variants that
did not help. At long frontiers block scoring and attention grow with the
context.

**TTFT.** A 64-token prefill takes 0.36 s (`ds4-bench`); through the server
a ~60-token chat prompt took 0.59-0.61 s to its first token (`d1eff63e`).
A short prompt's prefill streams the weights of every expert its tokens
route to (~70-110 of 128 per layer for 64 tokens), ~0.3 s of DRAM reads,
and the expert kernel runs near that rate there. Grouping the routes on the
device instead of the host (one sync per MoE layer) measured no change at
64 tokens or 8K and was not kept. Boot prewarms the first-use module loads
(MMQ, cuBLAS, the decode kernels: 1.3 s once), which had added 0.2 s to a
server's first request.

**Decode round (2026-10-24 evening).** 8K 18.3 -> 20.2 tok/s, 124K 14.6
-> 19.0, 256K 8.8 -> 14.7, every step with identical tokens and logits:
the scorer, top-k, indexer and router kernels take their row count as a
template argument; the top-k is one pass; the short kernels launch with
PDL; the indexer GEMVs run as one grid; the SSD tier reads its misses
through AIO; and the host argmax lost a 0.53 ms dependency chain that
`-ffast-math` had made of it (most of the idle time between tokens). An 8K
token now takes ~49.6 ms against ~42 ms for its ~10.2 GB at ~240 GB/s:
Q4_K GEMVs ~27.9 ms (~230 GB/s), routed experts ~15 ms (~210 GB/s), the
short kernels ~6 ms. The expert kernels stream at 86-94% of GB10's
measured streaming read in `tests/test_minimax_decode` (R13 gate/up 142
us, down 77 us per layer); an L2 prefetch of each warp's next task ran
slower. Past 128K plain decode is bounded by the index keys
(72 bytes per token and layer, 4.5 GB per token at 1M) and the SSD tier's
misses (~350 per token at 256K, ~13 ms through AIO), so 20 tok/s there
takes DSpark.

**DSpark** (8K frontier, 128 greedy tokens, `d7479ef9` era): plain 17.75
tok/s; draft 2 21.57 tok/s (2.46 tokens per step, verify 99.7 ms, draft
13.8 ms); draft 3 21.57 (2.95, 121.6 ms); at `344dee7d` draft 3 ran 24.2
tok/s over 64 tokens. Greedy output equals plain decoding token for token.
Verify rows run the one-row expert kernels per row, so an expert two rows
share is read twice; a batched verify as GLM's is the next step.

## Image and video input

Chat requests carry images (`image_url`, PNG or JPEG) and videos
(`video_url`, MP4) as base64 data; up to four per request, on the serial
lane. The host prepares them as the shipped processor does
(`crates/ds4-core/src/minimax_vq/media.rs`, contract: the handoff's
`tools/minimax_vision_ref.py`):

- `smart_resize` to multiples of 28 (images 3,136-451,584 pixels, at most
  576 language rows; video frames up to 602,112), PIL's antialiased bicubic,
  CLIP mean/std, patches of 3 x 2 x 14 x 14 in the processor's order (frame
  pairs, then 2 x 2 blocks row-major); a still image is one frame doubled;
- videos are decoded by ffmpeg at 1 frame per second (bicubic to the
  resized size, at most 768 frames, an odd count repeating its last frame);
- the template's `]<]image[>[` becomes `]<]start of image[>[` + one image
  pad per 2 x 2 patches + `]<]end of image[>[`; `]<]video[>[` becomes, per
  frame pair k, the text `]<]{2k:.1f} seconds[>[` and the same span with
  video pads.

The vision part (`vision-bf16.gguf`, the source tower in BF16) is mapped at
open and never resident: a request streams its 1.7 GB through V4.1's 8 MiB
pinned window layer by layer, then drops the part's page cache
(`ds4_minimax_vision.inc`). The tower is the reference's: F32 patch
embedding, pre-LN, 32 CLIP layers (16 heads x 80, 3-D RoPE on 78 dims,
bidirectional attention per segment: an image, or at most four frame pairs
of a video), the projector per patch and the 2 x 2 merge MLP. A 504 x 728
image (468 rows) encodes in 1.8 s cold, 0.35 s with the part cached; a 6 s
clip at 364 x 644 (3 pairs, 897 rows) in 0.84 s.

The prefill overwrites each pad's embedding row with its visual's row. A
media prompt extends the session checkpoint only under the same media tag
(geometry and pixels), so identical pads over other pixels prefill again; a
pad without rows is refused, and snapshots (disk KV, partial reuse) refuse
media prompts. `tests/minimax_vision` checks the tower against the reference
rerun in BF16 and in F32: to the F32 run the engine's relative RMS is 0.0316
(mean row cosine 0.99860) against the BF16 reference's own 0.0302 (0.99883);
the host's pixels match the reference's grid, with 1,746 of 2.2M values one
or two steps apart (PIL against torchvision).

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
| `DS4_MINIMAX_KV_STREAM=0` | SSD tier: selective resolves for prefill chunks too (A/B of the streamed reads) |
| `DS4_MINIMAX_TRACE=DIR` | per-layer tensor dumps of one forward (`DS4_MINIMAX_TRACE_POS`) |
| `DS4_MINIMAX_ATTN_SIMT=1` | F32 SIMT attention at every width instead of the tensor-core kernels |
| `DS4_MINIMAX_MMA_WS=0` | prefill routed experts on `glmvq_mma8_kernel` instead of the warp-specialized kernels |
| `DS4_MINIMAX_SCORE_TILED=1` | the tiled SIMT block scorer at every width |
| `DS4_MINIMAX_VERIFY_CHECK=1` | re-run each DSpark verify row as a decode and print the logit difference |
| `DS4_MINIMAX_DSPARK_MARKOV=0` | DSpark proposals without the Markov bias |
| `DS4_MINIMAX_DSPARK_GATE=0` | draft on every step (no rate gate) |
| `DS4_MINIMAX_DSPARK_NOCTX=1` | DSpark block attends to itself only (no context window) |
| `DS4_MINIMAX_DSPARK_TAP=entry\|norm` | tap the residual entering the tap layers, or the normed output |
