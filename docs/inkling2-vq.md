# Inkling-2-Small VQ: format, execution and serving

Inkling-2-Small VQ (family `Inkling2Vq`, architecture `inkling2_vq`) is a
GVQ1 conversion of
[thinkingmachines/Inkling-2-Small-MXFP8](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8/tree/086a047e8b7ff2ac902dd52355328ddb66b16914):
[Baekpica/Inkling-2-Small-VQ-GGUF](https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF)
(`a6db8b8b`, recipe `inkling2s-hvq8-r12x48-q4k-nvfp4kv-ssdkv-embssd-v1`).
This page describes how the CUDA backend runs it on one DGX Spark (GB10), how
its native 1,048,576-token context lives across device memory and the SSD,
how the release's DSpark drafter and vision tower run, and what was measured. The arithmetic
contract is `tools/vq_calib/inkling2_ref.py` (transformers `109a7d19`
`modeling_inkling2.py`, eager attention); the drafter follows TokenSpeed
`python/tokenspeed/runtime/models/inkling2_mtp.py` at `c485a46a`.

## Model and format

```
embed ─▶ [ attn-res ─▶ norm(s) ─▶ q k v r gate sink ─▶ conv ─▶ clip ─▶ attention (+rel bias, sink, gate)
           ─▶ o ─▶ conv ─▶ +h | restart ─▶ attn-res ─▶ norm ─▶ router ─▶ 8 of 384 experts ─▶ conv ─▶ +h ] x48
       ─▶ attn-res ─▶ norm ─▶ head ─▶ 30 tanh(z / 30)
```

| Part | Shape | Storage |
|---|---|---|
| Attention | 48 layers, 48 query / 8 KV heads x 128; full layers 5, 11, ..., 47 (dense, dynamic 4-tap K/V conv, q/k/v inputs from three attention-residual queries), the rest sliding over 1,024 keys (static 4-tap conv); q and k L2 clips at 22.627, v unweighted RMS, relative bias over 1,024 offsets from a clipped rank-16 profile, a sink logit per head, sigmoid output gate | Q4_K q/k/v/r/gate/o, BF16 sinks, scales, profiles and taps |
| Attention residuals | each layer reads the inputs of layers 0, 6, ..., 42 beside its running partial through learned queries | F32 queries, BF16 norms |
| Convolutions | 4-tap causal convolutions of K/V inputs, attention and MLP outputs, `(conv + x) / sqrt(5)` | BF16 taps |
| Routed experts | 384 x top-8, FFN 2,304; sigmoid + bias for selection, weights over their sum x 8; SwiGLU `4 tanh(g/4) sigmoid(g) 25 tanh(u/25)`; F32 routed sum in ascending expert order | GVQ1 K4096 12-bit hVQ on all 48 layers |
| Router | 384 x 4,608 per layer | BF16, applied in F32 |
| Embedding | 201,024 x 4,608 (200,058 tokenizer rows) | BF16 `embedding` part, read by the host one row per token |
| Head | 201,024 x 4,608 | Q4_K |
| DSpark drafter | 5 layers (0, 1, 3, 4 sliding over 1,031 keys, 2 global), block 8, `rnn_hidden` Markov head of rank 256 | Q8_0 + BF16 `drafter` part (mapped only when drafting) |
| Vision tower | 23 ViT layers (1,024 wide, 16 heads, axial 2-D RoPE, layer scale), 2x2 merge to 4,608 | BF16 `vision` part (paged in per image) |

The release is a directory: `release.json` names 52 GGUF parts (48 expert
parts, `body-nonexpert.gguf`, `embed-tokens-bf16.gguf`, `mtp-q8_0.gguf`,
`vision-bf16.gguf`; 118,182,291,328 bytes). It carries no tokenizer or
template: `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `processor_config.json` of the source revision sit
beside the parts, each checked by SHA-256 against the pins in
`crates/ds4-core/src/inkling2_vq/release.rs`. The host checks every part's
SHA-256, binds the HF names and copies the trunk (105.98 GiB) to the device;
the embedding part stays mapped.

Input renders the official `chat_template.jinja` through the shared engine
(`tests/fixtures/chat-template/models/inkling2`, 11 Python renders reproduced
byte for byte). The template opens no generation block: the model writes
`<|message_model|>` itself.

## K/V tiers

Sliding layers keep a BF16 ring of the window behind one prefill chunk (chunk
+ 1,023 rows). Full layers keep every position as NVFP4 pages: per 16 values
an E4M3 scale `max(amax / 6, 2^-9)` and E2M1 codes (the MiniMax VQ encoder),
one 147,456-byte block per 128 positions, 9,216 bytes per position over the
eight full layers.

- **Resident up to 262,144 positions** (2.25 GiB of full-layer blocks;
  `DS4_INKLING2_KV_RESIDENT_CTX` moves the threshold, 0 sends every block past
  the first forward's window to the file).
- **Past it, blocks live in an unlinked `O_DIRECT` file** in
  `DS4_KV_OFFLOAD_DIR` (default `~/.cache/ds4/kv-offload`), `[layer][block]`,
  reserved with `fallocate` at graph creation (6.75 GiB at 1M). Pinned memory
  holds each full layer's window (the blocks a forward writes: chunk / 128 + 2)
  and two 256-block read buffers (37.7 MB each). A forward first slides the
  window to the block of its first row, appending the complete blocks it
  leaves to the file.
- **Every forward past the threshold reads the file region**: full layers
  attend densely. An I/O thread reads the forward's 256-block segments in
  layer order, each into the buffer the attention two reads back released;
  the attention runs per key segment (resident blocks, file segments, then
  the window) and carries `(m, l, acc)` across them. GB10's NVMe gives 4.3 GB/s
  for 4 MB `O_DIRECT` reads, 5.6 at 18 MB, 5.9 at 36 MB; two or three
  concurrent streams reach 5.4-5.7 in all, so reads stay single and large.

The tiers run the same kernels on the same bytes. A 3.9K-token chat at
512-row chunks with 24 greedy tokens: resident 1,024 and 0 produce identical
results, prefill logits bit-identical to the resident run, decode logits
within the noise of a configuration change (mean top-20 KL 5.6e-4, 24/24
tokens; 512 vs 1,024-row chunks: 2.8e-3, tokens split at step 10). The
decode difference is the split merge's F32 order when the running maximum
moves to a later segment.

## DSpark drafting

```
target forward ─▶ taps: the partial entering layers 5, 11, ..., 41 and the normalized output
     ─▶ per draft layer: dflare_norm(sum_j softmax(fuse_weight)_lj tap_j) ─▶ k_ctx, v_ctx ─▶ k_norm_ctx + RoPE ─▶ ring
trial  [anchor embedding, mask x 7] at p.. ─▶ 5 x [ norm ─▶ q/k/v ─▶ norms + RoPE (theta 1e6) ─▶ attention over
       the context and the whole block ─▶ o ─▶ add-norm ─▶ SwiGLU 18,432 ─▶ add-norm ] ─▶ head on rows 1..7
       ─▶ rnn_hidden: z = rnn_proj[state, w1[prev], hidden_k, f_rnn], GRU state, proposal k =
          argmax 30 tanh((base_k + fold(tanh z_out)) / 30)
verify the target forwards the anchor and the proposals once; commit keeps the accepted prefix
```

`fold = head . hidden_out` maps the recurrence's output into the vocabulary
once at graph creation (TokenSpeed folds the BF16 head in FP64; here the Q4_K
head's F32 sums, BF16 stored). Every target forward writes its rows' context;
the Markov stage's mixture of the last committed row is the next trial's
`f_rnn`. Rejected positions' target K/V and draft ring rows are rewritten
before any read. The convolution histories cannot be cut back by position:
a trial saves them (6.5 MB) and keeps the verify rows' convolution inputs, and
commit recomputes the histories from the kept rows. The global draft layer
keeps the last 65,536 positions (TokenSpeed keeps all of them): a named
deviation that changes proposals only.

`--mtp-draft N` (2..7) maps the drafter part and drafts N tokens per trial.
MiniMax's rate gate (`DS4_INKLING2_DSPARK_GATE=0` disables it) decodes plainly
for a cooldown while trials keep tokens slower than plain decodes.

Acceptance, 256 greedy tokens after four chat prompts (code, explanation,
Korean travel plan, arithmetic), committed tokens per trial:

| Drafts | code | explain | Korean | math | est. tok/s (mean) |
|---|---|---|---|---|---|
| 7 (reference, teacher-forced) | 3.52 | 2.90 | 2.51 | 4.62 | |
| 7 | 3.08 | 2.59 | 2.35 | 4.20 | 27.4 |
| 3 | 2.59 | 2.15 | 2.42 | 3.12 | 30.3 |
| 2 | 2.29 | 2.06 | 1.95 | 2.55 | 29.0 |

The reference applies TokenSpeed's formulas to taps dumped from the engine
over the plain greedy text; per anchor it matched the native trial's kept
count at 11 of 14 positions. Throughput estimates divide kept tokens by draft
plus verify time (draft 13-17 ms, verify 63-98 ms for 3-8 rows); plain decode
runs ~24 tok/s here. The arithmetic answer stays identical to plain greedy
decoding; the other three split at near-ties (a 5-8-row verify runs the
prefill expert path). On raw Italian prose (ds4-bench, 4K frontier) drafts
mostly miss: plain 24.42, 3 drafts 16.43 without the gate and 21.88 with it.
Drafting costs ~4% of prefill (the context writes).

The draft block's attention stages each ring tile once for all eight rows
(29.5K context: draft 23.7 -> 22.1 ms). The same change in the target's
verify attention (rows staged once over the SSD tier's buffers) ran slower,
113.9 -> 120.8 ms per 8-row verify at 29.5K (all on SSD): the rows' scores
serialize inside one CTA, and re-reading the pinned buffers was not the cost.
Verify rows (2-8) instead run one MMA tile per split and KV head: the rows'
6 query heads per KV head fill 48 query slots of a BF16 `m16n8k16` over each
key tile, partials merged as for one row. At 29.5K, all on SSD, 7 drafts:
verify 113.8 -> 102.7 ms, 21.43 -> 23.38 tok/s; a short code answer 26.74 ->
28.06 tok/s. One decode row keeps the SIMT kernel.

Past the resident threshold a verify reads the file region once for all its
rows, so drafts pay most there. A ~937K-token document with a question
(7 drafts, gate off): 127 tokens in 61 trials, 2.08 per trial, draft 22 ms,
verify 1,053 ms (1,185 with the SIMT verify attention): 1.94 tok/s against
0.86 for plain decoding at 1,040,000 tokens; MemAvailable stayed at or above
2.37 GiB with the default threshold.

## Image input

```
image ─▶ EXIF orientation ─▶ alpha over an 8-pixel 204/153 checkerboard ─▶ pad right and bottom to
40-pixel tokens with -1 ─▶ (x - 127.5) / 127.5 ─▶ [tokens][40 x 40 x 3]
  ─▶ 4 patches per token (2 x 2 of 20 x 20) ─▶ patch embedding + 32 x 32 positions resized bilinearly
     to the patch grid, CLS in front ─▶ pre LN ─▶ 23 x [ x + ls1 attn(rope q, rope k, v) ; x + ls2 MLP ]
  ─▶ 4 patches per row (4,096) ─▶ proj_in, GELU, proj_out ─▶ the placeholder rows (unnormalized)
```

The template writes one `<|unused_200054|>` per image; the host expands it to
one placeholder per 40 x 40 token (`ceil(h / 40) x ceil(w / 40)`, no resize:
Inkling2ImageProcessor at `109a7d19`). JPEG decodes through libjpeg-turbo, as
PIL does. The processor has no maximum; one image is refused above 4,096
tokens (6.5 MP), which bounds the tower's activations. Each image is one
bidirectional attention segment; the tower's weights stream from the mapped
part per layer and its page cache is dropped afterwards
(`ds4_inkling2_vision.inc`, V4.1's linears, attention and GELU, MiniMax's
LayerNorm).

Checked against transformers' tower transcribed in PyTorch (BF16, SDPA) on the
same pixels, six images from 8 to 1,647 tokens (PNG with and without alpha,
JPEG with EXIF rotation): host pixels equal the processor's bit for bit; rows
cos mean 0.9991-0.99986 (minimum 0.9876), relative RMS mean 0.017-0.041,
where the reference against itself with SDPA's math backend gives 0.9996-
0.99986 (minimum 0.9884) and 0.016-0.028. Answers name the colors of
synthetic images, a cartoon and the text of a Chinese trading-app screenshot.
Encoding takes 0.06 s for 18 tokens, 0.23 s for 510 and 1.37 s for 1,647.

A prompt resumes on the live state only under the same images (FNV-1a of the
image bytes): a same-size image with identical placeholders prefills again,
and a follow-up turn over the same image resumes at its copy. Every request
encodes its images again, also when the state is reused.

## Quality

Teacher-forced through the production host (base `c57543fb` plus this
family's changes; prompts synced, then each next token evaluated), against
the release's own references:

- **Reference forward of the release bytes** (`tools/vq_calib/goldens_inkling2.py`,
  NVFP4 K/V): `The capital of France is` and a chat question, every position
  top-1 equal (6/6, 26/26); last position KL 0.0018 after one prefill, 0.0053
  token by token; mean KL over the reference's top 32 per position 0.008 and
  0.005.
- **The release's held-out images** (8 cases, 342 answer tokens;
  `calibration/images_heldout_manifest.json`, scores in
  `quality/eval/release/image` of the release): NLL per answer token 0.7101
  (0.7013 with 512-row prefill chunks), the release's reference 0.6936, the
  original model 0.6878. The engine's two chunkings differ by 0.024 per
  token over the cases, as much as the engine and the reference (0.025), and
  agree on 337 of 342 top-1 tokens. The original weights move by 0.016 when
  their K/V switches from FP8 to NVFP4 (the release's `EVALUATION.md`).

## Performance

GB10 at 2,184-2,190 MHz, `ds4-bench`, `speed-bench/promessi_sposi.txt`,
2,048-row chunks, 32 decoded tokens per frontier (prefill tok/s / decode tok/s):

| Frontier | Resident | All on SSD (resident 0) |
|---|---|---|
| 2K-8K | 1,101-1,148 / 24.8-24.9 | |
| 16K | 1,114 / 24.07 | 1,106 / 24.04 |
| 32K | 1,023 / 23.33 | 1,015 / 18.91 |
| 64K | 902 / 22.20 | 894 / 9.63 |

At 64K on the SSD every token reads 604 MB (5.8 GB/s). 1,040,000 tokens at the
default threshold: prefill 268.2 tok/s on average (3,878 s), decode 0.86 tok/s
(7.17 GB of file blocks per token, ~6.2 GB/s), 1,372.6 GiB read in 252.7 s of
I/O; MemAvailable stayed near 5.8 GiB.

## Memory

`ds4-server --check-config` prices the trunk (113,799,088,128 bytes), the K/V
of the requested context (resident blocks, the tier's pinned buffers and
attention state), the row workspace (1.11 GiB at 2,048 rows), the image
reserve (1.26 GB: the tower at 4,096 tokens and the rows of four images) and,
with MTP, the drafter part (1.87 GB) and its runtime (0.63 GiB at 1M). A 1M
context quotes 123.6 GB without MTP and 126.2 GB with it against ~126.8 GB
available (4 GiB floor included). Measured at 1M: MemAvailable ~5.8 GiB
without the drafter, 2.7 GiB with it (minimum 2.37).

## Serving

One serial lane. Full-layer rows are positional; the convolution histories
have no positional index, so a sync copies them, with every sliding ring's
1,023 rows below, where its final chunk starts (a resent turn resumes there)
and at its end (a re-rendered continuation resumes there however long the
answer that displaced the rings was): 174 MB per copy, two copies. A resume
restores the copy, restarts the SSD tier's window at the resumed block (one
block read back from the file) and drops the draft rings' displaced rows.
`ds4 --serving-check` (3,781-token prompt, 2,048-row chunks): prefix, rewind
and continue reuse work (top-1 17/17; prefix and rewind exact); disk KV,
snapshots and banks are neither declared nor implemented.

## Not yet supported

- No disk KV, snapshots or banks.
- Images above 4,096 tokens; audio and video (the model has neither).

## Diagnostic switches

| Variable | Effect |
|---|---|
| `DS4_INKLING2_PREFILL_CHUNK` | prefill rows per chunk (default 2,048, at most 4,096) |
| `DS4_INKLING2_KV_RESIDENT_CTX` | largest context whose full-layer K/V stays resident (default 262,144; 0 = SSD tier past the first window) |
| `DS4_KV_OFFLOAD_DIR` | directory of the SSD tier's unlinked file |
| `DS4_INKLING2_KV_BF16=1` | full layers keep BF16 K/V rows (the source's own; no SSD tier) |
| `DS4_INKLING2_DSPARK_GATE=0` | draft on every step (no rate gate) |
| `DS4_INKLING2_VERIFY_MMA=0` | verify rows use the one-row SIMT attention |
