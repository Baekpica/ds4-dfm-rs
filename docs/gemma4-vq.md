# Gemma-4 26B A4B VQ: format, execution and serving

Gemma-4 26B A4B VQ (family `Gemma4Vq`, architecture `gemma4_vq`, variant
`gemma4-a4b-vq8`) is a dense-VQ conversion of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
at revision `4d7ae498`, with the assistant drafter from
`google/gemma-4-26B-A4B-it-assistant` at revision `6e5aaaf4`. This page
describes what the release contains, how the CUDA backend runs it, what the
serving checklist measures, and what is not yet proven.

The checkpoint is not a DeepSeek-family model: it has two attention
geometries, a dense FFN beside its routed experts, tanh-GELU expert
activations, a tanh-30 vocabulary cap and a per-block layer scalar. The VQ
codebook is the generic dense one from the VQ_tools repository, not the
scalar FP4 family.

## Model and format

```
embed × 53.0 ─▶ [ rms ─▶ GQA (sliding 1024 or full) ─▶ +h
                  rms ─▶ dense FFN 2112 ─▶ +h
                  rms ─▶ router top-8 of 128 ─▶ experts (FF 704) ─▶ +h
                  × layer_scalar ] × 30 ─▶ rms ─▶ tied head ─▶ tanh-30
```

| Part | Contents | Storage |
|---|---|---|
| `trunk` | 30 layers. Sliding layers (25): 16 query and 8 KV heads at head_dim 256, window 1,024, default RoPE theta 1e4. Full layers (5, 11, 17, 23, 29): head_dim 512, 2 KV heads, proportional RoPE (0.25 of the dims, 40 of 128 rotary pairs), theta 1e6, `attention_k_eq_v` so the value projection is the key projection. Hidden 2,816; dense FFN 2,112; 128 routed experts, top-8, FF 704; RMS eps 1e-6; embedding scale √2816 | GVQ1 expert matrices (R13+8: 12+1-bit payload plus an 8-bit GVR8 residual plane and one 256-word E4M3 book per stack), Q4_K dense and attention projections, Q8_0 dense down, BF16 norms, layer scalars and router |
| `embedding` | 262,144 × 2,816 token table, tied to the head | BF16, host-read one row per token |
| `mtp` | the assistant drafter: 4 layers (3 sliding at head_dim 256, 1 full at 512), hidden 1,024, FF 8,192, its own tied 262,144 × 1,024 head, `pre_projection` 5,632→1,024 and `post_projection` 1,024→2,816, `use_ordered_embeddings` false | Q4_K projections, BF16 norms and layer scalars |

The release is a directory: `release.json` names the three GGUF parts, their
roles, their SHA-256 and their source revisions; the trunk part carries the
tokenizer and the chat template; `config.json` is the checkpoint config
(19 top-level keys) and is byte-identical to the copy the Rust side validates
against.
The R13+8 artifact is 10,686,852,608 bytes (9.95 GiB): trunk 8,588,461,824,
embedding 1,476,395,648, mtp 621,995,136, 23,805 tensors.

The residual stage is what distinguishes this artifact from the original
13-bit build: each routed-expert matrix adds one GVR8 plane and each stack one
256-entry E4M3 book, so the reconstruction is `bf16(gain · (C1 + C2))`.
Measured on the same checkpoint, the expert weight error is 23.3772 percent
mean relL2 at about 2.22 bits per weight, against 39.2868 percent for the
13-bit build at 1.6377 (the two builds' weight-space reports, recorded in the
port handoff section 4f). The mechanism, the byte account and the encoder are
in the VQ_tools repository (`docs/codebook.md` and the
`gemma-4-26b-a4b-family-r8` recipe).

## KV layout

Every layer keeps a BF16 key cache and a BF16 value cache; row bytes are
`kv_heads · head_dim · 2 · sizeof(uint16_t)`, so 8,192 bytes for a sliding
layer and 2,048 for a full one.

| Layer type | Rows | Slot |
|---|---|---|
| sliding (25) | `min(G4_WINDOW + chunk, ctx)` | `position % rows` |
| full (5) | `ctx` | `position` |

The ring is window plus one prefill chunk, not the window alone: a chunk
writes rows ahead of the rows it reads, and a ring of exactly the window would
let a later row of the same chunk overwrite a slot an earlier row still needs.
The default chunk is 512 rows (see the limits below), so at ctx 2,048 the
caches total 356,515,840 bytes (0.33 GiB).

## Execution on a 12 GiB card

The engine maps the 9.37 GiB release for device access and keeps the
embedding part on the host: each token's row is gathered from the page cache.
The prefill chunk defaults to 512 rows (`DS4_GEMMA4_PREFILL_CHUNK` overrides
it). The Q4_K projections and the expert routes run grouped over the chunk,
the Ling-3.1 / A.X-K2 prefill shape: each expert matrix is restored once per
chunk instead of once per route. The grouped Q4 FMA order is not bit-identical
to the one-row path, the same trade the sibling families make;
`DS4_GEMMA4_EXACT_PREFILL=1` restores the one-row per-route path for the
exactness gate, which is byte-identical at chunks 1..22.

SSD streaming is optional and off by default: `--ssd-streaming` moves the
routed experts through a bounded device slot cache (4 GiB unless
`--ssd-streaming-cache-experts N|GB` overrides it) and preads each VQ1 record
from the artifact. Books, norms, attention, the dense FFN and the token table
stay resident; `--ssd-streaming-cold` advises eviction of consumed expert
pages.

The stream map has two arms, chosen by the card, and neither changes the
other (`ds4_gemma4_map.inc`). On coherent CUDA host page tables (GB10) the
call is the shared GLM53 contract unchanged: the canonical mmap stays
unregistered and the resident spans are read through ATS. A discrete card has
no pageable memory access, so the same call installs the same stream state
without the coherence facts and makes the resident 2.35 GiB device-resident
instead, through the engine's own unit materializer (one O_DIRECT copy per
span, published as a device range); the experts stay host-cold and reach the
device only through the slot pool's pread. On a discrete card the bind-time
header gather also needs each cold record's first 128 bytes on the device, so
one 128-byte range per record is published for the bind and the truthful
record is restored before the first routed read, which re-validates it from
the pread staging.

Measured on the 4070 SUPER, ctx 1024, 32 greedy tokens, `--mtp --mtp-draft 6`:
resident 65.0-74.8 prefill / 3.74-4.34 gen t/s and 1.54-1.82 s first token;
streamed 51.5-52.5 / 11.1-13.1 and 0.51-0.63 s; with
`DS4_WEIGHT_RESIDENCY_MTP=eager` the 4 GiB pool reaches 13.28 gen t/s and
0.51 s first token. Prefill pays the cold reads (4.92 GiB read, 99% slot-cache
hit rate over the run); decode gains because the routed experts are resident
in VRAM instead of read over PCIe. The frontier logits are bit-identical
between the two. A pool sized to all 3,840 experts (7.01 GiB) does not fit
the 4070's graph budget at ctx 1024; the 4 GiB pool is the operating point.

## The verify head

The drafter's verify block runs the target's own head over up to seven rows.
The family's own kernel (`cuda/gemma4_head.cuh`,
`g4_head_gemv_rows_kernel`) reads the 1.48 GB tied BF16 table once and
accumulates every input row in the same eight-FMA sequence and XOR reduction
as the one-row kernel, so each row is bit-identical to the per-row call it
replaces; `tests/gemma4_head_rows` checks every output byte for 1 through 8
rows (4,096 of 4,096 exact on both sm_89 and sm_121). The ids come from
`ds4_gpu_argmax_rows_tensor` on the device (n int32), and the commit reads
only the kept frontier row instead of the whole 7.34 MB block. Measured on
the GB10, ctx 1024, draft 6, device-resident: 18.38 -> 20.21 gen tok/s
(+10.0%); plain decode is unchanged (6.76 -> 6.76). The exact-path byte gate
still fails, identically with and without this kernel (the exact switch does
not cover every batched piece of the verify forward), so it is a known
limitation, not a property of the head.

The forward is the reference's order, layer by layer, with every rounding it
makes: seven RMS norms per block, q/k norm and RoPE per head, the routed
experts with float32 router weights and one bf16 rounding per expert, the
dense FFN with tanh-GELU, the block layer scalar and the tanh-30 vocabulary
cap. The reference semantics, with file and line citations, are in
[gemma4-reference-semantics.md](gemma4-reference-semantics.md).

## Long-context decode and the drafter

The drafter runs four layers, one row per step, six steps per round. Each
step's attention used the oracle kernel: one warp per query head, every warp
reading its own K/V row from global memory. The full drafter layer has 16
query heads over the trunk's two KV heads, so eight warps read the same key
row eight times, six times per round. `ds4_gpu_g4_attn_fast` already had the
same arithmetic (one CTA per KV head, one shared 16-key tile) but refused
`keys != 0`; the drafter passes `keys = start` because the trunk holds only
the rows before the query. The guard now allows `keys` with the oracle's
bound, and `gemma4_mtp_step` takes the fast kernel when
`DS4_GEMMA4_ATTN_FAST=1`. Both kernels walk the same keys in the same order
with the same per-lane dot, warp reduction and float32 softmax, so the fast
arm is the oracle's bit pattern: the 1,081-token gate (MTP draft 6, 24 greedy
tokens, all eight fast options on) is byte-identical (115 bytes, `cmp`
clean).

Measured on the Spark, 13,447-token prompt, `ds4-bench`, 17 greedy tokens:

| arm | prefill tok/s | gen tok/s | TTFT s |
|---|---:|---:|---:|
| no MTP, four fast options | 312.0 | 4.55 | 0.23 |
| no MTP, eight fast options | 315.5 | 4.83 | 0.21 |
| MTP draft 6, four fast options, before the guard fix | 314.1 | 2.36 | 0.76 |
| MTP draft 6, four fast options, after | 312.1 | 3.69 | 0.47 |
| MTP draft 6, eight fast options, after | 316.2 | 3.84 | 0.45 |
| MTP draft 2, eight fast options, after | 314.7 | 4.83 | 0.28 |

The guard fix is worth 56 percent of the MTP decode rate at this length
(2.36 to 3.69 with the same four options) and the Phase-3 four add another
4 percent. The drafter is still a net loss at 13.4K: the speculative log
keeps 1-2 tokens per round, so six draft steps cost more than they save;
draft 2 matches plain decode. At 2,048 tokens the same prompt keeps
7/1/1/1/2/3/1/4/3 and reaches 15.06 gen tok/s. The remaining per-round cost
is the seven-row verify and the drafter's own per-step weights.

An nsys kernel summary of the 13,447-token prefill (chunk 2,048, eight fast
options) is the prefill picture: the attention tile kernel is 72.2 percent
of GPU time (53.1 percent the full-layer geometry, 19.1 percent the sliding
one), the two direct VQ expert GEMMs 14.7 percent, the batched dense Q8 down
7.0 percent, and everything else under 4 percent. The full-layer tile runs
656 ms per chunk on average at this length, at about 171 G MAC/s; the
sliding tile 47 ms. The attention kernel, not the expert GEMM, is the next
prefill lever.

## Tensor-core prefill attention

`cuda/gemma4_attn_mma.cuh` is the Phase 8 kernel: one template per geometry,
`<256, 2>` sliding and `<512, 8>` full, one CTA per (KV head, row tile), 8
warps, 32-key tiles. The sliding CTA covers 64 tokens (8 tokens x 2 query
heads per warp, full D = 256, 32 KiB shared). The full CTA covers 8 tokens
(2 tokens x 8 query heads per warp, two warps per M group split D = 512 into
two 256-dim halves, 32 KiB K/V plus 16 KiB score exchange). Q stays in
registers, the softmax is f32 and online once per 32-key tile, P rounds to
bf16 for the P*V mma, and the key loop starts at `(begin_all / 32) * 32` so
a token sees the same tile sequence at every chunk size. The dispatch takes
rows >= 16 and keys == 0 only; the MTP verify widths (2..8) stay on the
bit-exact tile kernel. `DS4_GEMMA4_ATTN_MMA=1` enables it; it is off by
default because it is a parity path, not a bit-identical one.

Synthetic parity (`tests/test_gemma4_primitives`, both geometries, rows
16/17/64, first 0/1000/1025/4096/8192, ring, window, shared-prefix and
poisoned-slot cases): max absolute error 1.4e-3 against `g4_ref_attn`,
inside the Phase 8 tolerance (2.5e-3). A one-layer probe on the real model
(`DS4_GEMMA4_ATTN_CMP=1`, layer 0 sliding and layer 5 full, ctx 256) measures
the attention output against the tile kernel at relL2 4.05e-4 and 3.94e-4.

The model gate on the pinned 256-token prompt is exact for the design's
eight greedy tokens `[6113, 1698, 236761, 28333, 506, 877, 6910, 3136]`,
with KL(tile || mma) 8.6e-5 and relL2 0.101 (the KL gate is 5e-3; the relL2
gate of 6e-2 is 1.7x over, the logits' tanh-30 scale is harsh on that
metric). On the longer repo-docs prompt the greedy text diverges after the
first sentence (a near-tie word choice), and the frontier KL is 0.026 at
256 tokens and 0.10 at 2,048. That is routing propagation, not a kernel
error: the per-layer attention is four ten-thousandths away from the tile,
and the MoE's top-8 is discontinuous. The path is therefore explicit,
measured and off by default; the tile kernel remains the exact reference.

Measured on the Spark, ds4-bench, eight fast options, 17 greedy tokens:

| frontier | tile prefill tok/s | MMA prefill tok/s | ratio | decode |
|---|---:|---:|---:|---:|
| 2,048 | 576.1 | 977.1 | 1.70x | 12.6 (both) |
| 13,447 | 309.5 | 1,061.4 | 3.43x | 4.82 (both) |

The 13,447-token prefill falls from 43.5 s to 12.7 s of GPU work, and the
9.66K open-grok prompt from about 34 s to about 10 s. Decode is unchanged
because rows == 1 takes the decode kernel.

## Lanes

The family serves two lanes from the same binary.

The serial session owns one graph and one checkpoint. A prompt resumes at its
common prefix; rows the new prompt rewrites land on their own slots, so
prefix, rewind and continue are exact. `ds4_session_save_payload` writes the
DSV4 v3 payload with the family tag `GVK4`: 13 u32 header words (magic,
version, ctx, chunk, layers, tag, reserved, token count, window, sliding rows,
full rows, vocabulary, reserved), the token ids, the logits, then every
layer's key and value cache whole. A restore validates every field and leaves
the graph failed and at position 0 until every row and the logits are in, so a
partial load is never a usable prefix. Snapshots use the same payload.

The banked lane (persistent) gives each bank its own KV caches of both
layouts; graph 0's GVQ context, expert scratch and working rows are shared,
because the scheduler serializes bank forwards under the engine generation
lock. Every cache row is position-indexed, so copying the first
`min(pos, rows)` rows copies exactly the slots the positions below `pos`
occupy: any prefix of a bank is exact, which makes the fork and partial reuse
copies (or an in-place rewind) and removes the need for checkpoint slots.
`ds4_cont_bank_save_payload` serializes one bank through the same DSV4 writer.
At ctx 2,048 on the 12 GiB card the fit chose two banks with 4.23 GiB spare
(headroom: 4 GiB floor plus 2 GiB boot burst).

## Serving checklist

`ds4 --serving-check` on the R13+8 artifact at ctx 2,048 with prompt
"Hello" (sm_89, 2026-11-03):

```
  category  item      declared   result   detail
  reuse     prefix    qualified  works    kept 21, top-1 17/17, exact
  reuse     rewind    qualified  works    kept 21, top-1 17/17, exact
  reuse     continue  qualified  works    kept 38, top-1 17/17, exact
  disk      session   qualified  works    kept 38, top-1 17/17, exact
  snapshot  session   qualified  works    kept 38, top-1 17/17, exact
  banks     warm      qualified  works    kept 21, tokens 16/16
  banks     fork      qualified  works    kept 21, tokens 16/16
  banks     partial   qualified  works    kept 21, tokens 16/16
  banks     disk      qualified  works    kept 21, tokens 16/16
  banks     width     qualified  works    tokens 16/16 + 16/16
result: pass (works 10, FAIL 0, WRONG 0, undeclared 0)
```

The check is a structural and numerical comparison against the live state it
saved; it is not a quality claim.

## Quality evidence

The reference gate on the GB10 compared post-softcap full-vocabulary
distributions at the last position against transformers at commit
`cc1451df` (fp32 reference, same ids, thinking off):

| Probe | ids | reference argmax | port argmax | cosine | KL ref‖port | top-10 |
|---|---|---|---|---|---|---|
| A short, R13+8 | 28 | 818 | 818 | 0.9906 | 0.000548 | — |
| B long, R13+8 | 1,243 | 1509 | 1509 | 0.9901 | 0.419508 | — |
| A short, 13-bit | 28 | 818 | 818 | 0.9879 | 0.002371 | 6/10 |
| B long, 13-bit | 1,243 | 1509 | 1509 | 0.9873 | 0.003730 | 8/10 |
| C window edge, 13-bit | 1,024 | 1509 | 1509 | 0.9801 | 0.00038 | 8/10 |
| D first eviction, 13-bit | 1,025 | 1509 | 1509 | 0.9800 | 0.00038 | 8/10 |

B's R13+8 KL is one token's 0.6-nat deficit at the sampled position, not a
tail widening; the distributions are close, not equal, and the residual is
the encoding's error. No greedy text, perplexity or benchmark was compared
here. The reference gate also found the port's rotary pair counts (64 on
sliding layers, 128 on full ones, where the reference rotates 128 of 128 and
64 of 256); that fix is `1d86327d` and the primitive test pins it.

## Limits and open work

- The qualified context is not recorded yet: `ctx_max` is 262,144 but
  `qualified_ctx` is unset, so the serving plan reports the family as
  qualified at the tested size only.
- The default prefill chunk is 512 rows, the native bind's `min(ctx, 512)`
  (`gemma4_prefill_cap`, ds4_gemma4_bind.inc) and the shape the bench measures
  in the VQ-prefill and attention rounds. The graph groups the Q4 projections
  and the expert routes over the chunk; `DS4_GEMMA4_EXACT_PREFILL=1` restores
  the one-row-per-route path for the byte-exactness gate (chunks 1..22). A
  one-row serving default stalled the Spark lane on 2026-11-04: 295 prompt
  tokens took 13.03 s at chunk 1 against 0.447 s at chunk 512, and an 8.5K
  prompt produced no chunk inside the client's 600 s idle timeout.
- The drafter part is in the artifact and validated by the inventory; the
  draft path (`--mtp-draft`) is declared and gated by
  `tests/gemma4_mtp_gate` (greedy off/on, prefix reuse, snapshot restore, a
  stop inside an accepted block, and sample-and-match at temp 0.8). Its
  exact-path byte gate fails identically with and without the fused verify
  head, because the exact switch does not cover every batched piece of the
  verify forward.
- The live tool-tail renders the canonical Gemma tool responses inside the
  model turn that emitted the calls (render.rs, commit be901c46); a suffix
  that is not a tool result is refused rather than mis-rendered.
- An omitted `reasoning_effort` gets no thought channel: the Gemma-4
  template opens `<|channel>thought` only for an explicit thinking request
  (chat_template.jinja), so the server's Gemma default is `none`
  (`ServerConfig::no_think_default`, set from `ModelSyntax::Gemma4`). An
  explicit `none`/`low`/`medium`/`high` is still honored. The sibling VQ
  families keep their own defaults.
- `ds4-server` passes its HTTP gate on the GB10 (P9, 2026-11-04: boot,
  /v1/models, non-stream, determinism, SSE stream, the max_tokens refusal).
  On the 12 GiB 4070 the serving quote used to price the whole release
  directory, including the 622 MB MTP part a no-MTP boot never maps; the
  Gemma release span now excludes that part (the rule Ling-3.1 applies to
  its embedded MTP), and the card boots device-resident at ctx 16384 with
  the FP8 full-layer KV: 9.37 GiB copied in 2.8 s, 10.16 GiB of the 12 GiB
  used, 20.95 tok/s against 2.66 tok/s host-mapped. The banked lane is
  refused at that size (serial only, reuse=exact); the quote admits ctx
  32768 with about 0.2 GiB of margin. `run-gemma4.sh` starts it (serve
  start/stop/status/logs, chat, plan, bench), and the open-grok model entry
  is `gemma4-vq-4070` on port 8099. `--ssd-streaming` still takes the
  shared GLM-5.3 branch ("requires one GLM-5.3 GGUF shard") rather than the
  Gemma discrete map; that remains open.
- The serial session has no ring-resume guard yet: a divergent continuation
  that reaches back past the rows the ring still holds can read overwritten
  slots. The banked lane is exact and the checklist does not exercise it.

## Reproduce

```
# build (this host: CUDA 13.3, sm_89)
make cuda CUDA_ARCH=sm_89

# the checklist on the R13+8 artifact
DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock \
  ./ds4 -m /data/vq-artifacts/gguf-family-it-r8 --cuda -c 2048 \
  --serving-check -p "Hello"
```

The artifact directory and the reference checkpoint are not committed; the
paths above are this machine's. `DS4_LOCK_FILE` must point away from
`/tmp/ds4.lock` when another ds4 server holds it.
