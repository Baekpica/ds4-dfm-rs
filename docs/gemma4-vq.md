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
layer and 2,048 for a full one. With `DS4_GEMMA4_FULL_KV=fp8` the five full
layers instead keep one combined E4M3 row per token (K, V, then one f16
scale per KV head for each: 2,056 bytes against 4,096), and that format
selects the tensor-core attention kernels described under the full-layer KV
lane.

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
cap. The reference formulas the gates compile against are pinned in
`ds4_gemma4_ref.h`, and the synthetic cases live in
`tests/test_gemma4_primitives.cu`.

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
bit-exact tile kernel. `DS4_GEMMA4_ATTN_MMA=1` enables it; the launcher turns
it on by default (`DS4_GEMMA4_MMA=0` keeps the tile kernel). It is a parity
path, not a bit-identical one, and the numbers below bound the trade.

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
and the MoE's top-8 is discontinuous. The path is explicit, measured and
enabled by the launcher; the tile kernel remains the exact reference
(`DS4_GEMMA4_MMA=0`).

Measured on the Spark, ds4-bench, eight fast options, 17 greedy tokens:

| frontier | tile prefill tok/s | MMA prefill tok/s | ratio | decode |
|---|---:|---:|---:|---:|
| 2,048 | 576.1 | 977.1 | 1.70x | 12.6 (both) |
| 13,447 | 309.5 | 1,061.4 | 3.43x | 4.82 (both) |

The 13,447-token prefill falls from 43.5 s to 12.7 s of GPU work, and the
9.66K open-grok prompt from about 34 s to about 10 s. Decode is unchanged
because rows == 1 takes the decode kernel.

Server end-to-end with the MMA prefill and the split-K decode (no MTP,
bf16 KV, ctx 262144), old prefill/decode in parentheses:

| prompt | prefill tok/s | decode tok/s | TTFT |
|---|---:|---:|---:|
| 1,696 | 911.8 | 26.23 | 1.86 s |
| 6,459 | 944.3 | 19.28 | 6.84 s |
| 11,668 | 856.7 | 19.05 | 13.62 s |
| 13,557 | 837.4 (287.6) | 17.98 (4.55) | 16.19 s (47.17) |
| 20,186 | 831.7 (232.7) | 17.20 (3.31) | 24.27 s (86.57) |

## The full-layer KV lane

The five full layers keep their rows in one of two formats
(`DS4_GEMMA4_FULL_KV`, typed knob, compiled default bf16). The format decides
which attention kernel runs, not just how many bytes the row holds:

- bf16 rows take the tensor-core kernels: `g4_attn_mma_full_kernel<false>`
  (cuda/gemma4_attn_mma.cuh) for prefill and
  `g4_attn_split_mma_kernel<false>` (cuda/gemma4_attn_split_mma.cuh) for a
  decode row.
- E4M3 rows take the same two kernels with `FP8 = true`. The tile is decoded
  from E4M3 to f16 on the way into shared memory, the mma runs the f16 form,
  and the mask, online softmax, half exchange, split division, partials and
  combine are the bf16 path's own code.

Before that port the E4M3 lane had only `g4_attn_fp8_kernel`
(cuda/gemma4_attn_fp8.cuh:36): one warp per (row, head), a scalar loop over
every visible key that decoded each K and V element, used for prefill and
decode alike. nsys on the Spark (`--trace=cuda`, ds4-bench, chunk 2048, one
cold frontier, gen 1) put that kernel at:

| frontier | `g4_attn_fp8_kernel` | share of GPU kernel time | all other families |
|---|---:|---:|---:|
| 2,048 | 0.72 s | 28.2 % | 1.83 s |
| 32,768 | 277.1 s | 90.5 % | 29.1 s |
| 65,536 | 1,113.7 s | 95.0 % | 58.6 s |

Every other family scales linearly in the token count (1.83 s, 29.1 s,
58.6 s for 16x and 32x the rows); only the full-layer attention grew with the
context, because each chunk of 2,048 rows attends over everything before it.

### The E4M3 tensor-core port

`g4_fp8_decode8` (cuda/gemma4_attn_mma.cuh) is the whole new mechanism: eight
E4M3 codes per uint2, the hardware `e4m3x2 -> f16x2` conversion, the product
with the row's f16 scale in fp32, and rn rounding to f16 — the decode stage of
the Solar warp-specialized kernel (cuda/mmq/ds4_fattn.cu:1557 `fp8_ws_scale4`);
the tile write and the two-stage buffer are Solar's
`fp8_ws_decode` (1557-1594's sibling at 1568), specialised here to the
shared-memory tile the Gemma kernel already stages. The scalar kernel's decode
is the same conversion with the product kept in fp32, so the tensor-core arm
differs from it by one f16 rounding of K and V.

Everything else is the bf16 kernel's arithmetic: the same swizzle, ldmatrix,
32-key causal mask, f32 online softmax, half-order score exchange and
`m16n8k16` mma, on f16 fragments instead of bf16. The tile's f16 slot is
exactly one 16-byte shared chunk, which is why the ldmatrix addressing and the
swizzle are unchanged between the two arms; only the staging loop differs
(`cp.async` of 16 bf16 bytes versus eight E4M3 codes decoded into the same
slot). The dispatch reuses the existing knobs: `DS4_GEMMA4_ATTN_MMA` for the
prefill width and `DS4_GEMMA4_ATTN_SPLIT` + `DS4_GEMMA4_ATTN_SPLIT_MMA` for
the decode, both already typed, with the scalar kernel as the fallback.

Synthetic parity (`tests/test_gemma4_primitives`, full geometry, rows
16/17/64, first 0/1000/1025/4096/8192, ring, shared-prefix, decode split
spans 512/128/64 and the tile edges 63/64): max absolute error 1.4e-4
prefill (sm_89) and 3.2e-6 decode against `g4_ref_attn`, inside the 2.5e-3
gate; the E4M3 arm is closer to its scalar reference than the bf16 arm is to
the tile, because f16 keeps 11 significand bits against bf16's 8.

A one-layer probe on the real model (`DS4_GEMMA4_ATTN_CMP=1`, ctx 256) puts
layer 5 (full, E4M3) at relL2 1.82e-4 against the scalar FP8 kernel, max
2.9e-3. Layer 0, the bf16 sliding kernel through the same probe, is 4.03e-4 —
unchanged by this unit.

The model gate, ctx 256, the same prompt and method on both arms: the eight
design greedy tokens `[6113, 1698, 236761, 28333, 506, 877, 6910, 3136]`
match exactly; KL(scalar || tensor-core) 1.7e-3 (gate 5e-3) and logits relL2
0.32. For calibration the bf16 lane on the same prompt and method is KL
9.8e-5 and relL2 0.159, so the E4M3 arm is twice as far in relL2 and 17x in
KL while staying inside the KL gate; the per-layer probe above shows that
this is propagation through 30 layers of MoE routing, not kernel drift.

Measured on the Spark, ds4-bench, cold frontier, prefill chunk 2048 (one
machine, one session, scalar and tensor-core arms back to back):

| frontier | scalar prefill tok/s | tensor-core prefill tok/s | ratio |
|---|---:|---:|---:|
| 2,048 | 489.6 | 770.1 | 1.57x |
| 32,768 | 97.9 | 746.6 | 7.63x |

The same pair at 65,536, measured at the engine's default prefill chunk in the
same session, is 47.2 against 460.3 tok/s (9.75x).

Decode at 32,768 with 16 greedy tokens: 0.50 tok/s scalar, 19.27 tok/s
tensor-core (38.5x). At 2,048 the scalar kernel was 28.2 percent of GPU time,
so the 1.57x already exceeds the 1.39x that removing it entirely would give
at that frontier (the two arms share everything else); the gain grows with the
context exactly as the scalar kernel's share did.

The same session, the same chunk, the two lanes side by side at 32,768:

| lane | prefill tok/s | decode tok/s (gen 16) |
|---|---:|---:|
| E4M3 rows, tensor cores | 746.6 | 19.27 |
| bf16 rows, tensor cores | 804.6 | 18.71 |

So the E4M3 lane now prefills at 93 percent of the bf16 lane and decodes
above it. The remaining prefill gap is the decode instruction stream: the
bf16 tile is copied with `cp.async` and costs no ALU, while every E4M3 element
costs a conversion, a multiply and a rounding before it can be staged. Both
lanes keep the same arithmetic after the tile (mask, online softmax, mma).

The bf16 lane, for reference, from the earlier session:

| frontier | FP8 prefill tok/s | bf16 prefill tok/s | ratio |
|---|---:|---:|---:|
| 2,048 | 783.2 | 1,003.0 | 1.28x |
| 32,768 | 107.4 | 1,029.1 | 9.6x |
| 65,536 | 56.1 | 970.4 | 17.3x |

with 0.52 against 31.80 tok/s decode at 32,768. Those scalar numbers are the
same kernel measured before this port, at the engine's default prefill chunk;
they are kept as the historical record of the lane, and only ratios measured
back to back in one session should be compared across the two tables.

The deployed route, end to end: the Spark launcher (bf16 lane, ctx 262144,
chunk 2048) answered a cold 32,370-token request with 31.1 s to the first
token, 1,041.9 prefill tok/s and 30.1 decode tok/s (attach-mode sweep, one
frontier, 4 greedy tokens). Those 32,768 tokens are the 305 s FP8 prefill in
the table above.

On the 12 GiB card the lane is now the whole story. With the FP8 rows,
`run-gemma4.sh` boots device-resident at ctx 65536, no SSD: 10.12 GiB resident
at boot, 11.45 GiB with the session graph, 833 MiB free, and a cold
61,932-token request answers in 55.7 s. The measured ladder on that card is
ctx 32768 at 11.11 GiB, 49152 at 11.28, 65536 at 11.45; 98304 is refused by
the serving quote before it boots, so 65536 is the ceiling for this release.
The session graph needs `DS4_SESSION_GRAPH_HEADROOM_MB=0` (and the launcher
sets it): the engine's own 1 GiB headroom refuses a 32K graph that actually
fits in the free VRAM it leaves.

The cost is memory, and the port does not change it: the full layers' cache is
967 MB at 32,768 and 1.30 GB at 65,536 in FP8, against 1.30 GB and 1.97 GB in
bf16, and 5.4 GB of full-layer cache at the 262,144-token ceiling. On the GB10
that is unremarkable. On the 12 GiB card the FP8 lane is what makes the 64K
context fit at all, and the tensor-core port is what makes that lane worth
running.

Selecting the lane is the knob alone; the sweep records the lane it measured
in its meta file, so a CSV can no longer outlive the reason for its numbers.

## The request path

A 2K-64K sweep on the Spark showed a TTFT that grew with the context but
not with the prefill: at 59,392 tokens the server prefilled 2,130 new
tokens at 852 tok/s and reported 35.0 s to the first token. The engine's
own prefill timing accounted for 2.5 s of it. A repeat request with the
prefix fully cached (one token prefilled) still cost 34.5 s, which put the
cost outside the engine: the host tokenizer merged one rank per full scan
of the symbol list, so a prompt's cost grew with the square of its length.

`crates/ds4-core/src/tok/gemma4.rs` now finds the lowest pair with a
min-heap keyed on `(rank, left, right, stamp_left, stamp_right)`. The
merge order is unchanged (one rank group per round, leftmost first, and a
pair a round creates waits for the next round), and the gate is a
byte-identical token stream against the old scan on a 60 KB slice of
`speed-bench/promessi_sposi.txt` (17,141 symbols) plus the tokenizer parity
test against the C oracles.

Measured on the GB10, one server, one route, the same 215 KB prompt:

| request | before | after |
|---|---:|---:|
| 61,872 tokens, cold | 98.20 s TTFT (62.9 s prefill) | 62.92 s TTFT (62.8 s prefill) |
| 61,871 tokens, fully cached | 34.46 s TTFT | 0.12 s TTFT |
| 60 KB tokenize, 4070 host | 7.43 s | 0.27 s |

After the change a request's TTFT is its prefill: 62,921 ms against 62,841
ms of prefill at 984.6 tok/s, and 120 ms for a request that only has to
restore a 61,871-token checkpoint.

## Disk KV parking

`--kv-disk-dir` lets the server park a session's whole cache on disk and
restore it on a later request instead of re-prefilling. The store is keyed
by the SHA-1 of the rendered text prefix and the restored token stream is
re-validated against that text, so a restore is only taken when the model,
the quantization, the context and the tokens all agree (`crates/ds4-kv`,
`generate.rs`). The launcher exposes it as `DS4_GEMMA4_KV_DIR` (empty keeps
the lane off), with `DS4_GEMMA4_KV_SPACE` (default 32G) and
`DS4_GEMMA4_KV_COLD_MAX` (default 65536).

Three defects had to be fixed before the lane could pay for itself on this
card. Each is measured, and the first is why an empty store was the only
thing the lane ever wrote:

- The family had no KV identity class (827a5af1). `routed_quant_bits`
  matched GGUF aliases and HF names only, while the release names its
  experts `model.layers.N.experts.M.gate_proj`, so it read 0 and
  `kv_identity` returned None. The failure was silent: every store was
  skipped.
- Non-ASCII piece text (`tok.rs`, `piece_text`). The printer applied GPT-2's
  byte alphabet to every vocabulary, so the Italian piece `▁però`
  (id 41843, U+00F2 in the vocabulary) printed byte 0xF2 instead of its
  UTF-8 bytes. Re-rendering a prefix then produced a byte the prompt never
  contained, so the text key no longer matched the prompt. The pinned
  tokenizer.json decoder is Replace(U+2581 -> " "), ByteFallback, Fuse with
  no ByteLevel stage, which is what `utf8_piece_text` records
  (`tok/gemma4.rs`); byte-level vocabularies keep the old arm.
- The chat anchor. With no user/assistant token ids, `chat_anchor_pos`
  returns -1 and a store fell back to trim 32 / align 2048, which cut a
  10.1K chat turn back to 8,192 tokens. `generation_anchor` cuts at the
  start of the last `<|turn>model\n` marker instead. The start is the
  `<|turn>` special token, which splits the encoder's input, so the stored
  tokens stay a prefix whatever the reply begins with; a cut after
  `model\n` instead was measured to miss on the very next turn, because the
  reply's first character merges with the newline.

A sweep with every request cold (an evicting prompt between them), same
prompts in both arms, 64 greedy tokens on the second turn. A is the first
turn of a conversation; B is the same conversation plus a short follow-up,
served after the live session was dropped. One server per arm:

| prompt | A off | A on | B off | B on | B gain |
|---|---:|---:|---:|---:|---:|
| 8,294 tokens | 6.26 s | 7.74 s | 6.65 s | 0.66 s | 10.0x |
| 16,243 | 12.54 s | 14.30 s | 12.94 s | 0.70 s | 18.5x |
| 31,909 | 26.29 s | 28.64 s | 26.79 s | 0.74 s | 36.4x |
| 45,737 | 39.78 s | 42.54 s | 40.23 s | 0.80 s | 50.5x |

The lane-on B rows restore prefix minus 446 tokens of tail; the off rows
re-prefill the whole prompt. Decode rate is 50-55 tok/s in both arms, so the
lane moves the first token only. The A delta is the 1.10 s checkpoint write
plus a 4-6 percent prefill-rate difference between the two boots; the store
cost itself is the paired measurement below.

Measured on the 4070 (ctx 65536, FP8 full-layer KV, chunk 512, store on the
root NVMe), temperature 0:

| case | lane off | lane on |
|---|---:|---:|
| new conversation, 9.7K prompt | 7.44 s TTFT | 8.55 s TTFT |
| returning chat turn, 10.1K | 7.69 s TTFT (full prefill after evict) | 0.52-0.95 s TTFT (9,700 restored; the first return also parks the evicted session, about 1.7 s) |
| identical 62K prompt, after its cold store | 57.08 s TTFT | 1.00-2.22 s TTFT (61,440 restored) |
| full-prompt restore, 9,652 tokens, trim/align 0 | 7.27 s TTFT, prefill 9,652 | 0.31 s TTFT, prefill 0 |

The cost is the write on the cold turn: 1.10 s measured per 989 MB
checkpoint. The store sits on the root NVMe (WD_BLACK SN850X, nvme0n1p2),
whose own throughput at that path is 3.9 GB/s O_DIRECT write, 4.8 GB/s read
and 2.7 GB/s with fdatasync, so the disk is not the limit. The device writes
about 1.98 GB per checkpoint: the native payload is staged to a temporary
file, then copied into the final KVC file with its header and text, and both
files are fsynced (store.rs write_payload_file, format.rs stage_stream). The
effective rate is therefore about 0.9 GB/s of file for about 1.8 GB/s of
device traffic, and a single-write path would roughly halve the cost. At the
32 GB budget that is about thirty parked prompts; eviction is score-ordered.

A restore is not required to reproduce a fresh prefill bit for bit, and on
this card it cannot be made to: two identical fresh prefills of the same
10,104-token prompt, same server, temperature 0, returned different texts,
while the restored path repeated identically across three requests. The
full-prompt restore did return the same 16 greedy tokens as a fresh prefill
in the tested sample, which is the strongest equality the path supports
without a logits dump.

## Split-K decode attention

`cuda/gemma4_attn_split.cuh` is the long-context decode fix. The fast decode
kernel is one CTA per KV head: two CTAs on the five full layers and eight
64-thread CTAs on the sliding ones, so most of the machine is idle while the
key loop runs. The split kernel divides each KV head's visible key range
across `splits = ceil(visible / span)` CTAs, where the span is
`DS4_GEMMA4_ATTN_SPLIT_SPAN` (default 64, 16..4096; the Rust host rejects a
value outside that range and the legacy C hosts still clamp), each running the
fast kernel's own per-key dot and online softmax
over its range and writing the unnormalized O plus (m, l). A combine kernel
joins the partials with `m = max m_s`, `l = sum l_s * exp(m_s - m)` and
`o = sum o_s * exp(m_s - m) / l`. The per-key arithmetic is unchanged; only
the softmax combine order differs, so this is a parity path. The synthetic
gate (sliding decode, ring wrap, full decode, shared prefix) measures max
error 6.6e-7 against the reference. On the 24-token greedy check, 23 tokens
are identical and the last word flips on a near-tie. The span sweep that set
the default is below.

`cuda/gemma4_attn_split_mma.cuh` replaces the per-split kernel on the full
layers (D = 512, group = 8) with tensor cores: the eight query heads of a KV
head sit in rows 0-7 of `m16n8k16`, the key tile in N, head_dim 512 in K, and
P V is a second MMA over 64-key tiles with P rounded to bf16. K and V share
one 64-key staging buffer, the online softmax rescales once per tile instead
of once per key, and the split count, scratch layout and combine kernel are
the scalar path's. `DS4_GEMMA4_ATTN_SPLIT_MMA=1` selects it (the launcher
default); 0 keeps the scalar per-key kernel. The synthetic gate (span
512/128/96, ring wrap, shared prefix, tile edges, single key) measures max
error 5e-6 against the f32 reference on sm_89 and sm_121, inside the 2.5e-3
class-B gate.

The MMA path against the scalar one, eight fast options, the span-128 default
of that build:

| frontier | split-K scalar | split-K MMA | ratio |
|---|---:|---:|---:|
| 256 | 27.60 | 28.29 | 1.025x |
| 1,024 | 26.62 | 27.41 | 1.030x |
| 13,447 | 24.09 | 26.38 | 1.095x |

Measured on the Spark, ds4-bench, the launcher's fast set, the persistent MoE
and the 2,048-row prefill chunk, 64 generated tokens, 2026-11-04. The default
is 64: it wins at every measured context.

| frontier | span 128 | span 64 | span 32 | span 16 |
|---|---:|---:|---:|---:|
| 256 | 36.96 | 38.31 | - | - |
| 1,024 | 35.87 | 37.25 | - | - |
| 13,447 | 33.96 | 35.72 | 34.56 | 33.30 |

The same frontier with the lighter arm (eight fast options, split-MMA, no
persistent MoE, shipped chunk) reads 26.41 / 27.28 / 26.85 / 25.98 gen tok/s
for 128 / 64 / 32 / 16, and 1,005.7 prefill tok/s; 64 is ahead there too.

The greedy 32-token sequences are identical at all three contexts, and at
13,447 the frontier logits dump is byte-identical between the two paths. The
earlier landing numbers were the fast kernel's 4.81 against split 18.96 at
13,447, before the span knob and the persistent MoE.

MTP draft 6 at 13,447 moves 3.87 -> 6.40 tok/s with the split, but plain
split decode still beats it: the seven-row verify is the remaining MTP cost
and stays on the tile kernel. The long-context serving setting is therefore
no MTP plus the split path; MTP remains a short-context tool. The launcher
enables the split by default (`DS4_GEMMA4_SPLIT=0` keeps the single-CTA
kernel).

## Typed knobs

Every arithmetic knob above is also a typed option: the `--gemma4-*` flags and
the `DS4_GEMMA4_*` variables funnel through `Gemma4Options` (strict ranges,
named errors), reach the engine as `ds4_engine_options.gemma4`, and are echoed
in the plan JSON and by `--check-config`. The surface is
`crates/ds4-core/src/gemma4_knobs.rs`; the C side reads it through
`ds4_gemma4_knobs.h`, and the legacy environment path remains only for the C
hosts.

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
  its embedded MTP), and the card boots device-resident with the FP8
  full-layer KV: 9.37 GiB copied in 2.8 s, 10.16 GiB of the 12 GiB used at
  ctx 16384, 20.95 tok/s against 2.66 tok/s host-mapped. The banked lane is
  refused at that size (serial only, reuse=exact). Since the E4M3 tensor-core
  port and with `DS4_SESSION_GRAPH_HEADROOM_MB=0`, the launcher boots at ctx
  65536: 11.45 GiB of the 12 GiB with the session graph, 833 MiB free, and a
  cold 61,932-token request in 55.7 s; 98304 is refused by the quote.
  `run-gemma4.sh` starts it (serve start/stop/status/logs, chat, plan, bench),
  and the open-grok model entry is `gemma4-vq-4070` on port 8099.
  `--ssd-streaming` still takes the shared GLM-5.3 branch ("requires one
  GLM-5.3 GGUF shard") rather than the Gemma discrete map; that remains open.
- The E4M3 full-layer lane runs the tensor-core kernels since the port above:
  `g4_attn_mma_full_kernel<true>` for prefill and
  `g4_attn_split_mma_kernel<true>` for decode, selected by the existing
  `DS4_GEMMA4_ATTN_MMA` and `DS4_GEMMA4_ATTN_SPLIT`/`_MMA` knobs. Two gaps
  remain and neither is new in kind: the MTP verify widths (2..8 rows) stay on
  the scalar kernel exactly as they do in bf16, and the E4M3 prefill runs at
  93 percent of the bf16 lane because every element is decoded on the way into
  the tile. The scalar FP8 kernel is still reachable with
  `DS4_GEMMA4_ATTN_MMA=0` and `DS4_GEMMA4_ATTN_SPLIT_MMA=0`.
- The sweep's top frontier is skipped. At 65,536 the byte-estimated prompt
  lands a few tokens over the booted context and the server refuses it
  ("bridge error 1: prompt exceeds context"), so both committed 2K-64K CSVs
  end at 63,488. The bytes-per-token ratio is calibrated on an 8,192-byte
  slice, and the last frontiers' text tokenizes slightly denser than that
  slice. The repair is either a few hundred tokens of headroom in the boot
  context or a trim-and-retry in the sweep; neither is in this unit.
- The serial session has no ring-resume guard yet: a divergent continuation
  that reaches back past the rows the ring still holds can read overwritten
  slots. The banked lane is exact and the checklist does not exercise it.
- A request that loads a checkpoint stores nothing deeper when it finishes.
  The only stores on that path are the previous live session (before the
  load) and the continued interval (10,000 tokens, aligned to 10,240), so a
  growing conversation restores one turn behind: turn N parks at its own
  marker only when turn N+1 begins. Storing at the request's end would let
  every turn restore its full own prefix; it is a policy change, not a
  correctness one.
- The parked payload is the whole cache, sliding rings included, so it is
  about 989 MB at ctx 65536 almost independently of the prefix length. A
  later slice can narrow the full layers to the live window if the budget
  or the write time ever bites (the payload format already has the field).
- Each checkpoint is written twice: the native payload goes to a staged file
  and `stage_stream` copies it into the final KVC file. With both fsyncs the
  device sees 1.98 GB per 989 MB checkpoint, which is most of the 1.10 s
  save. Handing the native save the final file and an offset would remove
  the copy and one of the two writes.
- `--kv-cache-cold-max-tokens 0` does not stop the lane from writing: it only
  disables the cold-path store, while the store taken before a load (the
  previous live session) still fires. Measured: an arm configured that way
  wrote a checkpoint per request anyway.

## Expert placement: three knobs, llama.cpp's shapes

The routed experts are 7.02 GiB of the 9.37 GiB map (the other 2.35 GiB is the
resident set: attention, norms, books, the token table and the router). Three
typed knobs decide where they run, and each mirrors a llama.cpp flag:

- `DS4_GEMMA4_CPU_MOE=1` (`-cmoe`): every expert stays host-cold in the mmap
  and the top-8 runs on the CPU (cuda/gemma4_cpu_moe.cuh, AVX2 R13+8 GEMV,
  ascending expert-id accumulation, 20 threads by default).
- `DS4_GEMMA4_MOE_CPU_LAYERS=N` (`-ncmoe N`): layers below N host-cold with
  CPU top-8, layers at or above N device-resident on the GPU. The accepted
  range is 0..999 and the C side clamps to the 30 routed layers, so 30, 99 and
  999 all mean "every layer", exactly as `-ncmoe 99` does in llama.cpp
  (common/arg.cpp:2778-2787). The dispatch chooses per layer
  (ds4_gemma4_graph.inc), so a mixed model needs no route split: each layer
  keeps its own accumulation order.
- `DS4_GEMMA4_MOE_CACHE_MIB=N` (`--moe-cache-mib`): every expert stays
  host-cold and N MiB of VRAM slots are filled from the mmap as the router
  asks (ds4_gemma4_stream.inc, the same pool `--ssd-streaming` uses, with
  `MOE_HOST` implied). A decode miss takes `DS4_GEMMA4_MOE_MISS`: `evict`
  (default) or `cpu` to compute it on the host.

`DS4_GEMMA4_MOE_TRACE=1` prints a per-layer routing histogram at exit
(`top8/top16/top32/top64` shares and the eight hottest ids), which is the
sizing input for the cache.

## Measured, 4070 SUPER, 2026-11-05

Cold server per arm, greedy 32 tokens, prefix cache empty. TTFT is the first
token; VRAM is boot/peak.

| mode | ctx | prompt | TTFT | decode | VRAM |
|---|---|---|---|---|---|
| resident | 8192 | 7,555 | 5.87 s | 54.8 tok/s | 10,121/10,861 MiB |
| resident | 32768 | 8,294 | 6.46 s | 54.6 | 10,121/11,103 |
| resident | 32768 | 31,909 | 26.45 s | 52.5 | 10,121/11,113 |
| resident | 65536 | 31,909 | 26.43 s | 51.8 | 10,121/11,433 |
| CPU MoE | 8192 | 7,555 | 362.8 s | 17.3 | 3,617/3,673 |
| CPU MoE | 65536 | 8,294 | 370.0 s | 18.7 | 4,177/4,233 |
| CPU MoE | 65536 | 31,909 | 1,410.0 s | 19.2 | 4,177/4,233 |
| cache 4 GiB + miss=cpu | 8192 | 7,555 | 22.1 s | 28.0 | 7,713/7,773 |
| cache 4 GiB + miss=cpu | 32768 | 8,294 | 24.3 s | 26.9 | 7,953/8,015 |
| cache 4 GiB + miss=cpu | 32768 | 31,909 | 81.6 s | 28.5 | 7,953/8,025 |
| cache 4 GiB + miss=cpu | 65536 | 31,909 | 82.0 s | 27.6 | 8,273/8,345 |

Reading the table: the host-compute lane costs about 1.6 ms per token per CPU
layer, so six layers turn a 7,555-token prefill from 5.9 s into 42 s (99
tok/s) for 734 MiB of VRAM, and thirty layers turn it into six minutes. The
CPU arms ran while unrelated jobs shared the host, so their decode is a floor,
not a ceiling. The cache lane is three times slower at prefill than the
resident lane and lands at 28 tok/s decode; its prefill cost is the slot pool
re-staging VQ expert records into a 4 GiB pool, not disk traffic (see below).

Two corrections to earlier readings:

- The `reread=72.85 GiB` line the stream reports is VQ expert payload
  re-uploaded into the pool: `reread_bytes` counts records `seen` before
  (ds4_gemma4_stream.inc:469 and :511) and with the host source the bytes come
  from the mmap, so `read=0.00 GiB` and `io=0.0 ms` on the same line. It is
  PCIe upload time, not disk.
- Token identity must be defined against `DS4_GEMMA4_PERSIST=0`: the
  persistent MoE decode is near-exact (26 of 30 layers bit-identical). With
  persist off, the all-GPU host-mapped arm and the all-CPU arm produced the
  same 32-token reply (md5 d9eb05c0b72d at ctx 8192), which is the parity the
  placement change has to keep.

Open defects on the new lanes, both reproduced:

- With `DS4_GEMMA4_MOE_CPU_LAYERS` between 1 and 29 at ctx 8192, a
  7,555-token prompt is refused with "prompt exceeds context" while the same
  prompt is accepted at N=0 and N=30, and 6,597 tokens is accepted at N=6.
  The check is `prompt->len > s->ctx_size` (ds4.c:78685), so the session ctx
  is being right-sized below the advertised 8,192 on those lanes.
- `DS4_GEMMA4_MOE_CACHE_MIB` shows the same refusal at ctx 8192 with a 2 GiB
  and a 4 GiB pool, so the right-sizing is tied to the promoted-stream map
  rather than to how much VRAM the lane leaves free.

## The expert fill, 4070 SUPER, 2026-11-05

One prompt (7,239 tokens at ctx 8192, chunk 2048) with the reply md5
57a0da50adbfbc90ca06b5e73cb6d842 in every row, so the changes below are
token-identical. `copy` is the host read into the pinned ring, `submit` the
cudaMemcpyAsync submissions, `wait` the device drain.

| step | TTFT | prefill | where the time went |
|---|---|---|---|
| cache 4 GiB, chunk 512, linear scan | 22.0 s | 329 tok/s | scan 2.3 s, 71.5 GiB of rereads |
| + chunk 2048 | 13.8 s | 524 | rereads 18.5 GiB, upload 7.1 s |
| + pool 7 GiB | 12.2 s | 594 | rereads 0.0 GiB, scan 2.3 s, copy 4.9 s |
| + per-layer slot map | 10.1 s | 718 | scan 0.02 s |
| + warm the mapping at open | 5.71 s | 1,279 | copy 787 ms for 6.51 GiB (8.27 GiB/s) |

What the numbers say:

- The fill's ceiling is the source, never the card. `tests/bench_gemma4_staging`
  reads the same 0.39 MiB records at 12.8 GiB/s pageable, 18.5 pinned,
  8.4 through the mapping and 6.4 by pread. The device side needs 43.5 ms to
  submit and 1.4 ms to drain 6.5 GiB once the records are staged.
- Pinning the staging buffer does not help on its own: 12.31 s queued against
  12.23 s blocking at a 7 GiB pool. The 3.9x came from removing page faults -
  the same records read 1.36 GiB/s on first touch and 8.3 GiB/s afterwards in
  one process - so the engine touches every expert record once at open
  (`ds4: Gemma-4 expert warm`, 7.01 GiB in 4.98 s, disk-bound at 1.41 GiB/s)
  and `DS4_GEMMA4_MOE_WARM=0` skips it.
- The staging ring must hold one whole layer (768 slices): a shorter ring
  wraps mid-layer and that wrap waits behind the previous layer's compute.
  With a whole layer it never wraps, and a batch that fills free slots is not
  ordered behind compute at all, so the fill runs under the previous layer.
- A 7 GiB pool holds all 3,830 expert records that fit and re-reads nothing
  after the first pass; 6 GiB re-reads 2.2 GiB and 4 GiB re-reads 18.5 GiB for
  the same prompt, because every 512-row chunk re-walks all thirty layers.

The next unit is the one this table argues for: keep the GPU as the only
expert compute unit, stream each layer's used experts during prefill (llama.cpp offloads any `MUL_MAT_ID` above 32
tokens, ggml-cuda.cu:182, and `sched_copy_experts` uploads only the used set,
llama-context.cpp:2780), and upload decode misses into a slot-mapped cache
instead of computing them on the host (llama-moe-cache.cpp:496-506 bypasses
its cache above 32 tokens for exactly that reason).

## Reproduce

```
# build (the GPU arch: sm_89 on the 12 GiB card, sm_121 on the Spark)
make cuda CUDA_ARCH=sm_89

# the checklist on the R13+8 artifact
DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock \
  ./ds4 -m <release-dir> --cuda -c 2048 \
  --serving-check -p "Hello"
```

The artifact directory and the reference checkpoint are not committed; the
paths above are this machine's. `DS4_LOCK_FILE` must point away from
`/tmp/ds4.lock` when another ds4 server holds it.
