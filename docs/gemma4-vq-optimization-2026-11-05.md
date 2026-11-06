# Gemma-4 26B A4B VQ optimization, 2026-11-03 to 2026-11-05

Rounds measured while jlFN ported the family
([Baekpica/ds4-dfm-rs#141](https://github.com/Baekpica/ds4-dfm-rs/pull/141)),
on the R13+8 release ([family page](gemma4-vq.md)). Two hosts:

- one DGX Spark (GB10, `sm_121`), outside the clock policy: busy SM clocks
  were 2,398–2,460 MHz, above the 300–2,200 MHz range, and no power or
  thermal samples were recorded. Absolute rates here do not compare with
  other families' GB10 rows; ratios measured back to back in one session
  do.
- one RTX 4070 SUPER (12 GiB, `sm_89`), the limited-VRAM reference.

`fast set` is eight switches (`ATTN_FAST`, `ATTN_PREFILL`, `VQ_PREFILL`,
`Q8_BATCH`, `FUSE_ROPE`, `ROUTER_PAR`, `HEAD_GEMV`, `ROUTER_DOT`, each
`DS4_GEMMA4_*`), now on by default with the MMA and split-K attention and the
persistent MoE.

## R1: fused verify head

The drafter's verify block runs the target's head over up to seven rows.
`g4_head_gemv_rows_kernel` (`cuda/gemma4_head.cuh`) reads the 1.48 GB tied
BF16 table once and accumulates every row in the one-row kernel's eight-FMA
sequence and XOR reduction, so each row is bit-identical to the per-row call
(`tests/gemma4_head_rows`: 4,096 of 4,096 bytes for 1–8 rows, `sm_89` and
`sm_121`). The ids come from `ds4_gpu_argmax_rows_tensor` on the device, and
the commit reads only the kept frontier row instead of the 7.34 MB block.

GB10, ctx 1024, draft 6, device-resident: 18.38 → 20.21 tok/s decode
(+10.0%); plain decode unchanged (6.76).

## R2: drafter attention on the fast kernel

Each draft step's attention used the oracle kernel: one warp per query head,
each reading its own K/V row, so the full drafter layer's 16 query heads read
each of the trunk's two KV rows eight times. `ds4_gpu_g4_attn_fast` (one CTA
per KV head, one shared 16-key tile) refused `keys != 0`; the drafter passes
`keys = start`. The guard now accepts the oracle's bound. Both kernels walk
the same keys in the same order with the same per-lane dot, warp reduction
and F32 softmax: the 1,081-token gate (draft 6, 24 greedy tokens) is
byte-identical.

GB10, 13,447-token prompt, `ds4-bench`, 17 greedy tokens:

| Arm | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---:|
| no MTP, four fast switches | 312.0 | 4.55 | 0.23 |
| no MTP, fast set | 315.5 | 4.83 | 0.21 |
| MTP draft 6, four switches, before the guard fix | 314.1 | 2.36 | 0.76 |
| MTP draft 6, four switches, after | 312.1 | 3.69 | 0.47 |
| MTP draft 6, fast set, after | 316.2 | 3.84 | 0.45 |
| MTP draft 2, fast set, after | 314.7 | 4.83 | 0.28 |

The drafter stays a net loss at 13.4K: rounds keep 1–2 tokens, so six draft
steps cost more than they save; at 2,048 tokens the same prompt keeps
7/1/1/1/2/3/1/4/3 and decodes 15.06 tok/s. The 13,447-token prefill trace
(chunk 2,048, nsys) put the attention tile kernel at 72.2% of GPU time (53.1%
full layers, 19.1% sliding), the two direct VQ expert GEMMs at 14.7% and the
batched dense Q8 down at 7.0%.

## R3: tensor-core prefill attention

`cuda/gemma4_attn_mma.cuh`: one template per geometry (`<256, 2>` sliding,
`<512, 8>` full), one CTA per (KV head, row tile), eight warps, 32-key tiles.
The sliding CTA covers 64 tokens (8 tokens × 2 query heads per warp, 32 KiB
shared); the full CTA covers 8 tokens (2 tokens × 8 query heads per warp, two
warps per M group splitting D = 512, 32 KiB K/V plus 16 KiB score exchange).
Q stays in registers, the softmax is F32 and online once per tile, P rounds
to BF16 for the P·V MMA, and the key loop starts at `(begin_all / 32) * 32`
so a token sees the same tiles at every chunk size. Rows ≥ 16 with
`keys == 0` take it; the MTP verify widths (2–8) stay on the tile kernel.

Synthetic parity (`tests/test_gemma4_primitives`, both geometries, rows
16/17/64, first 0/1000/1025/4096/8192, ring, window, shared prefix and
poisoned slots): max absolute error 1.4e-3 against `g4_ref_attn` (gate
2.5e-3). A one-layer probe (`DS4_GEMMA4_ATTN_CMP=1`, layers 0 and 5, ctx 256)
puts the output at relL2 4.05e-4 and 3.94e-4 against the tile kernel. On the
pinned 256-token prompt the eight design tokens match, KL(tile ‖ MMA)
8.6e-5; on a longer prompt the greedy text diverges after the first sentence
(a near tie) and the frontier KL is 0.026 at 256 tokens and 0.10 at 2,048,
routing propagation through 30 MoE layers rather than kernel error.

GB10, `ds4-bench`, fast set, 17 greedy tokens:

| Frontier | Tile prefill tok/s | MMA prefill tok/s | Ratio | Decode |
|---:|---:|---:|---:|---:|
| 2,048 | 576.1 | 977.1 | 1.70× | 12.6 (both) |
| 13,447 | 309.5 | 1,061.4 | 3.43× | 4.82 (both) |

## R4: split-K decode attention

The fast decode kernel runs one CTA per KV head: two CTAs on the full layers,
eight 64-thread CTAs on the sliding ones. `cuda/gemma4_attn_split.cuh`
divides each KV head's visible keys over `ceil(visible / span)` CTAs, each
running the fast kernel's per-key dot and online softmax and writing
unnormalized O plus (m, l); a combine kernel joins them (`m = max m_s`,
`l = Σ l_s·exp(m_s − m)`, `o = Σ o_s·exp(m_s − m) / l`). Only the combine
order changes: max error 6.6e-7 on the synthetic gate; 23 of 24 greedy tokens
match, the last flips on a near tie.

`cuda/gemma4_attn_split_mma.cuh` runs the full layers (D = 512, group 8) on
tensor cores: the eight query heads of a KV head are rows 0–7 of
`m16n8k16`, keys in N, head_dim in K, then P·V over 64-key tiles with P in
BF16. Max error 5e-6 against the F32 reference (`sm_89`, `sm_121`). Split-K
scalar against MMA, fast set, span 128: 27.60 → 28.29 tok/s at 256, 26.62 →
27.41 at 1,024, 24.09 → 26.38 at 13,447.

Span sweep (GB10, fast set, persistent MoE, chunk 2,048, 64 tokens,
2026-11-04), decode tok/s:

| Frontier | Span 128 | Span 64 | Span 32 | Span 16 |
|---:|---:|---:|---:|---:|
| 256 | 36.96 | 38.31 | – | – |
| 1,024 | 35.87 | 37.25 | – | – |
| 13,447 | 33.96 | 35.72 | 34.56 | 33.30 |

64 is the default. Greedy 32-token sequences are identical at all three
contexts. MTP draft 6 at 13,447 moves 3.87 → 6.40 tok/s with the split, below
plain split decode: the seven-row verify stays on the tile kernel.

## R5: full-layer E4M3 KV on tensor cores

E4M3 full-layer rows (`DS4_GEMMA4_FULL_KV=fp8`) had only
`g4_attn_fp8_kernel`: one warp per (row, head), a scalar loop over every
visible key. nsys (chunk 2,048, one cold frontier):

| Frontier | `g4_attn_fp8_kernel` | Share of GPU time | All other kernels |
|---:|---:|---:|---:|
| 2,048 | 0.72 s | 28.2% | 1.83 s |
| 32,768 | 277.1 s | 90.5% | 29.1 s |
| 65,536 | 1,113.7 s | 95.0% | 58.6 s |

`g4_fp8_decode8` decodes eight codes per `uint2` with the hardware
`e4m3x2 → f16x2` conversion, multiplies by the row's F16 scale in F32 and
rounds to F16 into the BF16 kernels' shared tile (the decode stage of Solar's
warp-specialized FP8 kernel, `cuda/mmq/ds4_fattn.cu`). Mask, online softmax,
exchange, split partials and combine are the BF16 path's; the tensor-core arm
differs from the scalar kernel by one F16 rounding of K and V. Synthetic max
error 1.4e-4 prefill and 3.2e-6 decode; layer 5 probe relL2 1.82e-4; on the
256-token prompt the eight design tokens match, KL 1.7e-3 (BF16 lane: 9.8e-5).

GB10, cold frontier, chunk 2,048, one session:

| Frontier | Scalar prefill tok/s | Tensor-core prefill tok/s | Ratio |
|---:|---:|---:|---:|
| 2,048 | 489.6 | 770.1 | 1.57× |
| 32,768 | 97.9 | 746.6 | 7.63× |
| 65,536 | 47.2 | 460.3 | 9.75× |

Decode at 32,768: 0.50 → 19.27 tok/s. Beside the BF16 lane at 32,768: 746.6
against 804.6 prefill, 19.27 against 18.71 decode; the gap is the per-element
decode the BF16 `cp.async` copy does not pay.

## R6: host tokenizer

A 2K–64K sweep reported TTFT growing with context but not with prefill: at
59,392 tokens the server prefilled 2,130 tokens at 852 tok/s and answered in
35.0 s. A repeat with the prefix fully cached still took 34.5 s: the host BPE
merged one rank per scan of the symbol list. The merge now pops the lowest
pair from a min-heap keyed on `(rank, left, right, stamp_left,
stamp_right)`; the merge order is unchanged and the token stream is
byte-identical on a 60 KB slice of `speed-bench/promessi_sposi.txt` and the
C oracles' parity test. The order itself was not tokenizers' (one rank group
per round, created pairs deferred to the next round): the PR review found
that runs of 32 or more spaces, tabs or newlines split differently, and the
merge now ranks created pairs at once (`tok/gemma4.rs` tests).

| Request (GB10, one server, 215 KB prompt) | Before | After |
|---|---:|---:|
| 61,872 tokens, cold | 98.20 s TTFT (62.9 s prefill) | 62.92 s TTFT (62.8 s prefill) |
| 61,871 tokens, fully cached | 34.46 s TTFT | 0.12 s TTFT |
| 60 KB tokenize, 4070 host | 7.43 s | 0.27 s |

## R7: disk KV parking

Three defects kept the disk lane from paying for itself:

- No KV identity class: `routed_quant_bits` matched GGUF aliases and HF
  names only, the release names its experts `model.layers.N.experts.M.*`,
  so `kv_identity` returned None and every store was skipped silently.
- Non-ASCII piece text: the printer applied GPT-2's byte alphabet to every
  vocabulary, so `▁però` printed byte 0xF2 instead of UTF-8 and a re-rendered
  prefix no longer matched its text key. The pinned decoder (Replace U+2581,
  ByteFallback, Fuse) is now the family's own printer (`tok/gemma4.rs`);
  byte-level vocabularies keep the old arm.
- The chat anchor: with no user/assistant token ids a store fell back to trim
  32 / align 2,048 and cut a 10.1K turn to 8,192 tokens. `generation_anchor`
  cuts at the start of the last `<|turn>model\n`; a cut after `model\n`
  missed on the next turn, because the reply's first character merges with
  the newline.

4070, every request cold (an evicting prompt between them), 64 greedy
tokens: A is a conversation's first turn, B the same plus a short follow-up
after the live session was dropped.

| Prompt tokens | A off | A on | B off | B on | B gain |
|---:|---:|---:|---:|---:|---:|
| 8,294 | 6.26 s | 7.74 s | 6.65 s | 0.66 s | 10.0× |
| 16,243 | 12.54 s | 14.30 s | 12.94 s | 0.70 s | 18.5× |
| 31,909 | 26.29 s | 28.64 s | 26.79 s | 0.74 s | 36.4× |
| 45,737 | 39.78 s | 42.54 s | 40.23 s | 0.80 s | 50.5× |

The cost is the cold turn's write: 1.10 s per 989 MB checkpoint at ctx
65,536. The device sees about 1.98 GB per checkpoint, because the native
payload is staged to a file and copied into the KVC file, both fsynced.

## R8: expert placement on a 12 GiB card

4070, cold server per arm, 32 greedy tokens, empty prefix cache; VRAM is
boot/peak MiB.

| Mode | ctx | Prompt | TTFT | Decode tok/s | VRAM |
|---|---:|---:|---:|---:|---|
| resident | 8,192 | 7,555 | 5.87 s | 54.8 | 10,121/10,861 |
| resident | 32,768 | 31,909 | 26.45 s | 52.5 | 10,121/11,113 |
| resident | 65,536 | 31,909 | 26.43 s | 51.8 | 10,121/11,433 |
| CPU MoE | 8,192 | 7,555 | 362.8 s | 17.3 | 3,617/3,673 |
| CPU MoE | 65,536 | 31,909 | 1,410.0 s | 19.2 | 4,177/4,233 |
| cache 4 GiB, miss cpu | 8,192 | 7,555 | 22.1 s | 28.0 | 7,713/7,773 |
| cache 4 GiB, miss cpu | 65,536 | 31,909 | 82.0 s | 27.6 | 8,273/8,345 |

The host lane costs about 1.6 ms per token per CPU layer: six layers turn a
7,555-token prefill from 5.9 s into 42 s for 734 MiB less VRAM. The CPU arms
shared the host with other jobs, so their decode is a floor. Token identity
is defined against `DS4_GEMMA4_MOE_PERSIST=0` (the persistent MoE is
bit-identical on 26 of 30 layers): with it, the all-GPU host-mapped arm and
the all-CPU arm return the same 32 tokens.

## R9: expert fill

One 7,239-token prompt (ctx 8,192), the same reply md5 in every row:

| Step | TTFT | Prefill tok/s | Where the time went |
|---|---:|---:|---|
| cache 4 GiB, chunk 512, linear scan | 22.0 s | 329 | scan 2.3 s, 71.5 GiB of re-uploads |
| + chunk 2,048 | 13.8 s | 524 | re-uploads 18.5 GiB, upload 7.1 s |
| + pool 7 GiB | 12.2 s | 594 | no re-uploads, scan 2.3 s, copy 4.9 s |
| + per-layer slot map | 10.1 s | 718 | scan 0.02 s |
| + warm the mapping at open | 5.71 s | 1,279 | copy 787 ms for 6.51 GiB |

The source is the ceiling: a standalone staging probe (not kept) read the
0.39 MiB records at 12.8 GiB/s pageable, 18.5 pinned, 8.4 through the mapping and 6.4
by pread, and the device drains 6.5 GiB in 1.4 ms once staged. Pinning alone
is neutral (12.31 s against 12.23 s); the gain is removing first-touch page
faults (1.36 → 8.3 GiB/s), hence the warm pass at open (7.01 GiB in 4.98 s,
disk-bound). The staging ring holds one whole layer (768 slices) so it never
wraps behind compute.

## R10: session quote and boot admission

The quote priced the KV rows only. With a slot pool at chunk 4,096 it
approved an arm the card could not hold: the first forward died 65 KiB short
in the mmq pool allocator with 31.75 MiB free. The quote now prices the
session workspace (`gemma4_session_memory`: activation + linear + routed
scratch), the slot pool, and on host-cold lanes only the 152 promoted spans
(2.345 GiB) plus one 128 B header per cold record (23,040, 2.81 MiB) instead
of the 9.37 GiB mapping.

| Arm (ctx 8,192) | Spans | Pool | KV | Workspace | Total | Free | Plan |
|---|---:|---:|---:|---:|---:|---:|---|
| pool 7,168 MiB, chunk 2,048 | 2.35 | 7.00 | 0.66 | 0.80 | 10.81 GiB | 11.37 | admits |
| pool 5,632 MiB, chunk 4,096 | 2.35 | 5.50 | 1.05 | 1.56 | 10.46 GiB | 11.37 | admits |
| pool 6,144 MiB, chunk 4,096 | 2.35 | 6.00 | 1.05 | 1.56 | 10.96 GiB | 11.37 | admits |

The quote runs before CUDA exists and cannot see the runtime's own 533 MiB,
so the last row also passes the plan. The engine decides at boot:
`gemma4_session_fit` keeps 384 MiB free at peak (`G4_SESSION_FIT_MARGIN`) and
the server refuses to listen, naming the deficit, when the serial session
does not fit. The 5,632 MiB arm boots and serves (peak 11,401 of 12,282 MiB);
the 6,144 MiB arm is refused at boot.
