# Qwen3.8-Flash-Next prefill rounds (2026-09-06)

Follow-up to the 2026-09-04 long-context rounds
(`qwen38-long-context-prefill-2026-09-04.md`).  Same host (DGX Spark GB10,
driver 610.43.02, CUDA 13.3), same artifact
(`Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16`), the
production weight owner (`ds4_weight_server --backend vmm`, aligned Q8
artifacts on) resident throughout, one fresh `ds4-bench` worker per run with
the production worker environment (`DS4_QWEN_PREFILL_CHUNK=8192
DS4_QWEN_PLE_CACHE_MB=2048 DS4_QWEN_PLE_WORKERS=16 DS4_MEMGOV=observe`).
Corpus: Italian prose (`promessi_sposi.txt`).  Two shapes per variant, three
runs each, medians: a cold single-shot 8,192-token prefill (the TTFT shape
of one production request) and a cold single-shot 65,536-token prefill,
each followed by 32 greedy tokens.  Base is `main` at `0510117`.

## What the profile said

nsys on the base binary's cold 8,192-token prefill (`base-cold8k.sqlite`):
6.61 s wall, 5.16 s GPU busy, 1.46 s idle, 1.06 s of it before the layer-1
`qwen38_ple_gather_kernel`: the prompt's first chunk has nothing queued for
it, so its SSD-PLE pages are read while only the token embedding and
decoder layer 0 run.  The busy part, per 8K chunk (same at 64K except the
QSA terms):

| kernel | s per 8K chunk | share of busy |
|---|---:|---:|
| dense Q8_0 MMQ (`mul_mat_q`, 334 launches) | 0.88 | 17.1 % |
| fused QSA attention (12 layers) | 0.75 | 14.5 % |
| routed expert-down worklist (main + tail) | 0.54 | 10.5 % |
| routed gate/up worklist pair | 0.52 | 10.0 % |
| HC residual + norm | 0.43 | 8.3 % |
| activation quantize (514 launches) | 0.33 | 6.5 % |
| GDN recurrent (36 layers) | 0.27 | 5.3 % |
| HC fused mix | 0.20 | 3.8 % |
| moe_sum | 0.17 | 3.4 % |
| MMQ output sanitize passes (387) | 0.13 | 2.5 % |
| weighted SwiGLU | 0.13 | 2.4 % |
| expert-down main pack | 0.06 | 1.2 % |
| shared-expert gate + add | 0.09 | 1.7 % |

## Round 1: opening chunk (`7a5f872`)

Every prompt now starts with a 2,048-row chunk (`qwen4exp_prefill_rows`,
both the serial session loop and the continuous banked scheduler).  The
opening chunk exposes only its own pages (~0.3 s); the full-size chunk
behind it is queued after layer 1 by the existing lookahead and read while
the opening chunk's remaining 46 layers run.  Later chunks keep the cap.
`DS4_QWEN_PREFILL_OPENING` overrides the opening rows (0 = cap).

Review follow-up (`6101811`): the split only pays when the chunk behind the
opening one is at least as long.  A short trailing chunk runs its layers at
low occupancy and hides few reads: 2,304 rows as 2,048 + 256 measured
1126.7 tok/s against 1211.6 as one chunk (-7 %), 2,049 rows as 2,048 + 1
1169.9 against 1181.3 (-1 %), and balancing the two halves (1,025 + 1,024,
1,536 + 1,536) 1127.4 / 1127.9 against the split's 1169.9 / 1179.7 (-4 %).
Prompts shorter than two opening chunks now stay one chunk (2,049 /
2,304 / 3,072 rows: 1181.1 / 1206.1 / 1219.0 tok/s, the single-chunk rate);
prompts of 4,096 rows and more, so every shape measured here, are unchanged
(cold 8K 1362.9 tok/s after the change).

| shape | base `0510117` | round 1 |
|---|---:|---:|
| cold 8,192 tokens | 1214.8 (1212.7 / 1214.8 / 1217.2) | **1325.7** (1322.3 / 1325.7 / 1329.6), +9.1 % |
| cold 65,536 tokens | 1382.7 (1380.7 / 1382.7 / 1382.9) | **1401.1** (1396.6 / 1401.1 / 1408.5), +1.3 % |

PLE gather wait of the 8K run 1.30 s -> 0.57 s (blocked wait 1.04 -> 0.36 s).
Decode after the prefill unchanged (24.5 / 23.9 tok/s).

## Round 2: MoE glue traffic

Two passes per layer moved data for nothing at prefill widths:

- The fused expert-down MMQ read its 512-column main input from a packed
  copy of the SwiGLU rows (`qwen4exp_pack_expert_down_main_kernel`, 378 MB
  per 8K layer) although the entry already read the 128-column tail in
  place with a row stride.  Both halves now read the 640-wide rows in place
  (`ds4_mmq_moe_tail_impl` takes the main stride); the pack kernel stays
  only for the paired-bank decode path, which keeps the separate main GEMM.
- The routed gate/up pair quantized the activation once per assignment
  slot (`quantize_mmq_q8_1` through `ids_src1`), reading every token row
  top-k times, ~0.84 GB per 8K layer.  It now quantizes once per token and
  scatters the 144-byte Q8_1 blocks into the expert-sorted layout the
  worklist tiles stream (`ds4_q8_1_mmq_gather_rows`, ~0.23 GB written, the
  23 MB compact source mostly from L2).  The first attempt read the
  token-compact buffer through a per-column map inside the tile instead:
  bit-identical, but the gate/up kernel went 2.39 -> 2.92 ms per launch
  (2,048 rows) and prefill lost 3 %, so the tiles keep their contiguous
  loads.  `DS4_MMQ_NO_YIND` restores the slot-gathered quantize.

Both changes are bit-identical by construction (same quantized values,
same tiles, same dots); the fixture (`tests/test_qwen4exp_moe`) checks the
pairs against the generic routed matmul under balanced and narrow-tile
routing (production routes ~5 rows per expert, so the 8- and 16-wide
worklist tiles carry most of the work) and the fused down against the
separate main + tail.

| shape | round 1 | round 2 |
|---|---:|---:|
| cold 8,192 tokens | 1325.7 | **1348.1** (1347.6 / 1348.1 / 1349.2), +1.7 % (in-place read alone: 1336.5) |
| cold 65,536 tokens | 1401.1 | **1422.4** (1421.1 / 1422.4 / 1422.6), +1.5 % |

## Round 3: one-pass block output (`974d706`)

`qwen4exp_moe_sum_shared_kernel` forms the block output in one pass: it
sums the top-10 routed rows (non-finite dropped, as `moe_sum` did) and adds
`shared * sigmoid(gate_logit)` with separate roundings (`__fmul_rn`,
`__fadd_rn`), replacing `moe_sum`, the shared-expert gate kernel and the
in-place add, i.e. two [rows x hidden] round trips per layer.  Bit-identical
to the three (fixture, including a dropped non-finite value); the decode and
paired-bank paths take the same kernel.  Per 6,144-row chunk: 2.72 ms
(`moe_sum`) + 0.6 + 1.1 ms (gate, add) -> 2.94 ms.

Rejected in the same round: the weighted SwiGLU folded into the up launch's
store (a ds4-side copy of `mul_mat_q_process_tile` whose write back reads the
gate the first launch stored and writes `silu(gate) * up * w[pair]`, so up
is never stored and the separate pass goes).  Bit-identical on the fixture,
but the up launch went 4.49 -> 5.84 ms per 6,144-row launch (+0.13 s per
chunk; the epilogue's gate reads follow the scattered `ids_dst` stores)
against 0.095 s of SwiGLU pass removed: cold 8K 1348.1 -> 1338.8 tok/s.
The scattered read costs more than the coalesced pass it replaces; the
patch is kept under `scratch/qwen-prefill-opt-20260906/r3-swiglu-store-rejected.patch`.

| shape | round 2 | round 3 |
|---|---:|---:|
| cold 8,192 tokens | 1348.1 | **1362.1** (1352.0 / 1362.1 / 1367.4), +1.0 % |
| cold 65,536 tokens | 1422.4 | **1439.7** (1428.7 / 1439.7 / 1443.9), +1.2 % |

## Round 4: K=2560 dense D2R (`feature/qwen-prefill-opt-20260906-r4`)

The dense Q8_0 D2R kernel already accepted K % 128 and K <= 4096, but the
weight owner's `ds4_repack_q8_candidate` only built aligned artifacts for
`dims[0] % 1024 == 0` (plus the 2560x640 shared-expert special case).
Qwen GDN qkv/z and QSA q are K=2560, so they never reached the tier and
stayed on `mul_mat_q` (17 % of an 8K chunk on the round-3 binary).

The candidate predicate now also admits the D2R prefill contract
(K % 128, K <= 4096, M % 128, M >= 2048).  The catalog mirror matches.
Additive artifacts: 160 tensors / 0.94 GiB -> 278 / 3.62 GiB
(+2.68 GiB in the owner).  `DS4_MMQ_DENSE_D2R=0` is the kill switch
(same binary, same owner).  Fold order differs from mmq: value-parity,
not bit-parity.  The first engaged launch is GDN qkv
`M=10240 N=2048 K=2560` on the opening chunk.

On the 6,144-row chunk, 84 D2R launches take 0.29 s and the remaining
`mul_mat_q` (o_proj K=6144, index M=640, in_a/in_b) is 0.24 s; the
standalone sanitize pass on that chunk falls 387 -> 303 launches
(0.13 s -> 0.030 s) because D2R writes every element through its
isfinite epilogue.

| shape | round 3 same-hour off | round 4 |
|---|---:|---:|
| cold 8,192 tokens | 1353.3 (1344.8 / 1353.3 / 1364.5) | **1391.1** (1386.2 / 1391.1 / 1393.1), +2.8 % |
| cold 65,536 tokens | 1431.1 (1419.7 / 1431.1 / 1440.2) | **1504.6** (1498.0 / 1504.6 / 1508.5), +5.1 % |

Decode after the prefill unchanged (24.5 / 23.9 tok/s).

## Round 5: HC mix q8 emit (`feature/qwen-prefill-opt-20260906-r4`)

GDN qkv+z and QSA q+index all read the same F32 HC mix, and each dense
entry quantized it again.  After a successful mix the existing
`cuda_norm_q8` registry now publishes one `quantize_ref` of `mixed`;
D2R `preq` (K gate widened from %1024 to %128 so K=2560 matches the
launch) and mmq `preq` consume it.  Kill switch
`DS4_CUDA_NO_NORM_Q8EMIT=1`.  Bit-identical to the per-GEMM quantize
(same kernel, same buffer).  Decode width never emits (`rows < 64`).

| shape | round 4 same-hour off | round 5 |
|---|---:|---:|
| cold 8,192 tokens | 1394.4 (1388.1 / 1394.4 / 1407.7) | **1406.0** (1401.9 / 1406.0 / 1408.3), +0.8 % |
| cold 65,536 tokens | 1504.1 (1503.1 / 1504.1 / 1507.1) | **1518.4** (1518.1 / 1518.4 / 1521.4), +0.9 % |

Decode after the prefill unchanged at 8K (24.6 tok/s); 64K 23.9 vs 23.2
is run noise (emit is off at decode width).  First engaged logs:
`HC mix emits producer q8` then `dense q8 D2R consuming producer q8`
at the 2,048-row opening chunk.

## Cumulative and the production shape

Cold single-shot `ds4-bench` prefill, `main` `0510117` -> `974d706`: 8,192
tokens 1214.8 -> 1362.1 tok/s (+12.1 %), 65,536 tokens 1382.7 -> 1439.7
tok/s (+4.1 %), 196,608 tokens 1297.5 -> 1325.6 tok/s (+2.2%, one
run each).

Production server shape (the canonical two-bank command: 196,608 configured
context, `DS4_QWEN_BATCH=1`, 8,192-token chunks, 2048 MiB PLE cache, 16 page
workers, `--mtp-draft 2`; three fresh workers with fresh disk-KV directories
per binary, `thinking` disabled, non-streaming, the API's
`timings.prefill_tok_s`), the 2026-09-02 card prompts, medians of three:

| binary | corpus3 8,259 tok (repeated passage) | corpus.txt 7,937 tok (markdown, cold PLE) | x-prompt 8,036 tok | corpus3 + 256 greedy tokens: prefill / decode / ms per MTP step |
|---|---:|---:|---:|---:|
| `0510117` | 1363.3 | 1406.1 | 1570.0 | 1403.4 / 31.3 / 52.7 |
| `974d706` | **1394.2** (+2.3%) | **1467.5** (+4.4%) | **1584.2** (+0.9%) | 1453.4 / 29.8 / 53.7 |

The 256-token continuation of the chat prompt differs between the two
binaries (`text_sha` in `api-results.tsv`), and so does its MTP acceptance
(1.65 -> 1.60 tokens per step), which is the whole decode difference: the new
binary with `DS4_QWEN_PREFILL_OPENING=0` reproduces the old text and the old
decode rate (31.4 tok/s, 1.65 tokens per step) exactly, so the opening chunk's
prefill GEMM shapes (as `DS4_QWEN_PREFILL_CHUNK` would) move a near-tie token,
not the kernels.  That single run also shows the opening chunk's cost when a
prompt's pages are already cached: the x-prompt (PLE-light) ran at 1611.4
tok/s with the opening chunk off against 1584.2 with it on, while the cold-PLE
markdown prompt ran at 1432.4 against 1467.5.  On the same-layout Uncensored
sibling artifact (owner swapped to it, same protocol): repeated prompt
1372.1 -> 1401.5 tok/s (+2.1%), markdown prompt 1402.1 ->
1476.7 (+5.3%), x-prompt 1557.0 -> 1587.5 (+2.0%).

## Open boundaries

- Remaining dense `mul_mat_q` after round 4 (0.24 s / 6.5 % of the
  6,144-row chunk): o_proj at K=6144 (D2R dispatch keeps K<=4096; deep K
  measured slower on mmq at 8192), index_qk at M=640, GDN in_a/in_b at
  M=48.  Raising the K cap for 6144 is untested.
- 303 MMQ output sanitize passes on the 6,144-row chunk (0.030 s, 0.8 %;
  was 387 / 2.5 % before D2R retired its own outputs).  Every remaining
  Qwen consumer could still guard at read as the routed path does
  (`DS4_ROUTED_OUT_GUARDED`).
- A per-column token map inside the worklist tile (reading the token-compact
  activation directly) is slower than the scatter (+22 % on the gate/up
  kernel): the contiguous 18 KB tile loads matter more than the extra 0.2 GB
  of DRAM writes.
- The opening chunk trades ~1.7 % on prompts whose pages are already cached
  (x-prompt 1611 -> 1584 tok/s, single runs) for the SSD overlap it buys on
  cold prompts; a probe of the page cache before choosing the opening rows
  (`ds4_ple_store` has the lookup, not the API) would make it conditional.
- The fused QSA attention (14.5 % at 8K, growing with depth) and the HC
  residual + norm (8.3 %, at bandwidth) are unchanged from the 2026-09-04
  rounds; the HC mix_down + inject GEMMs still read the BF16 rows twice.
