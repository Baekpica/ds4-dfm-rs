# Qwen3.8-Flash-Next prefill rounds (2026-09-07)

Follow-up to the 2026-09-06 rounds (`qwen38-prefill-2026-09-06.md`, six
rounds through `abdf25c`).  Same host (DGX Spark GB10, driver 610.43.02,
CUDA 13.3), same artifact
(`Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16`), the
production weight owner (`ds4_weight_server --backend vmm`, aligned Q8
artifacts, 278 tensors / 3.62 GiB of D2R candidates) resident throughout,
one fresh `ds4-bench` worker per run with the production worker environment
(`DS4_QWEN_PREFILL_CHUNK=8192 DS4_QWEN_PLE_CACHE_MB=2048
DS4_QWEN_PLE_WORKERS=16 DS4_MEMGOV=observe`).  Corpus: Italian prose
(`promessi_sposi.txt`).  Two shapes per variant, three runs each, medians:
a cold single-shot 8,192-token prefill and a cold single-shot 65,536-token
prefill, each followed by 32 greedy tokens.  Each round is measured against
its own kill switch on the same binary in the same hour (off / on
interleaved), so the pairs share the owner and the clock; the rounds are
not stacked onto the 2026-09-06 series as one same-hour measurement.  Base
is `main` at `ef37468`.

## What the profile said

nsys on the base binary's cold 8,192-token prefill
(`scratch/qwen-prefill-opt-20260907/base-nsys-cold8k.sqlite`, 1427.9 tok/s
under the tracer): the two prefill chunks span 5.36 s, 4.80 s of GPU busy
and 0.57 s idle.  The idle is the PLE gather at layer 1: 0.31 s + 0.06 s in
the 2,048-row opening chunk (its pages come from the SSD) and 0.15 s in the
6,144-row chunk (pages already prefetched; the host leases 105K rows through
two store locks each while the GPU waits).  Busy time by kernel over both
chunks:

| kernel | s | share of busy |
|---|---:|---:|
| fused QSA attention (12 layers, 24 launches) | 0.735 | 15.3 % |
| routed gate/up worklist pair | 0.661 | 13.8 % |
| routed expert-down worklist (main + tail) | 0.646 | 13.5 % |
| dense Q8_0 D2R (268 launches) | 0.550 | 11.5 % |
| HC residual + norm | 0.429 | 8.9 % |
| GDN recurrent (36 layers) | 0.265 | 5.5 % |
| HC fused mix | 0.199 | 4.1 % |
| moe_sum + shared gate | 0.189 | 3.9 % |
| activation quantize (773 launches, 8 per layer) | 0.187 | 3.9 % |
| weighted SwiGLU | 0.123 | 2.6 % |
| GDN conv / gated norm | 0.184 | 3.8 % |
| cuBLAS BF16/TF32 (HC projections, router) | 0.213 | 4.4 % |
| remaining `mul_mat_q` (index M=640, in_a/in_b M=48) | 0.066 | 1.4 % |
| gate/up Q8 gather, sanitize, id maps, indexer, PLE gather | 0.150 | 3.1 % |

The routed pair reads 943 MB of Q4_K weights per layer in 4.8 ms on the
6,144-row chunk (196 GB/s) and the fused down 577 MB in 4.4 ms, so both
sit near the memory floor at this chunk size; the HC residual + norm moves
~600 MB per launch at ~240 GB/s.  The fused QSA attention is 52 ms per
layer on the 6,144-row chunk (8.5 us per row at ~2,050 selected slots) and
its share grows with depth: on the 2026-09-06 cold 64K trace it was 16.7 %
of the busy time, the single largest kernel.

## Round 1: SwiGLU quantized straight into the expert-down (`1273c68`)

The routed block wrote the weighted SwiGLU as a [assignments x 640] F32
mid (157 MB per 6,144-row layer, `swiglu_weighted_kernel`) that the fused
expert-down read back through two gathered `quantize_mmq_q8_1` passes, one
for the 512-column K-quant main operand and one for the 128-column tail.
The fused entry now takes the gate / up rows and the router weights
directly (`ds4_gpu_qwen4exp_routed_down_fused_swiglu_tensor` ->
`ds4_mmq_*_moe_bounded_*_tail_swiglu`): one warp per 128-value block
computes `silu(g) * u * w` with the same expression and quantizes it with
`quantize_mmq_q8_1`'s lane mapping, shuffle order and rounding into both
operands' scale layouts (D4 / DS4), so the mid is never written or read
and three launches per layer become one.  The fixture
(`tests/test_qwen4exp_moe`) checks the expert-down output of the new entry
byte-for-byte against the SwiGLU pass + mid-based entry for Q5_K+Q5_0,
Q6_K+Q5_0 and the MTP block's Q8_0+Q8_0, non-finite gate / up values
included.  Decode widths and the paired-bank path keep the SwiGLU pass
and the mid.  Kill switch `DS4_QWEN_NO_SWIGLU_Q8_EMIT=1`.

| shape | round 1 off (same hour) | round 1 on |
|---|---:|---:|
| cold 8,192 tokens | 1432.9 (1430.5 / 1432.9 / 1433.6) | **1457.9** (1450.5 / 1457.9 / 1464.9), +1.7 % |
| cold 65,536 tokens | 1559.5 (1554.3 / 1559.5 / 1564.8) | **1581.5** (1576.2 / 1581.5 / 1581.9), +1.4 % |

Decode after the prefill unchanged (24.5 / 24.6 tok/s at 8K, 24.1 / 23.8
at 64K).

## Round 2: fused QSA attention, fewer barriers and L1-resident queries

The fused prefill QSA kernel (`qwen4exp_qsa_attention_fused_gqa12_kernel`,
one 256-thread block per (row, KV head), 32-slot tiles) was profiled with
ncu on the model-free probe (`DS4_QWEN_PROFILE_QSA=1 tests/test_qwen4exp_qsa`,
8,025 rows x 2,051 slots, owner down): issue slots 33 % busy, 0.57
eligible warps per scheduler out of 4, FMA pipes 24 %, L1/shared pipe
71 % busy.  Stalls per issue: long scoreboard 3.4 (the query rows: L1
hit rate 10 %, because the 64 KiB of keys and values a tile gathers
stream through the ~25 KiB L1 left beside two blocks' shared memory, so
the block's 12 KiB of query rows were re-fetched from L2 every tile),
MIO throttle 2.2 + short scoreboard 2.0 (shared-memory pipe), barrier
1.6.  Shared stores ran at a 4-way bank conflict (the four head groups a
lane set writes at once sit 96 words apart).

Changes, all bit-identical (same FMAs in the same order; the 8,192-token
frontier logits of `ds4-bench --dump-frontier-logits-dir` are
byte-identical between the round-1 binary and each step):

- **2a (neutral, kept as scaffolding):** the value tile is gathered into
  registers together with the key tile and parked in the key tile's
  storage once the scores are done, so the PV loop reads shared memory
  instead of gathering 1 KiB value rows on the critical path.  No
  measurable change (r1 1455.0 / 1582.0 vs 1454.6 / 1581.5 tok/s): the
  kernel is not bound by that latency.
- **2b:** four barriers per tile instead of six.  The next tile's slot
  ids are staged into a second buffer during the PV loop, and the softmax
  warps sum the eight warp partials themselves in warp order (no reduce
  pass, no reduced copy); probabilities get their own buffer.  Probe
  65.5 -> 62.5 ms.
- **2c (rejected):** key / value gathers and slot-id loads through
  `ld.global.nc.L1::no_allocate`, meant to keep the query rows
  L1-resident.  Probe 62.5 -> 84.0 ms: the 10 % L1 hit rate was already
  the query rows (12 KiB of the 76 KiB a tile pulls through L1), and the
  gathers lose more without allocation than the queries gain.
- **2d:** the score partials are XOR-swizzled by head group
  (`qsa_part_slot`), so the stores are bank-conflict-free (ncu: 424 M
  conflicts on 272 M shared stores before).  Probe 62.6 -> 61.9 ms.

Round 2 as committed (`30826d6` = 2a + 2b + 2d), probe 65.5 -> 61.9 ms,
against the round-1 binary in the same hour:

| shape | round 1 (same hour) | round 2 |
|---|---:|---:|
| cold 8,192 tokens | 1455.0 (1452.7 / 1455.0 / 1457.8) | **1474.8** (1465.8 / 1474.8 / 1479.9), +1.4 % |
| cold 65,536 tokens | 1576.9 (1570.0 / 1576.9 / 1585.8) | **1597.6** (1594.8 / 1597.6 / 1600.5), +1.3 % |

Decode after the prefill unchanged.  The kernel is still bound by the
shared-memory pipe and gather latency (ncu after 2b/2d not re-taken);
the remaining structural options are a row-group union of selections
(one gathered tile serving several rows) and a PV mapping with fewer
probability broadcasts (six heads x two dims per thread), both left open.

## Round 3: PLE gather leases a tile under one lock

The base trace's GPU idle is the layer-1 PLE gather: 0.15 s in the
6,144-row chunk although every page of that chunk was prefetched during
the opening chunk.  The gather (`ds4_qwen38_ple_cuda_gather`) leased its
105K rows one at a time, each through `cache_request_page` and
`cache_wait_ready` (two store-lock acquisitions plus a stats lock) and
released them one at a time from the stream callback (one lock each):
~1.4 us of host time per row while the GPU waited, ~145 ms per chunk.

`ds4_ple_store_acquire_ready_rows` now leases the leading rows of a
tile whose pages are already resident and ready under one lock (same LRU
touch, same reference count, no queueing), and
`ds4_ple_store_release_rows` releases a tile under one lock; only a row
whose page is still loading takes the old blocking path, from which the
walk continues as before.  Same rows, same bytes: the 8,192-token
frontier logits are byte-identical.  Kill switch
`DS4_PLE_NO_BATCH_ACQUIRE=1`.

The lease alone moved the 8K number by nothing (1478.6 / 1471.4, three
runs each): the host's non-SSD share of the gather (`row_acquire` minus
`blocked_wait`, plus `enqueue` in the `PLE gather split` line) fell from
~80 ms to ~30 ms of acquire but the enqueue side grew from ~80 to
~110 ms, because every 256-row tile still costs a pageable descriptor
copy, a launch and a lease-release host function the stream waits on,
and with the acquire out of the way that chain paces the tiles.  Tiles of
4,096 rows (`DS4_PLE_CUDA_TILE_ROWS`, 256 = old) with the lease, against
256-row tiles with the per-row lease, same hour, one binary:

| shape | round 3 off (256-row tiles, per-row lease) | round 3 on (4,096-row tiles, batched lease) |
|---|---:|---:|
| cold 8,192 tokens | 1482.9 (1475.7 / 1482.9 / 1487.9) | **1504.5** (1494.6 / 1504.5 / 1507.0), +1.5 % |
| cold 65,536 tokens | 1615.2 (1611.7 / 1615.2 / 1626.7) | **1644.3** (1643.7 / 1644.3 / 1647.3), +1.8 % |

Host-side non-SSD gather time per run: 8K 128-182 -> 61-68 ms, 64K
1,292-1,421 -> 609-625 ms; the SSD wait of the opening chunk (0.31-0.41 s
per run) is untouched and is most of the remaining run-to-run noise.
Larger tiles keep helping at 64K (single runs: 4,096 / 16,384 / 65,536
rows 1644 / 1650 / 1660 tok/s; enqueue 580 / 411 / 174 ms) and are flat
at 8K (1504 / 1501 / 1506); the committed default is 16,384 rows
(`6e036c4`), which pins at most 64 MiB of pages per tile while two banks
gather alternately into one 2 GiB cache.  Decode after the prefill
unchanged.

## Cumulative and the production shape

Cold single-shot `ds4-bench` prefill, `main` `ef37468` against the final
binary (`6e036c4`), interleaved in the same hour, three runs each:

| shape | `ef37468` | `6e036c4` |
|---|---:|---:|
| cold 8,192 tokens | 1429.9 (1420.1 / 1429.9 / 1434.7) | **1504.6** (1493.9 / 1504.6 / 1509.8), +5.2 % |
| cold 65,536 tokens | 1557.2 (1553.7 / 1557.2 / 1557.2) | **1648.4** (1647.9 / 1648.4 / 1650.4), +5.9 % |
| cold 196,608 tokens (one run) | 1439.9 | **1513.0, +5.1 %** |

Decode after the prefill unchanged (24.5 / 24.5 tok/s at 8K, 23.4 / 24.0
at 64K, noise).  The host's non-SSD share of the PLE gather per run went
195-231 -> 39-44 ms at 8K and 1.88-1.94 s -> 0.44 s at 64K; the opening
chunk's SSD wait (0.34-0.40 s per run) is unchanged.  Every round is
bit-identical to the kernels it replaces: the 8,192-token frontier logits
of the final binary are byte-identical to the base binary's.

Production server shape (the canonical two-bank command: 196,608
configured context, `DS4_QWEN_BATCH=1`, 8,192-token chunks, 2048 MiB PLE
cache, 16 page workers, `--mtp-draft 2`; three fresh workers with fresh
disk-KV directories per binary, `thinking` disabled, non-streaming, the
API's `timings.prefill_tok_s`), the 2026-09-02 card prompts, medians of
three:

| binary | corpus3 8,259 tok (repeated passage) | corpus.txt 7,937 tok (markdown, cold PLE) | x-prompt 8,036 tok | corpus3 + 256 greedy tokens: prefill / decode / ms per MTP step |
|---|---:|---:|---:|---:|
| `ef37468` | 1481.7 (1480.3 / 1481.7 / 1485.1) | 1555.9 (1554.7 / 1555.9 / 1561.3) | 1677.7 (1672.8 / 1677.7 / 1683.6) | 1552.4 / 30.3 / 53.8 |
| `6e036c4` | **1559.9** (1556.8 / 1559.9 / 1560.7), +5.3 % | **1652.6** (1645.4 / 1652.6 / 1655.3), +6.2 % | **1761.5** (1759.2 / 1761.5 / 1762.2), +5.0 % | 1647.2 / 30.4 / 53.6 |

The 175-token greedy continuation of the chat prompt (it stops at the
model's end token) is byte-identical between the two binaries (same
`text_sha`), as is its MTP acceptance (1.63 tokens per step), so the
decode figures differ by run noise only.  The same protocol on the
same-layout Uncensored sibling artifact (owner swapped to it): repeated prompt
1482.5 -> **1557.4 tok/s** (+5.1 %), markdown prompt 1546.3 -> **1651.4**
(+6.8 %), x-prompt 1676.8 -> **1761.8** (+5.1 %); its 256-token greedy
continuation is byte-identical before and after (30.6 / 30.6 tok/s, 1.63
tokens per step).

The published 2K-64K incremental sweep (same protocol as `abdf25c`:
`--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
--mtp-draft 2`, one warm session, aligned-Q8 owner, 2 GiB PLE cache) on
`6e036c4`: mean prefill **1,248.8 tok/s** (was 1,235.9), mean decode
**28.4 tok/s** (was 28.4); 2K 1,050.1 / 64K 1,182.9 tok/s prefill (was
1,032.6 / 1,198.5), no MTP quench.  The warm 2,048-token steps see little
of these rounds (no opening chunk, one 2K PLE gather per step, short QSA
rows), which is why the sweep moves ~1 % while the cold 8K-64K shapes move
5-6 %.  Graph: `docs/qwen38-long-context-throughput.{svg,png}`.

## Open boundaries

- The fused QSA attention is still bound by the shared-memory pipe
  (probabilities broadcast to 256 threads are about half of its shared
  load wavefronts) and gather latency; a row-group union of selections
  and a six-heads-by-two-dims PV mapping are the untested structural
  options.  `ld.global.nc.L1::no_allocate` gathers were measured and
  rejected.
- The opening chunk's SSD wait (~0.35 s per cold prompt) is the largest
  idle left at 8K; in `ds4-bench` it sits behind ~0.37 s of session graph
  allocation that the production server does once at warm-up, so a
  gather issued before that allocation would help only the bench.
- The PLE gather's per-tile chain (pageable descriptor copy, launch,
  lease-release host function) still costs ~0.4 s of host time per 64K
  run at 16,384-row tiles; mapped pinned descriptors would remove the copy.
- The routed pair and expert-down worklists sit near the weight-bandwidth
  floor at 6,144-row chunks (196 / 131 GB/s effective) and the HC residual +
  norm at ~240 GB/s; `moe_sum` re-reads the 629 MB assignment-major
  expert-down output per layer, inherent to summing across expert tiles
  deterministically.
