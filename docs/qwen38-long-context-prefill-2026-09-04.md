# Qwen3.8-Flash-Next long-context prefill (2026-09-04)

Follow-up to the tiled QSA block scorer (`9420234`).  Same host (GB10,
driver 610.43.02, CUDA 13.3), same artifact
(`Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16`, main),
owner `ds4_weight_server --backend vmm --repack-q8-aligned`, worker env
`DS4_QWEN_PREFILL_CHUNK=8192 DS4_QWEN_PLE_CACHE_MB=1024
DS4_QWEN_PLE_WORKERS=16 DS4_PLE_LATENCY_STATS=1 DS4_MEMGOV=observe`, one
fresh `ds4-bench` worker per run.  Corpus: Italian prose
(`promessi_sposi.txt`, natural n-gram statistics; PLE behaviour is
corpus-dependent, see the 2026-09-02 report).  Kernel and store
statistics quoted per 8,192-token prefill chunk.

## What the profile said

An nsys trace of the 2K..64K sweep at `9420234` put the GPU idle for 3.8 s
of an 11.4 s 0-8K chunk and 12.3 s of a 20.8 s 56-64K chunk, all of it in
the gaps between `qwen38_ple_gather_kernel` tiles: the host was waiting for
SSD-PLE pages.  The store statistics of a cold 65,536-token prefill
confirmed it without the profiler: 59.5 s of a 123.7 s run inside the PLE
gather.  Kernel time grew only 7.5 -> 8.5 s per chunk over the same depth
(block scorer 6 -> 50 ms per QSA layer, attention reduce 34 -> 64 ms,
attention scores 98 -> 117 ms; GDN recurrent flat at 34.6 ms per layer).

## Changes

1. **PLE page cache eviction was first-fit, not LRU** (`ds4_ple.c`,
   `cache_request_page`): the victim scan stopped at the first evictable
   way, so a page a prefetch had just brought into a set was evicted by the
   next request for that set.  A chunk re-read up to a third of its pages
   at gather time, and more page workers made it worse (64 workers:
   339 tok/s at 32K).  Now the least recently touched evictable way is
   chosen.  Associativity 4 -> 16 ways (4-way sets overflowed ~7 % of the
   time for an 8K chunk in 1 GiB).
2. **Next-chunk lookahead** (`ds4.c`, `qwen4exp_ple_lookahead`): the
   prefill loops (serial session and the banked scheduler) pass the
   following chunk's tokens; after this chunk's rows are gathered at layer
   1, their n-gram rows are hashed from the post-chunk state and queued, so
   the sidecar reads (~90K IOPS bursts of ~1.5 s) overlap the remaining
   decoder layers.  The first chunk of a prompt still waits (~1.5 s for
   8K rows).  Kill switch `DS4_QWEN_PLE_NO_LOOKAHEAD=1`.  Cache sizing:
   one 8K chunk is ~553 MiB of pages; one prefill stream needs 1024 MiB,
   two alternating banks 2048 MiB (README).
3. **Targeted reader wake-ups** (`ds4_ple.c`): page workers broadcast the
   state condition only for a page a reader is sleeping on (or when a
   request waits for a set to free up), instead of on every completed read.
4. **GDN recurrent state in registers** (`ds4_cuda.cu`,
   `qwen4exp_gdn_recurrent_kernel`, 128-wide path): each thread keeps its
   32 state rows for the whole chunk instead of two loads and a store per
   element per token.  Explicit `fmaf` keeps the compiler's contraction of
   the old form, so outputs and stored state are bit-identical (fixture
   digest `3298b52068fc78d9 / e422b571956a7b0e` before and after).  The
   move also exposed a benign shared-memory race between a token's output
   read of `partial[]` and the next token's Q/K norm partials; those now
   use their own words.  Model-free profile (8,025 rows): 33.1 -> 16.7 ms.
5. **Fused QSA prefill attention** (`ds4_cuda.cu`,
   `qwen4exp_qsa_attention_fused_gqa12_kernel`): one block per (row, KV
   head) walks the selected slots in 32-token tiles with the keys in shared
   memory, an online softmax and register-resident value accumulation, so
   the rows x heads x slots score scratch (1.6 GB per 8K bank, written and
   read three times) is gone from prefill; the scratch now covers decode
   widths only (graph measured 9.86 -> 8.33 GiB at 64K).  Scores differ
   from the per-slot kernel by fp32 reordering only (fixture max 1.3e-7).
   Model-free profile (8,025 rows, 2,051 slots): 104.1 + 63.6 -> 67.2 ms.
   Kill switch `DS4_QWEN_QSA_NO_FUSED=1`; decode widths (<= 8 rows) keep
   the split reduce.
6. Store statistics now report blocked row acquisitions, their wait time,
   the gather enqueue share and a per-second read timeline
   (`DS4_PLE_LATENCY_STATS=1`).

## Measurements

Cold single-shot prefill (`ds4-bench --ctx-start N --ctx-max N`, 32 greedy
tokens after), tok/s, one run each unless noted:

| prefill | base `9420234` | this round | PLE gather wait (base -> new) |
|---|---|---|---|
| 32,768 tokens | 597 (lookahead only, first-fit cache) | **1180** | 25.8 s -> 2.4 s |
| 65,536 tokens | 530 / 572 (two runs) | **1163 / 1169** (two runs) | 59.5 s -> 3.8 s |

Store sensitivity at 32K before the eviction fix (lookahead on, 16-way,
first-fit): 16 workers 597 tok/s, 32 workers 539, 64 workers 339, 4 workers
786, 2048 MiB cache 978 - faster page completion meant more READY victims
for the first-fit scan.  With true LRU the 16-worker, 1024 MiB shape gives
the 1180 above and the reads run in ~1.5 s bursts of 80-94K IOPS per chunk
(store timeline) while the GPU computes.

Depth sweep (`--ctx-start 2048 --ctx-max 65536 --step-incr 2048`, each
frontier a 2,048-token incremental prefill on a warm session, then 128
greedy tokens), same corpus, both binaries in the same hour:

| context | base tok/s | new tok/s | gain | base decode | new decode |
|---|---|---|---|---|---|
| 2,048 | 772.0 | 931.0 | +20.6 % | 24.73 | 25.34 |
| 4,096 | 839.4 | 1062.0 | +26.5 % | 24.90 | 25.13 |
| 8,192 | 816.5 | 1060.4 | +29.9 % | 24.65 | 24.86 |
| 16,384 | 780.0 | 1065.0 | +36.5 % | 24.51 | 24.90 |
| 24,576 | 683.6 | 1060.9 | +55.2 % | 24.60 | 24.51 |
| 32,768 | 697.4 | 1024.9 | +47.0 % | 24.44 | 24.40 |
| 40,960 | 676.3 | 1003.8 | +48.4 % | 24.35 | 24.29 |
| 49,152 | 688.4 | 994.2 | +44.4 % | 24.06 | 24.41 |
| 57,344 | 698.3 | 988.0 | +41.5 % | 24.19 | 24.09 |
| 65,536 | 675.2 | 974.1 | +44.3 % | 24.12 | 24.02 |

Means: 2K-16K 799.8 -> 1042.3 tok/s, 32K-64K 684.4 -> 1000.1 tok/s
(+46 %); the 8K -> 64K slope shrinks from -17 % to -8 %.  Decode mean
24.44 -> 24.51 tok/s (unchanged; `--mtp-draft` off in the bench).  A
2,048-token step cannot use the lookahead (the next step's tokens are not
known), so this table isolates the eviction fix, the wake-ups and the two
kernels; the single-shot rows above add the lookahead.

nsys of the new binary on the cold 64K prefill (1156.6 tok/s under the
profiler; `scratch/qwen-longctx-opt-20260904-r2/nsys-ABC2-64k*`), per
8,192-token chunk:

| chunk | wall | GPU busy | GPU idle | top kernels |
|---|---|---|---|---|
| 0-8K | 7.39 s | 6.06 s | 1.33 s (1.07 s first-chunk PLE reads) | MMQ 0.88, fused QSA 0.75, GDN 0.62 |
| 24-32K | 6.81 s | 6.51 s | 0.30 s | fused QSA 0.96, MMQ 0.88, GDN 0.63 |
| 56-64K | ~7.3 s | 6.99 s | 0.28 s | fused QSA 1.06, MMQ 0.87, block scorer 0.74, GDN 0.63 |

Against the base trace of the same depth: idle 3.8 / 12.3 s -> 1.3 / 0.3 s,
QSA scores + reduce 1.19 + 0.41 / 1.37 + 0.65 s -> fused 0.75 / 1.06 s, GDN
1.24 / 1.25 s -> 0.62 / 0.63 s.  The remaining depth slope (6.06 -> 6.99 s
busy) is the block scorer (0.07 -> 0.74 s, linear in visible blocks) and the
fused attention's K/V gather (0.75 -> 1.06 s).

## Open boundaries

- The first chunk of a prompt still waits for its own pages (~1.5 s per
  8,192 tokens at ~90K IOPS); hiding it needs the reads to start before
  the graph is entered (at admission), or a smaller first chunk.
- The lookahead holds two chunks' pages in cache; with 1024 MiB the
  previous chunk's cross-chunk n-gram hits are evicted earlier (+16 % reads
  at 64K).  Two alternating banks need 2048 MiB.
- The fused QSA kernel is still ~3x above its FMA bound on the model-free
  probe; the value gather (one row per thread iteration) and the K tile
  gather are the remaining costs, and at 64K the K/V traffic itself
  (~68 GB per layer, partly L2-missing) bounds it.  Sharing K/V tiles
  across a group of consecutive query rows (union of their selections)
  would cut that traffic but depends on selection overlap, unmeasured.
- Decode paths are untouched: the split reduce serves rows <= 8, the GDN
  kernel is bit-identical, and `test_qwen4exp_verify` (48 steps, 0 logit /
  0 state mismatches, via the owner manifest) and `test_qwen4exp_batch`
  pass on the final objects.
- `tests/cuda_long_context_smoke` (DeepSeek substrate regression) is flaky
  on this host while the 80 GiB weight owner is resident: its overflow-path
  `cudaMemcpy` fails with "invalid argument" in about half the runs, at
  `9420234` (3/6 passes) as much as here (2/6, interleaved runs, same
  conditions).  Not a regression of this round; retry with the owner down.

