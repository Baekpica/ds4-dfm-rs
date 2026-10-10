# What the qwen38 serving path teaches the ds41 route

Owner question (2026-10-10): Qwen3.8-Flash-Next is already served by this
repo (`ds4_ple.c`, `cuda/qwen38_ple.cu`, the qwen38 family) and it is fast;
study it and take what makes DeepSeek V4.1 fast the same way.

This document is the detailed reference: every learnable piece with its
exact source in this repo or in the engine (`/data/YoungAi`, commit
3946dbc), the ds41 counterpart it would touch, and the test that proves it
helps. The plan and gates are in `docs/deepseek41-port-plan.md` 6.20-6.21.

Honest frame, first: the qwen38 route's headline speed is mostly its
architecture (6B active MoE, GDN linear attention, QSA sparse attention,
HF model card + arXiv 2608.30320), which cannot be bolted onto a trained
V4.1. What transfers is the serving machinery around the off-accelerator
n-gram table, plus two MTP optimizations the qwen38 campaign adopted
beyond its engine and proved safe. Absolute numbers across the two
families are not comparable (different contexts and active parameters);
the comparable target for ds41 is port == its own engine.

## 1. The PLE pipeline (the n-gram table on SSD, served fast)

Qwen3.8-Flash-Next keeps 51B n-gram embedding parameters off the
accelerator ("N-gram Embedding ... more amenable to offloading", HF model
card; "deterministic addressing enables host-memory offloading and
asynchronous prefetching", arXiv 2608.30320 2.3). This repo serves the
table from SSD through `ds4_ple.c` + `cuda/qwen38_ple.cu`:

- Bounded set-associative 4 KiB page cache, 2 GiB default
  (`crates/ds4-core/src/serving_host.rs:36,39-41`; the cache contract
  `ds4_ple.h:141-147`; env `DS4_QWEN_PLE_CACHE_MB`).
- A loader worker pool (`worker_count`; the campaign used 16,
  `docs/qwen38-ple-fp8.md` reproduce block, `DS4_QWEN_PLE_WORKERS`).
- Speculative prefetch: "Queue all pages touched by the supplied global
  row ids and return without waiting for I/O" (`ds4_ple.h:161-169`).
- Ready-row leases: the gather consumes only rows whose pages are
  resident, and a tile ends early at a row whose page is still loading —
  the GPU never blocks on the SSD (`ds4_ple.h:205-221`,
  `cuda/qwen38_ple.cu:27-35`; 16384-row tiles amortize the descriptor
  copies).
- Stats that made the tuning honest: `page_requests`, `cache_hits`,
  `cache_inflight_hits`, `cache_misses`, `cache_evictions`
  (`ds4_ple.h:83-87`), completed page reads per second (`:99-103`),
  `DS4_PLE_LATENCY_STATS=1`.
- Measured state (`docs/qwen38-perf-2026-09-14.md`): ~1420-1550 t/s
  prefill and ~28-29.5 t/s decode at 8K/64K with MTP draft 2.

## 2. The principle, and where ds41 stands

The principle is one sentence: never let the GPU wait on the SSD. The
engine already implements it for the V4.1 engram, with the same shape as
the PLE lease:

- Prefetch at forward start, overlapped with the first layers' GPU work:
  `core_v41_forward.c:390` ("engram 行读盘与前面几层的 GPU 算重叠") ->
  `v41_engram_prefetch` (`core_v41_engram.c:311-341`): submit round 0;
  rounds larger than the queue go to a dedicated fetch thread
  (`:119-150`, `:325-341`).
- Per-layer wait that immediately submits the next layer's round
  (`v41_ejob_wait`, `core_v41_engram.c:344-379`).
- Graph arm/serve with a pinned flag per layer so captured decode waits
  on the host's read completion, not the other way around (`:259-281`).

The port has NOT ported that overlap. Unit P (2026-10-10, commit b7f5ad8)
ported the batched reads themselves (`crates/ds4-core/src/uring.rs`,
mirroring `core_v41_ering.c`) but keeps the block-synchronous shape: the
feed's `prepare` runs every layer's round to completion before the block's
GPU work starts (`crates/ds4-core/src/v41_run.rs` `prepare` ->
`prepare_round`; the feed contract `ds41_forward.h:41-53`, called at
`ds4_ds41_forward.inc:1038`, consumed per layer at `:440-466`). At 354
tokens that cost 1.3 s vs the engine's 1.247 s (gate, 2026-10-10); the
remaining gap is exactly this overlap, and it grows with block size.

## 3. What does not transfer

- The PLE page cache. The engine deliberately rejected page caching for
  the engram: O_DIRECT first, plain read + `FADV_RANDOM` as the fallback,
  because kernel readahead churned the model's mapped pages and made
  temperature-0 runs non-reproducible (`core_v41_engram.c:32-58`, the
  2026-09-15 conviction). Row hit rate is near zero by design (random row
  ids). Do not add a cache to the engram.
- The PLE tile/descriptor machinery: a PLE row is 160 values (320 B
  BF16 / 160 B FP8, `docs/qwen38-ple-fp8.md` storage table); an engram
  position reads 24 rows of 264 B across two planes. Different shape,
  different batching; the io_uring round is the right primitive for ours.
- GDN, QSA, Gated Residual: architecture of a different trained model.

## 4. Q-1: the MTP compact draft vocabulary

What the qwen38 route does (`ds4.c`): the draft scores a shortlist — the
first 65536 BPE IDs plus non-normal token types plus IDs observed by the
shifted input stream — while the target output and verification stay full
vocabulary. State: `mtp_vocab_gpu` / `mtp_vocab_ids` / `mtp_vocab_seen`
(`ds4.c:26778-26782`), reset with the graph and on bank copy/checkpoint
restore (`qwen_mtp_vocab_reset`, `ds4.c:26843-26845`); the kill switch is
`DS4_QWEN_MTP_FULL_VOCAB=1` (`ds4.c:27368-27373`).

Evidence it is safe (`docs/qwen38-perf-2026-09-14.md`, "MTP output
vocabulary"): median decode +3.04% at 8K and +3.87% at 64K, prefill
unchanged, every pair matched all 248320 prefill logits and all 128
generated token IDs exactly; acceptance moved 53.66% -> 51.81% while mean
cycle time fell 52.93 -> 50.67 ms — report acceptance and speed together.
GPU argmax alone was rejected (+0.47%, below the 2% bar).

The ds41 counterpart: the draft's head is full-vocabulary today —
`mk_bias` [V] (`ds4_ds41_draft.inc:46,165`), logits [cap x DS4_N_VOCAB]
(`:155`), the engine's own bias write over `DS4_N_VOCAB`
(`core_v41_draft.c:277`; the port cites the engine function by function,
`ds4_ds41_draft.inc:3`). A ds41 shortlist would shrink only the draft's
scoring head (and its argmax), exactly as qwen38's does; the verify and
the target output stay full-vocabulary.

Doctrinal status: a NAMED deviation. The engine has no shortlist, so the
`[dspark]` trace (draft ids per round) will differ from the engine's by
design; the generate gate's spec leg must treat that diff as
informational and gate on emitted ids instead. Kill switch:
`DS4_V41_MTP_FULL_VOCAB=1` (naming mirrors the qwen38 pair).

## 5. Q-2: the MTP prefix-cache preparation

What the qwen38 route does: an MTP prefix consumes shifted target hidden
states; its attention output and MoE result are discarded, so only the QSA
raw/pooled index history and K/V are carried forward. The cache-only
preparation skips query projection, query normalization, RoPE, selection,
attention output, output projection and the following HC/MoE when no draft
logits are requested (`ds4.c:27536` area, `DS4_QWEN_MTP_FULL_PREFIX=1`
restores the full computation).

Evidence (`docs/qwen38-perf-2026-09-14.md`, "MTP prefix cache
preparation"): median prefill +0.90% at 8K and +2.55% at 64K, decode
+0.84%/+0.27%, every pair matched all frontier logits and all 128 ids
exactly; a diagnostic prefix trace shrank from 78 kernels / 136.47 ms to
27 kernels / 21.99 ms of GPU time.

The ds41 counterpart: the port's draft arm and prefix path
(`ds4_ds41_draft.inc:125-198`, citing `core_v41_draft.c:94-175`). The
change is the same shape: when no draft logits are requested, skip the
parts of the draft prefix whose outputs are discarded, keep the carried
state (the verify's `spec_q` path, `ds4_ds41_draft.inc:58-59`) intact.
Kill switch: `DS4_V41_MTP_FULL_PREFIX=1`.

## 6. The tests (make sure it helps)

Protocol for every candidate (adopted from the qwen38 campaign,
`docs/qwen38-perf-2026-09-14.md` "Protocol"): fresh processes, alternating
A/B pairs, graphics-clock medians recorded, adoption bar >= 2% end to end
on the target phase with no repeatable > 2% loss on the other, exact
ids/logits parity required.

Parity gates that must stay green, on the Spark, source-built engine
oracle, canonical boot:

- `tests/ds41_sampling_gate.sh` (cases A-D byte-identical),
- `tests/ds41_wire_gate.sh` (anthropic + responses + smoke),
- `tests/ds41_chat_gate.sh` (354-token prompt + generated ids + response
  message),
- `tests/ds41_generate_gate.sh` with `DS41_FEED_DIGEST` (the feed bytes on
  device must equal the host digest).

Per candidate:

- P2 (the overlap): all four gates; prefill not worse than 1.3 s at 354
  tokens; the 835-token long prompt (`tests/ds41-long-prompt.txt`) as the
  big-block case (where the engine's fetch-thread path matters); the
  port's host-gaps line must show the engram prep out of the critical
  path. No kill switch needed (it mirrors the engine) — the fallback is
  the current synchronous path when the ring is unavailable.
- Q-1 (draft shortlist): the emitted-id gates above (sampling, wire,
  chat); the generate gate's spec leg with the `[dspark]` trace diff
  marked informational; a decode A/B at the chat gate's context (and one
  longer context) with >= 2% median gain; `DS4_V41_MTP_FULL_VOCAB=1`
  restores the engine-identical draft for control runs.
- Q-2 (prefix cache): same emitted-id gates; a prefill A/B (354-token and
  835-token prompts) with >= 2% median gain; `DS4_V41_MTP_FULL_PREFIX=1`
  as the control.

Rollback for all three: each lands as its own commit; Q-1/Q-2 default to
the engine behavior until their A/B passes, then flip defaults with the
kill switch retained; reverting the commit restores the engine path.

## 7. Order

1. P2 — the engine's overlap structure (no doctrinal change; the qwen38
   PLE lease is the same principle).
2. Q-1 — the draft shortlist (owner-gated, measured).
3. Q-2 — the prefix cache (owner-gated, measured).

Plan sections: `docs/deepseek41-port-plan.md` 6.21 (units and gates) and
6.20 (unit P's evidence, which these build on).
