# Qwen3.8-Flash-Next benchmark reference

This reference consolidates the recorded benchmark numbers, the source-declared
limits and the candidate protocol for Qwen3.8-Flash-Next (qwen4exp; the qwen35
sibling shares its self-parity harnesses). Future sessions use it as an
optimization target reference for the host and port. It is not live state: it
retains the recorded host, artifact, binary hashes and workload, and it does
not qualify a new build.

Source of truth for every number and bound below is the repo record it calls
out, read first-hand. Do not re-derive a fresh session; read the named file or
install the artifact and queue the miss.

## Artifact and host identity

- Base main: Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/MQ-Q5-SSD-PLE-BF16.
  The FP8 PLE sidecar is an E4M3FN companion with the same manifest and
  checksum list.
- Uncensored sibling loads as its own main GGUF after stopping the Base owner;
  it reuses the same local FP8 sidecar files.
- DGX Spark GB10, driver 610.43.02, CUDA 13.3, sm_121a; user-managed graphics
  clock range 300-2200 MHz; measured graphics-clock medians 2190 MHz
  (docs/qwen38-perf-2026-09-14.md:10-13).
- Owner: ds4_weight_server --backend vmm --repack-q8-aligned, one resident
  owner; each measurement uses a fresh guarded worker
  (docs/qwen38-long-context-prefill-2026-09-04.md:3-8).
- Corpus: speed-bench/promessi_sposi.txt, SHA-256
  f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f. PLE
  behaviour is corpus-dependent (docs/qwen38-ple-fp8.md:90-91).

## Source-declared limits

Read from code, not inferred from a doc summary:

| Bound or switch | Recorded value | Source |
|---|---|---|
| Runtime context cap | 1,048,576 (rope_orig_ctx 262144 x Yarn factor 4; a session above it is refused) | crates/ds4-core/src/serving.rs:792-810; crates/ds4-core/src/shape.rs:935-990; ds4.c:1637-1649; ds4.c:73199 |
| Qualified context / banks / prompt | 262144 / 2 / none | crates/ds4-core/src/serving.rs:807-809 |
| Capabilities (banks, reuse, disk, snapshot, MTP) | persistent/qualified; partial/qualified; qualified; qualified; Embedded on the bank lane/qualified | crates/ds4-core/src/serving.rs:793-803 |
| Scheduler chunk fence and candidates | 8192; 256, 512, 1024, 2048, 4096, 8192 | crates/ds4-core/src/serving.rs:18-23 |
| DS4_QWEN_PREFILL_CHUNK | default 256; bounded 1..16384; clamped to ctx | ds4.c:73138-73143 |
| DS4_QWEN_PREFILL_OPENING | default 2048; bounded 0..16384 (0 = cap) | ds4.c:37228-37237 |
| DS4_QWEN_PLE_CACHE_MB | default 2048; only 512, 1024, 2048 accepted, others ignored and forced 2048 | ds4.c:37267-37275 |
| DS4_QWEN_PLE_WORKERS | default 32; bounded 1..64; the campaign's measured runs used 16 | ds4.c:37276-37277; docs/qwen38-ple-fp8.md; docs/qwen38-perf-2026-09-14.md:12-13 |
| Recorded combined serving ceiling | 262144 context, two banks, separate empty 32 GiB disk-KV directories; a larger retrieval prompt failed under the 4 GiB/PSI floor and is retained | docs/qwen38-perf-2026-09-14.md:180-212 |

The runtime context cap is an admission bound, not a qualification claim; the
generated table says so explicitly (docs/serving-capabilities.md:40). Consult
the release ledger and family contracts for qualification evidence.

## Benchmark protocol

Keep these shapes when a candidate is measured:

- Card sweep: one warm session, 32 frontiers, a 2,048-token incremental
  prefill and 128 greedy outputs per frontier; MTP draft 2; 2 GiB PLE cache;
  16 page workers; 8192-token chunk cap. Prefill includes embedded-MTP prefix
  maintenance; snapshot save/restore runs outside the timers; NVTX disabled
  and no Nsight collector runs. Existing PLE latency counters stay on. The
  measured implementation is `84d183c` with benchmark SHA-256
  `423683df045ad27f12a19b1ca53b78c622f8558559a1805549a05984a6782046`; the
  later `2b7aee7` carries the KV format guard and brace/comment fixes, and
  final live checks use that commit (docs/qwen38-ple-fp8.md:83-99).
- Long-context rounds: cold single-shot `--ctx-start N --ctx-max N` with 32
  greedy tokens after; and an incremental `--ctx-start 2048 --ctx-max 65536
  --step-incr 2048 --gen-tokens 128 --mtp-draft 2` sweep on one warm session.
  Two binary shapes per variant, three interleaved runs each, medians
  (docs/qwen38-long-context-prefill-2026-09-04.md:71-98,
  docs/qwen38-prefill-2026-09-06.md:195-205).
- 2026-09-14 candidate protocol: independent 8192/65536 frontiers, 128
  generated tokens, three alternating A/B pairs per frontier, graphs-clock
  medians recorded, adoption threshold >=2% end-to-end on the target phase
  with no repeatable >2% loss on the other principal metric
  (docs/qwen38-perf-2026-09-14.md:18-21).
- Production server shape: the canonical two-bank command (196,608 configured
  context, DS4_QWEN_BATCH=1, 8192-token chunks, 2048 MiB PLE cache, 16 page
  workers, --mtp-draft 2), fresh disk-KV directories per binary, thinking
  disabled, non-streaming, the API's timings.prefill_tok_s
  (docs/qwen38-prefill-2026-09-06.md:210-227).

## Recorded numbers

Paired FP8 vs BF16 PLE card sweep, 2026-09-08 (docs/qwen38-ple-fp8.md:110-111,
benchmarks/qwen-ple-fp8-2026-09-08/summary.json):

| Artifact | PLE format | Prefill tok/s | Decode tok/s | Change |
|---|---|---|---|---|
| Base Q5 | BF16 | 1,245.1 | 28.59 | - |
| Base Q5 | FP8 | 1,323.1 | 28.93 | +6.3% / +1.2% |
| Uncensored Q5 | BF16 | 1,247.2 | 28.93 | - |
| Uncensored Q5 | FP8 | 1,310.8 | 28.96 | +5.1% / +0.1% |

Each cell is the median of three arithmetic means over the 32 frontiers; CSV
metrics are prefill_tps and gen_tps, matching the original card plot. For the
base model, median whole-process PLE row-acquisition time decreased in the
FP8 run (docs/qwen38-ple-fp8.md:128).

Optimization rounds:

- 2026-09-04 long-context: cold single-shot 32K 597 -> 1180 tok/s; 64K
  530/572 -> 1163/1169; PLE gather wait 25.8 -> 2.4 s at 32K, 59.5 -> 3.8 s
  at 64K. Incremental 2K-16K mean 799.8 -> 1042.3, 32K-64K 684.4 -> 1000.1
  (+46%); decode mean 24.44 -> 24.51, unchanged with MTP off
  (docs/qwen38-long-context-prefill-2026-09-04.md:71-98).
- 2026-09-06 prefill rounds: cold 8K 1214.8 -> 1362.1 (+12.1%); cold 64K
  1382.7 -> 1439.7 (+4.1%); 196,608 tokens 1297.5 -> 1325.6 (+2.2%) under
  one run each. Published incremental sweep mean prefill 1,235.9 (was
  1,163.5) and mean decode 28.4 (was 28.0); 2K 1,032.6 and 64K 1,198.5
  tok/s. Production server shape medians on three card prompts: 1363.3 ->
  1394.2 (+2.3%), markdown 1406.1 -> 1467.5 (+4.4%), x-prompt 1570.0 ->
  1584.2 (+0.9%); the 256-token continuation decode 31.3 -> 29.8 tok/s with
  MTP acceptance 1.65 -> 1.60 tokens per step
  (docs/qwen38-prefill-2026-09-06.md:195-227).
- 2026-09-14 perf campaign: 8K control about 1427.3 prefill / 27.8 decode;
  64K about 1515 / 28.4. Adopted MTP compact draft vocabulary: median decode
  +3.04% at 8K and +3.87% at 64K, with prefill changes -0.10%/-0.14% and
  acceptance 53.66% -> 51.81% while mean cycle time 52.93 -> 50.67 ms. MTP
  prefix-cache preparation: median prefill +0.90% at 8K and +2.55% at 64K,
  decode +0.84%/+0.27%. A diagnostic prefix trace shrank 78 kernels /
  136.47 ms -> 27 kernels / 21.99 ms. Uncensored combined paths: prefill
  +0.71%, decode +3.34%, acceptance 44.32% -> 42.70%
  (docs/qwen38-perf-2026-09-14.md:38-49, 147-157, 222-233). Rejected in the
  same campaign, each before an adoption sweep: GPU-argmax alone (+0.47%),
  QSA Tensor Core (-14% prefill), PLE descriptor pool (decode -3.14%/-7.09%),
  QSA PV 6x2 (+0.07%/+0.08%), selected-K/V pair reuse and HC shared
  mix_down/inject (:105-108, 39-48, 117-132, 269-330).

## Optimization target

The comparable target for the ds41 port is port == its own engine, not the
Qwen headline rate: the Qwen route's headline speed is mostly its architecture
(6B active MoE, GDN linear attention, QSA sparse attention); it cannot be
bolted onto a trained V4.1, and absolute numbers across the two families are
not comparable because contexts and active parameters differ
(docs/ds41-qwen38-serve-lessons.md:12-20). Transfer, with the campaign's
adoption bar and parity gates, only the serving machinery around the
off-accelerator n-gram table plus two MTP optimizations.

## Transferable pieces

1. The PLE ready-row lease principle, "never let the GPU wait on the SSD".
   The engine already implements that overlap for the V4.1 engram: prefetch
   at forward start, per-layer wait that immediately submits the next layer's
   round, a dedicated fetch thread for rounds larger than the queue, and a
   pinned flag per layer so captured decode waits on host read completion
   (`core_v41_forward.c:390`; `core_v41_engram.c:311-341`, `:344-379`,
   `:119-150`, `:259-281`). The port has not ported that overlap: its feed's
   prepare runs every layer's round to completion before the block's GPU work
   starts (`crates/ds4-core/src/v41_run.rs` prepare -> prepare_round, the
   feed contract `ds41_forward.h:41-53`, called at `ds4_ds41_forward.inc:1038`,
   consumed per layer at `:440-466`). At 354 tokens the synchronous shape
   costs 1.3 s against the engine's 1.247 s, and the gap grows with block size
   (docs/ds41-qwen38-serve-lessons.md:21-72). The Qwen lease itself is the
   same principle: the gather consumes only rows whose pages are resident and
   a tile ends early at a loading row (`ds4_ple.h:205-221`,
   `cuda/qwen38_ple.cu:27-35`).
2. The MTP compact draft vocabulary (Q-1). The qwen38 draft scores a
   shortlist: the first 65536 BPE IDs plus non-normal token types plus IDs
   observed by its shifted input stream; target output and verification stay
   full vocabulary. State `mtp_vocab_gpu` / `mtp_vocab_ids` / `mtp_vocab_seen`
   (`ds4.c:26778-26782`), reset with the graph and on bank copy/checkpoint
   restore (`qwen_mtp_vocab_reset`, `ds4.c:26843-26845`), kill switch
   `DS4_QWEN_MTP_FULL_VOCAB` (`ds4.c:27368-27373`, and the CPU-selection
   control `DS4_QWEN_MTP_CPU_ARGMAX`). The ds41 counterpart: the port's draft
   head is full-vocabulary (`mk_bias` [V], `ds4_ds41_draft.inc:46,165`; logits
   [cap x DS4_N_VOCAB], `:155`; the engine's own bias write over
   `DS4_N_VOCAB`, `core_v41_draft.c:277`). A ds41 shortlist shrinks only the
   draft's scoring head and its argmax; the verify and target output stay
   full-vocabulary. Doctrinal status: a named deviation, because the engine
   has no shortlist, so the [dspark] trace will differ by design; the generate
   gate's spec leg treats that diff as informational and gates on emitted ids
   instead. Kill switch `DS4_V41_MTP_FULL_VOCAB` (mirroring the qwen38 pair).
   Report acceptance and speed together
   (docs/ds41-qwen38-serve-lessons.md:94-111;
   docs/qwen38-perf-2026-09-14.md:38-53).
3. The MTP prefix-cache preparation (Q-2). An MTP prefix consumes shifted
   target hidden states; its attention output and MoE result are discarded,
   so only the QSA raw/pooled index history and K/V are carried forward. The
   cache-only preparation skips query projection, query normalization, RoPE,
   selection, attention output, output projection and the following HC/MoE
   when no draft logits are requested (`ds4.c:27536` area,
   `DS4_QWEN_MTP_FULL_PREFIX` restores the full computation). The ds41
   counterpart is the port's draft arm and prefix path
   (`ds4_ds41_draft.inc:125-198`, citing `core_v41_draft.c:94-175`): skip the
   parts of the draft prefix whose outputs are discarded and keep the verify's
   carried spec_q path (`ds4_ds41_draft.inc:58-59`) intact. Kill switch
   `DS4_V41_MTP_FULL_PREFIX` (docs/ds41-qwen38-serve-lessons.md:120-140).
4. The parity gates and adoption bar. Gates that must stay green, on the
   Spark with the source-built engine oracle and canonical boot:
   `tests/ds41_sampling_gate.sh` (cases A-D byte-identical),
   `tests/ds41_wire_gate.sh` (Anthropic + Responses + smoke),
   `tests/ds41_chat_gate.sh` (354-token prompt + generated ids + response
   message), `tests/ds41_generate_gate.sh` with `DS41_FEED_DIGEST` (feed bytes
   on device must equal the host digest). Protocol for every candidate: fresh
   processes, alternating A/B pairs, graphics-clock medians recorded, adoption
   bar >=2% end to end on the target phase with no repeatable >2% loss on the
   other, and exact ids/logits parity required
   (docs/ds41-qwen38-serve-lessons.md:145-179;
   docs/qwen38-perf-2026-09-14.md:18-21).

## What does not transfer

- The PLE page cache. The engine deliberately rejected caching for the
  engram: O_DIRECT first, plain read + FADV_RANDOM as the fallback, because
  kernel readahead churned the model's mapped pages and made temperature-0
  runs non-reproducible (`core_v41_engram.c:32-58`); row hit rate is near
  zero by design (random row ids). Do not add a cache to the engram
  (docs/ds41-qwen38-serve-lessons.md:74-79).
- GDN, QSA and gated residual: architecture of a different trained model,
  not servable machinery (:83-86).
- The PLE tile and descriptor machinery: a PLE row is 160 values (320 B
  BF16 / 160 B FP8, docs/qwen38-ple-fp8.md storage table); an engram position
  reads 24 rows of 264 B across two planes, a different shape and batching
  (:81-83).

## Reproduce the card sweep

Source: docs/qwen38-ple-fp8.md:77-80. The same binary and aligned-Q8 owner
serve both PLE formats; FP8 uses the FP8 directory and csv, BF16 the BF16
directory and its own csv:

```bash
DS4_QWEN_PLE_WORKERS=16 DS4_PLE_LATENCY_STATS=1 \
DS4_QWEN_PLE_DIR="$fp8_directory" \
./ds4-bench --cuda -m "$main_gguf" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 \
  --gen-tokens 128 --mtp-draft 2 --csv fp8.csv
```

Plot the published curves:

```bash
python -m pip install 'matplotlib==3.11.1'
python docs/benchmarks/plot-qwen-ple-fp8.py
```

## Measurement caveats and limits

- The card sweep is explicitly not a cold 64K prefill or a 256K service
  throughput measurement (docs/qwen38-ple-fp8.md:86-89); a short run proves a
  rate, not sustained performance.
- The paired FP8/BF16 result is the recorded automatic policy including
  embedded-MTP prefix maintenance, not an isolated FP8 arithmetic or MTP
  speedup. FP8 changes embedding values, which can change generated tokens
  and draft acceptance; one BF16 run autoquenched and remains in the
  published data (:123-126).
- Profiled timings are diagnostic only; the 2026-09-14 profiling changed the
  baseline's speculative scheduling, so its aggregate decode rate is not
  used for adoption (docs/qwen38-perf-2026-09-14.md:52-53).
- Acceptance alone hides a gain; report acceptance and speed together
  (:50-53), as the compact-vocabulary candidate did (53.66% -> 51.81% while
  mean cycle time fell 52.93 -> 50.67 ms).
- No broad task-quality or long-context retrieval accuracy claim follows from
  these inference and throughput checks (docs/qwen38-ple-fp8.md:176-177).

## Evidence index

Read first-hand before a new candidate:

- [FP8 PLE pair guide](qwen38-ple-fp8.md)
- [Long-context prefill rounds, 2026-09-04](qwen38-long-context-prefill-2026-09-04.md)
- [Prefill rounds, 2026-09-06](qwen38-prefill-2026-09-06.md)
- [Prefill rounds, 2026-09-07](qwen38-prefill-2026-09-07.md)
- [Draft/prefix campaign, 2026-09-14](qwen38-perf-2026-09-14.md)
- [Image/text/agent gates, 2026-09-07](qwen38-image-2026-09-07.md)
- [Image input contract](qwen38-image-input-spec.md)
- [Published card sweep receipt](benchmarks/qwen-ple-fp8-2026-09-08/summary.json),
  [12 CSVs and artifact receipt](benchmarks/qwen-ple-fp8-2026-09-08/),
  [plot script](benchmarks/plot-qwen-ple-fp8.py)
- [Performance guide Qwen sections](performance.md#qwen) and
  [Uncensored section](performance.md#qwen-uncensored)
- Project-memory context: the repo's own memory records the unit P + qwen38
  study at `/home/leandro/.opengrok/memory/ds4-dfm-rs-df08e5bf/MEMORY.md:2239-2253`;
  `/home/leandro/.opengrok/memory/ds4-602f7992/MEMORY.md:55-63` holds a
  llama.cpp-vs-ds4 comparison on a Qwen-family card whose artifact identity
  was not verified here: do not reuse that as Qwen3.8-Flash-Next evidence.
