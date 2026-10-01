# Model-family history

Retained C-line integration and workload records from the
[model-family guide](ds4-dfm-model-families.md). Use the
[serving contract](serving-contract.md) for current launch procedures.

## Historical integration evidence

The following C release and optimization results preserve their original
commits, artifacts and workloads. They are not fresh v0.1.0 Rust-host gates
or instructions to reuse a recorded owner/process.

## Integration evidence for `v0.6.3-dfm`

This cut absorbs Entrpi `v0.6.3` (`d92d93a`) — typed refusal for
schema-constrained output, chunked request bodies, think-dial
observability (the cont completion line and the `think_modes`
counter family), the full-1M decode dispatch (HG-before-cap,
live-scalar fallback, exact-full bank restore), the whole-prompt
depth fence, and best-fit trim victims — on the same GB10 host
(driver 610.43.02, CUDA 13.3, `sm_121a` cubins only).

One family-side reconciliation was required beyond conflict hunks:
upstream's exact-full bank restore fix (Inc 4 audit Finding 2)
covered only the DeepSeek payload lane, while the Solar, EXAONE,
and Motif-3 cont bank restores carried the same `>= seq_cap`
off-by-one. The family batch contexts share `bank_hist` (seq_cap
slots) and the admission install bound, so the three family lanes
now accept the exactly-full payload a full bank legitimately
persists.

Scope facts verified in review: the engine-side depth fence guards
the DeepSeek/GLM metal session path — the four family sessions
branch out of `ds4_session_sync` before it and chunk by the shared
default prefill cap (≤ 4096 under default env) — while the
server-side fence covers every family's serial lane; best-fit trim
victims operate on the VMM slab lane only (family banks keep fixed
CUDA allocations and remain non-reclaimable); the full-1M HG
dispatch is DeepSeek MLA-only (head_dim-512 guard). The shared
surfaces — typed refusal, chunked bodies, think counters, the cont
completion line — reach every family through the common request
machinery.

Gates on this binary: extractor self-test, `ds4_test --server`
(including the new v0.6.3 refusal/fence/think units), the
split-GGUF test, `test-model-family-kernels`, `test-mmq-parity`,
`cuda-regression` (including the new substrate overflow leg), Motif
loader/tokenizer/reference/CUDA, EXAONE kernels/reference, Solar
loader/tokenizer/KDA/KDA-prefill/KDA-chunk/gates/KV plus the full
forward integration, and dots3 loader/tokenizer — all passed. Bare
`ds4_test` model-dependent DeepSeek GPU tests were not rerun (no
DeepSeek GGUF on this host, as in previous cuts).

A live VMM owner + worker gate on the Motif MQ87-88 artifact
(aligned-artifact owner, 644 exported ranges, worker at `-c 2048`,
32 banks) answered all four API surfaces on the continuous route
with 4 requests and 0 failures. The v0.6.3 typed `response_format`
refusal answered HTTP 400 in the native envelope on the family
lane, and the new `cont chat ... think=... finish=...` completion
line and `ds4_requests_think_total` counters were observed live.

The published Motif-3 and Solar tables below are unchanged: no
remeasure was run for this cut and earlier tags are not moved.

## Integration evidence for `v0.6.2-dfm`

This cut absorbs Entrpi `v0.6.2` (`d183482`) — the v0.6.1/v0.6.2
memory-truth arc: honest decode credit, transient serial-graph leases,
the serial idle reaper, GRAPH_EXEC pool truth, ctx-aware defaults,
live commit-rate feedback, `--no-serial`, manifest content identity,
the governed cont bank plan, the packed work floor, derived fit
headroom, eviction-aligned trim victims, and the continuous ledger
reconciliation line — on the same GB10 host (driver 610.43.02,
CUDA 13.3, `sm_121a` cubins only).

Two family-side reconciliations were required beyond conflict hunks:

- Upstream's rider #48 content fingerprint stats the model path; the
  DFM split-GGUF models map shards into one logical range. The weight
  server now fingerprints that logical mapping (identical layout to the
  engine's `model_open_split`), so split models keep booting and the
  Motif single-file import reports `content identity verified`.
- Upstream's v0.6.2 Inc 3 recency array (`bank_last_use`) is stamped by
  `bank_hist_reset`, which the family persistent-bank lanes share. The
  Solar/EXAONE/Motif batch contexts now allocate it; without the fix the
  first cold family admission crashed the worker (reproduced under gdb).

Gates on this binary: server unit suite, extractor self-test,
split-GGUF test, `test-model-family-kernels`, `test-mmq-parity`,
Motif loader/tokenizer/reference/CUDA six groups, EXAONE
kernels/reference, Solar loader/tokenizer/KDA/prefill/chunk/gates/KV
plus the repaired full forward integration, dots3 loader/tokenizer,
and `make cuda-regression` — all passed. A live VMM owner + worker gate
on the Motif MQ87-88 artifact answered all four API surfaces
(4 requests, 0 failures, continuous route, 32 banks at `-c 2048`).

Same-host `ds4-bench` parity against the `v0.6.0-dfm` band (owner with
aligned Q8 artifacts, context-32768 corpus, greedy): 8K prefill
519.90 / 518.02 tok/s, 8K decode run 515.84 prefill + 12.62 decode
tok/s, 32K decode run 445.03 prefill + 9.68 decode tok/s.

A later Motif-only optimization series on the same `dfm` line
(`d03bd89` HG16, `b0db5a1` SWA→HMMA, `91823ca` MoE D2R,
`a8e9e61` HG16 cp.async, `a09ff4f` FATTN TK=32) remesured 8K/32K and
then the strict 256K serial Chat gate on the same artifact and host.
Measured tip (`2c81427`, kernels through `a09ff4f`): 8K prefill
627.19 tok/s and decode 15.06 tok/s; 32K prefill 545.62 tok/s and
decode 12.95 tok/s; 32K OpenAI sentinels exact (546.7 / 12.8); 256K
OpenAI Chat 262,080-token prefill **238.59 tok/s** and 43 decode tokens
at **5.97 tok/s**, sentinels exact, `finish_reason=stop`. The
`v0.6.2-dfm` **tag is not moved**. The published table below still
shows the `v0.5.6.3-dfm` 8K/32K/256K rows; the remesure is recorded
after that table and is not a new tag.

## Integration evidence for `v0.6.0-dfm`

This cut absorbs Entrpi `v0.6.0` (`c8956e0`) on the same GB10 host
(driver 610.43.02, CUDA 13.3, `sm_121a` cubins only). The gates below
are fixture, unit, and structural GGUF checks on this binary. The Motif
8K/32K/256K published numbers remain those of `v0.5.6.3-dfm`.

| Family | Gate | Result |
|---|---|---|
| DeepSeek | `ds4-eval --self-test-extractors`, `ds4_test --server`, `tests/test_split_gguf` | passed. No DeepSeek GGUF on this host, so model-dependent GPU tests were not rerun. |
| Solar Open2 | `test-solar-loader` / `test-solar-tokenizer` on MXQ-v1 11 shards; CUDA KDA, chunked prefill, gates, compressed KV | passed |
| K-EXAONE | `test-exaone-kernels` vs CPU (no model path: routed-expert matmul skipped); tokenizer load of the 3-shard MXQ | passed |
| Motif-3 | official-final CUDA fixtures (BF16, router, PolyNorm, mHC, expanded/latent GDLA); `test-motif3-loader` / `test-motif3-tokenizer` on `Motif-3-MQ87-88-FIT.gguf` | passed |
| dots3-note | 10-shard loader/tokenizer; CPU/GPU forward; 1600-token chunk/ring and prefix reuse; DSA boundary; 256K resident allocation/cleanup; 4K Chat | passed on `dots3-note-prev-MQ87` |
| Shared | `test-model-family-kernels` | passed |

The merge keeps DFM family generate, split-GGUF remaps, and aligned
mixed-quant remainder caching. Upstream's memory governor, own-reserve
trim, and two-phase reclaim are in; `ds4_batch_ctx_reclaim_prepare`
stays `UNSUPPORTED` for EXAONE/Motif/Solar because those banks use
fixed CUDA allocations. The Motif CUDA fixture also required restoring
the DFM rule that a current whole-model device copy wins over a stale
range keyed by a recycled host address.

The published Motif 8K `ds4-bench` point requires the VMM owner's
aligned Q8 artifacts (the `q8 pair prefill using aligned Q8_0 artifacts`
path). `--no-repack-q8-aligned` falls through to the raw Q8 pair kernel
and is not that point. A `v0.6.0-dfm` remeasure on the aligned-Q8 owner
stayed in the same band and is not a new published number.

## Integration evidence for `v0.5.6.3-dfm`

The following production GGUF integration gates were run on the same GB10
host and release line with a 2,048-token development context. The Motif row
also includes the later strict long-context gate documented below:

| Family | Weight-owner evidence | Server evidence |
|---|---|---|
| DeepSeek V4 Flash | 80.76 GiB base plus 6.49 GiB DSpark; 72.56 GiB aligned artifacts | detected DSpark automatically; one Chat request completed with zero failures |
| Solar Open2 250B | 11 shards, 88.97 GiB; 32.23 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| K-EXAONE 236B A23B | 3 shards, 85.56 GiB; 30.16 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| Motif-3 | 94,162,541,472-byte canonical GGUF; owner used for that run exported 7.00 GiB raw plus 80.68 GiB in 153 aligned expert artifacts | all four API surfaces, strict 262,080-token prompt plus decode, and three concurrent 196K-context banks passed |

The Motif artifact is 94.16 GB, or 87.6957 GiB; 87.70 is its binary GiB size,
not its decimal GB size. The owner used for that run exported 207 VMM ranges: 54 raw
ranges plus 153 aligned Q2_K and IQ2_XXS expert artifacts. The worker imported
those ranges without a duplicate model copy.

## Motif-3 DGX Spark performance evidence

The 32K/256K HTTP gates used
[`593d251`](https://github.com/Baekpica/ds4/commit/593d2511a10694f5a33fbafbd997ca24e819a853).
The 8K `ds4-bench` throughput below used
[`cc2f277`](https://github.com/Baekpica/ds4/commit/cc2f27712482318aef4d83c30f59974739166990)
(FATTN occupancy: drop the Q shared tile so three CTAs fit on GB10), built with CUDA 13.3
as `sm_121a` on one DGX Spark GB10 running driver 610.43.02 and Linux
6.17.0-1029-nvidia. The server used the production MQ87-88 artifact, the VMM
owner above, a 4,096-token prefill chunk, greedy sampling, no thinking, no
speculation, and one request at a time.

| Gate | Interface | Prompt | Prefill | Decode | Correctness |
|---|---|---:|---:|---:|---|
| 8K | `ds4-bench` | 8,192 | 519.55 tok/s | 64 tokens at 12.28 tok/s | throughput fixture; prefill-only 519.55, decode-run 516.17 / 12.28 |
| 32K | OpenAI Chat | 32,768 | 82.649 s; 396.47 tok/s | 43 in 4.799 s; 8.96 tok/s | beginning, middle, and end sentinels exact |
| 256K | OpenAI Chat, `-c 262144` | 262,080 | 1,492.375 s; 175.61 tok/s | 43 in 17.072 s; 2.52 tok/s | all sentinels exact; `finish_reason=stop`; 262,123 total tokens |

The two HTTP gates were non-streaming, so they do not provide an independent
network-visible time-to-first-message measurement. The table reports the
server's prompt-complete and decode timings and makes no separate TTFM claim.

The 256K session reported 4,422,546,432 bytes (4.119 GiB) of latent KV and
rotated-key payload. Including the default 4,096-token execution graph, its
physical worker allocation was 9.703 GiB. Source-GGUF mapping RSS remained
29,632 KiB after inference, and engine shutdown left 637,251,584 bytes of CUDA
module/driver state, below the 896 MiB lifecycle gate. During the full request,
the worker and owner both remained at `VmSwap: 0`; system memory retained about
12 GiB available. Loaded clock samples remained between 2,398 and 2,411 MHz,
so the earlier 611 MHz pin did not recur.

### Motif-3 remesure on the `v0.6.2-dfm` line (2026-08-21)

Same host, same MQ87-88 GGUF, same aligned-Q8 VMM owner (`--reserve-gb 24`),
same 4,096-token prefill chunk, greedy, no thinking, no speculation. Engine
tip `2c81427` (kernels through `a09ff4f`). The 256K cell used the official
`context-262144-server.txt` Chat fixture and `DS4_SERVER_COALESCE_MAX=1`
(serial lane, `-c 262144`).

| Gate | Interface | Prompt | Prefill | Decode | Correctness |
|---|---|---:|---:|---:|---|
| 8K | `ds4-bench` | 8,192 | 627.19 tok/s | 64 tokens at 15.06 tok/s | throughput fixture |
| 32K | OpenAI Chat | 32,768 | 546.7 tok/s | 12.8 tok/s | beginning, middle, and end sentinels exact |
| 256K | OpenAI Chat, `-c 262144` | 262,080 | 1,098.433 s; 238.59 tok/s | 43 in 7.205 s; 5.97 tok/s | all sentinels exact; `finish_reason=stop`; 262,123 total tokens; `cached_tokens=0` |

Versus the `v0.5.6.3-dfm` published 256K row this is +35.9% prefill and
+137% decode. The 256K worker held 10,429 MiB with 4.119 GiB of latent KV;
owner and worker `VmSwap` stayed 0; available memory stayed 11–12 GiB;
SM clocks sampled 2,411–2,496 MHz. Concurrent 256K banks are still not
claimed. Evidence:
`scratch/motif3-opt-v062/logs/sent-256k-summary.txt`
(response SHA-256
`f4aafb4c969c46889daceb64feb01177c4682e75efff555a6539202f78cd42aa`).

Nsight Systems on the final 32K prefill ranked aggregate CUDA kernel time as
expanded FATTN 15.5%, paired Q8 projection 11.0%, latent attention 9.7%, BF16
rounding 8.5%, W_UV value projection 7.5%, routed gate/up 7.2%, and QK absorb
4.1%. Focused 4,096-row Nsight Compute runs measured:

| Kernel | Before | Final | Reduction |
|---|---:|---:|---:|
| expanded FATTN | 55.79 ms | 28.83 ms | 48.3% |
| Motif group-5 QK absorb | 38.91 ms | 10.97 ms | 71.8% |

The final strict gate JSON and server log were retained with SHA-256
`b8551d5c96a0bdc1b6244275b79a5ac9ac9f8932862a93f6256ff51df00d7a9f` and
`90b064268bcc31498e653d27fcf5087064910bf9a6963f4fe239dc295b0fbeda`,
respectively.

### Motif-3 196K multi-bank serving evidence

The persistent-bank extension at `03b7002` and `cf605e0` was built as
`sm_121a` and run with `-c 196608`, three banks, an 8,192-token prefill chunk,
and `--no-spec`. The explicit 6 GiB batch-fit headroom left the measured
configuration at three banks instead of the conservative default reducing it
to two.

| Gate | Result |
|---|---|
| API surface | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/messages` each returned HTTP 200 with the native response shape |
| 8K cold prefill | 8,214 prompt tokens at 266.3 tok/s; `LONG_OK` returned exactly |
| Single decode | 192 output tokens; 490.4 ms TTFT, 12.9 tok/s decode, 15.350 s HTTP wall time |
| Three simultaneous Chat requests | 192 output tokens each in 24.885--25.030 s; 23.01 aggregate output tok/s; server log `served=3 fallback=0` |

After the gates, `/v1/stats` reported 11 completed requests, zero failures,
zero serial requests, zero continuous-batch failures, three total and zero
live banks, and zero speculative drafts. The VMM owner used 90,119 MiB, the
worker used 22,283 MiB after the 8K requests, and the system retained about
6.5 GiB available without an OOM event. Loaded SM clock remained 2,411 MHz;
the 611 MHz pin did not recur.

## Solar Open2 DGX Spark performance evidence

The [14 September 2026 campaign](solar-open2-optimization-2026-09-14.md)
is the current clock-capped record: disk-KV restart reuse, HTTP partial
fork, and default-on warp-specialized K-FP8/V-FP4 prefill attention
(`DS4_SOLAR_FATTN_WS=0` restores the pair kernel). SM clock 2190 MHz
inside the 300–2200 MHz cap. Cold `ds4-bench` medians of three, Promessi
Sposi, 4,096-token chunks, 64 greedy tokens, byte-identical 196,608
logits and 64 IDs:

Round 1 FATTN_WS (`DS4_SOLAR_FATTN_WS=0` restores pair):

| Prompt tokens | Prefill off → on | Decode off → on |
|---:|---:|---:|
| 8,192 | 1,050.86 → 1,075.76 tok/s | 17.40 → 17.44 |
| 65,536 | 731.24 → 927.50 tok/s | 13.06 → 13.02 |

Round 2 skip Q3 handoff down sanitize (`DS4_CUDA_MOE_HANDOFF_SANITIZE=1`
restores the pass), measured with WS on:

| Prompt tokens | Prefill off → on | Decode off → on |
|---:|---:|---:|
| 8,192 | 1,073.59 → 1,095.61 tok/s | 17.43 → 17.43 |
| 65,536 | 925.19 → 943.18 tok/s | 13.02 → 13.01 |

The [September 12 Rust-host campaign](solar-open2-optimization-2026-09-12.md)
records guarded 8K/64K controls and four unretained attention candidates
at uncapped clocks. [Round 5](solar-open2-optimization-2026-09-12-r5.md)
introduced the WS kernel as opt-in after 64K froze the host; the 14
September A/B is the 64K full-model sample under the cap.

The historical HTTP numbers below used
[`b2e52b9`](https://github.com/Baekpica/ds4/commit/b2e52b9048ba339327539212de1c47d009dde126)
on `origin/dfm`, built with CUDA 13.3 as `sm_121a` on one DGX Spark GB10
(driver 610.43.02, Linux 6.17.0-1029-nvidia). The GGUF is MXQ-v1 11 shards
(`Solar-Open2-250B-MXQ-v1`, 95,533,532,160 bytes). A long-lived VMM owner
(`--backend vmm --scope base --reserve-gb 16`, 453 derived aligned artifacts)
served a restartable worker at `--cuda -c 196608` with three persistent banks
and a 4,096-token prefill chunk. Requests were OpenAI Chat with thinking
disabled, exact-cold (`cached_tokens=0`), and 128 decode tokens. Each cell is
the median of three. Loaded SM clocks stayed between 2,411 and 2,561 MHz.
`banks_total=3` still admitted after the runs.

| Depth | Prompt tokens | Prefill | Decode p50 | Decode API |
|---|---:|---:|---:|---:|
| 8K | 8,222 | 1,050.7 tok/s | 19.05 tok/s | 18.9 tok/s |
| 64K | 66,761 | 804.5 tok/s | 13.07 tok/s | 14.1 tok/s |

On the same host and artifact, before this default-path series, 8K decode was
17.5 tok/s and 64K average prefill was 710 tok/s. The landed commits are
`3651787`, `5d2a96c`, `fd3a426`, `7563969`, `262ff8b`, and `b2e52b9`.
`test-solar-kv` reported 512-token GQA2 vs one-head `rel_rms=0` and split vs
direct `rel_rms=8.45e-7`. Incremental `T(64K)−T(60K)` last-4K is not a
published metric. 1,048,576-token serving is not claimed.
