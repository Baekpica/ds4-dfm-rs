# DeepSeek V4.1 Flash (ds41) on ds4-dfm-rs

DeepSeek V4.1 Flash is the family the `ds41` port brings to this tree: a
40-layer mixture-of-experts model whose routed-expert weights are 8-dimensional
vector quantization, served from one base GGUF and the original Engram tables,
with optional sidecar and post-train directories. This is the family's model guide: the artifact contract,
the shape this tree binds, the serving surface, the gates that back it and the
recorded benchmark numbers.

[JLFN](https://github.com/JLFN) ported DeepSeek V4.1 Flash serving to
`ds4-dfm-rs` in [PR #80](https://github.com/Baekpica/ds4-dfm-rs/pull/80),
from the [YoungAi engine](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5).
The VQ GGUF and GR/RB sidecars are distributed by
[YoungAi](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash);
the original model and Engram weights come from
[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

It is not live state. It retains the recorded host, artifact, binary hashes and
workload, and it does not qualify a new build. The executable plan and its
per-unit evidence are in the [port plan](deepseek41-port-plan.md); the engine
function inventory is in the [port inventory](deepseek41-port-inventory.md).
Every number and bound below is taken from a record in this repository, read
first-hand: the plan, the [prefill/decode report](ds41-perf-2026-10-11.md), the
[speed sweep](ds41-speed-bench-2026-10-12.md), the [serving contract](serving-capabilities.md)
and its source `crates/ds4-core/src/serving.rs`.

The [October 13 review qualification](benchmarks/ds41-spark-review-2026-10-13/README.md)
records fresh Engram-on source parity and the full 2K–40K screenshot sweep at
300–2200 MHz. Decode median is 41.0 tok/s; 40K prefill is 1,070.6 tok/s.

The [subsequent 2K–64K campaign](benchmarks/ds41-spark-64k-2026-10-13/README.md) adds three fresh full-prefill
sweeps, separate Prefill/Decode tables and a graph crediting JLFN's
DeepSeek V4.1 port. It reaches 66,608 actual input tokens; median run means are
1,039.65 prefill / 38.17 decode tok/s on the unchanged runtime.

The [warm-path campaign](benchmarks/ds41-warmpath-2026-10-13/README.md) measures
what a full-prefill sweep cannot see: the per-request asset prelude the port used
to pay. Holding the sidecar mount and the engram open per process takes
warm-prefix TTFT from 190.9 to 15.1 ms on the 54-token prose case and from 265.7
to 99.1 ms on the 10.4k-token one, with the cold first request and decode both
unchanged or better.

## Artifact and host identity

Recorded in the port plan §2, measured over the Spark mount. The base is
`DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`.

| item | recorded value |
| --- | --- |
| base | 113,556,639,424 bytes, 40 parts |
| base sha256 | `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654` |
| sidecars | five domain dirs `…-grrb-{vqfin41_vqhalf_a_n8192,code_fit_n15360,law_fit_n15360,med_fit_n15360,sci_fit_n15360}-engine`, each `gr_Lnn.bin` (1,044,492 B) + `rb_Lnn.bin` + `manifest.txt` |
| post-train | `posttrain-experimental-20260924/`: `gr_L39.bin` + `base.fnv` (20 B) |
| engram | not in the artifact: two official shards read in place via `--engram-dir`, `model-00047-of-00048.safetensors` 101,535,150,936 B and `model-00048-of-00048.safetensors` 101,537,926,640 B |
| vision | the model's own tower sidecar `DeepSeek-V4.1-Flash.vision.gguf`, 971,066,016 B, via `--vision-gguf`: DeepSeek-ViT (32 blocks, 1024-dim, 16 heads, patch 14, downsample 3) + the 2-layer aligner + the three delimiter rows + 40 per-layer `exp_probs_b_vl` router biases |

`--v41-no-engram` is a diagnostic that removes the Engram contribution.
Results with it do not qualify the complete model's quality or performance.
The vision sidecar is optional: without it the family serves text unchanged,
and an image request is refused with HTTP 400 by name rather than answered
without the picture.

The October 11 report records NVIDIA GB10 (DGX Spark), driver 580.142 and
CUDA 13.0 (docs/ds41-perf-2026-10-11.md:5-7). The five-case record omits its
driver, CUDA version and clocks. Corpus for the frontier sweep:
`speed-bench/promessi_sposi.txt`, SHA-256
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`, the same
fixture the other campaigns use.

Boot line the port is driven with (docs/ds41-speed-bench-2026-10-12.md:17-20):

    ./ds4-server --cuda -m <base>.gguf --zchain <domain sidecar> \
        --engram-dir <shards_dir> --mem-budget-mb 110000 \
        [--posttrain <dir>] --host <addr> --port <port>

The fresh resident run used the [host memory guard](host-memory-guard.md)
with a 2 GiB reserve and `--psi-full-limit 0`: stalls are recorded, while
the available-memory floor and cgroup limits remain enforced. The model's
`--mem-budget-mb` does not configure that external host reserve.

## The model: shape and tensor contract

`SHAPE_V41_FLASH` (`crates/ds4-core/src/shape.rs:581-636`) is the shape this
tree binds. It has no `n_hash_layer` (V4.1 drops it) and keeps the two values
the shape comment flags as real, not typos: `rms_eps` 1e-20 and route scale 1.5.

| field | value | field | value |
| --- | ---: | --- | ---: |
| layers | 40 | heads | 64 |
| embedding | 5120 | KV heads | 1 |
| vocab | 129280 | head dim / value dim | 512 / 512 |
| routed experts | 384 | rotary dim | 64 |
| experts used / shared | 6 / 1 | q / o LoRA | 1280 / 1024 |
| expert FF | 2304 | sliding window | 128 |
| indexer heads / dim | 32 / 128 | indexer top-k | 512 |
| hyper-connections | 4 | sinkhorn iters | 20 |

The artifact's tensor contract was dumped from the artifact itself and is
recorded in the port plan §6.6.1. Globals: `token_embd.weight`, `output_norm.weight`,
`output.weight` and the engram maps (`engram.token_map|multipliers|primes|offsets`).
There is no `output_hc_*`: the head reuses the last `ffn_pre`. Every layer
carries the hyper-connection triples (`hc_attn_*`, `hc_ffn_*`), MLA attention
(`attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv`, `attn_kv_a_norm`,
`attn_sinks`, `attn_output_a`, `attn_output_b`), `ffn_norm`, `ffn_gate_inp`,
`exp_probs_b.bias`, the single **`ffn_exps_vq.blob`** (the artifact carries no
per-expert gate/up/down tensors) and the shared experts.

- Source layers carry `attn_compressor_kv|norm`, `indexer.wk|k_norm` and, for
  index sources, `indexer.attn_q_b|proj`. Which layers those are comes from
  metadata arrays, not a formula: `kv-sources=[2,8,14,20]`,
  `index-sources=[2,8,14,20,24,28,32,36]`.
- Engram layers 1 and 14 add `engram_wkv` (fp8_32x32), `engram_q`, `engram_k`.
- Three MTP draft towers `mtp.{0,1,2}` mirror a layer's set including
  `ffn_exps_vq.blob`, plus five shared heads; the towers' metadata expert count
  is 128 (the main MoE stays at 384 top-6).

The routed-expert blobs use DQVL v3 with mixed index widths: inspected layer 0
payloads use 13 bits and layer 14 payloads use 12 bits. The decode family is
`cuda/ds41_vq_*.cuh`. Loader closure on the real file: the
inspector prints `identify`, `validate: ok`, `layout: ok` and
`bind: slots=1000 bound=1000 required-missing=0` (port plan §6.6.1).

Context is the model metadata `deepseek4.context_length` = 1,048,576; `--ctx`
is rejected for this family (recorded at port plan §6.16, item 3, read from the
artifact).

## Serving contract and limits

The family row is generated into [serving capabilities](serving-capabilities.md)
from `ServingCaps` (`crates/ds4-core/src/serving.rs:663-684`).

| property | value |
| --- | --- |
| banks | serial / none |
| prefix reuse (plan surface) | exact / present; auto, exact and off controls |
| disk KV | present, not qualified (`--kv-disk-dir`) |
| snapshots | none |
| MTP | embedded, qualified |
| speculation lane | serial, draft min 1 |
| host | CUDA |
| context cap | none — the runtime context is the model metadata |
| qualified context / banks | none / 1 |

The [review report](benchmarks/ds41-spark-review-2026-10-13/README.md#serving-evidence)
adds a bounded 3,725/3,726-token disk-restart gate with exact output/logits and
four negative controls. That result does not promote every artifact, sampling
mode or context in the generic capability table.

The native run keeps one prompt-end KV checkpoint and reuses the previous
prompt's prefix automatically (unit U-B, `--no-kv-reuse` disables it); it is
reported as usage cached tokens. `--prefix-reuse off` disables both live and
disk reuse; auto and exact select the prompt-end checkpoint. Partial reuse and
`--max-seqs > 1` are refused by name
(`v41_serving_contract_refuses_what_it_lacks`). The per-family sampling profile
is not adopted on this path: sampling is request-driven.

For device sampling, an identical prompt is prefilled again: the checkpoint
does not store the sampler's position slot. This avoids an inherited
uninitialized-state read. Greedy and penalty sampling retain full-prompt hits;
device sampling still reuses a prefix when a suffix forward fills those slots.

`/v1/completions` follows the common Rust raw-continuation contract. The pinned
YoungAi server adds a chat frame by default; use its `raw: true` request field
for equivalent source inputs. Chat completions use the family template.

Server flags (engine spellings, `crates/ds4-server/src/bin/ds4-server-rs.rs:241-263`):
`--engram-dir`, `--v41-no-engram`, `--no-dspark` / `--dspark`, `--dspark-verify`,
`--no-graph`, `--no-kv-reuse`, `--emit-trace`, `--v41-prof`, `--zchain`,
`--posttrain`. The boot line prints the resolved context and the KV-reuse state.
The common `--mtp-mode off|auto|on` controls the same embedded drafter.
`--mtp-draft` cannot change its trained block size; an external `--mtp` path
is refused. `--dspark-verify` selects a verification width within that block.
Native knobs: `DS4_V41_CHUNK`, `DS4_V41_CKV_BYTES`, `DS4_V41_IDXK_BYTES`,
`DS4_V41_MAX_ENGRAM`, `DS4_V41_GEMV_MAX_TOK`, `DS4_V41_DRAFT_GROWS`,
`DS4_V41_MKCACHE_SLOTS`. There is no settable prefill chunk on the port yet
(`serving.rs:1616-1620`).

## Port status and gates

Subsystems closed, each on the Spark against the source-built oracle unless
noted, all recorded in the port plan:

- Loader (P0-P3): tensor types 40-44, the VQ decode oracle, the engram loader,
  the sidecar readers and the sidecar/post-train `base.fnv` gate; the artifact binds
  1000/1000 tensors.
- Forward (P4): the eager single-state forward is bit-exact against the golden
  set on the bare variant (per-layer `x_Lnn`/`y_Lnn`, `hce_Lnn` and logits;
  port plan §6.12).
- Draft/spec (P5): the three-tower MTP drafter arms from metadata; the greedy
  spec round's emitted ids and `[dspark]` trace equal the engine golden, and
  the port's spec ids equal its non-spec ids.
- Serving (P6): the sampling gate (pinned seed byte-identical, the port samples
  through `cuda/ds41_sample.cuh`), the wire gate (`/v1/messages`,
  `/v1/responses`, the smoke request), the chat gate and the DSML tool path
  (port plan §6.16-§6.19, §6.23).
- Engram read path (unit P): `io_uring` round reads (`crates/ds4-core/src/uring.rs`),
  then the cross-block prefetch (below).
- KV prefix reuse (unit U-B): the in-process checkpoint gated
  (`tests/ds41_kv_reuse_gate.sh`); the disk half rides the KVC store when
  `--kv-disk-dir` is set (present, not qualified).
- Lifecycle (unit Z) and post-train (PT) are recorded in the port plan §6.18
  and [the post-train plan](ds41-posttrain-plan.md).

Gate scripts live in `tests/` as `ds41_*_gate.sh`
(`generate`, `serving`, `sampling`, `wire`, `chat`, `graph`, `kv_reuse`,
`kv_disk`, `long`, `toolcall`, `serving_http`) plus the perf sweep
`ds41_perf_sweep.sh`.

Not present on this path, and refused or absent by name: banks and snapshots
(no `ds4_session`), partial prefix reuse, the per-family sampling
profile, and `--v41-chunk`. The remaining open item against the engine is the
decode residual (~4% of wall, in the verify step; docs/ds41-perf-2026-10-11.md
"Limits and open boundaries").

## Recorded performance

Two recorded campaigns on the same GB10 shape. Cross-day wall comparisons on
this host are not valid; only same-window A/B or a matched binary pair is.
Decode varies with content and draft acceptance. The prose sweep sits near
40-44 t/s, while the five-case record reaches the mid-50s to mid-60s on code
and tables; that record lacks acceptance counters, so it cannot establish the
cause of the spread.

Five-case benchmark
([ds41-spark-2026-10-13](benchmarks/ds41-spark-2026-10-13/README.md), DGX Spark
GB10, same GGUF and coding sidecar, temperature 0, greedy, single stream, median
of 3, port checkout `2fb144b`):

| case | prompt tok | decode t/s | cold prefill t/s | warm TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| prose_short | 54 | 36.26 | 82.1 | 190.9 |
| code_short | 54 | 54.33 | 91.4 | 196.6 |
| table_short | 43 | 65.41 | 79.1 | 194.1 |
| prose_long | 10356 | 31.08 | 1286.0 | 265.7 |
| table_long | 10377 | 59.14 | 1283.4 | 273.4 |

Recorded decode therefore spans 31.1-65.4 t/s across content; the short-case
cold prefill (82-91 t/s) is a 43-54-token prompt and is a latency number, not a
throughput one.

22-frontier prose sweep (docs/ds41-speed-bench-2026-10-12.md,
`speed-bench/gb10_spark_ds41_862fd9a6.csv`, corpus promessi_sposi, 128 greedy
tokens, clock median 2405 MHz): prefill ramps from 852.6 t/s at the cold first
request to a 1201.9 t/s peak at 18537, holds ~1170-1200 to 26624 and ends at
1117.9 t/s at 45315; decode is 40.0-43.7 t/s on this prose corpus with
speculation live (`tok_per_step` 2.67-2.91, about 1.8 accepted drafts per
round). The two decode dips (29.9 at 28672, 28.7 at 34816) track
draft-acceptance variance, not interleave.

2026-10-11 prefill/decode cells (docs/ds41-perf-2026-10-11.md), same prose
corpus: the line rate is the server's own `[v41] prefill` timing; decode is
plain greedy.

| context | prefill line t/s | decode t/s |
| ---: | ---: | ---: |
| 354 | 501.0 | 40.5 |
| 839 | 413.1 | 38.0 |
| 2048 | 979.6 | 39.5 |
| 8192 | 916.9 | 39.4 |

Measured A/Bs on this side:

- Cross-block engram prefetch: +15.5% and +15.3% on the prefill line rate at
  8192 across two passes in opposite order, decode within noise, emitted
  content identical (`DS41_FEED_NO_PREFETCH` is the kill switch).
- KV prefix reuse: at 2048, turn 2 restores 2149 tokens and prefills only the
  145-token suffix at 2929.7 t/s instead of re-prefilling all 2294 at 844.4 t/s;
  fresh-prompt content sha unchanged.
- Side-stream forks (r6, build `1e7d8f24`): prefill at or above the baseline on
  every frontier (+1.4% at 2048, +0.3% median); decode median 41.20 -> 41.70 t/s.

## Optimization target

The comparable target for the ds41 port is port == its own engine, not the
headline rate of another family: those are different architectures and different
active parameters, so the numbers are not comparable. What transfers from the
qwen38 route is the serving machinery around the off-accelerator table plus the
two MTP optimizations, each as a measured candidate with a kill switch and the
repo's adoption bar: fresh matched processes, a reproducible end-to-end gain
beyond noise, and the family's numerical contract. There is no fixed percentage
floor; report memory, work and performance costs in both phases. The detailed study is
[what the qwen38 serving path teaches the ds41 route](ds41-qwen38-serve-lessons.md);
the candidates and gates are port plan §6.21.

## Measurement caveats and limits

- Single samples: one run per cell, no interleaved medians except where an A/B
  is named. The sweep is a full-prefill per frontier (single-turn renders never
  extend the previous prompt), so `prefill_cached_tokens` reads 0 there and the
  turn-2 reuse path is measured separately.
- Decode is content-dependent (draft acceptance), so read it per workload: the
  five-case record spans 31.1-65.4 t/s and the prose sweep 40.0-43.7. The
  five-case benchmark ran on a different build (`2fb144b`) than the sweep.
- Clock medians in the ds41 records (2405, and 2434-2447 in the cells) sit above
  the documented 300-2200 MHz user-managed range; reconcile before comparing
  against the qwen38 records, which also quote a different driver/CUDA.
- No broad task-quality or long-context accuracy claim follows from these
  inference and throughput checks.
- The October 11 cell campaign did not reach 32K or 64K; the later October 12
  sweep reached 45,315 prompt tokens. Its 47,104-65,536 frontiers remain open.
  The cell campaign left a ~40-50 ms setup residual and cold first-block terms open.
- The port is served on the `ds41` branch and is not in a release ledger; the
  gates above are this repository's own recordings.

## Evidence index

Read first-hand before a new candidate:

- [Port plan](deepseek41-port-plan.md) and [port inventory](deepseek41-port-inventory.md)
- [Five-case benchmark, 2026-10-13](benchmarks/ds41-spark-2026-10-13/README.md)
- [Prefill/decode report, 2026-10-11](ds41-perf-2026-10-11.md)
- [Speed sweep, 2026-10-12](ds41-speed-bench-2026-10-12.md)
- [Post-train plan](ds41-posttrain-plan.md)
- [What the qwen38 serving path teaches the ds41 route](ds41-qwen38-serve-lessons.md)
- Sweep data: `speed-bench/gb10_spark_ds41_862fd9a6.csv` (+ `.csv.meta.txt`),
  `speed-bench/gb10_spark_ds41_1e7d8f24.csv` (+ `.csv.meta.txt`),
  `speed-bench/ds41_engine_vs_fork_compare.png`
- Shape and contract in code: `crates/ds4-core/src/shape.rs:581`,
  `crates/ds4-core/src/serving.rs:663`, `docs/serving-capabilities.md`
- Gates: `tests/ds41_*_gate.sh`, `tests/ds41_perf_sweep.sh`,
  `tests/capture_ds41_golden.sh`
