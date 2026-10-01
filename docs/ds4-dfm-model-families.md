# ds4-dfm model families on DGX Spark

`ds4-dfm-rs` is the independent Rust-host continuation of DwarfStar DFM
(독자 파운데이션 모델, 독파모). v0.1.0 is the first independent Rust-host
baseline. Rust owns the host runtime; native CUDA/MMQ/VMM remains the compute
backend. See [LINEAGE.md](LINEAGE.md) for
the inherited C release history and the [release ledger](releases/v0.1.0.md)
for qualification and workload limits.

The runtime also carries explicit non-DFM family ports, including dots3-note,
Qwen3.8, GLM 5.3 Flash, K2-Horizon, Inkling Small, Step 3.7 Flash,
Ling-3.0-flash-VL, MiMo-V2.6-Flash and Naive-N0.5-Flash. Inclusion does not classify
those source models as Korean DFM. The [repository README](../README.md#supported-model-families)
links the supported artifacts; this guide records their state and release limits.

The reference target is one NVIDIA DGX Spark with a GB10 GPU and 128 GB of
unified memory. Bonsai has separately recorded RTX 4070 SUPER gates; inherited
Metal and other accelerator paths require their own checks.

[Families](#integrated-families) · [Qwen](#qwen-release-scope) ·
[GLM](#glm-53-flash-release-scope) · [K2](#k2-horizon-375b-release-scope) ·
[Reuse](#partial-prefix-reuse) · [Limits](#current-limits) ·
[Historical evidence](model-family-history.md)

## Design contract

ds4 is not a general GGUF runtime. A model is accepted only when its GGUF
metadata and tensor layouts match an explicit Rust
[shape contract](../crates/ds4-core/src/shape.rs) and native execution path.
Rust owns identification, validation, tensor bind plans, tokenizer/chat
behavior and lifecycle policy. Native code owns weight upload, CUDA/MMQ/VMM,
graphs and numerical state. Adding a family must cover both sides and their
state, API and correctness gates.

The implementation stays close to upstream's style:

- model selection is a small enum and direct dispatch;
- shared arithmetic reuses the existing CUDA primitives and aligned weight
  artifacts;
- different attention, recurrent state, or expert math gets a
  direct family path;
- no plugin registry, graph framework, or broad abstraction layer is added;
- external MTP sidecars require the exact DeepSeek, Inkling or Step family contract;
  Ling-3.0-flash-VL has no predictor block at all;
  DSpark uses explicit DeepSeek or Naive artifact contracts. dots3-note
  executes its embedded MTP block
  only on the explicitly enabled serial path described below.

This keeps the changes reviewable for a possible future upstream contribution.

## Integrated families

| Family | Shape selected from | Native state/runtime |
|---|---|---|
| DeepSeek V4 Flash / PRO | `general.architecture=deepseek4` | Entrpi compressed KV and continuous graph |
| Solar Open2 250B | `general.architecture=solar-open2` | recurrent KDA state plus compressed GQA KV |
| K-EXAONE 236B A23B | `general.architecture=exaone-moe` | LLLG full/sliding GQA KV |
| Motif-3 | `general.architecture=motif3` | normalized latent KV, rotated `k_pe`, and SWA rings |
| [dots3-note Preview](#dots3-serving) | `general.architecture=dots3note` (legacy `dots3-note`) | dual-geometry latent KV, DSA keys, and SWA rings |
| Qwen3.8 Flash Next SSD-PLE | `general.architecture=qwen4exp` | Q5 main + four SSD-PLE sidecars, GDN/QSA state, embedded MTP, still images |
| [Prism Bonsai 2 27B](BONSAI.md) | `general.architecture=qwen35` | pinned PQ2_0 with Prism Hadamard-fold metadata; CPU reference and serial CUDA text; banks, snapshots, disk KV, drafting and media unsupported |
| GLM 5.3 Flash | `general.architecture=glm5-next` | exact Q2 main + vision sidecar |
| K2-Horizon 375B A23B | `general.architecture=k2-horizon` | full-attention GQA KV, partial NeoX RoPE, shared-expert MoE |
| [Inkling Small](inkling-small.md) | `general.architecture=inkling` | MQ85GB source-interleaved GQA, four-tap convolution, embedded media encoders, optional eight-layer MTP-BF16 |
| [Step 3.7 Flash](step37-initial.md) ([serving](step37-serving-2026-09-13.md)) | `general.architecture=step35` | MQ83 full/sliding GQA, post-SiLU expert clamps, optional Q8 MTP and F16 vision |
| [Ling-3.0-flash-VL](ling3-flash-vl.md) | `general.architecture=bailingmoe3` | 35 recurrent KDA blocks and 7 latent MLA blocks, 512 grouped-sigmoid experts, separate Qwen3-VL mmproj |
| [MiMo-V2.6-Flash RL / MOPD](#mimo-release-scope) | `general.architecture=mimo2` | 9 full-attention and 39 SWA-128 layers, 256 experts top-8, three embedded MTP blocks; artifact-specific gates |
| [Naive-N0.5-Flash](#naive-release-scope) | `general.architecture=naive_n05_flash` | MQ87; 39 SWA-128 and nine DSA layers, BF16 GQA, E4M3 indexer history, continuous banks and disk KV |

The scheduler implementation may differ because the model states differ, but
the operator and client contract is the same. Changing `-m` to a GGUF from a
different supported family selects the corresponding runtime in the same
binary. Shared flag names and the requested / effective / qualified plan
live in the [serving contract](serving-contract.md). The
[generated capability table](serving-capabilities.md) records server lanes,
reuse, disk, MTP and bounds directly from `ds4_core::serving_caps`; the core
tests reject documentation drift. Dated campaign reports retain their scope.

## Qwen release scope

The initial Rust RC qualification covered:

- [`MQ-Q5-SSD-PLE-BF16`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF), three main GGUF shards;
- four shared BF16 SSD-PLE sidecars referenced by that Q5 layout;
- embedded MTP with `--mtp-draft 2`;
- text and base64 PNG/JPEG input on the three message APIs;
- 196,608 two-bank serving and 262,144 one-bank serving, in addition to the
  earlier exact/configured 262,144-token gates.

Q6, original safetensors, and a resident BF16 GGUF were not release gates and
are not implied by this claim.

For the optional FP8 PLE sidecar with the existing base and
[Uncensored](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF)
main GGUFs, see [selection, validation and paired 64K sweeps](qwen38-ple-fp8.md).

Rust normalizes ordered image parts, bounds and owns payload bytes, places
image tokens, and owns decoded-pixel cache identity. Decoding reuses the pinned
[`vendor/stb_image.h`](../vendor/stb_image.h) through a narrow native image ABI;
vision and CUDA execution stay native. No general multimedia layer or Rust
image dependency was added.

Image limits match the frozen C behavior:

- user messages only;
- PNG or JPEG data URIs only;
- at most four images;
- at most 10 MiB decoded per image and 20 MiB per request;
- remote URLs, files, SVG, GIF, WebP, malformed base64, and invalid image
  content are rejected.

See [`QWEN_V065_RESTAMP_2026-08-31.md`](rust-migration/QWEN_V065_RESTAMP_2026-08-31.md)
and [`qwen38-image-input-spec.md`](qwen38-image-input-spec.md).
Measured image latency and agent checks:
[`qwen38-image-2026-09-07.md`](qwen38-image-2026-09-07.md).

### Qwen owner and cache

The following Qwen reference keeps a VMM weight owner resident across worker
restarts. Use the first Q5 shard as `MODEL`; the PLE sidecars remain alongside
the model or in the selected FP8 directory. See the [shared launch procedure](serving-contract.md#weight-owner-and-worker).

```sh
MODEL=/path/to/supported-model.gguf
MANIFEST=/tmp/ds4-weights.manifest

./ds4_weight_server \
  --base "$MODEL" \
  --manifest "$MANIFEST" \
  --backend vmm \
  --scope base \
  --reserve-gb 32
```

Wait for both `broker listening` and `ready manifest=...`, then start the
worker in another durable session:

```sh
DS4_CUDA_WEIGHT_IPC_MANIFEST="$MANIFEST" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
./ds4-server --cuda -m "$MODEL" -c 196608 \
  --host 127.0.0.1 --port 8000 --no-update-check
```

Qwen Q5 release runs additionally set a bounded SSD-PLE cache. Size it from
the prefill chunk: a chunk's sixteen 320-byte PLE rows per token land on
about 1.08 4 KiB pages each, so an 8,192-token chunk needs ~553 MiB of pages,
and the engine prefetches the *next* chunk's pages while the current chunk's
decoder layers run (`DS4_QWEN_PLE_NO_LOOKAHEAD=1` disables that). One prefill
stream therefore wants at least one chunk in cache (1024 MiB with slack); two
banks that alternate chunks want two (2048 MiB, the maximum). Sixteen page
workers already saturate the sidecar reads at ~90K IOPS in bursts that overlap
compute, so more workers do not help. A prompt's first chunk has nothing
queued for it, so every prompt opens with a 2,048-row chunk whose remaining
decoder layers hide the reads of the full-size chunk behind it; prompts
shorter than two opening chunks stay one chunk, since a short trailing
chunk costs more than the reads it hides
(`DS4_QWEN_PREFILL_OPENING` sets the opening rows; `0` opens at the chunk
cap). This reference shape asks the shared Rust scheduler for two persistent
banks:

```sh
DS4_QWEN_BATCH=1 \
DS4_QWEN_PLE_CACHE_MB=2048 \
DS4_QWEN_PLE_WORKERS=16 \
DS4_QWEN_PREFILL_CHUNK=8192 \
DS4_SERVER_COALESCE_MAX=2 \
DS4_CUDA_WEIGHT_IPC_MANIFEST="$MANIFEST" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
./ds4-server --cuda -m "$MODEL" -c 196608 --mtp-draft 2 \
  --cont-width 2 --host 127.0.0.1 --port 8000 --no-update-check
```

### Qwen YaRN long contexts

Qwen contexts through 262,144 tokens retain the native factor-1 rotary path.
Larger server contexts select a static YaRN factor from the requested context:
factor 2 through 524,288, factor 3 through 786,432, and factor 4 through
1,048,576. This follows the
[`Qwen3.8-Flash-Next` 1M recipe](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#processing-ultra-long-texts)
and the
[`transformers` YaRN equations](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py);
the underlying method is described in the
[`YaRN` paper](https://arxiv.org/abs/2309.00071).

The 1M configuration uses one bank and a smaller prefill chunk:

```sh
DS4_SESSION_GRAPH_FIT=0 \
DS4_QWEN_BATCH=1 \
DS4_QWEN_PLE_CACHE_MB=512 \
DS4_QWEN_PLE_WORKERS=16 \
DS4_QWEN_PREFILL_CHUNK=256 \
DS4_SERVER_COALESCE_MAX=1 \
DS4_SERVER_FORK=0 \
DS4_SERVER_FORK_PARTIAL=0 \
DS4_CUDA_WEIGHT_IPC_MANIFEST="$MANIFEST" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
./ds4-server --cuda -m "$MODEL" -c 1000000 -n 256 --cont-width 1 \
  --host 127.0.0.1 --port 8000 --no-update-check
```

`DS4_SESSION_GRAPH_FIT=0` is an explicit fit-check override, not a claim that
the requested context fits the machine. On a 128 GB DGX Spark, the Q5+Sidecar
run recorded the following staged boundary on 2026-09-01:

| Configured context | YaRN factor | Largest prompt run | Result |
|---:|---:|---:|---|
| 196,608 | 1 | text and JPEG smoke | PASS, native-context regression |
| 524,288 | 2 | 524,240 tokens | HTTP 200, 215.4 prefill tok/s, zero census faults |
| 1,000,000 | 4 | 300,040 tokens | HTTP 200, 261.4 prefill tok/s, text/JPEG smoke, zero census faults |

The 524K run peaked at about 30.6 GiB in the worker and finished 47 tokens
below its context cap. A complete 1M-token prompt is **not** claimed: its
53.56 GiB graph plan plus the roughly 80.65 GiB weight owner exceeds the
machine's 121.63 GiB usable unified-memory budget. Use the native context for
ordinary short requests because static YaRN can reduce short-context quality.

Large GGUFs can exhaust unified or system memory. During validation, load one
production model at a time, observe accelerator activity and per-process memory
with tools available on your platform, and confirm serving processes have
exited before reclaiming host resources.

### Qwen derivatives

The same `qwen4exp` runtime also accepts these explicit Q5 layouts:

| Artifact | Evidence and limits |
|---|---|
| [Qwen3.8 Flash Next Uncensored](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF) | Base/Uncensored PLE comparisons and scoped snapshot, fork and restart gates are recorded separately in the [FP8 PLE guide](qwen38-ple-fp8.md). |
| [Swift1.5-Qwen3.8 Flash Next](https://huggingface.co/Baekpica/Swift1.5-Qwen3.8-Flash-Next-Mixed-Quant-GGUF) | Q5 backbone with the shared official FP8 PLE. Its published card records bounded 262,144-context, two-bank text/tool/image checks with MTP draft 2. Full-length 262K input and cross-process disk restore were not tested. Throughput remains unmeasured; the card graph is a Qwen Base reference. |

These are artifact-specific records. Base Qwen qualification does not establish
all quantizations, contexts or feature combinations for its derivatives.

## GLM 5.3 Flash release scope

RC.4 follows the explicit GLM 5.3 Flash graph and vision implementation in
the official [`antirez/ds4`](https://github.com/antirez/ds4) upstream, pinned
for this port at
[`110afdd`](https://github.com/antirez/ds4/commit/110afdd8886586f18fc9b28bc5533152dd10e728).
The Rust host keeps the KDA, DSA, hyper-connection mixing, MoE, and
[`vision encoder`](https://github.com/antirez/ds4/blob/110afdd8886586f18fc9b28bc5533152dd10e728/ds4_glm53_vision_gpu.cuh)
execution native.

The verified artifact set is exactly:

- `GLM-5.3-Flash-Q2.gguf` — 96,505,816,384 bytes;
- `GLM-5.3-Flash-Vision-Encoder.gguf` — 1,127,280,960 bytes, SHA-256
  `ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a`.

The following command reproduces the RC.4 live smoke shape:

```sh
MODEL_DIR=/path/to/GLM-5.3-Flash-Mixed-Quant-GGUF

./ds4-server --cuda \
  -m "$MODEL_DIR/GLM-5.3-Flash-Q2.gguf" \
  --vision "$MODEL_DIR/GLM-5.3-Flash-Vision-Encoder.gguf" \
  --model-id GLM-5.3-Flash-Q2 \
  -c 256 -n 8 \
  --host 127.0.0.1 --port 8000
```

The current GLM graph is serial and has an explicit 2,048-token context cap,
enforced by host admission and native session creation, including lazy graphs.
Snapshots, disk KV, continuous banks and MTP are unsupported.
OpenAI Chat text and inline PNG image requests were served live on one DGX
Spark; model-free parsing gates also cover the equivalent Responses and
Anthropic inline-image forms. PNG and JPEG are accepted, with at most four
images per request. Q4, FP8, full GLM 5.3, Metal, ROCm, distributed serving
and SSD streaming were not RC.4 gates and are not implied by this support entry.
Current boundary checks are listed in the [K2/GLM gates](releases/v0.1.3-k2-glm-gates.md).

## K2-Horizon-375B release scope

This branch follows the IFM
[`K2-Horizon-375B-A23B`](https://huggingface.co/IFM/K2-Horizon-375B-A23B)
graph: 61 full-attention GQA layers, partial NeoX RoPE on 64 of 128 dims,
three leading dense MLPs, sigmoid top-8 routing with one shared expert, and
no MTP. Execution stays native. The GGUF architecture is `k2-horizon`; it
does not widen the K-EXAONE LLLG/QK-norm contract.

The verified artifact set is exactly the public MQ87 split, 93,091,935,552
bytes (86.698621 GiB) across four shards:

- `K2-Horizon-375B-A23B-MQ87-00001-of-00004.gguf`
- `K2-Horizon-375B-A23B-MQ87-00002-of-00004.gguf`
- `K2-Horizon-375B-A23B-MQ87-00003-of-00004.gguf`
- `K2-Horizon-375B-A23B-MQ87-00004-of-00004.gguf`

Expected inventory: 842 tensors (`Q8_0=429`, `F32=239`, `IQ1_S=100`,
`IQ2_XXS=50`, `IQ1_M=16`, `IQ2_XS=8`). Official FP8 checkpoints are not a
runtime input.

On GB10, whole-map `cudaHostRegister` of the 86.70 GiB mmap fails. The
existing VMM materializer then promotes every unit (95/95, 0 cold) so CUDA
graphs never capture the unregistered mmap. The v0.1.0 gate accepted a 32K
first boot with `DS4_MEMGOV=enforce`. The 2026-09-17 native lifecycle gate
passed at context 1,024. A separate 32K explicit-serial raw disk gate passed
with three fresh processes: append and sibling requests restored 547 tokens,
and all four results matched fresh cold controls. The 4 GiB / PSI30 guard
remained active after startup pressure settled; minimum sampled availability
was 4.83 GiB. Identical whole prompts and early edits still replay cold under
K2's zero-rewind policy. This short gate does not qualify filled-32K prompts
or disk reuse through Chat or the continuous lane. See the
[commands and limits](releases/v0.1.3-k2-glm-gates.md) and
[evidence, including earlier failures](benchmarks/serving-v013-2026-09-17/k2.json).

The following command reproduces the historical continuous serving shape with
in-process VMM. External weight-owner import remains unqualified. The capability
marker for snapshots/disk KV remains conservatively `present`; the narrower
serial raw disk gate above has its own qualification. Context and concurrency
remain 32K and one bank; K2 has no MTP contract.

```sh
MODEL=/path/to/K2-Horizon-375B-A23B-Mixed-Quant-GGUF/K2-Horizon-375B-A23B-MQ87-00001-of-00004.gguf

./ds4-server --cuda \
  -m "$MODEL" \
  --model-id K2-Horizon-375B-A23B-MQ87 \
  -c 32768 --cont-width 1 \
  --host 127.0.0.1 --port 8000
```

CLI 32K raw-token smoke returned token `33785` with default memgov. HTTP
Chat, XML tool call/result continuation, streaming, and concurrent requests
passed on the same one-bank 32K setup. Official IFM `high` thinking is the
gated path. The 524,288-token metadata context, `low`/`medium` think
variants, other quants, Metal, ROCm, and distributed serving were not
gates and are not implied by this support entry.

## MiMo release scope

### MiMo RL

The [RL serving report](mimo2-serving-2026-09-25.md) records a 256K two-bank
text plan with MTP off: partial reuse, disk continuation and serial
image/video/audio input passed on GB10. The prior 512K serial-text and 256K
serial-media/DFlash gates are separate. A 1M one-bank text plan answered a
1,040,506-token prompt; two banks did not fit.

### MiMo MOPD

[MiMo-V2.6-Flash-MOPD mixed quant](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-MOPD-Mixed-Quant-GGUF)
uses the `mimo2` architecture and the RL artifact's mixed-quant tensor recipe,
with its own weights, media projector and DFlash sidecar. The
[September 30 report](benchmarks/2026-09-30-mimo2-mopd-spark.md) records its
plain-text performance and exact-logit/token checks. MTP/DFlash are inactive
in those comparisons; speculative acceleration remains separately unqualified.
RL's long-context and media gates are not transferred to this artifact.

## Naive release scope

[Naive-N0.5-Flash MQ87](naive-n05-flash.md) supports continuous banks, partial
checkpoints, disk KV and an external DSpark sidecar. Main-only buffered
retrieval and disk continuation passed at 256K/two banks and 512K/one bank.
DSpark acceleration remains unqualified; see the guide's commands and gates.

## Common serving surface

Every family uses the same `ds4-server` [HTTP surfaces](ds4-api-surface-matrix.md).
Native state, lane eligibility and supported input modalities remain explicit.

The model-family dispatch covers prompt rendering, generated-message parsing,
tool-call syntax, streaming tails, thinking controls, and generation stop
tokens. `--model-id` sets the `/v1/models` id for every family. When it is
omitted, the server parses the GGUF path: a parent directory ending in
`GGUF` or containing `Mixed-Quant` (the usual artifact bucket) wins,
otherwise the file stem with any `-00001-of-00011` shard suffix removed.
A listening port is not an acceptance result; `/v1/models`, a real
generation request, and settled `/v1/stats` counters must all pass.

## Disk-KV contract and limits

`--kv-disk-dir` and `--kv-disk-space-mb` are shared host policy. Payload support
is family-specific: the recorded lifecycle gates cover DeepSeek compressed
KV, Solar recurrent KDA plus GQA state, EXAONE full/sliding LLLG rings,
Motif latent KV plus rotated `k_pe` rings, and dots3 full/SWA latent KV plus
DSA keys. Tagged layouts and bounded payload ranges protect restores.

Qwen's continuous bank runtime uses the configured/native-fitted `max_seq`,
including width one. Its [bank payload restore](../crates/ds4-core/src/batch.rs)
goes directly to the native loader. The standalone Rust `Session` APIs use
Rust's [payload prefix parser](../crates/ds4-core/src/payload.rs); v0.1.0 adds
its missing `QWN3` family mapping. This also repairs serial disk caching for
non-streaming, non-thinking, tool-free Chat with `return_token_ids=true`,
which [routes to serial](../crates/ds4-server/src/route/mod.rs) even when a
continuous lane is available.

The release gate saved and restored a 512-token Qwen Q5 session, including a
bounded payload inside another file. Whole-file and fresh-session range loads
preserved all 248,320 frontier logits and eight greedy decode steps exactly.
An HTTP restart reused all 916 prompt tokens; the pre-fix server rejected the
same saved family and recomputed them. An exactly full 2,048-token context
also restored all frontier logits exactly through both load paths. A fresh
`max_seq=1` continuous-bank restart restored 920 tokens and computed a
21-token suffix. See the [release ledger](releases/v0.1.0.md) and the
[historical bank gates](rust-migration/QWEN_V065_RESTAMP_2026-08-31.md).

GLM 5.3 rejects snapshots and disk KV; its serial context cap is 2,048,
including lazy graph creation. K2's qualified scope remains one bank at 32K
with in-process VMM and no MTP. K2 snapshots and disk KV are implemented but
marked `present`; their lifecycle and external weight-owner import are not
qualified. The [K2/GLM gates](releases/v0.1.3-k2-glm-gates.md) distinguish
short native checks, restart reuse and context-boundary checks.

Example for a validated payload family, within its measured context limit:

```sh
./ds4-server -m "$MODEL" --cuda -c 131072 \
  --kv-disk-dir /path/to/ssd/ds4-kv --kv-disk-space-mb 32768
```

Successful loads remain on disk until the configured space-budget eviction
removes them, so more than one restart can reuse a prefix. The cache is ordinary
SSD persistence, not active-bank offload: context length and concurrency must
still fit unified memory before the worker starts. Its quant identity comes from
the first populated routed-expert layer, including dense-first model families.

## Partial prefix reuse

Solar and Motif-3 continuous banks additionally reuse prompts that diverge
inside a retained conversation. Both families share a
32-slot, demand-mapped, LRU checkpoint pool (`ds4_partial_checkpoint`):
Solar snapshots its 157.5 MiB KDA recurrent state, Motif-3 only each SWA
layer's 128-row window (39 layers, 5.48 MiB/slot). Request boundaries are
semantic checkpoints; long prefills and decode add stride-aligned ones
(`max(4096, ctx/24)` rounded to 4096). A partial fork restores the nearest
checkpoint at or below the token LCP, copies the positional rows (Solar GQA,
Motif-3 full-attention latent) from the source bank, and replays only the
gap. `DS4_SERVER_FORK_PARTIAL=0` disables capture and even the VA
reservation.

K-EXAONE captures its 36 local LLLG windows and copies the full-attention
prefix. Enable this `present` capability with `--prefix-reuse partial` on
the bank lane; `auto` still selects qualified exact reuse. Wrapped native
fork/truncate checks matched full-vocabulary logits and 16 greedy tokens.
A two-bank, 1,024-context HTTP check matched cold output after append and
edit; restart restored 304 of 327 prompt tokens. These are functional checks,
not speed measurements. The [scoped evidence](benchmarks/serving-v013-2026-09-17/exaone.json)
predates the required disk identity footer; the integrated restart gate
remains separate.

dots3-note partial reuse is also opt-in and marked `present`. It captures
33 local MLA windows and copies full MLA and DSA prefix rows. K2's
full-attention contract remains exact-only.

Verified on this host: Solar 6K/10K branches of a 12K source 2.85x/4.62x
TTFT (`docs/solar-partial-reuse-2026-08-21.md`); Motif-3 7.1K/14.1K
branches of a 16.8K source 2.18x/6.50x TTFT, byte-identical output, +0.23%
capture cost (`docs/motif3-partial-reuse-2026-08-22.md`).

## dots3 serving

dots3-note MQ87 is text-only CUDA. Banks, partial reuse and MTP are separate
`present` capabilities. `--max-seqs auto` keeps the serial default, and
`--mtp-mode auto` leaves MTP off. Explicit selections report their unqualified
status in the serving plan.

Choose text banks with ordinary decoding:

```sh
./ds4-server --cuda -m "$MODEL" --ctx 4096 --max-seqs 2 \
  --prefix-reuse partial --mtp-mode off
```

Or enable the embedded predictor on one serial session:

```sh
./ds4-server --cuda -m "$MODEL" --ctx 4096 --max-seqs 1 \
  --prefix-reuse exact --mtp-mode on --mtp-draft 3
```

The MTP draft limit is three tokens. MTP with multiple banks or an external
`--mtp` file is rejected. Each bank owns its complete runtime workspace;
weights are shared and the partial-checkpoint pool is priced separately.

Plain and MTP snapshots have distinct layouts and cannot be loaded across
those modes; bank payloads contain plain target state. Serial MTP extension
from an unfinished prefill chunk currently replays the prompt from zero.
An aligned frontier can extend directly. Replayed tokens report zero cached
tokens. Greedy HTTP checks require `reasoning_effort: "none"` as well as
`temperature: 0`; the default thinking mode is a different sampling path.

The [September 17 evidence](benchmarks/serving-v013-2026-09-17/dots3.json)
covers 4K two-bank append/edit/fork/restart and identity rejection, plus
serial MTP/plain cold parity, restart and disconnect recovery. Native tests
crossed the local-ring and DSA boundaries, checked all accepted-prefix
lengths, and clipped trials to a real 32-token context tail. The edited
arithmetic answer was wrong on both bank paths; counting outputs matched
but hit their length limit. These are functional checks, not broad quality
or long-context qualification.

No speedup is claimed: the recorded serial MTP samples were slower than
plain decoding. The
[September 6 measurements](dots3-optimization-2026-09-06.md) cover serial
plain decoding. The [bank](../tests/test_dots3_batch.c) and
[MTP](../tests/test_dots3_mtp.c) numerical/lifecycle gates are separate.

## Weight owner and inference worker

Use the [shared owner/worker procedure](serving-contract.md#weight-owner-and-worker).
Family-specific sidecars and launch limits remain below and in the linked recipes.

For a split model, `MODEL` is its first shard. DeepSeek can place a DSpark
drafter beside the base model; the standard launch resolver attaches it
automatically when its expected file name is present. Inkling accepts its exact
MTP-BF16 sidecar; use the [full base+MTP owner launch](inkling-small.md#serving)
for the tested configuration. Step accepts its documented Q8 MTP sidecar.
Naive accepts its Q8 DSpark sidecar through `DS4_DSPARK_MODEL`; see
[Naive serving](naive-n05-flash.md#serving).
Other families do not accept external MTP or DSpark attachments; dots3-note
uses only its [embedded serial predictor](#dots3-serving).

## DGX Spark memory hygiene

Follow the [shared model replacement procedure](serving-contract.md#replacing-a-model)
and [host memory guard](host-memory-guard.md). Preserve unrelated owners and workers.

## Current limits

- dots3-note is text-only, with serial default and explicit bank/MTP paths
  described [above](#dots3-serving). The source 524,288-token metadata is
  preserved. Historical 262,144-context allocation and short 4K server checks
  do not qualify a 524,288-token prefill or the new bank/MTP paths.
- dots3-note Spark throughput (2026-09-06, `docs/dots3-optimization-2026-09-06.md`):
  8,192-token cold prefill 278.3 → 604.3 tok/s and greedy decode 11.66 →
  16.78 tok/s on the serial lane after six rounds (tensor-core latent
  attention / value / absorb, split-K decode attention, grouped decode
  value projection, fused attention-side launches). Short-context only.
- dots3-note DSA above top-2048 has a deterministic 2,600-token smoke; exact
  CPU/GPU parity is gated in the dense-equivalent range at or below 2,048.
- The [historical Motif-3 evidence](model-family-history.md#motif-3-196k-multi-bank-serving-evidence) records three persistent banks at
  `-c 196608` on the reference Spark. It is not a new candidate memory gate;
  concurrent 256K banks are not claimed.
- The Motif-3 256K result validates one strict serial request on this exact
  artifact and GB10 host. It does not validate concurrent 256K banks or other
  accelerators.
- Motif-3 serving uses plain decoding; it does not accept MTP or DSpark
  support models.
- Historical Solar Open2 serving evidence includes `-c 196608` with three
  banks. The source 1,048,576-token metadata is not a measured Spark pass.
  Uncapped 64K host freezes are recorded in the
  [September 7](solar-open2-optimization-2026-09-07.md#campaign-closure-and-limits)
  and [September 12 round 5](solar-open2-optimization-2026-09-12-r5.md)
  reports. The [14 September campaign](solar-open2-optimization-2026-09-14.md)
  completed 64K WS under a 300–2200 MHz cap; keep that cap on GB10.
- Solar, EXAONE, and Motif-3 serial snapshots now reject corrupted family tags,
  and their continuous banks restore into a different idle bank before a
  one-token warm suffix. The CUDA lifecycle gates passed on the production
  mixed-quant GGUFs. DeepSeek retains its compressed-KV format; GLM session
  snapshots are unsupported.
- The 2026-08-15 Motif restart gate persisted a 738-token bank as 43.82 MiB,
  then restored all 738 cached tokens in 33.1 ms and computed only the
  24-token suffix. The cache file remained after the successful load.
- Disk KV reduces repeated prefill across eviction or restart. It does not lower
  the resident KV allocation of a live bank. Select context and bank width
  from the exact artifact, memory policy and measured headroom.
- Model cards contain only verified behavior and performance. Profiling
  results, failed experiments, and proposed kernels belong in the technical
  reports until a release gate validates them.

## Profiling order

Use the [ds4-perf profiling guide](prefill-decode-optimization-playbook.md)
for the instrumented benchmark, doctor/scout commands and evidence rules.

Do not use a 256K run as the first performance experiment. Establish a short
correctness baseline, profile an 8K or 16K prefill and a separate decode
window with Nsight Systems, then use Nsight Compute only on kernels that rank
as material bottlenecks. Keep one change per measurement and require both
the focused fixture and a full-model A/B before changing the default path.

## Historical integration evidence

The inherited reports now live in [model-family history](model-family-history.md):
C releases v0.6.3, v0.6.2, v0.6.0 and v0.5.6.3, plus the recorded Motif/Solar
throughput and bank gates. Their measurements and workload limits are preserved.

<a id="integration-evidence-for-v063-dfm"></a>
<a id="integration-evidence-for-v062-dfm"></a>
<a id="integration-evidence-for-v060-dfm"></a>
<a id="integration-evidence-for-v0563-dfm"></a>
<a id="motif-3-dgx-spark-performance-evidence"></a>
<a id="motif-3-remesure-on-the-v062-dfm-line-2026-08-21"></a>
<a id="motif-3-196k-multi-bank-serving-evidence"></a>
<a id="solar-open2-dgx-spark-performance-evidence"></a>
