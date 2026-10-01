# DwarfStar / ds4-dfm-rs

**DwarfStar** (`ds4`) runs a deliberately limited set of large open-weight
models on local machines. Model loading, chat templates, tool calls, KV reuse,
the HTTP server and the coding agent are built and tested together.

`ds4-dfm-rs` is the independent Rust-host continuation of
[`Baekpica/ds4`](https://github.com/Baekpica/ds4). Rust owns the host and serving
policy; the optimized C/CUDA/Metal backend remains native. The release reference
is the 128 GB NVIDIA DGX Spark / GB10. Support is specific to the
[listed model artifacts](#supported-model-families) and their recorded gates.

[Quick start](#quick-start) · [Models](#supported-model-families) ·
[API](#http-compatibility) · [Documentation](docs/README.md)

## What you can do

<a id="so-what-can-i-do-with-this-software"></a>

- Serve OpenAI Chat, Completions and Responses, or Anthropic Messages.
- Run the CLI and built-in DeepSeek DSML coding agent.
- Use official [chat templates](docs/chat-templates.md), tool history and REPL turns.
- Keep conversations through persistent banks, prefix reuse and disk checkpoints
  where the model's state contract supports them.
- Use family-specific MTP, drafters and media input within their verified limits.
- Profile prefill and decode with [ds4-perf](docs/ds4-perf.md).

## Performance and release evidence

The native paths are tuned for prefill and decode on the reference hardware.
These are recorded examples in **tokens/second**, with different workloads;
the row links explain the artifact, hardware and measurement. Models without
readily documented paired results are omitted.

| Model / artifact | Prefill | Decode | Workload / evidence |
|---|---:|---:|---|
| Qwen3.8 Flash Next Q5, FP8 PLE | 1,323.1 | 28.93 | [2K–64K sweep, MTP 2](docs/performance.md#qwen) |
| Qwen3.8 Uncensored Q5, FP8 PLE | 1,310.8 | 28.96 | [2K–64K sweep, MTP 2](docs/performance.md#qwen-uncensored) |
| Solar Open2 250B MXQ-v1 | 1,095.61 | 17.43 | [8K + 64, plain](docs/performance.md#solar) |
| Motif-3 MQ87-88 | 627.19 | 15.06 | [8K + 64, historical C](docs/performance.md#motif) |
| dots3-note MQ87 | 604.3 | 16.78 | [8K + 64, plain](docs/performance.md#dots3) |
| K2-Horizon MQ87 | 641.94–644.78 | 13.07–13.34 | [8K + 64, two samples](docs/performance.md#k2) |
| Inkling Small MQ85GB | 452.58 | 13.07 | [8K + 64, plain](docs/performance.md#inkling) |
| Step 3.7 Flash MQ83 | 1,194.64 | 22.77 | [2K + 64, MTP 3](docs/performance.md#step) |
| Ling-3.0-flash-VL MQ-Q5 | 1,889 | 24.65 | [8K + 64, plain](docs/performance.md#ling) |
| MiMo-V2.6-Flash-RL mixed quant | 1,217.76 | 24.44 | [8K + 128, plain](docs/performance.md#mimo) |
| MiMo-V2.6-Flash-MOPD mixed quant | 1,206.29 | 24.18 | [8K + 128, plain](docs/performance.md#mimo-mopd) |
| Naive-N0.5-Flash MQ87 | 506.16 | 17.44 | [8K + 32, plain](docs/performance.md#naive) |
| Prism Bonsai 2 27B PQ2_0 | 1,023.3 | 17.50–18.00 | [2,140 + 64, RTX 4070 SUPER](docs/performance.md#bonsai) |

![Qwen3.8 Flash Next Q5 paired BF16 and FP8 PLE throughput](docs/qwen38-ple-fp8-base.png)

*Qwen on one DGX Spark. [Results, conditions and raw evidence](docs/performance.md#qwen).*

The [performance guide](docs/performance.md) collects these references and the
other published curves. The [release ledgers](docs/README.md#release-ledgers)
record qualification separately from throughput.

## Quick start

### Build

The repository pins Rust 1.98.0, `rustfmt` and `clippy` in
[`rust-toolchain.toml`](rust-toolchain.toml). CUDA builds require the local
CUDA toolkit and C/C++ build tools.

```sh
git clone https://github.com/Baekpica/ds4-dfm-rs.git
cd ds4-dfm-rs
make cuda-spark
```

| Target | Backend |
|---|---|
| `make cuda-spark` | DGX Spark / GB10, `sm_121a` |
| `make cuda-generic` | Detected local NVIDIA GPU |
| `make cuda CUDA_ARCH=sm_N` | Explicit CUDA architecture |
| `make` on macOS | Inherited Metal backend |
| `make cpu` | CPU reference and diagnostics |

Production binaries are `ds4`, `ds4-server`, `ds4-bench` and `ds4-agent`, all
Rust hosts. The `*-c` binaries and `ds4-eval` remain C behavior oracles.
The old `*-rs` names are compatibility aliases.

### Run and verify

Models are downloaded separately. Use a [supported artifact](#model-zoo),
provide its template and required sidecars, and pass the first shard with `-m`.
This short text example uses a 2,048-token context; select longer contexts and
additional features from the [family guide](docs/ds4-dfm-model-families.md).

```sh
MODEL=/path/to/supported-model-00001-of-000NN.gguf

./ds4-server --cuda -m "$MODEL" -c 2048 -n 128 \
  --host 127.0.0.1 --port 8000 --model-id local-model --no-update-check
```

Check discovery, runtime state and a real generation:

```sh
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/stats
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}]}'
```

The four production binaries' `--help` output is the authoritative flag
reference. Use `--check-config` to inspect requested, effective and qualified
settings; see the [serving contract](docs/serving-contract.md#inspect).

<a id="weight-owner-and-worker"></a>
<a id="qwen-yarn-long-contexts"></a>

For shared weights and longer sessions, follow the
[owner/worker launch](docs/serving-contract.md#weight-owner-and-worker),
[Qwen cache and YaRN setup](docs/ds4-dfm-model-families.md#qwen-release-scope)
and [memory guard](docs/host-memory-guard.md).

## Supported hardware and backends

| Backend | Scope |
|---|---|
| NVIDIA DGX Spark / GB10 | Release reference; see each artifact's recorded workload gates. |
| Other NVIDIA CUDA GPUs | Build path retained; qualification requires device-specific checks. Bonsai has recorded RTX 4070 SUPER evidence. |
| macOS Metal | Inherited build and source path; separate live checks required. |
| CPU | Reference and debugging. |

Large models, active KV and workspaces must fit together. Disk KV persists
checkpoints; it does not offload live banks. Inspect ownership and memory before
loading a model, and preserve unrelated servers and resident owners.

## Supported model families

<a id="model-zoo"></a>

Only explicit GGUF layouts and execution paths are accepted. The table combines
the supported artifacts and Model Zoo; see also the
[`DS4-Mixed-Quant-for-Spark`](https://huggingface.co/collections/Baekpica/ds4-mixed-quant-for-spark)
collection. Feature and context limits remain specific to each artifact.

| Model | GGUF architecture | Artifact | Runtime guide / scope |
|---|---|---|---|
| DeepSeek V4 Flash / PRO | `deepseek4` | [antirez GGUF](https://huggingface.co/antirez/deepseek-v4-gguf/tree/main) | Flash live oracle, DeepSeek MTP and DSpark |
| Solar Open2 250B | `solar-open2` | [Baekpica mixed quant](https://huggingface.co/Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF) | KDA/GQA state and persistent banks |
| K-EXAONE 236B A23B | `exaone-moe` | [Baekpica mixed quant](https://huggingface.co/Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF) | LLLG KV; [partial reuse limits](docs/ds4-dfm-model-families.md#partial-prefix-reuse) |
| Motif-3 | `motif3` | [Baekpica mixed quant](https://huggingface.co/Baekpica/Motif-3-Mixed-Quant-GGUF) | Latent KV, SWA rings and persistent banks |
| dots3-note Preview | `dots3note` / `dots3-note` | [Baekpica mixed quant](https://huggingface.co/Baekpica/dots3-note-prev-Mixed-Quant-GGUF) | [Serial default; opt-in banks and MTP](docs/ds4-dfm-model-families.md#dots3-serving) |
| Qwen3.8 Flash Next | `qwen4exp` | [Baekpica Q5 + SSD-PLE](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF) | [BF16/FP8 PLE, embedded MTP, banks and images](docs/ds4-dfm-model-families.md#qwen-release-scope) |
| Qwen3.8 Flash Next Uncensored | `qwen4exp` | [Baekpica Q5 + SSD-PLE](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF) | [Separate Base/Uncensored gates](docs/qwen38-ple-fp8.md) |
| Swift1.5-Qwen3.8 Flash Next | `qwen4exp` | [Baekpica Q5 + FP8 SSD-PLE](https://huggingface.co/Baekpica/Swift1.5-Qwen3.8-Flash-Next-Mixed-Quant-GGUF) | [Qwen runtime; bounded serving, throughput unmeasured](docs/ds4-dfm-model-families.md#qwen-derivatives) |
| Prism Bonsai 2 27B | `qwen35` | [Pinned PQ2_0](docs/BONSAI.md) | Serial CUDA text and CPU reference; [limits](docs/BONSAI.md) |
| GLM 5.3 Flash | `glm5-next` | [antirez Q2](https://huggingface.co/antirez/glm-5.3-flash-gguf/blob/main/GLM-5.3-Flash-Q2.gguf) + [vision](https://huggingface.co/antirez/glm-5.3-flash-gguf/blob/main/GLM-5.3-Flash-Vision-Encoder.gguf) | [Serial text/image; 2,048-token cap](docs/ds4-dfm-model-families.md#glm-53-flash-release-scope) |
| K2-Horizon 375B A23B | `k2-horizon` | [Baekpica MQ87](https://huggingface.co/Baekpica/K2-Horizon-375B-A23B-Mixed-Quant-GGUF) | [32K one bank, IFM tools, no MTP](docs/ds4-dfm-model-families.md#k2-horizon-375b-release-scope) |
| Inkling Small | `inkling` | [Baekpica MQ85GB](https://huggingface.co/Baekpica/Inkling-Small-Mixed-Quant-GGUF/tree/main/MQ85GB) + [MTP-BF16](https://huggingface.co/Baekpica/Inkling-Small-GGUF/tree/main/MTP-BF16) | [Serial text/image/audio input](docs/inkling-small.md) |
| Step 3.7 Flash | `step35` | [Baekpica MQ83 + MTP + vision](https://huggingface.co/Baekpica/Step-3.7-Flash-Mixed-Quant-GGUF) | [Opt-in text banks/MTP, disk KV; images serial](docs/step37-serving-2026-09-13.md) |
| Ling-3.0-flash-VL | `bailingmoe3` | [Baekpica MQ-Q5 + BF16 mmproj](https://huggingface.co/Baekpica/Ling-3.0-flash-VL-Mixed-Quant-GGUF) | [Images, persistent banks, disk KV and YaRN](docs/ling3-flash-vl.md) |
| MiMo-V2.6-Flash-RL | `mimo2` | [Baekpica mixed quant](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF) | [256K two-bank text and serial media; longer-context limits](docs/mimo2-serving-2026-09-25.md) |
| MiMo-V2.6-Flash-MOPD | `mimo2` | [Baekpica mixed quant](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-MOPD-Mixed-Quant-GGUF) | [Own text/performance gates; drafting remains separate](docs/ds4-dfm-model-families.md#mimo-mopd) |
| Naive-N0.5-Flash | `naive_n05_flash` | [Baekpica MQ87](https://huggingface.co/Baekpica/Naive-N0.5-Flash-Mixed-Quant-GGUF) | [Banks, partial reuse and disk KV; draft acceleration unqualified](docs/naive-n05-flash.md) |

<a id="qwen-release-scope"></a>
<a id="glm-53-flash-release-scope"></a>
<a id="k2-horizon-375b-release-scope"></a>

Detailed [Qwen](docs/ds4-dfm-model-families.md#qwen-release-scope),
[GLM](docs/ds4-dfm-model-families.md#glm-53-flash-release-scope) and
[K2](docs/ds4-dfm-model-families.md#k2-horizon-375b-release-scope) artifact and
release scopes live in the [model-family guide](docs/ds4-dfm-model-families.md).
The [generated capabilities](docs/serving-capabilities.md) describe the runtime
plan; dated family gates retain their narrower workload boundaries.

## HTTP compatibility

| Surface | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Completions | `POST /v1/completions` |
| OpenAI Responses | `POST /v1/responses` |
| Anthropic Messages | `POST /v1/messages` |
| Model discovery | `GET /v1/models` |
| Runtime state | `GET /v1/stats`, `GET /metrics` |

Buffered and SSE responses retain each surface's tool, reasoning, finish and
error forms. Native state and serial/continuous/static routing stay
family-specific. The [API matrix](docs/ds4-api-surface-matrix.md) documents
supported fields, media shapes, routing and explicit refusals.

<a id="compatibility-boundary"></a>

The server is one trust domain and has no authentication or tenant isolation.
Use an authenticating proxy or a separate server per trust domain. See the
[compatibility boundary](docs/ds4-api-surface-matrix.md#compatibility-boundary)
for CLI, HTTP, cache and distributed contracts.

## Architecture

```text
CLI / HTTP / agent → ds4-core → ds4-sys → native CUDA / MMQ / VMM
                    Rust host policy      native compute and GPU state
```

Rust owns catalog validation, rendering, admission, scheduling, model/session
lifetime and KV policy. GGUF loading remains mmap-backed; device handles stay
behind the opaque [native bridge](native/bridge/ds4_bridge.h).
See [architecture](docs/rust-migration/ARCHITECTURE.md) and
[FFI rules](docs/rust-migration/FFI_CONTRACT.md).

## Documentation

| Guide | Use it for |
|---|---|
| [Documentation index](docs/README.md) | Find current guides, releases and dated reports |
| [Serving contract](docs/serving-contract.md) | Options, resolved plans, owner/worker and reuse |
| [Model families](docs/ds4-dfm-model-families.md) | Artifacts, media, context and state limits |
| [Chat templates](docs/chat-templates.md) | Official assets and continuation |
| [Performance](docs/performance.md) | Representative results and measurement conditions |
| [Memory guard](docs/host-memory-guard.md) | Large-model admission and operational limits |
| [Changelog](CHANGELOG.md) | Independent and inherited release history |

<a id="status"></a>

The current repository version is **v0.1.3**. Its
[ledger](docs/releases/v0.1.3.md) records common serving controls and scoped
P0–P4 gates. Earlier ledgers preserve the Rust baseline, performance workflow
and shared Jinja qualification. Dated reports describe their recorded build,
artifact and workload; they do not describe live processes.

## Contributing

<a id="testing"></a>
<a id="repository-layout"></a>
<a id="design-philosophy"></a>
<a id="how-to-use-this-project"></a>

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).
The contribution guide owns the ordered host-parity checks and native gates;
the [architecture guide](docs/rust-migration/ARCHITECTURE.md) maps the crates.
Family additions need explicit artifact/state contracts. Inference changes
need correctness and measured prefill/decode evidence on the affected path.

## Lineage

<a id="motivations"></a>
<a id="why-this-repository-exists"></a>
<a id="ai-full-disclosure"></a>

```text
antirez/ds4 → Entrpi/ds4 → Baekpica/ds4 (DFM) → Baekpica/ds4-dfm-rs
```

The project follows useful open weights that fit personal and workstation
machines. It keeps model mechanics explicit and adds only the abstractions
shared by proven families. DFM refers to 독자 파운데이션 모델 (독파모).
The split preserves Git ancestry, authorship and the native backend;
[LINEAGE.md](docs/LINEAGE.md) records the exact refs and port policy.

Development uses coding-agent assistance, with humans owning scope, ideas,
testing and debugging. See the [development context](docs/LINEAGE.md#development-context).

## License and acknowledgements

MIT; see [LICENSE](LICENSE). Existing notices are preserved. This project
builds on [antirez/ds4](https://github.com/antirez/ds4),
[Entrpi/ds4](https://github.com/Entrpi/ds4),
[Baekpica/ds4](https://github.com/Baekpica/ds4), and the engineering and
selected kernels of [llama.cpp](https://github.com/ggml-org/llama.cpp) / GGML.
The [MMQ vendor record](cuda/mmq/VENDOR.md) retains the upstream pin and notices.
