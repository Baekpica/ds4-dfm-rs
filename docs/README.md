# Documentation

Start with the [repository README](../README.md). Use the guides below for
operation and development; dated reports retain their recorded commit,
artifact and workload. They do not describe live processes or qualify a new build.

<a id="current-guides"></a>

## Start here

| Goal | Guide |
|---|---|
| Build and run | [Quick start](../README.md#quick-start) |
| Select a model and download its artifact | [Model Zoo](../README.md#model-zoo) and [family contracts](ds4-dfm-model-families.md) |
| Understand the performance examples | [Performance references](performance.md) |

## Operation

| Guide | Covers |
|---|---|
| [Serving contract](serving-contract.md) | Shared options, requested/effective/qualified plans, owner/worker and reuse |
| [Generated capabilities](serving-capabilities.md) | Runtime lanes, modes, bounds and recorded qualification |
| [Chat templates](chat-templates.md) | Official input assets, normalization and continuation |
| [API matrix](ds4-api-surface-matrix.md) | HTTP objects, routing, media forms and unsupported requests |
| [Memory guard](host-memory-guard.md) | Large-model admission, job limits and cleanup |

## Models

Choose the exact artifact from the [Model Zoo](../README.md#model-zoo).
The family guide owns shared state contracts and the Qwen/GLM/K2 recipes.
Other recipes and dated gates retain their hardware and workload limits.

| Guide | Covers |
|---|---|
| [Model families](ds4-dfm-model-families.md) | Architecture selectors, native state, context, snapshots and reuse |
| [Qwen](ds4-dfm-model-families.md#qwen-release-scope) | Q5, images, owner/cache setup and YaRN |
| [Qwen derivatives](ds4-dfm-model-families.md#qwen-derivatives) | Uncensored and Swift artifact-specific evidence |
| [Qwen FP8 PLE](qwen38-ple-fp8.md) | Sidecar selection, KV compatibility and paired card sweeps |
| [GLM](ds4-dfm-model-families.md#glm-53-flash-release-scope) | Exact Q2 + vision, serial graph and 2K cap |
| [K2](ds4-dfm-model-families.md#k2-horizon-375b-release-scope) | MQ87, IFM tools, one-bank serving and disk limits |
| [dots3](ds4-dfm-model-families.md#dots3-serving) | Separate opt-in text banks and serial MTP |
| [Bonsai](BONSAI.md) | Pinned ternary PQ2_0, CPU reference and serial CUDA |
| [Inkling](inkling-small.md) | MQ85GB/MTP, media input and serial serving |
| [Step](step37-serving-2026-09-13.md) | MQ83, opt-in banks/MTP, partial fork and disk KV |
| [Ling](ling3-flash-vl.md) | MQ-Q5, images, persistent banks and YaRN 256K |
| [MiMo RL](mimo2-serving-2026-09-25.md) | Recorded mixed serving, bank reuse and memory limits |
| [MiMo MOPD](ds4-dfm-model-families.md#mimo-mopd) | Separate artifact, text performance and DFlash boundaries |
| [Naive](naive-n05-flash.md) | MQ87, banks, disk KV, long-context gates and drafter limits |

## Development

| Guide | Covers |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Ordered host checks, native gates and evidence requirements |
| [Repository instructions](../AGENTS.md) | Shared development and agent rules |
| [Architecture](rust-migration/ARCHITECTURE.md) | Rust/native ownership, crates and source layout |
| [FFI contract](rust-migration/FFI_CONTRACT.md) | Opaque inference boundary |
| [ds4-perf](ds4-perf.md) | Inspect, scout, compare, optimize and serving workloads |
| [Optimization playbook](prefill-decode-optimization-playbook.md) | Execution-path diagnosis and numerical proof |
| [Speed benchmarks](../speed-bench/README.md) | Manual sweep CSVs and plotting |
| [Lineage](LINEAGE.md) | Provenance, development context and selective upstream ports |

## Recorded evidence

### Release ledgers

| Ledger | Recorded scope |
|---|---|
| [v0.1.3](releases/v0.1.3.md) | Common serving controls and scoped P0–P4 qualification |
| [v0.1.2](releases/v0.1.2.md) | Shared Jinja and Inkling checkpoint |
| [v0.1.1](releases/v0.1.1.md) | Performance workflow and FP8 PLE |
| [v0.1.0](releases/v0.1.0.md) | Independent Rust-host production baseline |
| [K2/GLM boundary gates](releases/v0.1.3-k2-glm-gates.md) | Native lifecycle, short restart and context checks |

### Performance and serving reports

The [performance guide](performance.md) explains the README examples.
These reports preserve successful gates, rejected experiments and unresolved limits.

- **Qwen:** [September 4 long prefill](qwen38-long-context-prefill-2026-09-04.md),
  [September 6](qwen38-prefill-2026-09-06.md),
  [September 7 prefill](qwen38-prefill-2026-09-07.md),
  [September 7 image/text/agent](qwen38-image-2026-09-07.md),
  [September 14 draft/prefix](qwen38-perf-2026-09-14.md).
- **Solar:** [September 7](solar-open2-optimization-2026-09-07.md),
  [September 12 rounds 1–4](solar-open2-optimization-2026-09-12.md),
  [September 12 round 5](solar-open2-optimization-2026-09-12-r5.md),
  [September 14 capped-clock](solar-open2-optimization-2026-09-14.md),
  [August 21 partial reuse](solar-partial-reuse-2026-08-21.md).
- **Motif:** [historical throughput and integration gates](model-family-history.md),
  [August 22 partial reuse](motif3-partial-reuse-2026-08-22.md).
- **dots3:** [September 6 optimization](dots3-optimization-2026-09-06.md).
- **K2:** [September 5](k2-optimization-2026-09-05.md),
  [continuation campaign](k2-optimization-2026-09-05-cont.md).
- **Inkling:** [rounds 1–12](inkling-optimization-2026-09-10.md),
  [13–15](inkling-optimization-2026-09-11.md),
  [16–18](inkling-optimization-2026-09-11-r16.md),
  [19–21](inkling-optimization-2026-09-12.md),
  [22–24](inkling-optimization-2026-09-12-r22.md),
  [25–27](inkling-optimization-2026-09-12-r25.md),
  [September 22](inkling-optimization-2026-09-22.md).
- **Step:** [initial integration](step37-initial.md),
  [BASE campaign](step37-optimization-2026-09-13.md),
  [SWA/MTP rounds](step37-optimization-2026-09-13-r2.md),
  [capped-clock campaign](step37-optimization-2026-09-13-r3.md).
- **Ling:** [GB10 campaign and long-context sweep](ling3-flash-vl.md#cuda-campaign-gb10),
  [raw CSVs and receipt](benchmarks/ling3-flash-vl-2026-09-17/).
- **MiMo RL:** [September 23](benchmarks/mimo2-2026-09-23/README.md),
  [September 24](benchmarks/mimo2-2026-09-24/README.md),
  [September 25 serving](mimo2-serving-2026-09-25.md),
  [September 26 text](benchmarks/mimo2-2026-09-26/README.md),
  [September 26 media](benchmarks/mimo2-media-2026-09-26/README.md).
- **MiMo MOPD:** [September 30 controls and proofs](benchmarks/2026-09-30-mimo2-mopd-spark.md).
- **Naive:** [September 30 campaign](benchmarks/naive-2026-09-30/README.md),
  [serving](benchmarks/naive-2026-09-30/serving.md),
  [long-context gates](benchmarks/naive-2026-09-30/long-context.md).
- **Bonsai:** [serving](releases/bonsai-serving-2026-09-30.md),
  [prefill](releases/bonsai-perf-2026-09-30.md),
  [split-K decode](releases/bonsai-splitk-2026-09-30.md),
  [session state](releases/bonsai-session-2026-09-30.md),
  [CUDA graph](releases/bonsai-cuda-graph-2026-09-30.md),
  [CPU reference](releases/bonsai-cpu-reference-2026-09-30.md).

### Migration and inherited history

- [Migration evidence](rust-migration/README.md): frozen C oracle, host parity,
  shared engine gaps and the completed repository split.
- [Model-family history](model-family-history.md): inherited C integration and
  Motif/Solar workload reports.

Completed migration plans and old process handoffs remain in the
[pre-cleanup documentation tree](https://github.com/Baekpica/ds4-dfm-rs/tree/ac750d61ef0306d30cc595081883f61ec847c3d8/docs).
Recorded PIDs are historical.

## Design records

- [Qwen image input contract](qwen38-image-input-spec.md): original design;
  use the family guide and dated image gates for implemented scope.
- [W2A16 fused-dequant GEMM proposal](ds4-w2a16-fused-dequant-gemm-design.md):
  proposed work, not a released path.
- [DeepSeek model-card synopsis](../MODEL_CARD.md): upstream model information,
  separate from DS4 artifact and runtime validation.
