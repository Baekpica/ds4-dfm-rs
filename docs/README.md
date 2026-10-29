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
| [MiniMax-H3 API](minimax-h3/api-usage.md) | Start the separate video host, create/poll/download jobs and manage retention |
| [SSD expert streaming](ssd-streaming.md) | Bounded GLM expert cache, options and memory policy |
| [GLM Prefill on Spark, 2026-10-08](glm53-prefill-2026-10-08.md) | Large batches, partial offload, numerical contracts and measured adoption |

## Conversion tools

| Guide | Covers |
|---|---|
| [VQ calibration](vq-calibration.md) | `tools/vq_calib/`: activation statistics, importance-weighted Lloyd for shared-codebook GVQ, 9/10-bit GVQ v2, the A.X-K2 VQ pipeline |

## Models

Choose the exact artifact from the [Model Zoo](../README.md#model-zoo).
The family guide owns shared state contracts and the Qwen/GLM/K2 recipes.
Other recipes and dated gates retain their hardware and workload limits.

| Guide | Covers |
|---|---|
| [Model families](ds4-dfm-model-families.md) | Architecture selectors, native state, context, snapshots and reuse |
| [Qwen](ds4-dfm-model-families.md#qwen-release-scope) | Q5, images, owner/cache setup and YaRN |
| [Qwen derivatives](ds4-dfm-model-families.md#qwen-derivatives) | Uncensored, Swift, Darwin and ThinkingCap artifact-specific evidence |
| [Qwen FP8 PLE](qwen38-ple-fp8.md) | Sidecar selection, KV compatibility and paired card sweeps |
| [Qwen benchmark reference](qwen38-benchmark-reference.md) | Consolidated source limits, protocols, campaign numbers, candidate adoption and optimization target |
| [GLM](ds4-dfm-model-families.md#glm-53-flash-release-scope) | Q2 + vision; recorded serial 2K qualification |
| [GLM Uncensored](glm53-uncensored.md) | Mixed recipe, SSD/bank contracts and current verification boundaries |
| [GLM VQ](glm53-vq.md) | GVQ1 format concept, GB10 execution paths, numerical contract and optimization log |
| [K2](ds4-dfm-model-families.md#k2-horizon-375b-release-scope) | MQ87, IFM tools, one-bank serving and disk limits |
| [dots3](ds4-dfm-model-families.md#dots3-serving) | Separate opt-in text banks and serial MTP |
| [Bonsai](BONSAI.md) | Pinned ternary PQ2_0, CPU reference and serial CUDA |
| [Inkling](inkling-small.md) | MQ85GB/MTP, media input and serial serving |
| [Step](step37-serving-2026-09-13.md) | MQ83, opt-in banks/MTP, partial fork and disk KV |
| [Ling](ling3-flash-vl.md) | MQ-Q5, images, persistent banks and YaRN 256K |
| [Ling-3.1 VQ](ling31-flash-vq.md) | GVQ1 release directory (v2 recipe), FP8 latent KV, embedded MTP, 262K banks beside a 106 GiB trunk, 2K–64K sweep |
| [Beam-1.0 VQ](beam-vq.md) | GVQ1 release directory, FP8 full-layer KV, Harmony header forcing, DFlash2 drafter and its losslessness gate, 1M in one bank, 2K–64K sweep |
| [A.X-K2 VQ](axk2-vq.md) | GVQ1 v2 (9/10-bit) release directory, MLA with the DSA indexer, FP8 latent and key rows, 262K in one bank, reference-parity receipt, 2K–64K sweep |
| [A.X-K2 optimization campaign](axk2-optimization-2026-10-29.md) | Three prefill and three decode adoptions, paired evidence, costs, fallbacks and the October 29 publication sweep |
| [MiMo RL](mimo2-serving-2026-09-25.md) | Recorded mixed serving, bank reuse and memory limits |
| [MiMo MOPD](ds4-dfm-model-families.md#mimo-mopd) | Separate artifact, text performance and DFlash boundaries |
| [Naive](naive-n05-flash.md) | MQ87, banks, disk KV, long-context gates and drafter limits |
| [IQuest-Q1](iquest-q1.md) | Canonical mixed weights, hybrid Q8 KV, embedded MTP and pending Spark gates |
| [DeepSeek V4.1 Flash](ds4-dfm-model-families.md#deepseek-v41-flash-release-scope) | Serial lane, metadata context, sidecar/engram mounts and the embedded MTP contract |
| [DeepSeek V4.1 Flash (ds41)](deepseek41-flash.md) | Artifact, 40-layer shape, serving contract, gates and recorded benchmarks |
| [DeepSeek V4.1 Flash (plan)](deepseek41-port-plan.md) | Proposal: the vq8sh14 artifact, the C engine as source of truth, the gap inventory and the golden-set gate |
| [DeepSeek V4.1 post-train (plan)](ds41-posttrain-plan.md) | The ③ producer, judge, apply and supervising HTTP API: units, ports and gates |
| [MiniMax-H3](minimax-h3/README.md) | BF16 video/audio diffusion: model files, build, CLI, reference and validation |
| [MiniMax-H3 local API](minimax-h3/local-v1-contract.md) | Video tasks, conditioning, supported settings and limits |

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
| [v0.1.4](releases/v0.1.4.md) | Cumulative model/VQ support, inference work, audit repairs and bounded GB10 smoke |
| [v0.1.3](releases/v0.1.3.md) | Common serving controls and scoped P0–P4 qualification |
| [v0.1.2](releases/v0.1.2.md) | Shared Jinja and Inkling checkpoint |
| [v0.1.1](releases/v0.1.1.md) | Performance workflow and FP8 PLE |
| [v0.1.0](releases/v0.1.0.md) | Independent Rust-host production baseline |
| [K2/GLM boundary gates](releases/v0.1.3-k2-glm-gates.md) | Native lifecycle, short restart and context checks |

### Performance and serving reports

The [performance guide](performance.md) explains the README examples.
These reports preserve successful gates, rejected experiments and unresolved limits.

- **Ling-3.1 1019:** [October 28 follow-up](ling31-optimization-2026-10-28-followup.md),
  per-change throughput, numerical gates and Decode variability.
- **MiniMax-H3:** [October 26 BF16 validation](minimax-h3/validation-2026-10-26.md),
  three matched h3.c outputs, a native video API gate and measured generation cost;
  [October 27 speed baseline](minimax-h3/speed-2026-10-27.md), the port against
  h3.c on one job, phase by phase; [that job with both attention arms on one
  binary](benchmarks/h3-jobA-2026-10-27/README.md); [the attention sweep across
  sequence lengths](benchmarks/h3-attention-2026-10-27/README.md); and [why the
  tensor path is the default](minimax-h3/tensor-default-2026-10-27.md).
- **MiniMax VQ:** [October 27 prefill/decode campaign](minimax-vq-optimization-2026-10-27.md),
  three adopted rounds each; [fresh 2K–64K DSpark graph](benchmarks/minimax-vq-spark-64k-2026-10-27/README.md),
  raw clocks, exact output checks and observed repeat ranges.
- **Ponytail audit:** [October 26 repairs, cleanup and local-model 8K smoke](benchmarks/2026-10-26-ponytail-audit/README.md).
- **Qwen:** [September 4 long prefill](qwen38-long-context-prefill-2026-09-04.md),
  [September 6](qwen38-prefill-2026-09-06.md),
  [September 7 prefill](qwen38-prefill-2026-09-07.md),
  [September 7 image/text/agent](qwen38-image-2026-09-07.md),
  [September 14 draft/prefix](qwen38-perf-2026-09-14.md),
  [benchmark reference](qwen38-benchmark-reference.md),
  [October 2 Darwin images](qwen-vision-2026-10-02.md) and
  [follow-up rounds](qwen-vision-followup-2026-10-02.md).
- **Solar:** [September 7](solar-open2-optimization-2026-09-07.md),
  [September 12 rounds 1–4](solar-open2-optimization-2026-09-12.md),
  [September 12 round 5](solar-open2-optimization-2026-09-12-r5.md),
  [September 14 capped-clock](solar-open2-optimization-2026-09-14.md),
  [August 21 partial reuse](solar-partial-reuse-2026-08-21.md).
- **Motif:** [historical throughput and integration gates](model-family-history.md),
  [August 22 partial reuse](motif3-partial-reuse-2026-08-22.md).
- **GLM VQ:** [packed restore](glm53-vq-restore-2026-10-15.md),
  [register MMA prefill](glm53-vq-mma-2026-10-15.md),
  [nonexpert Q4_K candidate](glm53-vq-q4k-2026-10-15.md),
  [cooperative Q4 decode](glm53-vq-q4-stream-2026-10-16.md),
  [direct VQ expert decode](glm53-vq-decode-2026-10-16.md),
  [fused FP8 projections](glm53-vq-fp8-2026-10-16.md),
  [selective KDA Q4 assessment](glm53-vq-kda-q4-2026-10-16.md),
  [KDA Q4 speed baseline](glm53-vq-kda-q4-speed-2026-10-16.md),
  [direct BF16 KV handoff](glm53-vq-kv-bf16-2026-10-16.md),
  [latent decode attention](glm53-vq-latent-2026-10-16.md),
  [sparse attention through 8K](glm53-vq-sparse-2026-10-16.md),
  [VQ banks and snapshots](glm53-vq-state-2026-10-16.md),
  [selective DSA Q4 assessment](glm53-vq-dsa-q4-2026-10-16.md),
  [prefill chunk assessment](glm53-vq-chunks-2026-10-16.md),
  [workspace allocation assessment](glm53-vq-work-arena-2026-10-16.md).
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
- **Naive:** [October 1 additional rounds](benchmarks/naive-2026-10-01/README.md),
  [September 30 campaign](benchmarks/naive-2026-09-30/README.md),
  [serving](benchmarks/naive-2026-09-30/serving.md),
  [long-context gates](benchmarks/naive-2026-09-30/long-context.md).
- **Bonsai:** [serving](releases/bonsai-serving-2026-09-30.md),
  [prefill](releases/bonsai-perf-2026-09-30.md),
  [split-K decode](releases/bonsai-splitk-2026-09-30.md),
  [session state](releases/bonsai-session-2026-09-30.md),
  [CUDA graph](releases/bonsai-cuda-graph-2026-09-30.md),
  [CPU reference](releases/bonsai-cpu-reference-2026-09-30.md).
- **DeepSeek V4.1:** [vision review: image KV, tools and bounded numerical gates, October 14](benchmarks/ds41-vision-review-2026-10-14/README.md),
  [vision weight staging: paired TTFT, decode costs and multi-image gates, October 14](benchmarks/ds41-vision-staging-2026-10-14/README.md),
  [warm-path campaign: the asset-open cache, October 13](benchmarks/ds41-warmpath-2026-10-13/README.md),
  [2K–64K Prefill/Decode tables and graph, October 13](benchmarks/ds41-spark-64k-2026-10-13/README.md),
  [capped-clock sweep and review qualification, October 13](benchmarks/ds41-spark-review-2026-10-13/README.md),
  [five-case benchmark, October 13](benchmarks/ds41-spark-2026-10-13/README.md),
  [prefill/decode, October 11](ds41-perf-2026-10-11.md),
  [speed sweep, October 12](ds41-speed-bench-2026-10-12.md),
  [raw 22-frontier CSV](../speed-bench/gb10_spark_ds41_862fd9a6.csv),
  [r6 build CSV](../speed-bench/gb10_spark_ds41_1e7d8f24.csv),
  [engine/fork compare chart](../speed-bench/ds41_engine_vs_fork_compare.png).

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
