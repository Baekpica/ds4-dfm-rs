# Documentation

Use the current guides for development and operation. Dated reports describe
their recorded commit, model artifact and workload; they do not qualify a new
release candidate or describe live processes.

## Current guides

| Document | Purpose |
|---|---|
| [Repository README](../README.md) | Build, run, validated model scope and limitations |
| [Repository instructions](../AGENT.md) | Shared agent/development rules; AGENTS.md and CLAUDE.md point here |
| [Contributing](../CONTRIBUTING.md) | Host CI, native gates and performance validation |
| [Architecture](rust-migration/ARCHITECTURE.md) | Rust host and native compute ownership |
| [FFI contract](rust-migration/FFI_CONTRACT.md) | Opaque native inference boundary |
| [v0.1.0 release ledger](releases/v0.1.0.md) | Production baseline qualification and workload limits |
| [v0.1.1 release ledger](releases/v0.1.1.md) | Performance workflow, FP8 PLE validation and limits |
| [v0.1.2 release ledger](releases/v0.1.2.md) | Inkling checkpoint and shared Jinja qualification |
| [v0.1.3 release ledger](releases/v0.1.3.md) | Serving Parity plan (P0–P4) and current contract |
| [Serving contract](serving-contract.md) | Common options, requested/effective/qualified, inspect |
| [Chat templates](chat-templates.md) | Official input grammar, local assets and continuation |
| [API surface matrix](ds4-api-surface-matrix.md) | Wire contracts, routing and unsupported behavior |
| [Model families](ds4-dfm-model-families.md) | Family contracts and dated Spark evidence |
| [dots3 serving](ds4-dfm-model-families.md#dots3-serving) | Separate opt-in text banks and serial MTP; snapshot and qualification limits |
| [Qwen FP8 PLE](qwen38-ple-fp8.md) | Sidecar selection, KV compatibility and paired card sweeps |
| [Qwen performance, September 14](qwen38-perf-2026-09-14.md) | Draft/prefix A/B, rejected QSA PV / pair-reuse / HC, serving limits |
| [Memory guard](host-memory-guard.md) | Admission policy and operational limits |
| [ds4-perf](ds4-perf.md) | Inspect, scout, compare, optimize and evidence contracts |
| [Optimization playbook](prefill-decode-optimization-playbook.md) | Execution-path diagnosis and numerical proof |
| [Speed benchmarks](../speed-bench/README.md) | Manual sweep CSVs and plotting |
| [Lineage](LINEAGE.md) | Upstream provenance and repository split |

## Recorded evidence

- [Migration evidence](rust-migration/README.md): frozen C oracle, host parity,
  C-shared gaps and the completed repository split.
- Qwen: [long-context prefill, September 4](qwen38-long-context-prefill-2026-09-04.md),
  [September 6](qwen38-prefill-2026-09-06.md),
  [September 7](qwen38-prefill-2026-09-07.md),
  [image/text/agent benchmarks, September 7](qwen38-image-2026-09-07.md),
  and [September 14 draft/prefix plus rejected QSA PV](qwen38-perf-2026-09-14.md).
- K2: [September 5 optimization](k2-optimization-2026-09-05.md) and
  [continued campaign](k2-optimization-2026-09-05-cont.md).
- [dots3, September 6](dots3-optimization-2026-09-06.md) and
  [Solar, September 7](solar-open2-optimization-2026-09-07.md), including
  unsuccessful experiments and remaining limits.
- [Solar, September 12](solar-open2-optimization-2026-09-12.md): four-round
  closure, guarded baseline evidence and unchanged acceptance limits.
- [Solar, September 12, round 5](solar-open2-optimization-2026-09-12-r5.md):
  opt-in warp-specialized K-FP8/V-FP4 prefill attention (byte-exact, 2.6x
  at 64K depth) and the GB10 power trip behind both host freezes.
- [Solar, September 14](solar-open2-optimization-2026-09-14.md): disk-KV
  and HTTP partial fork, default-on FATTN_WS under a 300–2200 MHz cap,
  64K prefill +26.8%.
- Partial reuse: [Solar, August 21](solar-partial-reuse-2026-08-21.md) and
  [Motif, August 22](motif3-partial-reuse-2026-08-22.md).

Completed migration plans, superseded status files and old process handoffs
were removed from the working tree. Their original contents remain in the
[pre-cleanup documentation tree](https://github.com/Baekpica/ds4-dfm-rs/tree/ac750d61ef0306d30cc595081883f61ec847c3d8/docs).
Do not reuse recorded PIDs or assume those services are running.

## Design records

- [Ling-3.0-flash-VL integration](ling3-flash-vl.md): the bailingmoe3 hybrid
  KDA/MLA stack, grouped sigmoid routing, the Qwen3-VL vision tower and the
  Qwen-parity serving surface it was sized against. GB10 throughput:
  [8K campaign, long-context rounds and the 2K–64K sweep](ling3-flash-vl.md#cuda-campaign-gb10),
  raw data under [`benchmarks/ling3-flash-vl-2026-09-17/`](benchmarks/ling3-flash-vl-2026-09-17/).

- [Step 3.7 integration](step37-initial.md): MQ83 text/image serving, MTP,
  numerical and KV verification; measured qualification limits.
  GB10 throughput: [BASE artifacts](step37-optimization-2026-09-13.md),
  then [SWA HMMA, chunk 2048, MTP verify GQA](step37-optimization-2026-09-13-r2.md).
  [Banked MTP, partial fork and disk KV](step37-serving-2026-09-13.md);
  [300–2200 MHz matched optimization rounds](step37-optimization-2026-09-13-r3.md).

- [Inkling Small integration](inkling-small.md): MQ85GB/MTP serving,
  multimodal checks and remaining qualification.
- Inkling MQ85GB: [rounds 1–12](inkling-optimization-2026-09-10.md),
  [rounds 13–15](inkling-optimization-2026-09-11.md),
  [rounds 16–18](inkling-optimization-2026-09-11-r16.md),
  [rounds 19–21](inkling-optimization-2026-09-12.md),
  [rounds 22–24](inkling-optimization-2026-09-12-r22.md), and
  [rounds 25–27](inkling-optimization-2026-09-12-r25.md), GB10 prefill/decode
  measurements and numerical controls.
- [Qwen image input contract](qwen38-image-input-spec.md): original design;
  use the current README and dated image evidence for implemented scope.
- [W2A16 fused-dequant GEMM proposal](ds4-w2a16-fused-dequant-gemm-design.md):
  proposed work, not a released execution path.
- [DeepSeek upstream model-card synopsis](../MODEL_CARD.md): upstream model
  information, separate from DS4 artifact and runtime validation.
