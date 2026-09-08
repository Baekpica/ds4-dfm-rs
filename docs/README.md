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
| [v0.1.0 release ledger](releases/v0.1.0.md) | Release qualification, evidence and workload limits |
| [API surface matrix](ds4-api-surface-matrix.md) | Wire contracts, routing and unsupported behavior |
| [Model families](ds4-dfm-model-families.md) | Family contracts and dated Spark evidence |
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
  [September 7](qwen38-prefill-2026-09-07.md), and
  [image/text/agent benchmarks, September 7](qwen38-image-2026-09-07.md).
- K2: [September 5 optimization](k2-optimization-2026-09-05.md) and
  [continued campaign](k2-optimization-2026-09-05-cont.md).
- [dots3, September 6](dots3-optimization-2026-09-06.md) and
  [Solar, September 7](solar-open2-optimization-2026-09-07.md), including
  unsuccessful experiments and remaining limits.
- Partial reuse: [Solar, August 21](solar-partial-reuse-2026-08-21.md) and
  [Motif, August 22](motif3-partial-reuse-2026-08-22.md).

Completed migration plans, superseded status files and old process handoffs
were removed from the working tree. Their original contents remain in the
[pre-cleanup documentation tree](https://github.com/Baekpica/ds4-dfm-rs/tree/ac750d61ef0306d30cc595081883f61ec847c3d8/docs).
Do not reuse recorded PIDs or assume those services are running.

## Design records

- [Qwen image input contract](qwen38-image-input-spec.md): original design;
  use the current README and dated image evidence for implemented scope.
- [W2A16 fused-dequant GEMM proposal](ds4-w2a16-fused-dequant-gemm-design.md):
  proposed work, not a released execution path.
- [DeepSeek upstream model-card synopsis](../MODEL_CARD.md): upstream model
  information, separate from DS4 artifact and runtime validation.
