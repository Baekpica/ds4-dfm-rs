# Rust host: current contracts and migration evidence

The C-to-Rust host migration and repository split are complete. The independent
`ds4-dfm-rs` product keeps CUDA/MMQ/VMM native. The genesis tag
`ds4-dfm-rs-genesis` resolves to `fe7733fb4f7e18204b6ea0a00fe3b136d2029b17`.

Current development follows the [architecture](ARCHITECTURE.md),
[FFI contract](FFI_CONTRACT.md), [repository instructions](../../AGENT.md)
and [v0.1.0 release ledger](../releases/v0.1.0.md). The directory name is
retained for existing links; those two boundary documents remain current.

## Frozen migration evidence

The original campaign reproduced the C `v0.6.5-dfm` lineage (`d02e2a4`) and
Qwen's post-tag feature cut (`4d40d97`). Its results describe those exact
artifacts, commands and commits. They do not mark the current candidate's
production gates green.

| Document | Recorded scope |
|---|---|
| [BASELINE.md](BASELINE.md) | Immutable C oracle, environment and proof protocols |
| [PARITY_MATRIX.md](PARITY_MATRIX.md) | Numerical, token, KV, wire and performance parity evidence |
| [QWEN_V065_RESTAMP_2026-08-31.md](QWEN_V065_RESTAMP_2026-08-31.md) | Q5+Sidecar text/image, family tests and ABBA evidence |
| [SPLIT_READINESS.md](SPLIT_READINESS.md) | August 31 genesis decision and evidence identities |
| [ENGINE_GAPS.md](ENGINE_GAPS.md) | Recorded C-shared failures behind qualified PASS* cells |

Superseded status reports and seeding plans have been removed. Their
[status snapshot](https://github.com/Baekpica/ds4-dfm-rs/blob/ac750d61ef0306d30cc595081883f61ec847c3d8/docs/rust-migration/STATUS.md)
and [split plan](https://github.com/Baekpica/ds4-dfm-rs/blob/ac750d61ef0306d30cc595081883f61ec847c3d8/docs/rust-migration/DFM_RS_SPLIT_PLAN.md)
remain available at a fixed historical commit. Seeding instructions and
temporary campaign constraints are not current development steps.

See the [documentation index](../README.md) for current API/family guides and
post-split performance reports, and [LINEAGE.md](../LINEAGE.md) for provenance.
