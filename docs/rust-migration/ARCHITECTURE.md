# Architecture

Rust owns host orchestration. Native CUDA/MMQ stays the GPU backend.
The C tree in the `v0.6.5-dfm` lineage remains the behavior oracle until a
subsystem has passed the parity matrix and is promoted. Qwen behavior is
frozen at the post-tag C cut `4d40d97`.

## Independent Rust-host baseline

[v0.1.0](../releases/v0.1.0.md) defines the first independent Rust-host release
with a stable host/runtime boundary and a native performance-observability
workflow. The RC series established migration, parity, and ownership; the
independent host also owns observability and performance orchestration.

`ds4-perf` runs benchmarks and Nsight in separate processes. Optional NVTX
annotations belong to `ds4-cli`, which uses NVIDIA's maintained Rust SDK
directly. They do not extend `ds4-core`, `ds4-sys`, or the inference bridge.
`ds4-sys` remains the unsafe boundary for ds4's native backend.

```text
Workflow: ds4-perf scout -> diagnosis -> code change -> A/B + proof

ds4-cli -- optional --> NVIDIA nvtx
   |
ds4-core
   |
ds4-sys -> native CUDA / MMQ / VMM
```

## Strangler, not rewrite

```text
C implementation          Rust host
████████████████████      ░░░░░░░░░░░░░░░░░░░░

        ↓ stepwise replacement, every step boots

C                         Rust
██████████████░░░░░░      ░░░░░░░░░░██████████

        ↓

C                         Rust
██░░░░░░░░░░░░░░░░░░      ░░██████████████████
```

Forbidden:

```text
ds4.c  →  ds4.rs          # file translation
“read C, write a new engine”
```

Required: an executable that still loads a model and produces tokens
after every phase.

## Target process shape

```text
┌─────────────────────────────────┐
│            Rust host            │
│ Model / Session / KV            │
│ Scheduler / Server              │
│ API / Distributed               │
│ Memory Policy                   │
└────────────────┬────────────────┘
                 │
           narrow native ABI
           (ds4_bridge.h)
                 │
                 ▼
┌─────────────────────────────────┐
│       Native GPU backend        │
│ ds4_cuda.cu                     │
│ cuda/mmq/*.cu                   │
│ cuBLAS                          │
│ CUDA Driver API / VMM           │
└─────────────────────────────────┘
```

`ds4_cuda.cu` and `cuda/mmq/` are **not** migration targets. Do not
change kernel behavior because the host language changed.

Late-form CUDA boundary (Phase 8/9):

```text
backend_create / load_weights / session_create
prefill / decode / kv_save / kv_load / backend_destroy
```

These stay opaque. Rust application code must not see `CUstream`,
`CUgraphExec`, `CUmemGenericAllocationHandle`, device pointers, MMQ
descriptors, or kernel scratch.

## C tree responsibilities (oracle)

From `AGENT.md` and the current layout:

| Unit | Role |
|---|---|
| `ds4.c` | GGUF load, metadata, tokenizer, CPU reference, session, KV payload, graph orchestration, backend dispatch |
| `ds4_cli.c` | CLI / REPL |
| `ds4_server.c` | four wire surfaces, workers, streaming, disk-KV policy; continuation registry and bounded single-owner FIFO ported to the Rust shadow |
| `ds4_bench.c` / `ds4_eval.c` / `ds4_agent.c` | tools |
| `ds4_kvstore.c` | KVC file format, eviction, prefix, trailers |
| `ds4_web.c` | blocking sockets, poll, subprocess |
| `ds4_distributed.c` | C oracle: pipelined prefetch, snapshot gather, `ds4_dist_session_*`. Rust `ds4-dist` owns blocking HELLO/WORK/RESULT + route plan. |
| `ds4_cuda.cu` + `cuda/mmq/` | production GPU |
| `ds4_metal.m` + `metal/` | macOS backend (keep compiling; not the DFM gate) |
| `ds4.h` | existing narrow engine/session API used by C CLI/server |

`ds4.h` is already narrower than the internals of `ds4.c`. The Rust
boundary is **narrower still**: `ds4_bridge.h`, not a bindgen of
`ds4.h`. Hundreds of internal structs must not appear in Rust.

## Planned crate map

```text
Cargo.toml
crates/
├── ds4-sys/      unsafe FFI only — the single unsafe boundary
├── ds4-core/     safe Model / Session + host-owned shape catalog / mmap GGUF identify / tensor inventory / bind plan / host load apply / host validate / host vocab apply / host bind lookup / host layout / MTP+DSpark sibling catalogs / tokenizer / session ledger / DSV4 prefix / live memgov census snapshot
├── ds4-cli/      ds4 / ds4-bench / later ds4-agent
├── ds4-kv/       KVC format + store policy (Phase 4)
├── ds4-web/      agent web utility (Phase 5; blocking I/O)
├── ds4-server/   route_decide + HTTP door + parsers + projectors + admit + bounded owner FIFO + /metrics memgov porcelain + live census overlay + family render + tool-schema/invoke + generated-tool parse (Phase 7)
├── ds4-dist/     distributed codecs + runtime (Phase 6)
└── ds4-perf/     standalone benchmark / Nsight orchestration and diagnosis
native/
└── bridge/
    ├── ds4_bridge.h
    └── ds4_bridge.c    # wraps existing C; does not reimplement inference
```

Empty scaffolds are allowed in Phase 1. Do not fill a crate by
inventing a new engine.

Final pre-split tree (see work instruction §27) also keeps
`cuda/mmq/`, `metal/`, `tests/parity`, `tests/proof`, and a `legacy/`
holding C oracles that have not met the delete bar.

## Ownership model the port must make compile-time

```text
Model
 └── Session
      ├── KV
      ├── backend state
      ├── scratch
      └── continuation
```

Questions the types must answer:

- Who owns the session?
- Who owns KV?
- When is GPU allocation released?
- Which object outlives an HTTP stream?

`Drop` on a Rust handle calls the native destructor. Do not leak
engine/session lifetime into HTTP worker code.

## Loading contract

Keep mmap-backed loading:

```text
mmap → metadata/index → lazy tensor access
```

Do not slurp a GGUF into `Vec<u8>`. Do not eagerly copy the full
model. The VMM weight-owner / worker split stays the Spark serving
lifecycle; Rust must speak the same IPC manifest contract.

## Concurrency during migration

Use `std::thread`, channels, `Mutex` / `Condvar`, and blocking
sockets. Do not introduce Tokio, an async scheduler, or a new HTTP
stack until `ds4-dfm-rs` exists. Language migration and concurrency
redesign must be separable.

The current Rust server shadow lets client threads read/parse and drain
bounded output, while the caller thread remains the sole owner of the
non-`Send` native engine and executes FIFO jobs to completion. This
preserves C ownership; it does not yet prove live multi-client rolling
batch width.

Distributed frames use explicit `encode_*` / `decode_*` integer
codecs. Do not `#[repr(C)]` a Rust struct onto the wire.

## Serving lanes (frozen)

From `docs/ds4-api-surface-matrix.md`:

| Lane | C entry |
|---|---|
| serial | `generate_job` |
| continuous | `generate_continuous_jobs` |
| static | `generate_batch_jobs` |

Surfaces: `POST /v1/chat/completions`, `/v1/completions`,
`/v1/messages`, `/v1/responses`. Routing is `route_decide` over the
needs word. Rust reimplements that decision; it does not improve it.

## What stays C without blocking cut-over

- CPU reference forward (`ds4.c` CPU path) — oracle, not production
- Metal backend — keep the macOS compile/smoke; DFM gate is CUDA
- CUDA/MMQ/VMM — permanent native backend

The goal is **production host runtime without a C dependency**, not
deletion of every `.c` / `.cu` file.

## Promotion

Until Phase 9:

| Binary | Role |
|---|---|
| `ds4` / `ds4-server` / `ds4-bench` / `ds4-agent` | C oracle |
| `ds4-rs` / `ds4-server-rs` / `ds4-bench-rs` | Rust candidate, same C core then growing Rust host |

After parity:

| Binary | Role |
|---|---|
| `ds4` / `ds4-server` / `ds4-bench` / `ds4-agent` | Rust |
| `ds4-*-c` | C oracle, kept until repo split |

Delete C sources only when Rust exists **and** unit parity, live
parity, performance, and a minimum soak are green. Keep the C
implementation reachable from this tag for comparison.
