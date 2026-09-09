# Architecture

Rust owns host orchestration. Native CUDA/MMQ stays the GPU backend.
[v0.1.0](../releases/v0.1.0.md) defines the first independent Rust-host release
with a stable host/runtime boundary and a native performance-observability
workflow. Its production qualification is tracked separately from the
[pre-split parity evidence](PARITY_MATRIX.md).

## Ownership and execution

```text
CLI / API clients / coding agent
                |
Rust host: model/session ownership, scheduling, serving,
           KV/state policy, distributed orchestration
                |
        ds4-core (safe host API)
                |
        ds4-sys (native inference FFI)
                |
        native/bridge/ds4_bridge.h
                |
Native: CUDA / MMQ / VMM / graphs / device state
        Metal and CPU reference backends retained
```

Rust's `Model`, `Session`, snapshots and batch wrappers own opaque native
handles. Sessions borrow their model; thread-affine handles stay on the
execution owner. Their `Drop` paths release the matching native resources.
HTTP readers and output writers do not take ownership of GPU state.

Rust owns catalog identification/validation, tensor bind plans and tokenizer
behavior on the production host path. Native code owns weight upload,
allocation, numerical execution and GPU payloads. GGUF access stays mmap-backed;
do not read the entire model into a Rust byte vector. A resident VMM weight
owner can share allocations with fresh workers through the existing manifest
and broker contract. See the [memory guard](../host-memory-guard.md).

## Crates and binaries

For Jinja-backed artifacts, input grammar runs the model's official Jinja
behind the local `ds4-core::chat_template` adapter. API normalization,
tokenizer/media processing and output protocols remain separate. REPL reuses
the server's host-only output parser. See
[template assets and continuation](../chat-templates.md), including
the DeepSeek V4 encoder exception. New families with official Jinja do not need
another imperative input renderer.

| Crate | Responsibility |
|---|---|
| `ds4-cli` | CLI, benchmark, coding agent |
| `ds4-server` | HTTP parsing/rendering, admission, scheduling, streaming, tools |
| `ds4-core` | Safe model/session API, catalog, tokenizer, host state |
| `ds4-kv` | KVC metadata, persistence policy, indexes and codecs |
| `ds4-dist` | Distributed protocol and blocking orchestration |
| `ds4-web` | Blocking agent web helpers |
| `ds4-sys` | ds4 native inference bindings and reviewed OS adapters |
| `ds4-perf` | Inspect, scout, compare and bounded performance experiments |
| `ds4-perf-gpu` | Optional CUDA calibration and CUPTI collection via cudarc |

`make` builds Rust production names `ds4`, `ds4-server`, `ds4-bench`, and
`ds4-agent` when a backend build is selected. Cargo retains the `*-rs` binary
names; Make provides those names as compatibility aliases. `ds4-c`,
`ds4-server-c`, `ds4-bench-c`, and `ds4-agent-c` remain C behavior oracles.
`ds4-eval` and native family/kernel tests also remain native.

`ds4_cuda.cu`, `cuda/mmq/`, and other hardware kernels are not Rust migration
targets. Native helper and oracle retention is intentional; it is not a claim
that inference has no C/CUDA dependency. Retire a C host oracle only after
its Rust replacement has unit/live parity, performance and soak evidence;
keep the baseline reachable in Git.

## Host concurrency and state

The host uses blocking sockets, `std::thread`, channels, mutexes and condition
variables. `ds4-server` owns serial, continuous and static scheduling over the
configured/native-fitted bank capacity. Native calls execute the GPU work;
Rust controls admission, placement, cancellation and continuation policy.
Family-specific lane restrictions remain explicit in the
[API surface matrix](../ds4-api-surface-matrix.md).

The four generation surfaces are Chat Completions, Completions, Messages and
Responses. Their shared scheduling does not erase prompt/tool/stop contracts.
Distributed frames use explicit integer codecs, not Rust/C memory layouts.
KVC metadata and cache policy are Rust-owned; opaque numerical snapshot and
payload state remains native.

## Performance observability

```text
ds4-perf -> machine/calibration -> scout/fit/NCU -> compare/decision
    |
    +-- optional ds4-perf-gpu: CUDA calibration and CUPTI collection

ds4-cli -- optional perf-nvtx --> NVIDIA nvtx
   |
ds4-core -> ds4-sys -> native CUDA / MMQ / VMM
```

NVTX is a Rust-host side dependency. `ds4-cli` uses NVIDIA's `LocalRange`
directly around measured `ds4.prefill` and `ds4.decode` operations. Profiling
handles do not cross `ds4-core`, `ds4-sys`, or the native bridge. Ordinary
inference builds do not enable the SDK. `ds4-perf` is not an inference dependency.

CUDA profiling calls and CUPTI callbacks use maintained cudarc bindings in
`ds4-perf-gpu`. Its unsafe adapter is separate from native inference; the
orchestration executable and inference packages do not expose device handles.

The [profiling guide](../ds4-perf.md) defines the build, capability checks,
versioned evidence, comparison policy and bounded automatic experiments.
Nsight Systems is structural evidence; it does not establish memory-bound
versus compute-bound behavior, correctness, or end-to-end improvement.

## Boundary changes

Follow [FFI_CONTRACT.md](FFI_CONTRACT.md). Keep model/session/device structures
opaque; no CUDA streams, device pointers, graph handles or VMM handles in
application code. Host-owned metadata descriptors may cross the documented
ABI; native internal layouts may not.

A native ABI change updates the header, implementation, Rust declarations,
safe wrappers and relevant parity tests together. Do not add another CUDA
context, allocation stack or kernel runtime behind the Rust host. Preserve
explicit numerical and wire contracts; change them only with scoped evidence.

The [migration index](README.md) retains the frozen C baseline, parity matrix
and genesis decision. Those records describe their original commits and
workloads, not the current release candidate's qualification.
