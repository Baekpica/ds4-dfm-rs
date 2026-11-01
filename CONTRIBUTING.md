# Contributing

Read [AGENTS.md](AGENTS.md) and the [architecture](docs/rust-migration/ARCHITECTURE.md).
Rust owns the production host; CUDA/MMQ/VMM remains native. Keep each change
scoped and report the exact commands, commit/build, hardware, model artifact,
workload and failures. The [v0.1.0 ledger](docs/releases/v0.1.0.md) defines the
release bar; a version bump or host-only CI pass does not close live GPU gates.
Backend changes require correctness and speed evidence. Accept a speed
regression only when a necessary correctness repair justifies it explicitly.

## One implementation

Host behavior (HTTP, parsing, templates, sampling, scheduling, KV policy,
tools, CLI) is written once, in Rust. Native C/CUDA keeps only work that
computes on tensors or devices. The C host executables (`ds4-c`,
`ds4-server-c`, `ds4-bench-c`) and C host logic in `ds4.c`
that Rust now owns, such as the C sampler, are deprecated: do not add
features or fixes to them. Move remaining host logic from C to Rust when a
change touches it.

The `tests/parity` oracles are frozen copies of the old C behavior. When
Rust changes a behavior on purpose, delete or narrow the parity case in the
same change; never mirror the change into C.

## Rust host checks

Use the pinned `rust-toolchain.toml`.
The [Host parity workflow](.github/workflows/host-parity.yml) runs four
Makefile groups as parallel jobs on a hosted CPU runner; `make ci` runs all
four locally:

| Target | Runs |
|---|---|
| `ci-lint` | rustfmt, Clippy, all-target and native-host type checks |
| `ci-rust` | serialized workspace tests (after building the parity oracles) |
| `ci-parity` | every C oracle / Rust host parity suite, once |
| `ci-model-free` | shared native gates that need no model |

CI holds only checks that cover every family. Add a check to a group in the
Makefile, never as a workflow step. A check of one family's own code joins
that family's `gate-<family>` target (`gate-ds41`, `gate-iquest`,
`gate-qwen`, `gate-glm53`, `gate-mimo2`); run it when the family changes.

Use a narrower affected-crate test during development. `cargo test -p ds4-perf`
is model-free and tests CSV parsing, diagnosis, capability detection and process
orchestration without requiring Nsight. Keep original goldens; do not refresh
them to conceal a mismatch.

## Native builds and live gates

On Linux, plain `make` prints help. Select the actual CUDA backend:

```sh
make cuda-spark                     # DGX Spark / GB10
# or: make cuda-generic
# or: make cuda CUDA_ARCH=sm_N
```

CUDA builds produce the Rust hosts, `ds4-eval` and the weight server. The
deprecated C host oracles build only on request
(`make ds4-c ds4-server-c ds4-bench-c`); proof and diagnostic
targets that use them declare their own dependencies.

On macOS, `make` selects the inherited Metal build. `make cpu` builds the
deprecated C reference executables; it is not a production CPU performance
target. Avoid
large CPU inference on macOS because of the documented VM failures.

`./ds4-server --version` reports the Rust package version;
`make print-version` reports the Git-derived native build stamp. Rust and native
objects must come from the same commit.

Native checks include `make cuda-regression`, `make test-model-family-kernels`
and `make test-mmq-parity`, according to the affected path. `make test` and
`ds4_test` retain native regression coverage; `ds4_test --server` exercises
the deprecated C server and is no gate for the Rust server. DeepSeek-specific tests
require their documented DeepSeek fixtures, not an arbitrary supported GGUF.
See the [baseline/proof protocols](docs/rust-migration/BASELINE.md),
[parity matrix](docs/rust-migration/PARITY_MATRIX.md), and
[current family scope](README.md#supported-model-families).

Start tool verification with a short, single-frontier GPU smoke. Run production,
capture/eager, long-context, KV, concurrency and soak gates when required by the
change or release claim. Preserve explicit artifact, context, token-count,
cache and MTP settings. A short smoke does not substitute for those gates.

## Performance and profiling

Use `ds4-bench` for throughput. Its sweep rows measure the newly computed
prefill suffix at each frontier; a cold single-frontier workload is a different
protocol. Reuse the original fixture and protocol for every comparison.

The [profiling guide](docs/ds4-perf.md)
shows `inspect` / `scout` / `compare` / `optimize` and
`make ds4-bench-perf` build. NVTX uses the optional official Rust SDK in the
benchmark host; native inference ABI changes are not needed for annotations.

Compare fresh unprofiled processes with the same model, prompt, backend,
context/output length, memory policy, owner, cache state and clock conditions.
Report prefill and decode separately. Nsight explains structure; profiled TPS
is not an unprofiled baseline. Pair any optimization with full-vocabulary logits,
greedy tokens and the relevant state/correctness gate. Retain raw evidence in
ignored `scratch/`; publish only scoped summaries and intended fixtures.

Inspect existing GPU processes before loading a model. Keep intended weight
owners resident across bounded workers; do not co-reside an independent model
copy or unrestricted NCU replay with a large owner. Follow
[host-memory-guard.md](docs/host-memory-guard.md), including its known limits.

## Quantization and bug reports

Quantization work also needs the
[official-continuation scorer](gguf-tools/quality-testing/README.md) against the
same manifest for old and new artifacts. Those DeepSeek vectors are scoped
reference evidence, not a gate for every family.

For a generation or API failure, retain the request shape, runtime SHA/build,
model identity, server stdout/stderr and relevant profiler artifacts. Use small
reproducible inputs and exclude credentials from shared logs.

## Documentation changes

Use the [documentation index](docs/README.md) to choose the owning guide.
Keep operational instructions separate from dated evidence; retain the exact
artifact, workload and qualification boundaries when summarizing a result.

For Markdown-only edits, check local links and heading anchors, review tables,
images and code fences, and run `git diff --check`. Inference builds and live
model gates are unnecessary for prose-only changes. The
[serving capability table](docs/serving-capabilities.md) is generated; preserve
its generator/check contract rather than editing its cells by hand.
