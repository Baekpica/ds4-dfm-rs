# Contributing

Read [AGENT.md](AGENT.md) and the [architecture](docs/rust-migration/ARCHITECTURE.md).
Rust owns the production host; CUDA/MMQ/VMM remains native. Keep each change
scoped and report the exact commands, commit/build, hardware, model artifact,
workload and failures. The [v0.1.0 ledger](docs/releases/v0.1.0.md) defines the
release bar; a version bump or host-only CI pass does not close live GPU gates.
Backend changes require correctness and speed evidence. Accept a speed
regression only when a necessary correctness repair justifies it explicitly.

## Rust host checks

Use the pinned `rust-toolchain.toml`. Build the C parity oracles before running
all workspace tests; otherwise some oracle-dependent tests can skip work.
The [Host parity workflow](.github/workflows/host-parity.yml) runs this sequence
on a hosted CPU runner:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked
make -j1 test-kv-parity
make -j1 test-web-parity
make -j1 test-dist-parity
make -j1 test-server-parity
make -j1 test-catalog-parity
make -j1 test-tokenizer-parity
make -j1 test-session-parity
make -j1 test-agent-parity
cargo test --workspace --locked -- --test-threads=1
cargo check --workspace --all-targets --locked
```

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

On macOS, `make` selects the inherited Metal build. `make cpu` builds the C
reference executables; it is not a production CPU performance target. Avoid
large CPU inference on macOS because of the documented VM failures.

Native checks include `make cuda-regression`, `make test-model-family-kernels`
and `make test-mmq-parity`, according to the affected path. `make test` and
`ds4_test` retain C/native regression coverage; `ds4_test --server` does not
replace the Rust server parity and live API gates. DeepSeek-specific tests
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

The [profiling guide](docs/prefill-decode-optimization-playbook.md#local-scout-with-ds4-perf)
shows the canonical `ds4-perf doctor` / `scout` workflow and
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
