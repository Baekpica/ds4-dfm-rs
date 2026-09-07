# Repository instructions

`ds4-dfm-rs` is the independent Rust-host continuation of DwarfStar DFM.
The default `ds4`, `ds4-server`, `ds4-bench`, and `ds4-agent` are Rust hosts.
CUDA/MMQ/VMM and hardware-specific kernels remain native. C executables
with `-c` suffixes are retained behavior oracles, not the production hosts.

Start with [README.md](README.md), the [documentation index](docs/README.md),
and the [architecture](docs/rust-migration/ARCHITECTURE.md).
The [v0.1.0 gate ledger](docs/releases/v0.1.0.md) separates release preparation
from verified production qualification. Dated reports establish only their
recorded commit, artifact and workload; old PIDs and handoffs are not live state.

## Inference Performance Optimization Guidelines

When optimizing inference performance, prioritize end-to-end execution-path efficiency over isolated kernel micro-optimizations.

Start from measured prefill and decode wall time. Decompose the execution into kernels, dispatch paths, memory transformations, routing/indexing operations, synchronization, and fallback implementations.

The primary optimization objective is **fast-path coverage**: maximize the fraction of the inference graph executed by the runtime's best hardware-appropriate primitives.

Before optimizing an already-fast kernel, search for:

- operations unexpectedly falling back to slower kernels;
- irregular tensor shapes that can be decomposed into native fast-path tile or iteration sizes;
- repeated quantization, format conversion, normalization, permutation, or activation materialization;
- redundant intermediate writes and subsequent accumulation passes;
- routing, sorting, indexing, or expert-map construction whose complexity grows with tokens or experts;
- separate kernels that can legally share an accumulator, worklist, quantized activation, routing map, or intermediate representation;
- dense operations that can reuse an optimized routed primitive through an identity or single-expert mapping;
- prefill operations incorrectly using decode-oriented kernels, or vice versa.

Prefer eliminating a slow execution path over making an already-fast path incrementally faster.

Treat prefill and decode as distinct workloads. Use explicit width- or shape-dependent dispatch where their optimal execution strategies differ.

Model-specific specialization is acceptable and encouraged when it produces measurable gains, but isolate specialization behind explicit capability predicates based on tensor shape, quantization format, topology, and workload width. Generalize the dispatch policy rather than forcing specialized kernels into a generic implementation.

For every optimization:

1. Record the specific bottleneck and its measured contribution to end-to-end latency.
2. State why the existing path is inefficient.
3. Preserve or explicitly define the numerical contract.
4. Add focused parity or regression tests.
5. Keep a fallback or kill switch for new execution paths where practical.
6. Benchmark with fresh processes and comparable workloads.
7. Report both kernel-level improvement and end-to-end prefill/decode impact.
8. Verify that improving prefill does not regress decode, and vice versa.

When reviewing the profiler, optimize in roughly this order:

unexpected fallback paths → repeated transformations → redundant memory traffic → routing/indexing overhead → launch fragmentation → fast-path utilization → individual optimized-kernel tuning.

The guiding principle is:

**Optimize execution-path topology before optimizing individual kernels.**

# Agent Notes

Support is limited to explicit model-family/artifact contracts listed in
[README.md](README.md#supported-model-families). Keep the Rust host and native
backend small and direct; this is not a generic GGUF runner.

## Goals

- Keep production inference on the validated GPU execution paths. DGX Spark
  CUDA is the release reference; inherited Metal support needs separate checks.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Keep new host/control-plane code in Rust; do not add a C++ host layer.
  Existing CUDA/MMQ implementation stays native.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not load independent huge model copies concurrently. Use the intended
  weight owner and bounded workers; inspect ownership and memory first.
- Preserve unrelated servers and resident owners. Stop only processes owned by
  the task. See [host-memory-guard.md](docs/host-memory-guard.md) for the guard's
  admission rules and documented limits.
- Prefer short GPU smoke tests for build verification
  (Metal on macOS, CUDA on Linux).

## Layout

- `crates/ds4-cli`: production CLI, benchmark and coding agent.
- `crates/ds4-server`: HTTP, admission, scheduling, streaming and tools.
- `crates/ds4-core`: model/session ownership, host catalog and tokenizer.
- `crates/ds4-kv`, `ds4-dist`, `ds4-web`: KV policy, distributed runtime and web helpers.
- `crates/ds4-sys` + `native/bridge`: narrow native inference boundary.
- `crates/ds4-perf`: standalone benchmark/profiler orchestration and diagnosis.
- `ds4.c`: native engine, GPU state/execution and retained compatibility helpers.
- `ds4_cli.c`, `ds4_server.c`, `ds4_bench.c`, `ds4_agent.c`: C host oracles.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu`: CUDA backend. Single TU; mirrors `ds4_metal.m`'s role on
  NVIDIA. CUDA env vars and dispatcher behavior are documented in
  `misc/cuda-env-vars.md`; CUDA MTP specifics in `misc/cuda-mtp/README.md`.
- `cuda/mmq/`: vendored llama.cpp ggml-cuda matmul kernels + ds4-side adapter.
  See `cuda/mmq/VENDOR.md` for the upstream pin and re-sync procedure.
- `tools/ds4_weight_server.cu`: optional CUDA weight-server sidecar for
  multi-process testing. See `misc/proof-harness/README.md`.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material. A few
  reference docs are force-added (`cuda-env-vars.md`, `cuda-mtp/`,
  `proof-harness/`, `ANTHROPIC_LIVE_CONTINUATION.md`, `RESPONSE_API.md`).

## CUDA captured-decode rules

- Captured decode kernels that consume `pos0`, `n_comp`, `n_index_comp`,
  `raw_start` / `raw_row`, `n_raw`, selected-row counts, or scratch pointers
  MUST read live substrate state (`g_decode_dev` / `g_layer_dev[il]`) or be
  keyed by regime into the graph cache. By-value kernel arguments are baked at
  graph queue time and replay stale. Reference: 7c4b84d, a1cff19, 8fb3c54.
- Long-context captured-vs-eager parity (essay prompt, n=1024, FP32, every
  enabled overlay) is a release gate, not a smoke test. See `make proof-cuda-long`.
- Optimization commits land with a correctness proof AND a speed proof. The
  proof harness records both: `tests/ds4_proof.py --scenario ...`. Skipping the
  correctness proof on the grounds that "we already had it before" is how the
  pos0 regression slipped past three previous commits.

## MTP / compressed-KV decode rules

- Lossless = verified tokens AND committed compressed-KV == the accepted-prefix transition. Not bit-identity (cross-width MoE order differs), not tokens alone (carried state sets future logits).
- Emit compressor state in the verify forward, roll back rejects; never re-emit in a second pass (a commit-reforward emit was ~100% wrong).
- Gate on ‖Δcompressed-cache‖ and long context, not short-prompt tokens — short context hides cache corruption behind raw/windowed KV.
- FP vs structural is element-wise magnitude (rel ≈1e-6 vs ≈1); a late token flip is not FP.
- Size GPU-readback buffers to the tensor and fail loud — undersized returns a constant → false "identical".
- At D>=1 the batched verify (width 1+D) is NOT bit-identical to a width-1 decode (cross-width MoE-order FP, same as N=1 vs N=8). At **N=1** the committed cache IS invariant to rejected-draft VALUES (width-matched, no cross-row terms) — validate the rollback there expecting bit-identical cache; comparing D>=1 vs width-1 mode-0 conflates inherent FP with real bugs.
- At **N>1** the committed cache is NOT value-invariant to rejected drafts, and that is NOT a bug: the in-forward verify is itself draft-VALUE-dependent via cross-row MoE-reduction FP — holding cur's input identical, varying a causally-masked rejected draft shifts the verify logits by FP while the argmax (committed token) is unchanged, and that FP rides an expert flip into the committed KV (deep-layer ‖Δ‖ ≈ O(10) while the final logit moves ≈0.03 and the token does not flip). So N>1 losslessness is TOKEN-LEVEL: gate the committed-row FRONTIER (counts) + token stream invariance to rejected-draft values; treat committed-cache VALUES as inherent-FP-noisy (informational), and add a byte-exact restore self-check (read back each restored lane vs its checkpoint source) to guard the capture/restore plumbing independently. Don't chase the N>1 cache-value diff as a rollback bug — bisect: the divergence onset is a cur-row (M=1) logit FP in the forward, not the restore (which is byte-exact).

## Testing

Use the [contribution checks](CONTRIBUTING.md): Rust host parity and serialized
workspace tests are model-free; CUDA/family/proof gates need their exact
fixtures. On Linux, `make` prints help: select `make cuda-spark`,
`make cuda-generic`, or an explicit CUDA architecture for native validation.
Use live server tests only when intentionally testing the API surface.

For ds4-perf, run `cargo test -p ds4-perf`. Build optional instrumentation with
`make ds4-bench-perf` after the CUDA build. Use NVIDIA's official Rust `nvtx`
SDK directly in `ds4-cli` under `perf-nvtx`; keep its default features disabled
and enable only `std` unless a measured need justifies more. Use `LocalRange`
for the measured `ds4.prefill` / `ds4.decode` operations. Do not add custom NVTX
FFI, route it through `ds4-core`/`ds4-sys`/the bridge, or annotate each token or
kernel. Ordinary inference builds remain independent of this optional SDK.

Multi-process testing (proof harness, multi-profile sweeps, MTP correctness
work that loads base + MTP gguf into the same device) goes through
`ds4_weight_server`. See `misc/proof-harness/README.md`. Single-process
runs hit the same prefill ceiling without a sidecar via the in-process
VMM arena, which is on by default.

## Common Rules
- When writing something intended for human consumption, (comment, commit message, reply to prompt) use as few words as possible. Pick every word meticulously to reduce the volume to a strict minimum. Be down to the point. Less is more.

- Avoid superlatives and praise. Stop telling me I am absolutely right. Give me the cold hard truth.

- Avoid magic numbers and strings by extracting recurring or meaningful values into descriptive constants (const) or enums. Keep self-explanatory, one-off values inline to avoid clutter. If a value comes from a spec (e.g. HTTP 200 OK), use a constant regardless.

- Reduce code indentation. Avoid Arrow Anti-Pattern. Leverage early return and continue.

- Keep function names short. Less than 30 characters.

- Use enums instead of booleans for function parameters.

- Let the reader of the code breathe. Add empty lines between logical blocks of code.

- Add a small, to the point, comment to explain *what* the block does and *why*. Use examples when possible. Propose ASCII drawings to explain complete systems.

- Treat member visibility changes as a breaking design shift. Keep all fields and functions private unless external access is strictly required by the design. Prompt the user for explicit approval before changing any access modifier from private to internal or public.

- Program to levels of abstraction. Lower-level mechanics (e.g., raw hardware I/O, sector parsing, direct socket streams) must be encapsulated in a dedicated driver/abstraction layer. Expose clean, high-level APIs to the rest of the application so calling code works with domain concepts, not raw implementation details.

- Don't touch blocks of code unrelated to the feature you implement. e.g. Don't add comments to a block of code if you did not create it or modify it. As much as possible try to minimize the number of changed lines when implementing a feature.

- Preserve the inference boundary: host → ds4-core → ds4-sys → native backend.
  Host-only libraries such as the official NVTX SDK are side dependencies;
  they do not belong in the inference ABI. Keep raw device handles out of
  application code and follow [FFI_CONTRACT.md](docs/rust-migration/FFI_CONTRACT.md).

- Always use {}, even on a one-line "if" statement.

When you write a commit message, follow these 7 rules:
Rule 1: Separate the subject line from the body with a single blank line.
Rule 2: Limit the subject line to 50 characters (72 is the absolute hard limit).
Rule 3: Capitalize the first letter of the subject line.
Rule 4: Do not end the subject line with a period.
Rule 5: Use the imperative mood in the subject line (e.g., "Fix bug," "Add feature," 
        not "Fixed" or "Adds"). Test formula: It must complete the sentence: "If applied,
        this commit will [your subject line here]".
Rule 6: Wrap the body text manually at 72 characters to prevent Git formatting issues.
Rule 7: Use the body to explain what and why vs. how. Assume the code explains the how;
        the message must explain the context and reasoning. 

- If the prompt indicates that a bug is being fixed, don't write the fix right away. First write the test. Observe it failing. Then write the fix. And observe the test passing. 
