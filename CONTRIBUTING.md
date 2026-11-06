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

## Model families and large contributions

A new family or a change across several layers (format, kernels, serving,
docs) is reviewed against this list. Each rule comes from a defect a
review had to fix; meet it before opening the PR, not after.

### What a PR must not contain

- **One-off experiments.** Probes, staging benches, sweep drivers, variant
  scripts and scratch measurements stay outside the tree. A program enters
  only as a gate: a Makefile target that `gate-<family>`, CI or a documented
  fixture runs. Record what an experiment measured (numbers, command,
  commit, host) in the family's dated optimization log instead.
- **Host-specific scripts and paths.** No `run-<family>.sh`, launcher,
  advisor or per-machine tool: every family serves through the common
  `ds4-server`, `ds4` and `ds4-bench` commands, and its page documents the
  flags and switches. No absolute paths, user names, model directories,
  personal ports or deployment names in code, comments, docs or records
  (`$HOME/...` placeholders in a record's meta are the limit).
- **References outside the tree.** Comments and docs cite files in this
  repository, pinned upstream revisions (`transformers cc1451df`, a
  checkpoint's `config.json` revision) or public artifacts. Not handoff
  notes, review documents, chat logs, private repositories or another
  tool's paths (`crates/vq-gguf/...`, `docs/... in VQ_tools`).
- **Work-log vocabulary.** Milestone names (`P6`, `Phase 8`, `UNIT 7`,
  `PHASE B`) mean nothing to the next reader. A comment states the fact and
  the measurement that supports it.
- **Debug output to files.** Diagnostics print to stderr behind a family
  switch. The server writes only operator-configured paths (KV directory,
  `--expect-plan`, `--trace`); never `/tmp` or a fixed name.

### Family integration checklist

**Input.** Render the official template (`chat_template.jinja`, the
artifact's `tokenizer.chat_template` or `tokenizer_config.json`) through the
shared Jinja engine ([chat templates](docs/chat-templates.md)). No
imperative renderer, no hand-built prompt and no prompt traces. Check in
reference renders (`tests/fixtures/<family>/`) and a test that reproduces
them.

**Tokenizer.** Match the reference tokenizer, not a plausible BPE:
tokenizers' `merge_all` ranks a pair a merge creates at once, and a round
by round merge splits long whitespace runs differently. Pin the pre- and
post-processing the reference declares, the byte fallback and the decoder
(a SentencePiece vocabulary prints UTF-8, not GPT-2 bytes). Add model-free
tests for the edge cases (whitespace runs, created pairs, ties, byte
tokens, non-ASCII pieces) and, where the reference is available, a
differential run against it. Dispatch by family inside `tok`; do not add
public fields to shared structs for one family.

**Output protocol.** One parser module (`crates/ds4-server/src/tools/<family>.rs`)
for reasoning, content and tool calls. Parsers terminate on any input
(every loop advances, nesting is bounded) and report a malformed call as
`ok: false` with its text. The reasoning channel stays out of `content` in
all three stream protocols (chat completions, messages, responses) and in
the token scan (`SemAccum`); test each. A request that asks for reasoning
in any API form gets it. The shared malformed-call retry writes DSML text:
a family without its own recovery suffix belongs in
`syntax_skips_recovery`.

**Declarations match evidence.** `serving_caps` declares `Qualified` only
for what `ds4 --serving-check` or the family's gate passed on the artifact,
recorded on the family page; implemented but unqualified is `Present`.
`Qualified` changes behavior: an `Auto` MTP mode drafts whenever a drafter
is given, so a drafter that loses to plain decode stays `Present`. The
context defaults to the native maximum (`native_ctx_default`); a small card
lowers `-c`, the family does not. Regenerate
[serving-capabilities.md](docs/serving-capabilities.md) with
`cargo run -q -p ds4-core --example serving_docs`.

**State and reuse.** A serial rewind or a bank fork resumes only where the
shared ring rule holds (the ring, counted from the newest row written, still
holds the window, never below a restored snapshot); native tracks the
high-water mark and Rust asks native. Bank copies, forks and payloads carry
every KV layout the graph can hold (BF16 and FP8 rows alike). Test the rule
model-free.

**Switches.** Kernel and placement switches are `DS4_<FAMILY>_*`
environment variables read where they are used. No typed option surface,
CLI flag or ABI field for them. A measured fast path is the default with a
`=0` kill switch; an unadopted path is opt-in. Every switch is in the
family page's table.

**Shared and deprecated code.** Change shared files only where the family
needs it, keep their measured comments, word new comments
family-neutrally and gate every new branch on the family: an early return
added for one family must not skip another family's error path. Deprecated C
host logic (`vocab_load`, token text, the C sampler, the C hosts) takes no
features; a family gets at most the early return that keeps it out of the
other families' code ([C continuation](docs/rust-migration/C_CONTINUATION.md)).

**Build hygiene.** Every header an object includes is on that object's
Makefile rule (`ds4.o` for `ds4_<family>_*.inc`, `ds4_cuda.o` and each CUDA
gate for `cuda/*.cuh`); new test binaries go in `.gitignore` and
`BUILD_PROGRAMS`. No module-wide `#[allow(dead_code)]`, no unused functions
and no new warnings against main (`cc -Wall -Wextra` on `ds4.c`, `cargo
check --all-targets` on the touched crates).

### Evidence and records

- A performance claim names the commit or binary, the artifact (part
  hashes), the host, the clocks and the workload. On GB10 the reference is
  the 300–2,200 MHz policy; a record taken outside it says so beside its
  numbers, and its absolute rates are not compared with other families'.
- Sweep records live in `speed-bench/` as CSV, `.meta.txt` and plot, the
  `ds4-bench --csv` format; the record is data, its driver is not committed.
- Docs, README rows and declarations agree with the code's defaults. A
  default changed in the PR changes every page that states it.

### Family documentation

- `docs/<family>.md` in the sibling form ([A.X-K2 VQ](docs/axk2-vq.md),
  [Beam VQ](docs/beam-vq.md)): model and format, execution, output protocol,
  serving with the common commands, performance, quality, switches, limits.
  Operational facts only; rounds and their history go in
  `docs/<family>-optimization-<date>.md`.
- A row in the [README family table](README.md#supported-model-families),
  the [documentation index](docs/README.md) and
  [model families](docs/ds4-dfm-model-families.md), and a `CHANGELOG.md`
  Unreleased entry naming the contributor and the PR.

### Before opening the PR

Run `make ci-lint`, `make ci-rust` (or the affected crates' tests during
development), the family's `gate-<family>`, and a native build for the
architectures the PR claims. The description lists what changed, every gate
run with its result and host, and what is unverified. Split a large port by
layer (format and bind, kernels, serving, docs) so each part can be reviewed
and reverted alone.

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
