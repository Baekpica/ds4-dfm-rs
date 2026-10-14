# DeepSeek V4.1 vision review on DGX Spark

Review of [JLFN's PR #83](https://github.com/Baekpica/ds4-dfm-rs/pull/83),
starting at `009c76dfa3dc687b2b384d1c34e81db52b116f6d` on main
`3513f26c399ede191bb9e01ca4e9e1f3cd9293ad`. This record covers image
correctness, cache identity, tool results, admission and redundant tower work.
It does not extend the family's serial serving or qualified-context limits.

## Source and artifact contract

The reference is **fodelf/YoungAi**. The base port remains pinned to
[`7fb26cc`](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5);
vision behavior was checked against
[`846c929c`](https://github.com/fodelf/YoungAi/tree/846c929c5d11ef4f89d6406e8504e6e66db3ce69).
The `/data/YoungAi` path was absent on this host; exact pinned, read-only
checkouts supplied the engine source and its bug, speed, MTP and run records.

- The Engram image boundary follows
  [`core_v41_engram.c:66–93`](https://github.com/fodelf/YoungAi/blob/846c929c5d11ef4f89d6406e8504e6e66db3ce69/src/core/core_v41_engram.c#L66):
  encountering an image pads that lookup and all earlier n-gram positions.
- Per-position image identity follows
  [`core_v41_vision.c:207–213`](https://github.com/fodelf/YoungAi/blob/846c929c5d11ef4f89d6406e8504e6e66db3ce69/src/core/core_v41_vision.c#L207).
  The port additionally binds encoded image bytes, preprocessing configuration
  and sidecar bytes. Appending an image preserves the existing prefix;
  replacing or reordering images invalidates the affected checkpoint.
- Port-specific work: Rust HTTP block ordering, resource admission, native
  extent checks, model-lifetime cleanup and disk serialization. The Responses
  image-array `function_call_output.output` is a host extension.
- The optimization skips tower execution for already-cached image rows and
  encodes only the required image suffix. Per-image arithmetic is unchanged.
  `DS41_VISION_REENCODE=1` restores full reencoding for comparison. No tower
  weights or embedding cache is retained across requests.

All vision artifacts are under
`~/workspace/ds4-exaone/models/YoungAi-DeepSeek-V4.1-Flash`.
The [vision receipt](vision-artifact.json) pins Hugging Face revision
`5d89e9ca59452ec191121ebe5ce2a998191648e0`: 971,066,016 bytes, SHA256
`b50974c405aafbd941f8e8767cf90231b522aa4f2fcb506eaea68fbf8b409af5`,
verified against LFS. The base GGUF and original Engram shards retain the
[previous verified identities](../ds41-spark-review-2026-10-13/README.md#model-and-source).

Host: DGX Spark / GB10, CUDA 13.4.92, driver 615.71.09. The user-managed
300–2200 MHz range was preserved; no clock-setting command was run.
Build: `make cuda-spark`, then `make ds4 ds4-server` after final ABI fixes.
[Binary hashes](binaries.sha256) and the [build input snapshot](build-inputs.json)
identify the tested dirty-tree patch on the PR head. Subsequent documentation
edits do not alter the tested runtime.

## Review fixes and model-free checks

Every behavior fix first failed a focused regression test, then passed:

| Boundary | Fixed behavior |
| --- | --- |
| Engram | Image-aware n-gram padding, checked against an independent C fixture |
| Live/disk KV | Per-position signatures; saved-prefix comparison; strict payload length; old disk ABI safely misses |
| Model lifecycle | Drop native image/KV state and Rust pointer-keyed mount/Engram caches |
| Failed mount | Invalidate the previous mount before native writes, so a failed B mount cannot make a later A retry skip restoration |
| Media admission | Validate headers, decoded bytes, expanded context and aggregate working memory before decoding/allocating all images |
| FFI | Validate patch slice extents, loaded tower geometry and contiguous maps, including the raw golden runner |
| Rust API | Keep pixels and private signatures inseparable; callers rebuild image inputs instead of mutating prepared payloads |
| HTTP | Preserve Anthropic mixed image/tool order; accept Responses tool image arrays; remove the ds41 image-only four-image cap |
| Gate scripts | Missing output, failed requests and incomplete goldens fail the gate instead of passing silently |

[Host checks](host-checks.jsonl) passed: formatting, Clippy, all eight host
parity targets, affected native gates, **1,711 serialized workspace tests**
and all-target checking. Additional ds41 registration, source-release, MoE,
GR and trace targets passed. The actual native image/golden boundary fixture
passed all **37 cases** under ASan and UBSan. The shell gate mock covers
18 scenarios, including 17 cases previously capable of a false pass.

## Numerical and state checks

One model open ran **31 generations / 8,555 emitted tokens**, three official
golden fixtures and **12 comparisons**. [Audit](gpu-audit.json),
[cases](gpu-stats.csv), [comparisons](gpu-comparisons.csv) and
[actual graph/rollback evidence](gpu-graph-evidence.csv) all pass.

| Golden fixture | Tower row cosine mean | Minimum | Max absolute error |
| --- | ---: | ---: | ---: |
| Market panel | 0.999804 | 0.990113 | 0.01953 |
| Bug screenshot | 0.999875 | 0.996753 | 0.01768 |
| Homer | 0.999716 | 0.990216 | 0.02148 |

The [raw golden metrics](golden-metrics.log) also record exact PNG pixels and
the JPEG decoder's maximum four-level pixel difference. JPEG end-to-end row
cosine is 0.999868 mean / 0.997257 minimum. These are the upstream fixture
gates, not a broad image-understanding benchmark.

- Cold/full hit and one-token suffix reuse match tokens and every finite
  full-vocabulary logit. Changed and reordered images miss and match cold
  references. Forced full tower and suffix-only tower match on text follow-up
  and appended-image cases.
- Image plus essay: **5,782 input tokens / 1,024 outputs**, bare and code GR,
  eager and captured. Pure decode matches every full-vocabulary logit and
  token; each captured run executes 1,022 graph steps.
- Fixed-5 MTP matches those output tokens, with actual rejected-draft rollback
  checks. Bare: 482 checks / 470 with rejects; GR: 463 / 454, in both eager
  and captured modes. MTP dumps contain only single-decision rows and are
  explicitly **not** full speculative-logit equality evidence.

Connection loss interrupted the wrapper's clock/guard telemetry during load;
the native driver continued and its complete receipts passed the independent
audit. This diagnostic run therefore does **not** establish continuous clock
or memory-guard coverage. Its timings are not speed measurements. A prior
partial run stopped for two additional fixes is superseded by this run.

## Live HTTP and disk restart

All **11 requests passed in two fresh, sequential processes**, with pure
decode, Engram, code GR and an isolated 1 GiB disk cache. The
[report](api-report.json) and [cases](api-results.csv) record:

- The exact request replay after restart restores 219/219 image-prompt tokens
  and matches returned choices, token IDs and all 16,547,840 logit bytes.
- Changed content and reordered equal-geometry images miss live and disk KV.
- Chat returns an actual screenshot call; its original call ID and an image
  result produce assistant text. Both Anthropic mixed block orders
  preserve two image spans and return text.
- Responses image tool output works; extending explicit history with a
  second image reuses the original 236-token prefix.

Both task-owned processes stop with no GPU owner or listener left. Diagnostic
readbacks are excluded from performance claims. These are bounded protocol
and state checks; their free-form prose is not an image-quality oracle. Chat
follow-up and Responses extension reach the 64-token cap; they establish
continuation, not completed answers.

## Performance method

The original PR's whole-workload measurements identified repeated image
encoding on cached text follow-ups: small-image tower 5.59 s of 5.98 s TTFT;
four-image tower 9.11 s of 9.63 s TTFT. Appending an image lost all 226 cached
prefix tokens. Five/eight images returned HTTP 400.

A bounded Nsight Systems 2026.3.2 trace used the original CLI and the same
Homer tower shape twice. The [interval analysis](profile-tower.json) excludes
the tail of model loading: 525 H2D copies / 1,944,327,680 bytes / 49.83 s,
versus 1,816 kernels / 0.258 s total. It omits the first patch-embed transfers.
The two tower wall times were 25.52/24.97 s under profiling; profiler overhead
is not a speed result. An earlier incomplete interactive capture is rejected.

The completed correctness run contains two same-input comparisons, each with
32 emitted tokens and identical saved prefixes:

| Case | Full reencoding wall | Suffix-only wall | Native prefill before / after |
| --- | ---: | ---: | ---: |
| Image then text follow-up | 9.311 s | 1.588 s | 8.2 / 0.5 s |
| Append another image | 9.543 s | 8.855 s | 8.5 / 7.8 s |

The first case skips a 7.71 s tower pass. The second encodes one image instead
of two. These are **single observations within one process**, including
diagnostic readback, with exact output/logit parity. They are not repeated
fresh-process speed qualification or a decode-speed claim. Cold tower times
varied substantially during this campaign; the cause remains unproven.

The user requested shipping the reviewed fixes before additional optimization.
The six-process A/B queue was cancelled before its first pair; the original-PR
cold timing control was stopped during its request. Neither supplies a
completed performance result. Further profiling and performance qualification
belong on a new branch. Five/eight-image preprocessing is model-free tested;
the queued live five/eight-image sweep did not run.

## Reproduce

The [archive](reproduce/manifest.json) preserves exact harness inputs and
paths. From the repository root, restore them without replacing different
local files:

```sh
python3 docs/benchmarks/ds41-vision-review-2026-10-14/reproduce/setup.py
python3 scratch/pr83-review/vision-gate/prepare.py
```

Build the production CUDA binaries, then link the one-open driver against
that build's exact `libds4_core-*.rlib` using
`scratch/pr83-review/vision-gate/harness.mk`. The target freezes native/Rust
inputs before and after linking and never rebuilds production code. Run it
under `tools/host_memory_guard.py`, with the four artifact paths, existing
`tests/fixtures/vision`, generated `vision-gate/inputs`, a fresh output
directory, `--long scratch/vq-pr80-81/live-fixtures/long-context-essay.ids
--mtp`. Run `vision-gate/audit.py OUT NATIVE_LOG` afterward.

For HTTP/disk checks:

```sh
python3 scratch/pr83-review/api-gate/run.py ./ds4-server \
  scratch/pr83-review/api-gate/result-final --port 8099
```

Use fresh output directories; stop other task-owned runs first and preserve
unrelated GPU owners. No fixture or script sets clocks.
Full native logs, logits, traces and disk files stay in the ignored local
`scratch/pr83-review` evidence directory; compact results and hashes are
published here. CSV copies use LF line endings; audit hashes identify the
original raw files.

This record does not qualify arbitrary context lengths, concurrency, banks,
session snapshots, remote URLs, other artifacts, or general image quality.
