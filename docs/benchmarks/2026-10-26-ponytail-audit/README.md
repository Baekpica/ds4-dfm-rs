# Ponytail audit repairs and bounded qualification

Baseline: `02a6b5be83231f847d89433a90a48334bdfce247`.
Implementation: `4a6d030d` (22 separate commits). Host integration ran at
`8ee84316`; the final ACL correction also passed the affected CLI tests,
agent parity and rebuilds.
This report records the approved 19 defects and scoped cleanup. It does not
adopt the audit's speculative 52,600-line deletion estimate.
Machine-readable receipts: [host/native checks](validation.json),
[model runs and artifact identities](model-receipts.jsonl), and
[local experiment cleanup](local-cleanup.json).

## Defects

A2/A3 identify the supplied audit and its numbered finding. Each row has a
focused regression; the original failures and passing receipts remain in
`scratch/ponytail/` in this checkout. C oracle golden data is unchanged.

| Finding | Change | Verification |
|---|---|---|
| A2-3 / A3-1 | Checked FFI shapes, buffer lengths and overflow before layer-slice/MiMo native calls; private session shape | Undersized input/output and invalid visual layouts reject before FFI; valid boundaries pass |
| A2-1 / A3-2 | Agent write/edit use same-directory staging and atomic replacement; preserve mode/ownership, Linux access ACLs and symlink targets; refuse hardlinks | Partial-write, ACL, flush and rename fault injection retain original bytes; inherited staging ACLs cannot change original access |
| A3-3 | Edit rereads after approval and checks replacement anchors again | Concurrent unrelated edits survive; removed or ambiguous anchors reject |
| A3-5 | KV trailer updates stream old payload into a replacement file | Failed replacement retains the old payload, replay mapping and identity |
| A3-12 | Trailer growth participates in disk budget/eviction; protect the replacement target | Insufficient capacity fails without removing the original |
| A2-2 | Agent KV uses the server's artifact/sidecar/settings identity | Different or absent identity rebuilds from saved text instead of loading foreign KV |
| A2-9 / A3-15 | File-based checkpoint write; metadata-only catalog and pinned-file restore | 8 GiB sparse payload under a 512 MiB address-space limit; titles/IDs/text preserved |
| A2-4 / A3-4 | Prefetch transfers exclusive socket-read ownership during snapshot restore | Nonempty multichunk restore through the default prefetch path |
| A3-6 | Connection-local partial-frame reader survives idle timeouts | Header/body fragments separated beyond the 25 ms timeout |
| A2-7 / A3-8 | EOF connections are reaped after pending work drains | Repeated transfers with a live coordinator do not accumulate connections/threads |
| A2-5 / A3-10 | Shared Completion cursor sends newly safe bytes and final remainder | Serial/continuous/push stop splits, UTF-8 splits and final flush |
| A2-6 | Responses result-only continuation skips leading system/developer instructions for call-ID matching | Valid continuation succeeds with and without instructions |
| A3-11 | Structured Anthropic tool results retain the existing failure marker | Final Jinja input distinguishes empty success from empty failure |
| A2-10 | Four omitted model families participate in failed-generation synchronization | Native generation changes invalidate Rust checkpoints |
| A3-7 | Continuous owner consumes settled results and removes slots immediately | 2,000 completions with a held bank keep storage bounded; deferred fallback drains in order |
| A2-8 / A3-9 | A watcher owns the unreaped child and enforces the process-group deadline independently | Timeout without polling, normal exit, manual stop and Drop; lifecycle test isolated from concurrent test threads |
| A2-12 / A3-16 | Fixed-buffer log statistics and bounded seek/tail | 128 MiB sparse log under 32 MiB additional address-space allowance |
| A2-11 / A3-13 | Docker installs the pinned Rust 1.98 toolchain; excludes host build products | Clean CUDA base reproduces missing Cargo; clean-base and final-source images build all production hosts |
| A3-14 | GGUF directory offsets use checked arithmetic | Malformed 32-byte input returns an error instead of panicking |

## Cleanup and preserved scope

- Default CUDA aliases build Rust hosts; all four C host sources and explicit
  targets remain. Proof/Bonsai targets declare their C dependencies.
- CLI workers reuse the existing server entry. Public adapters/signatures remain.
- Remove the always-admit host governor hook; keep native admission, refusal
  reasons and metrics.
- Remove the rejected Q8_1 fused MoE, Ling MLA tile/TK=32 and GLM dense-GEMM
  candidates, including their exclusive fixtures, dispatch and current guidance.
- Consolidate 174 identical Make rules. Compiler-generated dependency files
  cover transitive headers. `clean` names known outputs explicitly.
- Remove the copied QA approval gate/report; retain the real `tests/run.sh`
  checks. Wire three model-free IQuest suites into host CI.
- Stop tracking the two generated think/nothink prompt splits. The existing
  generator's `--render-splits-only` reproduces their 12,086,017 bytes exactly.

Individual cleanup commits (net lines, including their tests/current docs):

| Cleanup | Commit | Net lines |
|---|---|---:|
| CLI worker reuse | `768bd58e` | -162 |
| Always-admit hook | `8980c117` | -155 |
| Copied QA gate/report | `e4cd567d` | -1,566 |
| CUDA C-host build split | `38ac9906` | +4 |
| Make rules and depfiles | `ec156a5b` | -327 |
| Q8_1 fused MoE | `2d48e761` | -760 |
| Ling MLA variants | `4da463d2` | -217 |
| GLM dense candidate | `51f1325d` | -332 |

At implementation commit `4a6d030d`, `git diff --numstat 02a6b5be HEAD`
reports 2,797 added and 4,336 removed lines excluding those two generated
text files: **1,539 net fewer lines**. The generated data contributes another
225,938 removed text lines; it is not counted as code. This report and its
receipts are subsequent documentation additions.

Metal, C oracles, public model/bridge APIs, snapshot gather/merge, decode2
and its proof configuration remain. Broad VQ/MTP/JSON/Terminal abstractions,
CPU variants, vendor features, ctrlc replacement and undocumented bulk
campaign/probe removal were excluded. Missing CI wiring alone is not proof
that a test has no unique validation role.

## Host and native verification

- `cargo fmt --all -- --check`, workspace Clippy and workspace all-target
  check passed. Clippy reports warnings; a warning-free build was not a gate.
- All eight C-oracle parity gates passed. Serialized
  `cargo test --workspace --locked -- --test-threads=1` passed; ignored
  hardware/manual tests were not silently counted as executed. Its 107
  summaries include nested subprocess tests, so their 1,933 pass entries
  are not a count of unique tests.
- `make -j2 cuda-spark` and explicit `ds4-c ds4-server-c ds4-bench-c
  ds4-agent-c` builds passed with CUDA 13.4, `CUDA_ARCH=sm_121`.
- Linux 174 and Darwin 138 Make dry-run recipes matched their prior commands
  apart from depfile flags. Transitive-header rebuild and isolated cleanup
  of 612 known output/dependency paths passed; unrelated files were preserved.
  This is not a macOS build or Metal execution qualification.
- Retained SASS is strictly identical after each native deletion: Q8_1
  cleanup, 1,070 CUDA and 768 MMQ functions; Ling cleanup, 1,069 CUDA and
  32 FATTN functions; GLM cleanup, 1,066 CUDA functions. Only the candidate
  instantiations disappeared. Corresponding GPU fixture outputs matched the
  saved baseline. Five GLM lazy-allocation ASan/UBSan cases also passed.
- `docker build --no-cache -t ds4-ponytail-final -f docker/Dockerfile .`
  passed on the clean CUDA 13.0 base. Image ID:
  `sha256:af78d72992d6c183bfddfe4f03548fc682c84e59237cacbb3ea5058cbd6d28e1`.
  All four production hosts print usage with `docker run --rm --gpus all`.
  Rust server `--help` retains its pre-existing exit code 2; the other three
  exit 0. Without GPU driver injection this GPU image lacks `libcuda.so.1`.
  The final ACL correction was rebuilt from a clean source layer on those
  same prerequisite layers as `ds4-ponytail-final-acl`, image ID
  `sha256:61801cd85ebd398b01f46ee1de03d16d3fc95b2597790d1e11eb307a67537b32`.
  Its four help commands have the same documented results.

## Model smoke protocol

Model root: `/home/sunghoon/workspace/ds4-exaone/models`.
The directory inventory has 34 entries, but directories are not individual
models. The qualification set contains 21 selected base-model artifacts:
one current GLM release, Ling recipe v2, the other distinct model entries,
and the two separately published DeepSeek VQ/iVQ artifacts. Historical GLM
revisions/experiments, Ling v1 and a standalone-ineligible DFlash sidecar
are excluded. Exploratory runs of historical GLM files do not increase the
qualification count.

One model at a time on GB10, fresh process/session, zero prefix reuse,
approximately 8,192 prompt tokens and up to 32 plain decode tokens. The
existing `speed-bench/promessi_sposi.txt` corpus is used. No MTP, media,
maximum-context, sustained concurrency or quality benchmark is implied.
`host_memory_guard.py` retains a 2 GiB reserve and trip floor. Early jobs
recorded PSI without a stop threshold; after the interrupted Ling load
below, remaining jobs use the documented PSI 40% threshold.
Before each run, inspect nvtop and btop, confirm prior processes exited,
clear page cache, and confirm recovery. The 300–2200 MHz clock policy is
preserved and observed clocks are recorded.

For ordinary families, the existing `ds4-bench` uses `--ctx-start 8192
--ctx-max 8192 --ctx-alloc 16384 --gen-tokens 32 --warmup-prefill 0
--mtp-draft 1 --dump-frontier-logits-dir ...`, with
`DS4_MTP_SPEC_DISABLE=1`. A successful receipt requires 8,192 fresh prefill
and 32 generated tokens, finite positive rates, a finite winning logit,
valid token IDs and nonempty decoded text. Masked vocabulary logits may be
nonfinite. Read the generated continuation as an additional smoke check.

The reported load/setup interval includes guard admission, model open,
tokenizer/session preparation and any existing boot prewarm. It is separate
from measured prefill/decode and is not a pure disk-loading benchmark.

Canonical-owner cases start `ds4_weight_server --backend vmm --scope base`
with `--no-repack-iq2-aligned --no-repack-q8-aligned
--no-repack-q2k-aligned`. The worker imports its manifest; both processes
share one guarded scope and one weight allocation. These rows qualify this
explicit configuration, not an aligned-repack performance configuration.
Qwen variants select the locally supplied `PLE-FP8` directory through
`DS4_QWEN_PLE_DIR`. Every attempt retains its exact command/environment.

The two ds41 artifacts use the existing HTTP server because the benchmark
entry does not route that family. Each starts with its local Engram shards,
`--mem-budget-mb 110000 --no-dspark --no-kv-reuse`, then receives one raw
`/v1/completions` request with approximately 8,192 tokenizer-counted prompt
tokens, `max_tokens: 32`, `temperature: 0` and `stream: false`. Its receipt
checks fresh prefill, positive finite rates and nonempty text. HTTP does not
expose the benchmark's frontier-logit dump. For these rows, load/setup is
server start to `/v1/models` readiness inside the admitted guard; request
wall time is recorded separately.

The initial Inkling direct attempt spent 577.7 seconds in load/setup and
hit the harness's 600-second total deadline during the request. Its partial
result is discarded; subsequent bench jobs allow 1,800 seconds for loading
and the short workload while retaining the memory floor. IQuest direct
loading failed at canonical routed-weight copying; its documented VMM owner
configuration is recorded separately. K2 external owner stopped before
readiness; that topology was already unqualified in the family guide.
These failed attempts are retained alongside successful configurations.

The first 16 successful artifact runs preceded the final ACL correction.
They are retained under `scratch/ponytail/models-pre-acl/`, together with
failed attempts. Although this correction changes agent file writing, the
rebuilt benchmark was not byte-identical. Final qualification therefore
uses fresh runs on the rebuilt binaries; the earlier passes are not counted
as final-build passes.

## Final-build model results

All **21 selected artifacts** completed 8,192 fresh prompt tokens and
32 plain decode tokens on the frozen `4a6d030d` binaries. The table is a
single-run smoke, not a comparative speed qualification. MiniMax's raw
continuation and extra official-chat result are qualified separately below.

| Selected model | Configuration | Load/setup s | Prefill tok/s | Decode tok/s | Min available GiB |
|---|---|---:|---:|---:|---:|
| IQuest Q1 | VMM owner | 55.06 | 105.17 | 6.86 | 32.97 |
| Inkling Small | VMM owner | 55.79 | 230.90 | 12.53 | 34.84 |
| K-EXAONE | VMM owner | 58.37 | 365.33 | 12.89 | 29.64 |
| Motif 3 | VMM owner | 57.02 | 263.59 | 12.12 | 22.23 |
| Solar Open2 | VMM owner | 59.39 | 581.33 | 15.68 | 25.28 |
| K2 Horizon | Direct | 112.44 | 591.28 | 12.12 | 90.63 |
| Ling 3.0 Flash VL | VMM owner | 52.48 | 2208.94 | 26.00 | 35.79 |
| MiMo 2.6 Flash MOPD | VMM owner | 57.34 | 707.18 | 19.89 | 29.39 |
| Naive N0.5 | VMM owner | 54.34 | 391.54 | 16.44 | 32.79 |
| Step 3.7 | VMM owner | 53.86 | 983.30 | 18.30 | 35.40 |
| dots3 note | VMM owner | 53.27 | 217.37 | 15.29 | 30.22 |
| Darwin 180B RSI | VMM owner | 49.75 | 727.06 | 25.14 | 35.95 |
| Qwen 3.8 Flash Next | VMM owner | 49.13 | 726.73 | 25.17 | 35.91 |
| Swift 1.5 / Qwen 3.8 | VMM owner | 50.15 | 725.49 | 25.05 | 35.83 |
| ThinkingCap / Qwen 3.8 | VMM owner | 50.65 | 725.87 | 25.18 | 35.89 |
| GLM 5.3 VQ (HF a3f50caf) | Direct | 53.09 | 1193.12 | 23.96 | 21.42 |
| Beam 1.0 VQ | Direct | 41.84 | 1396.82 | 26.50 | 22.04 |
| Ling 3.1 VQ v2 | Direct | 70.53 | 939.65 | 16.93 | 7.62 |
| MiniMax M3.1 VQ | Direct | 50.22 | 1155.27 | 21.15 | 11.60 |
| DeepSeek V4.1 1017 VQ | HTTP | 150.84 | 952.30 | 30.40 | 7.32 |
| DeepSeek V4.1 1017 iVQ | HTTP | 150.37 | 947.30 | 30.50 | 7.34 |

Final executable SHA-256 values:

```text
9b1d15aa0a438c6332274aee9339e6c34b2a53b630e7ff1ce0435d976f8cb58c  ds4
33612c0c0238b1b8e689bcd644ab5557ada8b20c30ed611d080cdf1608148dba  ds4-server
11c5f9591b85d5eeeb68e923054bd3b1f30cd6abfd8f5aca0b301634dff284d3  ds4-bench
a52a5bd91dad6ea5df9dac1e8488ddde79eb5e7a732fa5bb04ab03ff7828e751  ds4-agent
ac42040a89c173f6b86e9b5dcb475e7a692b0c6534319be779ba18adadc7b76b  ds4_weight_server
```

All owned model processes exited. Final nvtop showed no compute process;
btop showed about 118 GiB available after cache reclamation.

## Artifact identity and excluded history

GLM is one published model, not twelve models inferred from directory
names. At verification time HF `Baekpica/GLM-5.3-Flash-VQ-GGUF` main is
[`a3f50caf1d63912adf9ad2b63973a140cdfe4f2d`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF/tree/a3f50caf1d63912adf9ad2b63973a140cdfe4f2d).
Its local snapshot is `GLM-5.3-Flash-VQ-Q4K-v3-GGUF`: the 19-layer R13+8
release, despite the older directory name. The current tensor manifest is
`77835380296e7ff85c00a7f80859eab3494516464247a1a0e01d494ab0c3d85d`.
All 76 files were checked against the pinned HF revision: SHA-256 for
LFS files and Git blob SHA-1 for regular files, including the full weight
payloads. Every digest matched; the verification receipt records each file.
The [October 19 record](../glm53-vq-spark-64k-19-layers-2026-10-19/README.md)
explains the succession from all-R13 to 14 and then 19 residual layers.

Other GLM directories are retained revisions or development experiments.
The early directory-wide sweep is preserved as exploratory evidence,
including the old Q4_K recipe's inventory rejection and the NonExpert
experiment's layer-3 prefill failure. They are not failures of the selected
latest HF revision, which completed the final-build smoke. No compatibility
change or additional run for those historical experiments was made.

Ling recipe v1 is superseded by v2 in the same model repository, so only
`Ling-3.1-flash-VQ-GGUF-v2` is selected. `Beam-1.0-501B-A23B-DFlash2` is a
drafter sidecar with no standalone base GGUF and is excluded from plain
base-model qualification.

## MiniMax output check

MiniMax completed the 8,192-token raw-book benchmark, but its 32-token
continuation repeated `my...`. That receipt is a runtime/cold-prefill pass,
not evidence of acceptable free-form continuation quality. A fresh official
chat-template run (`--nothink --temp 0 --mtp-draft 1`) then processed an
8,391-token formatted prompt and answered the requested `READY`, followed
by its end-of-turn token. Both selected-token logits and top-k log
probabilities were finite. Preserve both observations; do not turn the
second short answer into a broad quality claim or an explanation of the
first result's cause.

## Interrupted load and recovery

The first final-build Ling v2 attempt has no valid workload result. During
weight loading, full-memory PSI rose above 70% while MemAvailable remained
about 12 GiB. At 17:32:30 KST, systemd-oomd killed the parent PTY scope; at
17:32:45 it also killed the Hermes gateway. The kernel logged
`NV_ERR_NO_MEMORY`, and the surviving benchmark subsequently reported a
broken output pipe. This was an actual host-pressure interruption, not a
completed inference result or a mere observation timeout.

The task processes exited, nvtop showed no compute process, and cache
reclamation restored about 118 GiB available. Hermes was already active
again when checked; it was not manually restarted by this task. The repeat
uses the Ling campaign's documented PSI 40% guard threshold and passed
8K/32 without pressure growth (minimum available about 7.6 GiB). The
interrupted receipt and system/kernel logs remain in `scratch/ponytail/`.
A 2 GiB available-memory floor alone does not prevent system-level OOM
policy from acting on pressure.

## Local experiment cleanup

At the user's request, remove only experiment directories without external
references or live users. The search covered ds4 workspaces, operator
scripts/runtime/configuration, home/HF-cache symlinks, process mappings/open
files and Docker mounts. Historical benchmark references also cause a
directory to be retained. No model symlink target or live consumer was found
for the four removed directories:

- `GLM-5.3-Flash-VQ-Q4K-GGUF`
- `GLM-5.3-Flash-VQ-Q4K-NonExpert-GGUF`
- `GLM-5.3-Flash-VQ-Q4K-NonExpert-v3-GGUF`
- `GLM-5.3-Flash-VQ-Q4K-v3-KeepHead-GGUF`

This unlinked 296 file entries, reclaiming about **21.53 GiB**. Shared
hardlink inodes retain their other names; no shared file was rewritten.
The latest GLM snapshot's file identities, sizes, modification times and
metadata hashes are unchanged. Seven historical GLM directories remain
because recorded experiments reference them. These local artifact deletions
are separate from the source-code line counts above.

## Reproduction and limits

PR #112 review found a transient ACL exposure: restoring the original group
mode could activate a different inherited named-user ACL on the staging file.
The follow-up applies the original ACL before restoring mode bits.
`tests/agent_acl_window.py <ds4-cli lib-test binary>` delays ACL removal with
`strace` and checks effective staging access. It failed before the reorder
and passed afterward with 169 observations; `make -j1 test-agent-parity`
also passed (107 agent tests, one manual fault test ignored).
[Follow-up receipts](review-followup.json) record source and log hashes.
Only agent file metadata ordering changed; the CUDA/model results above
remain the recorded `4a6d030d` runs, not new release-tag measurements.

Integration commands run for this change:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
make test-kv-parity test-web-parity test-dist-parity test-server-parity
make test-catalog-parity test-tokenizer-parity test-agent-parity test-session-parity
make -j2 cuda-spark
make -j2 ds4-c ds4-server-c ds4-bench-c ds4-agent-c
```

The final ACL change also passed `cargo test -p ds4-cli --locked`,
`cargo clippy -p ds4-cli --all-targets --locked`, formatting and
`make test-agent-parity`. Fault injection uses the existing ignored manual
file-preservation test under `strace`, including fsync/rename/ACL failures.

The six fully removed tracked paths are `cuda/glm53_dense_attn.cuh`,
`tests/test_glm53_dense_gemm.cu`, `tests/qa-gate.sh`,
`qa-evidence/qa-report.md`, and
`gguf-tools/imatrix/dataset/rendered_prompts_{think,nothink}.txt`.
The generated prompt files remain reproducible locally.

Atomic KV replacement needs temporary disk space and rewrites the payload;
it is deliberately not an in-place trailer update. Agent writes reject
hardlinked targets before modifying them. The checks cover Linux access
ACLs, modes, ownership and symlink targets.

No Metal execution, maximum-context/media/MTP qualification or sustained
production load test was run. `/data/YoungAi` is absent on this host, so
there is no ds41 reference-engine comparison or port-specific change.
