# C golden baseline

Historical oracle for the completed migration, frozen on 2026-08-31. The
commands and results below describe that campaign; do not repeat its branch
setup. Current release qualification is tracked in the
[v0.1.0 ledger](../releases/v0.1.0.md).

The migration campaign used the `v0.6.5-dfm` release lineage as its immutable
correctness oracle. Qwen behavior is frozen at the final C feature cut
`4d40d97`; see
[QWEN_V065_RESTAMP_2026-08-31.md](QWEN_V065_RESTAMP_2026-08-31.md).

Do not treat the C tree as “legacy reference only” while porting.
If Rust and C disagree, Rust is wrong until a documented C bug is
fixed in C first, identified as its own commit, and then ported.

## Frozen identity

```text
C baseline tag:     v0.6.5-dfm
C baseline commit:  d02e2a4777a34a9f52fd987453b3ea1801fac52e
VERSION file:       v0.6.5-dfm
git describe:       v0.6.5-dfm
Annotated tag:      07a7fc7a5d68e32170cb320b7a4d69042c85bb3a
                    (peels to d02e2a4)
Parent merge:       d02e2a4 Merge Entrpi v0.6.5 into ds4-dfm as v0.6.5-dfm
Upstream absorbed:  Entrpi v0.6.5 (addc0c4)
Qwen golden cut:    4d40d97 (post-tag image and two-bank completion)
rust-host import:   d0384d9 (parents 09e33bf + 4d40d97)
```

The original `rust-host` branch was created from `v0.6.3-dfm`. The baseline
was deliberately advanced by the two-parent merge `d0384d9`; later `dfm`
tips remain outside the frozen oracle.

```bash
git merge --no-ff 4d40d97
```

The `dfm` branch must not be merged again into `rust-host`. Semantics are
frozen at the identities above.

## Host recorded at branch creation

Recorded 2026-08-24 on the reference DGX Spark. This is the
environment identity, not a new throughput run.

```text
CUDA:          13.3 (nvcc V13.3.73, cuda_13.3.r13.3/compiler.38244171_0)
Driver:        610.43.02
Target GPU:    NVIDIA GB10 / sm_121a
Compute cap:   12.1
Host memory:   121 GiB unified
Kernel:        Linux 6.17.0-1031-nvidia
Build:         make cuda-spark
               (CUDA_ARCH=sm_121, gencode compute_121a/sm_121a)
Rust toolchain (host, not yet in tree): rustc 1.98.0 / cargo 1.98.0
```

`make cuda-spark` remains the only production build. Cubins must be
`sm_121a`. `-arch=sm_121a` alone is the documented silent-wrong-arch
failure.

## GB10 OPP-C goldens

The existing architecture-specific native drift snapshot is retained as
historical provenance:

```text
tests/proof/expected/cuda-opp-c-full-sm121a-v0.6.3-dfm.json
```

It records five 512-token profiles from scenario plan
`9da8afca7e450d7ad026b038edeffb6837f9b03c37a951f61f7e21fec0559d6b`
on GB10 / `sm_121a`. All selected-token streams have MD5
`6f1a24a5adf105053e9bed63fded9337`.

The stable-runner snapshot was written by a detached build of
`v0.6.3-dfm@516456f` and then checked exactly with the current C oracle on
2026-08-24.
The detached baseline binary SHA-256 was
`42c841df96bf38b9a0ade75bd116cff2fac11883072a9829e96c12e8da5f73a9`;
all five cells matched in both runs. This validates provenance without
replacing the older generic golden, which remains untouched.

With the persisted `CUDA_ARCH=sm_121` build configuration,
the historical file remains available by its explicit path.

That snapshot proves the older `v0.6.3-dfm` native path. It is not silently
relabelled as a `v0.6.5-dfm` result.

The active resumed-campaign snapshot is separate:

```text
tests/proof/expected/cuda-opp-c-full-sm121a-v0.6.5-dfm-4d40d97.json
```

It was generated from the detached Qwen C golden cut
`4d40d97f1e575400237a6e5cef21d7f74404a38d` with all embedded cubins verified
as `sm_121a`. The frozen C binary SHA-256 is
`c175d9a302194417442673a2d8b1ec35af0b3f6dbdc71b788f0c5e5da5d8d164`.
Its five 512-token cells use expanded-plan SHA-256
`7072c94b00a68e4a9dd9bc52f06a235aeb5670de46902370167aa50d61539b8c`;
all selected-token streams retain MD5 `6f1a24a5adf105053e9bed63fded9337`.
The portable plan-input SHA-256 is
`df4065094b7364e57861bba6049f98f1bf45c405a3847fcb62702c75944bbeaa`;
it excludes only the proof runner path, not any model, prompt, profile, or
runtime setting.

The frozen binary first wrote the new file and then independently checked it
5/5 under the 109/115 GiB memory guard. Evidence is under
`scratch/rust-host-live/v065-full-restamp-20260831-184200/c-oracle-oppc{,-check}/`.
With `CUDA_ARCH=sm_121`, `make proof-cuda-opp-c` now selects this v0.6.5 file.
The current C-native committed-snapshot proof and the Rust-host parity proof
both passed on 2026-08-31. Their retry logs are
`g1b-logs/proof-{cuda-opp-c,rust-cuda-opp-c}-retry.log` under the same evidence
root.

## What `v0.6.3-dfm` itself proved

From `docs/ds4-dfm-model-families.md` § Integration evidence for
`v0.6.3-dfm` (commit `516456f`):

- Absorbs typed schema-format refusal, chunked request bodies,
  think-dial counters / completion line, full-1M decode dispatch
  (DeepSeek MLA), whole-prompt depth fence, and best-fit trim victims.
- Family-side fix: Solar / EXAONE / Motif-3 continuous banks accept
  an exactly-full payload (`18dbd1e`).
- Gates on this binary: extractor self-test, `ds4_test --server`
  (including v0.6.3 refusal/fence/think units), split-GGUF,
  `test-model-family-kernels`, `test-mmq-parity`, `cuda-regression`,
  Motif loader/tokenizer/reference/CUDA, EXAONE kernels/reference,
  Solar loader/tokenizer/KDA/prefill/chunk/gates/KV + forward,
  dots3 loader/tokenizer.
- Live Motif MQ87-88 VMM owner + worker (`-c 2048`, 32 banks): four
  API surfaces, continuous route, 4/0. Typed `response_format`
  refusal is HTTP 400 in the native envelope.

`v0.6.3-dfm` did **not** remeasure Motif or Solar throughput. Earlier
tags are not moved. The published tables below remain the Spark
evidence band. Before the Phase 3 FFI performance gate, remeasure the
representative cells on this SHA (or a documented cherry-pick) with
AB/BA or ABBA. Do not treat a later `dfm` tip as the C number.

## Representative published Spark cells

Authoritative write-up: `docs/ds4-dfm-model-families.md`.
All cells are GB10, driver 610.43.02, CUDA 13.3, `sm_121a`.

### Motif-3 (production MQ87-88, aligned-Q8 VMM owner)

Published `v0.5.6.3-dfm` table (`593d251` HTTP, `cc2f277` 8K bench):

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Motif-3 MQ87-88 | 8,192 (`ds4-bench`) | 519.55 | 12.28 (64 tok) | not claimed (non-stream bench) | aligned-Q8 owner path | source-GGUF map RSS 29,632 KiB after 256K cell (that cell) |
| Motif-3 MQ87-88 | 32,768 Chat | 396.47 | 8.96 (43 tok) | no independent TTFM | — | — |
| Motif-3 MQ87-88 | 262,080 Chat | 175.61 | 2.52 (43 tok) | no independent TTFM | worker 9.703 GiB; latent KV 4.119 GiB | owner+worker `VmSwap: 0` |

Later remesure on the `v0.6.2-dfm` line (engine tip `2c81427`, kernels
through `a09ff4f`; **not** a moved tag):

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Motif-3 MQ87-88 | 8,192 (`ds4-bench`) | 627.19 | 15.06 (64 tok) | not claimed | aligned-Q8 owner | — |
| Motif-3 MQ87-88 | 32,768 Chat | 546.7 | 12.8 | no independent TTFM | — | — |
| Motif-3 MQ87-88 | 262,080 Chat | 238.59 | 5.97 (43 tok) | no independent TTFM | worker 10,429 MiB; latent KV 4.119 GiB | `VmSwap: 0`; ~11–12 GiB available |

196K multi-bank Motif cell (`03b7002` / `cf605e0`): 8,214-token cold
prefill 266.3 tok/s; 192-token decode 490.4 ms TTFT, 12.9 tok/s;
owner 90,119 MiB, worker 22,283 MiB after the 8K requests.

`--no-repack-q8-aligned` is not the published Motif point.

### Solar Open2 (MXQ-v1, `b2e52b9`)

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Solar-Open2-250B MXQ-v1 | 8,222 Chat | 1,050.7 | 19.05 p50 / 18.9 API (128 tok) | not separately published | 3 banks at `-c 196608` | — |
| Solar-Open2-250B MXQ-v1 | 66,761 Chat | 804.5 | 13.07 p50 / 14.1 API | not separately published | same | — |

### Unvalidated on this host

- dots3-note 524,288-token prefill + decode: **unvalidated**
- Motif concurrent 256K banks: **not claimed**
- Solar 1,048,576-token serving: **not claimed**
- DeepSeek full-model GPU tests: no DeepSeek GGUF on this host

## Performance gate numbers (provisional)

Rust may ship a host change only when a same-host AB/BA or ABBA
remeasure against the C binary from this baseline (or an identified
cherry-pick) stays inside:

| Metric | Rust allowance |
|---|---:|
| Prefill | ≥ 97% of C |
| Decode | ≥ 98% of C |
| TTFT | ≤ +5% vs C |
| Host RSS | ≤ +5% vs C |
| GPU resident | no meaningful increase |
| Token correctness | existing contract |

A 2–3% regression with no explained cause (extra memcpy, allocation,
lock scope, FFI granularity, scheduler change) is a fail.

## Proof harness that must remain reachable

These Makefile targets are release gates, not smokes. The Rust
execution path must become able to run the same contracts:

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
```

`proof-cuda-long` is captured-vs-eager parity (essay prompt, n=1024,
FP32, every enabled overlay). Migration does not relax it.

## Artifacts this baseline serves

Paths are host-local; identities are from the family doc.

| Family | Serving input |
|---|---|
| Motif-3 | `Motif-3-MQ87-88-FIT.gguf` (94,162,541,472 bytes canonical) |
| Solar Open2 | `Solar-Open2-250B-MXQ-v1` 11 shards (95,533,532,160 bytes) |
| K-EXAONE | 3-shard MXQ (85.56 GiB) |
| dots3-note | `dots3-note-prev-MQ87` 10 shards (80.156 GiB) |
| Qwen3.8 Flash Next | `MQ-Q5-SSD-PLE-BF16` 3 shards (83,274,984,384 bytes) + shared 4-file SSD-PLE sidecar |

## Re-measure checklist (before Phase 3 performance sign-off)

Record SHA, GGUF revision, exact command/env, token counts, cold/warm,
cached tokens, TTFM/TTFT, prefill/decode wall and tok/s, memory peak,
swap, clocks, and the correctness fixture result. Prefer:

```text
C → Rust → Rust → C
```

on the same Motif 8K `ds4-bench` cell and the same Solar 8K Chat cell
used above. Do not update this file’s published historical rows to
hide a regression.
