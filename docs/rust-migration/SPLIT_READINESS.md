# ds4-dfm-rs split readiness

Historical decision: **GREEN for guarded repository seeding**. The split
completed at genesis `fe7733fb4f7e18204b6ea0a00fe3b136d2029b17`. This record
preserves the original decision and evidence; seeding is not a pending task.
Use the [v0.1.0 ledger](../releases/v0.1.0.md) for current qualification.

Decision date: 2026-08-31 KST

This document closes the pre-split Rust-host campaign. It does not authorize a
CUDA/MMQ rewrite: the independent repository keeps the Rust host over the same
native backend.

## Immutable identities

| Item | Identity |
|---|---|
| Release baseline | annotated `v0.6.5-dfm` tag object `07a7fc7a5d68e32170cb320b7a4d69042c85bb3a`, peeling to `d02e2a4777a34a9f52fd987453b3ea1801fac52e` |
| Qwen C behavior cut | `4d40d97f1e575400237a6e5cef21d7f74404a38d` |
| Rust live-family freeze | `eb4ba7790da11f037cacc199a2d48d769956347b` |
| Rust host implementation freeze | `d126e56877390e9522dde333a34f0d582c3e246c` |
| Recommended split ref | annotated tag `ds4-dfm-rs-genesis`, created on the commit that first contains this green document |
| Destination scaffold | `b01d1fa4172a5c957fe1232774629a192493efe4` |
| Destination | `https://github.com/Baekpica/ds4-dfm-rs`, public non-fork, default branch `main` |
| License SHA-256 | `c4c2d4cef8d91b06bf07911b49edd662c3fbcad027acb135c816ec0281d17247` |

The baseline and Qwen cut are ancestors of the implementation freeze. A commit
cannot contain its own SHA, so the literal genesis SHA is resolved by
`git rev-parse ds4-dfm-rs-genesis^{}` after this document is committed. The
annotated tag and the target repository lineage document are the immutable
record of that SHA.

## Gate result

```text
Baseline:                v0.6.5-dfm
Rust host parity:        PASS
CUDA backend:            unchanged / native
Model families:          PASS
API surfaces:            PASS
KV:                      PASS
Performance:             PASS
Known Rust regressions:  NONE
C-shared gate gaps:      E-2, E-3, E-6
Long soak:               Qwen Q5+Sidecar only, PASS
```

The current 60-cell host matrix is PASS 57 + PASS* 3, FAIL 0, BLOCKED 0.
Every PASS* cell was re-run against a C control; no C-pass/Rust-fail cell was
accepted or removed.

| Group | Result |
|---|---|
| G0 host, parity, shared kernels | 13 PASS at `d126e56` |
| G1 DeepSeek KV, API, proof | 23 PASS; no repeated long soak |
| G2 Motif family, bank shutdown, barrier, ABBA | 9 PASS; E-4 and E-5 fixed and re-run |
| G3 Solar | 8 PASS + 1 PASS*(E-3) |
| G4 K-EXAONE | 2 PASS + 1 PASS*(E-2) |
| G5 dots3-note | 2 PASS + 1 PASS*(E-6) |

E-2 and E-6 were reproduced by exact `v0.6.5-dfm` sm_121a controls. E-3 was
reproduced by the frozen C server. E-1 remains an older C-shared Motif stats
backlog entry, not a Rust-only failure or a current matrix exception. The full
ledger is [ENGINE_GAPS.md](ENGINE_GAPS.md).

## Qwen release scope

Only the requested primary deployment was loaded: three Q5 main shards and
four shared BF16 SSD-PLE sidecars. Q6, original safetensors, and resident BF16
GGUF validation were intentionally out of scope.

```text
b8c8633f475351008d68981f4b37b409669b3dabd1620d6ead12584e648fe523  Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-00001-of-00003.gguf
14a3514bbf080266d501b07aa9da73ee1e0bfcf5ffd6bf0669034ed4b3fd35c9  Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-00002-of-00003.gguf
92dfbce1fcc6f722ac0ef7ec100bda02fbeb7b1b5cfecd1b57e51e18f82df111  Qwen3.8-Flash-Next-MQ-Q5-SSD-PLE-BF16-00003-of-00003.gguf
c13cfa3065b2343de314d31c0061fdb4471505e385d98549be7772d60ae684db  ple/ple-bf16-00001-of-00004.bin
9e957bfeafe75e779c209b1447c7d728cf6acb6e3f809ba0a89df30bba675c3c  ple/ple-bf16-00002-of-00004.bin
4aac201fd0dc517a14608f5721b1d05933c2d4deb092ddf215492ecdb7c7a383  ple/ple-bf16-00003-of-00004.bin
1e55bf840e6f37d76df4ec4274e13bf04e62ba823d989966b1a7c251cedff3b3  ple/ple-bf16-00004-of-00004.bin
```

Qwen passed text and image serving, three image API surfaces, all four text
surfaces, image-aware live/disk KV, C↔Rust four-way restore, width-2 barriers,
forced static routing, MTP off/on target parity, configured 262K, exact 262K,
196K production smoke, and C→Rust→Rust→C ABBA. The exact 262K gate produced
248,320 finite logits, argmax 198 in both hosts, zero packed-f32 mismatches,
and 99.64% Rust/C prefill. The Qwen-only soak ran 7,202.3 seconds with
3,610/3,610 successful requests, 158 barriers, 79 image requests, zero
request/census/governor failures, and zero process swap.

Qwen measurement binaries:

```text
b13a093b9232f61185a4e47fdbb5b3ca3babed0cb0a2d8f8eeae074da2b1aa2a  C server
2bdbbc044216705bb1077feba279b48efa9028cba42207cd03ef4e7d1416635f  soak/live Rust server
bd021db126e7add9fd700d3a720c4e339915cd37279dac1cce356b826ca8e385  focused Rust server
e7d344dcebf008530436bf7b4a07f8a9eef175a075a72c6b4b59c721dbd9212d  exact-262K C bench
62844f312b521cf3503bc719690b1135d9392e5e8a0dec15149d3812d9a1f431  exact-262K Rust bench
```

## Genesis build identity

The final host-only compatibility fix makes Rust honor
`DS4_SERVER_CLIENT_SNDBUF` through a safe `ds4-sys` socket adapter. Its real
socket test, full server C-parity suite, workspace tests/checks, clippy, and all
13 G0 cells pass at `d126e56`.

```text
8d4c014adce67db8257f29044db9e047635a10a3080720e636ffd6b192d1370e  ds4
68c812b21f299f675f10c31d94b5084d0c3683bc032049fcb11b2dc4c3d6b615  ds4-server
1bfbca83d159eb867480fa1e66fd0167f30f003dce9dc08b9be1f0f6710e14a5  ds4-bench
5e3c2177d35dd58d2316fd20f0ca0905484a20fac829b759e1d3bf91158ba51c  ds4-agent
106cf79a1e9dfea6b52c927aa1d9b69c31f5a03362d8c32d00432ffc55872730  ds4-c
e6faa67aa6273c0ec0e14d93d53099b6f610eacb376b0ee73ac41aa2f1696530  ds4-server-c
47998eec17da65516ce5ba1a9f4c880fa1edc2c5aa7c3ac312ef755ae5cf19c4  ds4-bench-c
127d8a26b9cdc018ad9ed46a17e44ae5f0805ca942f240277813ec7f1dd3d635  ds4-agent-c
48ff2698a72f403d8a4189fab6da9e7d188897d7c306bf6cb92bfae55eab532d  ds4-eval
8a491b7ed1038ab5a11f0f4e45923d325cde8bce3e199978dcfd3c51a89d6b00  ds4_weight_server
```

Each default Rust binary and its `*-c` oracle have distinct hashes and inodes.
The CUDA build config SHA-256 is
`2c15fe72b58438bfc80cc4e1f0b90596a1443847e126971b6ee33d1ae506c0ec`
(`CUDA_ARCH=sm_121`, sm_121a SASS path).

## Architecture audit

- Rust owns HTTP/API, admission and routing, scheduler orchestration,
  model/session handles and lifecycle policy, KV metadata/persistence policy,
  memory policy, agent/web host behavior, and distributed orchestration.
- Native C/CUDA retains VMM, CUDA Graph, CUDA/MMQ/fused attention/MoE/vision
  kernels, weight upload, opaque GPU session execution, and layer evaluation.
- The bridge exposes 68 functions through `native/bridge/ds4_bridge.h`; CUDA
  handles, device pointers, and raw native structs do not enter safe Rust.
- `rg -n 'unsafe \{' crates/` reports 97 sites: 61 reviewed core native/mmap
  adapters, 27 localized CLI POSIX/linenoise adapters, 7 `ds4-sys` sites, and
  2 test-only raw-FD fixtures. Production server/KV/dist/web crates have zero.
- C/Rust KVC payloads, EXT_TOOL_MAP, KTM trailer, recurrent payload, image
  replay key, and explicit distributed codecs remain wire-compatible.

## Evidence index

| Evidence | Workspace path |
|---|---|
| Classified full-family ledger | `scratch/rust-host-live/v065-full-restamp-20260831-184200/gates-results.txt` |
| Final 13-cell host audit | `scratch/rust-host-live/v065-full-restamp-20260831-184200/final-audit-d126e56/` |
| Qwen evidence root | `scratch/rust-host-live/qwen-v065-restamp-20260831/` |
| Qwen hard gate | `scratch/rust-host-live/qwen-v065-restamp-20260831/hard-gate-20260831-154010/` |
| Qwen exact 262K | `scratch/rust-host-live/qwen-v065-restamp-20260831/direct-ab-262144-20260831-181624/` |
| Qwen focused MTP/static | `scratch/rust-host-live/qwen-v065-restamp-20260831/focused-20260831-183929/` |

The host was NVIDIA GB10, driver 610.43.02, CUDA 13.3.73, Linux
6.17.0-1031-nvidia, rustc/cargo 1.98.0. Production models were loaded one at a
time. Each C/Rust ABBA cell used fresh owner/worker processes; teardown checked
PIDs and CUDA compute before `/usr/local/bin/clear_cache`. `nvtop` and `htop`
monitors remained active.

## Guarded split procedure

1. Create and push annotated tag `pre-genesis-scaffold-b01d1fa` on the target
   scaffold.
2. Re-read remote `main`; require exact
   `b01d1fa4172a5c957fe1232774629a192493efe4`.
3. Create annotated source tag `ds4-dfm-rs-genesis` on this green document's
   commit.
4. Seed target `main` with
   `--force-with-lease=refs/heads/main:b01d1fa4172a5c957fe1232774629a192493efe4`.
5. Push the `v0.6.5-dfm` provenance tag and `ds4-dfm-rs-genesis` tag.
6. Re-run the same build/parity suite and a Qwen Q5+Sidecar release smoke in
   the target clone before adding repository identity and `v0.1.0-rc.1` only.

No filter, squash, rebase, golden refresh, or automatic merge back to `dfm` is
permitted. The scaffold archival tag makes the guarded branch replacement
recoverable.

## Decision

The pre-split gate is GREEN. `Baekpica/ds4-dfm-rs` may be seeded using the
guarded procedure above.
