# Ternary Bonsai 2 27B (Prism Bonsai 2 27B)

[Documentation index](README.md) | [Repository README](../README.md)

Prism Ternary Bonsai 2 27B is the model the `qwen35` family was ported for: a
dense 26.90 B trunk of 64 blocks, 48 gated delta-net layers and 16
gated-attention layers, a dense SwiGLU FFN, no MoE and no MTP block. Every
matmul weight is PQ2_0 (ternary, 2.125 bits per weight) and the Prism exporter
stores it already Hadamard-folded, so the engine rotates the activation instead
of the weight. The norms are F32 and the two ssm gates BF16, as the exporter
writes them.

The artifact used on this host:

    /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
    6.71 GiB, 851 tensors (402 pq2_0, 353 f32, 96 bf16)
    general.architecture = qwen35

`download_model.sh` in this tree has no Bonsai target (it lists only the
DeepSeek V4 downloads); the GGUF above is the one already present on this box.
Its `bonsai-pq2` target lives in the sibling tree only.

## What this tree can do today

One binary, `./ds4-c`, serves both backends. What it can do depends on how it
was built:

| Build | Command | Can run |
| --- | --- | --- |
| CUDA-linked | `make ds4-c CUDA_ARCH=sm_89` | `--cuda` (device graph) and `--cpu` (CPU reference) |
| CPU-only | `make cpu` | `--cpu` only; not CUDA-linked, so `--cuda` is refused by the script |

The CUDA build is the one this recipe uses; `run-bonsai.sh` never rebuilds
anything, it only reports what is present.

The diagnostic generator `--first-token-test` is the reproducible oracle path:
greedy, with `-p "<prompt>"`.
`DS4_QWEN35_STEPS=<n>` sets the greedy step count (default 16) and
`DS4_QWEN35_TOKENS=<comma ids>` replaces the prompt with raw token ids, which
is how the reproducible parity gate is run. `DS4_QWEN35_LOGITS=<file>` dumps
the prompt pass's n x 248320 f32 logits. `DS4_QWEN35_SESSION=1` drives that
same prompt and greedy loop through a real `ds4_session` (create, sync, eval)
instead of a state local to the diagnostic, on either backend.

## The session path

`ds4_session_*` is wired for this family on both backends. The session owns the
whole trunk state: the CUDA graph on `--cuda`, the CPU reference on `--cpu`.
Both support create, sync with prefix reuse, eval, rewind-by-replay, invalidate
and the context bound; the table below says what each backend does differently.

| | CUDA session | CPU session (reference) |
| --- | --- | --- |
| state | one device graph, fp16 k/v + recurrent state | float trunk state in host memory |
| prefill | `DS4_QWEN35_PREFILL_CHUNK` rows per forward (512 default, 1024 max); a chunk the device cannot hold is halved until the graph opens | one token per forward, as the reference always runs |
| speed | ~26 ms per token | ~3 s per forward |

Refused by name rather than pretended: batching (the session decodes one row per
eval), MTP and DSpark drafting, SSD/disk KV, tensor parallelism, distributed
ranks, and KV snapshots (`ds4_session_payload_bytes` returns 0 and both payload
paths refuse). The attention row kernel is the row-exact one for every chunk
size: this tree does not select the token-tile kernel from the graph.

`run-bonsai.sh session` drives both backends through the session and diffs each
against the CPU reference. The CLI's plain generation path (`-p`, no
`--first-token-test`) routes this family through the session as well.

## Environment requirement: copy the model to the device

Every CUDA run on this host needs:

    DS4_CUDA_COPY_MODEL=1

The 6.71 GiB mmap cannot be pinned here (`RLIMIT_MEMLOCK` is 8192 KiB, and that
is also the hard limit, so it cannot be raised). Without the variable the
backend falls back to lazy per-range materialisation and dies part-way through
the trunk with `Bonsai matmul failed for blk.<n>.<tensor>`; the tensor that
fails depends on allocation order, the symptom does not. With the variable the
log reads `CUDA copying 6.71 GiB model to device memory` and the copy costs
about 0.7 s. `run-bonsai.sh` sets it for every `--cuda` run.

## Running

```sh
make ds4-c CUDA_ARCH=sm_89

./run-bonsai.sh status              # what is built, which artifact, what can run
./run-bonsai.sh                     # generate from the default prompt (CUDA)
./run-bonsai.sh "a prompt"          # same, your prompt
./run-bonsai.sh cuda ["prompt"]     # the CUDA graph explicitly
./run-bonsai.sh cpu ["prompt"]      # the CPU reference (the oracle)
./run-bonsai.sh compare ["prompt"]  # both backends, diffed token for token
./run-bonsai.sh ids                 # the explicit five-id parity gate
./run-bonsai.sh session ["prompt"]  # both backends through the session, diffed
./run-bonsai.sh bench [tokens]      # decode rate with /usr/bin/time
./run-bonsai.sh help
```

Env overrides: `DS4_BONSAI_MODEL`, `DS4_BONSAI_BIN`, `DS4_BONSAI_BACKEND`
(`cuda`|`cpu`), `DS4_BONSAI_STEPS`, `DS4_BONSAI_WAIT`, `DS4_BONSAI_LOG`.

The engine holds a single global lock (`/tmp/ds4.lock`), so only one
`ds4`/`ds4-c` process runs at a time; a second refuses to start by design. The
script waits for the slot (up to `DS4_BONSAI_WAIT` seconds, default 300)
instead of failing, and retries if the engine still refuses.

### status

```
$ ./run-bonsai.sh status
binary:  /data/ds4-dfm-rs/ds4-c
         present, built 2026-09-30 09:30:17
         CUDA-linked: runs --cuda and --cpu (both backends)
model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
         present, 6.71 GiB
gpu:     NVIDIA GeForce RTX 4070 SUPER, 180 MiB, 12282 MiB
         no compute process
slot:    busy (another ds4/ds4-c holds /tmp/ds4.lock); runs will wait
entry:   --first-token-test (greedy diagnostic), and the session path
         DS4_QWEN35_STEPS=<n>, DS4_QWEN35_TOKENS=<comma ids>,
         DS4_QWEN35_SESSION=1 (drive the same run through a session),
         DS4_QWEN35_PREFILL_CHUNK=<n> (rows per CUDA prefill forward)
supported: generate, cuda, cpu, compare, ids, session, bench, status, help
refused:   server (no serving path in this tree)
runbooks:  make bonsai-cuda-check, make bonsai-cuda-parity,
           make test-qwen35-cuda, make test-qwen35-session,
           make test-qwen35-session-multichunk, make test-qwen35-rows,
           make pq2-0-test, make bonsai-fold-selftest, make bonsai-ref-check
```

### ids (the reproducible parity gate)

`ids` is byte for byte what `make bonsai-cuda-parity` runs: the same five ids
through both backends at the same step count, diffed.

```
$ ./run-bonsai.sh ids
model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
tokens:  760,6511,314,9338,369 (explicit ids; the prompt is ignored)
steps:   8

--- CUDA graph ---
waiting for a free device slot (another ds4 process is running)...
backend:      cuda
wall:         1.61s for the prompt plus 8 greedy tokens (includes the model load)
decode rate:  4.97 tokens/s (steps / wall; steady state is higher)
peak rss:     7616 MiB
continuation:
   Paris.
The capital of Germany is

--- CPU reference (the oracle) ---
backend:      cpu
wall:         42.12s for the prompt plus 8 greedy tokens (includes the model load)
decode rate:  0.19 tokens/s (steps / wall; steady state is higher)
peak rss:     6911 MiB
continuation:
   Paris.
The capital of Germany is

--- token-for-token diff ---
IDENTICAL: both backends print the same 8 ids for the same
           explicit prompt; this is the reproducible parity gate
```

### compare (both backends on one prompt)

```
$ ./run-bonsai.sh compare
model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
prompt:  The capital of France is
steps:   16, generated by both backends

--- CUDA graph ---
backend:      cuda
wall:         2.39s for the prompt plus 16 greedy tokens (includes the model load)
decode rate:  6.69 tokens/s (steps / wall; steady state is higher)
peak rss:     7616 MiB
continuation:
  The user is asking a simple factual question: "The capital of France is"

--- CPU reference (the oracle) ---
backend:      cpu
wall:         139.25s for the prompt plus 16 greedy tokens (includes the model load)
decode rate:  0.11 tokens/s (steps / wall; steady state is higher)
peak rss:     6916 MiB
continuation:
  The user is asking a simple factual question: "The capital of France is"

--- token-for-token diff ---
IDENTICAL: all 16 generated token ids agree, so the CUDA graph
           reproduces the CPU reference on this prompt
           (per-backend token lines kept at misc/scratch/bonsai-{cpu,cuda}.tokens)
```

The `-p` prompt is chat-templated by the CLI (25 tokens here), which is why
its continuation is not the bare "Paris" stream `ids` prints. The CPU side is
about 3 s per forward, so a 16-step compare takes just over two minutes.

### bench

```
$ ./run-bonsai.sh bench 8
tokens:  8 greedy, one forward each, after the prompt
timed:   /usr/bin/time

backend:      cuda
wall:         2.15s for the prompt plus 8 greedy tokens (includes the model load)
decode rate:  3.72 tokens/s (steps / wall; steady state is higher)
peak rss:     7616 MiB
continuation:
  The user is asking a simple factual question
```

The rate here divides steps by total wall, which includes the one-off 0.7 s
model copy and the prompt pass; it is a floor, not the steady-state rate. A
64-step run on the same box:

```
$ ./run-bonsai.sh bench 64
wall:         3.67s for the prompt plus 64 greedy tokens (includes the model load)
decode rate:  17.44 tokens/s (steps / wall; steady state is higher)
peak rss:     7616 MiB
```

## Validation

Runbooks in this tree (all read `DS4_BONSAI_MODEL`):

| Target | Covers |
| --- | --- |
| `make bonsai-cuda-check` | the greedy stream on the CUDA graph alone |
| `make bonsai-cuda-parity` | CPU reference vs CUDA graph streams, diffed (what `ids` runs) |
| `make test-qwen35-cuda` | the CUDA kernels against the in-process CPU reference, no model file needed |
| `make test-qwen35-session` | the session path on both backends (plain, prefix reuse, rewind, invalidate, context bound), diffed against the in-process CPU reference |
| `make test-qwen35-session-multichunk` | the same scenarios with `DS4_QWEN35_PREFILL_CHUNK=2`, so the prefill crosses many chunk boundaries |
| `make test-qwen35-rows` | every tensor of the artifact through this tree's row reader |
| `make pq2-0-test` | the PQ2_0 block format against the Prism reference dequantizer |
| `make bonsai-fold-selftest` | the fold round-trips and the gated-delta-net permutation |
| `make bonsai-ref-check` | greedy decode on the CPU reference |

Live check re-run for this recipe (commit `7818a7a`):

```
$ make bonsai-cuda-check
ds4: prompt 25 token(s); next-token top-5: 760(21.9915)The 90700(16.0678)Thinking ...
ds4: diagnostic run completed on the native cuda path.
token 25: 760 The
token 26: 1156  user
token 27: 369  is
token 28: 9859  asking
token 29: 264  a
token 30: 4145  simple
token 31: 57879  factual
token 32: 3296  question
token 33: 25 :
token 34: 328  "
token 35: 760 The
token 36: 6511  capital
```

Recorded evidence for the CUDA unit (commits `8c887e4`, `349e93b`, `7818a7a`,
RTX 4070 SUPER, sm_89, nvcc 13.3):

- `make test-qwen35-cuda CUDA_ARCH=sm_89` prints `PQ2_0 CUDA parity: PASS`,
  covering the ported kernel set, the gdn output gate on both selectors and
  eight attention cases.
- `make bonsai-cuda-parity` prints `bonsai cuda parity: PASS`: the CPU
  reference and the CUDA graph print the same eight ids for
  `DS4_QWEN35_TOKENS=760,6511,314,9338,369` at 8 steps (11751 Paris, 13 .,
  198, 760 The, 6511 capital, 314 of, 9564 Germany, 369 is), and the same for
  16 steps.
- logits on those five prompt tokens: CUDA against the CPU reference max|d|
  0.08093, 0.264% rms relative, argmax 5/5; this tree's CUDA against the
  sibling tree's CUDA on the same ids is max|d| 0.00000 (bit-identical).
- decode rate about 26 ms per token in steady state: 5 prompt plus 64 greedy
  forwards in 3.09 s wall including the 0.7 s model copy, 7.8 GB RSS.

## Limitations

- **No serving surface.** The engine now accepts a session on either backend,
  and the family's serving caps declare its minimal plan (serial lane, no
  banks, batching, snapshots or MTP, partial prefix reuse, any host) with no
  qualified limits, but serving has not been exercised for this family:
  the model identity and aliases (`prism-bonsai-2-27b`, `-chat`/`-reasoner`),
  the qualified context and the disk-KV notice belong to the serving unit, and
  `run-bonsai.sh server` refuses by name until they land.  Do not read the caps
  as a qualified service.
- **No batching, SSD/disk KV, MTP or session snapshots.** Each is refused by
  name rather than pretended: the session decodes one row per eval, the payload
  paths refuse, and the engine gate rejects the MTP/DSpark sidecars, tensor
  parallelism and distributed ranks.
- **The CPU reference is the slow path.** About 3 s per forward against ~26 ms
  per token on the card, which is why the CPU session runs the reference trunk
  and `./run-bonsai.sh session` takes minutes.
- **Greedy generation from a chat-templated prompt is a near-tie coin flip.**
  The CPU and CUDA logits differ by the 0.08093 above, and on the CLI's
  chat-templated prompt a continuation can flip between the backends (about
  0.1 on a 22-magnitude top logit). The reproducible gate is the explicit-token
  stream (`ids` / `make bonsai-cuda-parity`); a long chat continuation is not
  evidence. See
  [the CUDA graph report](releases/bonsai-cuda-graph-2026-09-30.md), section 4.

## Report

[Recorded evidence for the CUDA unit](releases/bonsai-cuda-graph-2026-09-30.md)
and [the CPU reference unit](releases/bonsai-cpu-reference-2026-09-30.md), and
[the session unit](releases/bonsai-session-2026-09-30.md).
