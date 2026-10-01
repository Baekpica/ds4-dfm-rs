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
greedy, with `-p "<prompt>"`. The Rust host is the default binary
(`make ds4` produces `./ds4`; the C hosts keep the `-c` suffix and this recipe's
runbooks use them), and `make test-qwen35-rust-host` pins the two hosts to the
same ids.
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
paths refuse). Attention is chosen per batch, not per model: a chunk of 32 rows
or more takes the token-tile MMA kernel, and decode (or anything below that)
takes the row-exact kernel, which cuts a row's key range into up to 64 split
ranges and reduces them in attn_merge. Without that split this family's decode
collapsed as the context grew — 4.32 tokens/s at a 15k context, against 35.65
in /data/ds4 — and the split is what closed that gap.

`run-bonsai.sh session` drives both backends through the session and diffs each
against the CPU reference. The CLI's plain generation path (`-p`, no
`--first-token-test`) routes this family through the session as well.

## The server path

`ds4-server` (the Rust host; the C oracle `ds4-server-c` cannot serve this
family) serves it over the OpenAI-compatible surface, on either backend. The
family reports the ABI id 14, which the server maps to the Qwen ChatML syntax
this artifact's own `tokenizer.chat_template` declares, so rendering, the
thinking split and the tool XML are the Qwen ones rather than the DeepSeek
default an unmapped id would fall back to.

| | CUDA server | CPU server (reference) |
| --- | --- | --- |
| state | the CUDA graph session, same as `session` | the CPU reference session |
| lanes | serial only (`--max-seqs`, batching, MTP refused by name) | serial only |
| speed | ~1.4 s for a 36-token answer with 45 prompt tokens | ~3 s per forward |

```
$ ./run-bonsai.sh server
model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
prompt:  The capital of France is
ctx:     45056, memory floor 1G, port 8899
backend: cuda
binary:  /data/ds4-dfm-rs/ds4-server

server:  up (pid 1240413)
         listening on 127.0.0.1:8899 model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open ...
id:      Ternary-Bonsai-2-27B-PQ2_0

wall:    1.40s for the request
finish:  stop
answer:   Paris.
reasoning: The user is asking a simple factual question: "The capital of France is". I just need to complete the sentence. The capital of France is Paris.
usage:    {'prompt_tokens': 45, 'completion_tokens': 36, 'total_tokens': 81, ...}
```

The advertised id is the GGUF stem, as it is for every family in this tree; the
sibling C server's names (`prism-bonsai-2-27b`, `-chat`, `-no-think`,
`-nothink`, `-reasoner`, `prism/bonsai-2-27b`) are accepted aliases on
`/v1/models/<id>`, and the server does not validate the model field of a chat
request, so a client configured with the alias is served.

`server` is a smoke test: it starts the server, sends one request and stops it.
To keep one up for a client (open-grok, a script, curl), use `serve`:

```
$ ./run-bonsai.sh serve start        # -> serve with no argument does the same
model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
backend: cuda, ctx 45056, memory floor 1G, port 8899
server:  up (pid 1355296)
         listening on 127.0.0.1:8899 model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open ...
base_url: http://127.0.0.1:8899/v1
id:      Ternary-Bonsai-2-27B-PQ2_0 (aliases: prism-bonsai-2-27b*)
log:     /data/ds4-dfm-rs/misc/scratch/bonsai-serve.log
note:    one ds4 model at a time; this server holds the slot until stopped

$ ./run-bonsai.sh serve status
server:  running (pid 1355296)
base_url: http://127.0.0.1:8899/v1
serving: Ternary-Bonsai-2-27B-PQ2_0 ctx 45056
vram:    7474 MiB, 12282 MiB

$ ./run-bonsai.sh serve stop
server:  stopped (pid 1355296)
```

`serve logs [n]` tails the capture. The pid file and the log live in
`misc/scratch`, and `stop` only ever kills the pid that file records, so a
foreign `ds4-server` is never touched. A second `serve start` reports the
running server instead of starting another one, and if any other ds4 model
holds the single slot the start refuses and names the process.

An open-grok client block for it (port 8899, the id it advertises, and a
context the card can host):

```toml
[model.bonsai-local]
model = "prism-bonsai-2-27b"
base_url = "http://127.0.0.1:8899/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 45056
max_completion_tokens = 4096
supports_images = false
```

### The memory quote

The plan prices the bank from the family's own geometry: the 16
gated-attention layers carry the per-token key/value rows, the 48 gated
delta-net layers carry a fixed recurrent matrix and convolution window, and the
CUDA graph adds its chunk-sized transient buffers. The generic estimate charges
all 64 blocks as attention rows (four times the real KV) and refuses every
usable context on a 12 GiB card even with the memory floor at zero.

Measured with `--mem-floor-gb 1` on this host (`--check-config` reports the
plan before the model opens, so the last row also reports a real start):

| ctx | bank (CUDA) | plan total | free device memory at plan time | verdict |
| --- | --- | --- | --- | --- |
| 32768 | 2.39 GiB | 10.11 GiB | 11.32 GiB | opens |
| 40960 | 2.89 GiB | 10.61 GiB | 11.29 GiB | opens |
| 45056 | 3.14 GiB | 10.86 GiB | 11.29 GiB | opens; 7.30 GiB fresh, 10.57 GiB after requests |
| 49152 | 3.39 GiB | 11.11 GiB | 11.28 GiB | refused on a real start (opens on paper only) |
| 65536 | 4.39 GiB | 12.11 GiB | 11.28 GiB | refused |

`available` is the free device memory the quote reads at plan time, so the
verdict moves with whatever else is on the card; 45056 is the largest context
with a workable margin here and is what `DS4_BONSAI_CTX` defaults to. Device
use is not flat: a freshly started server holds 7.30 GiB (weights plus
transients), and the first requests bring the ctx-45056 bank (3.14 GiB) in, for
10.57 GiB steady state, which is the number to plan against and the one the
plan's 10.86 GiB total already quotes. The 4 GiB
default memory floor refuses all of these. The CPU reference allocates a float
key/value row for every block, so its own limit is roughly half these contexts
on a 32 GiB host; `DS4_BONSAI_CTX` and `DS4_BONSAI_MEM_FLOOR` (with
`--kv-disk-dir` refused by name) set the trade.

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
./run-bonsai.sh server ["prompt"]   # one chat request through the Rust server
./run-bonsai.sh serve [start|stop|status|logs]  # keep a server up for a client
./run-bonsai.sh bench [tokens]      # decode rate with /usr/bin/time
./run-bonsai.sh help
```

Env overrides: `DS4_BONSAI_MODEL`, `DS4_BONSAI_BIN`, `DS4_BONSAI_BACKEND`
(`cuda`|`cpu`), `DS4_BONSAI_STEPS`, `DS4_BONSAI_WAIT`, `DS4_BONSAI_LOG`, and
for `server` / `serve` `DS4_BONSAI_SERVER_BIN`, `DS4_BONSAI_CTX` (default
45056), `DS4_BONSAI_MEM_FLOOR` (default 1, GiB), `DS4_BONSAI_SERVER_PORT` and
`DS4_BONSAI_SERVER_TOKENS` (default 64: the reasoning block alone runs about 33
tokens before any content).

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
entry:   --first-token-test (greedy diagnostic), the session path, and
         the Rust server (./run-bonsai.sh server)
         DS4_QWEN35_STEPS=<n>, DS4_QWEN35_TOKENS=<comma ids>,
         DS4_QWEN35_SESSION=1 (drive the same run through a session),
         DS4_QWEN35_PREFILL_CHUNK=<n> (rows per CUDA prefill forward)
server:  /data/ds4-dfm-rs/ds4-server
         present; serves this family (id from the GGUF stem, aliases
         prism-bonsai-2-27b*) with DS4_BONSAI_CTX=45056 and
         DS4_BONSAI_MEM_FLOOR=1G on port 8899
supported: generate, cuda, cpu, compare, ids, session, server, serve, bench, status, help
refused:   batching, MTP/DSpark, SSD/disk KV, session snapshots, distributed
           ranks (each refused by name; --kv-disk-dir stops the server for
           this family rather than staying silently unused)
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
| `make test-qwen35-rust-host` | the Rust host (`./ds4`, the default binary) against the C host's pinned ids on both backends; this is what catches a host-shaped load that misses the family's GGUF-borne rope and fold configuration |
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

- **Serving is exercised but not qualified.** The server answers on both
  backends and refuses by name what it cannot do, but the family's serving caps
  carry no qualified context, bank or prompt limits, and only the serial lane
  is wired: batching, MTP/DSpark, SSD/disk KV, tensor parallelism and session
  snapshots are refused (`--kv-disk-dir` stops the server rather than staying
  silently unused) and the plan reports `reuse=exact` with a
  `partial_unqualified` warning. Do not read the caps as a qualified service.
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
and [the CPU reference unit](releases/bonsai-cpu-reference-2026-09-30.md),
[the session unit](releases/bonsai-session-2026-09-30.md) and
[the serving unit](releases/bonsai-serving-2026-09-30.md).
