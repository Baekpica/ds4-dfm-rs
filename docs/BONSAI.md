# Ternary Bonsai 2 27B (Prism Bonsai 2 27B)

[Documentation index](README.md) | [Repository README](../README.md)

Prism Ternary Bonsai 2 27B is the model the `qwen35` family was ported for: a
dense 26.90 B trunk of 64 blocks, 48 gated delta-net layers and 16
gated-attention layers, a dense SwiGLU FFN, no MoE. Every matmul weight is
PQ2_0 (ternary, 2.125 bits per weight) and the Prism exporter stores it
already Hadamard-folded, so the engine rotates the activation instead of the
weight. The norms are F32 and the two ssm gates BF16, as the exporter
writes them.

The artifact these numbers were measured with:

    Ternary-Bonsai-2-27B-PQ2_0.gguf
    6.71 GiB, 851 tensors (402 pq2_0, 353 f32, 96 bf16)
    general.architecture = qwen35

`download_model.sh` in this tree has no Bonsai target (it lists only the
DeepSeek V4 downloads), so the GGUF is placed in the model directory by hand
and named through `DS4_BONSAI_MODEL` (no default; the runbooks require it).

## What this tree can do today

Two hosts run this family. `./ds4-c` is the C host: the `--first-token-test`
greedy diagnostic, the `--cpu` reference and the per-kernel tests live there.
`./ds4` is the Rust host and the default binary: `-p` for a prompt,
`--token-ids` with `--predict` for raw ids, `--serving-check` for the
capability checklist. Both run the same C inference core.

| Build | Command | Can run |
| --- | --- | --- |
| CUDA-linked | `make ds4 ds4-c CUDA_ARCH=sm_89` (`make cuda-spark` on GB10) | `--cuda` (device graph) and `--cpu` (CPU reference) |
| CPU-only | `make cpu` | `--cpu` only; `--cuda` is refused |

The CUDA build is the one this page's numbers use. Nothing here rebuilds a
binary: the commands below run what `make` produced.

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

Refused by name rather than pretended: batching (the session verifies at most
the two rows of one draft block), DSpark drafting, SSD/disk KV, tensor
parallelism, distributed ranks, and KV snapshots (`ds4_session_payload_bytes`
returns 0 and both payload paths refuse). The family's own embedded MTP block
drafts on the serial lane (see "The draft loop"); the bank driver has no draft
round for it. Attention is chosen per batch, not per model: a chunk of 32 rows
or more takes the token-tile MMA kernel, and decode (or anything below that)
takes the row-exact kernel, which cuts a row's key range into up to 64 split
ranges and reduces them in attn_merge. Without that split this family's decode
collapsed as the context grew — 4.32 tokens/s at a 15k context, against 35.65
in the pre-port baseline — and the split is what closed that gap.

`DS4_QWEN35_SESSION=1` drives both backends through the session; run the same
prompt on each and diff the token lines:

```sh
DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_SESSION=1 DS4_QWEN35_STEPS=16 \
    ./ds4-c -m "$DS4_BONSAI_MODEL" --cuda --first-token-test -p "The capital of France is" \
    | grep '^token ' > misc/scratch/bonsai-cuda-session.tokens

DS4_QWEN35_SESSION=1 DS4_QWEN35_STEPS=16 \
    ./ds4-c -m "$DS4_BONSAI_MODEL" --cpu --first-token-test -p "The capital of France is" \
    | grep '^token ' > misc/scratch/bonsai-cpu-session.tokens

diff misc/scratch/bonsai-cpu-session.tokens misc/scratch/bonsai-cuda-session.tokens
```

The CLI's plain generation path (`-p`, no `--first-token-test`) routes this
family through the session as well.

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

```sh
DS4_CUDA_COPY_MODEL=1 ./ds4-server -m "$DS4_BONSAI_MODEL" \
    --backend cuda -c 45056 --mem-floor-gb 1 --host 127.0.0.1 --port 8899
```

The server reports the plan and the listener before it answers:

```
ds4-server-rs: request defaults temperature=1 (model) top_p=0.95 (model) top_k=0 min_p=0 max_tokens=393216 seed=per-request
family: qwen35
requested: reuse=auto max_seqs=auto mtp=auto ctx=45056 floor=1G disk=false
ds4-server-rs: listening on 127.0.0.1:8899 model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open host_vocab=yes
```

One request through it:

```sh
curl -s http://127.0.0.1:8899/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Ternary-Bonsai-2-27B-PQ2_0",
         "messages":[{"role":"user","content":"The capital of France is"}],
         "max_tokens":64,"temperature":0}'
```

The recorded answer on the CUDA server was one 36-token reply (`finish_reason`
stop) ending in `Paris.`, with `prompt_tokens` 45 and `completion_tokens` 36;
`/v1/models` answers the same id with `context_length` 45056.

The advertised id is the GGUF stem, as it is for every family in this tree; the
sibling C server's names (`prism-bonsai-2-27b`, `-chat`, `-no-think`,
`-nothink`, `-reasoner`, `prism/bonsai-2-27b`) are accepted aliases on
`/v1/models/<id>`, and the server does not validate the model field of a chat
request, so a client configured with the alias is served.

The server runs in the foreground and holds the single model slot until it
exits; stop it with the shell's own job control. It refuses by name what it
cannot do (see Limitations) instead of staying silently idle, and `--kv-disk-dir`
stops the start rather than being accepted and ignored.

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

Measured on this host through `ds4-server`'s admission plan
(`--mem-floor-gb 1`; `--check-config` prints the plan before the model opens,
so the last row also reports a real start):

| ctx | bank (CUDA) | plan total | free device memory at plan time | verdict |
| --- | --- | --- | --- | --- |
| 32768 | 2.39 GiB | 10.11 GiB | 11.32 GiB | opens |
| 40960 | 2.89 GiB | 10.61 GiB | 11.29 GiB | opens |
| 45056 | 3.14 GiB | 10.86 GiB | 11.29 GiB | opens; 7.30 GiB fresh, 10.57 GiB after requests |
| 49152 | 3.39 GiB | 11.11 GiB | 11.28 GiB | refused on a real start (opens on paper only) |
| 65536 | 4.39 GiB | 12.11 GiB | 11.28 GiB | refused |

`available` is the free device memory the quote reads at plan time, so the
verdict moves with whatever else is on the card; 45056 is the largest context
with a workable margin here and is the `-c` these commands pass. Device use is
not flat: a freshly started server holds 7.30 GiB (weights plus transients),
and the first requests bring the ctx-45056 bank (3.14 GiB) in, for
10.57 GiB steady state, which is the number to plan against and the one the
plan's 10.86 GiB total already quotes. The 4 GiB default memory floor refuses
all of these. The CPU reference allocates a float key/value row for every
block, so its own limit is roughly half these contexts on a 32 GiB host; `-c`
and `--mem-floor-gb` (with `--kv-disk-dir` refused by name) set the trade.

## Environment requirement: copy the model to the device

Every CUDA run on this host needs:

    DS4_CUDA_COPY_MODEL=1

The 6.71 GiB mmap cannot be pinned here (`RLIMIT_MEMLOCK` is 8192 KiB, and that
is also the hard limit, so it cannot be raised). Without the variable the
backend falls back to lazy per-range materialisation and dies part-way through
the trunk with `Bonsai matmul failed for blk.<n>.<tensor>`; the tensor that
fails depends on allocation order, the symptom does not. With the variable the
log reads `CUDA copying 6.71 GiB model to device memory` and the copy costs
about 0.7 s. Set it for every `--cuda` / `--backend cuda` run below.

## Running

```sh
make ds4 ds4-c CUDA_ARCH=sm_89     # make cuda-spark on GB10
export DS4_BONSAI_MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
export DS4_CUDA_COPY_MODEL=1       # every CUDA run on this host

# the greedy diagnostic, one token per graph call
./ds4-c -m "$DS4_BONSAI_MODEL" --cuda --first-token-test -p "The capital of France is"

# the same prompt on the CPU reference (the oracle)
./ds4-c -m "$DS4_BONSAI_MODEL" --cpu --first-token-test -p "The capital of France is"

# the Rust host over the same raw ids
./ds4 -m "$DS4_BONSAI_MODEL" --backend cuda \
    --token-ids 760,6511,314,9338,369 --predict 8 --temp 0

# both backends over the parity ids, diffed
make bonsai-cuda-parity

# one chat request through the server
./ds4-server -m "$DS4_BONSAI_MODEL" --backend cuda -c 45056 \
    --mem-floor-gb 1 --host 127.0.0.1 --port 8899
```

Switches that shape a run: `DS4_QWEN35_STEPS`, `DS4_QWEN35_TOKENS`,
`DS4_QWEN35_LOGITS`, `DS4_QWEN35_SESSION`, `DS4_QWEN35_PREFILL_CHUNK`,
`DS4_MTP_SPEC_DISABLE`, `DS4_MTP_SPEC_LOG`, `DS4_QWEN35_MTP_PROBE` and
`DS4_CUDA_COPY_MODEL` (all above).

The engine holds a single global lock (`/tmp/ds4.lock`), so only one
`ds4`/`ds4-c`/`ds4-server` process runs at a time; a second refuses to start by
design. Wait for the running one to finish, or stop the process you started.

### What is present

There is no status subcommand: check the pieces directly.

```sh
ldd ./ds4-c | grep libcudart          # CUDA-linked, so --cuda and --cpu both run
ls -l "$DS4_BONSAI_MODEL"             # 6.71 GiB (7,206,168,928 bytes)
nvidia-smi                            # free device memory against the plan above
fuser -v /tmp/ds4.lock 2>&1           # which process holds the single slot
```

What the binaries carry for this family: `./ds4-c` has `--first-token-test`
(greedy diagnostic), the session path and `--cpu`; `./ds4` has `-p`,
`--token-ids`/`--predict`, `--serving-check` and `--cuda`/`--cpu`/`--backend`;
`./ds4-server` serves the id from the GGUF stem with the aliases above.

Refused by name: batching, DSpark, SSD/disk KV, session snapshots and
distributed ranks (a real start stops rather than ignoring them; `--kv-disk-dir`
stops the server for this family).

Runbooks in this tree (all read `DS4_BONSAI_MODEL`): `make bonsai-cuda-check`,
`make bonsai-cuda-parity`, `make test-qwen35-cuda`,
`make test-qwen35-rust-host`, `make test-qwen35-session`,
`make test-qwen35-session-multichunk`, `make test-qwen35-rows`,
`make pq2-0-test`, `make bonsai-fold-selftest`, `make bonsai-ref-check`.

### The parity gate

`make bonsai-cuda-parity` is the reproducible gate: eight steps from the five
explicit ids on both backends, diffed, and it prints one line.

```
$ make bonsai-cuda-parity
bonsai cuda parity: PASS
```

Spelled out, that is `DS4_QWEN35_TOKENS=760,6511,314,9338,369
DS4_QWEN35_STEPS=8` through `./ds4-c --first-token-test` on `--cuda` and on
`--cpu`. The eight generated ids are 11751 Paris, 13 `.`, 198, 760 The,
6511 capital, 314 of, 9564 Germany, 369 is, so the stream continues
`Paris.` then `The capital of Germany is`.

On this host the CUDA run took 1.61 s wall for the prompt plus eight tokens
including the model load, with 7616 MiB peak RSS; the CPU reference took
42.12 s and 6911 MiB for the same stream. Those wall times are load-inclusive
floors, not the steady-state rate.

### Both backends on one prompt

```sh
./ds4-c -m "$DS4_BONSAI_MODEL" --cuda --first-token-test \
    -p "The capital of France is" | grep '^token ' > misc/scratch/bonsai-cuda.tokens
./ds4-c -m "$DS4_BONSAI_MODEL" --cpu --first-token-test \
    -p "The capital of France is" | grep '^token ' > misc/scratch/bonsai-cpu.tokens
diff misc/scratch/bonsai-cpu.tokens misc/scratch/bonsai-cuda.tokens
```

Recorded at `DS4_QWEN35_STEPS=16`: both printed

```
  The user is asking a simple factual question: "The capital of France is"
```

and the 16 generated token ids agreed. The `-p` prompt is chat-templated by the
CLI (25 tokens here), which is why its continuation is not the bare `Paris`
stream the parity ids print. The CPU side is about 3 s per forward, so a
16-step comparison takes just over two minutes.

### Decode rate

```sh
/usr/bin/time -f '%e s wall, %M KiB peak' \
    ./ds4-c -m "$DS4_BONSAI_MODEL" --cuda --first-token-test \
    -p "The capital of France is"
```

Recorded at `DS4_QWEN35_STEPS=64` with the current kernels, this host decodes
at about 44 tok/s plain (43.94 at a ~1k-token prompt, 44.75 on a 4-token one)
and 56-61 tok/s with the family's draft block armed; "The draft loop" below
holds the matched pairs. The rate this command recorded on 2026-09-30 predates
the family-owned PQ2_0 dense kernel and is superseded. `ds4-bench`
measures the same family from the Rust host and arms the block by default
(`DS4_MTP_SPEC_DISABLE=1` for the plain run, `--mtp-draft 2` for the wider
verify).

## The MTP artifact, reproduced byte for byte

`decent-jawfish/bonsai-2-27b-mtp` publishes `Bonsai-2-27B-PQ2_0-MTP.gguf`: this
artifact with the Qwen3.8-27B MTP block grafted in. It is 15 `blk.64.*`
tensors copied verbatim and unrotated from
`unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q2_K_XL.gguf`, with
`qwen35.block_count` set to 65 and `qwen35.nextn_predict_layers` set to 1. The
tensors are deliberately absent from `prism.hadamard.weight_names`: they are
stock Qwen weights in the original residual basis.

`crates/ds4-core/examples/bonsai_graft_mtp.rs` reproduces it. The published
`graft_mtp.py` is the reference; the tool keeps each tensor's byte count as the
gap to the next offset in its own file, so the source's inter-tensor padding is
carried over too, and it only range-fetches the 15 tensors (351,008,768 bytes),
not the 9.15 GiB source artifact:

    mkdir -p work

    # 1. the source header only (metadata and tensor table)
    curl -fsSL -r 0-$((48*1024*1024-1)) \
        https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q2_K_XL.gguf \
        -o work/qwen38_head.gguf

    # 2. a fetch script for the header plus the 15 tensor ranges
    cargo run -p ds4-core --release --example bonsai_graft_mtp -- \
        plan work/qwen38_head.gguf --size 9828981664 \
        --script https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q2_K_XL.gguf \
        > work/fetch_mtp.sh
    truncate -s 9828981664 work/qwen38_src.gguf
    SRC=work/qwen38_src.gguf bash work/fetch_mtp.sh

    # 3. the graft, then the digest
    cargo run -p ds4-core --release --example bonsai_graft_mtp -- \
        graft Ternary-Bonsai-2-27B-PQ2_0.gguf work/qwen38_src.gguf \
        Bonsai-2-27B-PQ2_0-MTP.gguf
    sha256sum Bonsai-2-27B-PQ2_0-MTP.gguf

Verified 2026-11-09 against the base artifact above: the output is
7,557,178,656 bytes and its sha256 is
`78df4279d40ebebdccfd2dae0e9d4847afee52e94f48f3542ae9437220dbd847`, the digest
the published repository records for its LFS object. The published model card
measures 41.9 tok/s without MTP against 60.5 tok/s with two drafts on a
24 GB RTX PRO 6000 slice, under llama.cpp at two drafts. This engine's own
single-draft loop is measured under "The draft loop" below.

### The block as a drafter

`blk.64` runs as a nextn module: `hnorm` over the trunk's hidden row and
`enorm` over the next token's embedding concatenate through `eh_proj`, one
full-attention layer with its own fp16 k/v and a SwiGLU feed-forward follow,
and the trunk's output head returns the draft. The checkpoint
(`Qwen/Qwen3.8-27B` `config.json`) settles the geometry the tensor widths
alone left open: 24 query heads, 4 KV heads, head_dim 256, `attn_output_gate`
true with a swish gate (so `attn_q` is 12288 = 2 x 24 x 256 and
`attn_output` takes 6144), partial rotary 0.25 (`n_rot` 64), and
`mtp_use_dedicated_embeddings` false.

Two properties matter for the numbers below:

- The block's weights are Q6_K, copied unrotated from a stock checkpoint, so
  its matmuls are plain while the trunk's PQ2_0 weights are Hadamard-folded.
  The residual stream stays in the original basis because the fold rotates
  each matmul's input (`W'(H a) = W a`), which is why the trunk's hidden state
  feeds the block directly.
- Its rows are filled as the trunk completes them: a finished chunk pairs each
  hidden row with the next token, and the chunk's last row waits for the
  sampled token.

Draft quality, measured greedily on the pinned prompt with
`DS4_QWEN35_MTP_PROBE=1` (a diagnostic that only reports): the draft's argmax
equals the trunk's next argmax 89/119 (0.748) and 42/59 (0.712), and the
12-step token stream is identical with the probe on and off.

## The draft loop

A speculative cycle is the block's proposal plus one two-row trunk pass:

```
 pos0                pos0+1
   |                     |
 [ first ]  [ draft ]        <- the verify block's two tokens
   |                     |
 trunk rows 0 and 1  -> logits row 0 (the target), row 1 (the bonus token)
   |
 keep = 1 + (draft == target)
```

- Draft: the block consumes its pending trunk row with the token just sampled
  and returns one proposal (the trunk's own output head over the block's row).
- Verify: one forward over both rows. Row 0 reproduces the plain decode's
  logits; row 1's logits are free once the draft is accepted, so an accepted
  cycle emits two tokens.
- Rollback: 48 trunk layers are gated delta-net with recurrent state. The
  `conv` and `gdn_scan` kernels already carry two snapshot slots, so a verify
  checkpoints the state and the conv history as they stand after row 0 and a
  rejected draft swaps those buffers into place — no replay of the committed
  row. The checkpoint costs 48 x (3.1 MB state + 0.12 MB history) = 155 MB and
  exists only for an artifact that carries the block. The k/v rows a rejected
  block wrote stay stale at positions that are never read: attention reads
  keys below the live position, and the next cycle overwrites them.
- The block advances only over accepted rows (`MTP keep`), so its k/v prefix
  never holds a rejected token.
- Sampling: a request with a temperature draws its targets from the same two
  verify rows with the host's sampler (sample-and-match), so drafting is not
  gated on temperature. `DS4_MTP_SPEC_DISABLE=1` restores plain decode, and
  `DS4_MTP_SPEC_LOG=1` prints one line per cycle.

Measured 2026-11-09, `ds4-c` (CUDA, sm_89, RTX 4070 SUPER, 12 GiB), greedy
`--temp 0`, against the same binary with `DS4_MTP_SPEC_DISABLE=1`:

| workload | plain | one draft | gain |
| --- | --- | --- | --- |
| 4-token prompt, 48 generated | 44.75 tok/s | 60.94 tok/s | +36% |
| 25-token prompt, 128 generated | 44.05 tok/s | 59.28 tok/s | +35% |
| ~1k-token prompt, 64 generated | 43.94 tok/s | 56.37 tok/s | +28% |

Every pair printed the identical completion, so the committed stream is the
plain decode's own at every context measured. The acceptance was 50 of 77
cycles (0.65) on the 25-token prompt; the earlier probe's 0.75 was measured on
the diagnostic prompt. Prefill costs what the fill costs: 1149.7 to 1110.2 t/s
at ~1k tokens (-3.4%) and nothing measurable on a short prompt, because a
chunk fills the block's k/v in one batched pass (the FFN and head are skipped
for rows no draft reads) instead of one weight-read per row.

**Two drafts do not pay.** `--mtp-draft 2` is implemented and deterministic
(59.30, 59.34, 59.41 tok/s over three runs against 59.13-59.28 for one draft)
but it is not faster, and its three-row verify changes a committed token on a
long run (the 128-token printing-press completion differs from plain decode
after about 40 tokens while staying fluent; a two-row block matches it
exactly). The reason is the acceptance it buys: first-draft acceptance 0.68 and
second-draft 0.57 (24 of 42) on that prompt, against 0.65 for one, so a cycle
yields 2.06 tokens for three verified rows instead of 1.65 for two. The second
draft chains through the block's own post-FFN hidden state, because the trunk
row for that position does not exist until the verify runs. One draft is the
default; the second checkpoint pair is allocated only when `--mtp-draft 2`
asks for it (155 MB more).

The Rust host reaches the same loop (59 cycles on a 96-token greedy run, the
same completion as plain decode, and 19 cycles on a temperature-0.8 run through
sample-and-match). The C host speculates on greedy requests only.

## Validation

Runbooks in this tree (all read `DS4_BONSAI_MODEL`):

| Target | Covers |
| --- | --- |
| `make bonsai-cuda-check` | the greedy stream on the CUDA graph alone |
| `make bonsai-cuda-parity` | CPU reference vs CUDA graph streams, diffed over the parity ids |
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
  is wired: batching, DSpark, SSD/disk KV, tensor parallelism and session
  snapshots are refused (`--kv-disk-dir` stops the server rather than staying
  silently unused) and the plan reports `reuse=exact` with a
  `partial_unqualified` warning. Do not read the caps as a qualified service.
- **No batching, SSD/disk KV or session snapshots.** Each is refused by name
  rather than pretended: the session verifies at most one draft block's two
  rows, the payload paths refuse, and the engine gate rejects the DSpark
  sidecar, tensor parallelism and distributed ranks.
- **The CPU reference is the slow path.** About 3 s per forward against ~26 ms
  per token on the card, which is why the CPU session runs the reference trunk
  and a 16-step CPU comparison takes minutes.
- **Greedy generation from a chat-templated prompt is a near-tie coin flip.**
  The CPU and CUDA logits differ by the 0.08093 above, and on the CLI's
  chat-templated prompt a continuation can flip between the backends (about
  0.1 on a 22-magnitude top logit). The reproducible gate is the explicit-token
  stream (`make bonsai-cuda-parity`); a long chat continuation is not
  evidence. See
  [the CUDA graph report](releases/bonsai-cuda-graph-2026-09-30.md), section 4.

## Report

[Recorded evidence for the CUDA unit](releases/bonsai-cuda-graph-2026-09-30.md)
and [the CPU reference unit](releases/bonsai-cpu-reference-2026-09-30.md),
[the session unit](releases/bonsai-session-2026-09-30.md) and
[the serving unit](releases/bonsai-serving-2026-09-30.md).
