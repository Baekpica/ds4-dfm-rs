# Prism Bonsai (qwen35) serving path — 2026-09-30

`ds4-server` (the Rust host) now serves the family on both backends: the whole
request path runs — plan, session, ChatML rendering, the thinking split, SSE
streaming — and `./run-bonsai.sh server` proves it end to end with one real chat
request. Getting there meant fixing five defects, two of which crashed the
server on its first request; none of them were visible from the CLI, which is
why every earlier gate stayed green.

Host: RTX 4070 SUPER (sm_89), `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`
(851 tensors, 6.71 GiB). Branch `feature/qwen35-port`, on top of `78c086f`.
Sibling reference: `/data/ds4` branch `bonsai`, tip `bbaf298`.

## What landed

- `ds4.c` — `ds4_engine_supports_batching` had no `QWEN35` arm, so the family
  fell through to the DeepSeek multi-sequence slab body and the server
  segfaulted in `metal_graph_alloc_raw_cap` before it ever listened. The family
  answers `false` now, like Inkling and GLM-5.3: its session is the trunk state
  and there is no multi-sequence graph for it.
- `ds4.c` — `ds4_engine_session_graph_fit_quote` had no `QWEN35` arm either, so
  the first real request segfaulted in `metal_graph_alloc_bytes_estimate`,
  again on the DeepSeek slab estimate. The family now answers budget-less
  (`fits=1, fail_open=1`) inside its own context bound, which is the truth:
  the session opens with its own chunk-halving loop, so there is no per-bank
  number to give and the host keeps its unquoted margin.
- `ds4-server-rs` — the continuous lane was requested for any family while the
  caps said otherwise; it is now gated on the family's own bank support
  (`Support::None` skips it), which is the same refusal the native side makes
  and removes a wasted native call for Inkling and GLM-5.3 as well.
- `crates/ds4-core/src/serving_host.rs` — the bank quote priced Bonsai with the
  generic `bank_kv_bytes`, which charges all 64 blocks as attention rows. The
  family is 16 gated-attention layers plus 48 gated delta-net layers whose
  state does not grow with the context, so the estimate was four times the real
  KV and refused every usable context even with the memory floor at zero. The
  new `qwen35_bank_bytes` mirrors the native allocations: fp16 key/value rows
  for the attention layers (all 64 in f32 on the CPU reference, which allocates
  a row per block), the recurrent matrix and convolution window per delta-net
  layer, and the CUDA graph's chunk-sized transient buffers.
- `crates/ds4-server/src` — the ABI id 13 (variant `Qwen35_27B`) fell through
  `syntax_for_model_id` to the DeepSeek syntax, so a served request would have
  rendered DSML for a ChatML model. `ModelSyntax::Qwen35` now maps to the
  family's own ChatML path (`render_qwen_chat_ex`, the Qwen tool envelope, the
  `<think>` split, `ChatFormat::Qwen4Exp`) exactly as the sibling C server maps
  its QWEN syntax. Vision, audio and video stay refused.
- `crates/ds4-server/src/models.rs` — the sibling's names
  (`prism-bonsai-2-27b`, `-chat`, `-no-think`, `-nothink`, `-reasoner`,
  `prism/bonsai-2-27b`) are accepted on `/v1/models/<id>`. The advertised id
  stays the GGUF stem, as it does for every family in this tree.
- `run-bonsai.sh server` — starts the Rust server, waits for the listener,
  reads the advertised id back from `/v1/models`, sends one chat request through
  the model's own template, prints the answer and stops the server. The
  subcommand is no longer refused; `ds4-server-c` (the C oracle) still cannot
  serve this family, so the mode uses `./ds4-server`.

## Evidence

`./run-bonsai.sh server` (CUDA, ctx 32768, memory floor 1 GiB):

    server:  up (pid 1240413)
             listening on 127.0.0.1:8899 model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open ...
    id:      Ternary-Bonsai-2-27B-PQ2_0

    wall:    1.40s for the request
    finish:  stop
    answer:   Paris.
    reasoning: The user is asking a simple factual question: "The capital of France is". I just need to complete the sentence. The capital of France is Paris.
    usage:    {'prompt_tokens': 45, 'completion_tokens': 36, 'total_tokens': 81, ...}

`DS4_BONSAI_BACKEND=cpu DS4_BONSAI_CTX=8192 ./run-bonsai.sh server` (the CPU
reference trunk, 283 s for the same 81 forwards):

    wall:    283.26s for the request
    finish:  stop
    answer:   Paris.
    reasoning: The user is asking a simple factual question: "The capital of France is". I just need to complete the sentence. The capital of France is Paris.

Both backends produce the same reasoning block and the same answer through the
server, which is the claim this unit can make. The sibling C server answers the
same prompt with `content: 'Paris'` and a differently worded reasoning block
(`The user asks: ...`), but it renders a different prompt (57 prompt tokens
against this tree's 45), so that comparison is coarse: the token-level
cross-tree claim rests on the explicit-id logits parity recorded in
[bonsai-cuda-graph-2026-09-30.md](bonsai-cuda-graph-2026-09-30.md) (same five
ids, max|d| 0.00000 between the two trees' CUDA paths).

Two more served surfaces:

    $ curl -s -o /tmp/alias.json -w 'http %{http_code}\n' \
        http://127.0.0.1:8899/v1/models/prism-bonsai-2-27b
    http 200
    {"id":"prism-bonsai-2-27b","object":"model",...,"name":"Ternary-Bonsai-2-27B-PQ2_0",...}

    $ curl -sN .../v1/chat/completions -d '{... "stream":true, "max_tokens":48}'
    sse lines: 38
    "content":"\n\n"   "content":"Paris"   "content":"."
    data: [DONE]

The plan's own numbers, from `--check-config` (bank = 2.39 GiB at ctx 32768
against 6.71 GiB of weights; `available` is free device memory at plan time):

    ctx=32768 floor=1G: ACCEPTED  total 10.11 GiB avail 11.32 GiB
    ctx=49152 floor=1G: ACCEPTED  total 11.11 GiB avail 11.32 GiB
    ctx=65536 floor=1G: REFUSED   total 12.11 GiB avail 11.32 GiB  quote_overflow
    ctx=32768 floor=4G: REFUSED   total 13.11 GiB avail 11.32 GiB  quote_overflow

`--kv-disk-dir` never reaches the engine for this family: the plan reports
`error: qwen35 session snapshots are unsupported (disk_unsupported)` and the
server does not start, where the sibling C server warns and serves on.

Regression gates re-run: `cargo test -p ds4-core` green (295 + 13 + 5 + 7 + 1 +
9 + 25 + 1 + 7 + 7, 0 failed), `cargo test -p ds4-server --lib` 314 passed with
the pre-existing `cache_identity::tests::bounded_sidecar_and_ple` failure
(verified failing at `78c086f` in a scratch worktree: it asserts that a
same-size rewrite changes the mtime+ctime snapshot, which this host's /tmp does
not give it).

## Traps worth recording

- A family without a bank runtime must be refused by `ds4_engine_supports_batching`
  and by the session fit quote. Both defaults end in the DeepSeek slab body, and
  both fail as a segfault inside a graph allocator rather than as an error, so a
  missing family arm is a crash, not a degraded path.
- The memory floor is device memory on CUDA, and this card leaves about 11.3 GiB
  free: 6.71 GiB of weights plus the 4 GiB default floor refuses every usable
  context. The launcher and the doc use 1 GiB and say why.
- The CPU reference allocates its key/value rows for all 64 blocks (the CUDA
  graph only for the 16 attention layers), which is 17.2 GiB at ctx 32768 and
  why the CPU leg of the server runs at 8192 on this 32 GiB host.

## Limits

- Serial lane only: no batching, no MTP/DSpark, no SSD/disk KV, no session
  snapshots, no tensor parallelism, no distributed ranks.
- The caps carry no qualified context, bank or prompt limit, and the plan warns
  `partial_unqualified` while reporting `reuse=exact`.
- The advertised id is the GGUF stem rather than the sibling server's
  `prism-bonsai-2-27b`; the sibling's names are accepted aliases only.
- `/v1/models` lists one entry, not the sibling's three (`-chat`,
  `-reasoner`).
