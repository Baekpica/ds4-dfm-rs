# Prism Bonsai (qwen35) session path — 2026-09-30

The family gained a real `ds4_session` on both backends: the CUDA graph and the
CPU reference trunk each own the whole trunk state, so session create, sync
with prefix reuse, eval, rewind-by-replay, invalidate and the context bound all
work, and `./run-bonsai.sh session` diffs them against the CPU reference.

Host: RTX 4070 SUPER (sm_89), `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
Branch `feature/qwen35-port-cpu`, stacked on the LGTM tip `442b0aa`.

## What landed

- `ds4.c`:
  - `ds4_session` holds `struct ds4_qwen35_gpu_graph *qwen35_graph` (CUDA) or
    `struct ds4_qwen35_ref_state *qwen35_ref` (CPU), never both; the family
    branch in `ds4_session_create` builds whichever the backend needs, and the
    graph is opened with a halving chunk loop so a chunk the device cannot hold
    still converges (a chunk of one is the per-token path).
  - `qwen35_session_pos` is the frontier (the graph's own position on CUDA, the
    sessions's counter on the CPU).  `qwen35_session_reset` drops the recurrent
    state; `qwen35_session_replay_if_stale` re-runs the kept tokens when the
    state is behind the checkpoint, which is what rewind and a failed decode
    need (the gated delta-net state cannot roll back).
  - create / sync / eval / free / invalidate / rewind / the payload paths / the
    speculative-argmax chain all take the family.
  - Syncing hands the trunk a chunk of up to `DS4_QWEN35_PREFILL_CHUNK` rows
    (512 default, 1024 max); every layer runs that many rows and the output head
    runs one, from a fixed one-row copy of the newest hidden state, so the
    pointer a kernel sees never depends on the chunk.
  - The engine gate accepts a Bonsai session on `--cpu` or `--cuda` (from
    `--first-token-test` or a plain generation request) and keeps refusing
    tensor parallelism, distributed ranks, SSD streaming, MTP/DSpark and
    steering by name.
- `crates/ds4-core/src/serving.rs`: the family's caps declare `HostNeed::Any`,
  which is what the runtime now does; the rest of the plan (serial lane, no
  banks/batching/snapshots/MTP, partial prefix reuse) was already declared and
  is unchanged.  Serving itself is not qualified yet.
- `run-bonsai.sh session` drives the CPU reference, the CPU session and the CUDA
  session, and diffs all three; `server` still refuses by name.
- `tests/test_qwen35_session.c` with `make test-qwen35-session` (both backends)
  and `make test-qwen35-session-multichunk` (`DS4_QWEN35_PREFILL_CHUNK=2`).

## Evidence

`make test-qwen35-session CUDA_ARCH=sm_89`, CUDA backend, 12 steps:

    FAIL 0 / PASS 20
    plain: position after sync equals the prompt length      PASS
    plain: decoded ids equal the CPU reference               PASS
    prefix reuse: ids equal the CPU reference across both    PASS
    rewind: the replay reproduces the reference ids          PASS
    invalidate: ids after the rebuild equal the reference    PASS
    context bound: the third decode past capacity is refused PASS
    long prompt (68 tokens, one 68-row chunk): the same five PASS
    qwen35 session path: PASS

`DS4_TEST_BACKEND=cpu ./tests/test_qwen35_session`, 4 steps: FAIL 0 / PASS 10,
the same five scenarios on the CPU reference trunk.  The CPU pass skips the
long prompt (about 3 s per forward) and runs fewer steps.

`./run-bonsai.sh session` (16 steps, chat-templated prompt):

    --- CPU session ---   IDENTICAL: the CPU session reproduces the CPU reference
    --- CUDA session ---  IDENTICAL: all 16 generated token ids agree, so the
                          CUDA session reproduces the CPU reference, which the
                          CPU session does too

The explicit ids gate is unchanged: `make bonsai-cuda-parity` PASS and
`./run-bonsai.sh ids` IDENTICAL, so the session path did not move the numbers
the earlier units pinned (`760,6511,314,9338,369` at 8 steps, then 16).

Timing from the session run: the CUDA session did the prompt plus 16 greedy
tokens in 2.37 s wall including the 0.7 s model copy; the CPU session and the
CPU reference each took about 2 minutes for the same 41 forwards.

## Traps worth recording

- The output head cannot be a `ds4_gpu_tensor_view` of the arena's last row:
  the view's offset is fixed at open while the chunk varies, so a 5-token
  prefill read row 18 of a 19-row arena and returned zero logits (the first
  test run failed exactly this way).  The one-row copy costs 20 KiB per forward.
- The host arrays are bounded by `DS4_MAX_LAYER`, which was 61 while this family
  has 64 blocks; that was fixed in `bc451af` (the CUDA graph had its own bound
  already).  The session's per-layer state inherits the fixed bound.
- `make cpu` leaves a CPU-only `ds4-c` that is newer than `ds4.o`, so the
  documented `make ds4-c CUDA_ARCH=sm_89` before GPU work is a no-op: remove the
  binary first, or the CUDA gates measure the CPU build.

## Limits

- Prefill is chunked but decode is one row per eval: no batching, no
  speculative decoding, no MTP.
- No snapshots (`ds4_session_payload_bytes` returns 0 and both payload paths
  refuse by name) and no disk KV.
- The attention row kernel stays the row-exact one for every chunk size; the
  token-tile kernel is never selected from the graph.
- The CPU reference trunk is a correctness oracle at about 3 s per forward, not
  a serving path.
