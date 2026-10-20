# The post-generation KV checkpoint

Landed on `feat/ds41-kv-reuse-post-decode`; the served measurement is the
branch's acceptance run, not something this note may assume.  The earlier
attempt at the same feature, its measurements and its revert are recorded
below, because its blocker is the thing this landing fixes.

## Why

The engine keeps one checkpoint, taken after the prefill and before the first
generated token (reference `/data/YoungAi/src/core/core_v41_api.c:192-195`, port
`ds4_ds41_kvreuse.inc`).  Everything the model then generates is re-prefilled on
the next turn, while the reference's own account of the cost is 31 tool calls
adding 5-9k tokens a step, 40k-190k re-prefilled a step, 40 s to 3 min a step at
a normal 1000 t/s prefill and 40 t/s decode (`core_v41_kvreuse.c:1-6`).

The exact-DSML replay makes the next turn's prompt begin with the bytes the
model itself sampled, so that block is reusable in principle: the prompt the
next turn sends contains the same tokens the state already holds.  Measured on
the served revision `4a0a157f`, all 14 `[v41] KV reuse` lines report
`reused == the previous prompt length` -- the replay reaches the prompt but the
checkpoint stops one turn short of it.

This branch adds the second checkpoint that closes the gap: `K_turn` (written as
`T` in the port), saved when the run ends, so the next turn reuses the block the
model just produced.  The reference has one checkpoint; this second one is a
named deviation.

## What it is

| | `K_pre` (the reference's) | `K_turn` (the port addition) |
| --- | --- | --- |
| Written | after the prefill, before the first token | when the run ends |
| Position | the prompt's token count | the prompt, then what the state committed |
| Logits row | yes (an identical prompt prefills nothing and samples from it) | no (its position is never a prompt end) |
| Serves | a prompt that starts with the saved prefix, at any length | a prompt strictly longer than the saved prefix |
| Spans | window ring, compressed KV by `ng_src` groups, index keys, compressor remainder rows, main-hidden ring + counters, per-layer `ng_src`/`cpend`/`win_from` | the same set, from the same copy loop (`v41_kv_save_spans` / `v41_kv_load_spans`) |

The restore tries `K_turn` first and falls through to `K_pre` untouched: a miss
on the turn checkpoint must not disturb the prompt-end decision.  Both die on
the same events (`v41_kvreuse_drop`, a model close, an amplifier change).

## What the invalidation keeps

The drop keeps the prompt of the checkpoint it invalidates and hands it over
once (`v41_kvreuse_stale_prompt`; the reference's `g_stale` and
`ds4_engine_v41_kv_stale_prompt`, `core_v41_kvreuse.c:40-56`), NULL for an image
prompt, whose vision rows no checkpoint holds.  The reference spends the idle
moment after a plugin switch re-prefilling that prompt under the new amplifier,
so the user's next message does not pay the whole prefill (`plugins_rewarm`,
6057991).

Nothing consumed it while the only switch this deployment had was a restart
(`dgx-recipe/scripts/ds41_install.sh:202`): the checkpoint is VRAM, so a
restarted process has nothing to warm, and the port's own switch stand-in fires
at the start of the run that observed the change, which is already inside a
request.  The live switch below is what gives the retention its consumer.

## The live switch, and the retention's consumer

The reference's switch is a queued job: `POST /api/models/plugins` queues a
switch task on the worker's own queue, the worker applies it between requests
(`ds4_engine_v41_switch_plugins`, `core_v41_state.c:96-115`: swap the two
directories, re-mount, drop the checkpoint), and then `plugins_rewarm`
re-prefills the dropped prompt under the new amplifier while the worker is idle
(`server_plugins.c`, 6057991).  This branch lands the same chain:

- `POST /v1/plugins` with `{"zchain": dir|null, "posttrain": dir|null}` answers
  `{"ok":true,"zchain":...,"posttrain":...,"digest":...}`, or 409 carrying the
  mount's own refusal.  `""` and `null` both mean "not mounted".
- `Model::v41_switch_sidecar` (`crates/ds4-core/src/v41_run.rs`) drops the mount
  cache first -- a switch re-reads the disk even for a directory this process
  mounted before (`core_v41_amp.c:129-130`) -- mounts the new set, and notes its
  identity, which is what makes the native drop both checkpoints and keep the
  prompt they described (`v41_kvreuse_stale_prompt`).
- The owner thread applies it, from its own control channel, when the inference
  queue is empty: the process-level route tables never change under a run.  The
  answer reaches the client before the rewarm starts.
- `DecodeIo::v41_rewarm` runs one 0-predict generate over the retained prompt --
  prefill and a prompt-end checkpoint, no decode -- and stops at the next
  prefill block if a request has arrived meanwhile, that request saving its own
  checkpoint.

Two named differences from the reference:

1. Ordering plumbing.  The reference's worker has one queue and one condvar, so
   its switch job wakes the worker.  This host's job channel is inference-typed,
   so the switch rides its own channel and the owner checks it between requests
   and under a bounded idle wait (250 ms) -- the same shape as the host's own
   stop polling (`serve.rs`, `owner_loop`).  A queued request is always served
   first, so the switch cannot jump a request that already arrived.
2. Durability.  The reference writes `serve_pick.txt` after a switch, so a
   restart comes back to the user's last choice (`server_plugins.c:86-89`).
   Here the domain is the deployment's (`ds41flash_1017_install.sh --domain`),
   so a switch lasts until the next restart, which comes back to the script's
   choice.

The disk half needs nothing: a store candidate is gated by the mount digest and
the payload class in its trailer
(`crates/ds4-server/src/v41_disk.rs:109-114`), so another domain's
checkpoint can never be imported after a switch.

Verified model-free only (`cargo test -p ds4-core`, `cargo test -p ds4-server`):
the switch re-reads a changed directory from disk and reports the new identity,
a refused directory writes no device table and notes no identity, the retained
prompt is handed over exactly once, the wire body and its refusals, the owner
applying a switch with the rewarm standing aside for a queued request.  The
served path is not yet exercised: no sidecar pair has been switched on the Spark
since this landing, so the rewarm's real cost and the first message's reuse
after a switch remain unmeasured.

## The key is the state's positions, not the emitted ids

`v41_kvreuse_save_turn` keys `K_turn` by the prompt followed by the ids the
state committed, and the run appends those ids where the state advances
(`ds4_ds41_draft.inc`): the pure decode step, the k=0 round, and the accepted
rows after the verify rollback keeps `1 + a` of them.  It never appends at the
emit sites, because the emitted list and the committed frontier are different
things:

- an earned-token break stops emitting while the accepted rows are already in
  the buffers;
- under the served route (MTP speculation) the frontier runs ahead of the
  emission, and under pure decode it runs one behind (the last emitted token
  launches no forward).

That difference is what the earlier attempt at this feature died on: keyed by
the emitted ids, the served route's guard refused to write at all.  Keyed by the
state, the buffer equals `n_past - np` by construction, and the run writes the
checkpoint whenever that identity holds (`rc == 0 && st.n_past > np && turn.n ==
st.n_past - np`).  If it does not hold -- a failed launch or a failed rollback
-- no checkpoint is written at all: a checkpoint whose key described other
tokens would be wrong KV, not slow KV.

The memcmp in `v41_kvreuse_pick` is then the whole correctness argument: a match
means the incoming prompt's first `n` tokens are, token for token, the tokens
the state forwarded, so the copied spans are that prompt's own prefix.  A
mismatch -- a retry, a re-rendered block, a different client's spelling -- is a
miss, and the prompt falls back to `K_pre` or to a cold prefill.

The exactness the reuse buys is FP-level, not bit-level: the suffix chunks from
the checkpoint position, so block boundaries differ, exactly the class the
reference itself names for its own reuse (`core_v41_kvreuse.c:16-19`, the same
class as changing `--v41-chunk`).

## Memory

`K_turn` is a second full checkpoint copy: the span set is dominated by the
window rings, so the served revision's `(checkpoint 18.3 MB)` at a ~350-token
prompt becomes about twice that.  At long context the compressed KV dominates --
the reference's own account puts it at ~0.9 GB per 1M tokens -- so the pair
costs about twice the single checkpoint there.  Neither is persisted by default:
the disk half (`v41_kvreuse_export` / `_import`) carries both classes, each
payload naming its own (the store writes them under "cold" and "continued", its
own reason vocabulary), while the served configuration still runs with
`disk=false`.

## The earlier attempt, and what changed

2026-10-19 (`f211d555`, `52edf0c8`, local commits, never pushed) saved the same
second slot, keyed by the emitted ids, and was reverted.  What it measured on
the Spark:

- the deterministic route (penalty set, speculation off): reuse 3772/3773 tokens
  against the engine's own 3725/3773 on the same prompt, so a 47-token gain per
  turn; drift against a cold run in the same class and size as the engine's own
  reuse (row RMS 0.15-0.24 either way, no row byte-identical);
- the served DSML route: the slot was never written -- `emitted N, state
  advanced N+1`, 39 times in one suite run -- so the rows the line's criterion
  measures were unchanged (B-01..B-04 still reserved).

It was reverted for having no client-visible gain where clients look.  This
landing keeps the slot and moves the key to the state, which is the served
route's blocker; the deterministic-route numbers above set the expected gain and
the expected drift class for the new run.

## Verification

Local, without a GPU (`make test-ds41-kv-import`, `make test-ds41-boundary`):
`tests/test_ds41_kv_import.c` carries the turn checkpoint's own cases -- the
continuing prompt, the committed-id mismatch (which must fall back to the
prompt-end checkpoint), the equal-length and shorter prompts (neither may be
served by it), the image identity including the rule that a sampled position
carries none, the sampled-route suffix, and the drop.  `test_ds41_model_lifetime`
covers the close path for both slots, and the boundary fixture asserts, for every
case it runs, that the buffer the run fills equals the state's own advance and
that the save was handed exactly that buffer.

The same file carries the disk half's class end to end: the payload's own class
byte (0 = prompt, 1 = turn) decides the slot the import lands in, an unknown
class byte or a class the payload's shape contradicts is refused, the turn key
is the prompt plus the committed ids (a cap too short writes nothing), and a
turn payload round-trips with both copies of the spans damaged beforehand.

Served, on the Spark (GB10) against the binary built from this branch
(`ds4-server` sha256 `785bd6b7...`, `--prefix-reuse exact`, domain `code`):

- `tests/ds41_kv_post_gate.sh` PASS.  At a 3725-token prompt the A run writes
  `[v41] KV turn checkpoint 3772 token (20.7 MB)`, D reuses `3772/3773 token
  (turn checkpoint 20.7 MB), prefill suffix 1`, and its first logits row differs
  from the cold run's by max |d| 1.0769 (RMS 0.18651) -- the reuse class, against
  the gate's bound of 2.0.  0 of 48 rows are byte-identical and the arms share 4
  of 48 leading ids before the first argmax flip, both reported rather than
  asserted; after a flip the two arms are simply in different continuations
  (rows 5 onward differ by 20 to 30).
- `tests/spark/ds41_dsml_full_gate.sh` PASS: fix rows 19, failures 0,
  **reserved 0**, unverified 0.  The four reserved rows flipped from a replay
  gain of 0 to 40 tokens:

  | Row | Surface | Before | After |
  | --- | --- | --- | --- |
  | B-01 | OpenAI chat | prompt 403, cached 348, gain 0 | cached 388, gain 40 |
  | B-02 | OpenAI chat, reordered echo | prompt 403, cached 348, gain 0 | cached 388, gain 40 |
  | B-03 | Anthropic | prompt 407, cached 352, gain 0 | cached 392, gain 40 |
  | B-04 | Responses | prompt 403, cached 348, gain 0 | cached 388, gain 40 |

  Everything else held: conformance 18 PASS / 0 FAIL, compliance 22 / 0, the
  client leg 3 / 0, the host legs and the C-oracle parity PASS, and the wire
  probe is unchanged (15 PASS, W-12 the known open `reasoning_tokens` row, W-15
  and W-16 refused as decided).
- The serve's own log carries the mechanism: 48 turn checkpoints written in one
  suite run, 9 of its reuse lines served by the turn checkpoint (`375/390`,
  suffix 15; `24049/24069`, suffix 20; 38.2 MB at 24k tokens).

The disk half, on the Spark (GB10, 2026-10-20, `make cuda-spark` from
`ba30c9ea`, sibling gates' route: its own boots, port 8099, `--kv-disk-dir`, the
`code` sidecar, 12000 prompt bytes, 48 tokens):

- `tests/ds41_kv_disk_turn_gate.sh` PASS.  A writes both classes at a 3725-token
  prompt (`[v41] KV turn checkpoint 3773 token (20.7 MB)`, two store entries);
  B, a fresh process whose prompt replays the model's own bytes, reports
  `[v41] KV disk turn checkpoint loaded (3773 token, 20.8 MB)` and `[v41] KV
  reuse 3773/3774 token (turn checkpoint 20.7 MB), prefill suffix 1`, prefilling
  1 token in 0.1 s where the cold arm of the same request prefills 3774 in
  4.4 s.  Its first logits row sits 0.9844 max / 0.19850 RMS from the cold run's
  -- the class the post gate measured above, against the same bound -- with 0 of
  48 rows byte-identical and 3 leading ids shared, reported rather than
  asserted.  No row is bit-comparable with A by construction: B's first row is
  the forward of the client's last token, a position A never computed, which is
  why the row verdict is the cold run's.  D, the prompt alone in a third fresh
  process, still lands on the prompt class (`3725/3725 token (checkpoint
  21.2 MB), prefill suffix 0`), so the turn entry does not shadow it at the
  prompt's own length.

`make proof-cuda-long` ran for this branch on the Spark (2026-10-20, the serve
stopped): six cells, all `OK`, `PROOF_V41_EXIT=0`.  The three eager arms report
`v41 graph captures=0`, the three capture arms 15 each, and all six produce the
same output (`sha 2718cb6d...`, `tok_md5 8a65268e...`, `gen=1024`, 249-252 s a
cell), so parity holds on `selected_token_ids_md5` for eager vs capture, +fp8-kv
and +fp8-kv-predecode.

The V4.1 shape that run needed (`7cd44c52`): the harness tokenizes each fixture
through the same binary (`--dump-tokens`) and drives the family's `--gen-ids`
with `--engram-dir`, without which the run dies ENOENT on the artifact's
recorded Engram shard paths (the converting host's); the eager arm is driven
with `--no-graph`, because the v41 route sets the graph switch itself
(`lib.rs:782`, `v41_run.rs:1739`), so a profile-level `DS4_CUDA_LAYER_GRAPHS=0`
cannot reach it and the arm would run captured.  A cell whose capture count
contradicts its arm fails the run, so the parity cannot pass vacuously.
