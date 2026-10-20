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
the disk half (`v41_kvreuse_export` / `_import`) still carries `K_pre` alone, and
the served configuration runs with `disk=false`.

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
covers the close path for both slots.

On the Spark, from the analysis host: `tests/ds41_kv_post_gate.sh` (the
instrument kept from the earlier attempt, its reuse-line pattern taught the
`turn` tag) for the exactness gate, `tests/spark/dsml_conformance.sh` for the
reserved B rows, `dsml_compliance.sh` and `open_grok_leg.sh` for the surface,
and `make proof-cuda-long` for long-context parity.  The engine-side criterion
is the serve's own line, `[v41] KV reuse N/M token (turn checkpoint ...)`, with
`N` past the previous prompt.
