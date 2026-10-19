# The post-decode KV checkpoint: measured, and why it is not landed

Attempted and reverted on 2026-10-19 (commits `f211d555`, `52edf0c8`, local only,
never pushed). This note records what was measured, the mechanism that stopped
it, and what a follow-up would have to solve. The instrument is kept:
`tests/ds41_kv_post_gate.sh`.

## What was attempted

B-01..B-04 ask that a continuation reuse the assistant block the model itself
emitted, instead of re-prefilling it. The engine stores one KV checkpoint, at
the prompt end, before the first generated token
(`/data/YoungAi/src/core/core_v41_api.c:195`; port `ds4_ds41_draft.inc`), so
everything generated after it is re-prefilled on the next turn.

The attempt added a second slot: `K_pre` stayed the engine's checkpoint,
unchanged, and `K_post` saved the same state at the end of decode over the
prompt followed by the committed generated tokens. A restore took the longest
matching prefix, so a client that replays the model's own bytes (the exact
DSML replay) would reuse the assistant block while a retry of the previous
prompt alone still resolved to `K_pre`, exactly as the engine does.

## What it measured (GB10, `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative`)

| Surface | Result |
| --- | --- |
| Deterministic route (penalty set), continuation | Reuse 3772/3773 tokens, suffix 1. The engine's own reuse on the same prompt: 3725/3773, suffix 48. Gain: 47 tokens per turn. |
| Drift against a cold run, extension | Row RMS 0.17/0.21/0.24/0.12, max ~1.0, 0 of 48 rows byte-identical. |
| Drift against a cold run, the engine's own reuse | Row RMS 0.15/0.22/0.23/0.23, max ~1.2, 0 of 48 rows byte-identical. Same class, same size. |
| Served DSML route (chat, MTP speculation) | The slot is never written: `[v41] post-decode checkpoint skipped: emitted N, state advanced N+1` (39 times in one suite run). |
| Conformance (quiet server) | 14 pass / 4 fail, row for row the R7 baseline: B-01..B-04 unchanged. |
| Compliance (quiet server) | 22 pass / 0 fail. |
| Host gates | `ds41 KV import` PASS with seven new slot cases, `model lifetime` PASS, `ds4.o` clean. |

## The mechanism that stopped it

`n_past` is documented as "positions already committed"
(`ds4_ds41_forward.inc:251`), and the checkpoint may only claim what the
buffers hold. In the deterministic route the frontier sits one **behind** the
emission count (the last emitted token launches no forward), which is why the
first version refused to write at all: `emitted 48, state at 3772 (expected
3773)`. Taking `n_past` as the authority fixed that, and the ceiling became one
token: a continuation reuses `|P| + N - 1`.

In the served route the offset is the other way: the spec loop's frontier runs
one position **ahead** of the emitted tokens, so the emitted id list cannot
describe the buffers and the guard refuses to write. That route is the one the
line's criterion measures, so the change bought nothing where clients look.

## Why it is not landed

No client-visible gain on the line's surfaces, a second ~21 MB checkpoint copy
per request wherever it does fire, and a named deviation from
`core_v41_api.c:192-195` for a benefit only reachable when a client sets a
penalty. The measurements are worth more than the code.

## What a follow-up needs

Resolve the served route's frontier offset (the emission-versus-state gap under
MTP) so the checkpoint can claim generated positions there, or keep the live
session state resident instead of checkpointing a copy. Either way the work sits
with the KV/session contract this line reserved, not with the DSML carriage.

## Evidence

- `docs/evidence/ds41-dsml-gate/post-decode-ckpt-52edf0c8-20261019/` — the first
  run's diagnostics (`state at 3772 (expected 3773)`) and row-dump hashes.
- `docs/evidence/ds41-dsml-gate/replay-post-52edf0c8-20261019/` — the three
  suites, contended and quiet, including the run whose zeros were another
  client's interleaved requests.
- Spark side: `~/ab-evidence/post-gate{,2,ref}` hold the gate runs' row dumps
  (72 MB, 144 MB and 120 MB) and the reference arm.
