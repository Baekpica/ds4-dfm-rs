# GLM-5.3-VQ: the serial lane gets the penalty face, and the two lanes agree

Unit 8e of the phased plan -- the serial/CLI penalty gap. Branch
`fix/glm53-vq-penalties`, commits `c704903` (the port) and `b020f15` (the
greedy-commit gate), 2026-10-18. Everything is syntax-neutral: the face is zero
unless a request carries one, so DSML, Qwen and V4.1 behavior is unchanged.

## The gap, and why it was not left as documentation

The repetition penalty reached the CONTINUOUS lane only. The server forwards the
request's face to the bank (`serve_cont.rs` admit) and `family_cont_penalize`
applies it, from the bank's own generated segment, before the host sampler
(`ds4.c:64934-64946`, `:65871`). The SERIAL lane samples through
`ds4_session_sample` (`ds4.c:76977`), which carries no token history and dropped
the face silently.

The unit's plan offered "port it or document GLM's face as continuous-only".
Porting won on this evidence:

- The serial lane is a designed fallback, not an error path: `route_decide`
  sends a request there whenever continuous admission refuses
  (`serve_cont.rs:` "continuous admission rejected; serial fallback"), and
  `DS4_SERVER_CONTINUOUS=0` routes every request there (measured: 5/5 requests,
  `route_decisions.cont_unavailable`).
- DRY is half of the pair that stops this artifact
  ([the think-cap record](glm53-vq-think-cap-2026-10-18.md)), so a GLM turn on
  the serial lane had no loop breaker at all.

What the port is: `ds4_session_sample_penalized` (`ds4.c:76993`) applies the
same `core_decode_penalize` over the session's own committed tokens from
`gen_start` (the request's prompt length), so the prompt stays outside the
penalized segment exactly as `cb->generated` keeps it outside on the continuous
lane. The row is copied first, as the V4.1 route copies its `rowbuf`
(`ds4_ds41_sample.inc:203-222`), so `s->logits` stays the model's own row. A
zero face, an unusable `gen_start` or an uncommitted checkpoint take the plain
sampler, which makes the call byte-identical to `ds4_session_sample`.

`Penalty::of` (generate.rs) is now the ONE face build for both lanes, and it
applies DRY's base / allowed-length fallbacks (1.75 / 2) that `parse.rs` cites
from `server_generate_v41.c:363-365`. Without them `ds4_decode_penalize`
refuses a face whose base is <= 1.0, so a client that sent only
`dry_multiplier` used to carry an inert penalty on either lane.

## The second half: the greedy commit

The first port was not enough, and the live A/B at `c704903` shows it. The
serial lane commits a greedy segment in one native call
(`DecodeIo::eval_greedy`), and that committed tail is an argmax chain inside the
engine: it reads no history, so it cannot carry a face. Only the first token of
each segment was penalized.

`b020f15` stands the commit down when a face is set -- the rule the continuous
lane already applies to its device trial and speculation ("penalties read the
token history, so the device trial cannot carry them", `ds4.c:64971-64983`). A
greedy turn without a face keeps its commit, so the narrowing is exactly the
penalty case (test `serial_lane_keeps_the_greedy_commit_without_a_penalty`).

## Measurement

The Spark, one build per lane, the same four request bodies, `temperature` 0
and 0.8 DRY or none, `max_tokens` 600-1000, `seed` 11, effort low. Two payload
shapes: the session's own query (48 characters) and a repeat probe ("Repeat the
word bird 300 times").

| Request | Continuous lane | Serial lane, `c704903` | Serial lane, `b020f15` |
| --- | --- | --- | --- |
| session query, DRY 0.8 | `stop` 218 / thought 636 / content 464 | `length` 1000 / 3762 / 0 | `stop` 218 / 636 / 464 |
| session query, no DRY | `length` 1000 / 4991 / 0 | `length` 1000 / 4991 / 0 | `length` 1000 / 4991 / 0 |
| repeat probe, DRY 0.8 | `length` 600 / 2213 / 0 | `length` 600 / 1535 / content 924 | `length` 600 / 2213 / 0 |
| repeat probe, no DRY | `length` 600 / 2463 / 0 | `length` 600 / 2463 / 0 | `length` 600 / 2463 / 0 |

Read: the no-face arms were already identical across the two lanes, so the
lanes agree wherever no face is involved. At `c704903` the faced arms did not
agree, which is what exposed the greedy commit. At `b020f15` all four arms
agree, to the token count and to the character count, on both lanes.

Note what a face does to this artifact at greediness even where the two lanes
now agree: the repeat probe stays inside its thought for the whole 600-token
budget either way, and only the DRY arm of the session query leaves the thought
at all. The penalty is a loop breaker, not a cure for the wandering; that
stands.

## What the serial lane cannot do on this box

With the batch banks resident, the serial lane refuses a long request outright:
a 13,335-token and a 2,012-token body both answered "server is temporarily at
capacity for a N-token serial request (no session graph fits beside the batch
banks); retry shortly", while the 48-character body served. So the reachable
part of this gap is the SMALL-request fallback -- the "1+2"-shaped turn the
owner's report started from -- and a long GLM request that falls to the serial
lane now gets a retryable error rather than a silent penalty-free runaway.
`DS4_SERVER_CONTINUOUS=0` was used to reach the lane for measurement.

## The CLI

`ds4-cli`'s two generate loops now pass their own DRY face (`--dry-multiplier`,
with the same base / allowed fallbacks) and turn speculation off when it is set,
the served lanes' rule. `--dry-multiplier 0` (the default) keeps every call
byte-identical to the plain sampler. The agent lane (`ds4-agent`) has no
penalty flags and is unchanged.

## Deployment

The Spark is back on the documented configuration on this build: continuous
lane, `DS4_SERVER_GLM_TEMP_MAX=0`,
`DS4_SERVER_DEFAULT_DRY_MULTIPLIER=0.8`, no thought caps, verified with the
session query (stop at 218 with an answer; 1000 `length` without the face).
`GLM53UNC_RUN_FLOOR_MB` had to drop from 8000 to 7900 to start: the post-load
figure was 7990 MB and the 8000 floor refuses on a 10 MB margin. That default is
still the owner's call.

## Limits

- One box, one artifact, one build per column, four request bodies; the armed
  cases are deterministic at `temperature` 0 and single samples above it.
- The comparison is cross-process for the lanes (the lane is a process-level
  switch), so the lanes agree on these probes, not as a proof of general parity.
- The face's numerics are the continuous lane's own `ds4_decode_penalize`; this
  unit moved where it is called from, not what it computes.
- Small-request fallback is what was measured reachable; a larger-memory host
  would put bigger requests on the serial lane and this unit has not measured
  those.

## Tests

- `serial_lane_applies_the_request_penalty_over_the_generated_segment`: fails
  before the port (4 plain samples, 0 penalized) and asserts the face, the
  `gen_start` and the absence of a greedy commit after it.
- `serial_lane_keeps_the_greedy_commit_without_a_penalty`: the gate's complement.
- ds4-core 422 tests, ds4-server 350 + every integration target, green except
  the pre-existing `cache_identity::tests::bounded_sidecar_and_ple`.
