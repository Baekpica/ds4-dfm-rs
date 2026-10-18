# GLM-5.3-VQ: the thought bound fires, the turn still does not end

Unit 8d of the phased plan -- bound a GLM chat turn's thought. Branch
`fix/glm53-vq-penalties`, commit `566bc6c`, 2026-10-18, binary built on the
Spark from that commit. It also CORRECTS
[the 2026-10-17 record](glm53-vq-sampler-2026-10-17.md): its greedy arm was
not greedy alone, and its "DRY does not stop the runaway" conclusion holds
only at the artifact's own temperature.

## What the unit set out to do

The previous record's evidence was that a GLM chat turn never leaves its
scratchpad: at the artifact's own temperature 4 of 5 runs spent the whole
budget inside the thought and no run emitted an end token. The machinery that
forces the close already existed, but only for tool turns: the reference
reserves post-thinking when a tool start is required or a tool result is in
the prompt (`tests/parity/dsml_c_oracle.c:445-446`, from `ds4_server.c` at
v0.6.3-dfm), and a chat turn has neither. The hypothesis was that the missing
end token was the unbounded thought, and that bounding it would make effort
`low` actually low without capping the temperature.

## The change

`DS4_SERVER_GLM_THINK_CAP_LOW/_HIGH/_MAX` (tokens, 0 = off, the default) give
the GLM scope a per-level forced-thought budget. `parse_request` resolves the
level's value into `ParsedRequest::think_cap`; the continuous and the serial
lane hand it to `sampling_override`, which uses it as the cap and as a second
reason to reserve post-thinking. The startup line reports the three values.
Every other family keeps 0, so the change is inert for them, and at 0 a GLM
turn is byte-for-byte the turn it was.

## The measurement method

One payload: the session's own replay, 13,335 prompt tokens, `max_tokens`
1000, effort low unless a row says otherwise. One box, one build.

The knobs that matter are of two kinds, and only one kind can be compared
soundly:

- temperature and the DRY penalty are request-level, so every arm that uses
  them is measured inside ONE server process (the table below). This is the
  sound experiment and the one the conclusions rest on.
- the thought budget is process-level (an environment variable), so the
  capped arms were necessarily measured on a different process than the
  uncapped ones, and the difference is reported as a within-process
  observation of the capped process only.

The replay's own row of the 2026-10-17 table (50 sentences at temperature
1.0, 0/5 stops) reproduces here byte-for-byte, arm for arm, which is what
makes the two records comparable at all.

## The sound A/B: one process, request-level settings

Serve environment: no temperature ceiling, DRY off, `think_cap=0/0/0`, so the
request body is the only thing that varies. Three seeds (`11 22 33`).

| Request | Completion / finish | Thought (chars) | Content (chars) | Stops |
| --- | --- | --- | --- | --- |
| temperature 1.0, DRY 0 | 1000 `length`, 1000 `length`, 1000 `length` | 4225, 4522, 0 | 0, 0, 3806 | 0/3 |
| temperature 1.0, DRY 0.8 | 945 `stop`, 1000 `length`, 1000 `length` | 3448, 4563, 0 | 974, 0, 3439 | 1/3 |
| temperature 0, DRY 0 | 1000 `length` x3, identical | 4369 | 0 | 0/3 |
| temperature 0, DRY 0.8 | 557 `stop` x3, identical | 1233 | 1393 | 3/3 |

What this settles:

1. At the artifact's own temperature the DRY penalty is marginal: 0/3 without
   it, 1/3 with it. That part of the previous record stands.
2. At greediness the penalty is decisive: without it the thought runs to the
   budget (0/3, 4369 characters, no content at all); with it every run stops,
   deterministically, at the same token with the same thought and the same
   answer (557, 1233, 1393).
3. Therefore the deployment that stops this prompt is greediness AND the
   penalty TOGETHER -- not greediness alone, as the 2026-10-17 table's
   "`temperature` 0.0 (greedy) **5/5**" row implies, and not the penalty
   alone. That row was measured on a server started with
   `DS4_SERVER_GLM_TEMP_MAX=0` and `DS4_SERVER_DEFAULT_DRY_MULTIPLIER=0.8`
   (the server's own config history, `glm53unc.log:983`, `:1058`), so it was
   greedy plus DRY, not greedy.
4. The identical 557 / 1233 / 1393 signature also appears in this record's
   capped-greedy process and in the 2026-10-17 deployment table, so the three
   agree on the working configuration.

## The capped process: the bound fires and does not end the turn

One process started with `DS4_SERVER_GLM_THINK_CAP_LOW=512`,
`_HIGH=768`, `_MAX=1024`, no ceiling and DRY off, so the temperature arms
below are request-level and the bound is the process-level variable.

| Request | Runs | Stops | Thought (chars) |
| --- | --- | --- | --- |
| temperature 1.0, low (cap 512) | 5 | 1/5 | 2233, 2336, 0, 2316, 2235 |
| temperature 1.0, high (cap 768) | 2 | 0/2 | 3405, 3497 |
| temperature 1.0, max (cap 1024) | 2 | 0/2 | 3740, 3241 |
| temperature 0, low (cap 512) | 3 | 0/3 | 2251 (identical) |

The bound works exactly as built: the thought ends at the level's budget
(~2,230-2,340 characters at 512 tokens, ~3,400-3,500 at 768), and at `max` the
budget (1,024) exceeds the 1,000-token turn, so it never fires and the thought
still swallows everything. The tight clustering across seeds is the signature
of a forced close, not of the model deciding.

It does not end the turn. Eight of the nine bounded runs at the artifact's own
temperature still hit the token budget; the thought was replaced by content
that wanders ("I plan the entry's unblurred letterform margins ... I assign
typed directions"). The one stop is the arm's only natural end token. At
greediness the bound is inert for termination too: 0/3 either way, the
difference being only where the tokens go (thought 4369 with no bound,
thought 2251 plus content 2243 with one).

Unit acceptance ("5/5 stops at low with a usable answer") is NOT met. This is
a null result, recorded as one.

## An unresolved anomaly worth knowing

The 2026-10-17 greedy row (399 `stop` / thought 972 / content 909, and 685 /
925 / 2462 for two other seeds) does not reproduce on any later process: the
same request against the same printed configuration returns 557 / 1233 / 1393,
six times across three processes. Every other arm in this record reproduced
byte-for-byte across restarts. The difference is unaccounted for; the earlier
row is the one to distrust, and greedy arms should be re-measured rather than
cited.

## Limits

- One prompt (the session's own replay), 2-5 seeds per arm, one box (GB10,
  300-2200 MHz user-managed clock range) and one build (`566bc6c`).
- At temperature > 0 the arms are samples, not rates; at temperature 0 they
  are deterministic within a process.
- The bound was never measured together with the working configuration
  (greedy + DRY 0.8). A deployment must not assume it is free there.
- The capped arms are single-process observations, so they say "the bound does
  not end the turn"; they cannot say what the bound costs in a config that
  does end it.

## The deployment

Restored to the configuration that made the serve usable, verified in the
server's own startup line: `DS4_SERVER_GLM_TEMP_MAX=0` and
`DS4_SERVER_DEFAULT_DRY_MULTIPLIER=0.8`, with `think_cap=0/0/0`. The bounded
thought stays off, which is also its default.

## The code keeps the knob

The mechanism is kept, default off: it is inert for every other family, it is
test-covered, it is the reproducible instrument for the measurement above, and
a replacement artifact may want a bounded scratchpad. Its measured value for
THIS artifact is zero, and the record says so.
