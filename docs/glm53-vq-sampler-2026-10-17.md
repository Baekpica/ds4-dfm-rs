# GLM-5.3-VQ answers, then runs on: the served lane gets the penalty face

Unit: the sampled-generation runaway of the uncensored VQ artifact on the
Spark. Branch `fix/glm53-vq-penalties`, 2026-10-17. Everything is scoped to
the GLM-5.3 syntax (`ModelSyntax::Glm53`) and to the continuous lane that
serves it; the DSML, Qwen and V4.1 paths are untouched, and every default is
the engine's own zero unless the GLM scope sets it.

## What the owner reported

An open-grok session on `glm-5.3-flash-vq`, `reasoning_effort` low: the model
answered the question, then kept going -- first into pseudo-word salad
(`ctor-per-line-end-area-volume-points-presence-footed-perspective frame uses
instancing inheritance-populo-ratio ...`), then into random letter strings
(`atptds-lp-bbmoqdt-vo-wwre-ohorw-it-gokgsxy-...`). The turn streamed for
minutes and never ended. The session's own request (13,335 prompt tokens,
including open-grok's system prompt, memory and tool descriptions) was still
running when the client was stopped.

## The reproduction

The session's own `chat_history.jsonl` messages replayed against the live
server at its own settings (effort low, `temperature` 1.0, `top_p` 0.95, no
`min_p`, 1,000-token cap):

| Run | Completion | Finish | Where the 1,000 tokens went |
| --- | --- | --- | --- |
| A1..A5 (`temperature` 1.0, five seeds) | 1000, 1000, 1000, 1000, 1000 | `length` x5 | four runs never left the thought; one ran content to 3,806 chars |

Not one of the five produced an end token. The same context at
`temperature` 0 stopped in 5/5 runs (399 and 685 tokens, the last four
byte-identical), which is what makes the artifact's own temperature the
finding rather than a client mistake: the card says 1.0, and 1.0 is what the
model cannot take.

## What it is not

- **Not MTP.** The serve was restarted with `GLM53UNC_MTP_MODE=off` and the
  same probe still ran to the cap (1,500 tokens, `length`, a loop on
  `one specific definite/one specific single/...`).
- **Not the template or the stop set.** A plain `What is a gay person?` at
  the same effort stopped at 207 tokens (`finish` `stop`); a derailleur
  question stopped at 586. A 2,463-token prompt with a short question
  answered `On a clear day the sky is blue.` and stopped. So the prompt
  rendering, the `<think>` block and the three GLM stop ids all work.
- **Not context length.** The 2,463-token prompt above is longer than the
  context where the 28-token-prompt runs collapse.
- **Not the tool dialect.** The reproduced run had no tool calls at all.

## What it is

The artifact is a measured lossy conversion. Its own card
(`Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`, read 2026-10-17) records the
routed experts at 14.4% relative MSE (0.143756-0.144532 per layer, the official
conversion's own number), Q4_K nonexpert matrices at 0.005, held-out prose PPL
**4.8555 -> 8.0797**, a Korean meeting-summary case that "degrade[s] under
greedy decoding", and the sentence "This is not a quality-equivalent
replacement of the source model."

On this engine that damage shows up as generation that does not terminate:
sampled decoding wanders, and the port had nothing to break a wander -- the
engine's sampler is `sample_top_p_min_p` (ds4.c:40678) with no repetition
control anywhere, and the harness sends `max_completion_tokens` 32768, so a
runaway is minutes of streaming rather than a wrong answer.

## The change

The V4.1 route already carries the reference's answer:
`core_decode_penalty.c` ported as `ds4_decode_penalize`
(`ds4_ds41_sample.inc:49-232`) -- frequency, presence, and DRY, the last of
which penalizes a candidate for continuing an already-seen sequence and
therefore also works at temperature 0 ("the only way to have 'deterministic
direct output + no loops'", ds4_ds41_sample.inc:71-75). This unit wires that
same code into the family lane GLM is served from, rather than inventing a
control:

- `ds4_cont_request` carries `freq_penalty`, `presence_penalty`,
  `dry_multiplier`, `dry_base`, `dry_allowed_length`; all zero leaves a row
  byte-identical, so every other family and caller is unaffected.
- `ds4_family_cont_bank` carries them per bank; `family_cont_penalize`
  applies them to the bank's host logits row from `cb->generated`, the same
  prompt-excluded scope the reference documents.
- A penalty turns the device trial off for that request
  (`family_cont_spec`): the trial picks its target rows by argmax on the
  device, so it cannot carry a history-reading penalty. This is the V4.1
  route's own rule for a penalty plus speculation.
- The server already parsed the OpenAI keys; it now forwards them to the
  lane. The GLM scope resolves their request-omitted values from
  `DS4_SERVER_DEFAULT_FREQ_PENALTY`, `_PRESENCE_PENALTY`, `_DRY_MULTIPLIER`,
  `_DRY_BASE`, `_DRY_ALLOWED_LENGTH` and reports them at startup, exactly
  like the temperature/top_p/min_p/effort knobs the previous unit added.
  Every penalty knob defaults to the engine's own zero: an untouched
  deployment behaves exactly as before.
- `DS4_SERVER_GLM_TEMP_MAX` pins the temperature of every GLM request when a
  deployment sets it (absent = serve what the client asked, the host rule).
  `0` means greedy, which is the one setting that measured stable.

## Measured effect

Spark, the same 13,335-token context, 1,000-token cap, five seeds each
(`11 22 33 44 55`). "Stopped" means the model produced its end token inside
the cap:

| Setting | Stopped | Ran to the cap |
| --- | --- | --- |
| `temperature` 1.0, no penalty (the artifact's own card values) | 0/5 | 5/5 |
| `temperature` 1.0, `min_p` 0.05 | 2/5 | 3/5 |
| `temperature` 0.7 | 3/5 | 2/5 |
| `temperature` 0.5 | 0/5 | 5/5 |
| `temperature` 0.0 (greedy) | **5/5** | 0/5 |
| `temperature` 1.0, DRY 0.8 | 1/5 | 4/5 |
| `temperature` 1.0, effort none (thought closed) | 0/5 | 5/5 |
| `temperature` 1.0, effort high | 0/5 | 5/5 |

DRY is a real control and a partial one: at greedy on a "write about 800
words" prompt the old run locked onto `gas-lit, gas-powered` and the
DRY-0.8 run has no repeated span of that shape at all -- but at the
artifact's own temperature it still left 4/5 runs running on (the model's
failure is not only a literal repeat; it is a distribution that will not
emit an end token). Since a penalty also switches the GLM drafts off, it
ships **off** by default: the plumbing is available, the measured benefit
is not worth a 2x decode cost unless a deployment sees loops.

The lever that did change the outcome was temperature, and only at 0. The
deployment therefore caps it (`DS4_SERVER_GLM_TEMP_MAX=0` on the Spark
launcher) rather than changing the artifact's values in the code: the cap
is a deployment decision, visible in the startup report, and one variable
to loosen (`0.7` restores sampled output at a measured 3/5 stop rate).

With the cap live (`GLM values: ... temp_max=0 ... dry=0.8x1.75^L from
L=2`, glm53unc.log:1122) the reproduced session stops:

| Deployment run | Result |
| --- | --- |
| same 13,335-token replay, client `temperature` 1.0, five seeds | **5/5 stop**, 557 tokens, 42 s wall each |
| "write about 800 words about bicycles" | 1,500 tokens, `length` -- see below |

Two things that run does NOT fix, and the owner should read them here
rather than discover them:

- **The answer it stops on is not always the answer.** In the replay the
  content is meta-commentary about how the model would source a citation
  ("I will first verify the source ... then the rustwright browser"),
  because the replayed request carries open-grok's system prompt but no
  tools. With tools present it emits tool calls instead (the live session
  did exactly that). Stopping is fixed; usefulness in an agent context is
  a separate question that only a live tool task answers.
- **A length-padding prompt still decays.** Asked for "about 800 words",
  the capped run reaches the essay length and then walks off into
  `T-tt-T-t-tt-Tt-t-...` character salad before the cap. DRY removed the
  literal repeats, not the decay; that is the quantization's damage, not a
  sampler setting.

## Limits, stated rather than implied

- **The serial path and the CLI take no penalties.** Both sample through
  `ds4_session_sample`, which has no token history. The continuous lane is
  what the server serves ("continuous rolling path=cont served=1 fallback=0"
  in the deployment log); a deployment that forces the serial lane for GLM
  gets the old behavior.
- **A penalty costs the GLM speculation.** With a penalty set, the request
  serves plain, so `tok_per_step` is ~1.0 instead of the ~2.1 the drafts give
  at temperature 0. That is why DRY ships off.
- **Greedy is deterministic, not correct.** It stops, and it can also lock
  onto a truncated-looking essay when a prompt asks for a length; DRY is the
  knob for that case.
- **It does not repair the artifact.** The weights stay lossy (14.4% relative
  MSE on the routed experts, prose PPL 4.86 -> 8.08 by its own card); this is
  a control on generation, not a quality fix.
- **The A/B is seed-sampled**, n=5 per setting, on one prompt and one
  context. It establishes that the runaway is the common case at the
  artifact's temperature and that greedy is not, and it is not a quality
  benchmark.

## Where the evidence lives

- Replay payload and outputs: `/tmp/oa_repro.json`, `/tmp/oa_repro.out`,
  `/tmp/rt_*.out`, `/tmp/g_*.out`, `/tmp/d_*.out` on the dev box.
- Server: `ds4_session_sample` (ds4.c:76914), the family lane's sample sites
  (ds4.c:65676/65821/65849), the penalty code (ds4_ds41_sample.inc:49-232).
- Artifact: the model card above; `FORMAT.md` in the artifact directory for
  the VQ layout.
