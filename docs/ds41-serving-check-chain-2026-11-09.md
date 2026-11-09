# DS4.1 serving-check chain gate (2026-11-09)

Question: the V4.1 `reuse/continue` checklist item gated only the frontier
token and reported the rest of the greedy chain. Can the chain itself be
gated, without failing on the FP difference a checkpoint resume introduces?

## Why a floor is needed

A hit resumes at the checkpoint's position (the prompt end), so its suffix
prefills from there while a cold run chunks from zero.
`ds4_ds41_kvreuse.inc:16-19` names that as an FP-level difference, the same
class as changing `--v41-chunk`. A greedy chain turns one near-tie flip into
a different continuation, so the old item compared only the first token.

That class can be measured instead of assumed: the same cold prompt run at
two prefill blocks yields the band a block boundary moves a row's top-1
lead. The hit may flip only a decision inside that band.

## Method

Two seams, both on existing engine surfaces:

1. Per-row logits. `DS41_GEN_LOGITS=<path>` (native diagnostic,
   `ds4_ds41_draft.inc`) writes one `[n_vocab]` f32 row per emitted
   position, in emission order. The checklist sets it to a scratch path per
   run, reads the rows back, and restores the caller's value.
2. A narrower prefill block. `--v41-chunk` was a recorded serving gap; the
   native now has `ds4_engine_v41_set_chunk` behind the bridge and a
   `V41RunOptions.chunk`. It narrows only (the engram feed's row capacity is
   sized from the default at open, `ds4_v41_feed_open`), 0 = 2048. The check
   runs the second cold reference at 1024; the serving path keeps 0.

The gate (`chain_gate`, `crates/ds4-core/src/serving_check.rs`):

- floor = the largest change of a row's top-1 lead between the two cold runs,
  over the rows whose prefix both still share (`chain_floor`); after their
  first token divergence a row is a different prompt, not the same decision.
- the hit's row must pick the reference's top-1, or a token whose lead is
  inside the floor; the first such flip ends the comparable chain, and
  anything else fails the item.
- the row counts must match: a different length is a different stopping
  decision.

The checklist also splits on the family's 2048 block now (`V41_BLOCK`), so
the prefix item keeps a whole block and its stream is exact. Before that it
split at half the prompt, off the grid, and the prefix item failed on the
resulting arithmetic change (14/16), not on reuse.

## Results

One window on the Spark, MTP off (`dspark: Some(0)`), prompt 5037 tokens from
`docs/deepseek41-port-plan.md`, binary `15c35429e43decee7d06b35f83abc13026`
(scratch tree `~/ds4-check-l5`, `make ds4 CUDA_ARCH=sm_121`). The live server
(`139aef25...`) was restored afterwards and answers on 8008.

Before the grid fix (`/tmp/l5/check.log`):

| item | result | detail |
|---|---|---|
| reuse/prefix | WRONG | kept 2518, tokens 14/16 |
| reuse/continue | works | kept 5052, rows 16/16 agree (floor 0.526) |
| disk/session | works | kept 5037, tokens 16/16 |

Final (`/tmp/l5/check2.log`):

| item | result | detail |
|---|---|---|
| reuse/prefix | works | kept 4096, tokens 16/16 |
| reuse/rewind | no | prefilled from zero: nothing kept |
| reuse/continue | works | kept 5052, rows 16/16 agree (floor 0.526 from 16 rows) |
| disk/session | works | kept 5037, tokens 16/16 |

`result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)`.

The 1024-block run is visible in the log as `[v41] pos 0+1024 ...`, the second
cold arithmetic.

## Reading

- The hit chain agreed with the cold reference at every emitted row on this
  prompt, so the gate's new power is not exercised here: it is the floor that
  makes a future flip reportable instead of silently accepted. The old item
  would have passed any chain divergence; the new one fails an unexplained
  one.
- The prefix item's WRONG was the checklist's own split, not reuse. With the
  split on the 2048 grid the kept prefix repeats the fresh block arithmetic
  and the stream is exact (16/16). This was a real checklist defect for the
  family: `grid` was 0, so every split was half the prompt.
- `rewind` stays cold: the divergent filler means T does not start with the
  kept prefix, so the engine correctly prefills from zero. That is the item's
  designed outcome for V4.1, not a failure.
- Cost: one extra cold prefill of the prompt per checklist run (the floor
  reference) and one full logits readback per emitted row on the dump path,
  both inside the checklist only.

## Limits

- One prompt, one artifact, one window. The floor (0.526) is this prompt's
  band; it is measured per run, not a constant of the model.
- Rows past the reference pair's first token divergence are not counted in
  the floor (`from 16 rows` says the pair agreed on all 16 here).
- `--v41-chunk` is a native and bridge knob; no CLI or serving flag exposes
  it yet. The serving path keeps the build's 2048.

## Evidence

Spark: `/tmp/l5/check2.log` (final, result pass), `/tmp/l5/check.log` (first,
result FAIL, prefix 14/16), `/tmp/l5/build2.log`, prompt
`/tmp/l5/prompt.txt`, scratch tree `~/ds4-check-l5` (binary `15c35429...`).
Live server restored: `139aef25...`, answering on 8008.
