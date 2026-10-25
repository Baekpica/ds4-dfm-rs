# H3 prompt writing — Codex instructions

Same job as `SKILL.md` beside this file, for a harness that reads `AGENTS.md`.

Read the guide for the mode before writing:

- base modes (T2VA, I2VA, FL2VA, L2VA):
  `/data/MiniMax-H3/skills/h3-prompt-writing/references/base-en.txt`
- full-reference mode (Ref2VA):
  `/data/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt`

The field names and their order are literal; do not paraphrase them.

That tree is a separate checkout, not part of this repository: if
`/data/MiniMax-H3` is missing, restore it from the upstream MiniMax-H3
repository before writing a prompt, and say the guide was unavailable rather
than inventing the field names.

## Mode to task

| Wanted | Send |
| --- | --- |
| text only | `t2va` |
| first frame (I2VA) | `fl2va`, keyframe at `frame_index: 0` |
| first and last | `fl2va`, keyframes at 0 and 1 |
| last frame only (L2VA) | `fl2va`, keyframe at `frame_index: 1` |
| ordered references | `ref2va` |

## Base modes

Three fields in this order: `integrated_multimodal_description`,
`overall_soundscape`, `non_diegetic_music`. Anchored modes put an alignment
sentence first, naming which picture lands at which second.

## Ref2VA

Six sections in this order: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Labels: `<Picture n>`, `<Video n>`, `<Audio n>`, numbered
per kind and identical in every section — `<Picture 1>` and `<Video 1>` coexist
in one request.

## Keep it usable

- Match the timeline to the requested duration.
- Name composition, subject, environment, action, camera and sound; drop
  "cinematic" and other unfalsifiable adjectives.
- Dialogue, lyrics and on-screen text stay in their original language; the
  description stays English.
- Never label a reference the request does not carry.

Generate with `h3-video/AGENTS.md`.
