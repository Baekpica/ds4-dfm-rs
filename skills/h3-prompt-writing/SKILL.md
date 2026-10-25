---
name: h3-prompt-writing
description: Write MiniMax-H3 video prompts in the structure the checkpoint was trained on: integrated_multimodal_description, overall_soundscape and non_diegetic_music for T2VA/FL2VA/I2VA, or the six Ref2VA sections with ordered reference labels. Use before generating a clip whenever the prompt must carry a timeline, shot alignment, reference labels, dialogue or sound.
---

# H3 prompt writing

Plain text plus two local guide files: no API, no account and no
harness-specific tooling. The frontmatter above is the skill format open-grok
and Claude Code read; `AGENTS.md` beside this file is the instruction-file form
for Codex.

The generation side is the `h3-video` skill. This one only decides the text.

## 1. Read the guide for the mode

Two files ship with the reference and are the authority for the field names,
section order and labels. Read the one for the request:

- base modes (T2VA, I2VA, FL2VA, L2VA):
  `/data/MiniMax-H3/skills/h3-prompt-writing/references/base-en.txt`
- full-reference mode (Ref2VA):
  `/data/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt`

Do not paraphrase the structure from memory: the field names are literal and
the checkpoint reads them in order.

That tree is a separate checkout, not part of this repository: if
`/data/MiniMax-H3` is missing, restore it from the upstream MiniMax-H3
repository before writing a prompt, and say that the guide was not available
rather than inventing the field names.

## 2. Map the request to a served task

| Wanted | Send |
| --- | --- |
| text only | `task: t2va` |
| first frame only (I2VA) | `task: fl2va`, one keyframe at `frame_index: 0` |
| first and last frame | `task: fl2va`, keyframes at 0 and 1 |
| last frame only (L2VA) | `task: fl2va`, one keyframe at `frame_index: 1` |
| ordered image/video/audio references | `task: ref2va` |

`i2va` and `l2va` are not task values on this server; the two rows above are
how the reference's own local scripts express them
(`scripts/readme/full-2k-i2va-h3-base.sh:9-16`).

## 3. Base modes: the three fields

T2VA, I2VA and FL2VA prompts are built from three named fields, in this order
(`references/base-en.txt`, section 2):

1. `integrated_multimodal_description` — the audiovisual timeline.
2. `overall_soundscape` — the ambient and physical sound of the scene.
3. `non_diegetic_music` — music the characters do not hear.

Anchored modes add an alignment instruction before them, spelling out which
picture lands at which second of the target clip (the reference guide gives
the exact sentence for I2VA, FL2VA and L2VA).

## 4. Ref2VA: six sections

`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`, in that
order (`references/ref-en.txt`). Keep reference labels identical in every
section.

Labels follow request order and the media kind:

| Condition in the request | Label |
| --- | --- |
| `type: image` | `<Picture n>` |
| `type: video` | `<Video n>`, with one `<t.t seconds>` per sampled block |
| `type: audio` | `<Audio n>` |

A video that carries its own soundtrack gets both an `<Audio n>` and a
`<Video n>` label. Numbering is per kind, not across kinds: `<Picture 1>`,
`<Picture 2>` count images only, `<Video 1>`, `<Video 2>` count videos only,
`<Audio 1>`, `<Audio 2>` count soundtracks only. One image plus one video are
`<Picture 1>` and `<Video 1>` (`h3_multimodal.c:290-320`).

## 5. Rules that keep a prompt usable

- Match the description to the requested duration; the timeline must end where
  the clip ends.
- Name the composition, subject, environment, action, camera and sound; avoid
  "cinematic", "beautiful" and other unfalsifiable adjectives.
- Keep dialogue, lyrics and visible scene text in their original language;
  write the surrounding description in English.
- Never leave a reference label unresolved, and never label a reference the
  request does not carry.
- For an anchored clip, say what happens between the frames rather than
  describing the frames again.

## 6. Then generate

Pass the finished text as the `prompt` field with the `h3-video` skill. The
prompt cap is 32000 characters; the body adds the anchors and references.
