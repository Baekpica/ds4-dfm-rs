# Conditioning tests: FL2VA anchors and Ref2VA references

The port serves text-to-audio-video today. Everything that takes an image, a
clip or a soundtrack is P12, and this page is the test matrix those units must
pass. The flags are the reference's own, read from
`/data/h3.c/main.c:47-56` with the option table at `:333-338`; none of them is
invented here.

Status: `h3-rs` does not parse these flags yet, and the Spark artifact has no
Ref2VA partition. The harness exists so the cases are fixed before the code
lands, and so a skipped case says which artifact is missing instead of failing
mysteriously.

## Two families, two artifacts

| Family | Flags | Checkpoint |
|---|---|---|
| FL2VA | `--first-frame PATH`, `--last-frame PATH` | `FL2VA/` only |
| Ref2VA | `--ref-image`, `--ref-image-size match\|max`, `--ref-video`, `--ref-silent-video`, `--ref-video-audio VIDEO AUDIO`, `--ref-audio` | `Ref2VA/` required (`h3.c:872-874`) |

The FL2VA anchors work against the artifact already on the Spark
(`/mnt/h3-artifacts/MiniMax-H3/FL2VA`, 135 GiB). Ref2VA is a separate
partition (`Ref2VA/transformer`, its own tokenizer and text encoder,
`h3.c:927-930`) that is not downloaded yet, so every `ref2va-*` case is
skipped with that reason until it is.

## What the port has and what it lacks

Read from the reference, not assumed. The conditioned run is a different DiT
load (`h3_dit_load_conditioned`, `h3_dit.c:1742`) and a different text encode
(`h3_multimodal_encode_*`), so "the pipeline is ported" does not extend to it.

| Piece | Reference | Port state |
|---|---|---|
| CLI flags | `main.c:47-56`, table `:333-338` | Absent; `h3-rs` has no conditioning flags |
| Keyframe row layout | `h3_host.c:349-366` | Ported (`sequence.rs:129-160`), unused: every caller passes `keyframes: &[]` |
| Reference row layout | `h3_host.c:367-433` | Absent: `build` returns `Unsupported::References` (`sequence.rs:21-24`) |
| Media read: probe, image, video, audio at 32 kHz | `h3_ffmpeg.c:77`, `:156`, `:269`, `:392` | Absent; only the write side (`mux_mp4`) is ported |
| Vision tower (Qwen vision) | `h3_vision_encoder.c`, `h3_vision_encode_bf16` (`h3_vision_encoder.h:27`) | Absent. Weights are present: `FL2VA/text_encoder` is 63 GiB and carries them |
| Multimodal presentation and decoder | `h3_multimodal.c:15` (FL2VA), `:43` (Ref2VA) | Absent; `text_forward::encode` is the plain prompt path |
| Anchor latents (VAE encoder) | `h3_video_encoder.c:22`, `h3_video_vae_encode` | Absent; only the decoder is ported |
| Conditioned DiT graph | `h3_dit.c:1742` `h3_dit_load_conditioned`, `:2408` `h3_dit_reset_run` | Absent; `Dit::load_with` takes no condition rows |
| Condition time rows in the schedule | `dit.rs` `time_rows` (`h3_dit.c:1666-1684`) | Ported, reachable only through the sequence |
| Seeded condition augmentation | `h3.c:323-360` `h3_augment_conditions` | Absent |
| Conditioning memo key | `h3.c:143-190` | Prompt-only in the engine (`engine.rs`, `PreparedKey`) |
| Audio reference preparation | `h3_ffmpeg.c:392` plus its presentation labels | Absent |
| Ref2VA partition | `h3.c:872-874`, `:927-935` | Not on the Spark: only `FL2VA/` is downloaded |

Unit order, each with its case from the matrix above:

1. Media read (`ffprobe`, image, video, audio) and the `h3-rs` flags for
   `fl2va-first` / `fl2va-first-last`. This needs the vision tower, the
   multimodal encoder and the VAE encoder, so it is the largest unit and the
   one that unlocks the operator's cartoon pilot.
2. Conditioned DiT graph and the seeded augmentation, gated by
   `fl2va-first-last` against the reference.
3. Reference row layout, Ref2VA loader and ordered presentation, gated by
   `ref2va-image` and `ref2va-image-audio` — after the Ref2VA partition is
   downloaded.
4. Video and audio references, gated by `ref2va-video`, `ref2va-silent`,
   `ref2va-video-audio` and `ref2va-mixed`.
5. Conditioning keys in the engine memo, gated by two consecutive identical
   conditioned requests.

Nothing above is claimed as ported. The only conditioning code in the tree
today is the keyframe row layout and the schedule's condition rows, and
neither is reachable from a request.

## Rules the cases respect

- References and frame anchors cannot be combined (`h3.c:594-597`).
- At most 12 references, of which 9 images, 3 videos, 3 audio inputs
  (`h3.c:589`, `h3.c:633-636`).
- Audio alone is refused; it needs an image or video reference
  (`h3.c:638-640`).
- Reference clips are 2-15 s each, total at most 15 s (model card,
  `README.md:75`), so the derived fixture is 2.5 s.
- Every case is a quality render: `--steps 20` at 512x512, seed 42, reuse 1.
  The operator reviews the picture; no case uses a short-circuit.

## Matrix

| Case | Frames | Flags | What it proves |
|---|---:|---|---|
| `fl2va-first` | 22 | `--first-frame IMG` | The anchor latent enters the packed sequence; the clip starts on the image |
| `fl2va-first-last` | 22 | `--first-frame IMG --last-frame IMG2` | Two anchors, both ends, same layout order as the reference |
| `ref2va-image` | 22 | `--ref-image IMG` | Multimodal image encoding, `match` sizing |
| `ref2va-video` | 22 | `--ref-video CLIP` | Clip decode plus its embedded soundtrack |
| `ref2va-silent` | 22 | `--ref-silent-video CLIP` | The same clip with the audio path omitted |
| `ref2va-video-audio` | 56 | `--ref-video-audio SILENT WAV` | Explicit replacement soundtrack |
| `ref2va-image-audio` | 56 | `--ref-image IMG --ref-audio WAV` | Ordering across modalities |
| `ref2va-mixed` | 56 | `--ref-image IMG --ref-silent-video CLIP --ref-audio WAV` | Ordered mixed stack |

`CLIP` and `WAV` are derived from a source clip, so the tree stores one small
PNG and nothing else. `IMG` is the operator's pilot,
`tests/fixtures/h3/costello-crocodile-2panel.png`; `IMG2` is its mirror.

## Harness

`tools/minimax_h3/ref_gate.sh` derives the fixtures, runs the selected cases
against one binary and writes a manifest with the MP4 digest, the decoded
frame digest and the audio digest:

```sh
tools/minimax_h3/ref_gate.sh \
  --binary ~/ds4-h3-port/h3-rs \
  --model /mnt/h3-artifacts/MiniMax-H3 \
  --clip /mnt/h3-artifacts/h3-fox-512-20.mp4 \
  --image tests/fixtures/h3/costello-crocodile-2panel.png \
  --out ~/ref-gate-port

# the same cases against the reference engine, then compare
tools/minimax_h3/ref_gate.sh --binary ~/h3-oracle/h3 ... --out ~/ref-gate-ref
tools/minimax_h3/ref_gate.sh --binary ~/ds4-h3-port/h3-rs ... \
  --out ~/ref-gate-port --compare ~/ref-gate-ref
```

`--case NAME` runs one case, which is how a unit is gated while the rest are
still unported.

## Acceptance

1. Both binaries exit 0 and write a 512x512 MP4 with the requested frame count
   and 32 kHz stereo audio.
2. The decoded frame digest and the audio digest are identical between the
   port and the reference for the same case and flags. The MP4 digest is
   compared too and is expected to match; a container-only difference is
   reported, not hidden.
3. The operator reviews the 20-step output of every case. Quality is the
   operator's call, not a hash.
4. A skipped case names the missing artifact; it never counts as a pass.

## Not covered here

`--ref-image-size max` is a second sizing policy on the same image case; it is
measured after `match` passes. Reference video/audio longer than the fixture,
and the 9-image/3-video/3-audio ceilings, are limit tests for a later unit.
