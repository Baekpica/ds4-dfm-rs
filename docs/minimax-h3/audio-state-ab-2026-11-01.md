# Frozen audio state: the A/B fails on picture

Date: 2026-11-01. Branch: `feat/sol-h3`, tree `d48df326`. This is the phase-B
acceptance receipt its own plan asked for, and it is a **fail**: the audio
state is wired, live and measurable, and it destroys the picture.

## Setup

The Spark (`gx10-e1d2`, driver 580.178.04, boot `122818b0`) was hard-synced to
the pushed tip: the three files copied over the tree in the previous session
were compared with the commit first (`ltx_audio_vae.rs` and
`h3_ltx_audio_gate.rs` identical, `jobs.rs` differing only in rustfmt
wrapping), moved aside, and `git reset --hard fork/feat/sol-h3` left the tree
clean at `d48df326`. `make h3-sol-rs` rebuilt it, and `nm` on
`native/h3/h3_dit_gpu.o` shows `h3a_pad_zero` and `h3a_pixel_norm` present, so
the object matches the commit rather than the previous session's copy.

Both arms consume the same stage-1 pair, `scratch/accept.bin` and
`scratch/accept.pcm` (672x384 at 124 stage-1 frames, seed 42, the accept
route's own inputs), the release schedule, the distilled LoRA at 0.8 and
`--sol`. They differ in exactly one flag: the control runs `--no-audio-state`,
the treatment encodes the PCM with
`/mnt/h3-artifacts/LTX-2.5/ltx-2.5-audio-vae-bf16.safetensors` and holds that
latent plus the cached audio prompt on every step. Stage 2 renders 1344x768 at
121 frames.

No GPU clock service is installed, by operator decision this session: "we dont
need to change the driver settings just continue". The box therefore ran
uncapped, and the samples the run took (every 5 s, `scratch/ab-clock.txt`,
67 samples) are the record: SM 208-2450 MHz, mean 2373; under load (>80%
utilization, 51 samples) mean 2405 MHz, peak 100.82 W, peak 80 C. `nvidia-smi
-lgc 3003` as root had already been refused (rc=4, "does not have permission to
change clocks"), and the device reports no supported-clock table, so no cap
could have been applied here anyway.

## Measured

| | control (`--no-audio-state`) | audio state on |
|---|---|---|
| encode the state | - | 1.174 s, 8x126x16, mean -0.22164, std 0.82102 |
| step 0 / 1 / 2 | 40.679 / 36.574 / 33.167 s | 44.932 / 40.492 / 36.859 s |
| denoise | 110.420 s | 122.283 s |
| LTX VAE decode | 15.597 s | 15.212 s |
| wall (one process, cold page cache for the first) | 174.812 s | 160.053 s |
| frames FNV | `8a0dbea4252603fb` | `d5fa3d0c00cfbbe3` |
| H.264 bitrate at one CRF | 3.34 Mbit/s | 7.57 Mbit/s |

The control arm reproduces the previous session's ref20 arm to 0.2 s per step
(40.898 / 36.598 / 33.046 s then 15.690 s of decode), which is what says this
box, this tree and this input are the ones the earlier numbers were taken on.

Two facts about the audio arm: the state is **live** (the frames hash moves,
and the denoise grows by 11.863 s, i.e. +4.25 / +3.92 / +3.69 s per step for
the audio self-attention, the two AV attentions and the audio feed-forward),
and the picture **collapses**. Every frame of `outputs/ab-audio.mp4` is a
smooth brown smear on snow - no fox, no paws, no texture - while
`outputs/ab-control.mp4` from the same latent, seed and schedule shows the fox.
This is not a subtle shift: it is the second frame of `ab-audio.mp4` at
`t = 0`, `t = 2.5` and `t = 5.0` being the same blob.

## What the instrument measured

`H3_STAGE2_TRACE=1` (`ltx_stage2.rs`, one `println!` per stream per step) was
added for this diagnosis and both arms were re-run with it
(`scratch/trace-rows.log`). Four things it settled:

- **The video state into the blocks is bit-identical between arms**: `in-video
  rows=16128 width=128 mean=-0.00023 rms=0.78651 max=19.12500` in both. The
  audio state does not touch the video latent before the blocks run, so
  `prepare`'s video half is not the fault.
- **The video's own RoPE is bit-identical**: `video-self-cos rows=16128
  width=2048 row_mean_min=-0.07252 row_mean_max=0.12550 distinct_rows=7356` in
  both arms. The audio path does not corrupt the video positional tables, and
  the video self-attention sees the same positions it sees with audio off.
- **The audio state is real data, varying over frames**: `in-audio rows=126
  width=128 mean=0.01770 rms=1.01021 max=26.0`, `in-audio-rows ... distinct_rows=126`,
  `audio-self-cos ... distinct_rows=126`. Its prompt context is real too
  (`prompt-cache.pt`'s `contexts.audio` is `[1, 1024, 2048]` BF16, std 0.9997,
  max 15.81, no zero rows).
- **Neither stream explodes**: after the 48 blocks the video state is `rms
  56.79996` with audio on against `rms 56.08978` without, and the step's video
  latent is `rms 0.75397 mean=0.00067` against `rms 0.76652 mean=0.00403`.
  Magnitudes are preserved in both arms.

So the fault is inside the A2V/V2A block path, and it corrupts the video's
*spatial* structure while leaving every aggregate statistic - mean, rms, max,
per-row spread of the positional tables - in the range the control produces. A
spatially uniform injection into all 16128 video tokens, repeated over 48
blocks and three steps, is what that looks like, and A2V is the only path from
the audio state into the video state
(`ltx_refiner.rs:700-760`, reference `transformer.py:328-397`).

The wiring was re-read and matches the reference line for line: the four
modulated operands are formed from the pre-A2V state, the A2V query comes from
the video and its context from the audio, the gate is table row 4 plus the
`cross_gate_adaln` value at the *other* modality's sigma (`transformer_args.py:388-410`),
and the gate is applied as `state + attention * gate` with no `tanh`
(`transformer.py:328-397`). A structural read is therefore not enough to find
it; the next step is a golden.

## What the next unit needs

The campaign's own method: a golden from the release's code, on one block, with
the audio state present. `h3_ltx_refiner_gate` already isolates one block
(`apply_one`) and compares dumps (`--expect`, `--dump`), and
`h3_ltx_refiner_gate --signals` dumps `edges-audio-state`,
`edges-audio-modulation`, `edges-audio-cross-scale-shift` and the video's two
cross tables. What is missing is the reference side: a torch probe over the
same synthetic inputs (the gate's `lcg` draws), one block, audio on, dumping the
post-A2V video state and the post-V2A audio state. A block-0 diff will name the
operand - query, context, gate or RoPE - that differs.

Until that diff passes, `--audio-vae` stays out of every shipped route.
`--no-audio-state` is bit-identical to the previous behaviour
(`8a0dbea4252603fb`, the same hash this tree produced before the audio state
existed), and the flag is now described as unfinished in `h3-sol-rs --help`.

## Artifacts

On the Spark, `~/ds4-sol-h3/`: `outputs/ab-control.mp4`,
`outputs/ab-audio.mp4`, `outputs/trace-control.mp4`,
`outputs/trace-audio.mp4`, `scratch/ab-run.log`, `scratch/ab-clock.txt`,
`scratch/trace-rows.log`, `scratch/ab-control.memory.jsonl`,
`scratch/ab-audio.memory.jsonl`, `scratch/ab-prep1.log`,
`scratch/ab-prep2.log`. Scripts: `scratch/ab-run.sh`, `scratch/trace-both.sh`,
`scratch/ab-prep1.sh`, `scratch/ab-prep2.sh`, `scratch/ctx-check2.sh`.
