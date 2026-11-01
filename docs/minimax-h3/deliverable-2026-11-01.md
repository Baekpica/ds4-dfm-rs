# The deliverable, and the draft it is measured against (2026-11-01)

Phase F. The campaign's product line is *the draft, enhanced, never replaced*:
stage 1 draws a clip, the latent path doubles it, and the refiner adds detail to
the picture it was given (`--stage2-anchor 0.7`). This is that pipeline as it
stands today, measured end to end on the Spark, with the draft beside it at 1:1.

**Result: 76.0 s for a 5.04 s 1344x768 clip with its soundtrack**, against
248.5 s before Phase D. The refined picture keeps the draft's animal and motion
- SSIM 0.957-0.960, PSNR 26.3-28.9 dB at five timestamps - which for a pipeline
whose whole job is to sharpen is the size an enhancement should be, not a
regression. The audio is byte-identical between the two clips.

Arms: `H3_LTX_ATTN_MMA=1 H3_SOL_MMA=1`, anchor 0.7, release sigmas. Both arms
are still switches; a `--stage2-attn` flag is owed before they are the default
(§6).

## 1. The route and what each stage cost

One process, the release's own geometry, the same draft both runs got
(`/mnt/h3/recipe.bin`, sha256 `2cb3e8d2...`):

| stage | seconds |
| --- | ---: |
| draft latent load (1.79 MB, 124 frames of 24x37x1008) | 0.256 |
| latent path: upscaler, adapter, crop to 16 frames | 2.860 |
| audio state: log-mel, audio VAE encode, conform | 1.156 |
| refiner load, LoRA merge, process setup (not logged) | ~18.1 |
| `noised_state` (anchor 0.7) | 0.000 |
| denoise step 0 / 1 / 2 | 11.943 / 11.805 / 11.742 |
| **denoise** | **35.490** |
| LTX VAE decode, 121 frames of 1344x768 | 15.130 |
| mux (x264 CRF 19, AAC 192k) | 0.594 |
| **whole window, host guard** | **76.017** |

The 18.1 s is inferred from the pair rather than logged: the draft-only run, which
never loads the refiner, spends 21.207 s in the guard window against 18.787 s of
logged stages, and this run spends 76.017 s against 55.486 s. It is the 34.58 GiB
weight file plus the LoRA merge. **That is what a resident route amortizes** -
Phase E's whole value, and it is per process, so it does not touch a single run.

Memory, from the guard's own log: the draft-only run's minimum available was
101.72 GiB, this run's 61.77 GiB, both starting from ~119 GiB and both recording
PSI-full 0.000 throughout.

## 2. The draft beside it, at 1:1

The control is `--transfer-only`: the same upscaler, adapter and LTX VAE decode
as the deliverable, with the refiner switched off. Same canvas, same 121 frames,
same soundtrack - so the difference between the two clips is the refinement and
nothing else.

| t | SSIM | PSNR |
| --- | ---: | ---: |
| 0.5 s | 0.9590 | 28.73 dB |
| 1.5 s | 0.9599 | 28.89 dB |
| 2.5 s | 0.9582 | 28.28 dB |
| 3.5 s | 0.9586 | 27.31 dB |
| 4.5 s | 0.9570 | 26.25 dB |

Read as what they are. SSIM ~0.96 with PSNR in the high twenties is the
signature of *same structure, different texture*: an enhancement that adds
detail moves every high-frequency sample, so a 48 dB figure here would mean the
refiner had done very little. The number drifts down 2.5 dB along the clip
because that is where the draft is softest - the snow texture and the fox's fur
are what the refiner has to invent, and 4.5 s is the frame with the most of it.

**This is not the class of number the plan carried.** `PSNR 48.80 / SSIM 0.9910`
belongs to a same-route code-change gate - two clips from the *same* route, one
variable apart - and this campaign has one of those too, freshly measured: the
sparse tensor-core arm against the dense arm, 48.13-48.55 dB and 0.9917-0.9925
across the same five timestamps (`sol-mma-2026-11-01.md` §5). Both gate classes
now exist in the tree and neither replaces the other: §2's says the refinement
still does the job it is for, and the arm gate says the last change did not move
the picture.

## 3. The soundtrack

Both clips carry the same audio, and it is not merely equal in level -
`cmp` of the two decoded F32 streams is clean, byte for byte:

| | mean volume | max volume | RMS |
| --- | ---: | ---: | ---: |
| source `recipe.pcm` (F32, 32 kHz, 2 ch, 5.175 s) | -60.2 dB | -34.5 dB | 0.00098 |
| `final-draft.mp4` (AAC 192k) | -60.4 dB | -44.1 dB | 0.00095 |
| `final-refined.mp4` (AAC 192k) | -60.4 dB | -44.1 dB | 0.00095 |

The source's -34.5 dB maximum is **the last frame of the file**: the peak sits at
interleaved value 331,198 of 331,200, at -0.0189 while the program around it
sits at -44 dB. It is the file ending mid-waveform, not a transient in the clip,
and the mux does not reproduce it - the encoded peak is the program's own
-44.1 dB, at 0.97 s.

So the honest statement of the level: **the H3 soundtrack for this prompt is
quiet ambience, about -60 dB mean and -44 dB peak**, carried faithfully through
the mux. If the operator expects an audible soundtrack, that is a stage-1 audio
property to look at on its own; it is not a mux defect, and the refinement
cannot touch it - the two clips' audio streams are identical.

## 4. The frames

`docs/minimax-h3/frames/deliverable-2026-11-01.png` - draft left, deliverable
right, 2.5 s. Same fox, same pose, same framing; the deliverable's edges and
contrast are the difference the SSIM table describes.

`docs/minimax-h3/frames/deliverable-strip-2026-11-01.png` - the deliverable at
0.5 / 1.5 / 2.5 / 3.5 s. The animal walks left to right through the frame,
noses the snow at 1.5 s and turns back at 3.5 s, one coherent fox through the
whole clip. That is the acceptance the operator asked for - *enhancing and never
changing* - and it is a picture judgement, which is why the strip is in the tree
beside the numbers.

## 5. Reproduce

```sh
bash scratch/final-arms.sh     # both clips, both guard logs
bash scratch/final-measure.sh  # the tables above and both figures
```

The two commands, with the artifacts as they are on the Spark:

```sh
./h3-sol-rs --latent recipe.bin --audio recipe.pcm --upscaler UPS --adapter ADP \
  --refiner REFINER --vae VAE --context CACHE --lora LORA --lora-strength 0.8 \
  --audio-vae AVAE --hash --transfer-only -o outputs/final-draft.mp4

env H3_SOL_MMA=1 H3_LTX_ATTN_MMA=1 ./h3-sol-rs ... --sol --stage2-anchor 0.7 \
  -o outputs/final-refined.mp4
```

## 6. What this does not claim

- **No QA PASS.** There is no `tests/qa-gate.sh` for this workstream and none is
  invented here; the campaign's gates are the ones named above.
- **The arms are switches, not the deliverable default.** `H3_LTX_ATTN_MMA` and
  `H3_SOL_MMA` are environment variables; a deliverable that uses them without
  the caller knowing is the wrong shape. A `--stage2-attn tensor` flag (and the
  bit-exactness decision it implies, since the one-block golden is pinned to the
  scalar kernels) is still the operator's call.
- **The draft is the campaign's, not re-rendered here.** Both clips got the same
  `recipe.bin`; this report measures the refinement, not stage 1. Stage 1's own
  cost and quality are in `recipe-run-2026-11-01.md`.
- **One clip, one seed.** The 0.7 anchor was chosen on this draft and confirmed
  at `--seed 7` (`subject-drift-2026-11-01.md`); a second prompt was not run.

## 7. Artifacts

- Clips: `outputs/final-draft.mp4`, `outputs/final-refined.mp4` on the Spark.
- Logs: `scratch/final-arms.log`, `scratch/final-draft.memory.jsonl`,
  `scratch/final-refined.memory.jsonl`.
- Figures: `docs/minimax-h3/frames/deliverable-2026-11-01.png`,
  `docs/minimax-h3/frames/deliverable-strip-2026-11-01.png`.
- Scripts: `scratch/final-arms.sh`, `scratch/final-measure.sh`.
- Code that produced it: `cf6bb201` (`feat/sol-h3`), the two arm switches and
  anchor 0.7 as described in `sol-mma-2026-11-01.md` and
  `ltx-attn-mma-2026-11-01.md`.
