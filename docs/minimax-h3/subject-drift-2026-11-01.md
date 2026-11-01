# Stage 2 changes the animal, and the mask is why (2026-11-01)

Status: **fixed by `--stage2-anchor`**, measured on the Spark; the release-
faithful alternative (first-frame conditioning) is priced at the end and not
built.

## 1. The symptom

`outputs/recipe-short.mp4`, the release's own recipe end to end: the draft is a
red fox - `outputs/id-transfer-2.5.png`, the same latent decoded through the
upscaler and adapter with the refiner barely moving (`--sigmas 0.001,0`), shows
the fox at 2.5 s - and by 2.5 s of the refined clip the animal is a dhole:
broad muzzle, short rounded ears, no white chin, a tail with no brush.
`outputs/recipe-structured.mp4` kept the fox because its draft was
anatomically sound; the same refiner kept it too (`outputs/id-sol.mp4`, a fresh
draft from the same short prompt and seed 42, is a fox at 2.5 s).

`docs/minimax-h3/frames/subject-drift-2.5s-2026-11-01.png` is the same frame
three ways: the draft's transfer decode, the release's route, and the anchored
route.

So the subject survives when the draft is unambiguous and is redrawn when it is
not, under the same code, schedule, LoRA and prompt. That is a property of the
route, not of one clip.

## 2. The mechanism, read from the release

Stage 2's video denoise mask is one in every route the release ships
(`runtime/stage2.py`, `Stage2Resolver`'s `prepare_multimodal_states` inherits
it from LTX-2's two-stage family and only overwrites the *audio* mask to zero,
`runtime/stage2.py:96-99`). The mask is read three times, and each read is the
same statement:

- `GaussianNoiser.__call__` (`components/noisers.py:30-37`) lerps the clean
  latent toward the draw, so at the release's first sigma the state is
  `0.090625 * draft + 0.909375 * noise`;
- `timesteps_from_mask` hands the blocks `mask * sigma` per token
  (`transformer_args.py:177-181`), so the model is told 909/1000;
- `post_process_latent` (`utils/helpers.py:461-463`) blends
  `mask * denoised + (1 - mask) * clean`, which at one contributes nothing
  from the draft at all.

Nine percent of the draft, once, at the noisiest step: that is the route's whole
draft input. Everything after it is the model's own velocity, so what the animal
becomes is decided by the model's prior plus nine percent of a latent. A draft
whose anatomy is already contradictory at that signal level gets resolved into
whatever the prior prefers.

## 3. Ruled out on the same draft, one variable at a time

| arm | what changed | step 0 | 2.5 s animal | lapvar | SSIM vs transfer |
| --- | --- | ---: | --- | ---: | ---: |
| `id-transfer` | `--sigmas 0.001,0` | 129.7 s | the draft's fox | 7.70 | 1.000 |
| `id-sol` | release, Sol on | 44.6 s | fox, heavier, mottled | 12.53 | 0.803 |
| `id-dense` | release, Sol off | 131.2 s | fox, heavier | 11.03 | - |
| `id-anchor-0.7` | mask 0.7 | 44.8 s | the draft's fox, sharpened | 9.99 | 0.901 |
| `id-anchor-0.5` | mask 0.5 | 45.7 s | the same fox, softer | 8.36 | 0.949 |
| `id-anchor-0.3` | mask 0.3 | 44.7 s | the draft, barely touched | 7.88 | 0.971 |

Dense attention draws the same animal as Sol (11.03 against 12.53 on the same
frame), so the port's own sparse routing is not what changes the subject: the
comparison that had never been run closes the last unvalidated stage-2 kernel
as a cause. The block, the sigma-to-timestep path, the text context and the
transfer were closed earlier (`recipe-run-2026-11-01.md` §2).

Identity across the clip, SSIM against the transfer decode at four timestamps:

| arm | 0.5 s | 2.0 s | 2.5 s | 4.0 s |
| --- | ---: | ---: | ---: | ---: |
| release | 0.762 | 0.786 | 0.803 | 0.814 |
| anchor 0.7 | 0.906 | 0.904 | 0.901 | 0.909 |

The drift is uniform along the clip, so it is the route, not one frame.

## 4. The fix: a video denoise mask the caller sets

`--stage2-anchor M` writes the same three reads with `M` instead of one:

- the state starts at `clean + M * (noise - clean)`, which is a coherent
  diffusion state at timestep `M * sigma`;
- the blocks are handed `M * sigma` per token, so they are told the truth about
  how noised the state is;
- `post_process_latent` blends `1 - M` of the clean draft back after every
  step, so the picture cannot drift away from the draft between steps.

`M = 1` is the release's route unchanged, byte for byte: the mask tensor is only
written when the flag names another value. `M = 0` is the release's own frozen
audio treatment, where the model reaches nothing, which is the limit that shows
the knob is the same statement the release already makes elsewhere.

Cost: nothing measurable. Step times at anchor 0.7 are 44.808 / 40.402 / 36.813
s against 44.577 / 40.446 / 37.033 s at anchor 1.

Robustness, the seed test (`--seed 7`, same draft, same schedule):

| arm | lapvar | SSIM vs transfer |
| --- | ---: | ---: |
| release, seed 7 | 12.60 | 0.805 |
| anchor 0.7, seed 7 | 9.89 | 0.900 |

Seed 42 and seed 7 land within 0.001 of each other at anchor 0.7, so the draw
moves texture only; at anchor 1 it is free to redraw the animal.

## 5. What this costs and what it does not

The anchor trades sharpness for identity on a straight line: 7.88 at 0.3,
8.36 at 0.5, 9.99 at 0.7, against 7.70 for the transfer decode and 12.53 for the
release's route. 0.7 is 1.30x the transfer's Laplacian variance with the draft's
animal; the release's 12.53 is 1.63x with a different animal. The product
requirement is "enhance, never change", which is the 0.7 point of that line.

## 6. The release-faithful alternative, priced and not built

`super_acceleration/stage2/official_compat_h3_refiner_diagnostic.py` restores
the parts of LTX-2.5's official two-stage Stage 2 that the shipped runtime omits
- "a full-resolution first-frame conditioning encoded by the LTX-2.5 Video VAE"
and the joint audio stream - and ships a `skip_0p9` schedule beside the default.
Both are the upstream authors' own answer to the same problem: at 0.909375 the
route redraws. First-frame conditioning would anchor the picture properly and
is a real port (the LTX video VAE encode of frame zero, the transformer's image
conditioning tokens, a second context on the request), so it is not in this
commit. The anchor delivers the same guarantee with the graph the port already
has, and the two are not exclusive: the anchor stays useful under image
conditioning for clips whose *later* frames drift.

`docs/minimax-h3/frames/anchor-before-after-2026-11-01.png` is the same draft
through both masks at 2.5 s and 4.0 s: the release's one draws the heavier
animal, the anchor draws the draft's fox. Stage 1 is deterministic, so the
recipe's own draft (`/mnt/h3/recipe.bin`, sha256 `2cb3e8d2...`) is the same
latent in `id-sol.mp4` and in `recipe-anchored.mp4`; only the mask differs.

## 7. Where the evidence lives

Spark, `~/ds4-sol-h3/outputs/`: `recipe-anchored.mp4` (the whole recipe at
anchor 0.7: stage 1 76.448 s, stage 2 172.027 s, 248.478 s total),
`id-transfer.mp4`, `id-sol.mp4`,
`id-dense.mp4`, `id-anchor-0.7.mp4`, `id-anchor-0.5.mp4`, `id-anchor-0.3.mp4`,
`id-seed7-release.mp4`, `id-seed7-anchor.mp4`, with four frames each
(`-0.5`, `-2.0`, `-2.5`, `-4.0`). Scripts: `scratch/identity-arms.sh`,
`scratch/anchor-arms.sh`, `scratch/seed-arms.sh`; logs beside them.
