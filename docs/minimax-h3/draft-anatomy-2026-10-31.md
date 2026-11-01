# The draft's anatomy: what holds it (2026-10-31)

Phase 1 of `docs/minimax-h3/plan-2026-10-31.md`. Same model, prompt family,
sparsity and seed, one variable per arm, all on the Spark at 672x384x124,
decoded with the H3 video VAE (`H3_VAE_ATTENTION=tensor`,
`H3_VAE_GEMM=bf16`) and checked at 0.5, 1.0, 2.0, 3.5 and 4.5 s.

## 1. The harness bug, and the arm set

The earlier harness exported `H3_DIT_SOL=` (empty) for its dense arms, and the
engine refuses that as a sparsity (`h3-rs: unsupported: H3_DIT_SOL= is not a
sparsity`), so dense 8 and dense 16 never ran; only `vsa8` was measured. Fixed
with `env -u H3_DIT_SOL`. A fixed seed is byte-exact: the seed-42 4-step arm
re-rendered to the same sha256 (`c7e18983...`), so every difference below is
the variable and nothing else.

| arm | seed | sparsity | prompt | steps | denoise s | lapvar 2.0 s | 0.5 s |
|---|---|---:|---|---:|---:|---:|---|
| vsa4 | 42 | 0.9 | short | 4 | 21.742 | 33.26 | head buried |
| dense4 | 42 | none | short | 4 | 28.585 | 25.99 | clean |
| vsa4p | 42 | 0.9 | structured | 4 | 22.147 | 33.52 | clean |
| vsa4s43 | 43 | 0.9 | short | 4 | 22.028 | 45.46 | clean |
| dense4s43 | 43 | none | short | 4 | 29.051 | 30.82 | clean |
| vsa4ps43 | 43 | 0.9 | structured | 4 | 22.763 | 31.76 | clean |
| vsa4s44 | 44 | 0.9 | short | 4 | 21.662 | 34.27 | clean |
| vsa4s45 | 45 | 0.9 | short | 4 | 21.794 | 50.07 | clean |
| vsa4ps44 | 44 | 0.9 | structured | 4 | 22.313 | 38.09 | clean |
| vsa8 | 42 | 0.9 | short | 8 | 44.454 | 30.22 | clean |
| dense8 | 42 | none | short | 8 | 58.081 | 30.84 | clean |
| dense16 | 42 | none | short | 16 | 116.708 | 44.09 | clean |

The structured prompt is the one written for this test: subject definition,
anatomy-preservation clause, lighting, camera, soundscape; 187 tokens against
15, and it costs nothing (TEXT 12.8 s against 12.3-14.3 s, denoise 22.1-22.8 s
against 21.7-22.0 s).

## 2. What the frames show

The defect the operator saw is seed 42's short prompt: at 0.5 s the head is
inside the body and at 1.0 s there is a second tail mass at the lower left.
The same latent re-rendered here reproduces it exactly, and the 1344x768
render changes nothing, so the defect is in the draft.

It is intermittent, not systematic. Of the four short-prompt 4-step seeds,
one (42) shows it and three (43, 44, 45) are clean at all five times. Of the
three structured-prompt seeds, none show it, and the fox is also better
placed: side profile, visible head, one tail, matching the prompt's
"left to right, eye level". Both 8-step arms and the 16-step arm are clean at
all five times.

Sparsity is not established as the cause. At seed 42 the dense arm is clean
where the sparse one is not; at seed 43 the sparse arm is clean and so is the
dense one. Two seeds per cell is not enough to separate them, and the cost
difference is real either way: dense is +6.9 s at 4 steps and +13.6 s at 8.

Sharpness ranks the arms only loosely, because each arm is a different sample:
33-50 across the 4-step arms, 30-31 at 8 steps (VSA and dense alike), 44 at 16
steps dense. Nothing here says a sparse 4-step draft is softer than a dense
one; the earlier 20.64 against 14.11 comparison is between the FP8 and BF16
arms of the full route at 1344x768, a different canvas, and stays out of this
table.

## 3. What it costs

Denoise at 672x384x124, BF16; the gap between the columns is what VSA saves
on a 9,753-row sequence:

| steps | VSA 0.9 | dense | note |
|---:|---:|---:|---|
| 4 | 21.7-22.8 s | 28.6-29.1 s | the release's step count |
| 8 | 44.5 s | 58.1 s | |
| 16 | - | 116.7 s | |

Whole arm, the 4-step VSA arm is about 70 s (TEXT 12-14, LOAD 7-11 warm, AUDIO
3, decode 20-23, mux 0.6). LOAD is page-cache dependent in these arms (7.1 to
11.2 s) and must not be compared across them.

## 4. What to adopt

The cheapest setting that passed every sampled time is 4 steps with a
structured prompt: the same 22 s of denoise the release spends, no extra
steps, and the page's own showcase prompts are of that shape (200 to 600
words with subject definitions and anatomy clauses, per the plan's reading of
the showcase page; the paper's own qualitative figures use the same shape,
Figures 11-12). The plain 15-token prompt is 1-in-4 at 4 steps on this subject.

The fallback when a clip does show a defect is 8 steps (+22.7 s), not dense
attention: dense costs +6.9 s at 4 steps and was clean in 2 of 2 seeds, which
does not separate it from VSA's 3 of 4. Dense cannot be made cheaper by
switching it on for the first step alone without a code change, and the
release's stage 1 runs every step sparse on purpose
(`runtime/stage1_ops/vsa.py:19-38`, `dense_first_n_steps: 0`).

## 5. The product clip

The adopted recipe rendered end to end, FP8 DiT plus the structured
prompt, seed 42: `outputs/faithful-fp8-structured-1344x768.mp4`, 4,460,596 B,
sha256 `8d818c8c...`, 5.175 s, H.264 1344x768 + AAC 32 kHz stereo, decode
clean. Phases: TEXT 15.696 (187 tokens), LOAD 28.974 (cold), DENOISE 15.898
(10 prefix blocks, realized exact share 0.2141 of 9,925 rows), AUDIO 2.975,
H3 VAE decode 64.648, WALL 133.569. Against the short-prompt FP8 clip the
denoise is +0.9 s and the prompt encode +2.4 s; the load difference is the
page cache, not the arm.

At 1.0 s the two clips separate in a crop of the same region: the short-prompt
arm has a buried head and a stair-stepped silhouette (what the operator
described as bad blurry pixels), the structured one has a formed head with
ears, eye and muzzle, fur texture along the body, and a clean silhouette. The
1344x768 render reproduces whatever structure the draft carries, so the alias
comes from the draft and the prompt is what removes it.

Its sharpness number is 15.45 against 20.64 for the short-prompt clip, and
that is not evidence of a softer picture: the aliased silhouette is a hard
edge and the Laplacian count rewards it. Laplacian variance ranks finished
pictures of one scene, and these are two scenes.

## 6. Open

- The defect rate at 4 steps is estimated from four seeds; six would say
  whether it is one in four or one in ten.
- Whether sparsity contributes needs a dense 4-step arm at seeds 44 and 45.
- The FP8 arm's seed-42 clip was clean at all four sampled times, so a
  structured prompt under FP8 is the product recipe to render next.
