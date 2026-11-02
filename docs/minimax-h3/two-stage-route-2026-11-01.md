# The two-stage route: a draft, one upscale and one refine step

Unit: `feat/rapid-h3`, 2026-11-01. Follows
[`resizer-algorithm-2026-11-01.md`](resizer-algorithm-2026-11-01.md), which
left the resizer at 3.886 s for a 32 x 34 x 58 draft and named the next unit:
the engine does not call it yet.

**Result.** `--draft 0.7` renders the reference's `ts_h2` route end to end. At
768 x 1344 x 141 and 6 steps the denoise falls from 471.3 s to 203.7 s
(draft 125.5, upscale 4.6, refine 78.2) and the wall from 912.7 s to 667.4 s,
with the draft's audio carried through the refine bit for bit. The clip is a
clip. It was not, for the first three attempts: the route rendered a lattice,
and finding out why is most of what this unit did.

## 1. The route, taken from the reference

Four numbers define the route and none of them is this port's choice: the draft
scale, the refine sigma, whether the refine re-noises or continues, and what
exactly the audio freeze holds. All four are read off the released pipeline
(`/data/Rapid-H3` at `d3bd39b`) against the ComfyUI it runs
(`rapid-h3/h3-comfyui:v0.42.1`, `serve.sh:13`), and cited below before a line
of code was written.

| fact | value | where |
| --- | --- | --- |
| draft scale | 0.7, rounded to 32 pixels per axis | `harness/workflow.py:50`, `:393-394` |
| draft schedule | 6 steps, shift 12, `simple` -> 1, .98361, .96, .92308, .85714, .70588, 0 | `harness/workflow.py:50`, `harness/fast_test.py:76` |
| upscale | video only, to the target dimensions, `align` 32 | `harness/workflow.py:411-415` |
| refine sigmas | `0.72, 0` - exactly one step | `harness/workflow.py:50`, `:418` |
| refine seed | re-noised: `x = sigma*noise + (1-sigma)*latent` | `comfy/samplers.py:993`, `comfy/model_sampling.py:94-97` |
| audio | `LTXVFreezeLatent` on the draft's audio latent | `harness/workflow.py:417`, `:448` |
| freeze mechanism | `noise_mask = 0` for the whole stream | `comfy_extras/nodes_lt_keyframes.py:984-1036` |
| frozen stream is kept | sampler keeps `latent_image` where the mask is 0 | `comfy/samplers.py:634-642` |

### 1.1 The refine re-noises; it does not continue

`SamplerCustomAdvanced` seeds its first step with
`noise_scaling(sigmas[0], noise, latent_image)`
(`comfy/samplers.py:993`), and H3 is a `ModelType.FLOW` model
(`comfy/model_base.py:115-118`), whose `noise_scaling` is
`CONST.noise_scaling` (`comfy/model_sampling.py:94-97`):

```python
return sigma * (s * noise) + (1.0 - sigma) * latent_image   # s = noise_scale = 1.0
```

`max_denoise` is false here - it is only true when `sigmas[0]` is at or above
`sigma_max` (`comfy/samplers.py:966-969`), and `sigma_max` for shift 12 is 1.0
against the refine's 0.72. So the refine's starting latent is

```text
x = 0.72 * noise + 0.28 * upscaled_draft
```

with `noise` the *same* `RandomNoise` node the draft sampled from
(`workflow.py:408` and `:349` both take `["11", 0]`), drawn at the full canvas.

The number to notice: the draft's own schedule ends at **0.70588**, and the
refine re-enters at **0.72** - slightly *above* where the draft stopped. The
route does not resume the draft's noise level; it treats the upscaled latent as
an x0 estimate and adds a step's worth of noise back.

### 1.2 What the audio freeze holds, and what it does not

`LTXVFreezeLatent` sets `noise_mask` to zeros over the whole stream
(`nodes_lt_keyframes.py:1006-1032`), `LTXVConcatAVLatent` carries that as the
audio half of a per-modality mask and defaults the video half to ones
(`comfy_extras/nodes_lt.py:957-965`), and `SamplerCustomAdvanced` hands the
latent's `noise_mask` to the guider as `denoise_mask`
(`comfy_extras/nodes_custom_sampler.py:1047-1055`).

In the sampling loop (`comfy/samplers.py:634-642`) a zero mask means two things
per step: the state handed to the model is rebuilt from `latent_image`, and the
model's output for those rows is replaced by `latent_image` again. So the audio
*state* is bit-for-bit the draft's audio at every step and at the end, while
the audio rows still enter the model - that is the point: they are what the
video's lip sync is conditioned on.

One consequence worth stating, because it is easy to get wrong in a port: the
frozen stream is **not** fed to the model as an untouched tensor. For H3 the
audio stream carries its own scaling (`comfy/model_base.py:2299-2318`
`scale_latent_inpaint`), and the audio timestep is derived from the video sigma
with the audio shift (`time_shift_sigma(sigma_v, 12, 3)`,
`comfy/ldm/minimax/model.py:36-39`). What is frozen is the audio latent, not the
arithmetic the model does around it.

### 1.3 The two stages do not share a conditioning row set

The draft runs `MiniMaxH3ImageToVideo` at the draft size and the refine gets the
full-size conditioning; in `encode_once="draft"` (the preset's setting,
`workflow.py:52`) the Qwen hidden states are the draft's and the keyframe
latents are re-encoded at full size (`workflow.py:438-441`). So the *text*
encode is paid once and the *conditioning rows* are per stage.

## 2. What the port added

- **`Draft { scale, sigma }`** on `RenderRequest`, with `Draft::RELEASED` =
  0.7 / 0.72. `Draft::canvas` is the reference's `round(scale * w / 32) * 32`
  with `round_ties_even`, because Python's `round` is half to even: 768 x 1344
  gives the preset's 544 x 928, and 544 x 928 at 0.5 gives 256 x 448 rather
  than 288 x 480.
- **`Schedule::refine(sigma)`** in `host.rs`: one evaluation and the terminal
  zero, with the audio sigma derived from the video one by inverting the video
  shift to the base grid and applying the audio shift - the same expression
  `Schedule::serving` uses per index, so the derived sigma agrees with the
  serving grid to one ulp (a test pins that).
- **`host::renoise`**: `x = sigma * noise + (1 - sigma) * latent`, in the
  reference's expression order.
- **`Dit::denoise_refine`**: one `forward` at the graph's only sigma, then
  Euler on the video buffer alone. The audio is a `&[f32]` - not updated, by
  signature - and still uploaded to the model as the step's conditioning.
- **The engine**: stage one renders at the draft canvas, the resizer takes the
  video latent to the delivery canvas, and stage two re-reads its anchors and
  references at its own canvas, re-noises and refines. The upscaler is opened
  once and kept; its directory comes from `H3_UPSCALER_DIR`, never from a path
  in the source.
- **`--draft <scale>` / `--refine-sigma <sigma>`** on `h3-rs --jobs`, refused
  without `--jobs` because the one-shot route has no second prepared graph.
- **The gate's `--draft` arm** asserts `draft_audio_hash == audio_hash` and the
  engine's `audio_frozen`, and prints `STAGES` per turn.

The prepared key had to grow a `sigma` field. Both refine schedules have
`steps == 1`, so `steps` alone let a request at 0.55 reuse a graph whose
modulation maps were baked at 0.72. The test that would have caught it is
`a_refine_sigma_is_part_of_the_prepared_key`.

## 3. The first clip was a lattice

Three renders, all wrong in the same way - a fine periodic grid at about 1.4
latent cells:

| render | settings | result |
| --- | --- | --- |
| delivery, 6 steps | 768 x 1344 x 141, `--draft 0.7` | lattice |
| small, 4 steps | 512 x 512 x 22, `--draft 0.7` | lattice |
| small, refine 0.02 | same, `--refine-sigma 0.02` | same lattice |

The third is the bisect: at sigma 0.02 the refine is 98 % the upscaled latent,
so the lattice was already in the resizer's output, not in the refine. The
control that followed is what made it conclusive - the same 352 x 352 canvas
rendered single-stage is a clean fox, so the draft latent was good and the
resizer was not.

The cause is one line: the node's ResBlock builds
`emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(...))`
(`minimax_h3_latent_upscaler_3d.py:168-170`), and the port applied the linear
to the raw 64-wide embedding, without the block's SiLU. Every block's
`h * (1 + scale) + shift` was therefore modulated by the wrong vector, and
eighteen of them compound: the output becomes mostly a per-channel constant
with a lattice on top, which is exactly what the frames showed.

The tell was in the statistics, not the picture: on a real 22 x 22 draft the
output's per-channel deviation was 7.06 against the input's 0.57, and the
per-channel means were the *same* numbers at 22 x 22 and at 34 x 58 and for two
different inputs (channel 1 at -10.7 and -11.1). An output that does not depend
on its input is a modulation path, not a data path.

## 4. Why two instruments could not see it

- **The gate compares device against a host transcription at 2 x 2 x 2.** The
  transcription had the same misreading, so it agreed to 3.5e-7 while both were
  wrong. This is the campaign's "a per-op probe set cannot prove a composition"
  in a new form: an instrument written from the same reading as the
  implementation cannot audit it.
- **The bench at 32 x 34 x 58 only times.** No correctness, so the one shape
  the product would use was never checked for value. Its 3.886 s stands - the
  fix changes no kernel - but it never said the stage was right.

Two additions, both kept:

- `h3_upscale_gate --shape T H W HO WO` threads a shape through `run` and the
  transcription, so any shape can be checked and not only timed.
- `h3_upscale_probe` (new, `tests/h3_upscale_probe`) runs the resizer on a
  BF16 latent the model actually produced (`h3-rs --save-latent`) and prints
  per-channel means and deviations on both sides. It is the instrument that
  found the lattice and then confirmed the fix.
- The gate also measures `deviation_ratio` - output deviation over input's -
  and requires 0.25 to 6. Measured: 3.0 on the gate's own near-constant input,
  1.1 on a real 22 x 22 draft, and 8 to 20 with the broken modulation. That
  check does not come from the transcription, so a shared misreading fails it.

After the fix, on the same real draft, the output tracks the input
(channel 0: -0.588/0.621 against the input's -0.603/0.533), and the reference
shape tracks it too (34 x 58 synthetic in -0.022/0.288, out 0.048/0.255 on
channel 0).

The F16 arm's error moved with the fix, from 6.5e-5 to 2.45e-4 of its 5e-4
bound: an activation the blocks were missing also changes what F16 rounds. The
arm is still inside its bound, but it now spends half of it.

## 5. The numbers

**Gate**, 512 x 512 x 22, 4 steps, seed 42, Spark `gx10-e1d2`:

| arm | result |
| --- | --- |
| direct | `RESULT PASS residency`, media identical between turns, turn 2 = 8.32 s wall |
| `--draft` | `RESULT PASS two-stage text_hit audio_frozen`, media identical, turn 2 = 29.27 s |

The two-stage turn's own split (turn 2): draft 2.218 s, upscale 0.153 s,
refine 1.117 s, load 19.526 s, `evaluated=5`. The load is *two* loads there,
and a repeat cannot hit: the engine holds one prepared graph and the two stages
are two canvases, so what a repeat saves is the prompt encode, which it does.

**Delivery clip**, 768 x 1344 x 141, 6 steps, seed 7, same prompt, one
process each, `H3_VAE_GEMM`/`H3_VAE_ATTENTION` left at the environment's word
(the decoder is fp32/scalar in both, and it dominates):

| phase | direct | two-stage |
| --- | --- | --- |
| prompt encode | 28.221 | 28.183 |
| DiT load | 20.898 | 30.758 (two) |
| denoise | 471.312 | 203.687 |
| - draft | - | 125.490 |
| - upscale | - | 4.570 |
| - refine | - | 78.196 |
| audio VAE | 3.508 | 3.564 |
| video VAE | 386.281 | 394.078 |
| mux | 0.878 | 0.925 |
| **wall** | **912.739** | **667.449** |

So the route is worth **57 % of the denoise and 27 % of the wall**, and the
video decode is now the wall's largest term in both arms.

The refine's single step is 78.2 s against the draft's 21.0 s, on 2.05x the
packed rows: the delivery-canvas step is 3.7x the draft's, not 2x, which is
what makes the route pay more than the token count suggests. That superlinearity
is the attention path, which is already on the backlog as its own unit.

The audio hash is identical across the refine in every run
(`4279f6eabc705b50` at the delivery canvas), and the first, broken-resizer
render produced the *same* audio hash with a different video - the resizer
touches video rows only.

## 6. What is not established

- **Quality against the single pass.** Both clips are plausible; the two-stage
  frame is softer in the fur and the mid-clip frame carries a ghost of the fox
  at the left edge. The reference's own README records that `ts_h2` gives up
  close-up detail and weakens push-ins, which matches what one pair of frames
  shows, but one prompt and one seed is not a quality measurement. The
  campaign's scoring harness (`harness/eval_clips.py`) is the instrument for
  that, and it is not wired to this port.
- **The 4-step draft.** This checkpoint is a 4-step distillation, and the
  reference's `ts_h2` uses 6 draft steps on the base model. Four plus one was
  measured here because it is what the checkpoint is for; six plus one is what
  the reference does. The prompt after the fix was run at 6 to match the direct
  arm, which is the fair comparison.
- **Residency for the two-stage route.** Two graphs cannot both be resident on
  one 128 GB host, so every clip pays two 61.7 GiB reads. Sharing the weights
  and preparing only the sequence per stage is the fix, and it is a unit of its
  own.
- **References and anchors in the second stage** are implemented (each stage
  reads its own) but only the no-media path was rendered. A first-frame clip
  through the route has not been looked at.
