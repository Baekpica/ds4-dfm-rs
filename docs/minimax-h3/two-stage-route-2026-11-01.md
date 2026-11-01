# The two-stage route: draft at 0.7, upscale, one refine step at 0.72

Unit: `feat/rapid-h3`, 2026-11-01. Follows
[`resizer-algorithm-2026-11-01.md`](resizer-algorithm-2026-11-01.md), which
left the resizer at 3.886 s for a 32 x 34 x 58 draft and named the next unit:
the engine does not call it yet.

## The route, taken from the reference

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

### 1. The refine re-noises; it does not continue

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

### 2. What the audio freeze holds, and what it does not

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

### 3. The two stages do not share a conditioning row set

The draft runs `MiniMaxH3ImageToVideo` at the draft size and the refine gets the
full-size conditioning; in `encode_once="draft"` (the preset's setting,
`workflow.py:52`) the Qwen hidden states are the draft's and the keyframe
latents are re-encoded at full size (`workflow.py:438-441`). So the *text*
encode is paid once and the *conditioning rows* are per stage.

## What this means for the port

- The draft canvas is `round(0.7 * w / 32) * 32` per axis, which is the same
  rounding `Canvas::new` already enforces (multiples of 32). 768 x 1344 gives
  544 x 928, the reference's own draft size.
- The refine needs its own prepared graph: its schedule is one sigma, not the
  draft's six, and its canvas is not the draft's. `Engine::prepare`-style keying
  already turns that into a cold load; it is the same 61.7 GiB read the branch
  pays once today, and it is the second time.
- The refine seed and the audio freeze are arithmetic on the host buffers the
  engine already owns, not kernels: `x = 0.72 * noise + 0.28 * upscaled`, and
  no `euler_step` on the audio.
- The audio sigma for the refine's modulation is derived from the video sigma
  the way the schedule builder derives it: invert the video shift to the base
  grid, apply the audio shift. That is the same expression `Schedule::serving`
  already uses per index.
