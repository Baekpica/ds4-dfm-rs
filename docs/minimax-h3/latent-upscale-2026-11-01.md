# The H3 latent upscaler, ported and gated

**Branch** `feat/rapid-h3`, off `feat/h3-native` at `5deedf2c`.
**Checkpoint** `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`,
690,592,672 bytes, sha256 `043e5a48…12eb55c2`, at
`/mnt/h3-artifacts/models/latent_upscale/` on the Spark.
**Licence** Apache-2.0 for the weights and the node (the H3 base weights carry
their own community licence; check that separately before redistributing).

## Why this model and not the LTX one

The second stage this port carried until `dd73b5a5` upscaled an **LTX** latent,
which first required an H3-to-LTX adapter to translate between two unrelated
latent spaces. That translation was lossy by construction — channel statistics
and temporal packing do not line up — so the LTX refiner was handed a latent
whose detail was already gone. The 1344 x 768 render measured 0.1382 normalized
detail on that route against 0.1625 for plain H3 at the same canvas.

The reference implementation's `ts_h2` recipe instead upscales an **H3** latent
with this resizer and then spends one denoise step at sigma 0.72 at full size.
Nothing is translated, so nothing is lost on the way in.

## What the model is

`LatentResizer3D` (`config.json`, confirmed against the checkpoint's 322
tensors): `conv_in` 24 -> 512, twelve `ResBlockEmb3D` alternating with six
`TemporalConv`, **trilinear resize**, twelve more of each, `norm_out`,
`conv_out` 512 -> 24. GroupNorm with 32 groups, per-block embedding from a
1 -> 64 -> 64 MLP over `scale - 1`, dropout inert at inference (the node calls
`.eval()`). 345,280,216 parameters.

The upscale is the trilinear resize; the convolutions either side of it
re-synthesize what the resize cannot invent. `TemporalConv` is depthwise over
time with a five-tap kernel, so a frame sees two frames either side.

Latents are normalized with the released per-channel `latents_mean` /
`latents_std` before the network and denormalized after, exactly as the node
does.

## Port

`crates/ds4-core/src/minimax_h3/latent_upscale.rs`, plus six kernels in
`native/h3/h3_dit_gpu.cu`: zero padding, the plain GroupNorm the embedding
modulates, the embedding's channel affine, the depthwise temporal convolution,
the trilinear height/width resize, an F32 add for the identity skips, and the
im2col plus GEMM convolution. `h3d_conv3d`, `h3d_group_norm_silu` and
`h3d_vae_linear` are reused unchanged.

Two things the port had to get right that were not obvious:

- **`group_norm_silu` reduces per temporal slice.** It was written for the
  encoder, which runs one frame at a time. PyTorch's GroupNorm reduces over
  `T * H * W` jointly, so using it here normalized a multi-frame latent with
  the wrong statistics. The gate caught it at 6.4e-3 relative; with the joint
  statistics the same check reads 2.1e-6.
- **The im2col has to walk the output geometry.** The row index runs over
  `out_depth * out_height * out_width`, which is the padded extent less the
  kernel's reach; dividing by the padded strides put every row after the first
  on the wrong voxel. That one was introduced with the GEMM path and caught by
  the gate at 0.8 relative.

## Gate

`tests/h3_upscale_gate` runs a 2 x 2 x 2 latent (4 x 4 out) through the device
path and through an independent f64 reference written from the model
definition, over the checkpoint's tensors read by name, sharing no code with
the port beyond the architecture's constants.

```
upscale 166ms for 8 voxels: max |device - reference| 5.025e-5,
  relative 2.135e-6 (scale 2.354e1, worst channel 18)
PASS
```

f32 device arithmetic against an f64 reference over twenty-four convolutions:
anything structural lands orders of magnitude outside the 5e-4 bound.

## Cost

`--bench 32 34 58` (63,104 latent voxels in, 252,416 out, 2 x spatial):

| arm | load | upscale |
|---|---|---|
| F32 GEMM | 1.03 s | 16.34 s |
| TF32 GEMM | 1.05 s | 16.75 s |

The reference implementation prices this stage at 1.7-3 s on the same GPU, so
the port is roughly 6x off, and the profile says why: TF32 changes nothing, so
the wall is **memory, not arithmetic**. The im2col materializes 27x the input
per convolution — 14 GB per full-size convolution here, 840 GB over the stage —
and cuBLAS then reads it back. At 273 GB/s of unified bandwidth that traffic is
~3 s; the measured 16 s means the expansion is running far below that. A fused
convolution that stages taps in shared memory instead of in HBM is the fix, and
it is the obvious next unit.

Sixty-five teraflops of f32 product would be ~3.5 s on this part at peak; the
F32 GEMM arm reaches 6.5 TFLOPS, which says the same thing from the other side.

## What it buys

A two-stage route costs a draft at 0.7x plus this stage plus one refine step
instead of a full six-step pass at delivery size. At the canvases the reference
uses (928 x 544 draft, 1344 x 768 refine, 124 frames) that is roughly 78 s of
draft denoise at the fast decoder arms, 16 s here, 64 s for one full-size step
and the decode on top — against 330 s for the single pass this branch renders
today. The face is the point: 1344 x 768 across a standing figure puts about
90 px on her face, and the refine step is what re-synthesizes eyes, lashes and
lips at the pixels the resize created.

The engine does not call this yet: `latent_upscale` is a module and a gate on
this branch, and wiring the two-stage request is the next unit.
