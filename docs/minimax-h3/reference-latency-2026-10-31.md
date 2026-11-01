# Where the reference's 56 s goes, and where ours goes

Status: open. One optimization landed (`7a5e9b04`), the sharpness cause is
measured, and one profile is missing before any latency target is promised.

## 1. The claim and its scope

`/data/Sol-h3-Spark/README.md:46`: "On a single NVIDIA DGX Spark at 1344x768,
T2V end-to-end latency is 56 s for a 5-second output."

`runtime/pipeline.py` defines the clock the claim is about: one monotonic
interval from request entry to the completed muxed MP4 (`:175-189`), recorded
per request as `e2e_s` beside `qwen_s`, `stage1_s`, `stage2_s` and
`stage2_phases_s` (`:194-197`). Warmup is excluded; the first formal request
includes the first Qwen encode (`absolute_paths`: "first_formal_request_
included": True). No `results.json` ships with the tree, so the 56 s has no
per-phase evidence on this unit. Everything below that is marked measured was
measured here; everything derived is labelled.

## 2. What the reference does differently, read from its code

| # | Mechanism | Source | What it removes |
|---|---|---|---|
| 1 | The H3 video is never decoded: Stage 1 saves no video, returns no frames, and the capture asserts `video_decode_calls == 0` and `h3_decoder_calls == 0` | `runtime/stage1.py` (`save_video=False`, `return_frames=False`, capture checks; `resolved_config` records `"video_decode": False`); `README.md` "The latent adapter replaces an intermediate H3 VAE decode and LTX VAE encode" | Our 109.2 s H3 VAE decode at 672x384x124 |
| 2 | Models stay resident between requests; Stage 2 prompt contexts are computed offline, so Gemma and the connector never load online | `README.md` "Why the models can remain resident"; `docs/validation.md` "Gemma and the connector are absent from online inference" | Our 28.2 s `TEXT` and 41.7 s `LOAD` per process |
| 3 | Stage 1 attention is 90% sparse over 64-token tiles on cuDNN BSA, 200 calls and 200 fused merges per request | `runtime/stage1_ops/vsa.py` `configure_profile`; `docs/validation.md` | Part of our denoise (our VSA exists as `H3_DIT_SOL`, off by default) |
| 4 | Stage 1 DiT is W8A8 FP8 after the BF16 LoRA merge | `runtime/stage1.py` `resolved_config`: "native W8A8 after BF16 LoRA merge; exact BF16 AdaLN lookup" | Our 57.5 % BF16 GEMM share per evaluation |
| 5 | Stage 2 is three BF16 updates with Triton Sol (141 Sol calls, 3 dense), the distilled LoRA at 0.8, and the decoder streamed in 16-frame chunks straight into ffmpeg | `runtime/stage2.py` `_checked_attention_stats`, `write_full_vae_video`; `runtime/stage2_ops/writer_chunking.py`; `docs/validation.md` | Our 110.5 s denoise and our whole-clip decode-and-mux |

## 3. Our phases, measured on this Spark

Shapes: draft 672x384x124, refine 1344x768x121. Dense BF16 stage 1 (no
`H3_DIT_SOL`), Sol-Attn and the distilled LoRA at 0.8 in stage 2.

| Arm | Phases | Sum |
|---|---|---|
| 4-step FastH3 draft, with video | 28.22 + 39.35 + 28.41 + 3.09 + 109.16 + 0.60 | 208.8 s |
| 4-step FastH3 draft, `--latent-only` | 28.24 + 41.73 + 28.76 + 3.11 + 0.00 | 101.8 s |
| 20-step base H3 draft, with video | 27.32 + 41.25 + 143.29 + 2.98 + 109.06 + 0.51 | 324.4 s |
| Transfer only | 0.31 + 2.83 + 15.14 + 0.83 | 19.1 s |
| Refiner, 3 updates | 2.83 + 41.0 + 36.6 + 33.0 + 15.7 + 0.79 | 129.9 s |

The latent-only flag is `7a5e9b04`. Its measurement is the first two rows:
one decode and one mux, 107.0 s of the 208.8, removed without touching text,
load, denoise or audio.

## 4. The sharpness finding, measured

Laplacian variance of the frame at 2.0 s, the same operator on every row:

| Input / output | 672x384 | 1344x768 |
|---|---|---|
| 4-step draft | 25.31 | - |
| 20-step draft | 35.93 | - |
| bicubic 2x of the same draft frame | - | 2.71 (4-step) / 3.65 (20-step) |
| transfer-only output | - | 3.82 (4-step) / 4.63 (20-step) |
| transfer-only output, projected back to 672x384 | 8.65 / 15.91 | - |
| 3-step refiner output | - | 10.42 |

Two conclusions. The transfer path is barely sharper than a bicubic resize:
the LTX VAE is asked to decode a latent the refiner was meant to denoise
first, and that direct decode softens it. The refiner is 2.25x sharper than
the transfer, and it changes the animal: at 2.0 s the draft's tall pointed
ears and narrow muzzle come back as short rounded ears and a blunt open
muzzle. So the sharp route alters and the faithful route is soft, on the same
latent.

## 5. The question that decides the plan

**Corrected 2026-11-01: this section's arithmetic is off by 1000x and its
conclusion is withdrawn.** One refiner step does read all 48 blocks - 34.58 GiB
measured from the loader - but 37.13 GB at this host's 246 GB/s is **0.151 s**,
not the 136 s the "0.7-1.0 TB/s" figure implies, and none of it is a wall. The
step is compute-bound at 9-14 TFLOPS against a 128.4 TFLOPS BF16 peak, which is
what the profile was wanted for; `refiner-update-anatomy-2026-11-01.md` has the
phase split, the per-block table and the corrected reasoning. The paragraph
below is kept as written, dated, with the correction above it.

One refiner step must read the transformer it uses: 48 blocks sum to 34.6 GiB
BF16 in the header of `ltx-2.5-22b-dev-transformer-bf16.safetensors`, of which
the video-only branch is about 24 GiB. Our 36.8 s per step then implies an
effective 0.7-1.0 TB/s, three to four times the 246 GB/s this host streams in
our own measurement (`docs/gb10-hardware.md`). One of the two numbers is
wrong, and the difference decides whether the refiner has 3x of headroom or
none. It needs one profile of a single step before anyone promises 56 s here.

## 6. Next, in order

1. Residency: serve the draft through `h3-server-rs` with the NVFP4 prompt
   encoder. Our own server record already shows `TEXT` 0.001 s and `LOAD`
   0.000 s on the second render in a process
   (`docs/minimax-h3/steady-state-2026-10-28.md`).
2. Profile one refiner step with `nsys`, then `ncu` on its largest kernel, to
   settle section 5.
3. Make the stage-2 sigma schedule a runtime value and test a one-step pass
   from a low sigma as the detail pass: the anchor stays, the model supplies
   the texture the direct decode loses.

## 7. The publish bar, checked against our phases

The upstream page (NVlabs/Sana, branch `sol-engine`, the same tree as
`/data/Sol-h3-Spark`; README, configs, `runtime/*.py` and `infer.py` diff
identical) publishes 56 s with a share per phase. Reading those shares:

| phase | share | seconds | ours |
|---|---|---|---|
| Qwen prompt encode | 3.8 % | 2.13 | 15.35 (NVFP4, loaded in-process) |
| H3 4-step generation | 35.2 % | 19.77 | 21.40 (VSA 0.9) |
| 2x upscale + VAE adapter | 2.4 % | 1.35 | 2.83 |
| LTX 3-step refinement | 44.0 % | 24.71 | 110.37 |
| LTX VAE decode | 12.5 % | 7.02 | 15.34 |
| MP4 + audio mux | 1.3 % | 0.73 | 0.86 |
| request overhead | 0.9 % | 0.51 | 35.14 + ~42 (model loads) |

Stage 1 is at parity (1.08x). **The refinement conclusion below is withdrawn
(2026-11-01):** 3 x 34.6 GiB over 24.71 s is 4.51 **GB/s**, not TB/s, so the
bar is nowhere near a streaming limit; at 8.24 s per update it asks for 50-73
TFLOPS, which is inside the 65-90 TFLOPS cuBLAS reaches on this part
(`refiner-update-anatomy-2026-11-01.md` §4). The bar stands; our achieved rate
is 5x off it. The paragraph below is kept as written, dated.

The old text: three passes of the
22B BF16 transformer is 3 x 34.6 GiB = 104 GiB, and 24.71 s for that is
4.5 TB/s against this host's measured 241-246 GB/s (`docs/gb10-hardware.md`).
Either their stage 2 runs different weights or the bar is not this artifact
on this part. No 56 s promise is honest until one `nsys` profile of a single
refinement step says whether the step is kernel-bound or stream-bound.

## 8. The anchored pass, measured

One 4-step draft, `--sigmas` swept, no `--sol` (the taus belong to the
release's intervals). Frame at 2.0 s; sharpness is Laplacian variance,
faithfulness is SSIM against the transfer-only decode of the same latent.

| route | lapvar | SSIM |
|---|---|---|
| transfer only | 2.82 | 1.000 |
| one step from 0.3 | 3.04 | 0.940 |
| one step from 0.45 | 3.30 | 0.923 |
| two steps from 0.6 -> 0.3 -> 0 | 3.67 | 0.893 |
| release, three from 0.909375 | 5.54 | 0.777 |

Content follows the start sigma exactly as `noised_state` says it must: at
0.3 the fox survives, at 0.909375 it is redrawn. Detail follows the same
knob, so the refiner cannot be both faithful and sharp: there is no sigma
that keeps the animal and restores texture. A denser pass also costs more
than the release's: one dense step measured 127.7 s against 36.8 s per
Sol step.

The detail for a faithful upscale therefore has to come from the upscaler
or from a better draft, and both are measurable:
- a 20-step draft lifts the draft's lapvar from 25.31 to 35.93 and the
  transfer decode from 3.82 to 4.63 (same-frame controls: bicubic 2.71 and
  3.65), so the chain tracks the draft's quality;
- the transfer decode beats a bicubic resize by 1.27-1.41x on those frames,
  which is the upscaler's whole contribution as measured so far. Isolating
  it needs the upscaled H3 latent decoded by the H3 VAE against a
  bicubic-upscaled draft decoded the same way.

## 9. The upscaler isolated: it works, and the LTX bridge was hiding it

One decoder (the H3 video VAE, `H3_VAE_ATTENTION=tensor`,
`H3_VAE_GEMM=bf16`), two latents, same draft and prompt, frame at 2.0 s:

| latent decoded | canvas | lapvar |
|---|---|---|
| the draft itself | 672x384 | 32.86 |
| the draft, bicubic 2x | 1344x768 | 3.34 |
| the draft, lanczos 2x | 1344x768 | 3.56 |
| the upscaler's own latent | 1344x768 | 14.11 |

The upscaler is 4.2x sharper than a bicubic resize at the same canvas (SSIM
0.847 against it, same pose, same ears, same muzzle), so it is adding real
structure. Decoded through the adapter and the LTX VAE instead, the same
latent yields 3.82 - almost the bicubic number. The loss is in the combined
adapter-plus-LTX-VAE path; an earlier note here blamed it on the adapter
returning a noised latent, and that is wrong: the adapter returns its model
output unnoised and the stage-2 noiser adds the noise afterwards
(runtime/stage2_ops/h3_ltx_adapter/adapter.py:70-102, then
official_compat_h3_refiner_diagnostic.py:324-332). What is established is
only where the detail disappears, not which of the two stages inside that
path causes it.

## 9a. External review of the identity question (Gpt-6-Astra)

An independent review read the release, the diagnostic, the pinned LTX-2
sources and the port, and confirmed the mechanism with the upstream noiser:
`torch.lerp(latent, noise, noise_scale)` then
`torch.lerp(clean_latent, latent, denoise_mask)` with an all-ones mask
(ltx-core `components/noisers.py:30-37`), which is algebraically
0.090625 x draft + 0.909375 x noise, and the per-step blend
`denoised * mask + clean * (1 - mask)` with mask one gives the clean draft a
zero coefficient (`utils/helpers.py:461-463`, `utils/samplers.py:25-36`).
Its findings, in its own order:

1. The published recipe is generative refinement, not content-locked
   super-resolution. Its only draft contribution is the noised initial state;
   the image conditioning that could anchor a frame exists in the diagnostic
   but is deliberately disabled by the release runtime
   (`runtime/stage2.py:38-44`).
2. The port's remaining recipe deviation is real and unmeasured: the release
   builds a clean, frozen audio state and runs the joint AV transformer
   (`runtime/stage2.py:78-99`), while the port passes `audio=None`.
3. The earlier sigma sweep conflates sigma with step count and with Sol being
   off for custom schedules, so it bounds the effect but does not isolate
   sigma.
4. The port already exposes the faithful route (`--h3-video upscaled` with
   `--h3-vae`), which returns before the adapter loads; it is not the
   published LTX route.


Consequence for the product route: a faithful sharp 2x exists today and it
does not use the LTX bridge at all. Draft latent -> upscaler -> H3 video VAE,
measured on the Spark: upscaler 2.83 s, H3 decode at 1344x768 62.25 s, mux
0.86 s. The adapter and the refiner are what cost content.

## 10. The upscale route needs nothing from LTX, and that is now enforced

`--h3-video` requires only the latent, the soundtrack, the upscaler and the
H3 video VAE. The adapter, the refiner, the LTX VAE and the prompt context
are required only by the LTX route, so the faithful arm can run without
them and without Gemma.

Measured on the Spark: `h3-sol-rs --latent ... --audio ... --upscaler ...
--h3-vae ... --h3-video upscaled` produced a clip byte-identical to the one
that also carried the LTX arguments, sha256
`8718b3eb6d5b3c3ed455dd4f07bb1234b37922b3d79c3085f0e2f3e3d9abb169`, 5.175 s,
1344x768, decode 60.29 s. Nothing on the LTX side reaches the picture.
