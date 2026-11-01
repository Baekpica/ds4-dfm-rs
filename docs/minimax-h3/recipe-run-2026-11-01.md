# The recipe end to end, and where the subject changes (2026-11-01)

Phase C. `misc/scratch/sol-recipe.sh` with every switch on - `H3_DIT_GEMM=fp8`
and `H3_DIT_SOL=0.9` in stage 1, NVFP4 prompt encoder, FastH3 at four steps,
672x384x124 latent only; x2 latent upscaler, H3-to-LTX adapter, LTX-2.5 dev
refiner at the release's sigmas, distilled LoRA 0.8, the frozen audio state,
LTX VAE, mux at CRF 19 / veryfast - run twice, one variable apart: the prompt.

| arm | stage 1 wall | denoise | stage 2 wall | the three steps | VAE decode | total | sha256 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| short | 78.317 s | 14.960 s | 186.116 s | 44.240 / 40.185 / 36.667 s | 16.041 s | 264.436 s | `14e3a6e0` |
| structured | 65.363 s | 15.892 s | 187.822 s | 44.795 / 40.632 / 37.218 s | 15.684 s | 253.188 s | `6aff5b74` |

Both are 5.175 s at 1344x768, h264 + aac 32 kHz stereo, and decode clean
(`ffmpeg -f null -`). Sol-Attn is what makes stage 2 usable: the same two
updates dense cost 127.806 / 128.318 s against 44.240 / 40.185 s routed.

## Where the subject changes

The same draft latent through the same refiner at three schedules, all at
2.5 s, with PSNR against the decoded transfer:

| arm | sigmas | what the frame is | PSNR vs transfer |
| --- | --- | --- | ---: |
| transfer | 0.001, 0 | the adapter output decoded, soft | - |
| enhance | 0.65, 0.35, 0 | the same fox, sharpened | 31.65 dB |
| release | 0.909375, 0.725, 0.421875, 0 | the same fox, release look | 24.86 dB |

The counter-example is the other draft. The short-prompt draft breaks anatomy
between 2.2 s and 2.9 s - the brush tail collapses to a stub and thin spikes
appear under the belly (`draft-anatomy-2026-10-31.md` has the arm table that
localises this to seed 42's short prompt) - and through the release schedule
the animal at 2.5 s is a cat: broad muzzle, short rounded ears, thin tail. The
same broken draft at 0.35 kept the fox.

## Ruled out, each measured rather than argued

1. **The block.** Every checkpoint of block 0 - both AdaLN norms, self-attn,
   text cross-attn, both AV cross operations, both feed-forwards - sits at the
   BF16 floor against the release's own block: rel_rms 2.6e-3 to 2.0e-2, cosine
   0.99998 or better, against a 5.4e-3 f32/bf16 floor. Receipt
   `av-cross-modulation-pair-2026-11-01.md`.
2. **The sigma-to-timestep path.** The release scales `sigma * 1000` and builds
   the per-token timestep as `denoise_mask * sigma`, never `1 - sigma`
   (`ltx-pipelines/.../helpers.py:486-500`,
   `transformer_args.py:177-181`); the port does the same
   (`ltx_refiner.rs:2827`).
3. **The text context.** The release's stage 2 is conditioned on
   `FIXED_PROMPT`, and its cache refuses any other prompt outright
   (`prompt_cache.py:73-75`, "case prompt cannot be used with fixed Stage2
   conditioning"). Our generic-prompt cache is the release's own conditioning,
   not a port shortcut.
4. **The transfer.** Decoding the upscaler+adapter output with the refiner
   barely moving (sigmas 0.001, 0) gives the same fox in the same composition,
   so the transfer is faithful; the refine moves the picture 24.86 dB, which is
   a re-synthesis, not a resample.

## What this means for the recipe

The result changes subject where the *draft* is anatomically impossible, not
where the port disagrees with the release: with a first sigma that replaces 91%
of the transfer, the refiner has to resolve the draft's contradiction, and the
text conditioning carries no subject to resolve it toward. Two measured
remedies, either one sufficient:

- the anatomy-explicit prompt, which removes the draft's break and keeps the
  fox at the release schedule (`outputs/recipe-structured.mp4`);
- a lower start - 0.65/0.35 keeps the fox even on the broken draft
  (`outputs/probe-enhance.mp4`), at 31.65 dB from the transfer.

Artifacts on the Spark: `outputs/recipe-short.mp4`, `outputs/recipe-structured.mp4`,
`outputs/probe-transfer.mp4`, `outputs/probe-enhance.mp4`, logs under
`scratch/phasec-*.log` and `scratch/probe-*.log`.
