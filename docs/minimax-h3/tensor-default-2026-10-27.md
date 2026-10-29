# Why the tensor-core attention is the default

Operator decision, 2026-10-27: the tensor path becomes the default and the
scalar path becomes an option, selectable from the CLI and from the API. This
file is the evidence for that choice, and it states what the evidence does not
cover as carefully as what it does.

## The claim

The tensor-core attention produces the same output quality as the byte-exact
scalar kernel and is substantially faster. Both halves need support, and the
second is much easier to establish than the first.

## The speed evidence

Every row is one binary with `H3_DIT_ATTENTION_MMA` as the only difference, so
each pair is its own control.

| Scope | Scalar | Tensor core | Ratio |
|---|---:|---:|---:|
| Attention kernel, 4,352 tokens | 295.57 ms/call | 10.07 ms/call | 29.4x |
| Attention kernel, 18,432 tokens | 5,320.93 ms/call | 180.78 ms/call | 29.4x |
| Kernel sweep median, 2,048 to 32,768 | 5,106.1 ms/call | 181.3 ms/call | 28.2x |
| Job A denoise, 768x768, 2 evals | 585.267 s | 40.294 s | 14.53x |
| Job A wall | 821.994 s | 277.019 s | 2.97x |
| 512x512, 20 steps, rain | 385.830 s | 64.353 s | 6.00x |
| 512x512, 20 steps, Earth | 409.684 s | 66.520 s | 6.16x |
| 512 wall, rain / Earth | 460.399 / 485.457 s | 139.418 / 141.562 s | 3.30x / 3.43x |

The kernel numbers are the honest headline, and they are stable: 29.36x and
29.43x at the two benchmark lengths, three rounds an arm agreeing to well under
one percent, and 28.2x on median across seven lengths. The job-level numbers are
smaller and explain themselves: the denoise gains 6.0-14.5x depending on how
large a share of a step the attention is, which is 96.3 percent at 18,432 tokens
and 86.2-86.6 percent at 4,352, and the wall gains only about 3x because the VAE
decode and the text encoder are untouched and now dominate. The VAE lands within
0.125-0.203 s of its baseline in every pair, which is itself the check that the
switch affects nothing else.

## The quality evidence

Rendered output, tensor arm against scalar arm, same prompt, seed, shape and step
count. PSNR and SSIM are FFmpeg's, computed on the decoded frames.

| Pair | Steps | PSNR y | PSNR avg | SSIM Y | SSIM all |
|---|---:|---:|---:|---:|---:|
| Job A, 768x768, 107 frames | 2 | 42.75 dB | 44.18 dB | 0.9829 | 0.9868 |
| 512x512, 56 frames, rain | 20 | 36.97 dB | 38.44 dB | 0.9721 | 0.9779 |
| 512x512, 56 frames, Earth | 20 | 30.13 dB | 31.85 dB | 0.9134 | 0.9389 |

At the kernel itself, the tensor arm against the scalar output is max absolute
1.2207031e-04 with rms 1.1488895e-05 at 18,432 tokens, and max 2.4414062e-04 with
rms 2.3465292e-05 at 4,352, with about 42 percent of values differing in their
last BF16 places. The clip structure is unchanged: same frame count, same
duration, same 32 kHz stereo audio layout, and the first output value is
byte-identical at both kernel lengths.

## What this establishes

- The deviation at the kernel is on the order of a few tenths of a percent
  relative, and its cause is known and named rather than mysterious: P rounds to
  BF16 for the second `mma`, a roughly 0.4 percent relative error on every
  softmax weight, plus a different reduction tree. Nothing else differs: the
  scores, the running maximum and sum, and the output accumulator stay FP32.
- Nothing in the output is structurally wrong. No dropped frames, no audio
  layout change, no NaN or saturation, and the VAE and text encoder produce
  their baseline results to within a fifth of a second.
- At the resolution and step counts the model actually runs, the two clips are
  close: 36.97 dB and 0.9779 on the rain subject at 20 steps.

## What this does not establish

- **No noise floor, so no band.** The handoff's condition for this arm was that
  it be gated on a noise floor measured on this host from a benign configuration
  change. That floor does not exist: every benign configuration change measured
  on this kernel is bit-identical, so the floor is zero and there is no
  established band inside which this deviation could be called noise-class. Any
  numeric threshold quoted here is a judgement, not a measurement of a bound.
- **The deviation is content-dependent, and Earth is the weak case.** At the
  same shape, steps, and measurement method, the Earth clip agrees less than the
  rain clip: 30.13 dB and 0.9389 against 36.97 dB and 0.9779. That is the honest
  spread, and it is not a measurement artifact: the Earth frame carries fine
  detail, cloud edges, a star field and city lights, where a perturbed weight
  shows more. A single figure from one prompt would have overstated the case.
- **The deviation compounds with depth.** Rain at 2 steps would agree much more
  closely than rain at 20; the 2-step Job A pair measures 42.75 dB and 0.9868
  while the same prompt at 20 steps measures 36.97 dB and 0.9779. The 2-step
  number flatters the tensor arm, and it is the number this campaign's benchmark
  setting produces, so it should not be quoted as the quality of the default.
- **No perceptual or semantic check.** PSNR and SSIM are pixel metrics. Nobody
  has looked at the two clips side by side and judged whether the difference is
  visible, and a pixel metric cannot answer that. The clips are in `~/h3-clips`
  on the driving machine as `h3-512-rain-{scalar,tensor}.mp4` and
  `h3-512-earth-{scalar,tensor}.mp4` precisely so that a person can.
- **No latent-level or token-level comparison.** The H3 DiT path exposes no
  logits or argmax, so the paired-NLL and top-1 method that `tests/minimax_quality`
  uses for the MiniMax text paths cannot be applied here. There is no equivalent
  instrument, which is why this file leans on render metrics.

## What would strengthen the case

1. A measured floor. A deliberately benign reorder in the scalar kernel, the
   butterfly in place of the down-shuffle in its reduction, keeps the arithmetic
   the same in kind, changes the order, and is bit-identical to nothing; it
   would give a real band to compare the tensor deviation against. Until it
   exists, the tensor arm's deviation is attributable but unbounded.
2. A 512 pair at 2 steps as well as 20, to separate the per-step deviation from
   its compounding.
3. A latent-level comparison, if the DiT ever exposes one.

## How to select each path

The tensor kernel runs unless something asks otherwise.

- CLI: `h3-rs --attention scalar` for the byte-exact kernel; `--attention tensor`
  is the default and may be passed explicitly.
- API: the create body takes an optional `attention` field, `"tensor"` or
  `"scalar"`. Absent means tensor, so no existing caller changes what it gets,
  and a wrong value is refused by name (`invalid_attention`) rather than
  silently defaulted.
- Native and test harnesses: `H3_DIT_ATTENTION=scalar` or `=tensor`. The older
  `H3_DIT_ATTENTION_MMA` is still honoured, 1 for tensor and 0 or empty for
  scalar, so the A/B scripts written before the inversion keep their meaning.
  `h3d_set_attention` overrides all of it for a host that chooses per run.

## Consequences of the decision

Making the tensor path the default changes what the released output is.

- The released render is no longer byte-identical to the reference. Job A's
  scalar render holds `69816f7f...`; the tensor render is `5cb28765...`. Every
  render hash recorded in this campaign describes the scalar option from here on,
  not the default.
- The reproducer's `out_fnv` gate (`c6196e81bc4634e0` at 4,352 and
  `5c1e3810028a8fa5` at 18,432) likewise describes the scalar option.
- The switch becomes inverted: the scalar path is what needs to be asked for,
  from the CLI and from the API, and it is the only path that carries the
  byte-exact guarantee.
