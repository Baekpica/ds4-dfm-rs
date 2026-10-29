# The 512x512, 20-step pair — scalar against tensor core

The same job run at the model's normal operating point rather than the campaign's
benchmark step count, so the video is actually denoised. Job A's 2 evaluations
are a step count chosen for comparability, and the two arms' renders at that
setting are under-denoised by design.

Command, one per arm, one subject at a time:

    h3-rs -d /mnt/h3-artifacts/MiniMax-H3 -p "<prompt>" \
        --width 512 --height 512 --frames 56 --steps 20 --seed <seed> -o <out.mp4>

`H3_DIT_ATTENTION_MMA` set selects the tensor arm; unset selects the scalar one.
Both arms are the same binary, so each pair is its own control.

## Subjects

| Scope | Prompt | Seed |
|---|---|---|
| `hq512-rain` | "A man wearing black coat is walking in the rain. He is listening the music. Cyberpunktic city night lights his gloomy face." | 42 |
| `hq512-earth` | The three-field Earth-rotating prompt in the same request shape the served API takes | 1 |

## Why the quality gap is wider here than at 2 steps

At 20 steps the arms agree less than at 2: luma PSNR 36.97 dB and SSIM 0.9779
here against 42.75 dB and 0.9868 for the 2-step job A pair. That direction is
expected and is not a defect. A per-step numerical difference compounds through
the denoise, so more steps means more divergence, and 20 steps is where the
model actually runs. The 2-step figure flatters the tensor arm; this one is the
honest number.

## Where the gain comes from, and where it does not

The denoise is 6.00x shorter, 19.292 s per step to 3.218 s. Lower than the
kernel's 29.4x, and lower than job A's 14.53x, for the same reason in both
cases: the attention is a smaller share of a step at this shape. Solving
`a(1 - 1/29.4) = 1 - 3.218/19.292` gives `a` = 0.862, so attention is 86.2
percent of the scalar step here against 96.3 percent at 18,432 tokens. The wall
keeps 3.30x of that, because the VAE decode and the text encoder sit outside the
attention and land on their own baselines: `FRAMES` within 0.125 s.

## Files

- `h3-512-both-arms.csv` — every phase for both arms and both subjects, the
  per-step figures, the speedups, both render hashes, the derived attention
  share, and the quality scores.
- `../h3-512-2026-10-27.png` — the rain pair, and `../h3-512-2026-10-27-earth.png`
  the Earth pair, both rendered by [`../plot-h3-512.py`](../plot-h3-512.py).

The clips themselves are not in the repository. They are on the Spark under
`/tmp/h3-hq-<subject>-<arm>.mp4` and were copied to `~/h3-clips/` on the machine
that drives it.
