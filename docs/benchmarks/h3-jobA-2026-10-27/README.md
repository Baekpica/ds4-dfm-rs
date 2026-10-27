# Job A, both arms, same binary

Job A is the 768x768, 107-frame, 2-step workload this campaign benchmarks with:
`-d /mnt/h3-artifacts/MiniMax-H3 --width 768 --height 768 --frames 107 --steps 2
--seed 42`, with the prompt "A man wearing black coat is walking in the rain. He
is listening the music. Cyberpunktic city night lights his gloomy face."

Both rows come from the same binary, built from `feat/h3-attn-mma`, and the only
difference between them is the `H3_DIT_ATTENTION_MMA` switch. That is the point
of the file: earlier rows in
[../../minimax-h3/speed-2026-10-27.md](../../minimax-h3/speed-2026-10-27.md)
compare the tensor arm against a baseline recorded from a build made before the
dormant path existed, and this is the like-for-like control.

## What to take from it

- The denoise is 14.53x shorter, 292.634 s per step to 20.147 s per step.
- The wall is 2.97x shorter, not 14x, because `FRAMES` and the text encoder are
  outside the attention and land on their own baselines: `FRAMES` within 0.203 s
  of the switch-off number.
- The switch-off render is byte-identical to the campaign's gate render,
  `69816f7f...`, 1,769,313 bytes. The dormant path therefore cannot have
  perturbed the released one, not even incidentally.
- The switch-on render differs, which is expected: the two arms cannot agree
  byte for byte, because `mma` changes the reduction tree and P rounds to BF16.
  Against the gate render it measures PSNR 42.75 dB luma and SSIM 0.9868.

`scale` is the attention's share of a step, derived as `a(1 - 1/29.4) = 1 -
20.147/292.634`, giving 96.3 percent. The remaining 10.7 s per step is the rest
of the DiT and is the next thing worth attacking if the step is to fall below
20 s.

## Files

- `h3-jobA-both-arms.csv` — every phase for both arms, the two speedups, both
  render hashes, the derived attention share, and the quality figures.
- `../h3-jobA-2026-10-27.png` — the figure, rendered by
  [`../plot-h3-jobA.py`](../plot-h3-jobA.py) in the same matplotlib convention as
  the rest of this directory: the headline pair on top with the values on the
  bars, the full phase breakdown below on a log axis, which is what shows that
  only the denoise moves.

The kernel-level sweep across sequence lengths, three runs per arm, is a
separate folder: [../h3-attention-2026-10-27](../h3-attention-2026-10-27).
