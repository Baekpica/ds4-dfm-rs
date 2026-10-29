# H3 attention — scalar against tensor core

The BF16 attention of the MiniMax-H3 DiT, the same kernel both ways, swept
across the sequence lengths the denoise actually uses. The numbers are the
`ms_per_call` the reproducer `tests/h3/attn_bm` prints, three fresh runs per arm,
and the plot is
[`../h3-attention-2026-10-27.png`](../h3-attention-2026-10-27.png), rendered by
[`../plot-h3-attention.py`](../plot-h3-attention.py).

**Scalar** is the released kernel, byte-exact against the reference and the arm
every gate reads. **Tensor core** is the same attention built on
`mma.sync.m16n8k16`, reached only by setting `H3_DIT_ATTENTION_MMA`; unset, the
scalar kernel runs unchanged. One binary serves both arms, so the switch is the
only difference between them.

## What the two arms are, and are not

The tensor arm is a benchmark arm, never a gated one. Three deviations are
inherent to using tensor cores and are named in the source:

1. the scores accumulate through `mma`, so the reduction tree replaces the
   scalar kernel's four-element lane partial and five-step shuffle;
2. the online softmax takes one maximum per 64-key tile instead of rescaling on
   every key row;
3. P rounds to BF16 for the second `mma`, where the scalar path keeps it in
   FP32.

Scores, the running maximum and sum, and the output accumulator stay FP32, so
the arithmetic is the scalar kernel's in kind and differs only in order.

## Files

- `h3-attention-<arm>-<repeat>.csv` — one file per arm per repeat, columns
  `sequence, heads, calls, ms_per_call, tokens_per_second, out_fnv`. The
  `out_fnv` column is the reproducer's own hash of the BF16 output, which is why
  the scalar files can be checked against the campaign's gate values directly.
- `summary.json` — written by the plot script: per-metric run means, the median
  of the run means, and the ratio between the arms.

`tokens_per_second` counts query rows attended per second, that is
`heads x sequence` over the call's wall.

## Method

- The switch is read once, in `h3d_create`, so a run cannot change arms midway.
- Every arm at every length is a fresh process, and `attn_bm` runs the kernel
  twice per invocation; the plotted band is the observed min-max across the
  three repeats and the line is the per-length median.
- The sweep is the kernel alone, not the model: no weights, no other layers, so
  it isolates the attention. The end-to-end effect is the denoise phase in
  [../../minimax-h3/speed-2026-10-27.md](../../minimax-h3/speed-2026-10-27.md).
- `out_fnv` is printed for both arms but compared only for the scalar arm. The
  tensor arm is expected to differ, so it is measured against the scalar output
  numerically instead; those figures are in the same speed file.
