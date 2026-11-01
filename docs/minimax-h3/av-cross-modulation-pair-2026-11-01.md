# The AV cross-modulation pair is (scale, shift), not (shift, scale)

Date: 2026-11-01. Branch `feat/sol-h3`, on top of `d48df326`.

## What was wrong

`BasicAVTransformerBlock` modulates the AV cross-attention operands through
`get_av_ca_ada_values` (`ltx2-pin/.../transformer.py:198-222`), which unpacks
the table pair as

```python
scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
```

- the first row of the pair is the **scale** and its second row is the shift.
  The block's own AdaLN unpacks the other way round (`shift, scale, gate = ...`,
  `:272`), and `ModalityScratch::ada(0)` is the shift buffer, so `av_modulate`
  wrote the pair into the wrong two buffers. The AU gate row is a single row
  (`:211-218`) and was unaffected, which is why every aggregate statistic of
  the audio state matched while the picture collapsed.

The four modulated operands were therefore normalized with a swapped affine,
and the A2V/V2A attentions injected a spatially uniform error into all video
rows of all 48 blocks. That is the smear recorded in
`audio-state-ab-2026-11-01.md` on the audio-conditioned arm.

## Evidence

`h3_ltx_refiner_gate --edges --signals --dump` writes every tensor one block
reads; `scratch/agold/block_golden.py` runs the release's own
`BasicAVTransformerBlock` over those dumps (stepped through the reference's own
methods, asserted bit-identical to `forward`), and `--compare` reports each
checkpoint. The f32 reference against the device's BF16 kernel has a floor of
`5.4e-3` rel_rms (the same block run in bf16 vs f32), and at rows=8,
audio_rows=4 two arms of block 0 measured:

| checkpoint | before | after |
| --- | --- | --- |
| `a2v-context` | cosine 0.040, rel_rms 1.01 | cosine 0.99998, rel_rms 7.2e-3 |
| `a2v-query` | cosine 0.060, rel_rms 1.05 | cosine 0.99998, rel_rms 6.9e-3 |
| `a2v-out` | cosine 0.015, rel_rms 1.00 | cosine 0.999999, rel_rms 1.4e-3 |
| `v2a-context` | cosine 0.006, rel_rms 1.03 | cosine 0.99997, rel_rms 7.9e-3 |
| `v2a-out` | cosine 0.088, rel_rms 1.00 | cosine 0.99983, rel_rms 2.0e-2 |
| `video-after-av` | rel_rms 3.3e-2 | rel_rms 6.4e-3 |
| `audio-after-av` | cosine 0.982, rel_rms 3.1e-1 | cosine 0.99999, rel_rms 6.5e-3 |
| block `video` | rel_rms 2.4e-2 | rel_rms 5.2e-3 |

The two remaining outliers are the raw text cross-attention outputs
(`video-text-out` 1.4e-1, `audio-text-out` 2.8e-1 rel_rms) whose own magnitude
is near zero against a residual rms near 50 (`max_abs` 0.41 and 1.48), so BF16
storage alone dominates their relative error; after the residual add the
checkpoint is at the floor (`video-after-sa-text` 4.5e-3).

Everything else - the video self-attention, its text cross-attention, both
feed-forwards and the whole audio branch - already sat at the floor, so the
port's block arithmetic was correct apart from this pair.

## What landed

- `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`: `av_modulate` writes the
  pair as (scale, shift), with the reference cited at the site.
- `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`: an `H3_BLOCK_DUMP=DIR`
  instrument that writes every state and AV operand of a block, unset by
  default, for the checkpoint diff above.
- `crates/ds4-core/examples/h3_ltx_refiner_gate.rs`: `--signals` now dumps the
  remaining block inputs (video cross sin, both contexts, the audio prompt
  modulation, the audio cross gate/cos/sin) and every block's audio state.

## Not yet measured

The audio-conditioned arm's picture, after the fix. The block is at the floor;
the render has not been re-run to completion.
