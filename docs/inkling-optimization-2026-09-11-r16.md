# Inkling MQ85GB prefill on GB10, 2026-09-11 (rounds 16–18)

Continuation of [rounds 13–15](inkling-optimization-2026-09-11.md). Starting
tree is merged main `1a32cd2` after PR #29. All measurements use MQ85GB on
one DGX Spark / GB10, CUDA 13.3.73, driver 610.43.02, `sm_121a` via
`CUDA_ARCH=sm_121`. MTP is off in the timed path.

These three rounds are new. Rounds 1–15 do not count toward this campaign.

Same-binary cumulative default vs all three new switches off:

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 222.96 → 328.16 | +47.18% | 1.25 → 13.78 | Improved |
| 2,048 | 229.45 → 342.31 | +49.19% | 1.27 → 17.26 | Improved |

IQ2 kill switches derepack to raw tiles, so control decode is the unaligned
fallback. Every comparison checks 1,200,348 logits (`max_abs=0`) and zero
token mismatches.

Retained commits: R16
[`19af7cf`](https://github.com/Baekpica/ds4-dfm-rs/commit/19af7cf9e41c25f86e531ae90fe516e317a01dee),
R17
[`da85ad3`](https://github.com/Baekpica/ds4-dfm-rs/commit/da85ad3f7e530ab27905714ed6eed3d93f650ba5),
R18
[`96499a9`](https://github.com/Baekpica/ds4-dfm-rs/commit/96499a9da719beaf11bfd28426a6a8f845903510).

## Protocol

- `speed-bench/promessi_sposi.txt`, 8192/2048 input tokens, 64 greedy output
  tokens, MTP off, context allocation input+65. Prefill chunk default is 512
  through round 17 and 1024 in round 18.
- Three unprofiled fresh processes per side, each preceded by a separate
  warmup, same resident base+MTP VMM owner.
- `ds4-perf scout --proof --repeats 3 --cache-policy warmup-then-fresh`.
  All six main shards, MTP, prompt, template/tokenizer files and IPC
  manifest hashed. Guard max/high 12/10 GiB, host reserve 12 GiB.
- Each retained round compares one new diagnostic switch with the default
  on the same binary. Full-vocabulary logits and generated IDs must match;
  prefill, decode and first-token latency pass `compare --regression` with
  verdict `Improved` (>1% nonoverlapping prefill, decode/first-token not
  slowed >3%).

## Round 16: fused IQ2_XXS experts from aligned SoA

Fused `w13` is IQ2_XXS with the K2 gate/up shape. The existing SoA
repack matched only `.ffn_{gate,up}_exps`, so the 38 GiB expert stack
stayed in 66-byte blocks. The loader now admits `.mlp.experts.w13_weight`,
replaces raw with the d/scale/qs SoA artifact, and keeps the MMVQ tile
reduction. Decode uses the same tile.
`DS4_INKLING_NO_IQ2_ALIGNED=1` derepacks to raw 66-byte tiles.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 263.31 → 319.16 | +21.21% | 2.20 → 14.04 | Improved |
| 2,048 | 272.45 → 334.87 | +22.91% | 2.28 → 17.69 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 263.93 / 263.31 / 262.54; default: 319.93 / 319.16 / 318.49 tok/s.
- 2,048 control: 272.45 / 271.92 / 272.50; default: 335.02 / 334.84 / 334.87 tok/s.

The kill-switch control derepacks SoA back to unaligned 66-byte tiles, so
decode on that path is the raw MMVQ fallback (~2.2 tok/s). Default decode
uses the aligned tile; this round does not claim a decode-only optimization.
Prefill is the retained win.

Native: aligned vs raw exact in `tests/test_inkling_batch`; 515-token
`test_inkling_forward` `max_abs=0`. Commit
[`19af7cf`](https://github.com/Baekpica/ds4-dfm-rs/commit/19af7cf9e41c25f86e531ae90fe516e317a01dee).

Benchmark binary SHA-256:
`ddfbe18ef3a48281defa6a2f03799e48f3bb835c8d778bf6acfd4f8a14abffb4`.

## Round 17: IQ2_XS down from aligned SoA

Fused `w2` is IQ2_XS in 74-byte blocks. After round 16 the XXS SoA path
still left down on unaligned loads. The loader admits
`.mlp.experts.w2_weight`, replaces raw with a d/scale/qs SoA artifact, and
keeps the same MMVQ tile reduction. Decode uses that tile.
`DS4_INKLING_NO_IQ2_XS_ALIGNED=1` derepacks to raw 74-byte tiles.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 261.75 → 320.37 | +22.40% | 2.32 → 13.81 | Improved |
| 2,048 | 271.66 → 335.65 | +23.55% | 2.40 → 17.32 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 262.33 / 261.68 / 261.75; default: 321.45 / 320.37 / 319.91 tok/s.
- 2,048 control: 271.98 / 271.61 / 271.66; default: 336.87 / 335.65 / 335.31 tok/s.

Same derepack caveat as round 16: the XS kill switch is the unaligned
fallback. Prefill is the retained win.

After this round the 8K candidate trace is 25.79 s prefill wall, GPU
coverage 99.8%. Top kernels: IQ2 tile 41.9% (10.64 s), shared Q8 tile
22.2%, BF16 linear tile 11.7%, grouped attention 9.0%.

Native: aligned-xs vs raw exact; 515-token forward `max_abs=0`. Commit
[`da85ad3`](https://github.com/Baekpica/ds4-dfm-rs/commit/da85ad3f7e530ab27905714ed6eed3d93f650ba5).

Benchmark binary SHA-256:
`a76ad8d688985e6c1cc7bb7238f6193fdf405d5c44efaf50d9018beb98891c70`.

## Round 18: default prefill chunk 512 → 1024

Launch gap was 0.07% of prefill wall, so the remaining linear 11.7% and
grouped-attention 9.0% are not launch-bound. 1024 tokens doubles expert
assignments per chunk versus 512 and halves chunk count on the 8K shape
(16 → 8). Graph scratch grows with cap (campaign context 8257: 645 MiB at
512, 974 MiB at 1024). Arithmetic stays chunk-invariant. 8192 as a default
previously regressed. `DS4_INKLING_PREFILL_CHUNK=512` restores the prior cap.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 319.76 → 326.48 | +2.10% | 13.78 → 13.78 | Improved |
| 2,048 | 335.35 → 341.61 | +1.87% | 17.28 → 17.27 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 320.45 / 319.76 / 319.17; default: 329.02 / 326.48 / 326.46 tok/s.
- 2,048 control: 336.31 / 334.69 / 335.35; default: 344.19 / 341.61 / 341.45 tok/s.

Decode and first-token envelopes stay within 0.3%. The 8K candidate trace
falls to 25.23 s; IQ2 tile 10.64 → 9.95 s with launches 1184 → 592. Linear
tile wall is essentially unchanged (2.97 → 2.93 s) at half the launches.
The gain is expert-tile fill from 2× assignments, not a new linear/attn
kernel.

Native: `tests/test_inkling_session --memory-quotes` walks cap 1024, asserts
the default is 1024, and checks campaign-context scratch at 1024 exceeds
512. 515-token forward `max_abs=0` (one chunk at 1024 vs two at 512).
`cargo test -p ds4-perf` accepts 1024 as a reviewed chunk value. Commit
[`96499a9`](https://github.com/Baekpica/ds4-dfm-rs/commit/96499a9da719beaf11bfd28426a6a8f845903510).

Benchmark binary SHA-256:
`95e548e537bb2ba2163bc76dbb7119f5d63365303fb27e01255392de8955a97e`.

## Rejected probes

- Shared Q8 SoA (additive, not replace): 8K overlapping Pass
  (319.39–320.14 vs 319.69–321.60 tok/s). Reverted.
- Linear panel at 512 rows: isolated 512×4096×4096 3704 vs tile 3713 µs.
  Reverted `PANEL_MIN_ROWS` to 4097.
- IQ2 XXS R=8 vs R=4: isolated 512-token tile 15.86 vs 12.44 ms. Reverted.
- Linear F32 fuse: isolated 512×4096×4096 fused 3879 vs staged 3601 µs.
  Reverted.

None of these count as a retained round.

## Cumulative

Same-binary default versus all three new switches off
(`DS4_INKLING_NO_IQ2_ALIGNED=1`, `DS4_INKLING_NO_IQ2_XS_ALIGNED=1`,
`DS4_INKLING_PREFILL_CHUNK=512`). The IQ2 kill switches derepack to raw
tiles, so the control decode path is the unaligned fallback. Prefill is
the campaign metric.

| Input | Prefill control → default (tok/s) | Gain | Decode control → default (tok/s) | Verdict |
|---|---:|---:|---:|---|
| 8,192 | 222.96 → 328.16 | +47.18% | 1.25 → 13.78 | Improved |
| 2,048 | 229.45 → 342.31 | +49.19% | 1.27 → 17.26 | Improved |

Every comparison checks 1,200,348 logits (`max_abs=0`) and zero generated-token
mismatches. Three throughput samples per side:

- 8,192 control: 223.06 / 222.96 / 222.54; default: 329.13 / 328.16 / 325.81 tok/s.
- 2,048 control: 229.70 / 229.42 / 229.45; default: 343.26 / 342.31 / 341.80 tok/s.

Two additional unprofiled `ds4-bench-perf` launches on the default path:

- 2,048: 344.06 / 343.13 prefill tok/s, 17.28 / 17.28 decode tok/s.
- 8,192: 328.93 / 327.41 prefill tok/s, 13.79 / 13.78 decode tok/s.

## Final validation

- `tests/test_inkling_session --memory-quotes` on the round-18 binary:
  default cap 1024, campaign-context scratch 973,649,152 bytes at 1024 vs
  644,879,616 at 512.
- 515-token `test_inkling_forward`: `max_abs=0`, committed KV/convolution
  93,327,360 bytes exact, chunks 2/3/7 exact.
- `tests/test_inkling_batch` and `cargo test -p ds4-perf` pass.
- Retained commits: R16
  [`19af7cf`](https://github.com/Baekpica/ds4-dfm-rs/commit/19af7cf9e41c25f86e531ae90fe516e317a01dee),
  R17
  [`da85ad3`](https://github.com/Baekpica/ds4-dfm-rs/commit/da85ad3f7e530ab27905714ed6eed3d93f650ba5),
  R18
  [`96499a9`](https://github.com/Baekpica/ds4-dfm-rs/commit/96499a9da719beaf11bfd28426a6a8f845903510).
