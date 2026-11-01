# The video VAE decode, priced at the product canvas (2026-10-31)

Phase 3 of `plan-2026-10-31.md`. The plan's brief was written from a share (the
H3 decode is "untiled" and is 77% of a warm request), and the first measurement
moved both claims. This unit prices the decoder at the canvas the product clip
actually decodes, finds the reason the product's own request fields never reached
it, fixes that, and states what is left before 30 s is reachable.

Spark `gx10-e1d2`, sm_121a, CUDA 13.0. `nvidia-clocks.service` does not exist on
this host (`systemctl status`: "could not be found"; persistence mode Enabled),
so every number here is at boost clocks: 2,327 MHz during the F32 arms (82 C) and
2,411 MHz during the fast ones (65-71 C, max 3,003). The drift pair in section 2
bounds the day's stability at 0.6%.

## 1. Which decoder runs, and what the release does

One decoder carries the product clip's H3 render, and the release never runs
it:

- **The release never decodes H3 video on the product route.** It sets
  `video_decode: false` (`configs/default.json:13`) and replaces the model's own
  `decode` with a refusal (`runtime/stage1.py:81-83`), recording
  `"video_decode": False` in the run's own receipt (`:320`).
- **The product clip decodes with the H3 video VAE at 1344x768.** That decoder
  is the phase this unit prices, and it is **not** untiled: it ports the
  reference's own chunk plan (7 latent frames, 22 frames out of the first chunk,
  then 17 new frames over a 5-frame overlap, `h3_video_vae.c:1092-1093`) and its
  own tile plan (a search over 256..320 pixels with a 64-pixel minimum overlap,
  `:658-731`). At the product canvas that plan is **7 chunks x 15 tiles = 105
  submissions of 2,805 rows** and 3,780 block calls.

The plan's own overhead is arithmetic, not waste: 105 x 2,805 = 294,525 rows run
where the clip's cells need 149,184, so **1.97x** - 1.49x of it the spatial halo,
1.32x the chunk overlap, both the reference's own rules.

## 2. The arms at the product canvas

New gate `crates/ds4-core/examples/h3_vae_gate.rs` (`make tests/h3_vae_gate`,
weights from `H3_VAE`, exit 3 when unset), one shape, one binary. `H3_VAE_GEMM`
and `H3_VAE_ATTENTION` are the released words; the gate reads none of them
itself.

The latent is `/mnt/h3/upscaled.bin`: 24x37x48x84, the file an isolation run
wrote at 12:44 from `/mnt/h3/recipe.bin`. It is **not**
the product clip's own draft (`scratch/clip.bin`, a different sample), because
that clip's prompt is not in its record (section 7; the tool is
`tools/make-h3-clip.sh`). Same shape, same plan, same weights; the decoder's cost
does not depend on the values.

| arm | `H3_VAE_GEMM` | `H3_VAE_ATTENTION` | seconds | frame hash |
|---|---|---|---:|---|
| the reference | fp32 | scalar | 429.046 | `3c591d6b9f4438db` |
| reference GEMM, tensor attention | fp32 | tensor | 232.369 | `e63be2b86ca832fb` |
| the campaign's GEMM | bf16 | scalar | 256.290 | `3b647ae3038203d9` |
| BF16 operands, scalar attention | bf16i | scalar | 243.679 | `0f4e59d0762fe417` |
| **the campaign's arm** | bf16 | tensor | 60.493 / 60.856 | `a098d3599109c8fe` |
| **the fastest arm** | bf16i | tensor | 48.225 / 48.476 / 48.493 / 48.658 | `2e071efbf843eec0` |

- The product clip's recorded 64.648 s is this `bf16`/`tensor` row plus its own
  load and its thermal state: the row is 60.5 s and the clip's own decode line
  started before the weights' upload. Its log did not state the arms, so the
  attribution is by time and not by its own record - section 5 is the fix.
- The reference's F32 route is **8.9x** the fastest arm at this canvas. That is
  the same "80 s choice" `vae-bf16-arm-2026-10-29.md` corrected, at 2.15x the
  rows (that job: 1944 blocks of 2532 rows): 429.0 s here against 110.4 s there
  scaled by 3.9 would predict 430 s, so the two hosts' states agree.
- Attention dominates when it is the reference's scalar kernel: 429.0 - 232.4 =
  **196.6 s**, 45.8% of the F32 arm. With the tensor kernel it is 14.6% of the
  fast one (section 3).
- Determinism: each arm reproduces its own hash, three runs for `bf16`/`tensor`
  and four for `bf16i`/`tensor`. Drift across the session: 60.493 against
  60.856 s, **+0.6%**.

At the engine gate's warm canvas (`512x512x22`, latent 7x32x32, one chunk, the
default search's 288-pixel plan = 2x2 tiles), the same binary:

| arm | tile | seconds |
|---|---|---:|
| fp32 + scalar (the code's default arm) | 288 | 12.991 |
| bf16 + tensor | 288 | 3.251 |
| fp32 + scalar | one 512 tile | 19.574 |
| bf16 + tensor | one 512 tile | 3.185 |

The residency gate measured 12.441 s for that phase - the **default** arm, not the
one its request asked for (section 5). The reference's tile search beats one big
tile by 1.5x in the F32 arm (attention is quadratic in the tile's rows: 4 x 2273
squared against 7173 squared) and by 2% in the fast one, so the search's area
heuristic is doing its job; `H3_VAE_TILE_PIXELS` (the reference's own override,
`h3_video_vae.c:666-673`, now ported) is a diagnostic, not a lever.

## 3. Where the fastest arm's 48.5 s goes

One nsys capture of the `bf16i`/`tensor` arm at the product canvas
(the Spark checkout's `scratch/vae-bf16i.nsys-rep`, split by
`misc/decn/decn-split.py`): decoder window **46.537 s**, 72,531 kernels, busy
44.225 s (**95.0%**), 2.314 s of gaps.

| kernel | calls | total ms |
|---|---:|---:|
| `nvjet_sm121_tst_mma_128x240x64` (w1) | 3,779 | 7,578.14 |
| `h3d_uncast_kernel` (widen) | 15,116 | 6,755.55 |
| `h3d_attention_mma_f32_kernel` | 3,780 | 6,455.45 |
| `nvjet_sm121_tst_mma_128x208x64` (w2) | 3,779 | 4,294.97 |
| `h3d_swiglu_f32_kernel` | 3,779 | 4,197.36 |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3` (qkv, out) | 7,558 | 3,986.61 |
| `h3d_cast_kernel` (narrow) | 15,116 | 3,616.18 |
| `h3d_video_qkv_rope_f32_kernel` | 3,780 | 3,471.15 |
| `h3d_scale_add_f32_kernel` | 7,558 | 2,185.16 |
| `h3d_rms_norm_f32_kernel` | 7,558 | 1,512.37 |
| the three outer GEMMs, their biases, the output norm | 624 | 171.80 |
| **total** | **72,531** | **44,224.74** |

Grouped: block GEMMs 15.86 s (35.9%), the two cast passes 10.37 s (23.5%),
attention 6.46 s (14.6%), the F32 elementwise chain 7.96 s (18.0%), RoPE 3.47 s
(7.8%), the outer GEMMs 0.11 s.

The GEMMs are done: 1.423 PFLOP (294,525 rows x 134.2 MFLOP a row-block x 36) over
15.86 s is **89.7 TFLOPS**, 75.8% of this clock's 118.4 BF16 peak and 83% of the
108 TFLOPS figure the hardware notes quote at 2.19 GHz. The attention runs at
37.7 TFLOPS (32% of the BF16 rate) because it is the F32-typed MMA kernel.

Two host costs are visible and small here, unlike job A's: the per-chunk stitch
and unpack are 0.232 s a chunk (1.62 s total), and the six large gaps between a
chunk's last kernel and the next chunk's first GEMM are 165-180 ms each (~1.03 s
of the 2.31 s of gaps). The 1.65 GB readback that job A paid is inside the window
as memcpy here: 3.58 GB over 520 copies in 0.065 s, because the arm's output rows
are already BF16.

## 4. The numeric contract, at this canvas

`bf16i`/`tensor` against `bf16`/`tensor`, both arms' own frames dumped as raw
8-bit RGB through the product's own fold (`render::rgb_f32_to_u8`), 124 frames of
1344x768:

| metric | value |
|---|---|
| PSNR (r, g, b) | 61.279 / 61.608 / 61.282 dB, average **61.387** |
| PSNR min / max a frame | 60.430 / 62.653 dB |
| SSIM (all) | **0.999525** |

That is the same order as the 2026-10-29 pair at job A's canvas (frames PSNR
60.56 average, SSIM 0.9847 on the muxed MP4, max 6 of 255): a quantization-level
deviation, four orders below the value scale, and invisible next to the draft's
own aliasing. It is a numerical change and it stays the operator's call; the
reference's F32 route remains the code's default.

## 5. The defect this found: the request's arms never reached the decoder

The engine's `RenderRequest` carries `vae_attention` and `vae_gemm`, the CLI has
`--vae-attention` and `--vae-gemm`, and `LoadOptions` maps both to
`set_vae_option` - on the **DiT's** context. The video decoder creates its own
context in `Decoder::load`, so every one of those settings was silently dropped
and only `H3_VAE_*` ever configured it. The residency gate asked for
`VaeAttention::Tensor` and `VaeGemm::Reduced` and measured the default arm
(12.4 s where 3.0 s was available).

Fixed by carrying the arms with the call: `vae_video::Arms { attention, gemm }`,
where `Some` states the value and `None` leaves the context with the environment's
word (the reference's path when unset). `RenderRequest` and `LoadOptions` state
`Some` (a request that carries a value means it), the CLIs' flags now default to
"not stated" so an exported `H3_VAE_*` still governs, and the two gates pass
`Arms::default()` because the environment **is** their switch.

Re-measured with `H3_VAE_*` unset: the engine gate's phases go 36.579 -> **26.676 s**
for the first request and 16.146 -> **6.677 s** for the warm one, its video phase
12.441 -> **3.119 s** and 12.479 -> **2.998 s**, the media hashes identical between
the two requests (`video=c63efab0...`, `frames=0afdebb4...`) and the frames hash
changed from `c1e6ad50...` because the arm changed.

A decode now prints the plan and the three arms it ran, which is what made this
findable and what a product log needs:

```
VAE latent=37x48x84 canvas=1344x768 tile=320 tiles=5x3 chunks=7 frames=124 \
    gemm=bf16 attention=tensor order=heads
```

The product clip's own 64.648 s could not be attributed from its log for exactly
this reason: the log recorded the time and not the computation.

## 6. What is left before the plan's 30 s

Measured pieces of the 48.5 s, and the projection each one supports (a projection,
not a measurement - the arithmetic is the table in section 3):

| lever | class | measured today | after |
|---|---|---:|---:|
| the two cast passes, removed by a BF16 activation chain | numerical | 10.37 s | 0 |
| the F32 elementwise chain, halved in bytes by the same chain | numerical | 7.96 s | ~4.0 |
| attention with BF16 P/V operands | numerical | 6.46 s | ~3.2 |
| RoPE in one pass instead of two (174 GB/s today) | schedule | 3.47 s | ~2.4 |
| the six chunk-boundary host gaps | schedule | 1.03 s | ~0 |
| **sum** | | | **-19.6 s -> ~29 s** |

So 30 s at this canvas is reachable, and it needs the full-BF16 chain (six new
kernels and a wider numerical surface, the decision `vae-bf16-arm-2026-10-29.md`
section 6 deferred) plus two scheduling changes. Nothing in the GEMM is left:
89.7 TFLOPS is the hardware.

## 7. What this does not establish

- **One latent, one canvas.** Every number is the product run's own latent at
  1344x768x124. The warm arms at 512x512x22 are a second shape; no
  other geometry was measured.
- **The BF16 chain is a projection.** No BF16 twin of the elementwise kernels
  exists, and section 6's "after" column is arithmetic on section 3's measured
  times, not a run.
- **Clock state.** No clock policy exists on this host; all numbers are at boost
  with the temperatures quoted. The 0.6% drift pair bounds one session, not a day.
- **The H3 video VAE encoder and the audio VAE** are untouched by all of this.
- **The product clip cannot be re-rendered from its record.** The tool that made
  it is `tools/make-h3-clip.sh`, but the structured prompt it was given lives in
  `draft-anatomy-2026-10-31.md` only as a description, so its frames, its
  64.648 s and its 133.569 s wall are a report and not a reproduction. Every arm
  above is a reproduction: the gate, the shape and the words are in this record,
  and each hash repeats. The tool now takes its arms from the environment
  (`H3_VAE_ATTENTION`/`H3_VAE_GEMM`, defaulting to the campaign's), so the
  fastest arm is one word away from a product run.
