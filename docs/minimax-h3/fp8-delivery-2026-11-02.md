# The release's FP8 route at the delivery canvas: 12.8% of a step, and detail the arm does not lose (2026-11-02)

Unit: handoff section 3 step 1. The eight-warp attention block left a delivery
step at 45.0 s, of which attention was ~27 s, so the next lever was the other
half: the release's own W8A8 E4M3 route, ported two units ago
([`fp8-route-2026-10-29.md`](fp8-route-2026-10-29.md)) and measured only at job
A's 768x768 canvas. This runs it at 1344x768x121, the job the campaign
delivers.

**The verdict the operator needs first.** The arm takes **12.8% off a
denoise step and 27 s off the job's wall** (386 -> 359 s), and at this canvas it
does **not** visibly soften the picture: detail energy falls 2.4%, and side by
side the two clips are equally sharp with *different invented detail*. What it
does cost is bytes - PSNR 22.1 dB against the BF16 arm - so it is not a drop-in
and stays an environment arm. That is the decision: **-12.8% of a step for a
render that is not the same render.** Commit: the one landing this record.
Spark `gx10-e1d2`, binary at `97ed7efd`, one binary for both arms
(`H3_DIT_GEMM=fp8`), sm_121a, CUDA 13.0.

## 1. The two renders

The delivery job as `outputs/turbo-delivery/run.log` records it: Turbo-v4
merged checkpoint, 1344x768, `--frames 121` (latent 37x48x84, 124 output
frames), 6 steps, seed 42, `--crf 15 --preset slow --vae-gemm bf16i
--vae-attention tensor`. The BF16 arm is `outputs/pipe-attn`, the FP8 arm
`outputs/fp8-attn`; same binary, same wrapper, the arm is one variable.

| arm | `TEXT` | `LOAD` | `DENOISE` | s/step | `AUDIO` | `FRAMES` | wall | clip md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 29.342 | 28.738 | 270.080 | 45.01 | 3.097 | 52.412 | **386** | `922e2ce8...` |
| FP8 | 28.640 | 39.640 | **235.419** | **39.24** | 3.085 | 49.202 | **359** | `cffb3fb9...` |

`DENOISE` -34.661 s, **-12.83%**; wall -27 s, **-7.0%**. Both arms' `TEXT` line
carries the same digest (`fnv=336a227b59fb846a`), which is the cross-check that
the arm touches the DiT's four projections and nothing upstream of them. The
clip md5 differs, as it must - this is different arithmetic, and section 4 is
the quality gate.

## 2. The `LOAD` column is page cache, not the foot of the conversion

Read alone, 28.7 s against 39.6 s looks like the price of converting 50 blocks'
four matrices to E4M3 at load, and it would eat a third of the win. It is not.
The fp8 records already found this phase page-cache sensitive
([`fp8-row-arm-2026-10-29.md`](fp8-row-arm-2026-10-29.md) section 5), so the
arms were alternated at a shape small enough to repeat - 512x512x22, two steps,
four runs:

| run | arm | `LOAD` | `DENOISE` (2 steps) |
|---|---|---:|---:|
| 1 | BF16 | 31.168 | 2.188 |
| 1 | FP8 | **28.109** | 1.418 |
| 2 | BF16 | 30.089 | 2.179 |
| 2 | FP8 | **26.711** | 1.422 |

FP8's load is *lower* both times, by 1-3 s: the reference drops the BF16
parameter it converted, and the amax and quantize run on the GPU behind the
read. The delivery job's `LOAD` gap was the checkpoint having been streamed out
of the page cache by the long render before it. The arm's real price is
`DENOISE` and nothing else, and its real gain on the job is the -34.7 s, not the
-27 s the wall shows.

## 3. Where the step's time went

`attn_bm` at this canvas's own sequence (37,722 rows, 56 heads) puts the
eight-warp block at **503.081 ms per call**, and a step is 50 of them, so
attention is 25.15 s of a step in **both** arms - the arm does not touch it.
That splits the step:

| | attention (50 calls) | everything else | step |
|---|---:|---:|---:|
| BF16 | 25.15 s | 19.86 s | 45.01 |
| FP8 | 25.15 s | **14.09 s** | 39.24 |

The non-attention part falls **-29.1%**, which is job A's own number (-29.3%)
reproduced at twice the token count. The step gains 12.8% instead
of 29% only because attention is now the majority of it: **64% of an FP8 step
is the attention block.** That is what reorders the work, and it is why step 2
of the handoff gets more valuable, not less, after this arm lands.

## 4. The quality half

`metrics.sh` (`~/ds4-test/`), against the BF16 clip, decoded yuv420p over all
124 frames:

| instrument | value |
|---|---|
| PSNR y / u / v | 22.104 / 36.107 / 36.419 dB |
| PSNR average | 23.783 dB (min 23.318, max 23.999) |
| SSIM Y | 0.717197 |
| SSIM All | 0.784757 |
| detail sigma 4 | 4.8851 -> **4.7678** (-2.40%) |
| detail sigma 8 | 8.1485 -> **8.0415** (-1.31%) |

Two things have to be read together. The chroma PSNR is 36 dB and the detail
energy barely moves - 2.4% - which says the picture is not being smoothed. The
luma PSNR is 22.1 dB, which says the two renders disagree about *content*
almost everywhere. The stills at frames 60 and 100 show both at once: the
subject, framing, wardrobe and lighting are identical and equally sharp, and the
church around her is not the same church - the FP8 arm invents a statue on the
left pier, a different window tracery, a different pew count, and moves the
candles. Neither is degraded; they are two valid answers, and the seed does not
choose between them once the projections round differently.

The arithmetic that causes it, at the delivery shape
(`tests/h3/fp8_gemm_bm 37722 5376 21504 0`): the two quantization steps are
**exact** (0 of 202,793,472 activation elements and 0 of 115,605,504 weight
elements off by more than half an E4M3 ulp), and the W8A8 product lands **rel
3.79% RMS**
against the BF16 linear over 811,173,888 elements, worst element 7.69. The same
shape at job A's 18,799 rows reads 3.79% too, so the divergence is the dtype's,
not the canvas's.

This is a smaller relative cost than job A's, and the two are not a like-for-like
pair: job A was 768x768, 107 frames, four steps, the NVFP4 encoder,
`H3_DIT_SOL=0.9`, and there the same instrument reported detail sigma 4 falling
24% (1.8839 -> 1.4284). The delivery canvas is the one that matters and it is
where the arm was least costly, so the difference between the canvases is worth
naming rather than averaging: on this job the softening the job A render showed
is not there.

## 5. What is left

- **The attention block is now 64% of a step**, at 503.081 ms per call. Handoff
  step 2 - a <=128-register variant that admits a second CTA, 16 warps instead
  of 8 - is worth up to 12.5 s of a 39.24 s step if it could halve the call,
  which is more than the FP8 arm just gave.
- **The per-row arm is the mitigation, already priced and already rejected on
  this evidence**: exact, keeps only 47% of the FP8 win, and the render showed no
  quality gain that pays for it
  ([`fp8-row-arm-2026-10-29.md`](fp8-row-arm-2026-10-29.md) section 5). Running
  it at this canvas is the check if the operator ever wants the win with less
  divergence.
- **The honest quality gate this needs is not a single frontier.** Per-pixel
  PSNR cannot say whether an arm is *better*; a teacher-forced NLL comparison
  against the BF16 arm over many positions can, and the campaign's own note says
  why single frontiers are useless here (`AGENTS.md`, "Frontier logits are
  chaotic").
- The arm stays opt-in. `H3_DIT_GEMM` is read once per load, default BF16, and
  a typo is refused rather than served as BF16.

## 6. Reproducing

    make CUDA_ARCH=sm_121 h3-rs tests/h3/fp8_gemm_bm tests/h3/attn_bm
    tests/h3/fp8_gemm_bm 37722 5376 21504 0        # the route at this shape
    tests/h3/attn_bm 37722 56 3                    # 503.081 ms per call
    H3_DIT_GEMM=fp8 ./h3-rs -d <Turbo-v4> -p "$(cat prompt.txt)" \
        --width 1344 --height 768 --frames 121 --steps 6 --seed 42 \
        --crf 15 --preset slow --vae-gemm bf16i --vae-attention tensor \
        -o outputs/fp8-attn/clip.mp4

The wrappers and instruments are `~/ds4-test/render-fp8.sh`,
`~/ds4-test/loadprobe.sh` (the alternating load probe) and
`~/ds4-test/metrics.sh` (PSNR, SSIM, detail energy); the stills are
`outputs/fp8-attn/stills/`, three frames from each arm.
