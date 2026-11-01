# The detail question, settled against the source of truth

**2026-11-01.** The operator rejected `outputs/final-refined.mp4` as "super bad
detail", then caught two diagnostic arms (`detail-release`, `detail-0.95`) for
turning the fox into a cat. This record answers the only question that
distinguishes a port defect from a property of the route: **run
`/data/Sol-h3-Spark`'s own stage 2 on the same draft latent and compare.**

Answer: the port is faithful. At the reference's own setting our picture is the
reference's picture - same feline, same detail, within 2 %. Detail and the fox
are not separable on this route, by the reference's own behaviour.

## 1. What was installed, and the three snags

The reference had never run on this box; there was no torch anywhere. Built a
venv (`~/torchenv`, never global, per the operator's instruction):

| piece | version | check |
| --- | --- | --- |
| torch | 2.12.0+cu130 | `cuda.is_available()` True, device `NVIDIA GB10`, cap **(12,1)** = the SM121 the reference asserts |
| torchaudio | 2.11.0+cu130 | |
| deps | numpy 2.5.4, transformers 5.14.1, safetensors 0.8.0, accelerate 1.14.0, av 18.0.0, imageio, imageio-ffmpeg, einops | all import |
| sources | `LTX-2` @ `d1511477`, `h3-upscaler` @ `d7c01b90` | upscaler source sha256 `744063b43e0f...` **equals** `configs/dependencies.json`'s `source_sha256` |
| kernels | `ltx-kernels` built, `all2all_cpp` imports | |

Three snags, each fixed on our side; the reference code was never edited.

1. **`ltx-kernels` would not build under pip's build isolation.** The overlay
   ships its own CUDA headers beside `/usr/local/cuda` and CCCL refuses the
   mixture ("CUDA compiler and CUDA toolkit headers are incompatible"). Fixed
   with `--no-build-isolation` plus `TORCH_CUDA_ARCH_LIST=12.1`; the default was
   `compute_120`, and this GPU is SM121. The opposite holds for `ltx-core` and
   `ltx-pipelines`, which need the `uv_build` backend, i.e. isolation **on**.
2. **`/data/Sol-h3-Spark` is a sparse checkout.** Its cone is
   `models/minimax_h3/{GB200,Sol-H3,Sol-H3-Spark,super_acceleration}` +
   `techniques/sparse_backends`, so `techniques/methods/` and
   `techniques/transforms/` are tracked but never materialised. `import
   techniques` then fails with a circular-import message that is really a
   missing directory. Recovered with
   `git archive HEAD techniques/methods techniques/transforms` (9 files) so the
   operator's working tree and cone were not widened.
3. **The prompt cache was rejected.** `validate_payload` compares each
   `tensor_manifest` entry against `{"shape": list(tensor.shape), ...}` and our
   cache stored `"shape"` as a `torch.Size`. A tuple subclass never equals a
   list, so `torch.Size([1,1024,4096]) != [1,1024,4096]` failed the whole
   cache even though every tensor was the right shape, dtype and finiteness.
   Rewritten to `~/solh3ref/prompt-cache.ref.pt` with **only that field
   changed**; prompt, recipe, fingerprint and both tensors are identical.

## 2. The input is our own draft, not a new seed

The reference stage 2 refines a stage-1 *capture*. Our draft is the same latent
in a different container, so it was converted rather than re-rendered - this is
only a controlled comparison if the input is identical.

- `h3_normalized` (1,24,37,24,42) bf16 from `/mnt/h3/recipe.bin`: 1,790,208 B,
  an exact match for the reference's `H3_SHAPE`. bf16 is widened by shifting
  each `uint16` into the high half of a float32, which is exact.
- `audio` (1,2,161333) f32 from `/mnt/h3/recipe.pcm`. Our PCM is 165,600 frames
  per channel; the wire contract is 121 frames at 24 fps = 5.0417 s at 32 kHz,
  so the tail **4,267 frames are dropped** (they belong to the padded canvas,
  which the reference does not emit).

Driver: `scratch/run-ref-stage2.py` starts
`torchrun --nproc-per-node=1 -m runtime.worker --module stage2`, then `prepare`
and `run`. Reference stages on our capture:

| phase | s |
| --- | ---: |
| joint video+audio stage 2 | 111.247 |
| VAE decode (tiled) | 7.088 |
| stream + h264 mux | 7.754 |
| h3 upscale + adapter | 2.296 |
| audio VAE encode | 0.374 |
| **run total** | **122.0** |

Output `refined_1344x768_121f.mp4`, 1,940,314 B, 1344x768, 121 frames.

## 3. The measurement

One metric, fixed before the comparison, invariant to a global contrast change
so a tone stretch cannot pass for sharpening:

`norm detail = std(3x3 Laplacian) / std(luma)`

| clip | luma_std | hp_std | **norm detail** | detail r | contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| draft (transfer only) | 48.43 | 2.871 | 0.0593 | - | 1.000 |
| **source of truth, its route** | 52.98 | 3.599 | **0.0679** | 0.0305 | 1.094 |
| **our port, same setting** | 53.49 | 3.704 | **0.0693** | 0.0260 | 1.104 |
| our port, anchor 0.7 (shipped) | 57.34 | 3.337 | 0.0582 | 0.1996 | 1.184 |

**Our port reproduces the reference at the same setting:** +2.1 % on normalized
detail, +0.9 % on contrast, and both repaint the texture almost entirely
(`detail r` 0.026 against 0.031). The two pictures are the same picture.

And that picture is a **feline**: both drifted, because mask 1.0 is the
release's generative route (9.06 % of the draft seen once at the first sigma).

The operator checked the reference's own output by eye and asked whether the
Python had "the same bug - it's not a fox, it's a cat". That is the finding in
the operator's own words: it is not a bug on either side. The route re-resolves
an animal the draft does not pin down, and the port reproduces the route.

## 4. The anchor is a tone knob, not a detail dial

Measured on the same draft, one arm per mask value:

| anchor | luma_std | hp_std | norm detail | detail r | tdiff r |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 49.36 | 2.930 | 0.0594 | 0.46 | 0.985 |
| 0.50 | 52.02 | 3.069 | 0.0590 | 0.31 | - |
| **0.70 (shipped)** | 57.34 | 3.337 | **0.0582** | 0.20 | 0.902 |
| 0.80 | 60.36 | 3.543 | 0.0587 | 0.13 | 0.840 |
| 0.85 | 61.31 | 3.628 | 0.0592 | 0.11 | - |
| 0.90 | 61.49 | 3.786 | 0.0616 | 0.08 | 0.735 |
| 0.95 | 59.08 | 3.773 | 0.0639 | 0.06 | 0.643 |
| 1.00 (release route) | 53.49 | 3.704 | 0.0693 | 0.03 | 0.494 |

Every mask up to 0.85 is **flat** against the draft. The +16 % absolute
high-frequency energy at 0.7 is entirely the 1.184x contrast stretch: the
anchor raises contrast and repaints texture, it does not find detail. Detail
appears only at 0.90+, and there the animal is a cat.

`--stage2-anchor 0.7` was chosen by `subject-drift-2026-11-01.md` to keep the
fox; it does, and this table is the price. The operator's two catches - 0.95 and
1.0 both cats - are the same carve-out the reference lives in.

## 5. What this does and does not say

- **Not a port defect.** At the reference's own setting the port's output is the
  reference's output on the same input. The block-level gates were the right
  instrument and they held.
- **The route has one detail level and one identity level, and they coincide.**
  Getting a sharper animal from this two-stage route requires the reference's
  own redraw, which changes the animal.
- **The ceiling is structural.** The LTX latent is 24x42 for a 1344x768 frame
  (32x spatial compression) while the draft's H3 latent is 24x42 for 672x384
  (16x); the refiner cannot exceed the upscaler's grid.
- **A native 1344x768 stage-1 render is 1.63x the draft's normalized detail**
  (`outputs/hq-native.mp4`, 0.0966) - but that is a different route and is not
  a stage-2 setting.

Not claimed: that the reference is "correct to look like this", or that 0.0693
against 0.0679 is a match to BF16 precision. It is a same-picture result at the
level this question needs.

## 6. Artifacts

| item | where |
| --- | --- |
| reference output on our draft | `~/solh3ref/stage2run/work-151708/out/refined_1344x768_121f.mp4` (Spark) |
| capture bundle | `~/solh3ref/capture/{capture.json,stage1_direct_tensors.pt}` |
| repaired prompt cache | `~/solh3ref/prompt-cache.ref.pt` |
| driver / builders | `~/ds4-sol-h3/scratch/{run-ref-stage2,make-capture,fix-cache,make-paths}.py`, `fetch-ltx.sh` |
| arms | `outputs/detail-{0.3,0.5,0.8,0.9,0.95}.mp4`, `outputs/img2img-s{50,65}.mp4` |
| drifted control, renamed | `outputs/control-release-catdrift.mp4` |
| three-way sheet | `docs/minimax-h3/frames/detail-vs-truth-2026-11-01.png` |
| port-vs-truth sheet | `docs/minimax-h3/frames/detail-port-vs-truth-2026-11-01.png` |

## 7. Reproduce it yourself: one prompt, three clips

`~/ds4-sol-h3/sol-prompt.sh` (tracked as `misc/scratch/sol-prompt.sh`) is the
comparison as a command instead of a procedure:

```sh
~/ds4-sol-h3/sol-prompt.sh "A red fox trots through fresh snow at dawn, crunching footsteps."
```

One prompt in, three clips out, all on the same draft:

| arm | what it is |
| --- | --- |
| `draft-transfer.mp4` | upscaler, adapter and LTX VAE decode, refiner off (`--transfer-only`) |
| `port-anchor0.7.mp4` | our refiner, the shipped deliverable route, tensor-core arms |
| `truth-reference.mp4` | `/data/Sol-h3-Spark`'s own Python stage 2, release sigmas |

plus the normalized-detail table of section 3 and `sheet-2.5s.png` at 1:1.

**The prompt drives stage 1 only.** Both refiners are conditioned on the
reference's fixed prompt ("4K, refined, high quality, cinematic detail, clean
textures, natural motion.") because that is the reference's contract -
`runtime/prompt_cache.py` refuses any other prompt for stage 2. Letting the
prompt into one refiner and not the other would compare conditioning, not code.

Cost, measured on this Spark: stage 1 79 s, draft control ~10 s, port ~80 s,
reference 122 s - about six minutes. One GPU at a time; the script refuses to
start beside another `h3-rs`/`h3-sol-rs`/`runtime.worker`.

**The stage-1 arm is exact, not approximate.** With the campaign's own prompt
and seed the script's latent hashes to `2cb3e8d2...` - the same digest as
`/mnt/h3/recipe.bin`. So a comparison run starts from a byte-identical draft,
which is what makes "same input, only the code differs" true rather than
hopeful. The reference arm needs 672x384 and 124 frames (a 24x37x24x42 latent
and 161333 audio frames); the script refuses any other canvas unless
`--skip-truth` says the reference is out of scope.

Its text is not always one line, so `-p FILE` reads the prompt from a file and
`-i IMAGE` appends a Ref2VA reference (`--ref-image`, in order, addressed as
`<Picture 1>`, `<Picture 2>`, ...). The full prompt is also written into the
run directory as `prompt.txt`.

## 8. A second prompt, the timings, and the anchor's other end

Run again on a textured prompt instead of the fox: a 260-word stop-motion beat
sheet (a felt raccoon stealing a strawberry off a tart), 361 text tokens
against the fox's 15, 2026-11-01 15:38:13-15:44:05.

| step | Rust (`h3-rs` + `h3-sol-rs`) | Python (`/data/Sol-h3-Spark`) |
| --- | --- | --- |
| stage 1 draft, 124 f @ 672x384 | 81 s | not applicable |
| capture into the reference's container | 3 s | - |
| stage-2 denoise, 121 f @ 1344x768 | **35.0 s** (3 steps of 11.6) | **112.8 s** |
| VAE decode + mux | 16.3 s | 15.4 s |
| stage-2 wall, including its own load | **90 s** | **157 s** (26 s session start + 124 s request + 7 s copy) |
| transfer-only control | 20 s | - |

The prompt itself only reaches stage 1, so prompt-to-finished-clip is 174 s in
Rust and was 194 s with the control. The Python reference cannot make a draft
at all: it needs one of ours first and then spends a further 157 s.

The denoise comparison is 3.2x, the end-to-end stage 2 is 1.7x, and the one
line Rust loses is the decode (16.3 s against 15.4 s). The earlier 7.1 s decode
figure for the reference counted only its VAE pass; the remaining 7.7 s is the
streaming decode and h264 mux, so the two decoders are effectively equal.

Detail on this prompt moved the picture:

| clip | norm detail | vs draft |
| --- | --- | --- |
| draft (transfer only) | 0.1212 | - |
| port, anchor 0.7 | 0.1246 | x1.03 |
| truth (reference Python) | 0.1742 | x1.44 |
| port, anchor 1.0 | **0.1809** | x1.49 |

On the fox all three arms sat at 0.058-0.069 and the reference added nothing;
here it adds 44 %, which at first looks like the port under-delivering. It is
the anchor, not the port: at 1.0 - the release's own generative route, no
draft preserved - the Rust arm measures 0.1809, 1.04x the source of truth, in
85 s and without regenerating the draft. So the ceiling in section 4 is a
ceiling on the *shipped anchor*, and the same binary reaches the reference's
detail when it is allowed the same freedom. What the draft-preserving anchor
buys is the draft's structure; what it costs is texture.

Rust only, no Python arm, from the same draft:

```sh
~/ds4-sol-h3/sol-prompt.sh --name rust-only -a 1.0 --skip-truth --skip-draft \
  --prompt-file scratch/raccoon-prompt.txt
```

The GPU over that run, sampled every 10 s: 33.9 W mean, 91.5 W peak, 69 C max,
2296 MHz SM mean (2483 peak), 42 % utilisation mean and 96 % peak. No
throttling and far from any power limit - this box is memory- and
latency-bound, which is why the resident Rust route wins: it keeps the 34.6 GiB
of refiner weights loaded and pipelines straight through, where the reference
re-dispatches and reallocates per request.

One operational note: the `outputs/` tree was emptied between 15:44 and 15:51
on 2026-11-01, so the artifact paths in section 6 and the three-arm clips from
this run no longer exist on the box. Every one of them is reproducible with the
commands above; the stage-1 latent regenerates to the same `2cb3e8d2...`
digest, so a rerun starts from the same draft.
