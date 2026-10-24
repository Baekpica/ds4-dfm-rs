# MiniMax-H3 after the port: speed, serving, delivery, conditioning

The default text-to-video path is complete and bit-identical to the reference
end to end (`port-plan.md`, E1-E9). This page is the work list from there:
P8.5 the upstream branch survey, P9 speed, P10 serving with a resident engine,
P11 delivery formats (YouTube, Instagram, the highest resolution the model
has), P12 the conditioning and editing surface, P12.5 the reference test suite
ported to Rust, P13 close.

Each phase ends with a gate that runs on the Spark and a record in
`port-plan.md`. A phase is not done because code exists; it is done when its
gate has produced the numbers and the hashes on the reference workload.

## The contract every phase inherits

These are the reference's limits, read from the source and not negotiable by a
delivery preset:

| Limit | Value | Source |
|---|---|---|
| Canvas edges | >= 32, multiples of 32 | `h3.c:488-520` |
| Canvas area | <= 768*1344 = 1,032,192 px | `h3.c:488-520` |
| Render canvas | same aspect to the pixel, no larger than output | `h3.c:488-520` |
| Frames | 5 + 17k, generation needs >= 22, aligned <= 362 | `h3.c:869-874` |
| Frame rate | 24, fixed | `h3_host.h:9` |
| Audio | 32 kHz stereo, f32, from the AudioVAE | `h3_audio_vae.c`, E7 |
| Pixel output | RGB24, scale 255, round half to even | `h3.c:731-742` |
| Resize | f64 bilinear, half-pixel centres, edge clamp | `h3_host.c:532-635` |
| Mux | rgb24 + f32le pipes, libx264 fast crf 18 yuv420p, AAC 192k, faststart | `h3_ffmpeg.c:670-682` |
| PCM conditioning | clamp to +-0.5, then `alimiter=limit=0.5` | `h3_ffmpeg.c:641-648` |

Consequences a later phase must not paper over: 1920x1080 (2,073,600 px) and
1080x1920 are above the released area, and 32 kHz audio cannot gain bandwidth
by resampling. Delivery presets upscale the model's own canvas with the
reference's bilinear resize; no preset can add detail the model never
produced.

Review rule (operator, 2026-10-24): every result shown to the operator is a
quality render, 20 evaluations at a real canvas. Short-circuited runs (2
evaluations, smallest canvas) are plumbing and parity checks only; they are
never presented as a result, because the operator reviews the picture every
time. A parity check may use 2 evaluations when its evidence is an MP4 hash
against the CLI at the same flags.

## P8.5 — The upstream M3.1 branch: what it can and cannot give H3

Runs before P9.1, because three of its rows change what P9 measures. It makes
no source change and adopts nothing on inspection alone.

What the branch is. `origin/feat/add-minimax-vq` at `fd7192f4`, 31 commits
over the merge base `600e465e`, read-only clone at
`/data/ds4-minimax-vq-ref` (clean, at `fd7192f4`, 260 MB). It is
MiniMax-M3.1-Flash: a 60-layer text MoE with 64 query / 4 KV heads, an MSA
block indexer, 128 experts top-4, NVFP4 K/V pages and a DSpark drafter
(`ds4_minimax.h:18-40`, `docs/minimax-vq.md:1-46`). It carries no H3 file:
`git ls-tree -r --name-only origin/feat/add-minimax-vq | grep h3` is empty.
Same repository, different model. H3 is a 50-block diffusion transformer over
a packed text+video+audio sequence with no autoregressive decode.

The KV question, answered before any candidate is weighed. H3 has no
attention KV cache:

- The reference's cache is conditioning memo + prepared DiT + video decoder.
  `h3_cache_info` holds exactly `embedding_entries`, `embedding_bytes`,
  `prepared_dit` and `video_decoder` (`h3.h:21-27`); `h3_cache_clear` frees
  exactly those three (`h3.c:53-64`); the decoder hit is keyed on the
  conditioning key plus the decoder shape (`h3.c:697-723`).
- The DiT recomputes attention over the whole packed sequence at every step.
  `qkv_capacity` (`h3_dit.c:1505-1516`) is a buffer bound, not a cache.
- The port matches that. The text tower's `key` and `value` tensors are
  projection weights (`text_forward.rs:458-465`) and per-forward scratch for
  every prompt token (`:521-529`), allocated inside `run` (`:199-200`) and
  released on return. Encoding is one full-sequence pass, no continuation.
- "Correct KV usage" for H3 therefore means three things, none of them a
  cache: scratch sized to the sequence and freed per call (holds today); a
  conditioning memo keyed on every input that changes the output
  (`h3_conditioning_key`, `h3.c:143-166`; `h3_prepared_key`, `:168-190`),
  which is absent today and is P10.3; and the VAE temporal-overlap blend on
  multi-chunk paths, ungated today and P9.5.
- The branch's K/V tier solves a problem H3 does not have. A claim that H3
  can "continue with the KV cache" is a new mechanism, as P12 already states.

Candidates, with a verdict per row:

| Candidate | Where | Verdict for H3 | What decides it |
|---|---|---|---|
| In-process session, resident engine, first-use module loads prewarmed at boot | `ds4_minimax_session.inc:1-2`; `d1eff63e` (1.3 s of module loads once, 0.2 s added to a first request) | Adopt in P10.1 | the reference does the same (`h3.c:697-723`); P10.6's two-request timing is the gate |
| A reuse key built from every input, media pixels included | `ds4_minimax_session.inc:26-34`: identical pads over other pixels prefill again | Adopt in P10.3 and P12.4 | mirror `h3_conditioning_key` (`h3.c:143-166`), never a prompt-only key |
| Capacity as a measured knob | `fd7192f4`: 256K tier, 128 tokens, 300/107/68 misses and 10.9/4.9/3.3 ms of reads per token, 14.74/15.75/16.71 tok/s at identical tokens | Adopt the method, not the mechanism | P9's residency and prefetch-depth sweeps report misses and identical hashes the same way |
| Disk K/V tier: unlinked O_DIRECT file, pinned LRU pool, slot table, AIO completions | `ds4_minimax_kv.inc:1-25`, `:334-372` | Reject for H3 | H3 streams 735 MiB per block in a strict ring with no random misses; the transferable contract items are already recorded at `port-plan.md:376-412` |
| Pinned host memory aliased into the device | `ds4_gpu_minimax_host_alloc`, `ds4_minimax.h` | Candidate for P9.1/P9.7 only | P9.1 must first show the gap is in the staging or copy path; then measure mmap, malloc staging and pinned staging, and confirm `pageableMemoryAccess` / `canMapHostMemory` on the Spark before designing around them |
| Row count as a kernel template argument | `344dee7d`, `docs/minimax-vq.md:64-70` | Candidate for the text encode, not the DiT | the DiT repeats one shape per render, so a runtime row count costs it nothing; text rows equal the prompt length and the encode is 20-30 s of a render (P10.3) |
| PDL on short kernels | `7d9adeba`, `docs/minimax-vq.md:280-284` | Candidate for P9.7 | attribute with PDL off, as `AGENTS.md` requires, and respect the known memcpy-in-chain hang |
| Media input plumbing: base64 only, ffmpeg sampling, page-cache drop after streaming a tower | `6fd141ff`, `docs/minimax-vq.md:290-328`, `crates/ds4-core/src/minimax_vq/media.rs` | Plumbing only, in P12 | semantics stay with `/data/h3.c` (`h3_multimodal.c`, `h3_vision_encoder.c`, `h3_video_encoder.c`) |
| Session snapshot payload | `ds4_minimax_bank_payload.inc:1-44` | Reject | H3's only reuse state is the conditioning memo plus residency |
| Gate record layout: one directory per run, one CSV per sweep, `tables.md`, `summary.json` | `docs/benchmarks/minimax-vq-spark-64k-2026-10-24/` | Adopt for P9 and P11 records | every number there names its commit and command; the same discipline settles the audio and preset questions |
| Kernel-local wins: epilogue gate loads, merged indexer grid, two-pass host argmax, warp-specialized experts, tensor-core block scoring | `85086da5`, `152fa899`, `2f89a938`, `4d47f9f6`, `0a1f2696` | Reject by name | each is bound to M3.1 structures H3 does not have (indexer, 128 routed experts, argmax sampler) |
| VQ weights for the DiT or the text tower | `vq-recipe.md`, `docs/minimax-vq.md:14-46` | Out of scope here | they change bytes; that is `port-plan.md` section 5's T2 rung with its own quality gate |

Work items:

1. Write `docs/minimax-h3/upstream-m31-survey.md`: one row per candidate
   above, verdict, `file:line`, and the measurement that would reverse a
   reject. Rejected rows are recorded so a later session does not re-litigate
   them.
2. Put the KV audit in the same file with the citations above, and link it
   from `openai-videos.md` where request reuse is discussed.
3. Feed P9: name the live candidates (pinned staging, row templating, PDL)
   beside the target each would change, with its trigger measurement. None is
   implemented before P9.1-P9.3 report.
4. Feed P10: prewarm at open, and a memo key built from every input.
5. Feed P12: reuse the media plumbing only; the conditioning semantics are
   the reference's.
6. Keep the branch out of the tree: no merge, no cherry-pick of family files.
   The clone stays read-only, like `/data/YoungAi`. Only a mechanism
   re-implemented in h3 code crosses over, with its citation.

Gate: the survey file exists with a verdict and a citation per row; the KV
audit matches `/data/h3.c` and `text_forward.rs`; P9's list names which
candidates are live and what measurement decides each. No binary change
belongs to this phase.

## P9 — Speed as a port

Goal: close the measured gaps without changing bytes. Every adoption is one
measured bottleneck, one scoped change, a fresh-process A/B, and for
scheduling-only changes the E9 hashes stay bit-identical (`70973d0a2c32e378`,
`541455fe622e267f`, `62b36816b6e1248e`, `4771cab28a1fae97`).

Measured 2026-10-24 on the Spark, 256x256, 22 frames, 2 evaluations, seed 1,
one prompt, same command line for both binaries (reference probe and
`h3_dit_gate`):

| Run | Wall | Video decode | Audio decode |
|---|---:|---:|---:|
| reference, run 1 | 64.08 s | 8.165 s | 0.799 s |
| port, run 1 | 84.20 s | 11.036 s | 0.921 s |
| port, run 2 | 89.74 s | 12.854 s | 0.921 s |
| port, run 3 | 106.52 s | 13.587 s | 0.921 s |

Two things the table shows that a single pair would hide: the port's video
decode is 35-66% slower at this shape, and its wall time drifts upward across
consecutive runs (84 -> 90 -> 107 s) while the reference stayed at 64.08 s.
The drift is consistent with the load path, but it is not proof; P9.1 and the
`nsys` attribution decide it.

Audio PCM hash was `e54867f428a2ce87` in both binaries, so the samples are
identical and the operator's audio complaint is not a decode defect (P9.5,
P11.4).

Targets in the order they are attacked; none may be skipped for a later one:

1. **Weight-load source (leading hypothesis, unverified).** The reference
   mallocs a staging buffer, `pread`s the tensor into it, copies with
   `cudaMemcpy`, then frees it (`h3_gpu_cuda.cu:454-478`). All three port
   loaders copy straight out of a read-only `mmap` of the shard
   (`vae_store.rs:100-112`, `weights.rs:146-170`, `text_weights.rs:105-125`).
   A file-backed mapping faults inside the copy; a `pread` fills anonymous
   memory with normal readahead. The port's wall is 20-25 s longer at a fixed
   workload, and the loads (text 62 GiB, DiT 61.7 GiB, video VAE 9.7 GiB)
   dominate it, so this is the first thing to measure.
   Harness: `tools/minimax_h3/p9_copy_bench.cu` (one 2 GiB range, mmap vs
   `pread` staging, same copy API, cold and warm pass). If confirmed, the
   loaders get one shared `pread` staging path with a reusable buffer; the
   uploaded bytes do not change.
2. **Video decode at a small canvas.** 11.0-12.9 s against 8.2 s here, and
   11.115 against 7.159 s in the E9 256x256 run, yet only +6% at 512x512
   (22.484 against 21.162). At 256 the reference takes its streaming
   single-chunk path (`run_decoder`, `h3_video_vae.c:458-531`: load one block,
   run it, free it), while the port always loads all 36 blocks
   (`Decoder::load`) and then runs `run_resident_tile`. Attribute with
   `nsys` before changing anything: sum kernel time, sum `cudaMemcpy` (H2D and
   D2H), and measure the host gaps; the reference's own `H3_PROFILE` teardown
   line (`h3_gpu_cuda.cu:306-318`) reports its GPU/encode/wait split for the
   same run. Candidate scoped changes, in order: the per-tile allocations
   inside `run_tile` (`Buffers::new`, `latent`, `post`, `patch_hidden`,
   `zero`); the streaming path for the single-tile single-chunk case; the
   22 MB output readback plus the CPU unpack.
3. **Wide-canvas denoise.** The port measures 12.25 s per evaluation at
   864x480 against 0.67 s at 256x256 for 6.3x the rows. The reference's own
   number at 864x480 has never been taken: measure it first, at 2 evaluations
   and the same prompt. If the reference is equally superlinear this is the
   model, not the port, and the row closes as measured.
4. **Audio decode.** 0.799 against 0.921 s here, 0.802 against 1.056 s in the
   E9 20-step run. Same method: attribute inside the decoder's load and graph
   (6.6 GiB read, 36 blocks), then one change.
5. **The multi-chunk VAE path is ungated.** Every E7/E8/E9 gate ran 22 frames,
   which is one 7-latent chunk and one tile. 39 frames is 2 chunks, 56 frames
   is 3, and the server's own default (512x512, 4 s = 107 frames, `video_api.rs:67-71`)
   is 6. Gate 39, 56 and 107 frames against the reference at 256x256 and
   512x512, on frames and on the temporal-overlap blend. This is correctness,
   not speed; it runs before any preset promises a duration.
6. **Resident decoder.** The reference keeps the video VAE decoder across
   calls when its session cache is on (`h3.c:697-723`), and the port re-reads
   9.7 GiB per decode. The port needs the same residency, but it belongs to
   P10's engine so the server gets it for free. P9 only records the re-read
   cost as the baseline.
7. **Only after 1-6, and only if measured:** CUDA graph capture for the DiT
   forward, the host sampler's thread use, pinned staging. No speculative
   change; the denoise is already at parity (109.435 s against 109.698 s at 20
   evaluations).

Method per `AGENTS.md`: whole-workload measurement first, attribute inside the
hot range, one scoped change with an explicit numerical contract, an A/B with
fresh processes, the observed clock and the page-cache state recorded, and a
correctness check. Record what the measurement does not prove. A sub-1%
repeatable gain with no extra memory is worth keeping; a large gain that
changes bytes is a different phase with its own quality gate. P8.5's verdicts
decide which launch-side candidates from the upstream branch reach this list;
none is implemented before the attribution above.

## P10 — The OpenAI Videos API, end to end, with a resident engine

Today `h3-server-rs` spawns the renderer per request (`CommandRenderer`,
`h3_video.rs:107-241`), so every `POST /v1/videos` pays the full load: at the
measured rates that is over a minute before the first denoise step. The
reference's own answer is an in-process session that keeps the text encoder,
the DiT and the video decoder resident and memoizes the conditioning
(`h3.h:180-184`, `h3.c:66-80`, `h3.c:697-723`, enabled by the CLI session at
`h3_cli.c:702`). The port mirrors that mechanism; it does not invent a
different one.

1. `minimax_h3::engine`: `Engine::open(model_dir)` owning tokenizer, text
   encoder, DiT and both VAEs; `Engine::render(spec)` reusing them.
   `h3-server-rs` uses it when built with `native`; `--renderer` stays as the
   operator escape hatch and keeps the current wire contract.
2. Admission: one GPU render at a time (one device), queue position in
   `GET /v1/videos/{id}`, cancel and timeout, and a store whose restart
   behaviour is documented. Retained jobs and rendered MP4s keep their
   current rules.
3. Conditioning memo, mirroring `h3_conditioning_cache_store/load`
   (`h3.c:223-309`): a repeated prompt skips the 20-30 s text encode.
4. The default request path is the one to gate: `POST /v1/videos` with only a
   prompt renders 512x512, 4 s (107 frames, 6 chunks) at 20 steps. Run it
   through a real client, poll, download `content`, `ffprobe` it, and compare
   its MP4 against `h3-rs` for the same spec.
5. The install script runs as a script, all subcommands, on a tree that was
   not built by hand: `preflight`, `fetch`, `verify`, `build`, `gate`,
   `render`, `start`, `test`, `status`, `stop`.
6. Two consecutive requests show residency and the memo: the second request's
   time excludes the load, and the server never re-reads 134 GiB.

Gate: the transcript, the MP4 hashes, the two-request timings, and the
`ffprobe` output in `port-plan.md`.

## P11 — Delivery presets: YouTube, Instagram, the highest resolution, audio

The model canvas is the quality ceiling; a preset chooses the canvas, the
delivered frame, the encode and the container. All presets still obey P9.5
(multi-chunk) and the contract table above.

The official request names geometry as a resolution tier plus an aspect ratio,
not as pixels: `resolution` in `480P | 768P | 2K` and `ratio` in
`adaptive | 21:9 | 16:9 | 4:3 | 1:1 | 3:4 | 9:16`
(`platform.minimax.io/docs/api-reference/video-generation-v2-create`, read
2026-10-24). The model card fixes the tier: "the shorter side is set to 768
pixels by default", and 2K is H3-Regenerate-2K, a hosted module this release
does not include. 480P and 2K therefore belong to other variants (H3-Lite/Max
and Regenerate-2K) and are refused by name; this checkpoint serves 768P.

Pixels then follow from the released cap: short edge 768, long edge
`round(768 * ratio)` snapped to a multiple of 32, clamped to `768*1344`
(`h3.c:490-500`). That rule is checked against the platform's own artifact,
not inferred: `assets/h3_direct_768p.mp4` is 1344x768, square pixels, 24 fps.
`round(768 * 16/9)` is 1365.33, which snaps to 1376, and 1376x768 is
1,056,768 px, above the cap, so the platform ships 1344x768 for 16:9. The same
clamp gives 768x1344 for 9:16, while 4:3, 3:4 and 1:1 are exact at 1024x768,
768x1024 and 768x768 — the reference's own canvas presets
(`webui/frontend/src/generated/options.ts`, `CANVAS_PRESETS`).

| Preset | Model canvas | Delivered frame | Scale | Notes |
|---|---|---|---|---|
| `16:9` | 1344x768 | 1920x1080 | anamorphic 1.6 % | the platform's 768P wide canvas, exact 16:9 frame |
| `9:16` | 768x1344 | 1080x1920 | anamorphic 1.6 % | phone Reel/Shorts, the same frame for both |
| `21:9` | 1344x576 | 2520x1080 | uniform 1.875x | 7:3, so no distortion; 1568x672 would exceed the cap |
| `4:3` | 1024x768 | 1024x768 | none | exact ratio at the tier canvas |
| `3:4` | 768x1024 | 768x1024 | none | exact ratio at the tier canvas |
| `1:1` | 768x768 | 768x768 | none | exact ratio at the tier canvas |
| `native_16x9` | 1344x768 | 1344x768 | none | platform parity: the file is the canvas |
| `native_9x16` | 768x1344 | 768x1344 | none | platform parity |
| `instagram_4x5` | 768x960 | 1080x1350 | uniform 1.406x | outside the official ratio set; exact 4:5 canvas |

The operator's earlier names stay as aliases of the same presets:
`reel_phone` is `9:16`, `youtube_1080p` is `16:9`, `instagram_1x1` is `1:1`,
`cinema_21x9` is `21:9`.

### Anamorphic delivery

Both official wide and tall frames are 1.75:1 at this tier, so the exact 16:9
and 9:16 frames are a non-uniform scale of the canvas: 1344x768 to 1920x1080
stretches x by 1.4286 and y by 1.4063, a 1.6 % difference. The reference
refuses any frame that is not the canvas aspect (`h3.c:505-520`), so this is a
named deviation, never an inference: the caller sets
`RenderRequest.presentation`, `render::OutputRule::Anamorphic` is the only rule
that skips the aspect check, and the engine reports both factors
(`RenderReport.scale`, printed as `anamorphic` in the one-line summary).

No new math is involved. The reference's resize kernel already scales x and y
independently (`h3_host.c:557-635`), and the port transcribes it; only the
validator refused the mismatch. Parity follows: the same kernel, the same
half-pixel centers, the same round-half-to-even write.

A preset may not stretch beyond a quarter (`MAX_DISTORTION`, checked at
construction): past that the squeeze reads as distortion rather than as a
rounding correction, and a canvas that bad is a preset error, not a request.

Work items:

1. The preset table landed in `presets.rs`, with the area, grid, even-frame
   and distortion rules checked at construction, so an impossible preset is
   refused by name.
2. `size` on the OpenAI surface accepts the preset names and the aliases in
   addition to `WxH`; a literal `WxH` stays the model canvas.
3. Resolution ladder measured on the Spark: 1344x768 at 22, 39 and 56 frames,
   20 steps. The unified 121 GiB has to hold 61.7 GiB of DiT plus activations
   that grow with rows; record the largest canvas x frames that fits rather
   than promise one.
4. Audio. The reference muxes `-c:a aac -b:a 192k` after the +-0.5 clamp
   (`h3_ffmpeg.c:641-648`, `:670-682`). Options to measure, each a named
   deviation if adopted: keep the clamp but raise the AAC bitrate; skip the
   clamp and let `alimiter` alone handle peaks; keep the PCM in a second
   output for mastering. Peak/RMS evidence with `ffmpeg -af astats`; the
   32 kHz rate is not changed.
5. The operator's audio complaint is closed here: identical PCM at the same
   settings is already measured (`e54867f428a2ce87` both binaries), so the
   gate is the muxed track at 22, 39 and 56 frames, plus the clamp question
   above. If the model's short-clip audio is simply poor, say so with the
   measurement and do not claim a fix.

Gate: one rendered file per preset, `ffprobe` geometry/duration/streams, an
audio statistics block, and a reference comparison at the model canvas. The
first gate must include a multi-chunk request (39 frames or more): the
107-frame run exposed a tile-stride bug in the video VAE
(`vae_video.rs`, `prepare_input_rows`) that 22-frame renders never reached.

## P12 — Conditioning and editing (the unported surface)

State the semantics before porting anything: H3 has no token-level KV
continuation and no latent-input API (`h3_params` carries no initial latent,
`h3.h:60-130`). What the reference calls a cache is model and conditioning
residency (`h3_cache_info`: embedding entries, prepared DiT, video decoder).
"Editing" in H3 means conditioned generation: first/last-frame anchors and
ordered Ref2VA references. Any claim of "continue my video with the KV cache"
would be a new mechanism, not a port.

1. **First/last frame (FL2VA).** Needs image decode, the vision encoder
   (`h3_vision_encoder.h:20-31`), the multimodal presentation
   (`h3_multimodal.c`, `h3_multimodal_encode_fl2va_bf16`), and the VAE encoder
   for the anchor latents (`h3_video_encoder.h:19-30`). The anchors enter the
   layout at `h3.c:335-346`.
2. **Ref2VA.** Ordered image/video/audio references through
   `h3_multimodal_encode_ref2va_bf16` and the Ref2VA partition of the
   artifact. Video references need frame sampling and audio resampling;
   audio references use the released preprocessing.
3. **The wire.** `input_reference` is parsed today and refused by name
   (`video_api.rs:386-405`). Once 1 lands, base64 `image_url` maps to the
   first-frame anchor. The server never fetches a URL (project rule).
4. **The session surface.** `h3-rs` gains `--first`, `--last`, `--ref-image`,
   `--ref-video`, `--ref-audio` with the reference's spellings, and the
   server's memo keys include them, as the reference's conditioning key does
   (`h3.c:149-180`).
5. Each unit gates against the reference at the same seed and params (hash on
   the final latents and frames), then gets an operator quality review.
6. The operator's pilot input is `tests/fixtures/h3/costello-crocodile-2panel.png`
   (432x171, sha256 `f364705a...`): a two-panel cartoon to animate as the
   first-frame test, at a legal canvas near its 2.53 aspect (576x224 or
   512x192). It is the same file the OpenAI `input_reference` case uses once
   the field is served.

## P12.5 — Port the reference test suite

`/data/h3.c/tests` holds 33 files and 9,177 lines. They are the reference's own
evidence, and the port currently reproduces only the text and DiT gates plus
whatever `cargo test` carries. This phase ports all of them to Rust, one test
module per reference file, with the same assertions and pinned fixtures.

Rules for the port:

1. One Rust test per reference file, named after it
   (`tests/h3/test_real_dit_block.rs`), and a header comment naming the
   reference path and the assertions it reproduces.
2. These are on-demand checks, not an automatic suite (operator, 2026-10-24):
   every one is a `make tests/h3-*` target the operator runs when a unit needs
   its evidence. Nothing in CI or in a hook runs them, and no push depends on
   them. They may be slow, artifact-backed and GPU-bound because of it.
3. Model-free tests need no artifact: tokenizer vectors, host arithmetic,
   safetensors schema, BF16 arithmetic, mux.
4. Artifact tests take the model root, and their golden tensors are
   regenerated from the reference build and pinned by sha256. A golden that is
   not pinned is not evidence.
5. A test that cannot be ported names why. Two do: `test_metal.c` and
   `test_text_metal.c` exercise the Metal backend, which the h3 port does not
   have (the Spark reference is CUDA). They are recorded here, not deleted
   quietly.
6. Benchmarks (`bench_dit.c`, `bench_video_vae.c`) become `ds4-perf`
   operations, not tests: they measure, they do not assert.

| Reference file | Lines | Pins | Port |
|---|---:|---|---|
| `test_device.c` | 23 | device probe | `ds4-sys` probe test |
| `test_host_portable.c` | 24 | RGB resize | `render` unit tests |
| `test_tokenizer_portable.c` + fixture | 60 | tokenizer fixture | `tokenizer` unit tests |
| `test_tokenizer.c` | 73 | model tokenizer.json vectors | artifact gate |
| `test_checkpoint_schema.c` | 102 | four-component safetensors schema | `layout`/`weights` gate |
| `test_av_mux.c` | 112 | container geometry and streams | model-free mux test |
| `test_h3.c` | 414 | host parsing, safetensors, terminal | split across host/terminal units |
| `test_bf16.c` | 1,714 | BF16 arithmetic breadth | model-free GPU-vs-reference test |
| `test_semantic_dit.c` | 120 | 256x256x22 latent, rel-max/abs/rel-L2 | artifact gate |
| `test_semantic_vae.c` | 264 | VAE block semantics | artifact gate |
| `test_cuda_runtime.c` | 107 | allocation, conversion, copy, file I/O | native gate |
| `test_cuda_primitives.c` | 201 | elementwise and normalization | native gate |
| `test_cuda_ops.c` | 229 | convolution, audio, VAE operators | native gate |
| `test_cuda_linear.c` | 310 | cuBLASLt F32/BF16 linear | native gate |
| `test_cuda_attention.c` | 352 | SDPA causal, GQA, head-major | native gate |
| `test_cuda_rope_tokens.c` | 234 | RoPE and token transforms | native gate |
| `test_audio_gpu.c` | 246 | audio primitives vs CPU | native gate |
| `test_real_prompt.c` | 171 | Qwen encoding, layer by layer | artifact gate |
| `test_real_dit_schedule.c` | 130 | AdaLN precompute, 50 blocks | artifact gate |
| `test_real_dit_block.c` | 677 | one real block against a golden | artifact gate |
| `test_real_dit.c` | 138 | full production DiT | artifact gate |
| `test_real_video_vae.c` | 104 | video VAE load, all blocks | artifact gate |
| `test_real_audio_vae.c` | 88 | AudioVAE decode stages | artifact gate |
| `test_real_audio_encoder.c` | 92 | AudioVAE encode stages | artifact gate (P12) |
| `test_real_video_encoder.c` | 104 | VAE encoder tile | artifact gate (P12) |
| `test_real_qwen_vision.c` | 132 | vision tower, all layers | artifact gate (P12) |
| `test_real_multimodal_text.c` | 188 | multimodal presentation | artifact gate (P12) |
| `test_real_ref_video_text.c` | 138 | reference video presentation | artifact gate (P12) |
| `test_metal.c` | 320 | Metal backend | not ported: no h3 Metal path |
| `test_text_metal.c` | 291 | Metal text | not ported: same reason |
| `bench_dit.c` | 1,924 | block benchmark with AB outputs | `ds4-perf` operation |
| `bench_video_vae.c` | 95 | VAE tile benchmark | `ds4-perf` operation |

Order: model-free first (they run everywhere and catch arithmetic drift), then
the artifact gates that already have code (prompt, schedule, block, DiT, VAEs),
then the P12-dependent ones (vision, multimodal, encoders) as their units
land. `make tests/h3-suite` runs every one by hand and prints, per artifact
test, either its result or the missing artifact; it is the operator's single
command, never an automatic step.

## P13 — Close

The rule 19 QA gate once, the evidence file, docs, memory, handoff, push.

## Unported reference surface and its disposition

| Surface | Reference | Disposition |
|---|---|---|
| SSD weight streaming | `h3.h:98`, `h3.c` streaming tier | Refused: a residency policy, same bytes; measured slower for this artifact. The port's `--ssd-streaming` exists and is bit-identical |
| `--layers 45/40` | `h3.h:82` | Not ported; `h3-rs` refuses anything but 50. Port only with the reference's own quality caveat and a gate |
| `--core-reuse N` | `h3.h:85` | Not ported. Speed knob that changes bytes; park behind a gate |
| `--token-reduction` | `h3.h:89` | Not ported. Changes composition; park behind a gate |
| `--reuse N` | `h3.h:79` | Ported and gated (reuse 1 and 3) |
| `--use-int8-row-fc2` | Metal/M5 only, CUDA no-op (`README.md:119-121`) | Refused |
| `--show`, `--zoom`, preview frames | `h3_terminal.c`, `h3.h:125-127` | Refused for the server; optional for the CLI only if the preview decoder is ported |
| Resident preview decoder | `h3.c` preview path | Not ported; only the preview needs it |
| CPU decode | `h3.c` CPU path | Refused; Linux CUDA is the reference |
| Metal backend | `h3_metal.m`, `metal/*` | Out of scope; CUDA is the release reference |
| h3c studio (FastAPI + React) | `webui/` | Refused: no Python in the deliverable. The OpenAI Videos API is the replacement |
| Postprocessing hook | `docs/POSTPROCESSING.md` | Out of scope |
| Device probe | `h3_device_cuda.cu` | Cheap; port in P13 so `h3-rs --info` prints the device |
| Tiled-VAE quality bench | `h3_vae_bench_quality` | P0 remainder; run with P9.5/P11 |
| Image/video/audio encoders | `h3_vision_encoder.c`, `h3_video_encoder.c`, `h3_audio_vae.c` encode half | P12 |

## What has been measured so far, and what has not

Measured: the E9 output stage and the 20-step parity (identical hashes at
256x256 and 512x512); the 256x256/22-frame/2-step decode comparison above;
identical audio PCM at that workload.

Not measured yet: the `mmap`-vs-`pread` copy hypothesis; the reference's
864x480 denoise; the 39/56/107-frame VAE path; any preset; any server
request through the in-process engine; the highest canvas that fits.

Nothing in this page has been adopted on the strength of a hypothesis. P9.1
is a hypothesis until `p9_copy_bench` and an A/B say otherwise.
