# MiniMax-H3 after the port: speed, serving, delivery, conditioning

The default text-to-video path is complete and bit-identical to the reference
end to end (`port-plan.md`, E1-E9). This page is the work list from there:
P8.5 the upstream branch survey, P9 speed, P10 serving with a resident engine,
P11 delivery formats (YouTube, Instagram, the highest resolution the model
has), P12 the conditioning and editing surface, P12.6 the remaining reference
knobs, P13 close, and P12.5 last: the reference test suite ported to Rust
(operator, 2026-10-24 — it is the evidence pass over a finished surface, not a
gate in front of it).

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
produced, and an anamorphic delivery changes the frame's aspect, not its
information.

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

1. **Media read.** `h3_ffmpeg.h` is the whole interface: `h3_ffprobe_visual_size`
   for the stream geometry, `h3_ffmpeg_read_image_f32(path, w, h, fit, &pixels)`
   for one F32 `[3,H,W]` image with `STRETCH` or `COVER` fitting,
   `h3_ffmpeg_read_video_f32(path, w, h, max_frames, &pixels, &frames)` for a
   `[3,T,H,W]` 24 fps clip trimmed to the released `5+17k` cadence, and
   `h3_ffmpeg_read_audio_f32(path, max_samples, truncate_at_limit)` for stereo
   32 kHz F32. The port has only the writer (`render.rs`), so this is a new
   host module around the ffmpeg CLI, not a device path.
2. **The vision tower.** `h3_vision_encoder.h`: 27 blocks, hidden 1152,
   FFN 4304, 16 heads of 72, patch 16, temporal patch 2, merge 2, a 48x48
   position grid, 5120-wide output and three deepstack additions. Input is F32
   `[T,3,H,W]` with `T` in `{1,2}` and both spatial edges multiples of 32.
   Weights are `model.visual.*` inside the **text** partition (the reference
   passes `text_path`, `h3.c:1347`), so the artifact already on disk covers
   it. Gate: `tests/test_real_qwen_vision.c` reproduced as a Rust test.
3. **The multimodal presentation.** `h3_multimodal.h`: FL2VA wraps one vision
   output per image as `<Picture n>: ` and then runs the decoder
   (`h3_multimodal.c:221`); Ref2VA keeps request order, adds one timestamp per
   video block and pending audio labels (`:303`), and a video block is a
   two-frame Qwen input sampled every 12 frames (`h3.c:1360-1379`,
   `h3_extract_vision_pair`). Gates: `test_real_multimodal_text.c`,
   `test_real_ref_video_text.c`.
4. **The video VAE encoder.** `h3_video_encoder.h`: `[3,T,H,W]` to a
   channel-major `[24,t,h,w]` latent, spatial edges multiples of 16, the
   released 256 px tiles with 64 px overlap. Anchors use `t = 1`, video
   references the whole clip. Gate: `test_real_video_encoder.c`.
5. **The conditioned DiT.** `h3_dit.h:50` `h3_dit_load_conditioned` and `:82`
   `h3_dit_reset_run`. Anchors occupy the leading spans of the video latent,
   one span each, seeded by `h3_augment_span(video, span, params->seed)`
   (`h3.c:333-346`); references occupy the layout's reference rows
   (`h3.c:344-370`). Gate: `test_real_dit.c`, then a final-latent and frame
   hash against the reference at the same seed.
6. **The wire and the session.** CLI spellings from `main.c:47-56` and the
   server's base64 `input_reference` (refused by name today,
   `video_api.rs:386-405`). The memo keys on media pixels as the reference's
   conditioning key does (`h3.c:149-180`), not on the prompt alone.
7. **Quality.** Every unit ends with a 20-evaluation operator render at a
   legal canvas near the pilot's 2.53 aspect (576x224 or 512x192 from
   `tests/fixtures/h3/costello-crocodile-2panel.png`), never a short-circuit.

Dependency order is 1, 2, 3, 4, 5, then 6; unit 1 and unit 2 are independent
and unit 2 can run in parallel with unit 1 only if two sessions exist.

**Unit 1 landed** (`minimax_h3::media`): probe, image read with STRETCH/COVER
Lanczos, 24 fps clip trimmed to `5+17k`, stereo 32 kHz audio with the truncate
rule, eight tests on generated media plus the pilot PNG.

**Unit 2 landed** (`vision.rs`, `vision_weights.rs`, `vision_forward.rs`,
`native/h3/h3_vision_gpu.{h,cu}`, `h3_vision_gate`): the whole tower, from the
patch rows to the final merger and the three deepstacks. Two corrections to
the survey above, both read from the reference this unit:

- The path is **351** tensors, not 350 as an earlier draft said: twelve per
  block (`h3_vision_encoder.c:250-287`, not eleven), plus the position table,
  the patch weight and bias, and four mergers of six.
- A deepstack merger normalizes `[merged_rows, 4608]`; the final merger
  normalizes `[patch_rows, 1152]`. Both then run the same 4608-wide projection
  pair (`:353-368`).

The unit's own test is the layout walk plus the host preparations. The device
side was gated on the Spark on 2026-10-24 against `vision_probe.c`: every
output (the final merger and all three deepstacks) hashed equal at 512x192 and
512x512, the merged hash also at 512x224/256/320/384/448, and 20 repeats at
two shapes were stable. The goldens are in `h3_vision_gate.rs`.

**Unit 3 landed** (`multimodal.rs`, `encode_multimodal` in `text_forward.rs`,
`Rope::build_multimodal` in `text_rope.rs`, `h3t_write` in the text-object ABI,
`h3_multimodal_gate`): the `<Picture n>: ` presentation, the axis-major
`[3, tokens]` mRoPE block with both tables rounded to BF16, the splice of the
tower's rows over the embedding lookup, and the deepstack add after layers 0,
1 and 2. Two facts read from the reference rather than assumed:

- The mRoPE axis cycles `0, 1, 2` only for half-indices below **60**; the last
  four keep axis 0 (`h3_text_encoder.c:551-553`).
- The tag mask covers the boundary ids too: tags are zeroed over
  `[start - 1, start + tokens + 1)` (`h3_multimodal.c:236-252`), which the
  gate's `zeros` field confirms (96 merged rows give 98 zeros).

Gated on the Spark on 2026-10-24 against `multimodal_probe.c` with the pilot
image at 512x192 and 512x512: the BF16 row hash, the eight-word edges and the
tag hash are equal at both shapes (PARITY `6ced02fc6feb0fe6` / `9f8deeb453edfecc`,
TAGS `7d8b4561c21600fb` / `0f10ec04569ce77b`). The `<Picture 1>: ` prefix is
six released BPE ids, not a hardcoded constant. Wall time with a warm page
cache: 5.245 s against 19.756 s at 512x192 (the reference read the shard cold)
and 7.028 s against 8.570 s at 512x512. The goldens are in
`h3_multimodal_gate.rs`.

Ref2VA (`h3_multimodal.c:243-360`) remains unported: it needs the ordered
audio/video reference surface from units 4-5, not just the presentation.

**Unit 4 landed** (`vae_encoder.rs`, `h3d_vae_encoder_pad` / `h3d_conv3d` /
`h3d_group_norm_silu` in `native/h3/h3_dit_gpu.cu`, `h3_video_encoder_gate`,
`video_encoder_probe.c`): the encoder graph, the tile plan, the stitched
border and the released normalization. Three facts read from the reference
rather than assumed:

- The encoder's tile axis has no size search, unlike the decoder's: the count
  is `ceil(extent/256)`, raised while `256*count - 64*(count-1) < extent`, and
  the leftover is handed out 16 px at a time, first gap first
  (`h3_video_encoder.c:611-651`). 512 px is three tiles with both gaps at 128.
- The blend is positional, not a fixed ramp: an incoming tile's first rows are
  mixed with the previous tile's tail at weight `position/overlap`, written at
  destination `start/16` (`:676-743`).
- The latent time is `div_ceil(frames, 4)` (`h3_host.c:30-32`), so five frames
  give `t = 2` and one frame `t = 1`, which is what the anchor path uses.

Gated on the Spark on 2026-10-24 against `video_encoder_probe.c` with the
pilot image at 512x192, the fox clip's five frames at 512x192 and the pilot
image at 512x320 (two tile rows, so a vertical 64 px blend): the pixel hash and
the latent hash are equal at all three (LATENT `ca995ec30255b666` /
`d039bbcf2a108277` / `e67374beb0e376bc`, 3, 3 and 6 tiles). Encoder wall time,
port against reference: 13.460/12.824 s, 45.258/45.195 s, 34.540/34.240 s. The
goldens are in `h3_video_encoder_gate.rs`; seven device-free tests cover the
tile plan, the stitch and the temporal rule.

**Unit 5 landed** (`condition.rs`, `Dit::set_conditions`,
`Engine::anchor_rows`, `--first-frame` / `--last-frame` / `--video-vae` in
`h3_dit_gate` and `dit_probe.c`): the anchor spans that lead the video latent,
from the image read to the conditioned run. Three facts read from the
reference rather than assumed:

- The order is encode, patchify, jitter: the anchor latent is patchified
  straight into its span (`h3.c:1300-1331`) and `h3_augment_conditions`
  (`h3.c:326-372`) rewrites the assembled buffer afterwards as
  `0.999*v + 0.001*normal(seed)`. Each span seeds a fresh PCG32 from
  `params->seed`, so two anchors of one request replay the same jitter
  sequence instead of continuing it; reference audio spans use `seed + 1`
  (`h3.c:358`), which FL2VA never reaches.
- A span is `latent_t(1) * latent_h * latent_w / 4` rows of 96, and the
  anchor count must cover the condition buffer exactly (`h3.c:333-343`).
- The rows live at offset 0 of `video_input` and are rewritten on every run:
  once at load (`h3_dit.c:1615-1624`, `:1685-1690`) and again by
  `h3_dit_reset_run` (`:2419-2437`), which also clears the core-reuse
  counters. `set_conditions` enforces the same element counts.

Gated on the Spark on 2026-10-24 against `dit_probe.c` with the pilot PNG as
both anchors at 512x192, five frames, four evaluations, seed 1: `ANCHORS
keyframes=2 elements=18432 fnv=fe11c049cced3a13`, `LAYOUT ... cond=192+0
target=192+16 latent=2x12x32`, `NOISE video=993dc9b32198325f
audio=094d71985b0c77d7` and `DENOISE video=dd0c9126e2bcbe37
audio=d27bae4151be2e5b` all equal, with the same edges. The decode leg then
stops at the reference's own two-token rule (`h3_video_vae.c:1222`): `t = 2`
supports one spatial tile and this canvas is two. The goldens are in
`h3_dit_gate.rs`; four device-free tests cover the jitter seed, the fresh
per-span generator, the patchified rows and the geometry refusal.

## P12.6 — The remaining reference surface

The surfaces that are neither conditioning nor the test suite, each with its
verdict so nothing stays "missing" without a reason:

1. **Device probe.** `h3_device_cuda.cu` / `h3_device_probe` fills
   `h3_device_info`; the port's `--info` prints components and geometry but no
   device line. Small CUDA call set, and `tests/test_device.c` is the gate.
2. **`--token-reduction`.** Real DiT behavior: `configure_token_reduction`
   (`h3_dit.c:378-470`) pairs video tokens in a middle-block window with an
   early-step window and a scale, controlled by `token_reduction_begin`,
   `_end`, `_early_steps`, `_early_end`, `_scale`. It changes compute and
   numerics, so the port needs the hash gate at 2 evaluations plus a measured
   end-to-end gain, and a kill switch.
3. **`--use-int8-row-fc2`.** Accepted by the reference and inert on CUDA:
   `dit->int8_mlp` requires `h3_gpu_has_int8_mlp(gpu)` (`h3_dit.c:1633`), and
   the CUDA backend returns 0 from it (`h3_gpu_cuda.cu:334`). The port parses
   the flag and records the zero, exactly as the reference does.
4. **`--show` / `--zoom` and the resident preview.** A terminal preview built
   from the denoise callback (`h3_dit_denoise_euler_preview`, `h3_dit.c:2861`)
   on the continue-event chain (`h3_gpu_continue`). It must never change the
   rendered file: the gate is a hash identical with the preview on and off.
5. **The tiled-VAE quality bench** (`h3_vae_bench_quality`): tile seams and
   temporal blend against the untiled decode, still unbuilt (P0 remainder).
6. **The reference tests** are P12.5 and cover units 1-5 above file by file.

## P13 — Close

The rule 19 QA gate once, the evidence file, docs, memory, handoff, push.

This is the point where the port is complete and runnable. The reference test
suite (P12.5) does not gate the close: it runs afterwards, as the evidence pass
over the finished surface (operator, 2026-10-24).

### P13.1 — The CI must pass on the branch

Recorded 2026-10-24 from the live check, not from a guess: `Host parity` is red
on PR 111 (Baekpica/ds4-dfm-rs, run 40127087166, job 120970719438). Format and
Clippy pass; the failure is in the step `C and Rust host parity`, inside
`make test-catalog-parity`, whose last command is `cargo test -p ds4-core`
(Makefile:731). That command builds every example target, and the four h3 gate
examples call `h3t_*`/`h3d_*` entry points that exist only in
`native/h3/*.o`, which a plain `cargo test` does not link. The log names
`h3t_read`, `h3t_embed`, `h3t_write`, `h3t_linear`, `h3t_upload`,
`h3t_rms_norm`, `h3t_silu_mul` and `h3t_rope_text` as undefined, then
`collect2: error: ld returned 1 exit status`.

The remaining steps have never run on this branch, because the job stops at
the first failure. Treat the CI as unknown past that point until the fix lands.

The fix, landed 2026-10-24:

1. `crates/ds4-core/Cargo.toml` declares the additive, empty `h3-gates`
   feature and one `[[example]]` entry per h3 gate with `required-features =
   ["h3-gates"]`, so `cargo test`, `cargo check --all-targets` and `cargo
   clippy --all-targets` skip them instead of linking them.
2. The `tests/h3_*_gate` make targets pass `--features h3-gates`. They already
   name the native objects and their fingerprints on the link line; only the
   feature was missing.

Verification record, replayed step by step on this host with
`/tmp/ci-host-parity.sh` (one PASS/FAIL per workflow step, log at
`/tmp/ci-host-parity.log`): 16 of 19 steps pass, including `make
test-catalog-parity`, the step that failed on PR 111. Three fail on code this
branch does not touch (`git diff origin/main...HEAD` is empty for every file
involved):

- `server-parity` and `workspace-tests`: `cache_identity::tests::
  bounded_sidecar_and_ple` (crates/ds4-server/src/cache_identity.rs:501) and
  two `ds4-perf` artifact tests (crates/ds4-perf/src/artifact.rs:449 and
  :489) rewrite a file to the same byte length and expect the stat-based
  identity to change. This host stamps consecutive writes with the same
  mtime/ctime — measured: five back-to-back `echo > f` on /data (xfs) and on
  /tmp (tmpfs) produced identical `%y %z` values for writes inside one ~4 ms
  tick — so the rewrite is invisible on this kernel and the assertion fails
  (4 of 5 runs). The same steps are green on main's Host parity runs
  (397139481086, 395545266396, 395387760326), whose runner has fine-grained
  timestamps.
- `v41-lifecycle`: `make test-vq-source-release` fails `resident(primary) ==
  0` (tests/test_vq_source_release.cpp:154): this host's `mincore` still
  reports the mapped source pages resident after the release advice, where
  the CI runner reports zero. The file and the gate are unchanged on the
  branch, and the step is green on the three main runs named above.

Still open after the replay: the PR check itself. The replay proves the step
that failed now passes; only the runner can prove the whole job. Confirm green
at the branch tip, not only locally, before the PR leaves draft.

Do not make the workflow pass by deleting steps or by excluding `ds4-core`:
the point of the job is that the branch's host code builds on a model-free
runner. The h3 device gates stay on-demand make targets (P12.5, rule 2) and
are not CI's job.

## P12.5 — Port the reference test suite (runs last, after P13)

Ordering decision (operator, 2026-10-24): this phase is moved to the end of the
work list. The port becomes complete and runnable at P13; the reference suite
is the evidence pass over a finished surface, not a gate in front of it, so it
runs after the close. The identifier P12.5 is kept because the handoff, the PR
body and the disposition table already name it that way; the position in this
file is the authority on when it runs.

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
