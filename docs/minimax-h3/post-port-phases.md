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

Surface superseded 2026-10-25: the routes are the official *local* `/v1` ones
with the SGLang body, not the OpenAI Videos body (`local-v1-contract.md`, P15).
The residency, admission and memo work recorded below stands unchanged.

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
   server's base64 `input_reference`, which is the clip's first frame
   (`video_api.rs:526-592`). The memo keys on media pixels as the reference's
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

**Unit 6 landed** (`video_api.rs`, `h3_video.rs`, `h3_engine.rs`,
`h3-rs --first-frame/--last-frame`, `docs/minimax-h3/api-usage.md`): anchors
reach the renderer from the wire and the CLI. Three facts read from the
reference rather than assumed:

- Anchors are paths (`h3.h:71-77`), and the CLI spellings are
  `--first-frame PATH` / `--last-frame PATH` (`main.c:47-49`). Neither side
  fetches a URL, so the served form accepts one base64 `data:` URI and refuses
  every other scheme by name.
- The image reader is told the format by the file name, not by sniffing
  (`h3_ffmpeg.c:143-160`), so a multipart part's extension picks the media
  type and an extension outside PNG/JPEG/WebP is refused instead of guessed.
- The OpenAI Videos `input_reference` is the clip's first frame; last-frame
  anchoring stays CLI-only because the surface carries one reference.

The decoded bytes become a request-scoped `AnchorFile` beside the output, the
server's only writable state, and are removed when the render returns.
`file_id` is 400 `unsupported_parameter`: this host has no uploads store.

Gated by 18 `video_api` tests and the 24 `h3_video` tests (data URI, multipart
part, `file_id`, non-data URL, unknown media type, anchor cleanup). The Spark
render of unit 6 — both anchors on the pilot PNG, 512x192, 22 frames, 20
steps, seed 1, "A red fox walking through snow" — is what opened unit 7: the
port's frame 0 was the prompt's fox while the reference's was the anchored
comic.

**Unit 7 landed** (`condition::anchor_embedding`, `AnchorPixels`, the same two
hosts, `h3_vision_gpu.o` on the `h3-rs` / `h3-server-rs` link line): an anchor
is a vision span in the prompt, not only a row span in the latent. Two facts
read from the reference rather than assumed:

- `h3.c:1345-1414` runs `h3_vision_encode_bf16` over every visual condition at
  the render size and passes the outputs to
  `h3_multimodal_encode_fl2va_bf16`; with no visual condition the run takes
  the plain `h3_text_encode_bf16` path (`h3.c:1424-1436`). The port therefore
  uses `encode_multimodal` only when an anchor exists.
- `h3_conditioning_key` folds the media into the key (`h3.c:143-166`), so the
  engine's text memo keys on the anchor rows as well as the prompt.

Measured on the Spark, 512x192, 22 frames, 20 steps, seed 1:
- One first-frame anchor with a prompt that animates the comic: the port and
  the reference both open on the anchored panels and animate (`~/unit7`).
- One anchor, then two anchors, both on the comic prompt: frame 0 is the comic
  in either layout (`~/diag-anchors`), so the earlier fox frame was never a
  placement bug.
- The unit-6 arguments after the splice open on the anchored comic, as the
  reference does, with `TEXT` growing from 6 to 214 tokens (`~/unit8`).

**Unit 8 landed** (`multimodal_probe.c` / `h3_multimodal_gate` taking one
`--image` per span, served forms exercised live): the anchor path is now
hash-gated, not only frame-gated. Measured on the Spark on 2026-10-25:

- Two images (the pilot PNG on both spans), 512x192, prompt `A red fox walking
  through snow`: `TOKENS 214 spans=2 zeros=196`, PARITY
  `14553bf8e35815b6`, TAGS `8db87703b7392495` — equal to the reference probe,
  edges included, 5.938 s against 17.492 s.
- Served forms, one live run each: a JSON `data:image/png;base64,` anchor and
  a multipart file part both render 512x192x22 with byte-identical MP4s
  (`cde156d793a2a9032143533b67520c5f`) whose frame 0 is the anchored comic;
  `file_id` answers 400 `unsupported_parameter`, an `https://` URL 400
  `invalid_value`, and no `*.anchor.*` file survives the render.
- The anchor-less path is unchanged: `h3-rs` at the install gate's arguments
  prints `TEXT tokens=6` and `HASH video=70973d0a2c32e378
  audio=f776839e1170733b frames=541455fe622e267f rgb8=62b36816b6e1248e`, the
  values recorded before the splice.
- The unit-5 conditioned `h3_dit_gate` run, repeated on the merged tree, still
  prints `ANCHORS fnv=fe11c049cced3a13`, `LAYOUT
  positions=a47f8985c9beac72`, `NOISE video=993dc9b32198325f
  audio=094d71985b0c77d7` and `DENOISE video=dd0c9126e2bcbe37
  audio=d27bae4151be2e5b`.

Open: the live two-request memo pair (same anchor hits, a different anchor
prefill-loads) is still unit 3's open item; the key itself is covered by
device-free tests.

## P12.6 — The remaining reference surface

The surfaces that are neither conditioning nor the test suite, each with its
verdict so nothing stays "missing" without a reason:

1. **Device probe.** `h3_device_cuda.cu` / `h3_device_probe` fills
   `h3_device_info`; the port's `--info` prints components and geometry but no
   device line. Small CUDA call set, and `tests/test_device.c` is the gate.
   **Landed 2026-10-25** (`native/h3/h3_device_gpu.{h,cu}`,
   `ds4_sys::h3_device`, `minimax_h3::device`, `h3_device_gate`,
   `h3_device_gpu.o` on the `h3-rs` link line). The probe is the reference's
   call set line for line (`h3_device_cuda.cu:17-50`): device 0, its
   properties, `snprintf("%.127s")` of the name, "CUDA sm_%d%d"
   (`:38-40`), `sysinfo(2)` totalram × mem_unit (`:41-45`), `totalGlobalMem`
   twice (`:46-47`), `unifiedAddressing` (`:48`). `apple_gpu_family` and
   `metal4` are the shared struct's Metal fields and stay zero
   (`h3.h:135-136`, `h3_device_cuda.cu:23`). Measured on the Spark: the
   reference's `h3_device_test` prints `ok: NVIDIA GB10 (CUDA sm_121), 121.6
   GiB GPU memory, unified=yes`; the port's gate prints that same line and
   `h3-rs --info` the same record (`physical_gib=121.6 working_set_gib=121.6
   max_buffer_gib=121.6`), and the reference's own `h3 --info` prints
   `physical memory / recommended GPU set / max GPU buffer` at 121.6 GiB each
   and `unified memory yes` (`~/unit9-device-port.log`, `~/unit9-ref-info.log`
   on the Spark).
2. **`--token-reduction`.** Real DiT behavior: `configure_token_reduction`
   (`h3_dit.c:378-470`) pairs video tokens in a middle-block window with an
   early-step window and a scale, controlled by `token_reduction_begin`,
   `_end`, `_early_steps`, `_early_end`, `_scale`. It changes compute and
   numerics, so the port needs the hash gate at 2 evaluations plus a measured
   end-to-end gain, and a kill switch.
   **Landed 2026-10-25** (`Reduction`/`ReductionWindow`, the pairing and
   parent maps, `reduced_rope_tables`, `h3d_token_pool`/`h3d_token_expand`,
   `--token-reduction` in `h3-rs` and `h3_dit_gate`). Read from the reference,
   not inferred: pooling halves one spatial axis of the video target
   (`reduced_width = (spatial_width + 1) / 2`, `:440-447`); a reduced row's
   RoPE position is the mean of its pair, the spatial coordinate scaled in F32
   (`:927-948`); the reduced AdaLN map takes each pair's first row
   (`:1034-1041`); the restore is `original + scale * (reduced - baseline)`
   with the prefix rows copied through (`h3_gpu_cuda.cu:3328-3351`); the window
   defaults to blocks 4..30 with 10 early steps ending at 40 below a 40-block
   end, and every bound is validated (`:383-439`). Measured on the Spark,
   512x192, five frames, both anchors, seed 1:
   - 20 evaluations: reference `6720bb389eeaef0b`/`3a88f90bede2da9b` off and
     `35fc6b4a61d34881`/`52dd7ce1c6106c42` on; the port prints those same four
     hashes. Denoise 9.637 -> 8.425 s in the reference, 9.539/9.615 -> 8.348 s
     in the port (12.5%, two off runs and two on runs).
   - Two evaluations: `7606d402d9cac721`/`6a8e8b2323eaea5a` off and
     `4dc60b25c9b1a6f2`/`8bbab603d0908469` on, equal on both sides; 0.991 ->
     0.821 s and 0.989 -> 0.812 s. That pair is confounded by weight
     page-cache warmth (off ran first after the load); the 20-step repeat is
     the measurement.
   - The off path is unchanged: the unit-5 four-step conditioned gate still
     prints `dd0c9126e2bcbe37`/`d27bae4151be2e5b`.
   - The kill switch works: `H3_DISABLE_TOKEN_REDUCTION=1` beside
     `--token-reduction` prints the off hash at two evaluations.
   - End to end at 22 frames and 20 steps with both anchors: both paths write
     an MP4 (`e9891e13929b2a9746908eeec31edb8b` off,
     `05f4ea3f99322a830d3993c99393430b` on), denoise 45.153 -> 32.733 s, and
     frame 0 of the reduced render is the anchored comic
     (`~/unit10-render-*.mp4`).
   The reference's fused pool/expand-with-AdaLN entries compute pool + adaln
   in one kernel; the port calls the two parts, which is the same arithmetic,
   so `H3_DISABLE_FUSED_TOKEN_POOL_ADALN` and `H3_DISABLE_FUSED_TOKEN_ADALN`
   are inert here and recorded as such; `--use-int8-row-fc2` is refused
   instead.
3. **`--use-int8-row-fc2`.** Accepted by the reference and inert on CUDA:
   `dit->int8_mlp` requires `h3_gpu_has_int8_mlp(gpu)` (`h3_dit.c:1633`), and
   the CUDA backend returns 0 from it (`h3_gpu_cuda.cu:334`). The port refuses
   the flag instead of accepting an inert switch (`h3-rs.rs:16-18`, port plan
   section 6.3); no output differs either way.
4. **`--show` / `--zoom` and the resident preview.** A terminal preview built
   from the denoise callback (`h3_dit_denoise_euler_preview`, `h3_dit.c:2861`)
   on the continue-event chain (`h3_gpu_continue`). It must never change the
   rendered file: the gate is a hash identical with the preview on and off.
   **Landed 2026-10-25** (`PreviewDecoder` in `vae_video`, `terminal.rs`,
   `Dit::denoise_with`, `--preview-dir`/`--show`/`--zoom` in `h3-rs`). Read
   from the reference: the preview asks the resident decoder for one frame
   from the middle chunk, its `FIRST_CHUNK_FRAMES / 2` frame
   (`h3_video_vae.c:953-975`), and only that frame is unpacked per tile
   (`:850`, `:874-877`); the frame is resized to the output canvas and handed
   to the CLI callback after every Euler step, evaluated or reused
   (`h3_dit.c:2969-2974`, `h3.c:755-830`); `--preview-dir` writes
   `step-%04d.ppm` through a scratch name (`main.c:196-213`); `--show`
   detects Kitty/Ghostty or the iTerm2 family and warns instead of failing on
   anything else (`h3_terminal.c:29-50`, `main.c:541-560`).
   Measured on the Spark, 512x192, 22 frames, 2 steps, both anchors, seed 1:
   the port's MP4 is byte-identical with the preview on and off
   (`3e16ffa2dd4392382f1593cfed8a4a31`) and both runs print `HASH
   video=701962e123127810 audio=fe06b9a6e575195f`; the preview wrote
   `step-0000.ppm` and `step-0001.ppm`, 294,927 bytes each; denoise 10.898 s
   with the preview against 4.538 s without it (two extra chunk decodes).
   OPEN: the reference's own CLI render for that command differs from the
   port's — its preview PPM against the port's is 47.2 dB PSNR and the MP4s
   differ — while the anchor-row DiT gate at the same shape matches bit for
   bit (`DENOISE video=2df00020bb691fb6 audio=4bd5b8fe0e7ec495`). The
   difference is therefore downstream of the latents: the CLI's spliced
   prompt or the multi-tile VAE decode. The `--decode` comparison was run
   without `--audio-vae`, so the reference probe stopped before `FRAMES`; the
   port printed `FRAMES frames=22 height=192 width=512
   fnv=e706c198483b3c0c`. Settle it with item 5 below (`~/unit12-*`).
   Correction, measured 2026-10-25: 47.2 dB was `step-0000` only, whose input
   is pure noise; `step-0001` is 24.75 dB, and the 22 delivered frames
   (`--frames-dir`, before the mux) differ at 23.3-27.3 dB. The decode itself
   is not the cause (it hashes identically, P12.7); the difference enters
   before it, in the CLI's conditioning assembly. Now P12.7.
5. **The tiled-VAE quality bench** (`h3_vae_bench_quality`): tile seams and
   temporal blend against the untiled decode, still unbuilt (P0 remainder).
6. **The reference tests** are P12.5 and cover units 1-5 above file by file.

**Core execution options landed 2026-10-25** (`LoadOptions`, `--layers`,
`--core-reuse`, `h3d_sub`):

- `--layers N` (25..=50) applies `configure_active_blocks`
  (`h3_dit.c:1181-1194`) and then the reference's default gate ranking
  (`:1196-1244`): the lowest mean |gate| over slots 2 and 5 is dropped unless
  `H3_DIT_LAYER_POLICY=uniform`. The port computes the same score from its own
  modulation tensors, over the same rows and slots. Deviation, named: every
  block is still loaded, where the reference skips inactive ones at load, so
  the port uses more memory for the same arithmetic.
- `--core-reuse N` (1..=6) implements `evaluate_core` (`h3_dit.c:2161-2165`):
  the first forward, every Nth one and the last step run the core; the others
  add the cached `hidden - core_input` delta to the freshly packed rows
  (`:2306-2312`). `set_conditions` resets the counter, as `h3_dit_reset_run`
  does (`:2437-2438`), and the token-reduction window only applies to steps
  that evaluate the core (`:2166-2167`).
- Measured on the Spark, 512x192, five frames, both anchors, seed 1, against
  `dit_probe`: layers 45 -> `f2874fa151da44d2`/`5d53523c468fa314`, layers 40 ->
  `a542928c46756e5a`/`0ca938b91455931a` at two evaluations; core-reuse 4 at 20
  evaluations -> `f7237de7a7c714c2`/`06d1170ae45c365f` (2.868 s against the
  reference's 2.939) and core-reuse 6 -> `aabb0a542ebe4980`/`a10445642d3fe201`
  (2.425 s against 2.437). Equal on both sides.

## P12.7 — Anchored lettering and the CLI conditioning split (opened 2026-10-25)

Operator report on `~/unit12-portprev/step-0001.ppm` (Spark): the two-panel
comic renders as intended but the balloon text cannot be read. Confirmed, and
two separate findings follow, one about the canvas and one about the port.

### The canvas, not the renderer

Measured 2026-10-25, 512x192, 22 frames, 2 steps, seed 1, both anchors:

- The reference is as soft as the port. Its own `step-0001.ppm`
  (`~/unit12-refprev`) and its own MP4 frames read identically; the balloon
  lettering is mush in both. So this is not a port defect and not a preview
  defect: the delivered frames (`~/u12-port-frames/frame-0021.ppm`, written by
  `--frames-dir`) are just as soft.
- The canvas is the caller's number (`h3.c:488-498`: multiples of 32, at least
  32, area at most `768*1344` = 1,032,192, `h3_host.h:8`). 512x192 is 98,304
  px, 9.5 % of the cap, and its VAE latent is 32x12: `h3_latent_canvas` is
  `w/16, h/16` (`h3_host.c:43-46`) with `H3_VAE_SPATIAL_RATIO` 16
  (`h3_host.h:11`).
- Both anchors pass through that canvas before the encode: the first frame is
  resampled with `H3_IMAGE_FIT_STRETCH`, the last with `H3_IMAGE_FIT_COVER`
  (`h3.c:1233-1247`), then encoded at the latent canvas above.
- The source is `tests/fixtures/h3/costello-crocodile-2panel.png`, 432x171,
  whose lettering is about 4 px tall (measured on a 6x nearest-neighbour crop;
  the strokes are one pixel). A glyph is therefore a quarter of a latent cell:
  the encoder has nowhere to put it.

The arithmetic bounds the improvement. One latent cell is 16 canvas pixels,
so a glyph needs about 32 canvas pixels to fill two cells. The largest
exact-aspect canvases under the cap are 1536x608 for the fixture's 48:19
(933,888 px) and 1536x576 for 8:3 (884,736 px); a panel is then 768 px wide
and a 20-character balloon line is about 12 px per glyph, under one cell.
Rendering larger helps definition and cannot make this page's lettering
readable: the fixture carries ~4 px glyphs, and upscaling to 1536 adds no
glyph detail. That bound is the source's, not the checkpoint's, and P12.9
measures the difference: the same canvas fed a page whose glyphs really are
18 px keeps its text.

Work, in order:

1. Render 1536x608 (48:19, no anchor stretch) and 1536x576 (8:3) at 2 steps,
   and read `step-0001` and final frame 21 against 512x192: does the lettering
   go from mush to shape, and what does it cost (denoise s, decode s, peak
   memory)? No port change is needed; `Canvas::new` enforces the reference's
   rules (`render.rs:127-140`, `h3.c:488-498`).
2. If a legible page is required, the composite is the honest route: run the
   motion at the largest canvas and keep the source's own balloon art in the
   delivered frames instead of asking the decoder to carry lettering it
   cannot.
3. A sharper source is its own question: the fixture is 432x171, so upscaling
   it into a larger canvas cannot add glyph detail.

**Landed 2026-10-25 — re-lettering with the workstation's image stack.** The
page can be made legible outside the checkpoint. `/data/imagegen` runs
stable-diffusion.cpp master-890 with Qwen-Image-2.1 Q6_K, the Qwen3-VL-8B
encoder and its VAE on this box's RTX 4070 SUPER; a single img2img pass from
the 432x171 source upscaled to 1536x608 (lanczos) restores the balloons:

    sd-cli --diffusion-model models/diffusion_models/qwen-image-2.1-Q6_K.gguf \
      --vae models/vae/qwen_image_2.1_vae_bf16.safetensors \
      --llm models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
      --backend te=cpu,diffusion=cuda0,vae=cpu \
      --params-backend te=cpu,diffusion=cuda0,vae=cpu -M img_gen \
      -i init-1536.png -W 1536 -H 608 --strength 0.5 --steps 28 \
      --cfg-scale 6.0 --sampling-method euler -s 7 -p "<two-panel prompt with
      both balloon texts quoted>" -n "blurry text, unreadable letters, ..." \
      -o out/h3-comic-reletter-1536-strength050.png

181 s, and both balloons read exactly, in the source's wording; the panel
art is redrawn but faithful in composition, characters and colours
(`/data/imagegen/out/h3-comic-reletter-1536-strength050.{png,log}`). So the
deliverable can carry readable lettering: deliver the re-lettered still, or
composite its balloon art into the H3 frames.

The price of a global pass, measured against the upscaled init: 16.95 dB PSNR
over the frame, 17.02 dB below y=220 and 19.89 dB in the margin strip, so
strength 0.5 redraws everywhere it is allowed to. The masked pass is therefore
the next step, and Gemini 4 Argon (asked through the agy session) gave the
numbers to use: inpaint the balloon regions at strength 0.65-0.70 (18-20 of
28 steps), dilate the mask 16 px past the ink rim, and composite in pixel
space with an 8-12 px Gaussian feather so the unmasked art is untouched; the
two-pass variant (0.35 global for line quality, then 0.70 on the balloons)
doubles the run to about 365 s. Its gates: art drift (unmasked region MAE at
most 12/255, PSNR at least 24 dB), invented words (OCR character error 0
against the transcript, Michelson contrast at least 0.80) and seams (gradient
ratio at most 1.25, ring MAE at most 6/255). `adetailer` needs a YOLOv8
detector that is not on disk and `upscale` is unconditioned ESRGAN, so
neither fits. Its sd-cli code offsets come from its own disassembly and are
unverified here; the drift figures above are ours.

### The decode agrees; the CLI path above it does not

Item 4's open difference (the reference's 22-frame CLI render against the
port's) is now split in two, and the decode is cleared. Same command, 217-token
prompt, 512x192, 22 frames, 2 steps, both VAEs:

- The DiT and the video VAE decode are bit-identical. The reference's own
  probe prints `FRAMES frames=22 height=192 width=512 fnv=6f49c158bf764365
  first=0.985459328 last=0.591346204` (`~/unit12-refdecode22.log`) and the
  port's gate prints the same four values (`~/unit12-long-port.log`), from
  `DENOISE video=0f3c1d473e28c1cc audio=218ed318614f2ba7` equal on both sides.
  The tiled spatial decode at this shape is therefore correct, and item 5's
  seams are a quality question, not a parity bug. Temporal chunk blending is
  not covered by this run: 22 frames are exactly one decoder chunk.
- The two CLIs still deliver different frames: `--frames-dir` at the same
  command writes 22 frames that differ at 23.3-27.3 dB PSNR (frames 0, 1, 11,
  21: 27.09, 23.27, 24.75, 25.53; `ffmpeg -lavfi psnr` on
  `~/u12-{ref,port}-frames`). The error is edge structure over the whole
  frame, not a tile seam (23.86/23.91/27.31 dB left/mid/right thirds) and not
  a phase offset (+1 px is worse: 20.48; +1 row: 19.49); a 0.5 sigma blur on
  the reference does not close it (25.19 dB).
- The untested stage is the CLI's own conditioning assembly. Both CLIs take
  the released multimodal presentation here (the reference: `h3.c:1407-1414`,
  `<Picture n>: ` plus the vision span per image,
  `h3_multimodal.c:200-232`; the port prints `TEXT tokens=217
  fnv=e596a5d1874ee131` for a 9-token prompt), while every gate that matches
  bit for bit supplies the anchors as VAE condition rows and a plain text
  prompt (`TEXT tokens=9`). The vision span, not the decoder, is where these
  two renders part.

Attribution, with `h3_ppm_diff` (`tools/minimax_h3/ppm_diff.c`, which reports
`PIXELS`/`PSNR`/`BBOX` and per-128-pixel `BAND` counts): 95,641 of 98,304
pixels differ, rms 14.75 of 255, and the x bands split evenly (24,395 /
22,874 / 23,923 / 24,449 of 24,576 each). Nearly every pixel moves, so this is
two different generations from different conditioning, not a filtered or
shifted image.

The 2x2 at 22 frames and 2 steps, both CLIs, with and without the anchors,
answers it (`~/u13-*`; 22 is the CLI's minimum, both refuse five with
`generation requires at least one trained 22-frame decoder chunk`):

- anchor-free: all 22 frames are byte-identical, 0 of 22 differing;
- anchored: frame 0 differs at 27.085 dB PSNR, rms 11.28, 95,331 of 98,304
  pixels, reproducing the unit-12 number exactly.

So the plain CLI path is correct and the anchored path is the defect. It is in
the DiT's row map, not in the splice and not in the decoder:

- the reference builds each row's AdaLN selector from the presentation's
  modality tags: `h3_dit_schedule_row_map` receives `text->tags`
  (`h3_dit.c:1028`) and uses `time_row * H3_DIT_MODALITIES + text_tags[index]`
  for text-segment rows (`h3_dit_schedule.c:449-456`);
- the presentation marks vision rows 0 and text rows 1 (`h3_multimodal.c:173`,
  `:182`), so this render's 192 vision rows select a different modulation slot
  than its nine prompt rows;
- the port hardcodes `time_row * MODALITIES + 1` for every text row
  (`dit.rs:573-584`) and never reads `Embedding::tags` (`text_forward.rs:74-79`).
  Plain text carries no tags, which is why every unanchored gate matched.

**Landed 2026-10-25.** `row_map` takes the tags and validates them
(`dit.rs:576-611`), `modulation_maps` and `reduced_maps` pass them, and
`load_with` reads them from the embedding, one line with the reference's
(`dit.rs:1571`). A tag set that does not cover the text rows, or a tag outside
the modality count, is refused the way `h3_dit_schedule.c:440` and `:453`
refuse it; two unit tests pin the untagged and the tagged map.

Measured on the Spark, 512x192, 22 frames, 2 steps, seed 1, both anchors:

- the anchored CLI's frames are now byte-identical to the reference's, 0 of
  22 differing (frame 0 md5 `9b48614dec433c052816063069fdbbb2` on both sides),
  against 27.085 dB before the change;
- the plain-text path did not move: the five-frame four-step gate reads
  `DENOISE video=12c4ef3f4203bf56 audio=c825b5090644a38c` on both the pre-fix
  and post-fix trees (`~/u13-ab-{head,fixed}.log`), and the 22-frame gate
  still reads `2df00020bb691fb6`/`4bd5b8fe0e7ec495`.

The tag fix is the port's own; the canvas finding above stands independently,
so anchored renders now match the reference exactly and still cannot carry
this page's lettering. The re-lettering pass is how the deliverable gets both.

A stale golden found on the way: the unit-5 four-step block in
`h3_dit_gate.rs`'s header (`LAYOUT rows=406 ... positions=a47f8985c9beac72`,
`DENOISE video=dd0c9126e2bcbe37`) does not reproduce on either side of this
change; the current tree prints `rows=409`,
`positions=e93dddf46639aa52` and `video=12c4ef3f4203bf56` for that command,
which is consistent with a different prompt in the original run (three more
tokens). Re-record it with the prompt spelled out.

## P12.8 — A prompt-only render at the 21:9 canvas (2026-10-25)

A fresh clip, not an anchored reproduction: the 432x171 page (the operator's
panel strip) supplied the prompt only, and no pixel of it entered the
conditioning. Canvas 1344x576, the 21:9 preset (`presets.rs:211-216`,
774,144 px against the 1,032,192 cap), 22 frames, 20 steps, seed 42.

Command (`~/ds4-h3-port`):

    ./h3-rs -d /mnt/h3-artifacts/MiniMax-H3 -p '<prompt>' \
      -o ~/u14-new.mp4 --width 1344 --height 576 --frames 22 --steps 20 \
      --seed 42 --hash --frames-dir ~/u14-frames

Prompt: an animated comic-book cartoon, flat ink outlines and watercolour
washes, a tropical swamp lagoon at midday, a lanky bather in red trunks
treading water in alarm beside a drifting log that opens a long green
alligator grin, a second bather sitting on the log and laughing, lily pads,
splashes, bold saturated colours, hand-drawn 24 fps look, no text and no
speech balloons.

Measured: TEXT 27.0 s, LOAD 106.3 s, DENOISE 804.5 s for 20 steps
(40.2 s per step), AUDIO 1.5 s, FRAMES 84.6 s, 1023.9 s total.
`frames=bc3d7ce34d585bd5 rgb8=6975023b964dc0c7`; the MP4 is 1344x576,
22 frames at 24 fps, 32000 Hz stereo, 817,823 bytes. Audio is not silent
(mean -27.5 dB, peak -11.8 dB).

A 2-step probe of the same shape measured 40.0 s per step against the
render's 40.2, so denoise cost here is linear in steps at this width.

The picture holds a readable comic frame: the wide canvas carries clean
inked linework, the log is the alligator's body and its jaws open across the
clip. Its defect is the open jaw, which late in the clip draws a second
mouth ring in the water (frames 7-21); frame 0 is clean. That is a model
artifact at this width, not a decode or conditioning fault, and it is the
thing a longer or deeper render would have to fix.

Unported surface this exposed: the reference's own reference conditioning,
`--ref-image` plus `--ref-image-size` (`main.c:440-449`), refused together
with frame anchors (`h3_cli.c:316-318`). The port carries `RefImage` in the
sequence and the DiT (`sequence.rs:32`, `dit.rs:610`) but no CLI or engine
path builds it, so an image-conditioned clip (Ref2VA) is not reachable yet.
A prompt is the only way to steer a picture into this engine today.

## P12.9 — A new lettered clip, and what the checkpoint really does with text

Operator, 2026-10-25: a new video, the page as reference only, and the text
read back by vision. Two results.

**The 16x bound was the source's, not the checkpoint's.** P12.7 reasoned that
no legal canvas carries this page's lettering. That holds for the 432x171
fixture, whose glyphs are ~4 px and gain nothing from upscaling. Feed the same
canvas a page that really has ~18 px glyphs and the line moves: the re-lettered
1536x608 art anchored at 1536x608 (1:1, no stretch, `H3_IMAGE_FIT_STRETCH`,
`h3.c:1233-1247`) round-trips at PSNR 28.90 dB (rms 9.15, max 131,
`ppm_diff` against the source page) and its balloons read in the decoded
frame: "IT ISN'T A LOG, COSTELLO! CAN YOU SWIM?" and "WHY, SURE! I'M THE
HUNDRED METER CHAMP! ER- MEET THE NEW CHAMP!". A glyph an 18 px tall spans a
latent cell; at 4 px it is a quarter of one, which is the whole difference.

Cost at that shape, 22 frames, 2 steps (`~/u15-probe`, 1536x608, one anchor,
912 anchor rows): TEXT 39.9 s, LOAD 103.3 s, DENOISE 182.3 s (91.1 s per
step), AUDIO 1.5 s, FRAMES 95.6 s. The anchor and the larger latent canvas
roughly double the per-step cost against 40.4 s at 1344x576 without one.

**The delivered clip is new, and its lettering is composited.** `u16` is a
fresh prompt-only render: 1344x576, 22 frames, 20 steps, seed 7, no anchor,
`DENOISE video=6e3b447a8f4bbcd7 audio=df6a25f98b33682a frames=737bdfe443bd30cc`
(`~/u16-new.mp4`, sha256 114537c8...). Nothing of the page's pixels enters it;
the prompt takes the scene and the gag. The alligator rises out of the water
between the two bathers by frame 21, which the page does not do.

The balloons are a static composite, authored here, not sampled: two
antialiased white ellipses with a 6 px black rim (`geq` on a hypot mask in
`ffmpeg`), DejaVu Sans Bold 26 and 23 px, overlaid at 12,14 and 560,8 so
neither covers a face. Delivered as `~/u16-lettered.mp4`, sha256
5c2b1fa2..., 599,209 bytes. The reason for the composite is the previous
paragraph: at 1344x576 the checkpoint would invent glyphs, and invented
glyphs are not the gag's dialogue.

Re-read from the encoded H.264, not from the pre-encode frames, both balloons
parse exactly: "IT ISN'T A LOG, COSTELLO! CAN YOU SWIM?" and "WHY, SURE! I'M
THE HUNDRED METER CHAMP! ER- MEET THE NEW CHAMP!". That is the operator's
vision check, and it is the only claim about readability here that was made
against the delivered file.

The composite is reproduced by `tools/minimax_h3/letterballoons.sh`; rerunning
it on `u16-new.mp4` wrote a byte-identical file (sha256 5c2b1fa2...), so the
delivered clip is a script result, not a hand edit.

**The longer cut, 39 frames.** The same prompt and seed at the same canvas,
with 39 frames instead of 22: TEXT 27.0 s, LOAD 106.1 s, DENOISE 2258.2 s for
20 steps, AUDIO 1.9 s, FRAMES 138.2 s, 2531 s total. The latent is 12x36x84,
1.71x the 22-frame token count, but the step costs 112.9 s against 40.4 s,
2.8x, so denoise at this width grows faster than tokens. Hashes
`video=2c3fd6319ed9db1b audio=1b8a70fe2911603c frames=014d65241aee7346`
(`~/u17-long.mp4`). Lettered by the same script to `~/u17-lettered.mp4`,
1,042,101 bytes, sha256 3809f8de..., 1.625 s at 24 fps with 52,000 samples of
audio (mean -36.5 dB, peak -15.3 dB). Both balloons re-read exactly from the
encoded file at frame 38; the crocodile surfaces between frames 0 and 38 and
its head sits under the second balloon without touching it.

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

### P13.2 — The close evidence (2026-10-25)

CI at the branch tip `7ed11690`: `Host parity` completed success, the only
check the repository reports. The P13.1 fix holds on the runner, not only in
the replay.

Rule 19 QA, run on `deepseek-v4.1-CC-flash` against base `origin/main` (the
default base is the pushed head and would pass trivially): 559 surfaces (186
files, 2 Makefile targets, 371 public items), every one present with a
disposition; 562 checks PASS, gate exit 0, report
`qa-evidence/qa-report.md` 919 lines ending `verdict: overall PASS`.

The first pass returned FAIL on two shipped documents that contradicted the
served wire: `docs/minimax-h3/openai-videos.md` and
`tests/fixtures/h3/README.md` still described `input_reference` as refused,
although `video_api.rs:526-599` and `h3_video.rs:794-820` accept a base64 data
URI or a multipart file part as the clip's first frame, with a live matrix
(`.png` 200, `.tiff` 400, `file_id` 400, `https` URL 400, data URI 200, and
the 16 MiB bound at both edges). Two more documents carried the same class of
error: `port-plan.md:1039` listed `--token-reduction`, `--show` and `--zoom`
as unported while all three are parsed and wired (`h3-rs.rs:686-690`), and
`post-port-phases.md` said the port parses `--use-int8-row-fc2` while it
refuses it (`h3-rs.rs:16-18`). All four were corrected against the code, the
unported-surface table gained the Ref2VA and interactive-session rows, and two
clippy warnings in new h3 code were removed (`render.rs:633`, a needless
`mut`; `vae_video.rs:362`, `is_multiple_of`). The re-verified pass is the one
recorded above.

Docs index: `docs/README.md` now indexes the post-port phase record and the
served Videos API beside the family index.

Open observations from the report, all judged unreachable or cosmetic and
recorded rather than fixed: the `tools/minimax_h3/README.md` index covers 5 of
13 tools; `ppm_diff.c` bounds a band to 8,192 px (no h3 canvas reaches it) and
prints `axis=y` bands its header does not name; `row_map` omits the
reference's two malformed-segment refusals, which `sequence::build` and
`Geometry::check` make unreachable.

## P14 — Ref2VA, and the code the port is missing

Opened 2026-10-25 on the operator's instruction ("lets start with that port we
were missing"). This is the one substantial surface the close left unported:
ordered image/video/audio references, plus the AudioVAE encode half it needs.

What the reference does, read from the source:

- A reference is `{kind, path, audio_path, include_embedded_audio}` with kind
  image, video, audio or video+audio (`h3.h:30-42`); image sizing is a policy,
  `match` (default) or `max` (`h3.h:44-47`).
- At most 12 (`h3.c:588`, `main.c:107-110`). Any reference requires the
  Ref2VA checkpoint component (`h3.c:872-875`), inventoried as
  `Ref2VA/transformer` (`h3.c:448-452`), and the prompt is encoded by
  `h3_multimodal_encode_ref2va_bf16` (`h3_multimodal.c:243`).
- References enter the conditioning key byte for byte, media and soundtrack
  included (`h3.c:155-161`), and are given their own noise spans per modality
  (`h3.c:323-360`).
- CLI: `--ref-image`, `--ref-image-size`, `--ref-video`, `--ref-silent-video`,
  `--ref-video-audio VIDEO AUDIO`, `--ref-audio` (`main.c:51-56`, `:333-335`).

What the port already has: the FFmpeg readers at the reference's shapes
(`media.rs`), the VideoVAE encoder (`vae_encoder.rs`, P12.4 gate), the vision
tower, `SegmentKind::RefImage` (`sequence.rs:32`) and the DiT's
COND/REF_IMAGE/REF_AUDIO row arms (`dit.rs:610`, `:1376`, `:2112`).

What it lacks, in the order it must be built:

1. **P14.1 — the AudioVAE encode half** (`h3_audio_vae.c:1275`, its stages
   `:746-1273`, 5 stages with strides `{2,4,4,5,5}` at `:714-720`). The port
   has the decode half only (`vae_audio.rs:386`). Gate:
   `tests/test_real_audio_encoder.c`'s assertions against a pinned golden.
2. **P14.2 — the Ref2VA component** beside the FL2VA transformer
   (`h3.c:448-452`): loader, tensor table and a gate that hashes it.
3. **P14.3 — reference media prep**: decode at the reference's sizes, encode
   tiles with `vae_encoder.rs`, audio with P14.1, honour the image-size
   policy.
4. **P14.4 — the Ref2VA presentation** (`h3_multimodal.c:243`): ordered
   references, their spans and tags.
5. **P14.5 — plumbing**: sequence rows, conditioning key, CLI flags, the 12
   cap, the checkpoint requirement.
6. **P14.6 — parity gate**: 512x512, 22 frames, 2 steps, two image references,
   hashes equal to the reference's, then one render.

Acceptance: the 2-step hashes equal on both sides for an image-only reference
set, a reference-free run unchanged (existing goldens), the 12-reference and
missing-checkpoint refusals, and one rendered clip.

### P14.1 — The AudioVAE encode half (2026-10-25)

Done. The encoder is `vae_audio::encode`. The five stages and their strides
are `h3_audio_vae.c:714-720`'s (`{2,4,4,5,5}`); each stage is `snake ->
dilated residual stack -> downsample`, then the final `snake` at 2048, the
`conv1` to 32, the attention branch's causal SDPA, attention pool and
projection, and the MLP's GEGLU, all read from `:746-1273`. The output is the
released inverse normalization (`:1253-1268`): channel-major `[32,2,t]`.

Five F32 primitives carry it, each citing its reference kernel:
`h3d_geglu_f32` (`h3_gpu_cuda.cu:2912-2920`), `h3d_snake1d_f32`
(`:2935-2943`, the plain snake, not the alias-free variant),
`h3d_audio_qkv_split_f32` (`:1829-1848`), `h3d_audio_attention_pool_f32`
(`:1874-1893`) and `h3d_sdpa_causal_f32` (the F32 branch of
`h3_attention_kernel`, `:1337-1419`). Layer norm reuses the existing
`h3d_layer_norm_f32` (`h3_dit_gpu.cu:1671`).

Evidence, on the Spark: media `p14-audio-2s.wav`, 64000 samples at 32 kHz
stereo, built from `u17-long.mp4` with `-af apad -t 2.0` (`duration_ts`
64000, sha256 `301bfd0f...`). The reference probe
`tools/minimax_h3/audio_encoder_probe.c` (built against `~/h3-oracle/libh3.a`,
ffmpeg 6.1.1) and the port gate `h3_audio_encoder_gate`, same media,
`--samples 64000`:

  PCM    samples=64000 fnv=edc4afefc0ec1d6a                both sides
  LATENT channels=32 stereo=2 t=80 values=5120
         fnv=d03a1d52c0a4352b
         first=0.217818677 last=0.404173136                both sides

Bit-identical, so there is no deviation to name. Times: reference encode
2.032 s; port 2.318 s plus 0.053 s read. The hashes reproduce across an
independent rerun.

Recorded reference behaviour: `h3_ffmpeg_read_audio_f32` (`h3_ffmpeg.c:450-484`)
reads exactly `max_samples*2*4` bytes, then one byte, then closes the pipe and
requires ffmpeg to exit 0. Media longer than the `-t` cut therefore makes a
still-writing ffmpeg take SIGPIPE and the probe report "could not decode a
stereo soundtrack"; a 3.25 s looped copy failed this way while the port read
it fine. The gate media must end at the cut (exactly 2 s here), or the
reference fails on its own reader, not on the encoder.

### P14.2 — The Ref2VA component (2026-10-25)

Done. `minimax_h3::partition` reproduces `h3_create`'s checkpoint check
(`h3.c:425-452`) without the device probe: `FL2VA/transformer/config.json` and
`FL2VA/tokenizer/tokenizer.json` must be regular files with the reference's
message (`h3.c:385-395`), the four `FL2VA` shard directories are inventoried
in `h3.c:430-433`'s order through the existing `h3_st_inventory_dir` port, and
`Ref2VA/transformer` is inventoried only when
`Ref2VA/transformer/model.safetensors.index.json` is a regular file
(`h3.c:444-452`). `Catalog::partition` is `h3.c:872-875`'s gate and returns
that refusal verbatim; `Partition::component` is the five-path switch of
`h3.c:927-936`, which reads *every* component from the reference partition and
not only the transformer. `Engine::open` now builds the catalog, so a broken
component fails at open (0.016 s here) instead of at the first render.

The gate is `tests/h3_partition_gate <model_dir> [--references N]`, host-only
(no device object, no CUDA library). It prints `h3_st_inventory_dir`'s four
counters per component, an FNV `table` over every tensor's shard, name, dtype,
shape, span and offset, an FNV `sample` over the first 4096 bytes of every
tensor, and the selected partition.

Evidence, Spark `gx10-e1d2`, root `/mnt/h3-artifacts/MiniMax-H3`. Both
partitions are installed in full (all 14 text-encoder shards, 13 transformer
shards), which `find` and `stat` confirm; a gvfs `ls` glob truncated its
listing and briefly suggested two shards were missing.

  fl2va/text_encoder     files=14 tensors=1058 tensor_bytes=66714780128
  fl2va/transformer      files=13 tensors=535  tensor_bytes=66280430144
  fl2va/video_vae/source files=1  tensors=560  tensor_bytes=10415484128
  fl2va/audio_vae        files=1  tensors=1087 tensor_bytes=605306340
  ref2va/*               the same five counters, every component
  PARTITION references=2 selected=Ref2VA      references=0 selected=FL2VA
  OPEN seconds=0.016

The reference's own reader prints the same eight totals
(`./st_probe <dir> | tail -1`, built from `h3.c` at `eae5d54`), so the
inventory is two-sided. The port's layout check accepts the reference
transformer unchanged: `minimax_h3_inspect --layout Ref2VA/transformer` exits
0 with `tensors=535 blocks=50 refiner=2 hidden=5376`, the same line FL2VA
prints, and the two index files agree on all 535 tensor names, shard
assignments and metadata.

Refusal, live: an FL2VA-only root (`~/p14-fl2va-root`, a symlink to `FL2VA`)
with `--references 2` prints `REFUSED references=2 ordered references require
the Ref2VA checkpoint` and exits 3, the reference's `h3.c:873-874` message.

Measured artifact fact, and it changes the P14.4/P14.5 plan: full sha256 of
every text-encoder shard, `audio_vae`, `video_vae/source` and the first
transformer shard, both partitions, diffs to one line. The tokenizer, text
encoder, video VAE and audio VAE are the *same bytes* in both partitions; only
the transformer differs (sample hash `14705eea42ade724` against
`484052d34b0ea2af`, shard 1 `0b338656...` against `902d6ae7...`). A reference
request therefore needs only the 62 GB Ref2VA transformer as new weight state.
The port still resolves every path through the partition, as the reference
does; reusing the FL2VA-resident encoder is a path substitution over identical
bytes and is left to P14.5, where the memory plan is decided.

### P14.3a — The reference media canvas policy (2026-10-25)

Done. `host::adapt_canvas`, `host::reference_image_canvas` and
`host::reference_video_canvas` port `h3_host.c:49-124` exactly: the 768-px
nominal canvas at the source aspect with the `H3_MAX_PIXELS` downscale
(`:49-76`), the down-only aspect-preserving image policy that matches the
target area or a short-edge cap (`:77-100`), and the video policy that keeps a
smaller source's own snapped size instead of enlarging it (`:101-124`). Every
axis snaps with `nearbyint`, which rounds ties to even, so the port uses
`round_ties_even`: `720 / 32` is exactly 22.5 and the reference answers 704,
not 736.

The gate is `tests/h3_canvas_gate` (host-only) against
`tools/minimax_h3/reference_canvas_probe.c` built from `/data/h3.c` at
`eae5d54`. Both run the same 170-row sweep (17 sources x {adapt, video} plus
17 x 4 targets x 2 caps for images) and print the same TSV rows:

  REF_EXIT=0  PORT_EXIT=0  DIFF_EXIT=0      170 rows, no line differs

Rows worth keeping: `ADAPT 17x9993 -> 32x24640` (the one-multiple clamp),
`VIDEO 1280x720 -> 1280x704` (ties to even) and `VIDEO 100x100 -> 96x96` (a
small source is snapped down, never enlarged). Three unit tests in `host.rs`
pin those values, so a later `round()` swap fails locally.

Still open in P14.3: the reference media prep itself (decode at the canvases
above, the `vae_encoder` tile encode, `vae_audio::encode` plus the
`[32,2,t] -> [2,t,32]` repack and the per-span jitter at `seed`/`seed + 1`).

### P14.3b — Reference media prep (2026-10-25)

Done. `minimax_h3::reference::prepare` turns one `Reference` into a
`Prepared`: the media canvas from P14.3a, the decode at that canvas, the video
VAE encode, the patchified rows, the Qwen vision blocks and timestamps, and the
audio rows. Citations: the visual half is `h3.c:1052-1128`, the patchify
`:1296-1328`, the timestamps `:1096-1114`, the audio source choice and
truncation `:1126-1170`, the `[32,2,t] -> [2,t,32]` repack `:1182-1197`, and
the per-span jitter with `seed` for video and `seed + 1` for audio
(`h3.c:344`, `:348`) — a fresh generator per span, as `augment_span` does for
anchors.

One reference rule the gate found, and it is a real constraint on any
reference request: the video reader trims decoded frames to `frames % 17 == 5`
(the released cadence, `h3_ffmpeg.c`), and a truncated soundtrack needs at
least 64000 samples = 48 decoded frames. 48 frames trims to 39, which is
52000 samples, so the smallest *accepted* request is 56 frames, which is what
the reference's own message says ("request at least 56 output frames").

Evidence, Spark, media `p143-image.png` (1344x576, one frame of
`u17-long.mp4`) and `p143-video-small.mp4` (256x128, 120 frames, 5 s, AAC
track), `--frames 56 --seed 42`:

  image       1344x576 frames=1  latent=1x36x84   row_elements=72576
              rows_fnv=4068bd8e1946076b  vision_blocks=1
  video       256x128  frames=56 latent=14x8x16   row_elements=43008
              rows_fnv=d3b99fd56eec6ab9  audio_t=93 audio_rows=5952
              audio_fnv=2d6f8000a6ab0c99 samples=73643
              vision_blocks=3 timestamps=0.250,1.250,2.000
  video+audio the same numbers: both forms cut the same track to the same
              length (samples 73643 each), so an independent second encode of
              the same pixels reproduces both hashes
  TOTAL       references=3 audio_samples=147286 within_15s=true 331.631 s

Three runs, the third rebuilt after the audio-source refactor, returned
identical hashes and TOTAL (331.631 s, 330.763 s, 332.454 s), so the
composition is deterministic on the Spark.

The gate is port-side. Its primitives each have a two-sided gate already
(canvas policy P14.3a, media readers, the video VAE encoder at P12.4, the
audio VAE encoder at P14.1); the two-sided proof of this composition is
P14.6's render parity, and nothing here claims more than that.

The 331 s is the VAE encoder's cost at this canvas, not a regression: 56
frames at 256x128 is ~2.2x the pixels-times-frames of the P12.4 gate's five
frames at 512x192 (45 s). A first attempt at 107 frames and 1344x576 was
aborted after ten minutes; P14.6 must size its parity clip with this cost in
mind.

## P15 — The local v1 contract (opened and redirected 2026-10-25)

History, because it reverses a committed decision. P15 first adopted the hosted
platform's `/v2` contract (commit 2093add4) on the reading that the official
repo's own scripts call it. A full read of those scripts the same day showed
two contracts, not one:

- every `/v2` caller uses `$MINIMAX_API_BASE` with a Bearer token — the hosted
  platform: `full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh:8`,
  `full-2k-i2va-h3-context-ir.sh:8`, `full-2k-i2va-h3-regenerate-2k.sh:39`;
- every local caller uses `$SGLANG_DEPLOYMENT_URL` or `localhost:30010` and
  the same three routes — `POST /v1/videos`, `GET /v1/videos/{id}`,
  `GET /v1/videos/{id}/content`: `full-2k-t2va-h3-base.sh:24-31`,
  `reproducible-768p-fl2va-request.sh:10-44`;
- the two `/v2` routes the local contract lacks, context-ir and regeneration,
  are the hosted-only features (`README.md:104`, `:170`).

The operator reversed the decision the same day. The served surface is the
local one, `/v1` with the SGLang body; `/v2` gets no route at all — not
implemented and not stubbed with a 501, because no local client speaks it.

The contract is `docs/minimax-h3/local-v1-contract.md`: three routes, five
body fields (`task`, `prompt`, `conditions`, `target`, `seed`), t2va and fl2va
at 768P, keyframes as data URIs, every other value refused by name.

The units, in order, each one commit with its own tests and evidence:

1. **P15.1 — the contract and the mapping, no HTTP**
   (`crates/ds4-core/src/minimax_h3/local_api.rs`): body parsing with unknown
   fields refused, task/role/condition validation, ratio to canvas
   (`presets::tier_canvas`), anchor aspect to canvas (`presets::canvas_for`,
   the same rule for a ratio stated as pixels), duration to aligned frames,
   anchor bytes to `AnchorImage`, and the 21:9 refusal with its numbers.
2. **P15.2 — the swap** (in the worktree, uncommitted): `/v1/videos` takes the
   SGLang body through `local_api`, and the OpenAI carriage is gone —
   `ImageRef`, `VideoRequest`, `parse_request`, the thirteen-field
   `VideoResource`, `VideoList`, `VideoDelete` and `ListQuery` deleted from
   `video_api.rs`, `/v1/models`, list and delete deleted from `h3_video.rs`.
   What remains there is the task record, the four states and the bounded
   store. Consequences, each measured by a test:
   - the create answer is the id alone, as `full-2k-t2va-h3-base.sh:24` reads
     it; the record is `{id, status, size, seconds, frames, seed, created_at,
     completed_at, error}` and does not echo the prompt;
   - `RenderSpec` is gone: the renderer takes the validated `Plan`, so the
     wire's resolved target is the render contract and the seed comes from the
     body instead of a startup flag. `--model` and `--seed` leave the binary
     with it;
   - the query string is dropped before routing, because the contract has no
     query parameter (`?variant=` was the OpenAI download knob);
   - the renderer argv gains `--seed` (`main.c:330`) beside `-p`, `-o`,
     `--width`, `--height` and `--seconds`.
3. **P15.3 — the engine request**: `last_frame` reaches `RenderRequest`
   (`engine.rs:71`) through both renderers and `AnchorFile`. Folded into
   P15.2, since `RenderSpec` disappeared and both anchors now ride the plan;
   `CommandRenderer` passes `--first-frame`/`--last-frame` (`main.c:331-332`).
   What is left for P15.6 is the live proof with a last-frame keyframe.
4. **P15.4 — refusals on the wire** (landed with P15.2): one test drives every
   row of the contract page's refusal table through `server.handle` and
   asserts the status and the code — eighteen bodies, from `invalid_json` to
   `unsupported_uri`, plus the four 404 rows and the 429 and 503 ones in their
   own tests.
5. **P15.5 — the stand-in gate on the 4070** (landed 2026-10-25,
   `docs/minimax-h3/v1-local-gate.txt`): the real server on a real socket, a
   renderer that writes an H.264/AAC clip from the argv the bridge builds, a
   fixed curl transcript over every route, and `ffprobe` of both downloads.
   It proves the wire, the mapping and the refusals; it says nothing about h3
   output, which is P15.6's job.
6. **P15.6 — the real engine gate**, on the host that holds the weights: t2va
   768P 16:9, then fl2va with a data-URI keyframe, poll to `completed`,
   download, `ffprobe`, and the engine's stage seconds. Open, with the wire
   proven and the render stopped 2026-10-25: the Spark clone answered
   `{"id":"video_1"}` and the record read `1344x768`, `4 s`, `107 frames`, then
   `in_progress` for 33 minutes before the operator stopped it. Measured
   per-step cost says why: 40.2 s at 4,536 tokens and 112.9 s at 9,072 (2.8x
   for 2x), so the gate shape's 32,256 tokens is hours at 20 steps. Two
   changes came out of that attempt: the in-process engine now emits the
   reference's progress channel (`STEP n/N seconds=`, `h3.c:649-676`), and the
   CLI renderer learned the reference flags (`h3-rs`, `main.c:429-490`).
   The next bounded gate is 768P, 4 s, `--steps 2`.
7. **P15.7 — ComfyUI and the operator skill**: the node lives outside this
   repo, in ComfyUI's own Python runtime; it calls the three routes and hands
   ComfyUI a local file. Then `h3-prompt-writing`, the one portable skill of
   the nine in `/data/MiniMax-H3/skills`.
8. **P15.8 — ref2va on the local contract** (landed in the worktree,
   2026-10-25): the ordered-reference path ported from the reference.
   - the ordered presentation, `h3_multimodal.c:243-360`: `<Audio n>: ` for a
     reference with a soundtrack, `<Picture n>: ` for an image, `<Video n>: `
     then one `<t.t seconds>` per sampled block, then the prompt
     (`multimodal::ref2va`); a video's blocks are the two-frame pairs
     `h3.c:832-855` extracts from channel-major pixels (`reference::vision_pair`).
   - the layout segments, `h3_host.c:367-433`: image, audio and video reference
     spans in request order, the cursor advancing by one, by the audio length
     and by `max(audio, video span)`; a video's rows carry `REF_IMAGE`, as the
     reference labels them. Both cases are pinned against `h3_layout_build`
     compiled from `/data/h3.c` at `eae5d54`.
   - the engine (`engine.rs`): references decoded at their own canvases, rows
     concatenated in order, soundtracks held to the 15-second set budget
     (`h3.c:1171-1176`), the Ref2VA partition selected by
     `Catalog::partition` (`h3.c:872-875`), and the reference rows in the memo
     key so a second request cannot reuse another clip's embedding.
   - the contract (`local_api.rs`): `ref2va` accepted with up to 12 references
     (`h3.c:588`), roles `reference`/`reference_video`/`reference_audio`,
     `base_video` refused, media type checked against the condition, and
     `auto` resolved from the first visual reference (a video's size probed at
     create from a temp copy, removed at once).
   - the renderer boundary: `--ref-image`, `--ref-video`, `--ref-silent-video`
     and `--ref-audio` in request order (`main.c:438-490`), temp copies removed
     after the render (`MediaFile::write_reference`).
   - the renderer CLI itself (a port gap found on the Spark, 2026-10-25):
     `h3-rs` parsed none of the reference flags the reference CLI carries
     (`main.c:51-56`, `:333-338`, `:429-490`), so only `--model-dir` could
     serve a reference. It now parses all six, including
     `--ref-video-audio VIDEO AUDIO` and `--ref-image-size match|max`, switches
     its component root to `Ref2VA` (`h3.c:872-875`), refuses anchors with
     references (`h3.c:597-600`) before reading a weight, and shares one port
     of `h3.c:1027-1200` with the engine (`reference::prepare_set`).
   - tests: presentation order and refusals, two layout cases pinned to the C
     oracle, the contract mapping and its refusals, the wire reaching the
     renderer, and the argv order and cleanup.
   - what is not proven here: no Ref2VA checkpoint on the 4070, so the live
     ref2va render is the same blocked gate as P15.6 (it needs the Spark).

Ordering constraint: the body swap and the engine request land before any
gate, and the real engine gate passes before the ComfyUI node is called done.
ref2va's live render rides the same host and gate as P15.6.

What the port knowingly does not provide: HTTP auth (loopback bind only),
moderation, quota, billing and token usage, callbacks, CDN URLs with expiry,
the hosted platform's routes, and `base_video`. The two hosted-only features
get no route rather than a stub.

Watermarking is not provided either, and nothing implements it (operator
decision, 2026-10-25). The create schema declares seven properties and no
watermark (`VideoGenerationV2Req` in `v2-video-generation.openapi.json`); the
field exists only on the two regeneration request schemas (`:1444`, `:1503`),
and regeneration is not served. So there is no create field to accept, no
branch to refuse, and no watermark ever applied to the output.


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
| `--layers 45/40` | `h3.h:82` | Ported and gated: gate-ranked mask (`h3_dit.c:1196-1244`), hashes equal at 45 and 40 |
| `--core-reuse N` | `h3.h:85` | Ported and gated: cached core delta (`h3_dit.c:2161-2165`, `:2306-2312`), hashes equal at 4 and 6 |
| `--token-reduction` | `h3.h:89`, `h3_dit.c:378-470` | Ported and gated (`h3d_token_pool`/`h3d_token_expand`, `--token-reduction` in `h3-rs`); hashes equal to the reference on and off |
| `--reuse N` | `h3.h:79` | Ported and gated (reuse 1 and 3) |
| Ref2VA ordered references | `h3_multimodal.c:243`, `h3_cli.c:173`, `:322`, `:332` (up to 12) | Component loader ported and gated (`partition.rs`, `h3_partition_gate`, P14.2); the reference media prep, the ordered presentation and the request plumbing are P14.3-P14.5. Only `SegmentKind::RefImage` (`sequence.rs:32`) and the DiT row-map arms (`dit.rs:610`) exist below them |
| Interactive session (`!` commands) | `h3_cli.c:625` and the handlers above it | Unported: the CLI is one shot |
| `--use-int8-row-fc2` | Metal/M5 only, CUDA no-op (`README.md:119-121`) | Refused |
| `--show`, `--zoom`, preview frames | `h3_terminal.c`, `h3.h:125-127` | Ported for the CLI (`--preview-dir` PPMs, Kitty/iTerm2 display); the server stays headless |
| Resident preview decoder | `h3.c` preview path | Ported (`vae_video::PreviewDecoder`, one frame per step from the middle chunk) |
| CPU decode | `h3.c` CPU path | Refused; Linux CUDA is the reference |
| Metal backend | `h3_metal.m`, `metal/*` | Out of scope; CUDA is the release reference |
| h3c studio (FastAPI + React) | `webui/` | Refused: no Python in the deliverable. The official local v1 surface is the replacement (P15); a UI is ComfyUI, outside this repo |
| Postprocessing hook | `docs/POSTPROCESSING.md` | Out of scope |
| Device probe | `h3_device_cuda.cu` | Ported and gated (`h3_device_gate`, `h3-rs --info`); field parity with the reference's `h3_device_test` |
| Tiled-VAE quality bench | `h3_vae_bench_quality` | P0 remainder; run with P9.5/P11 |
| Image/video/audio encoders | `h3_vision_encoder.c`, `h3_video_encoder.c`, `h3_audio_vae.c` encode half | Vision tower and the VideoVAE encoder ported and gated (`h3_vision_gate`, `h3_video_encoder_gate`); the audio VAE's encode half (`h3_audio_vae.c:1275`) is unported |

## What has been measured so far, and what has not

Measured: the E9 output stage and the 20-step parity (identical hashes at
256x256 and 512x512); the 256x256/22-frame/2-step decode comparison above;
identical audio PCM at that workload.

Not measured yet: the `mmap`-vs-`pread` copy hypothesis; the reference's
864x480 denoise; the 39/56/107-frame VAE path; any preset; any server
request through the in-process engine; the highest canvas that fits.

Nothing in this page has been adopted on the strength of a hypothesis. P9.1
is a hypothesis until `p9_copy_bench` and an A/B say otherwise.
