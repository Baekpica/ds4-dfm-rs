---
name: h3-video
description: Generate MiniMax-H3 video clips with sound on this machine through the local v1 server (POST /v1/videos, poll GET /v1/videos/{id}, download GET /v1/videos/{id}/content). Use when the user wants a short clip from text (T2VA), from a first or last frame (FL2VA/I2VA), or from ordered image/video/audio references (Ref2VA); also when a job must be started, polled, downloaded, or a refusal explained.
---

# H3 video (local v1 server)

Plain shell plus curl: no MCP server and no hosted API in the path, so any
agent that can run a command can use it. The frontmatter above is the skill
format open-grok and Claude Code read; `AGENTS.md` beside this file is the
instruction-file form for Codex.

One GPU box renders; one binary serves. The server is `h3-server-rs` from this
repository and the checkpoint lives on the same host. The wire is the official
local contract — three routes, five body fields, no `/v2`
(`docs/minimax-h3/local-v1-contract.md`).

## 1. Start the server

```bash
make h3-server-rs                       # build, on the host that has the weights
H3_MODEL_DIR=/mnt/h3-artifacts/MiniMax-H3 \
H3_STEPS=2 \
skills/h3-video/scripts/h3-serve.sh     # listens on 127.0.0.1:8090
```

`H3_STEPS` is the number of denoise evaluations (default 20). Two is the
smoke-test setting: the clip still renders and muxes, its quality is not the
release one. The first request after start pays the checkpoint load.

If the server answers 503 `renderer_disabled`, it was started without
`--model-dir` and without `--renderer`; restart it.

To serve another machine, bind the reachable interface:

```bash
H3_BIND=0.0.0.0:8090 H3_MODEL_DIR=/mnt/h3-artifacts/MiniMax-H3 \
  H3_STEPS=20 skills/h3-video/scripts/h3-serve.sh
# then, from any host on the network:
skills/h3-video/scripts/h3-video.sh --url http://192.168.1.91:8090 "..."
```

There is no authentication in this contract, so a bind beyond loopback lets
every host on that network queue renders and spend the GPU.
`tools/minimax_h3/systemd/h3-server-rs.service` and its `env.example` run it
as a user service; `loginctl enable-linger "$USER"` keeps it up without a
login.

## 2. Make one clip

```bash
skills/h3-video/scripts/h3-video.sh "A red kite over a grey sea at dusk, wind in the mic"
```

The script posts the body, prints the id, polls `status` until `completed`,
downloads the MP4, then probes it. Flags it accepts:

| Flag | Meaning |
| --- | --- |
| `--url URL` | server base (default `http://127.0.0.1:8090`) |
| `--ratio R` | `auto`, `16:9`, `9:16`, `4:3`, `3:4`, `1:1` (default `16:9`) |
| `--seconds N` | 4 through 15 (default 4) |
| `--seed N` | non-negative integer (default 1) |
| `--task T` | `t2va`, `fl2va`, `ref2va` (default: inferred from the conditions) |
| `--short-edge N` | must be 768; the checkpoint has one tier |
| `--keyframe FILE` | first-frame anchor (`role: keyframe`, `frame_index: 0`) |
| `--last-frame FILE` | last-frame anchor (`role: keyframe`, `frame_index: 1`) |
| `--ref-image FILE` | ordered reference (`role: reference`) |
| `--ref-video FILE` | ordered reference, its own soundtrack included (`role: reference_video`) |
| `--ref-audio FILE` | ordered reference (`role: reference_audio`) |
| `--prompt-file FILE` | read the prompt from a file instead of the argument |
| `--out FILE` | output path (default `h3-<seed>-<epoch>.mp4`) |
| `--poll N`, `--timeout N` | poll interval, give-up seconds |
| `--dry-run` | print the JSON body and stop |

Raw form, if the script is not wanted:

```bash
curl -sS -X POST http://127.0.0.1:8090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"task":"t2va","prompt":"a red kite","conditions":[],
       "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":4},
       "seed":1}'
# -> {"id":"video_1"}
curl -sS http://127.0.0.1:8090/v1/videos/video_1
curl -sS -o out.mp4 http://127.0.0.1:8090/v1/videos/video_1/content
```

## 3. The body is exactly five fields

`task`, `prompt`, `conditions`, `target`, `seed`. An unknown field is refused
by name, not ignored (`local_api.rs:58`).

- `task`: `t2va`, `fl2va`, `ref2va`. Image-to-video is `fl2va` with one
  keyframe at `frame_index: 0` — that is what the reference's own i2va script
  sends (`scripts/readme/full-2k-i2va-h3-base.sh:9-16`).
- `prompt`: non-empty, at most 32000 characters.
- `target.short_edge`: must be `768`. The checkpoint has one tier.
- `target.aspect_ratio`: `auto` takes the aspect from the keyframe or the first
  image/video reference; `21:9` is refused with its numbers.
- `target.duration_seconds`: 4 through 15. The file is slightly longer than the
  request: frames come out on the engine's grid, 5+17n, so 4 s becomes 107
  frames at 24 fps. The record's `frames` is the authority.

## 4. Canvases

| Ratio | Canvas |
| --- | --- |
| 16:9 | 1344x768 |
| 9:16 | 768x1344 |
| 4:3 | 1024x768 |
| 3:4 | 768x1024 |
| 1:1 | 768x768 |

Audio is always 32 kHz stereo; video is H.264 with AAC in MP4.

## 5. Anchors and references

- A keyframe `uri` is a `data:image/png;base64,...` (or jpeg, webp) URI, at
  most 16 MiB decoded. Paths and http URLs are refused: this server never
  reads a caller-named path and never fetches a URL.
- `ref2va` takes up to 12 ordered references. At least one image or video is
  required; audio alone conditions nothing. A video reference needs at least
  2 seconds of soundtrack, and all soundtracks share one 15-second budget.
- Keyframes and references never appear in the same request.

## 6. Refusals

The answer is `{"error":{"message","type","param","code"}}`. Read `code`:

`invalid_json`, `unknown_field`, `invalid_task`, `invalid_prompt`,
`unsupported_short_edge`, `invalid_aspect_ratio`, `invalid_duration`,
`invalid_condition`, `unsupported_uri`, `queue_full` (429),
`renderer_disabled` (503), `video_not_ready` and `content_not_found` (404).
The full table with every value is `docs/minimax-h3/local-v1-contract.md`,
section Refusals. Do not retry a 400: the body will not become valid.

## 7. Time, and keeping a test small

Denoise cost grows with the token count, so the same duration is much cheaper
on a small canvas than at 768P. Measured on the GB10 Spark
(`docs/minimax-h3/v1-local-gate.txt`):

- 512x512, 56 frames, `--steps 2`: 52.5 s load, 95.5 s denoise (47.75 s per
  step), 542 s wall.
- 768P 4 s (107 frames) is 32,256 tokens at 16:9. Its per-step cost has been
  measured only at smaller shapes (40.2 s at 4,536 tokens, 112.9 s at 9,072),
  so treat a 20-step 768P render as hours. Measured with real weights: 768x768,
  107 frames, `--steps 2`, 22 min 8 s wall, solo. `--steps 2` is not a
  minute-scale test at this shape.
- One served render has completed: the 768x768 two-step clip above, through
  `/v1/videos`, downloaded from `/v1/videos/video_1/content`. The wire and
  every refusal are proven, and the engine path is proven through the CLI
  (`h3-rs`) at 512x512, 56 frames, 8 steps: `DENOISE ... seconds=291.869`
  (36.5 s per step), video VAE 81.038 s, about nine minutes end to end.
  Quality at two steps is a preview; twenty is the release setting.

The served surface has no small canvas: `short_edge` is 768 and the duration
floor is 4 s, so the smallest clip this API can express is 1:1 at 768x768 with
107 frames. For a cheaper smoke test below that, use the CLI (`h3-rs`) on the
host — 512x512 with a few frames and `--steps 2` — because the wire refuses
every other size. Keep `--steps 2` on a shared server until a real render has
been timed at the shape being requested.

## 8. Check the file

`ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames
-select_streams v:0 out.mp4`. Compare `nb_frames` with the record's `frames`;
a mismatch means the renderer wrote something else than the plan.

## 9. Prompt

The checkpoint expects H3's prompt structure, not a sentence. Read the
`h3-prompt-writing` skill before writing a real prompt.
