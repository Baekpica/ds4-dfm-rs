# H3 video — Codex instructions

Same job as `SKILL.md` beside this file, written as an instruction file
because Codex reads `AGENTS.md` rather than skill directories. Install by
copying this file to `~/.codex/AGENTS.md`, or by keeping it where the agent
runs.

## Make a clip

The renderer is one HTTP service on this machine or on the Spark. Ask it for a
clip; it never executes tools and never reads a local path from the request.

```bash
# server side, on the host with the checkpoint
make h3-server-rs
H3_MODEL_DIR=/mnt/h3-artifacts/MiniMax-H3 H3_STEPS=20 \
  skills/h3-video/scripts/h3-serve.sh          # 127.0.0.1:8090 by default

# client side, anywhere on the network
skills/h3-video/scripts/h3-video.sh --url http://192.168.1.91:8090 \
  --seconds 4 --ratio 16:9 "a red kite over a grey sea"
```

The script posts `POST /v1/videos`, polls `GET /v1/videos/{id}` until
`completed`, then downloads `GET /v1/videos/{id}/content`. Flags: `--url`
(server base), `--ratio` (`auto`, `16:9`, `9:16`, `4:3`, `3:4`, `1:1`),
`--seconds` 4..15, `--seed`, `--task` (`t2va`, `fl2va`, `ref2va`, inferred
when omitted), `--short-edge` (must be 768), `--keyframe FILE` (first frame),
`--last-frame FILE`, `--ref-image`, `--ref-video`, `--ref-audio`,
`--prompt-file FILE`, `--out`, `--poll`, `--timeout`, `--dry-run`.

## The body

Five fields only: `task` (`t2va`, `fl2va`, `ref2va`), `prompt`, `conditions`,
`target`, `seed`. An unknown field is refused by name.

- Image to video is `fl2va` with one keyframe at `frame_index: 0`; there is no
  `i2va` task here.
- `target.short_edge` must be 768. Canvases: 16:9 1344x768, 9:16 768x1344,
  4:3 1024x768, 3:4 768x1024, 1:1 768x768.
- Durations land on the engine's 5+17n frame grid, so 4 s is 107 frames.

## Rules

- Condition URIs are `data:<mime>;base64,...` only; a path or an http URL is
  refused. A keyframe decodes to at most 16 MiB.
- Ref2VA takes up to 12 ordered references and needs at least one image or
  video.
- A refusal is `{"error":{"code":...}}` with a 4xx; do not retry it.
- 503 `renderer_disabled` means the server has no model and no stand-in.
- Cost: `--steps` is a server flag, not a request field. At 768P expect tens of
  minutes per clip at 2 steps and hours at 20; ask before starting a long one.
  The smallest clip the API can express is 1:1, 4 s (768x768, 107 frames).
- A bind beyond loopback publishes an unauthenticated renderer to the network.

Full contract: `docs/minimax-h3/local-v1-contract.md`. Prompt structure:
`h3-prompt-writing/AGENTS.md`.
