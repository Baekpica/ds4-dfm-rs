# Running the h3 local v1 host

`h3-server-rs` serves the local video contract of MiniMax-H3: the three routes
the official repository's own local clients call. This page is the operator
path. The contract itself, with every field and every refusal, is
[local-v1-contract.md](local-v1-contract.md). The upstream requests are in
the [official local client scripts](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/scripts/readme).

## Start the host

`make h3-server-rs` builds the binary with the in-process engine
(`--features native`).

| Flag | Default | Effect |
|---|---|---|
| `--bind ADDR:PORT` | `127.0.0.1:8090` | listen address (`h3-server-rs.rs:23`) |
| `--out-dir DIR` | `h3-videos` | MP4 directory, created at startup (`h3-server-rs.rs:24`) |
| `--model-dir DIR` | none | in-process engine root |
| `--steps N`, `--reuse N` | `20`, `1` | denoise evaluations, reuse window; engine only (`h3-server-rs.rs:28-29`) |
| `--renderer PROGRAM` | none | external renderer executable |
| `--renderer-arg ARG` | none | fixed renderer argument, repeatable |
| `--max-jobs N` | `8` | retained jobs, including completed and failed jobs |
| `--timeout-secs N` | `3600` | per-render timeout, external renderer only (`h3-server-rs.rs:26`) |

`--model-dir` and `--renderer` are exclusive (exit 2). A `--model-dir` build
without `native` exits 2; an unopenable checkpoint, a failed bind or out-dir
exits 1. With neither backend the banner says `renderer disabled` and every
create answers 503.

```bash
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 0 \
  --log scratch/h3-server.memory.jsonl -- \
  ./h3-server-rs --bind 127.0.0.1:8090 --out-dir /srv/h3-videos \
  --model-dir /path/to/MiniMax-H3 --steps 2 --reuse 1

# Alternative backend; use the same memory guard for a model renderer.
./h3-server-rs --out-dir /srv/h3-videos \
  --renderer /path/to/h3 \
  --renderer-arg -d --renderer-arg /path/to/MiniMax-H3
```

The native-engine example uses two steps to check execution. Use `--steps 20`
for the normal denoise setting. On GB10, the smallest square API request
below took about 22m20s at two steps, including status polling; see the
[measured workload](validation-2026-10-26.md#native-api-gate).

The engine keeps the prompt embedding and prepared DiT across requests and
serves one render at a time in arrival order; the engine channel is the queue
(`h3_engine.rs:3-7`, `engine.rs:13-16`, `video_api.rs:352-370`). The external
renderer runs per job as

```text
PROGRAM [fixed args...] -p P --width W --height H --seconds S
        [--first-frame P] [--last-frame P] [ordered reference flags...]
        --seed N -o PATH
```

stdin null, stderr tail kept (`h3_video.rs:312-315`, `:330-366`). The request
never chooses the program, an argument or a path (`h3_video.rs:21-27`).

## Create

`POST /v1/videos` takes the SGLang body and answers the id alone
(`h3_video.rs:518-548`); without a backend it answers 503 `renderer_disabled`
before touching the store.

```bash
video_id=$(curl --fail-with-body -sS http://127.0.0.1:8090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "t2va",
    "prompt": "A man in a black coat walks through a rainy neon-lit city.",
    "conditions": [],
    "target": {"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 4},
    "seed": 0
  }' | jq -er '.id')
```

```json
{"id":"<opaque id>"}
```

Use the returned id for polling and download; ids are not sequential names.
The shell examples require `jq`.

Exactly five fields, and an unknown one is refused rather than ignored
(`local_api.rs:55`, `:184-277`). The full table is the contract page; the
shapes this host enforces are:

| Field | Rule |
|---|---|
| `task` | `t2va`, `fl2va`, `ref2va` |
| `prompt` | non-empty, at most 32000 characters (`video_api.rs:25`) |
| `conditions` | at most two keyframes, or at most twelve references (`local_api.rs:52`, `:55`); a keyframe needs `frame_index` 0 or 1, unique, image only |
| `target.short_edge` | 768 |
| `target.aspect_ratio` | `auto`, `16:9`, `9:16`, `4:3`, `3:4`, `1:1` |
| `target.duration_seconds` | 4 through 15 (`local_api.rs:45-46`) |
| `seed` | non-negative integer, required |

A keyframe `uri` must be a base64 data URI (PNG, JPEG or WebP), at most
16 MiB decoded (`video_api.rs:30`, `local_api.rs:598-620`). Nothing is
fetched, no request names a path.

A ref2va body carries ordered references instead:

```json
{"task":"ref2va","prompt":"...","conditions":[
  {"type":"video","uri":"data:video/mp4;base64,...","role":"reference"},
  {"type":"audio","uri":"data:audio/mpeg;base64,...","role":"reference"}],
 "target":{"short_edge":768,"aspect_ratio":"auto","duration_seconds":5},"seed":0}
```

Order is the label order: a video's own soundtrack is `<Audio n>` before its
`<Video n>`, so the example above is the official script's pair. At least one
image or video is required; video/audio payloads are bounded by the 64 MiB
body cap, and the soundtracks share a 15-second budget
(`docs/minimax-h3/local-v1-contract.md`, "References").

## Poll

```bash
curl --fail-with-body -sS "http://127.0.0.1:8090/v1/videos/$video_id"
```

```json
{"id":"<opaque id>","status":"in_progress","queue_position":null,"size":"1344x768","seconds":10,"frames":243,"seed":0,"created_at":1792846990,"completed_at":null,"error":null}
```

The record is `video_api.rs:223-238`: the id, the state, its place in the
queue, the resolved target (`size`, `seconds`, `frames`), the seed and the
failure, in the field order above. It does not echo the prompt: a poll answer
would repeat up to 32000 characters on every request. A create pumps the store
at once (`h3_video.rs:539-543`), so a lone job reads `in_progress` on the
first poll, and a create behind a running render reads `queued` with its
`queue_position` (`video_api.rs:352-370`, `:446-456`).

| `status` | Meaning |
|---|---|
| `queued` | accepted, waiting for the single render slot; `queue_position` is its place in line |
| `in_progress` | holds the render slot; the engine runs it |
| `completed` | `completed_at` set, content available |
| `failed` | `completed_at` set, `error` carries `code` and `message` |

Only `completed` matters to the official client
(`full-2k-t2va-h3-base.sh:26-31`). Nothing expires: the store is in memory
only. A restart clears records while preserving MP4s in `--out-dir`; the
old files are not re-adopted or overwritten by new jobs.

Renderer failures land in `error.code`: `render_failed` (nonzero exit),
`render_timeout` (killed at `--timeout-secs`), `render_output_missing` (exit 0
without a non-empty file) and `renderer_spawn_failed`, with the last 4096 bytes
of stderr in the message (`h3_video.rs:447-500`, `:41`).

## Download

```bash
curl --fail-with-body -sS -o "$video_id.mp4" \
  "http://127.0.0.1:8090/v1/videos/$video_id/content"
```

The file on disk is `<out-dir>/<id>.mp4`; content is served with
`Content-Type: video/mp4` and `Content-Length`, and `HEAD` returns the same
headers with no body (`h3_video.rs:509-537`). Only a completed job has content:
any other status is 404 `video_not_ready`, and a vanished file is 404
`content_not_found` (`h3_video.rs:509-537`, `:539-545`). The local contract has
no query parameters, so `?variant=` and friends are not read; the query string
is dropped before routing (`h3_video.rs:587-647`).

## Errors

One body shape, all four keys always present, `param` null when no field
applies (`video_api.rs:56-113`):

```json
{"error":{"message":"aspect_ratio 21:9 is not served ...","type":"invalid_request_error","param":"target.aspect_ratio","code":"invalid_aspect_ratio"}}
```

| Code | Status | `param` | When |
|---|---|---|---|
| `invalid_json` | 400 | null | body is not a JSON object |
| `unknown_field` | 400 | null | a field the contract does not define |
| `invalid_task`, `invalid_prompt`, `invalid_target`, `invalid_seed` | 400 | the field | absent or the wrong type |
| `invalid_condition` | 400 | `conditions` | a bad role, index, type or count; ref2va without a visual, or with a keyframe |
| `unsupported_condition` | 400 | `conditions` | a reference role on t2va/fl2va, or `base_video` |
| `invalid_uri`, `unsupported_uri` | 400 | `conditions` | malformed data URI, a type that disagrees with the condition, an https URI or a bare path |
| `unsupported_short_edge` | 400 | `target.short_edge` | any tier but 768 |
| `invalid_aspect_ratio` | 400 | `target.aspect_ratio` | 21:9 or unknown |
| `invalid_duration` | 400 | `target.duration_seconds` | outside 4..15, or outside the frame range |
| `video_not_found`, `content_not_found` | 404 | `video_id` | unknown id, or a completed job whose file is gone |
| `video_not_ready` | 404 | `video_id` | content before completion |
| `unknown_url` | 404 | null | any route outside the three |
| `queue_full` | 429 | null | store at `--max-jobs` |
| `renderer_disabled` | 503 | null | no `--model-dir` and no `--renderer` |
| `server_busy` | 503 | null | 64 HTTP connections already active |
| `unsupported_encoding`, `request_too_large`, `invalid_request` | 400 | null | chunked, oversized or malformed HTTP |

`type` is `invalid_request_error` for 400 and 404, `rate_limit_error` for 429,
`service_unavailable_error` for 503 (`video_api.rs:64-113`).

## What is not served

- The hosted platform's `/v2/video_generation`, `/v2/h3_context_ir`,
  `/v2/video_regeneration` and their query routes: no route at all, 404
  `unknown_url`, not a 501 stub.
- `GET /v1/videos` (list), `DELETE /v1/videos/{id}` and `GET /v1/models`: the
  local scripts never call them, and the contract is three routes.
- `role: base_video` and 480P/2K: the reference roles and tiers this
  checkpoint does not provide.
- A non-data-URI condition and any caller-supplied path: nothing is fetched or
  opened from a request.
- Chunked request bodies: `Content-Length` only.

## Limits

- `--max-jobs` (default 8) caps retained jobs. There is no delete route, so
  at capacity every create is 429
  `queue_full` until the host restarts and clears its in-memory records.
  `--max-jobs 0` makes every create fail.
- The output directory is never pruned: one MP4 per job, kept until the
  operator removes it.
- Both backends render one job at a time in arrival order.
- At most 64 HTTP connections are active; further connections receive 503
  `server_busy`. Socket reads and writes have a 10-second inactivity timeout.
- `--timeout-secs` (default 3600, clamped to at least 1) bounds the external
  renderer only; the engine path has no per-job timeout.
- Steps and reuse are fixed at startup; the seed comes from the body.
- `Content-Length` is required; body cap 64 MiB, header cap 64 KiB
  (`http.rs:7-8`).
- One request per connection: every response carries `Connection: close`.
- No authentication: any client that can reach the listener can submit GPU
  work. The default bind is loopback; the operator controls network access.
