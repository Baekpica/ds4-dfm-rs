# Using the h3 OpenAI Videos API

`h3-server-rs` serves an OpenAI-compatible Videos API for MiniMax-H3. This page
is the operator path: start the host, call every route, follow the job
lifecycle, stay inside the limits. The wire shape and every deviation from the
OpenAI reference are analyzed in [openai-videos.md](openai-videos.md); this page
does not repeat that analysis.

Every flag, field, status and error code below is read from the code and cited
as `file:line`. Nothing is taken from OpenAI's documentation.

## Start the host

`make h3-server-rs` builds the binary with the in-process engine
(`Makefile:472-478`, `--features native`).

| Flag | Default | Effect |
|---|---|---|
| `--bind ADDR:PORT` | `127.0.0.1:8090` | listen address (`h3-server-rs.rs:17`, `:46-47`) |
| `--model ID` | `minimax-h3` | the only accepted model id (`h3-server-rs.rs:18`, `video_api.rs:77`) |
| `--out-dir DIR` | `h3-videos` | MP4 directory, created at startup (`h3-server-rs.rs:19`, `:183-189`) |
| `--model-dir DIR` | none | in-process engine checkpoint root (`h3-server-rs.rs:50-52`) |
| `--steps N` | `20` | denoise evaluations, engine only (`h3-server-rs.rs:21`, `:53`) |
| `--reuse N` | `1` | reuse window, engine only (`h3-server-rs.rs:22`, `:54`) |
| `--seed N` | `42` | sampler seed, engine only (`h3-server-rs.rs:23`, `:55`) |
| `--renderer PROGRAM` | none | external renderer executable (`h3-server-rs.rs:56-57`) |
| `--renderer-arg ARG` | none | fixed renderer argument, repeatable (`h3-server-rs.rs:58`) |
| `--max-jobs N` | `8` | retained jobs, also render concurrency (`h3-server-rs.rs:20`, `:59`) |
| `--timeout-secs N` | `3600` | per-render timeout, external renderer only (`h3-server-rs.rs:24`, `:60`, `:127-131`) |

`--model-dir` and `--renderer` are mutually exclusive; both exits 2
(`h3-server-rs.rs:192-196`). `--model-dir` in a build without `native` exits 2
(`h3-server-rs.rs:172-179`); a checkpoint that fails to open exits 1
(`h3-server-rs.rs:154-170`). A bind or out-dir failure exits 1
(`h3-server-rs.rs:183-189`, `:205-212`).

```bash
# In-process engine: one render at a time, the rest wait in arrival order.
./h3-server-rs --bind 127.0.0.1:8090 --out-dir /srv/h3-videos \
  --model-dir /mnt/h3-artifacts/MiniMax-H3 \
  --steps 20 --reuse 1 --seed 42
```

```bash
# External renderer: the reference CLI's own flags, no adapter.
./h3-server-rs --out-dir /srv/h3-videos \
  --renderer /home/leandro/h3-oracle/h3 \
  --renderer-arg -d --renderer-arg /mnt/h3-artifacts/MiniMax-H3 \
  --timeout-secs 3600
```

```bash
# No backend: the host still starts and answers 503 on create.
./h3-server-rs
```

Startup prints the bind, the model, the output directory and the backend
(`h3-server-rs.rs:217-234`); the last line is
`renderer disabled (POST /v1/videos answers 503)` when neither backend is set.

The engine keeps the prompt embedding and the prepared DiT across requests and
serves one render at a time, in arrival order; the wait inside the engine
channel is the queue (`h3_engine.rs:5-8`, `engine.rs:11-16`, `:375-380`,
`:413-422`). The external renderer is spawned once per job with
`PROGRAM [fixed args...] -p P --width W --height H --seconds S -o PATH`, its
stdin null, stdout discarded, stderr tail kept (`h3_video.rs:107-152`,
`:160-240`). The request never chooses the program, an argument, the working
directory or a path (`h3_video.rs:8-18`).

## Create a video

`POST /v1/videos` answers 200 with the `Video` object
(`h3_video.rs:359-379`). Without a backend it answers 503 `renderer_disabled`
before touching the store (`h3_video.rs:360-365`).

```bash
curl -s http://127.0.0.1:8090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4"}'
```

```bash
curl -s http://127.0.0.1:8090/v1/videos \
  -F 'prompt=A red cube rotates on a white background.' \
  -F 'size=512x512' \
  -F 'seconds=4'
```

| Field | Type | Rule |
|---|---|---|
| `prompt` | string | required, 1 to 32000 characters counted by `char` (`video_api.rs:42`, `:376-396`) |
| `model` | string | optional, must equal `--model` exactly (`video_api.rs:419-427`) |
| `size` | string | optional, preset name or literal `WxH`, default `512x512` (`video_api.rs:70`, `:429-470`) |
| `seconds` | string | optional, positive decimal, default `"4"` (`video_api.rs:74`, `:472-482`) |
| `input_reference` | object | recognised and refused (`video_api.rs:397-415`) |

Three encodings reach the same validation (`h3_video.rs:635-651`): JSON (absent
Content-Type means JSON; unknown keys ignored, `video_api.rs:294-356`),
`application/x-www-form-urlencoded`, and `multipart/form-data` (what
`openai-python` and the official curl example send). Form bodies accept only
those five names; any other field is 400 `unknown_parameter`, and a repeated
field keeps its last value (`h3_video.rs:752-772`).

The response is the job as submitted, before the render thread starts:

```json
{"id":"video_1","object":"video","model":"minimax-h3","status":"queued","progress":0,"created_at":1792846990,"completed_at":null,"expires_at":null,"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4","remixed_from_video_id":null,"error":null}
```

The server moves the job to `in_progress` before that response is written, so
the next GET already reports `in_progress` even while it waits behind another
render (`h3_video.rs:375-377`, `video_api.rs:693-704`).

## Size names and literal WxH

A preset name carries two geometries: the canvas the model renders and the
frame the MP4 carries (`presets.rs:114-145`, `:197-252`).

| `size` | Model canvas | Delivered frame | Scale |
|---|---|---|---|
| `16:9` | 1344x768 | 1920x1080 | anamorphic, 1.0159 horizontal over vertical |
| `9:16` | 768x1344 | 1080x1920 | anamorphic, 0.9844 |
| `21:9` | 1344x576 | 2520x1080 | uniform, 1.875 |
| `4:3` | 1024x768 | 1024x768 | none |
| `3:4` | 768x1024 | 768x1024 | none |
| `1:1` | 768x768 | 768x768 | none |
| `native_16x9` | 1344x768 | 1344x768 | none |
| `native_9x16` | 768x1344 | 768x1344 | none |
| `instagram_4x5` | 768x960 | 1080x1350 | uniform, 1.40625 |

Aliases resolve to the same presets: `reel_phone` is `9:16`, `youtube_1080p`
is `16:9`, `instagram_1x1` is `1:1`, `cinema_21x9` is `21:9`
(`presets.rs:254-260`, `:262-275`).

A literal `WxH` is the model canvas and the delivered frame at once: no resize
(`video_api.rs:429-469`). Only the model rules apply: both edges multiples of
32 and at least 32, and width times height at most 768*1344 = 1,032,192
(`video_api.rs:58-65`, `:235-254`). The spelling must be canonical: lowercase
`x`, digits only, no leading zeros (`video_api.rs:213-228`). So `720x1280` and
`1280x720` fail the multiple-of-32 rule and `1024x1792` fails the area cap;
`768x768`, `1024x768` and `512x512` pass.

Three things to know:

- The response `size` is the delivered frame, so a preset comes back as pixels
  (`video_api.rs:680`). The render canvas is never echoed.
- A preset's delivered frame may exceed the area cap (1920x1080 is 2,073,600
  pixels); the cap applies to the render canvas only (`presets.rs:148-192`).
- The external renderer receives `--width`/`--height` from the delivered frame
  (`h3_video.rs:143-146`), while the engine receives the render canvas and the
  delivered frame separately (`h3_engine.rs:50-61`). With `--renderer`, a
  preset name such as `16:9` hands the child 1920x1080, which the h3 canvas
  rules refuse. Use a literal `WxH` with `--renderer`.

## Duration

`seconds` is parsed as a positive decimal, converted at 24 fps and rounded
(`video_api.rs:91-93`). The rounded frame count must land in 5 to 362 and its
upward alignment to `5 + 17*n` frames must be at least 22
(`video_api.rs:58-65`, `:490-503`). In practice the window is about 0.23 s to
15.1 s; `"0.21"` is refused with 400 `invalid_value` on `seconds`
(`video_api.rs:505-513`).

The string is echoed unchanged in the job, while the clip is the aligned frame
count: `"4"` becomes 107 frames, 4.458 s (`video_api.rs:80-87`).

## Job lifecycle

| `status` | `progress` | Meaning |
|---|---|---|
| `queued` | 0 | accepted, present in the POST response |
| `in_progress` | 0 | a render thread owns it, possibly still waiting in the engine queue |
| `completed` | 100 | `completed_at` set, `content` available |
| `failed` | 0 | `completed_at` set, `error` carries `code` and `message` |

Statuses, `progress` and `error` come from `video_api.rs:159-177` and
`:693-736`. Progress never moves between 0 and 100; there is no partial
percentage, so a long render looks static.

Nothing expires: `expires_at` is always null (`video_api.rs:678`). A job lives
until it is deleted or the server restarts; the store is in memory only
(`video_api.rs:618-634`), and MP4s left in `--out-dir` by an earlier run are
not re-adopted.

## Poll a job

```bash
curl -s http://127.0.0.1:8090/v1/videos/video_1
```

```json
{"id":"video_1","object":"video","model":"minimax-h3","status":"in_progress","progress":0,"created_at":1792846990,"completed_at":null,"expires_at":null,"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4","remixed_from_video_id":null,"error":null}
```

A failed job carries the reason in the same object:

```json
{"id":"video_2","object":"video","model":"minimax-h3","status":"failed","progress":0,"created_at":1792846990,"completed_at":1792847001,"expires_at":null,"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4","remixed_from_video_id":null,"error":{"code":"render_failed","message":"renderer exited with status 1: ..."}}
```

Renderer failure codes: `render_failed` (nonzero exit), `render_timeout`
(child killed at `--timeout-secs`), `render_output_missing` (exit 0 without a
non-empty file), `renderer_spawn_failed` (`h3_video.rs:186-239`, `:391-432`).
The last 4096 bytes of the child's stderr ride the message (`h3_video.rs:41`).

## Download the MP4

```bash
curl -sS -o video_1.mp4 http://127.0.0.1:8090/v1/videos/video_1/content
```

```bash
curl -sSI http://127.0.0.1:8090/v1/videos/video_1/content
```

The file on disk is `<out-dir>/<id>.mp4` (`h3_video.rs:373-374`). Content is
served with `Content-Type: video/mp4` and `Content-Length`
(`h3_video.rs:498-502`). `HEAD` returns the same headers without a body, with
`Content-Length` equal to the file size (`h3_video.rs:298-310`). Only a
completed job has content; any other status is 404 `video_not_ready`
(`h3_video.rs:480-488`), and a completed job whose file is gone is 404
`content_not_found` (`h3_video.rs:490-496`). The optional `variant` query
accepts only `video`; anything else is 400 `unsupported_parameter` with
`param: "variant"` (`h3_video.rs:466-477`).

`HEAD` is wired for `/content` only; on any other route it answers 404
`unknown_url` (`h3_video.rs:341-356`).

## List jobs

```bash
curl -s 'http://127.0.0.1:8090/v1/videos?limit=5&order=desc'
```

| Query | Values | Default |
|---|---|---|
| `limit` | integer 0 to 100 | `20` |
| `order` | `asc`, `desc` | `desc` |
| `after` | a video id | none |

Limits are `video_api.rs:46-48` and `:842-863`; ordering is `:866-880`. The
response is `{"object":"list","data":[...],"first_id":...,"last_id":...,"has_more":...}`,
with `first_id`/`last_id` null on an empty page (`video_api.rs:556-577`).
`after` names the last id of the previous page; an id that is not in the store
is 400 `invalid_after` (`video_api.rs:778-791`). `limit=0` is legal and returns
an empty page. The `desc` default is an assumption, not reference behaviour
(`video_api.rs:867-869`).

## Delete a job

```bash
curl -s -X DELETE http://127.0.0.1:8090/v1/videos/video_1
```

```json
{"id":"video_1","deleted":true,"object":"video.deleted"}
```

Only completed or failed jobs may be deleted; a queued or in-progress job is
400 `video_not_ready` (`video_api.rs:809-831`). Deletion removes the MP4 from
`--out-dir` (`h3_video.rs:451-461`). Ids are `video_N` from a monotonic counter
and are never reused (`video_api.rs:669-673`).

## Models

```bash
curl -s http://127.0.0.1:8090/v1/models
```

```json
{"object":"list","data":[{"id":"minimax-h3","object":"model","created":1792846900,"owned_by":"ds4","shutdown_date":null}]}
```

`created` is the server's start time, not a model date (`h3_video.rs:505-519`).

## Errors

Every failure uses one body shape, with all four keys always present and
`param` null when no field applies (`video_api.rs:145-156`):

```json
{"error":{"message":"Invalid 'size': '720x1280': width and height must be multiples of 32 pixels.","type":"invalid_request_error","param":"size","code":"invalid_value"}}
```

| Code | Status | `param` | When |
|---|---|---|---|
| `invalid_json` | 400 | null | body is not a JSON object (`video_api.rs:294-310`) |
| `invalid_type` | 400 | field | wrong JSON type, or a file part where text is required (`video_api.rs:365-370`, `h3_video.rs:722-728`) |
| `missing_required_parameter` | 400 | `prompt` | no prompt (`video_api.rs:376-382`) |
| `string_below_min_length` | 400 | `prompt` | empty prompt (`video_api.rs:383-389`) |
| `string_above_max_length` | 400 | `prompt` | over 32000 characters (`video_api.rs:390-396`) |
| `model_not_found` | 400 | `model` | id differs from `--model` (`video_api.rs:419-427`) |
| `invalid_value` | 400 | `size`, `seconds`, `limit`, `order` | failed a field rule |
| `unsupported_parameter` | 400 | `input_reference`, `variant` | refused by name |
| `invalid_after` | 400 | `after` | cursor id not in the store |
| `unsupported_content_type` | 400 | null | not JSON, urlencoded or multipart (`h3_video.rs:659-686`) |
| `invalid_content_type` | 400 | null | multipart without a boundary, or a boundary over 70 bytes (`h3_video.rs:688-710`) |
| `invalid_multipart` | 400 | null | malformed or truncated multipart (`h3_video.rs:889-894`) |
| `invalid_urlencoded` | 400 | null | bad percent escape or non-UTF-8 value (`h3_video.rs:775-847`) |
| `unknown_parameter` | 400 | null | unknown form field (`h3_video.rs:765-770`) |
| `invalid_encoding` | 400 | null | form text value is not UTF-8 (`h3_video.rs:729-735`) |
| `request_too_large` | 400 | null | body over 64 MiB (`h3_video.rs:639-645`, `:1147-1151`) |
| `unsupported_encoding` | 400 | null | chunked request body (`h3_video.rs:1142-1146`) |
| `video_not_found` | 404 | `video_id` | unknown id (`h3_video.rs:523-528`) |
| `video_not_ready` | 404 | `video_id` | content before completion, or delete before completion (`h3_video.rs:480-488`, `video_api.rs:809-831`) |
| `content_not_found` | 404 | `video_id` | completed job, file gone (`h3_video.rs:531-536`) |
| `unknown_url` | 404 | null | no such route, including a trailing slash (`h3_video.rs:341-356`, `:547-567`) |
| `queue_full` | 429 | null | store at `--max-jobs` (`video_api.rs:660-668`) |
| `renderer_disabled` | 503 | null | no `--model-dir` and no `--renderer` (`h3_video.rs:360-365`) |

`type` is `invalid_request_error` for 400 and 404, `rate_limit_error` for 429
and `service_unavailable_error` for 503 (`video_api.rs:123-143`).

## What is refused today

- `input_reference`, by name: 400 `unsupported_parameter` on
  `input_reference`, whether it arrives as a JSON object, a form string or a
  multipart file part (`video_api.rs:397-415`, `h3_video.rs:762`).
- Every `POST /v1/videos` when the server started without `--model-dir` or
  `--renderer`: 503 `renderer_disabled` (`h3_video.rs:360-365`).
- `variant=thumbnail` and `variant=spritesheet`: only the MP4 exists
  (`h3_video.rs:466-477`).
- Characters, edits, extensions and remixes: no route, so 404 `unknown_url`.
- Chunked request bodies: `Content-Length` only (`h3_video.rs:1037-1040`,
  `:1075-1077`).

## Operational limits

- `--max-jobs` (default 8) caps retained jobs and therefore render threads. At
  capacity, create answers 429 `queue_full` until a finished job is deleted
  (`video_api.rs:660-668`). `--max-jobs 0` makes every create fail.
- With `--model-dir` renders are serial, in arrival order
  (`engine.rs:11-16`, `:375-380`). With `--renderer`, each job gets its own
  thread and child process, so up to `--max-jobs` run at once
  (`h3_video.rs:381-434`).
- `--timeout-secs` (default 3600, clamped to at least 1) bounds the external
  renderer only; the engine path has no per-job timeout
  (`h3_video.rs:186-228`, `h3-server-rs.rs:127-131`).
- Steps, reuse and seed are fixed at startup; a request cannot change them
  (`h3_engine.rs:26-41`).
- Bodies need `Content-Length`; the cap is 64 MiB and the header block 64 KiB
  (`http.rs:7-8`). Form bodies are capped at 64 fields and 64 parts
  (`h3_video.rs:48-49`).
- One request per connection: every response carries `Connection: close`
  (`error.rs:76-104`, `h3_video.rs:298-310`).
- No authentication: the dispatch matches method and route only, so no header
  is checked (`h3_video.rs:341-356`). Bind to loopback or a trusted network
  (`h3-server-rs.rs:17`).
