# Using the h3 OpenAI Videos API

`h3-server-rs` serves an OpenAI-compatible Videos API for MiniMax-H3. This page
is the operator path: start the host, call every route, follow the lifecycle,
stay inside the limits. The wire shape and its deviations from the OpenAI
reference are analyzed in [openai-videos.md](openai-videos.md); every flag,
field, status and error below is cited from the code as `file:line`.

## Start the host

`make h3-server-rs` builds the binary with the in-process engine
(`Makefile:472-478`, `--features native`).

| Flag | Default | Effect |
|---|---|---|
| `--bind ADDR:PORT` | `127.0.0.1:8090` | listen address (`h3-server-rs.rs:17`, `:46-47`) |
| `--model ID` | `minimax-h3` | the only accepted model id (`h3-server-rs.rs:18`, `video_api.rs:77`) |
| `--out-dir DIR` | `h3-videos` | MP4 directory, created at startup (`h3-server-rs.rs:19`, `:183-189`) |
| `--model-dir DIR` | none | in-process engine root (`h3-server-rs.rs:50-52`) |
| `--steps N`, `--reuse N`, `--seed N` | `20`, `1`, `42` | denoise evaluations, reuse window, seed; engine only (`h3-server-rs.rs:21-23`, `:53-55`) |
| `--renderer PROGRAM` | none | external renderer executable (`h3-server-rs.rs:56-57`) |
| `--renderer-arg ARG` | none | fixed renderer argument, repeatable (`h3-server-rs.rs:58`) |
| `--max-jobs N` | `8` | retained jobs, also render concurrency (`h3-server-rs.rs:20`, `:59`) |
| `--timeout-secs N` | `3600` | per-render timeout, external renderer only (`h3-server-rs.rs:24`, `:60`) |

`--model-dir` and `--renderer` are exclusive; both exits 2
(`h3-server-rs.rs:192-196`). A `--model-dir` build without `native` exits 2
(`:172-179`); an unopenable checkpoint, a failed bind or out-dir exits 1
(`:154-170`, `:183-189`, `:205-212`). With neither backend the banner prints
`renderer disabled (POST /v1/videos answers 503)` (`h3-server-rs.rs:217-234`).

```bash
./h3-server-rs --bind 127.0.0.1:8090 --out-dir /srv/h3-videos \
  --model-dir /mnt/h3-artifacts/MiniMax-H3 --steps 20 --reuse 1 --seed 42

./h3-server-rs --out-dir /srv/h3-videos \
  --renderer /home/leandro/h3-oracle/h3 \
  --renderer-arg -d --renderer-arg /mnt/h3-artifacts/MiniMax-H3
```

The engine keeps the prompt embedding and prepared DiT across requests and
serves one render at a time in arrival order; the engine channel is the queue
(`h3_engine.rs:5-8`, `engine.rs:11-16`, `:413-422`). The external renderer runs
per job as `PROGRAM [args...] -p P --width W --height H --seconds S -o PATH`,
stdin null, stderr tail kept (`h3_video.rs:107-152`, `:160-240`); the request
never chooses the program, an argument or a path (`h3_video.rs:8-18`).

## Create

`POST /v1/videos` answers 200 with the `Video` object (`h3_video.rs:359-379`);
without a backend it answers 503 `renderer_disabled` before touching the store
(`h3_video.rs:360-365`).

```bash
curl -s http://127.0.0.1:8090/v1/videos -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3","prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4"}'
```

| Field | Type | Rule |
|---|---|---|
| `prompt` | string | required, 1 to 32000 characters (`video_api.rs:42`, `:376-396`) |
| `model` | string | optional, must equal `--model` (`video_api.rs:419-427`) |
| `size` | string | optional, preset or `WxH`, default `512x512` (`video_api.rs:70`, `:429-470`) |
| `seconds` | string | optional, positive decimal, default `"4"` (`video_api.rs:74`, `:472-482`) |
| `input_reference` | object | recognised and refused (`video_api.rs:397-415`) |

JSON (absent Content-Type means JSON, unknown keys ignored), urlencoded and
multipart reach one validation path (`video_api.rs:294-356`,
`h3_video.rs:635-651`); `openai-python` and the official curl example send
multipart. Form bodies accept only those five names, any other field is 400
`unknown_parameter`, and a repeated field keeps its last value
(`h3_video.rs:752-772`). The response is the job as submitted: status `queued`
and the thirteen fields of `video_api.rs:534-553`. The server moves it to
`in_progress` before the response is written, so the next GET already says
`in_progress` even while it waits behind another render (`h3_video.rs:375-377`,
`video_api.rs:693-704`).

## Size names and literal WxH

A preset name carries two geometries: the canvas the model renders and the
frame the MP4 carries (`presets.rs:114-145`, `:197-252`).

| `size` | Model canvas | Delivered frame | Scale |
|---|---|---|---|
| `16:9` | 1344x768 | 1920x1080 | anamorphic, 1.0159 |
| `9:16` | 768x1344 | 1080x1920 | anamorphic, 0.9844 |
| `21:9` | 1344x576 | 2520x1080 | uniform, 1.875 |
| `4:3` | 1024x768 | 1024x768 | none |
| `3:4` | 768x1024 | 768x1024 | none |
| `1:1` | 768x768 | 768x768 | none |
| `native_16x9` | 1344x768 | 1344x768 | none |
| `native_9x16` | 768x1344 | 768x1344 | none |
| `instagram_4x5` | 768x960 | 1080x1350 | uniform, 1.40625 |

Aliases: `reel_phone` is `9:16`, `youtube_1080p` is `16:9`, `instagram_1x1` is
`1:1`, `cinema_21x9` is `21:9` (`presets.rs:254-260`, `:262-275`).

A literal `WxH` is the model canvas and the delivered frame at once, with no
resize (`video_api.rs:429-469`). Only the model rules apply: both edges
multiples of 32 and at least 32, width times height at most 768*1344 =
1,032,192 (`video_api.rs:58-65`, `:235-254`), canonical spelling with a
lowercase `x` and no leading zeros (`video_api.rs:213-228`). `720x1280` and
`1280x720` fail the 32 rule; `1024x1792` fails the area cap.

- The response `size` is the delivered frame, so a preset comes back as pixels;
  the render canvas is never echoed (`video_api.rs:680`).
- A preset frame may exceed the area cap (1920x1080 is 2,073,600 pixels); the
  cap applies to the render canvas only (`presets.rs:148-192`).
- `--renderer` gets `--width`/`--height` from the delivered frame
  (`h3_video.rs:143-146`), while the engine gets canvas and frame separately
  (`h3_engine.rs:50-61`). A preset such as `16:9` therefore hands the external
  program 1920x1080, which the h3 canvas rules refuse: use a literal `WxH`
  with `--renderer`.

## Duration

`seconds` is parsed as a positive decimal, converted at 24 fps and rounded
(`video_api.rs:91-93`). The rounded frame count must land in 5 to 362 and its
upward alignment to `5 + 17*n` frames must be at least 22
(`video_api.rs:58-65`, `:490-503`); in practice 0.23 s to 15.1 s, and `"0.21"`
is refused with 400 `invalid_value` (`video_api.rs:505-513`). The job echoes
the string, while the clip is the aligned count: `"4"` becomes 107 frames,
4.458 s (`video_api.rs:80-87`).

## Lifecycle

| `status` | `progress` | Meaning |
|---|---|---|
| `queued` | 0 | accepted, in the POST response |
| `in_progress` | 0 | owned by a render thread, possibly still queued in the engine |
| `completed` | 100 | `completed_at` set, content available |
| `failed` | 0 | `completed_at` set, `error` carries `code` and `message` |

Statuses and progress come from `video_api.rs:159-177`, `:693-736`. Progress
never moves between 0 and 100, so a long render looks static. Nothing expires:
`expires_at` is always null (`video_api.rs:678`). A job lives until deleted or
the server restarts; the store is in memory only (`video_api.rs:618-634`) and
old MP4s in `--out-dir` are not re-adopted.

```bash
curl -s http://127.0.0.1:8090/v1/videos/video_1
```

```json
{"id":"video_1","object":"video","model":"minimax-h3","status":"in_progress","progress":0,"created_at":1792846990,"completed_at":null,"expires_at":null,"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4","remixed_from_video_id":null,"error":null}
```

Renderer failures land in `error.code`: `render_failed` (nonzero exit),
`render_timeout` (killed at `--timeout-secs`), `render_output_missing` (exit 0
without a non-empty file) and `renderer_spawn_failed`, with the last 4096 bytes
of stderr in the message (`h3_video.rs:186-239`, `:391-432`, `:41`).

## Download

```bash
curl -sS -o video_1.mp4 http://127.0.0.1:8090/v1/videos/video_1/content
```

The file on disk is `<out-dir>/<id>.mp4` (`h3_video.rs:373-374`); content is
served with `Content-Type: video/mp4` and `Content-Length`
(`h3_video.rs:498-502`), and `HEAD` returns the same headers with no body
(`h3_video.rs:298-310`). Only a completed job has content: any other status is
404 `video_not_ready` (`h3_video.rs:480-488`), and a vanished file is 404
`content_not_found` (`h3_video.rs:490-496`). `?variant=` accepts only `video`
(else 400 `unsupported_parameter`, `h3_video.rs:466-477`). `HEAD` is wired for
`/content` only; elsewhere it is 404 `unknown_url` (`h3_video.rs:341-356`).

## List

```bash
curl -s 'http://127.0.0.1:8090/v1/videos?limit=5&order=desc'
```

| Query | Values | Default |
|---|---|---|
| `limit` | integer 0 to 100 | `20` |
| `order` | `asc`, `desc` | `desc` |
| `after` | a video id | none |

Limits are `video_api.rs:46-48`, `:842-863`; ordering `:866-880`. The response
is `{"object":"list","data":[...],"first_id":...,"last_id":...,"has_more":...}`,
with `first_id`/`last_id` null on an empty page (`video_api.rs:556-577`).
`after` names the last id of the previous page; an unknown id is 400
`invalid_after` (`video_api.rs:778-791`). `limit=0` is legal and returns an
empty page. The `desc` default is an assumption, not reference behaviour
(`video_api.rs:867-869`). `GET /v1/models` answers one model entry whose
`created` is the server's start time (`h3_video.rs:505-519`).

## Delete

```bash
curl -s -X DELETE http://127.0.0.1:8090/v1/videos/video_1
```

Answers `{"id":"video_1","deleted":true,"object":"video.deleted"}`. Only
completed or failed jobs may be deleted; a queued or in-progress job is 400
`video_not_ready` (`video_api.rs:809-831`). Deletion removes the MP4
(`h3_video.rs:451-461`). Ids are `video_N` from a monotonic counter, never
reused (`video_api.rs:669-673`).

## Errors

One body shape, all four keys always present, `param` null when no field
applies (`video_api.rs:145-156`):

```json
{"error":{"message":"Invalid 'size': '720x1280': width and height must be multiples of 32 pixels.","type":"invalid_request_error","param":"size","code":"invalid_value"}}
```

| Code | Status | `param` | When |
|---|---|---|---|
| `invalid_json` | 400 | null | body is not a JSON object (`video_api.rs:294-310`) |
| `invalid_type` | 400 | field | wrong JSON type, or a file where text is required (`video_api.rs:365-370`, `h3_video.rs:722-728`) |
| `missing_required_parameter`, `string_below_min_length`, `string_above_max_length` | 400 | `prompt` | absent, empty, or over 32000 characters (`video_api.rs:376-396`) |
| `model_not_found` | 400 | `model` | id differs from `--model` (`video_api.rs:419-427`) |
| `invalid_value` | 400 | `size`, `seconds`, `limit`, `order` | failed a field rule (`video_api.rs:451`, `:507`, `:858`, `:875`) |
| `unsupported_parameter` | 400 | `input_reference`, `variant` | refused by name (`video_api.rs:410`, `h3_video.rs:469`) |
| `invalid_after` | 400 | `after` | cursor id not in the store (`video_api.rs:786`) |
| `unsupported_content_type`, `invalid_content_type`, `invalid_multipart`, `invalid_urlencoded`, `unknown_parameter`, `invalid_encoding`, `request_too_large`, `unsupported_encoding` | 400 | null | malformed or oversized body, form or multipart (`h3_video.rs:635-651`, `:659-710`, `:775-847`, `:889-894`, `:1142-1151`) |
| `video_not_found`, `content_not_found` | 404 | `video_id` | unknown id, or a completed job whose file is gone (`h3_video.rs:523-528`, `:531-536`) |
| `video_not_ready` | 404 | `video_id` | content or delete before completion (`h3_video.rs:480-488`, `video_api.rs:809-831`) |
| `unknown_url` | 404 | null | no such route, including a trailing slash (`h3_video.rs:341-356`, `:547-567`) |
| `queue_full` | 429 | null | store at `--max-jobs` (`video_api.rs:660-668`) |
| `renderer_disabled` | 503 | null | no `--model-dir` and no `--renderer` (`h3_video.rs:360-365`) |

`type` is `invalid_request_error` for 400 and 404, `rate_limit_error` for 429,
`service_unavailable_error` for 503 (`video_api.rs:123-143`).

## Refused today

- `input_reference` by name: 400 `unsupported_parameter`, whether JSON object,
  form string or multipart file part (`video_api.rs:397-415`, `h3_video.rs:762`).
- Every create without `--model-dir` or `--renderer`: 503 `renderer_disabled`
  (`h3_video.rs:360-365`).
- `variant=thumbnail` and `variant=spritesheet` (`h3_video.rs:466-477`).
- Characters, edits, extensions, remixes: no route, 404 `unknown_url`.
- Chunked request bodies: `Content-Length` only (`h3_video.rs:1037-1040`,
  `:1075-1077`).

## Limits

- `--max-jobs` (default 8) caps retained jobs and render threads; at capacity
  create is 429 `queue_full` until a finished job is deleted
  (`video_api.rs:660-668`). `--max-jobs 0` makes every create fail.
- `--model-dir` renders serially in arrival order (`engine.rs:11-16`,
  `:375-380`); `--renderer` gives each job a thread and a child, so up to
  `--max-jobs` run at once (`h3_video.rs:381-434`).
- `--timeout-secs` (default 3600, clamped to at least 1) bounds the external
  renderer only; the engine path has no per-job timeout
  (`h3_video.rs:186-228`, `h3-server-rs.rs:127-131`).
- Steps, reuse and seed are fixed at startup (`h3_engine.rs:26-41`).
- `Content-Length` is required; body cap 64 MiB, header cap 64 KiB
  (`http.rs:7-8`), form bodies capped at 64 fields (`h3_video.rs:48-49`).
- One request per connection: every response carries `Connection: close`
  (`error.rs:76-104`, `h3_video.rs:298-310`).
- No authentication: dispatch matches method and route only
  (`h3_video.rs:341-356`). Bind to loopback or a trusted network
  (`h3-server-rs.rs:17`).
