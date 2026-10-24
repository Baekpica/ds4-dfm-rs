# OpenAI Videos API surface for MiniMax-H3

Status: served by the standalone `h3-server-rs` binary. The port's own renderer
lands at phase P8; until then generation is delegated to an operator-configured
external program (the renderer bridge).

## Verified wire reference

Fetched and checked on 2026-10-24:

- https://developers.openai.com/api/reference/resources/videos
- https://developers.openai.com/api/reference/resources/videos/methods/create
- https://developers.openai.com/api/reference/resources/videos/methods/list
- https://developers.openai.com/api/reference/resources/videos/methods/retrieve
- https://developers.openai.com/api/reference/resources/videos/methods/delete
- https://developers.openai.com/api/reference/resources/videos/methods/download_content
- https://developers.openai.com/api/reference/resources/models/methods/list

`platform.openai.com/docs/api-reference/videos` redirects to the first URL
above. The reference marks the Videos surface deprecated; Sora 2 and the Videos
API shut down on 2026-09-24 and the page is retained as historical
documentation. The field list below is still the carriage this server serves.

## Routes

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/videos` | create a job |
| GET | `/v1/videos` | list jobs |
| GET | `/v1/videos/{video_id}` | retrieve a job |
| GET | `/v1/videos/{video_id}/content` | download MP4 bytes |
| HEAD | `/v1/videos/{video_id}/content` | content headers only |
| DELETE | `/v1/videos/{video_id}` | delete a finished job |
| GET | `/v1/models` | list the configured model |

`POST /v1/videos` answers 200 with the `Video` object, as the reference does.

## Video object

Exactly the reference's thirteen fields:

| Field | Type | Value here |
|---|---|---|
| `id` | string | `video_N`, monotonic |
| `object` | string | `video` |
| `model` | string | the configured model id |
| `status` | string | `queued`, `in_progress`, `completed`, `failed` |
| `progress` | integer | 0 while queued/in progress, 100 completed, 0 failed |
| `created_at` | integer | unix seconds |
| `completed_at` | integer or null | set on completed and failed |
| `expires_at` | integer or null | always null |
| `prompt` | string or null | the request prompt |
| `size` | string | `WxH` |
| `seconds` | string | the requested duration |
| `remixed_from_video_id` | string or null | always null |
| `error` | object or null | `VideoCreateError`: `code`, `message` |

## Create body

| Field | Type | Notes |
|---|---|---|
| `prompt` | string, required | 1 to 32000 characters |
| `input_reference` | object, string or file part, optional | recognised and refused until P8 |
| `model` | string, optional | must equal the configured model id |
| `seconds` | string, optional | h3 range below; default `"4"` |
| `size` | string, optional | h3 canvas rule below; default `512x512` |

Accepted request encodings:

- `application/json`: the five fields above as JSON. Unknown JSON keys are
  ignored, as on the reference API.
- `application/x-www-form-urlencoded`: the same field names as form keys.
  `+` decodes to a space; `%XX` decodes to a byte; a malformed escape is 400
  `invalid_urlencoded`. A repeated key keeps its last value.
- `multipart/form-data; boundary=...`: RFC 7578 parts, which is what the
  official curl example and `openai-python` send. The boundary may be quoted
  or unquoted and is at most 70 printable ASCII bytes. Framing is CRLF;
  preamble and epilogue are ignored; part bodies may contain CRLF and the
  boundary string as long as it is not on a delimiter line. Each part needs
  `Content-Disposition: form-data; name="..."`; `filename` marks a file part.
  A repeated field keeps its last value. An unknown field is 400
  `unknown_parameter`. Text values must be UTF-8. The body is bounded by the
  server's 64 MiB request limit and at most 64 parts.

`input_reference` is decoded from all three encodings into the reference's
object shape and refused with 400 `unsupported_parameter`, whether it arrives
as a JSON object, a form string or a multipart file part.

All three encodings produce the same internal request and run through one
validation and submission path.

Example, the official multipart form:

```sh
curl http://127.0.0.1:8090/v1/videos \
  -F "model=minimax-h3" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "seconds=8" \
  -F "size=768x768"
```

Example, JSON:

```sh
curl http://127.0.0.1:8090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A calico cat playing a piano on stage","seconds":"8","size":"768x768"}'
```

## List query

| Parameter | Values | Bridge behaviour |
|---|---|---|
| `limit` | integer 0..100 | default 20 |
| `order` | `asc`, `desc` | default `desc` |
| `after` | video id | cursor; unknown id is 400 `after` |

Response: `object` (`list`), `data` (array of Video), `first_id`, `last_id`,
`has_more`. `first_id`/`last_id` are null for an empty page.

## Delete

`DELETE /v1/videos/{id}` answers
`{"id": "...", "deleted": true, "object": "video.deleted"}` and removes the
MP4. Only completed or failed jobs may be deleted; a queued or in-progress job
answers 400 `video_not_ready`. A missing id answers 404 `video_not_found`.

## Content

`GET /v1/videos/{id}/content` serves the MP4 with `Content-Type: video/mp4` and
`Content-Length`. `HEAD` returns the same headers without a body. The optional
`variant` query accepts only `video`; `thumbnail` and `spritesheet` are
refused with 400 `variant`. Content is available only for completed jobs
(404 `video_not_ready` otherwise) and only while the job is retained.

## Models

`GET /v1/models` answers the reference's list shape:

```json
{"object":"list","data":[{"id":"minimax-h3","object":"model","created":1700000000,"owned_by":"ds4","shutdown_date":null}]}
```

## Errors

Every failure uses the reference's `ErrorResponse`:

```json
{"error":{"message":"...","type":"invalid_request_error","param":"size","code":"invalid_value"}}
```

All four keys are always present; `param` is null when no field applies.
`type` is `invalid_request_error`, `rate_limit_error` or
`service_unavailable_error`.

Validation mapping:

| Failure | Status | `type` | `param` | `code` |
|---|---|---|---|---|
| malformed JSON / body not an object | 400 | invalid_request_error | null | `invalid_json` |
| unsupported create `Content-Type` | 400 | invalid_request_error | null | `unsupported_content_type` |
| multipart without a boundary, or a boundary over 70 bytes | 400 | invalid_request_error | null | `invalid_content_type` |
| malformed or truncated multipart framing | 400 | invalid_request_error | null | `invalid_multipart` |
| malformed percent escape in a form body | 400 | invalid_request_error | null | `invalid_urlencoded` |
| unknown form field | 400 | invalid_request_error | null | `unknown_parameter` |
| form text value is not UTF-8 | 400 | invalid_request_error | null | `invalid_encoding` |
| create body over 64 MiB | 400 | invalid_request_error | null | `request_too_large` |
| prompt missing | 400 | invalid_request_error | `prompt` | `missing_required_parameter` |
| prompt empty | 400 | invalid_request_error | `prompt` | `string_below_min_length` |
| prompt over 32000 characters | 400 | invalid_request_error | `prompt` | `string_above_max_length` |
| wrong JSON type for a field | 400 | invalid_request_error | that field | `invalid_type` |
| unknown model | 400 | invalid_request_error | `model` | `model_not_found` |
| size not a legal h3 canvas | 400 | invalid_request_error | `size` | `invalid_value` |
| seconds outside the h3 frame range | 400 | invalid_request_error | `seconds` | `invalid_value` |
| `input_reference` supplied | 400 | invalid_request_error | `input_reference` | `unsupported_parameter` |
| bad `limit` | 400 | invalid_request_error | `limit` | `invalid_value` |
| bad `order` | 400 | invalid_request_error | `order` | `invalid_value` |
| unknown `after` cursor | 400 | invalid_request_error | `after` | `invalid_after` |
| unknown video id | 404 | invalid_request_error | `video_id` | `video_not_found` |
| content not ready / not retained | 404 | invalid_request_error | `video_id` | `video_not_ready` / `content_not_found` |
| unknown route | 404 | invalid_request_error | null | `unknown_url` |
| store at capacity | 429 | rate_limit_error | null | `queue_full` |
| renderer not configured | 503 | service_unavailable_error | null | `renderer_disabled` |

## h3 deviations from the OpenAI/Sora contract

Every deviation is named; none is silent.

1. **Model id.** The reference allows `sora-2`, `sora-2-pro` and their
   snapshots, defaulting to `sora-2`. This server accepts only the id chosen at
   startup with `--model` (default `minimax-h3`); any other id is
   `model_not_found`.
2. **Sizes.** All four official `VideoSize` values are illegal for h3:
   `720x1280` and `1280x720` fail the multiple-of-32 rule, and `1024x1792` and
   `1792x1024` exceed the pixel cap. Accepted sizes follow the reference
   engine (`/data/h3.c` at `eae5d54`): width and height multiples of 32, at
   least 32, product at most `768 * 1344` (`h3.c:492-497`,
   `h3_host.h:8`). The default is `512x512`, the reference's safest
   development canvas, not the official `720x1280`.
3. **Seconds.** The reference allows the strings `"4"`, `"8"`, `"12"`. h3
   accepts any positive duration whose frame count at 24 fps (`h3_host.h:9`),
   rounded as in `main.c:78-91`, lands in the released 5..362 request range
   (`h3.c:518-519`) and aligns to at least the 22-frame decoder chunk
   (`h3.c:867-870`). In practice that is roughly 0.23 s to 15.1 s. The
   default stays `"4"`. The response echoes the requested string; the actual
   clip is aligned upward to `5 + 17*n` frames, so `"4"` becomes 107 frames
   (4.458 s).
4. **Id scheme.** The reference's example uses the `video_` prefix; this
   server uses `video_N` with a monotonic counter. Deleted ids are not reused.
5. **Progress.** The bridge cannot see inside the renderer, so progress is
   coarse: 0 while queued or in progress, 100 on completion, 0 on failure.
6. **input_reference.** The field exists in the reference. The bridge refuses
   it with 400 `unsupported_parameter` instead of dropping it; reference
   images arrive with the port's own renderer at P8.
7. **Content variants.** Only `variant=video` (the default MP4) exists.
   `thumbnail` and `spritesheet` answer 400.
8. **List defaults.** The reference states no default for `limit` or `order`.
   The bridge uses `limit=20` and `order=desc`; both are assumptions, not
   verified API behaviour.
9. **Capacity.** `--max-jobs` bounds retained jobs and therefore render
   concurrency. A create at capacity answers 429 `queue_full`; delete a
   finished job to make room.
10. **Unserved methods.** Characters, edits, extensions and remixes are not
    served (404 `unknown_url`). `expires_at` and `remixed_from_video_id` are
    always null.
11. **Renderer failure codes.** Inside `error.code`: `render_failed`
    (nonzero exit), `render_timeout`, `render_output_missing` (exit 0 without
    a non-empty output file), `renderer_spawn_failed`.
12. **Form field scope.** Form bodies (`urlencoded` and multipart) accept only
    the five official field names; an unknown form field is 400
    `unknown_parameter`. JSON keeps the reference's tolerance for unknown
    keys. Chunked request bodies are refused (400 `unsupported_encoding`).

## Renderer contract and security boundary

The executable, its fixed arguments, the output directory and the working
directory are chosen at startup, never by a request. The argv uses the
reference engine's own CLI spellings, verified in `/data/h3.c/main.c` at
eae5d54 (`main.c:22-29` usage, `:292-296` option table, `:361-381` switch),
so the oracle binary runs without an adapter:

```sh
h3-server-rs \
  --renderer /home/leandro/h3-oracle/h3 \
  --renderer-arg -d --renderer-arg /mnt/h3-artifacts/MiniMax-H3 \
  --out-dir /srv/h3-videos
```

Each job runs, with the fixed args before the request args:

```
/home/leandro/h3-oracle/h3 -d /mnt/h3-artifacts/MiniMax-H3 \
  -p P --width W --height H --seconds S -o /srv/h3-videos/video_N.mp4
```

`-p` and `-o` are the short forms the reference documents; `--width`,
`--height` and `--seconds` are long options. The bridge passes `--seconds`
unchanged and never computes frames: the reference converts the duration at
24 fps itself (`main.c:78-91`). The P8 port keeps the same CLI flags
(port-plan 6.3), so the renderer can be swapped without changing this surface.

- No shell: every value is a separate argv entry. A prompt containing `;`,
  `$()` or `--width` cannot become a command or an option.
- The request never supplies a program, an argument position, a working
  directory, or a filesystem path. The output path is built from the
  monotonic job id inside the fixed output directory.
- stdin is null, stdout is discarded, stderr is captured (last 4096 bytes) and
  attached to a failed job.
- Each render is bounded by `--timeout-secs` (default 3600). On timeout the
  child is killed and the job fails with `render_timeout`.
- Exit 0 without a non-empty output file fails the job with
  `render_output_missing`.
- Without `--renderer` the server still starts; every create answers 503
  `renderer_disabled`, in the OpenAI error body.

## Pointing an OpenAI client at it

```
base_url = http://<host>:<port>/v1
api_key  = any non-empty string (the bridge does not authenticate)
```

`GET /v1/models` works with any OpenAI-compatible client. `videos.create` in
`openai-python` sends `multipart/form-data`, which this surface accepts, so the
SDK works unchanged against the base URL. The same job as curl:

```sh
curl http://127.0.0.1:8090/v1/videos \
  -F "prompt=A red cube rotates on a white background." \
  -F "size=512x512" \
  -F "seconds=4"
```

The equivalent JSON body is accepted too:

```sh
curl http://127.0.0.1:8090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red cube rotates on a white background.","size":"512x512","seconds":"4"}'
```

Poll `GET /v1/videos/{id}` until `status` is `completed` or `failed`, then
download `GET /v1/videos/{id}/content`.

## P8

The port's own renderer replaces this bridge at phase P8. The HTTP surface,
wire types, ids, validation and error shapes stay; only the `Renderer`
implementation changes, and `input_reference` can then be wired to h3's
first/last-frame and Ref2VA conditioning. The P8 CLI keeps the same flags
(port-plan 6.3), so the bridge argv does not change when the renderer is
swapped.
