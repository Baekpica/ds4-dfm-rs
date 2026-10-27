# The local v1 video contract (MiniMax-H3)

This host serves the request shape used by the
[official local client scripts](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/scripts/readme),
pinned at `d21241f0a4b3acbb34c97dae47fa417b7065e438`:

- `full-2k-t2va-h3-base.sh:24-31`, `full-2k-i2va-h3-base.sh:24-38`,
  `full-2k-ref2va-h3-base.sh:24-42`: `POST $SGLANG_DEPLOYMENT_URL/v1/videos`,
  then `GET /v1/videos/{id}`, then `GET /v1/videos/{id}/content`;
- `reproducible-768p-fl2va-request.sh:10-44`: the same three calls against
  `http://localhost:30010`, with the full FL2VA body and one keyframe.

The tables below describe the implemented local limits and refusals.

## Routes

| Method | Path | Answer |
| --- | --- | --- |
| POST | `/v1/videos` | `{"id": "<id>"}` |
| GET | `/v1/videos/{id}` | the task record, whose `status` the client polls |
| GET | `/v1/videos/{id}/content` | the MP4 bytes |

The content route also accepts `HEAD`, returning its headers without a body.

The scripts read `.id` from the create answer and `.status` from the query
answer, and download once the status is `completed` (`full-2k-t2va-h3-base.sh:24`,
`:30`, `:31`). The port's state machine is `queued`, `in_progress`,
`completed`, `failed`; `completed` is the one value the official client
depends on.

### The task record

The query route answers one record, in this order
(`crates/ds4-core/src/minimax_h3/video_api.rs:223-258`):

| Field | Meaning |
| --- | --- |
| `id` | the create answer's id, an opaque identifier unique to the server instance and job |
| `status` | `queued`, `in_progress`, `completed`, `failed` |
| `queue_position` | 1 for the job the engine takes next, counted over the queued jobs; null while a job holds the single render slot and for a job that has run |
| `size` | the resolved canvas, for example `1344x768` |
| `seconds` | the requested duration |
| `frames` | the aligned frame count the engine renders |
| `seed` | the seed the body sent |
| `created_at` | submission time, seconds since the epoch |
| `completed_at` | null until the job completed or failed |
| `error` | null, or `{code, message}` on failure |

The prompt is not echoed: a poll answer would repeat up to 32000 characters on
every request. The content route answers `video/mp4` for a completed job only.

### Queue state

The store admits one render at a time. `--max-jobs` bounds retained jobs,
including finished jobs. There is no delete route or automatic expiry; a
full store refuses new creates until the server restarts. Output files are
preserved across restarts, but the in-memory task records are not restored.

## The create body

Exactly five fields. An unknown field is refused, not ignored: a typo in a
local contract should fail loudly.

| Field | Values | Where it comes from |
| --- | --- | --- |
| `task` | `t2va`, `fl2va`, `ref2va` | the three base scripts |
| `prompt` | non-empty string, at most 32000 characters | the scripts; the cap is the server's, `MAX_PROMPT_CHARS` |
| `conditions` | array of condition objects: at most 2 keyframes, at most 12 references | `[]` for t2va; one `keyframe` for fl2va and i2va; image/audio/video references for ref2va (`h3.c:588`) |
| `target.short_edge` | any multiple of 32, 32 through 992 | the scripts' `short_edge`; 768 is the tier the model card documents, and the canvas is built from the value (see "The canvas") |
| `target.aspect_ratio` | `auto`, `16:9`, `9:16`, `4:3`, `3:4`, `1:1` | the scripts use `16:9` and `auto`; the rest are the platform's named ratios, each resolving to a canvas of the requested short edge (see "The canvas") |
| `target.duration_seconds` | 4 through 15 | the scripts send 5, 8 and 10; the range is the platform document's enum, and every value aligns inside the engine's 22..362 frames |
| `seed` | non-negative integer | every script sends `seed` |

### A condition

| Field | Values |
| --- | --- |
| `type` | `image`, `audio`, `video` |
| `role` | `keyframe`, `first_frame` (fl2va anchors); `reference`, `reference_video`, `reference_audio` (ref2va). `base_video` is refused: it is the platform's source-video edit role |
| `frame_index` | `0` (first frame) or `1` (last frame); required for a keyframe, one per index, absent on a reference |
| `uri` | a `data:<media-type>;base64,<payload>` URI. The declared type must match the condition: `image/*` (PNG, JPEG, WebP, at most 16 MiB decoded), `video/*`, `audio/*` |

### References (ref2va)

The built-in decoder accepts PNG/JPEG/WebP images, MP4/MOV and Matroska/WebM
video, and WAV/MP3/AAC/M4A/FLAC/Ogg/AIFF audio. Playlists such as HLS, DASH
and concat are refused before their references are opened. Inputs use the
local file protocol only. An operator-supplied external renderer must enforce
its own media policy.

`task: ref2va` conditions with ordered references, at most 12 (`h3.c:588`),
each label following request order (`h3_multimodal.c:320-332`):

| Request | Reaches the engine as | Label |
| --- | --- | --- |
| `type: image`, `role: reference` | `--ref-image` | `<Picture n>: ` |
| `type: video`, `role: reference` or `reference_video` | `--ref-video` (its own soundtrack included) | `<Audio n>: `, `<Video n>: `, then `<t.t seconds>` per sampled block |
| `type: audio`, `role: reference` or `reference_audio` | `--ref-audio` | `<Audio n>: ` |

At least one image or video reference is required: ordered audio alone
conditions nothing (`h3_multimodal.c:283-285`). Keyframes and references never
appear in the same request (`h3.c:597-600`), and a video's soundtrack is cut to
the requested clip length, so it needs at least two seconds
(`h3.c:1144-1151`). Soundtracks share one 15-second budget across the whole
set (`h3.c:1171-1176`); the refusal names the budget. Video and audio payloads
are bounded by the request body's 64 MiB cap, which base64 inflates by 4/3.

## The canvas

`target.short_edge` is the short edge the caller asks for; `aspect_ratio` names
an aspect; the long edge follows that aspect on the released 32-pixel grid and
the released `768*1344` area clamps it (`presets::ratio_canvas`, verified
against the platform's own 768P artifact). At the 768 tier the canvases are:

| Ratio | Canvas at short_edge 768 |
| --- | --- |
| 16:9 | 1344x768 |
| 9:16 | 768x1344 |
| 4:3 | 1024x768 |
| 3:4 | 768x1024 |
| 1:1 | 768x768 |
| auto | the keyframe's own aspect, or the first image/video reference's (`presets::canvas_for`) |

A different short edge scales the same rule: 576 with 16:9 is 1024x576,
because 576 is on the grid and 1024x576 is inside the released area. The
record's `size` is the canvas the host resolved, which is what the job
rendered; a long edge that had to be clamped is reported there, not hidden.

The short edge is never clamped. It is the canvas's short side, so the only
canvases a request can produce have an area of at least its square, and the
released `768*1344` area caps it at 992: 1024x1024 is over the area, and 1024's
own long edge would clamp to 992, inverting the shape. A request above the
ceiling is refused rather than served as a different shape, which is how the
reference behaves too, since it takes width and height and refuses the area
outright (`h3_valid_params`, `h3.c:496`).

The file is the canvas: this contract has no separate delivery frame, so
there is no anamorphic scale anywhere in it.

`21:9` is refused at short_edge 768. 768P wants 1792x768 for it, which the
released 768x1344 area refuses; the widest canvas is 1344x768, a 1.333x
stretch, beyond the 1.25 bound (`presets::MAX_DISTORTION`). The refusal prints
those numbers and suggests 16:9, and lowering `short_edge` serves the ratio:
576 with 21:9 is 1344x576, inside the area and inside the bound.

## Refusals

| Case | Code |
| --- | --- |
| body is not JSON / not an object | `invalid_json` |
| a field the contract does not define | `unknown_field` |
| missing `task`, `prompt`, `target` or `seed` | `invalid_task`, `invalid_prompt`, `invalid_target`, `invalid_seed` |
| `task` not one of the three | `invalid_task` |
| ref2va with no image or video reference, or only audio | `invalid_condition` |
| ref2va with a keyframe, or a reference carrying `frame_index` | `invalid_condition` |
| more than 12 references | `invalid_condition` |
| `role: base_video` | `unsupported_condition` |
| a reference `uri` whose type disagrees with `type` | `invalid_uri` |
| a reference role on fl2va | `unsupported_condition` |
| a keyframe without `frame_index`, a bad index, a duplicate index | `invalid_condition` |
| t2va with a keyframe, fl2va without one | `invalid_condition` |
| an https or plain-path `uri` | `unsupported_uri` / `invalid_uri` |
| `short_edge` off the 32 grid, below 32, or above the 992 ceiling | `unsupported_short_edge` |
| `aspect_ratio` 21:9 or unknown | `invalid_aspect_ratio` |
| `duration_seconds` outside 4..15 | `invalid_duration` |

## Deviations from the official scripts

Each is deliberate and named:

1. **No HTTP auth.** The scripts call an SGLang deployment with no auth, so
   this host matches them: no `Authorization` header is read and no key
   exists. The listener's address is the operator's (`--bind`, default
   `127.0.0.1:8090`); a bind beyond loopback publishes an unauthenticated
   renderer to that network, which is the operator's decision, not a
   property of the contract. See "Deployment limits" below.
2. **A condition `uri` must be a data URI.** The scripts pass `https://` CDN
   URLs. This host never fetches a URL from a request body and never reads a
   caller-named filesystem path, so the same image travels base64-encoded.
3. **`auto` uses the first visual reference.** A video's pixel size is
   probed with FFprobe at create time from a temporary copy; an image's
   dimensions are read from its header.
4. **`short_edge` is the caller's, on the released grid.** 768 is the tier the
   model card documents and what the scripts send. The reference accepts any
   canvas that is a multiple of 32 with a product under `768*1344` (`h3.c`
   README, "Pick resolution and duration"), and its own scale points are 256,
   320 and 384, so a smaller `short_edge` is served as an execution preview
   rather than refused. 480P and 2K belong to the other variants and to
   H3-Regenerate-2K, which this checkpoint does not provide: neither is a
   short edge here, they are other released artifacts. The same area that
   bounds the long edge bounds the short one, so 992 is the ceiling and a
   larger request is refused instead of being served inverted ("The canvas").
5. **21:9 is refused at the tier**, with the numbers in the message; a
   `short_edge` the released area can hold serves it (see "The canvas").

## Deployment limits

The default listener is loopback. This server has no authentication; any
client that can reach the port can submit GPU work. The operator controls
network access and the output directory. Each accepted render writes an MP4;
the server does not prune output files. Steps and reuse are fixed at startup.

## Not served

The hosted platform's `/v2/video_generation`, `/v2/h3_context_ir`,
`/v2/video_regeneration` and their query routes. No local script calls them
(every `/v2` caller in the official tree uses `$MINIMAX_API_BASE` with a
Bearer token), and the two distinctive ones are hosted-only features. They
have no route here at all, rather than a stub that answers 501.

The OpenAI Videos body and `/v1/models` are not part of this contract either:
the path and the three routes are the official local ones, and the body is
the scripts' body.
