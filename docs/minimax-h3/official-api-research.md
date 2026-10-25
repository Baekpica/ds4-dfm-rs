MiniMax-H3 official API: research and implementation report

Status: research only, 2026-10-25. No source, test, build or git change accompanies this document. Sources were read in the fixed order:
the official repository at /data/MiniMax-H3 (read-only, HEAD d21241f, clean), then the official documentation pages. Nothing was run
against the hosted API, so every live-behavior statement is a statement about the documented contract.

1. Source inventory

Repository (read-only):

- README.md:56-58 - the Online API section; global doc link is the v2-create page, CN link is the same path on platform.minimaxi.com.
- README.md:104 - H3-Context-IR is "a hosted preprocessing and orchestration system ... not included in this open-source release".
- README.md:170 - H3-Regenerate-2K is "not yet open-sourced"; its API exists to validate official results.
- README.md:294-320 - the Full 2K Workflow: SGLANG_DEPLOYMENT_URL, MINIMAX_API_BASE, TOKEN, the three doc links (:314-316), and Base64
  data URLs for local H3-Base output (:319).
- README.md:303,305 - MINIMAX_API_BASE is https://api.minimaxi.com (CN) or https://api.minimax.io (Global).
- scripts/readme/full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh (:6-23 create, :26-31 query, :34-35
  download) - primary t2va body, endpoint, headers and download flow; the 2k variant (:8-22, :29) differs only in resolution.
- scripts/readme/full-2k-i2va-reference-768p-... (:11-29) and -2k-... (:11-29) - the first-frame item (image_url, role first_frame) and
  ratio adaptive.
- scripts/readme/full-2k-ref2va-reference-768p-... (:11-36) - video_url/reference_video and audio_url/reference_audio, duration 5, ratio
  adaptive; the "api-2k" script (:11-36) is the same at resolution 2K.
- scripts/readme/full-2k-t2va-h3-context-ir.sh (:8, :11-21, :28, :33) - POST /v2/h3_context_ir, the query, and extraction of
  .task.content.prompt; the i2va and ref2va context-ir scripts (:8-35) add image and reference items.
- scripts/readme/full-2k-t2va-h3-regenerate-2k.sh (:13-28, :32, :42, :47-48) - POST /v2/video_regeneration with a video_url base_video
  item and resolution 2K; the i2va and ref2va variants (:13-42) add first_frame and reference items.
- scripts/readme/reproducible-768p-{t2va,fl2va,ref2va}-request.sh (:11-35, :12-28, :12-32) - the local SGLang contract (POST /v1/videos
  with task, prompt, conditions[].type/uri/role/frame_index, target.short_edge/aspect_ratio/duration_seconds, seed; GET /v1/videos/{id};
  GET /v1/videos/{id}/content). This is the reference's local serving contract, not MiniMax's hosted one.
- A repo-wide grep for MINIMAX_API_BASE, /v2/, api.minimax, webhook, callback, idempotency, rate limit, upload, file_id and asset found
  no other API use: the eleven scripts above are the only Open Platform calls. There is no upload, asset-id, webhook or idempotency example.

Documentation reached with web_fetch:

- platform.minimax.io/docs/api-reference/video-generation-v2-create, -query, -list, -delete, -h3-context-ir, -regeneration.
- platform.minimax.io/docs/api-reference/video/generation/api/v2-video-generation.json - the OpenAPI 3.1 document behind those pages; it
  carries the complete schemas, error envelopes and enums.
- platform.minimax.io/docs/guides/video-generation.md, guides/rate-limits.md, api-reference/errorcode.md, api-reference/api-overview.md,
  api-reference/video-generation-t2v.md (v1), docs/llms.txt.
- CN mirror after a cross-host redirect from platform.minimaxi.com: platform.minimax.cn/docs/api-reference/video-generation-v2-create,
  -query, -regeneration.

Failed: the first CN fetch (platform.minimaxi.com) returned a cross-host redirect to platform.minimax.cn, which was then fetched
successfully; the /docs/en/ prefixed create page returned an empty body. Every other page was reached.

2. The official wire contract

2.1 Hosts and paths

- Global base URL https://api.minimax.io; CN base URL https://api.minimax.cn (every CN cURL example). The README's CN value
  https://api.minimaxi.com (:303) still resolves but the CN docs use api.minimax.cn; treat the README value as stale.
- POST /v2/video_generation, POST /v2/h3_context_ir, POST /v2/video_regeneration.
- GET /v2/query/video_generation/{task_id}, GET /v2/query/video_generation.
- DELETE /v2/video_generation/{task_id}.
- A v1 exists (POST /v1/video_generation, video-generation-t2v.md) with a different body
  (model/prompt/prompt_optimizer/duration/resolution/callback_url), Hailuo models and a base_resp status_code envelope. The docs do not
  call v1 deprecated. H3 is served on v2 only.

2.2 Auth and headers

- Authorization: Bearer API_key, required on every route (OpenAPI bearerAuth; each page's Authorizations block).
- Content-Type: application/json, required on the three POST routes.
- No other required header, idempotency key or version header is documented.

2.3 POST /v2/video_generation request

- model - string, required - MiniMax-H3, MiniMax-H3-Highspeed, MiniMax-H3-Lite, MiniMax-H3-Max. MiniMax-H3 is 768P/2K and 4-15 s;
  Highspeed is 768P, 4-15 s; Lite is 480P/768P, 4-15 s; Max is 480P/768P, 5-15 s (VideoGenerationV2Req.model).
- content - object array, required - each item has a type (text, image_url, video_url, audio_url) and an optional role. Every request
  must contain one non-empty text item; otherwise the 400 message is "invalid params, content must include a non-empty text item (prompt is
  required) (2013)".
- content[].text - string - prompt, at most 7000 characters.
- content[].image_url.url - string, required when type=image_url - a public URL, mm_file://{file_id}, or
  data:image/<format>;base64,<payload> with a lowercase format. video_url and audio_url take the same three forms
  (data:video/mp4;base64,... and data:audio/<format>;base64,...).
- content[].role - string - first_frame, last_frame, reference_image, reference_video, reference_audio. A single image with no role
  defaults to first_frame; last_frame must be paired with first_frame; reference roles and frame roles are mutually exclusive.
- resolution - string, required - 480P, 768P, 2K. MiniMax-H3 allows 768P and 2K; Lite and Max default to 768P.
- duration - integer, required - 4 through 15. Max does not support 4.
- ratio - string, optional in the schema - adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, 9:16; default adaptive. t2va requires a concrete ratio
  (not adaptive). i2va is always treated as adaptive and other valid values are ignored. r2va defaults to adaptive; Lite does not support
  adaptive.
- extra - object, optional - only prompt_expansion_mode (disabled, balanced, quality; default balanced), documented for MiniMax-H3-Max
  only; undeclared fields are forbidden.
- callback_url - string, optional - see 2.7.
- aigc_watermark - boolean, default false - listed by CN for this route, absent from EN and from the OpenAPI schema. The language
  versions disagree (2.9).

Media limits (content description): body <= 64 MB; image <= 30 MB, 256..5760 px per edge, aspect 0.4..2.5, first frame <=1, last frame <=1,
reference images <=9; video <=50 MB, MP4/MOV, H.264/AVC or H.265/HEVC with AAC/MP3, <=3 clips, 2..15 s each and <=15 s total, 23.976..60
fps; audio WAV/MP3, <=15 MB, <=3 clips, 2..15 s each and <=15 s total; 12 files across all types (video-generation.md).

2.4 Create response

200, application/json: { "task_id": "<string>" } only. The task object is not returned.

2.5 GET /v2/query/video_generation/{task_id} response

200 with { "task": Task }. Task fields (VideoTask):

- id - string - task id.
- model - string - e.g. MiniMax-H3.
- status - string - queued (waiting in queue), running (in progress), succeeded, failed, cancelled.
- error - object, absent on success - {code, message}; the list example shows code "1026", message "video description contains sensitive
  content".
- created_at, updated_at - integer - unix seconds at creation and at the last status change.
- content - object, after success - {url} for a video task, {prompt} for an h3_context_ir task.
- content.url - string - time-limited download URL; query again for a new URL after it expires.
- resolution - string; duration - integer (output seconds).
- usage - object, only when succeeded - total_seconds, input_seconds, output_seconds, input_image_count, input_audio_seconds,
  total_tokens, prompt_tokens, completion_tokens. Video tasks report the second-based fields and convert tokens from usage; H3-Context-IR
  reports tokens. input_audio_seconds is omitted when there is no reference audio.
- ratio - string - output aspect ratio; may be "" when it does not apply (the regeneration example).
- task_type - string - generation, h3_context_ir, regeneration.
- modality - string - video or text.

2.6 Other operations

- POST /v2/h3_context_ir - model (MiniMax-H3 only), content (same item schema), duration (required, 4..15), ratio (same rules),
  callback_url. Returns a task_id; on success content.prompt is the enhanced prompt. "This endpoint only returns an enhanced video prompt.
  It does not create a video generation task."
- POST /v2/video_regeneration - exactly one of source_task_id or content. source_task_id mode: model, source_task_id, resolution (2K
  only), callback_url, aigc_watermark; the source task must be owned, succeeded and within the 7-day window, and the mode needs whitelist
  access. base_video mode: model, content, resolution (2K only), callback_url, aigc_watermark; content must reproduce the original inputs
  and contain exactly one type=video_url item with role=base_video. A base video needs audio, 24 fps, edges divisible by 32, area between
  768x768 and 768x1344, and 107..362 frames in increments of 17.
- GET /v2/query/video_generation - page_num (starts at 1), page_size (example 20, no maximum stated), filter.status, filter.task_ids,
  filter.model, filter.task_type. Response {items: Task[], total}, total counting only the last 7 days.
- DELETE /v2/video_generation/{task_id} - status driven: queued becomes cancelled (no charge), succeeded or failed is deleted, running
  and cancelled are refused with an error. Response {task_id, action, status} with action/status in {cancelled, deleted}.

2.7 Lifecycle and callbacks

- Create returns a task_id; the client polls the query route; the guide recommends a 10 s interval; on succeeded the video is at
  content.url and is downloaded. Terminal failure states are failed and cancelled.
- Tasks are queryable only inside [T-7d, T); outside it the response is "invalid task_id". The HTTP status and code for that case are not
  stated.
- callback_url: the server first POSTs a verification body with challenge, which must be echoed within 3 s; then it POSTs an update on
  every status change. The push body matches the query response. Callback statuses are queued, running, succeeded, failed, cancelled.
- Rate limits: Video Generation V2 with MiniMax-H3 is 300 RPM and 30 maximum inflight tasks.

2.8 Error envelope and codes

Every documented failure is HTTP 400/401/402/422/429/500 with:

  { "type": "error",
    "error": { "type": "<kind>", "message": "<detail> (<code>)", "http_code": "<status string>" },
    "request_id": "<trace id>" }

Documented kinds: bad_request_error 400, authorized_error 401, insufficient_balance_error 402, unprocessable_entity_error 422,
rate_limit_error 429, server_error 500; the schema also names overloaded_error 529. The message ends with the internal code in parentheses.
Documented examples: 2013 invalid params, 1004 login fail, 1008 insufficient balance, 1026 sensitive content, 1002 rate limit, 1000
internal error. The error table adds 1001 timeout, 1024 internal error, 1027 output sensitive, 1045 task concurrency, 2022 model is not
exist, 2064 high load, 2066 processing timeout, 2069 model no longer available.

2.9 Hosted-only capabilities

- H3-Context-IR is not open-sourced (README.md:104; the page repeats it). It is a multi-stage pipeline over multiple hosted models.
- H3-Regenerate-2K is not open-sourced (README.md:170) and is the only documented 2K producer.
- MiniMax-H3-Highspeed, -Lite and -Max are hosted variants; Max is jointly released with fal.ai and post-trained there
  (video-generation.md).
- Moderation (422/1026/1027), balance (402/1008), account ownership, whitelist access, CDN hosting and URL expiry, and the 300 RPM / 30
  inflight quota are service behavior, not model behavior.
- EN and CN disagree on aigc_watermark for create (CN lists it, EN does not). Which is current is unknown.

3. The port's current surface

Read from crates/ds4-server/src/h3_video.rs, crates/ds4-server/src/bin/h3-server-rs.rs, crates/ds4-core/src/minimax_h3/video_api.rs and
docs/minimax-h3/openai-videos.md.

- Routes (h3_video.rs:601-621; openai-videos.md:24-38): POST and GET /v1/videos, GET and DELETE /v1/videos/{id}, GET and HEAD
  /v1/videos/{id}/content, GET /v1/models.
- Create body (video_api.rs:275-281, :337-401): prompt (1..32000 chars), input_reference (file_id or image_url data URI), model, seconds
  (decimal string), size (WxH or preset). JSON, urlencoded and multipart bodies decode to one request (h3_video.rs:675-710).
- Validation (video_api.rs:421-517): model must equal the configured id; size must be a legal h3 canvas or preset; seconds must align
  into 22..362 frames at 24 fps.
- Job object (video_api.rs:687-700): the thirteen OpenAI fields id, object, model, status, progress, created_at, completed_at,
  expires_at, prompt, size, seconds, remixed_from_video_id, error.
- Statuses (video_api.rs:166-182): queued, in_progress, completed, failed; the store moves to in_progress before the create response is
  written (h3_video.rs:429-431).
- Store (video_api.rs:790-1001): in memory, bounded by --max-jobs, ids video_N, delete for completed or failed only, list by
  limit/order/after.
- Errors (video_api.rs:102-162): OpenAI envelope {error:{message,type,param,code}} with types invalid_request_error, rate_limit_error,
  service_unavailable_error.
- Result delivery: GET /v1/videos/{id}/content streams the MP4 (h3_video.rs:519-557); there is no URL field in the job object.
- Backends (h3-server-rs.rs:17-24, :183-240): --model-dir runs the engine in process (h3_engine.rs:16-79); --renderer PROGRAM spawns the
  reference CLI with -p, --width, --height, --seconds, -o (h3_video.rs:160-240); with neither, create answers 503 renderer_disabled.
- Auth: none; any non-empty key is accepted (openai-videos.md:296-299).

Concrete differences from the official contract:

1. Paths differ completely; no /v2 route, no list filters, no status-driven delete.
2. Body and enums: a flat prompt/size/seconds document vs a content array plus model, resolution (768P/2K tier),
   duration (integer 4..15) and ratio (aspect enum); model ids are minimax-h3 only vs four MiniMax-H3 variants.
3. Create returns the full Job object with status queued vs {task_id}; statuses are in_progress/completed vs
   running/succeeded, and cancelled is missing.
4. Result delivery: a local byte endpoint vs task.content.url.
5. Metadata: no updated_at, resolution, task_type, modality or usage.
6. Errors: the OpenAI envelope and codes vs the type/error/http_code/request_id envelope.
7. Auth: none vs the required Bearer key.
8. Delete: completed/failed only vs the official cancel-or-delete state machine.

4. Implementation report

4.1 Shape

Add the official surface as new h3-only code; leave the OpenAI types untouched.

- crates/ds4-core/src/minimax_h3/task_api.rs - official wire types, validation, task store, serializers.
- crates/ds4-server/src/h3_official.rs - the /v2 route table, handlers, auth and the create-to-RenderSpec mapping.
- crates/ds4-server/src/bin/h3-server-rs.rs - new flags; both surfaces on one listener.

Reuse the Renderer trait, CommandRenderer, EngineRenderer and RenderSpec. RenderSpec gains one additive field, last_frame:
Option<AnchorImage>, because the engine already takes it (engine.rs:57-79, :430-431) and the official first-and-last-frame mode needs it;
the current server passes None (h3_engine.rs:66-69).

4.2 Route table

- POST /v2/video_generation - validate, submit, return {task_id}.
- GET /v2/query/video_generation/{task_id} - the Task object.
- GET /v2/query/video_generation - page_num/page_size/filter.* list.
- DELETE /v2/video_generation/{task_id} - cancel queued or delete finished.
- GET /v2/video_generation/{task_id}/content - the local stand-in for the object content.url names. This path is not in the official
  contract: the contract names only an opaque URL, never a download route. Serving the bytes under the port's own base URL is a named
  deviation.
- POST /v2/h3_context_ir and POST /v2/video_regeneration - refuse (hosted-only; 2K unavailable).

4.3 Request structs and field mapping

CreateTaskRequest:
- model: String. Accept MiniMax-H3 only; the other three are 400 with code 2022 "model is not exist", because the open
  checkpoint is the single H3-Base artifact (README.md:184-189; video-generation.md).
- content: Vec<ContentItem>, an enum on type: Text{text}, ImageUrl{image_url, role}, VideoUrl{video_url, role},
  AudioUrl{audio_url, role}. Enforce exactly one non-empty text item (the docs say one; a second item's fate is unstated, so
  refuse 2013), text <= 7000 characters, the first_frame default for a lone image, last_frame pairing, and role exclusivity.
- resolution: {480P, 768P, 2K}. Accept 768P; refuse 2K (2013, naming H3-Regenerate-2K as hosted) and 480P (a Lite/Max tier).
- duration: 4..=15. frames = duration * 24, aligned by align_frames (video_api.rs:84-98). Every value in 4..15 lands in
  107..362, matching the official 768P output rule quoted in the regeneration base_video spec (increments of 17).
- ratio: enum. t2va requires a concrete ratio; i2va forces adaptive; r2va is refused. A concrete ratio resolves through
  presets::tier_canvas (presets.rs:91-112), which reproduces the platform's own 768P artifact 1344x768 for 16:9
  (presets.rs:5-20). Use tier_canvas, not the P11 delivery presets (presets.rs:198-252): those upscale 16:9 to 1920x1080,
  a port delivery choice, not the platform output.
- extra: refused 2013 (prompt_expansion_mode is Max-only, and Max is refused).
- callback_url: refused 2013. A callback means POSTing to a caller-supplied URL, which the repo forbids ("no network fetch
  reachable from a request body", AGENTS.md).
- aigc_watermark: accept false or absent; refuse true 2013 (no watermark path). Record the EN/CN discrepancy.

Media handling:
- Only data: URIs are accepted. A public URL needs an outbound fetch from a request body, which is forbidden and absent from
  the engine (media.rs is path-based); mm_file:// needs a platform asset store this host lacks. Both are refused 2013.
- The data URI is decoded by the existing base64/extension checks (video_api.rs:523-590) into AnchorImage: png, jpeg, webp.
  Official HEIC/HEIF are refused 2013 and named.
- Reference video and audio URLs are unreachable for the same reason; reference-to-video is refused independently (4.6).

4.4 Response structs

CreateTaskResponse {task_id}. Task {id, model, status, error, created_at, updated_at, content, resolution, duration, usage,
ratio, task_type, modality}. TaskContent is Video{url} or Prompt{prompt}. TaskUsage, locally:
- total_seconds = output_seconds + input_seconds; output_seconds = the requested duration integer (the official example
  reports 5 for a 5 s request); input_seconds = 0.
- input_image_count = number of frame images supplied; input_audio_seconds omitted (no reference audio can be accepted).
- total_tokens, prompt_tokens and completion_tokens omitted: the platform converts them from metered usage by an unstated
  rule and the local engine has no metering. Zero would be a false statement.
- ratio = the resolved concrete ratio (the docs say the actual ratio is read from this field for adaptive; the i2va example
  does not pin whether the value is adaptive or concrete, so a concrete value is a recorded choice).
- task_type = "generation", modality = "video".

4.5 Task state machine

- create -> queued.
- the render worker acquires the single GPU slot -> running, updated_at set. This differs from /v1, which sets in_progress at create
  time; the official queued state must be real while a job waits behind another render.
- renderer returns a non-empty file -> succeeded, content.url and usage set, updated_at set.
- renderer fails, times out, writes nothing or cannot spawn -> failed with error {code, message}.
- DELETE in queued -> cancelled; the worker checks the flag before starting and skips the render.
- DELETE in succeeded or failed -> record and file deleted; response {task_id, action:"deleted", status:"deleted"}.
- DELETE in running or cancelled -> an error; the docs give no status or code, so use 400/2013 and record the gap.
- Query and list see only [T-7d, T); older tasks are purged with their files (the window is documented; deleting the file at the same
  boundary is a local retention choice).
- Query of an unknown or expired id: the docs say "invalid task_id" with no status; use 400/2013 and record the gap.

4.6 Refusals

- POST /v2/h3_context_ir and POST /v2/video_regeneration: 501 with the official envelope, a message naming the hosted-only service, and
  code 2013. Alternative: 400/2013, inside the documented status set but calling a valid request invalid. Recommendation: 501, documented
  as a deviation.
- resolution 2K: 400/2013. Model MiniMax-H3-Highspeed/-Lite/-Max: 400/2022.
- reference_image, reference_video, reference_audio: 400/2013 until P14 lands. The port refuses Ref2VA segments by name
  (port-plan.md:801-805); P14.2-P14.6 are open (post-port-phases.md:1089-1130) and only P14.1, the AudioVAE encoder, is done.
- callback_url, extra, aigc_watermark=true, public URLs, mm_file://, HEIC/HEIF, a second text item.
- Queue bound: 429 rate_limit_error/1045 (task concurrency), the closest documented code; the platform's 30 inflight is a quota, not the
  same limit.
- No renderer configured: every create fails; the official set has no 503, so use 500 server_error/1024 and name it.

4.7 Auth and result URL

Add --api-key (or MINIMAX_API_KEY): when set, every /v2 route requires Authorization: Bearer <key> and answers 401 authorized_error/1004
otherwise; when unset the route is open, a named deviation. One key owns the whole store; do not invent per-task ownership.

Add --public-base-url; content.url becomes <base>/v2/video_generation/<task_id>/content, default http://<bind>. It is stable while the task
is retained. The official URL is time-limited; the local analogue is the 7-day retention, and the docs must say the URL does not rotate.
This is the largest unavoidable deviation: there is no CDN.

4.8 Flags reused

--model-dir, --steps, --reuse, --seed, --renderer, --renderer-arg, --max-jobs, --timeout-secs, --out-dir, --bind keep their meaning. The
official surface adds --api-key and --public-base-url. The external renderer keeps the reference CLI spelling, and the official surface
never hands it a delivered-frame size the h3 canvas refuses (api-usage.md:88-95 records that the current 16:9 preset does exactly that).

4.9 Compatibility decision

Keep /v1 beside /v2 for one release, with /v2 documented as primary, then delete /v1.

Tradeoffs: keeping both costs a second store, envelope and route table and doubles the HTTP tests, but preserves openai-python clients and
the Spark receipt in openai-videos.md. Replacing at once removes a surface whose upstream reference the repo records as shut down on
2026-09-24 (openai-videos.md:21-24, a repo statement not verified live here), and whose body is not MiniMax's product. Recommendation: add
/v2, mark /v1 deprecated in its doc and banner, keep it one release, then remove it while keeping the Renderer trait and the engine
unchanged.

5. Phased units

Each unit is one commit with its own evidence and tests.

U1 - Official wire types and store (no HTTP). task_api.rs: parsing, content validation, Task object, status enum, envelope, 7-day window,
list filters, delete state machine. Tests: one per field and refusal; the exact official key set; transitions; window boundary. Evidence:
cargo test -p ds4-core.

U2 - Official routes. h3_official.rs plus h3-server-rs wiring: create, query, list, delete, content, auth, public base URL. Tests: a
FnRenderer drives create/poll/download/delete through the handler, mirroring h3_video.rs:1240-1870. Evidence: cargo test -p ds4-server.

U3 - Create-to-render mapping. resolution/ratio/duration to VideoSize and frames; tier_canvas for a concrete ratio; adaptive from the
anchor's probed size; data URI to AnchorImage; last_frame wiring and the additive RenderSpec field; usage and echoed metadata. Tests: each
ratio to canvas, each duration to frames, adaptive, last-frame argv, usage arithmetic; h3_video tests unchanged. Evidence: the same test
run.

U4 - Refusals. Hosted-only routes, 2K, the three other model ids, reference roles, callback_url, extra, aigc_watermark, URL forms, HEIC,
second text, renderer disabled, queue full. Tests: one per case asserting status, error.type, http_code and code. Evidence: cargo test plus
a curl transcript.

U5 - Operator documentation. New docs/minimax-h3/official-api.md (routes, fields, state machine, deviations); updates to api-usage.md and
the doc index; a deprecation note in openai-videos.md. Evidence: the diff and a local create/poll/download transcript.

U6 - Spark gate. One real 768P t2va render through /v2 (text plus 16:9), poll to succeeded, download content.url, ffprobe it, compare
geometry and frames against assets/h3_direct_768p.mp4 (1344x768); one i2va render with a data-URI first frame. Evidence in official-api.md
with commit, model path, clocks and ffprobe output.

U7 - Retire /v1 (operator decision). Delete the /v1 routes, OpenAI wire types and their tests; keep the Renderer trait, engine and worker.
Evidence: the full test suite and doc updates.

6. Open questions

1. The HTTP status and code for querying an expired or unknown task ("invalid task_id") and for DELETE on a running or
   cancelled task ("returns an error"); the docs state neither.
2. Whether a second text item is accepted, ignored or refused; the docs say "one", not "exactly one".
3. Whether aigc_watermark is current on create; CN lists it, EN and OpenAPI do not.
4. Whether extra.prompt_expansion_mode is ignored or refused for MiniMax-H3.
5. Whether ratio is adaptive or concrete in an i2va query response.
6. The exact lifetime of content.url; no duration is stated.
7. The default and maximum page_size.
8. Whether output_seconds is the requested integer or the actual aligned clip length.
9. Which code a hosted deployment returns for an unsupported model variant: 2022, 2069 or 2013.
10. Whether the 107..362-frame, increment-17 rule is the generation output rule or only a regeneration input constraint; it
    is stated only on the regeneration page.
11. The query-string serialization of filter.task_ids.

7. Risk notes

Each item is a place where the official contract depends on the hosted service or on unported model code. Each is a named deviation, never
a silent substitution.

1. URL expiry and CDN hosting. The official content.url is a time-limited CDN object; the local URL is a port route with no
   rotation, so a client that stores and later fetches it behaves differently. Prevention: 7-day retention, documented no-rotation.
2. Moderation. The hosted service answers 422/1026 or fails tasks with 1026/1027; the local host does none, so a local deployment
   emits content the hosted service would refuse. The operator must accept that explicitly.
3. Quota, billing and usage. 402/1008 and 300 RPM / 30 inflight do not exist locally (the bound is --max-jobs), and token usage
   cannot be reproduced, so those fields are omitted while the platform always reports them.
4. H3-Context-IR is hosted-only (README.md:104); clients that rely on prompt expansion must call the hosted service.
5. H3-Regenerate-2K is hosted-only (README.md:170) and the only documented 2K producer; a 2K request must be refused, never downgraded.
6. Ref2VA is refused (port-plan.md:801-805) and P14.2-P14.6 are open (post-port-phases.md:1089-1130); reference-to-video is unavailable.
7. Public-URL media, mm_file:// and file uploads need an outbound fetch or an asset store, both forbidden; refused. Only data URIs
   work, and the 64 MB cap makes video references impractical even after P14.
8. Callbacks are refused for the same outbound-request reason; a callback would be a new unit with its own security review.
9. Local task ids are chosen, not platform ids; the 7-day window is retention, and files are deleted at the same boundary.
10. Deleting a running task is refused, which is faithful, but the exact error is unknown; the CN/EN aigc_watermark disagreement
    makes any choice a guess, with the OpenAPI document as the tie-breaker here.

Bottom line

The official MiniMax-H3 contract is an asynchronous, Bearer-authenticated v2 task API: POST /v2/video_generation, /v2/h3_context_ir or
/v2/video_regeneration with a multimodal content array (text, image, video, audio) plus model, resolution, duration and ratio; a shared
query, list and cancel/delete surface over task objects with queued/running/succeeded/failed/cancelled states; a task_id only on create; a
time-limited content.url on success; and an error envelope with numeric codes. The port serves a different, deprecated-upstream OpenAI
Videos surface (flat prompt/size/seconds, /v1/videos, local byte download, no auth). The port must add the v2 route table, content-array
validation, the task object and state machine, the error envelope and Bearer auth, reuse the existing engine and renderer flags, and refuse
by name everything the open checkpoint cannot do: H3-Context-IR, H3-Regenerate-2K and 2K output, the three hosted model variants,
reference-to-video until P14, public-URL media, callbacks and moderation. The single decision the operator has to make is whether that
honest subset counts as official support: if yes, the v2 surface ships now and /v1 is kept for one release as a deprecated alias; if no,
the v2 surface waits for Ref2VA and 2K and the port keeps serving only /v1.
