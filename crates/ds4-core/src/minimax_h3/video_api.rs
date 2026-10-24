//! OpenAI Videos API wire types and in-memory job state for the h3 family.
//!
//! Wire reference (verified 2026-10-24):
//!
//! - <https://developers.openai.com/api/reference/resources/videos>
//! - <https://developers.openai.com/api/reference/resources/videos/methods/create>
//! - <https://developers.openai.com/api/reference/resources/videos/methods/list>
//! - <https://developers.openai.com/api/reference/resources/videos/methods/retrieve>
//! - <https://developers.openai.com/api/reference/resources/videos/methods/delete>
//! - <https://developers.openai.com/api/reference/resources/videos/methods/download_content>
//! - <https://developers.openai.com/api/reference/resources/models/methods/list>
//!
//! The reference marks the Videos surface deprecated and shut down on
//! 2026-09-24; it is retained as historical documentation, so its field list
//! is still the carriage we serve. Every field below exists in that reference.
//! h3-only deviations (model id, accepted sizes, seconds range, id scheme) are
//! named in `docs/minimax-h3/openai-videos.md`; none is invented silently.
//!
//! `Video` serializes the thirteen fields of the reference's `VideoResource`:
//! `id`, `object`, `model`, `status`, `progress`, `created_at`,
//! `completed_at`, `expires_at`, `prompt`, `size`, `seconds`,
//! `remixed_from_video_id`, `error`.
//!
//! No HTTP and no GPU live here: the server layer owns the renderer boundary.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

/// Canonical wire reference used to verify every field in this module.
pub const WIRE_REFERENCE: &str = "https://developers.openai.com/api/reference/resources/videos";

pub const VIDEO_OBJECT: &str = "video";
pub const LIST_OBJECT: &str = "list";
pub const DELETE_OBJECT: &str = "video.deleted";

/// `CreateVideoJsonBody.prompt.maxLength` in the reference OpenAPI document.
pub const MAX_PROMPT_CHARS: usize = 32_000;

/// `ListVideos.limit` is `minimum: 0`, `maximum: 100`; the reference states
/// no default. The bridge uses the common OpenAI page size and names it here.
pub const MIN_LIST_LIMIT: usize = 0;
pub const MAX_LIST_LIMIT: usize = 100;
pub const DEFAULT_LIST_LIMIT: usize = 20;

/// Reference frame cadence and limits, copied from `/data/h3.c` at eae5d54:
///
/// - `h3_host.h:9` `H3_FPS 24`.
/// - `h3_host.c:17-23` frames align upward to `5 + 17*n`.
/// - `h3.c:492-497` width/height multiples of 32, at least 32, product
///   at most `H3_MAX_PIXELS`.
/// - `h3.c:518-519` requested frames must be in 5..362 after alignment.
/// - `h3.c:867-870` generation needs at least one 22-frame decoder chunk.
pub const H3_FPS: u32 = 24;
pub const H3_ALIGN_BASE: u32 = 5;
pub const H3_ALIGN_STEP: u32 = 17;
pub const H3_MIN_FRAMES: u32 = 5;
pub const H3_MAX_FRAMES: u32 = 362;
pub const H3_MIN_GEN_FRAMES: u32 = 22;
pub const H3_CANVAS_MULTIPLE: u32 = 32;
pub const H3_MAX_PIXELS: u64 = 768 * 1344;

/// h3-safe default. The OpenAI default `720x1280` is not a legal h3 canvas
/// (720 is not a multiple of 32), so the bridge defaults to the reference's
/// safest development size, `/data/h3.c/README.md` "Pick resolution".
pub const DEFAULT_SIZE: &str = "512x512";

/// The OpenAI default is `4` seconds and 4 is legal h3 (aligned to 107
/// frames), so the default is unchanged.
pub const DEFAULT_SECONDS: &str = "4";

/// Jobs name their model id; the operator picks the real one at startup.
pub const DEFAULT_MODEL: &str = "minimax-h3";

/// Align a requested frame count upward the way `h3_host.c:17-23` does.
pub fn align_frames(requested: u32) -> u32 {
    let value = requested.max(H3_ALIGN_BASE);
    let remainder = (value - H3_ALIGN_BASE) % H3_ALIGN_STEP;
    if remainder == 0 {
        return value;
    }
    value + H3_ALIGN_STEP - remainder
}

/// Convert a duration to the frame count `main.c:78-91` requests at 24 fps.
/// `f64::round` is half-away-from-zero, matching C `llround`.
pub fn frames_from_seconds(seconds: f64) -> u32 {
    (seconds * H3_FPS as f64).round() as u32
}

/// OpenAI-shaped API error: `{"error":{"message","type","param","code"}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub code: &'static str,
    pub param: Option<&'static str>,
    pub message: String,
}

impl ApiError {
    pub fn new(
        status: u16,
        kind: &'static str,
        code: &'static str,
        param: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            kind,
            code,
            param,
            message: message.into(),
        }
    }

    pub fn bad_request(
        code: &'static str,
        param: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(400, "invalid_request_error", code, param, message)
    }

    pub fn not_found(
        code: &'static str,
        param: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(404, "invalid_request_error", code, param, message)
    }

    pub fn too_many(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(429, "rate_limit_error", code, None, message)
    }

    pub fn unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(503, "service_unavailable_error", code, None, message)
    }

    pub fn to_json(&self) -> String {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
        .to_string()
    }
}

/// The four lifecycle statuses of `VideoResource.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
}

impl VideoStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// `VideoCreateError`: `code` and `message` are the only required fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoError {
    pub code: String,
    pub message: String,
}

impl VideoError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    fn to_value(&self) -> Value {
        json!({ "code": self.code, "message": self.message })
    }
}

/// `VideoSize` as `width x height`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSize {
    pub width: u32,
    pub height: u32,
}

impl VideoSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Parse only the canonical form the reference documents (`720x1280`).
    /// Leading zeros or an uppercase separator are rejected.
    pub fn parse(text: &str) -> Option<Self> {
        let (width, height) = text.split_once('x')?;
        if width.is_empty() || height.is_empty() {
            return None;
        }
        if !width.bytes().all(|b| b.is_ascii_digit()) || !height.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let size = Self::new(width.parse().ok()?, height.parse().ok()?);
        if size.as_string() == text {
            Some(size)
        } else {
            None
        }
    }

    pub fn as_string(self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// `h3.c:492-497`: multiples of 32, at least 32, product under the cap.
    pub fn check_h3(self) -> Result<(), String> {
        if self.width < H3_CANVAS_MULTIPLE || self.height < H3_CANVAS_MULTIPLE {
            return Err(format!(
                "width and height must be at least {H3_CANVAS_MULTIPLE} pixels"
            ));
        }
        if !self.width.is_multiple_of(H3_CANVAS_MULTIPLE)
            || !self.height.is_multiple_of(H3_CANVAS_MULTIPLE)
        {
            return Err(format!(
                "width and height must be multiples of {H3_CANVAS_MULTIPLE} pixels"
            ));
        }
        let pixels = self.width as u64 * self.height as u64;
        if pixels > H3_MAX_PIXELS {
            return Err(format!(
                "canvas exceeds the h3 limit of {H3_MAX_PIXELS} pixels"
            ));
        }
        Ok(())
    }
}

/// `input_reference` object of `CreateVideoJsonBody`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageRef {
    pub file_id: Option<String>,
    pub image_url: Option<String>,
}

/// Parsed `CreateVideoJsonBody`, before h3 validation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoRequest {
    pub prompt: Option<String>,
    pub input_reference: Option<ImageRef>,
    pub model: Option<String>,
    pub seconds: Option<String>,
    pub size: Option<String>,
}

/// What the renderer bridge receives after validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderSpec {
    pub prompt: String,
    pub model: String,
    pub size: VideoSize,
    pub seconds: String,
    pub frames: u32,
}

/// Parse a create body. Field names are the reference's; unknown fields are
/// ignored, as an OpenAI client may forward extras.
pub fn parse_request(body: &[u8]) -> Result<VideoRequest, ApiError> {
    let root: Value = serde_json::from_slice(body).map_err(|error| {
        ApiError::bad_request(
            "invalid_json",
            None,
            format!("We could not parse the JSON body of your request: {error}"),
        )
    })?;
    let object = root.as_object().ok_or_else(|| {
        ApiError::bad_request(
            "invalid_json",
            None,
            "The request body must be a JSON object",
        )
    })?;

    let prompt = match object.get("prompt") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(wrong_type("prompt", "a string")),
    };

    let model = match object.get("model") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(wrong_type("model", "a string")),
    };

    let seconds = match object.get("seconds") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(wrong_type(
                "seconds",
                "a string such as \"4\", \"8\" or \"12\"",
            ))
        }
    };

    let size = match object.get("size") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(wrong_type("size", "a string such as \"512x512\"")),
    };

    let input_reference = match object.get("input_reference") {
        None | Some(Value::Null) => None,
        Some(Value::Object(reference)) => Some(ImageRef {
            file_id: optional_string(reference.get("file_id"), "input_reference.file_id")?,
            image_url: optional_string(reference.get("image_url"), "input_reference.image_url")?,
        }),
        Some(_) => return Err(wrong_type("input_reference", "an object")),
    };

    Ok(VideoRequest {
        prompt,
        input_reference,
        model,
        seconds,
        size,
    })
}

fn optional_string(value: Option<&Value>, field: &'static str) -> Result<Option<String>, ApiError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(wrong_type(field, "a string")),
    }
}

fn wrong_type(field: &'static str, expected: &str) -> ApiError {
    ApiError::bad_request(
        "invalid_type",
        Some(field),
        format!("Invalid type for '{field}': expected {expected}."),
    )
}

impl VideoRequest {
    /// Validate against the h3 contract and resolve defaults.
    pub fn validate(&self, model: &str) -> Result<RenderSpec, ApiError> {
        let prompt = self.prompt.as_deref().ok_or_else(|| {
            ApiError::bad_request(
                "missing_required_parameter",
                Some("prompt"),
                "Missing required parameter: 'prompt'.",
            )
        })?;
        if prompt.is_empty() {
            return Err(ApiError::bad_request(
                "string_below_min_length",
                Some("prompt"),
                "Invalid 'prompt': string is too short. It must be at least 1 character.",
            ));
        }
        if prompt.chars().count() > MAX_PROMPT_CHARS {
            return Err(ApiError::bad_request(
                "string_above_max_length",
                Some("prompt"),
                format!("Invalid 'prompt': string is too long. It must be at most {MAX_PROMPT_CHARS} characters."),
            ));
        }

        if let Some(reference) = &self.input_reference {
            // The wire field exists, but the P8 renderer is the first h3
            // backend that takes reference images. Refuse loudly instead of
            // dropping a reference the caller believes is applied.
            let which = if reference.image_url.is_some() {
                "image_url"
            } else if reference.file_id.is_some() {
                "file_id"
            } else {
                "empty"
            };
            return Err(ApiError::bad_request(
                "unsupported_parameter",
                Some("input_reference"),
                format!(
                    "'input_reference' ({which}) is not supported by the h3 bridge; \
                     the port's own renderer lands at phase P8."
                ),
            ));
        }

        let model_id = self.model.as_deref().unwrap_or(model);
        if model_id != model {
            return Err(ApiError::bad_request(
                "model_not_found",
                Some("model"),
                format!("The model '{model_id}' does not exist or you do not have access to it."),
            ));
        }

        let size_text = self.size.as_deref().unwrap_or(DEFAULT_SIZE);
        let size = VideoSize::parse(size_text).ok_or_else(|| {
            ApiError::bad_request(
                "invalid_value",
                Some("size"),
                format!("Invalid 'size': '{size_text}'. Expected width x height, for example '512x512'."),
            )
        })?;
        size.check_h3().map_err(|reason| {
            ApiError::bad_request(
                "invalid_value",
                Some("size"),
                format!("Invalid 'size': '{size_text}': {reason}."),
            )
        })?;

        let seconds = self.seconds.as_deref().unwrap_or(DEFAULT_SECONDS);
        let frames = validate_seconds(seconds)?;

        Ok(RenderSpec {
            prompt: prompt.to_string(),
            model: model_id.to_string(),
            size,
            seconds: seconds.to_string(),
            frames,
        })
    }
}

/// Mirror `main.c:78-91` plus `h3.c:518-519`: a duration is converted at
/// 24 fps, rounded, and must land in the released 5..362 request range.
/// The aligned value is also checked against `h3.c:867-870`.
fn validate_seconds(text: &str) -> Result<u32, ApiError> {
    let seconds: f64 = text.parse().map_err(|_| invalid_seconds(text))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(invalid_seconds(text));
    }
    let frames = frames_from_seconds(seconds);
    if !(H3_MIN_FRAMES..=H3_MAX_FRAMES).contains(&frames) {
        return Err(invalid_seconds(text));
    }
    if align_frames(frames) < H3_MIN_GEN_FRAMES {
        return Err(invalid_seconds(text));
    }
    Ok(frames)
}

fn invalid_seconds(text: &str) -> ApiError {
    ApiError::bad_request(
        "invalid_value",
        Some("seconds"),
        format!(
            "Invalid 'seconds': '{text}'. h3 accepts a positive duration whose frame count \
             at {H3_FPS} fps aligns into {H3_MIN_GEN_FRAMES}..={H3_MAX_FRAMES} frames."
        ),
    )
}

/// One `VideoResource`, exactly the reference's thirteen fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Video {
    pub id: String,
    pub model: String,
    pub status: VideoStatus,
    pub progress: u8,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub expires_at: Option<u64>,
    pub prompt: String,
    pub size: VideoSize,
    pub seconds: String,
    pub remixed_from_video_id: Option<String>,
    pub error: Option<VideoError>,
}

impl Video {
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "object": VIDEO_OBJECT,
            "model": self.model,
            "status": self.status.as_str(),
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "prompt": self.prompt,
            "size": self.size.as_string(),
            "seconds": self.seconds,
            "remixed_from_video_id": self.remixed_from_video_id,
            "error": self.error.as_ref().map(VideoError::to_value),
        })
    }

    pub fn to_json(&self) -> String {
        self.to_value().to_string()
    }
}

/// `VideoListResource`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoList {
    pub data: Vec<Video>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

impl VideoList {
    pub fn to_json(&self) -> String {
        json!({
            "object": LIST_OBJECT,
            "data": self.data.iter().map(Video::to_value).collect::<Vec<_>>(),
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        })
        .to_string()
    }
}

/// `VideoDeleteResponse`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoDelete {
    pub id: String,
}

impl VideoDelete {
    pub fn to_json(&self) -> String {
        json!({ "id": self.id, "deleted": true, "object": DELETE_OBJECT }).to_string()
    }
}

/// `GET /videos` query parameters, as received strings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListQuery {
    pub limit: Option<String>,
    pub order: Option<String>,
    pub after: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

struct JobEntry {
    video: Video,
    seq: u64,
    output: Option<PathBuf>,
}

struct StoreInner {
    jobs: Vec<JobEntry>,
    next_seq: u64,
}

/// Bounded in-memory job store. `max_jobs` caps retained jobs, which also
/// caps concurrent renders: every retained job owns at most one thread.
pub struct VideoStore {
    inner: Mutex<StoreInner>,
    model: String,
    max_jobs: usize,
}

impl VideoStore {
    pub fn new(model: impl Into<String>, max_jobs: usize) -> Self {
        Self {
            inner: Mutex::new(StoreInner {
                jobs: Vec::new(),
                next_seq: 1,
            }),
            model: model.into(),
            max_jobs,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    fn lock(&self) -> MutexGuard<'_, StoreInner> {
        // A poisoned lock only means another job thread panicked; the store
        // itself is still a valid list of plain values.
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Validate and insert a queued job. Returns the job and its render spec.
    pub fn submit(
        &self,
        request: &VideoRequest,
        now: u64,
    ) -> Result<(Video, RenderSpec), ApiError> {
        let spec = request.validate(&self.model)?;
        let mut inner = self.lock();
        if self.max_jobs == 0 || inner.jobs.len() >= self.max_jobs {
            return Err(ApiError::too_many(
                "queue_full",
                format!(
                    "The h3 bridge holds at most {} jobs; delete a finished job and retry.",
                    self.max_jobs
                ),
            ));
        }
        let seq = inner.next_seq;
        inner.next_seq += 1;
        let video = Video {
            id: format!("video_{seq}"),
            model: spec.model.clone(),
            status: VideoStatus::Queued,
            progress: 0,
            created_at: now,
            completed_at: None,
            expires_at: None,
            prompt: spec.prompt.clone(),
            size: spec.size,
            seconds: spec.seconds.clone(),
            remixed_from_video_id: None,
            error: None,
        };
        inner.jobs.push(JobEntry {
            video: video.clone(),
            seq,
            output: None,
        });
        Ok((video, spec))
    }

    pub fn start(&self, id: &str) -> bool {
        let mut inner = self.lock();
        let Some(job) = inner.jobs.iter_mut().find(|job| job.video.id == id) else {
            return false;
        };
        if job.video.status != VideoStatus::Queued {
            return false;
        }
        job.video.status = VideoStatus::InProgress;
        job.video.progress = 0;
        true
    }

    pub fn complete(&self, id: &str, now: u64, output: PathBuf) -> bool {
        let mut inner = self.lock();
        let Some(job) = inner.jobs.iter_mut().find(|job| job.video.id == id) else {
            return false;
        };
        if job.video.status != VideoStatus::InProgress {
            return false;
        }
        job.video.status = VideoStatus::Completed;
        job.video.progress = 100;
        job.video.completed_at = Some(now);
        job.output = Some(output);
        true
    }

    pub fn fail(&self, id: &str, now: u64, error: VideoError) -> bool {
        let mut inner = self.lock();
        let Some(job) = inner.jobs.iter_mut().find(|job| job.video.id == id) else {
            return false;
        };
        if matches!(
            job.video.status,
            VideoStatus::Completed | VideoStatus::Failed
        ) {
            return false;
        }
        job.video.status = VideoStatus::Failed;
        job.video.progress = 0;
        job.video.completed_at = Some(now);
        job.video.error = Some(error);
        true
    }

    pub fn get(&self, id: &str) -> Option<Video> {
        self.lock()
            .jobs
            .iter()
            .find(|job| job.video.id == id)
            .map(|job| job.video.clone())
    }

    pub fn output_path(&self, id: &str) -> Option<PathBuf> {
        self.lock()
            .jobs
            .iter()
            .find(|job| job.video.id == id)
            .filter(|job| job.video.status == VideoStatus::Completed)
            .and_then(|job| job.output.clone())
    }

    pub fn len(&self) -> usize {
        self.lock().jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn list(&self, query: &ListQuery) -> Result<VideoList, ApiError> {
        let limit = parse_limit(query.limit.as_deref())?;
        let order = parse_order(query.order.as_deref())?;
        let mut ordered: Vec<(u64, Video)> = self
            .lock()
            .jobs
            .iter()
            .map(|job| (job.seq, job.video.clone()))
            .collect();
        ordered.sort_by_key(|(seq, video)| (video.created_at, *seq));
        if order == SortOrder::Desc {
            ordered.reverse();
        }

        let start = match query.after.as_deref() {
            None => 0,
            Some(cursor) => {
                let position = ordered
                    .iter()
                    .position(|(_, video)| video.id == cursor)
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "invalid_after",
                            Some("after"),
                            format!("Invalid 'after': no video with id '{cursor}'."),
                        )
                    })?;
                position + 1
            }
        };
        let end = start.saturating_add(limit).min(ordered.len());
        let data: Vec<Video> = ordered[start..end]
            .iter()
            .map(|(_, video)| video.clone())
            .collect();
        Ok(VideoList {
            first_id: data.first().map(|video| video.id.clone()),
            last_id: data.last().map(|video| video.id.clone()),
            has_more: end < ordered.len(),
            data,
        })
    }

    /// Only completed or failed jobs may be deleted, per the reference's
    /// "Permanently delete a completed or failed video" description.
    pub fn delete(&self, id: &str) -> Result<(VideoDelete, Option<PathBuf>), ApiError> {
        let mut inner = self.lock();
        let index = inner
            .jobs
            .iter()
            .position(|job| job.video.id == id)
            .ok_or_else(|| unknown_video(id))?;
        if !matches!(
            inner.jobs[index].video.status,
            VideoStatus::Completed | VideoStatus::Failed
        ) {
            return Err(ApiError::bad_request(
                "video_not_ready",
                Some("video_id"),
                format!(
                    "Video '{id}' is {}; only completed or failed videos can be deleted.",
                    inner.jobs[index].video.status.as_str()
                ),
            ));
        }
        let job = inner.jobs.remove(index);
        Ok((VideoDelete { id: id.to_string() }, job.output))
    }
}

fn unknown_video(id: &str) -> ApiError {
    ApiError::not_found(
        "video_not_found",
        Some("video_id"),
        format!("Video with id '{id}' not found."),
    )
}

fn parse_limit(text: Option<&str>) -> Result<usize, ApiError> {
    let Some(text) = text else {
        return Ok(DEFAULT_LIST_LIMIT);
    };
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid_limit(text));
    }
    let limit: usize = text.parse().map_err(|_| invalid_limit(text))?;
    if !(MIN_LIST_LIMIT..=MAX_LIST_LIMIT).contains(&limit) {
        return Err(invalid_limit(text));
    }
    Ok(limit)
}

fn invalid_limit(text: &str) -> ApiError {
    ApiError::bad_request(
        "invalid_value",
        Some("limit"),
        format!(
            "Invalid 'limit': '{text}'. Expected an integer from {MIN_LIST_LIMIT} to {MAX_LIST_LIMIT}."
        ),
    )
}

fn parse_order(text: Option<&str>) -> Result<SortOrder, ApiError> {
    match text {
        // The reference states no default. "List recently generated videos"
        // and the other OpenAI list surfaces lead with newest, so the bridge
        // defaults to desc and documents that as an assumption.
        None => Ok(SortOrder::Desc),
        Some("asc") => Ok(SortOrder::Asc),
        Some("desc") => Ok(SortOrder::Desc),
        Some(other) => Err(ApiError::bad_request(
            "invalid_value",
            Some("order"),
            format!("Invalid 'order': '{other}'. Expected 'asc' or 'desc'."),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: &str) -> VideoRequest {
        VideoRequest {
            prompt: Some(prompt.to_string()),
            ..VideoRequest::default()
        }
    }

    fn store() -> VideoStore {
        VideoStore::new(DEFAULT_MODEL, 8)
    }

    fn value(text: &str) -> Value {
        serde_json::from_str(text).expect("valid JSON")
    }

    #[test]
    fn video_serializes_exactly_the_official_fields() {
        let (video, _) = store()
            .submit(&request("a red cube"), 1_700_000_000)
            .unwrap();
        let json = value(&video.to_json());
        let object = json.as_object().expect("object");

        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "completed_at",
                "created_at",
                "error",
                "expires_at",
                "id",
                "model",
                "object",
                "progress",
                "prompt",
                "remixed_from_video_id",
                "seconds",
                "size",
                "status",
            ]
        );
        assert_eq!(object["id"], "video_1");
        assert_eq!(object["object"], VIDEO_OBJECT);
        assert_eq!(object["model"], DEFAULT_MODEL);
        assert_eq!(object["status"], "queued");
        assert_eq!(object["progress"], 0);
        assert_eq!(object["created_at"], 1_700_000_000_u64);
        assert_eq!(object["completed_at"], Value::Null);
        assert_eq!(object["expires_at"], Value::Null);
        assert_eq!(object["error"], Value::Null);
        assert_eq!(object["remixed_from_video_id"], Value::Null);
        assert_eq!(object["prompt"], "a red cube");
        assert_eq!(object["size"], DEFAULT_SIZE);
        assert_eq!(object["seconds"], DEFAULT_SECONDS);
    }

    #[test]
    fn create_body_accepts_the_official_fields() {
        let body = br#"{
            "prompt": "a red cube",
            "model": "minimax-h3",
            "seconds": "8",
            "size": "768x768",
            "input_reference": null,
            "unknown": true
        }"#;
        let parsed = parse_request(body).unwrap();
        assert_eq!(parsed.prompt.as_deref(), Some("a red cube"));
        assert_eq!(parsed.model.as_deref(), Some(DEFAULT_MODEL));
        assert_eq!(parsed.seconds.as_deref(), Some("8"));
        assert_eq!(parsed.size.as_deref(), Some("768x768"));
        assert!(parsed.input_reference.is_none());

        let spec = parsed.validate(DEFAULT_MODEL).unwrap();
        assert_eq!(spec.size, VideoSize::new(768, 768));
        assert_eq!(spec.frames, 192);
    }

    #[test]
    fn lifecycle_moves_queued_to_completed() {
        let store = store();
        let (video, _) = store.submit(&request("p"), 10).unwrap();
        assert_eq!(video.status, VideoStatus::Queued);
        assert!(store.start(&video.id));
        let running = store.get(&video.id).unwrap();
        assert_eq!(running.status, VideoStatus::InProgress);
        assert!(store.complete(&video.id, 20, PathBuf::from("/tmp/video_1.mp4")));
        let done = store.get(&video.id).unwrap();
        assert_eq!(done.status, VideoStatus::Completed);
        assert_eq!(done.progress, 100);
        assert_eq!(done.completed_at, Some(20));
        assert_eq!(done.error, None);
        assert_eq!(
            store.output_path(&video.id),
            Some(PathBuf::from("/tmp/video_1.mp4"))
        );
        assert!(!store.start(&video.id), "completed jobs do not restart");
    }

    #[test]
    fn failure_records_the_official_error_payload() {
        let store = store();
        let (video, _) = store.submit(&request("p"), 10).unwrap();
        store.start(&video.id);
        assert!(store.fail(
            &video.id,
            30,
            VideoError::new("render_failed", "renderer exited with status 1"),
        ));
        let failed = store.get(&video.id).unwrap();
        assert_eq!(failed.status, VideoStatus::Failed);
        assert_eq!(failed.completed_at, Some(30));
        let json = value(&failed.to_json());
        assert_eq!(json["error"]["code"], "render_failed");
        assert_eq!(json["error"]["message"], "renderer exited with status 1");
        assert_eq!(
            json["error"].as_object().unwrap().len(),
            2,
            "VideoCreateError requires only code and message"
        );
    }

    #[test]
    fn ids_are_monotonic_across_deletes() {
        let store = store();
        let (first, _) = store.submit(&request("a"), 1).unwrap();
        let (second, _) = store.submit(&request("b"), 2).unwrap();
        assert_eq!(first.id, "video_1");
        assert_eq!(second.id, "video_2");
        store.start(&first.id);
        store.complete(&first.id, 3, PathBuf::from("/tmp/a.mp4"));
        store.delete(&first.id).unwrap();
        let (third, _) = store.submit(&request("c"), 4).unwrap();
        assert_eq!(third.id, "video_3", "ids never reuse a deleted slot");
    }

    #[test]
    fn list_paginates_with_a_cursor_in_both_orders() {
        let store = store();
        for index in 0..3 {
            let (video, _) = store
                .submit(&request(&format!("p{index}")), 100 + index)
                .unwrap();
            store.start(&video.id);
            store.complete(&video.id, 200 + index, PathBuf::from("/tmp/x.mp4"));
        }

        let first = store
            .list(&ListQuery {
                limit: Some("2".to_string()),
                ..ListQuery::default()
            })
            .unwrap();
        assert_eq!(first.data.len(), 2);
        assert_eq!(first.first_id.as_deref(), Some("video_3"));
        assert_eq!(first.last_id.as_deref(), Some("video_2"));
        assert!(first.has_more);

        let second = store
            .list(&ListQuery {
                limit: Some("2".to_string()),
                after: first.last_id.clone(),
                ..ListQuery::default()
            })
            .unwrap();
        assert_eq!(second.data.len(), 1);
        assert_eq!(second.first_id.as_deref(), Some("video_1"));
        assert!(!second.has_more);

        let ascending = store
            .list(&ListQuery {
                order: Some("asc".to_string()),
                limit: Some("2".to_string()),
                ..ListQuery::default()
            })
            .unwrap();
        assert_eq!(ascending.first_id.as_deref(), Some("video_1"));
        assert_eq!(ascending.last_id.as_deref(), Some("video_2"));
        assert!(ascending.has_more);

        let json = value(&first.to_json());
        assert_eq!(json["object"], LIST_OBJECT);
        assert_eq!(json["has_more"], true);
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn list_limit_zero_returns_an_empty_page() {
        let store = store();
        store.submit(&request("p"), 1).unwrap();
        let page = store
            .list(&ListQuery {
                limit: Some("0".to_string()),
                ..ListQuery::default()
            })
            .unwrap();
        assert!(page.data.is_empty());
        assert_eq!(page.first_id, None);
        assert_eq!(page.last_id, None);
        assert!(page.has_more);
    }

    #[test]
    fn list_rejects_bad_query_parameters() {
        let store = store();
        for (query, param) in [
            (
                ListQuery {
                    limit: Some("101".to_string()),
                    ..ListQuery::default()
                },
                "limit",
            ),
            (
                ListQuery {
                    limit: Some("-1".to_string()),
                    ..ListQuery::default()
                },
                "limit",
            ),
            (
                ListQuery {
                    order: Some("newest".to_string()),
                    ..ListQuery::default()
                },
                "order",
            ),
            (
                ListQuery {
                    after: Some("video_99".to_string()),
                    ..ListQuery::default()
                },
                "after",
            ),
        ] {
            let error = store.list(&query).unwrap_err();
            assert_eq!(error.status, 400);
            assert_eq!(error.param, Some(param));
        }
    }

    #[test]
    fn delete_only_removes_finished_jobs() {
        let store = store();
        let (video, _) = store.submit(&request("p"), 1).unwrap();
        let error = store.delete(&video.id).unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "video_not_ready");
        assert_eq!(error.param, Some("video_id"));

        store.start(&video.id);
        store.complete(&video.id, 2, PathBuf::from("/tmp/v.mp4"));
        let (deleted, output) = store.delete(&video.id).unwrap();
        assert_eq!(deleted.id, video.id);
        assert_eq!(output, Some(PathBuf::from("/tmp/v.mp4")));
        let json = value(&deleted.to_json());
        assert_eq!(json["object"], DELETE_OBJECT);
        assert_eq!(json["deleted"], true);

        assert_eq!(store.delete(&video.id).unwrap_err().code, "video_not_found");
        assert!(store.get(&video.id).is_none());
    }

    #[test]
    fn queue_full_refuses_with_a_rate_limit_error() {
        let store = VideoStore::new(DEFAULT_MODEL, 1);
        store.submit(&request("a"), 1).unwrap();
        let error = store.submit(&request("b"), 2).unwrap_err();
        assert_eq!(error.status, 429);
        assert_eq!(error.kind, "rate_limit_error");
        let json = value(&error.to_json());
        assert_eq!(json["error"]["code"], "queue_full");
        assert_eq!(json["error"]["param"], Value::Null);
    }

    #[test]
    fn unknown_model_names_the_model_param() {
        let parsed = VideoRequest {
            prompt: Some("p".to_string()),
            model: Some("sora-2".to_string()),
            ..VideoRequest::default()
        };
        let error = parsed.validate(DEFAULT_MODEL).unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "model_not_found");
        assert_eq!(error.param, Some("model"));
    }

    #[test]
    fn prompt_bounds_are_enforced() {
        let missing = VideoRequest::default().validate(DEFAULT_MODEL).unwrap_err();
        assert_eq!(missing.code, "missing_required_parameter");
        assert_eq!(missing.param, Some("prompt"));

        let empty = request("").validate(DEFAULT_MODEL).unwrap_err();
        assert_eq!(empty.code, "string_below_min_length");

        let long = request(&"x".repeat(MAX_PROMPT_CHARS + 1))
            .validate(DEFAULT_MODEL)
            .unwrap_err();
        assert_eq!(long.code, "string_above_max_length");
        assert_eq!(long.param, Some("prompt"));

        assert!(request(&"x".repeat(MAX_PROMPT_CHARS))
            .validate(DEFAULT_MODEL)
            .is_ok());
    }

    #[test]
    fn size_must_be_an_h3_canvas() {
        // Every official OpenAI size is illegal for h3: 720 is not a
        // multiple of 32, and 1792*1024 and 1024*1792 exceed the pixel cap.
        for official in ["720x1280", "1280x720", "1024x1792", "1792x1024"] {
            let error = VideoRequest {
                prompt: Some("p".to_string()),
                size: Some(official.to_string()),
                ..VideoRequest::default()
            }
            .validate(DEFAULT_MODEL)
            .unwrap_err();
            assert_eq!(error.param, Some("size"), "{official}");
        }
        for bad in ["512", "512X512", "0512x512", "512x", "x512", "abc"] {
            let error = VideoRequest {
                prompt: Some("p".to_string()),
                size: Some(bad.to_string()),
                ..VideoRequest::default()
            }
            .validate(DEFAULT_MODEL)
            .unwrap_err();
            assert_eq!(error.param, Some("size"), "{bad}");
        }
        for good in ["512x512", "1344x768", "768x1344", "32x32"] {
            assert!(
                VideoRequest {
                    prompt: Some("p".to_string()),
                    size: Some(good.to_string()),
                    ..VideoRequest::default()
                }
                .validate(DEFAULT_MODEL)
                .is_ok(),
                "{good}"
            );
        }
        assert_eq!(VideoSize::parse("512x512"), Some(VideoSize::new(512, 512)));
        assert_eq!(VideoSize::parse("512X512"), None);
    }

    #[test]
    fn seconds_must_land_in_the_h3_frame_range() {
        for bad in [
            "0", "-1", "abc", "NaN", "inf", "0.18", "0.21", "15.11", "16",
        ] {
            let error = VideoRequest {
                prompt: Some("p".to_string()),
                seconds: Some(bad.to_string()),
                ..VideoRequest::default()
            }
            .validate(DEFAULT_MODEL)
            .unwrap_err();
            assert_eq!(error.param, Some("seconds"), "{bad}");
        }
        for (good, frames) in [
            ("4", 96),
            ("8", 192),
            ("12", 288),
            ("0.23", 6),
            ("15.1", 362),
        ] {
            let spec = VideoRequest {
                prompt: Some("p".to_string()),
                seconds: Some(good.to_string()),
                ..VideoRequest::default()
            }
            .validate(DEFAULT_MODEL)
            .unwrap();
            assert_eq!(spec.frames, frames, "{good}");
        }
        assert_eq!(align_frames(1), 5);
        assert_eq!(align_frames(5), 5);
        assert_eq!(align_frames(6), 22);
        assert_eq!(align_frames(22), 22);
        assert_eq!(align_frames(23), 39);
        assert_eq!(align_frames(362), 362);
    }

    #[test]
    fn malformed_json_and_wrong_types_map_to_their_params() {
        let error = parse_request(b"{not json").unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "invalid_json");
        assert_eq!(error.param, None);

        let error = parse_request(b"[]").unwrap_err();
        assert_eq!(error.code, "invalid_json");

        let error = parse_request(br#"{"prompt": 7}"#).unwrap_err();
        assert_eq!(error.code, "invalid_type");
        assert_eq!(error.param, Some("prompt"));

        let error = parse_request(br#"{"prompt":"p","seconds":4}"#).unwrap_err();
        assert_eq!(error.param, Some("seconds"));

        let error = parse_request(
            br#"{"prompt":"p","input_reference":{"image_url":"data:image/png;base64,AA"}}"#,
        )
        .unwrap()
        .validate(DEFAULT_MODEL)
        .unwrap_err();
        assert_eq!(error.code, "unsupported_parameter");
        assert_eq!(error.param, Some("input_reference"));
    }

    #[test]
    fn error_shape_always_carries_all_four_fields() {
        let json = value(&unknown_video("video_9").to_json());
        let error = json["error"].as_object().unwrap();
        assert_eq!(error["message"], "Video with id 'video_9' not found.");
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(error["param"], "video_id");
        assert_eq!(error["code"], "video_not_found");
        assert_eq!(error.len(), 4);
    }
}
