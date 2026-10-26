//! Task state for the h3 family's local v1 surface.
//!
//! The served contract is `docs/minimax-h3/local-v1-contract.md`. The create
//! body and its mapping onto the engine live in `local_api.rs`; HTTP lives in
//! `ds4-server`. This module owns only what sits between them: the task
//! record, its four states and the bounded in-memory store.
//!
//! Frame cadence and limits are the reference's, copied from `/data/h3.c` at
//! eae5d54:
//!
//! - `h3_host.h:9` `H3_FPS 24`;
//! - `h3_host.c:17-23` frames align upward to `5 + 17*n`;
//! - `h3.c:518-519` requested frames must land in 5..362 after alignment.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::local_api::Plan;

/// The server's prompt cap. The scripts send an H3-Context-IR expansion,
/// which runs to a few thousand characters; this leaves room without
/// admitting an unbounded body into the transcript.
pub const MAX_PROMPT_CHARS: usize = 32_000;

/// Decoded anchor limit. The HTTP body cap is 64 MiB; an anchor is one still
/// image, and the reference's reader has no limit of its own
/// (`h3_ffmpeg.c:143-160`), so the server states one.
pub const MAX_ANCHOR_BYTES: usize = 16 * 1024 * 1024;

pub const H3_FPS: u32 = 24;
pub const H3_ALIGN_BASE: u32 = 5;
pub const H3_ALIGN_STEP: u32 = 17;
pub const H3_MIN_FRAMES: u32 = 5;
pub const H3_MAX_FRAMES: u32 = 362;

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

/// A refused request, with the code the contract page names.
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

/// The four lifecycle states. The official client polls `status` and
/// downloads once it reads `completed` (`full-2k-t2va-h3-base.sh:26-31`); the
/// other three names are this port's, since no script prints them.
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

/// Why a job failed: one code and one sentence.
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

/// A canvas in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSize {
    pub width: u32,
    pub height: u32,
}

impl VideoSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn as_string(self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

/// One conditioning image as the wire delivered it: the decoded bytes of a
/// `data:image/...;base64,` URI and the media type that names their format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorImage {
    pub media_type: String,
    pub bytes: Vec<u8>,
}

impl AnchorImage {
    /// The file extension the media type names. FFmpeg's image demuxer picks a
    /// decoder from the input name, so an anchor without one is refused rather
    /// than guessed (`h3_ffmpeg.c:143-160`).
    pub fn extension(&self) -> Option<&'static str> {
        match self.media_type.as_str() {
            "image/png" => Some("png"),
            "image/jpeg" | "image/jpg" => Some("jpg"),
            "image/webp" => Some("webp"),
            _ => None,
        }
    }
}

/// One task record.
///
/// `size` is the resolved canvas, `seconds` the requested duration and
/// `frames` the aligned count the engine renders. The prompt is not echoed:
/// a poll answer would repeat up to [`MAX_PROMPT_CHARS`] characters on every
/// request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Video {
    pub id: String,
    pub status: VideoStatus,
    pub size: VideoSize,
    pub seconds: u32,
    pub frames: u32,
    pub seed: u64,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub error: Option<VideoError>,
}

impl Video {
    /// The record as the query route answers it. `queue_position` is the one
    /// field the store derives at read time instead of storing: 1 for the job
    /// the engine takes next, null while a job holds the single render slot and
    /// for a job that has already run.
    pub fn to_value(&self, queue_position: Option<usize>) -> Value {
        json!({
            "id": self.id,
            "status": self.status.as_str(),
            "queue_position": queue_position,
            "size": self.size.as_string(),
            "seconds": self.seconds,
            "frames": self.frames,
            "seed": self.seed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error.as_ref().map(VideoError::to_value),
        })
    }

    pub fn to_json(&self, queue_position: Option<usize>) -> String {
        self.to_value(queue_position).to_string()
    }
}

/// A stored record plus the field the store derives at read time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoView {
    pub video: Video,
    /// 1 for the job the engine takes next, null otherwise.
    pub queue_position: Option<usize>,
}

impl VideoView {
    pub fn to_value(&self) -> Value {
        self.video.to_value(self.queue_position)
    }

    pub fn to_json(&self) -> String {
        self.to_value().to_string()
    }
}

struct JobEntry {
    video: Video,
    /// The validated plan, held until admission hands it to the renderer.
    plan: Option<Plan>,
    output: Option<PathBuf>,
}

/// A job the store has admitted. One device renders one request at a time
/// (`h3.c:697-723`), so admission belongs to the store, not to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    pub id: String,
    pub plan: Plan,
}

struct StoreInner {
    jobs: Vec<JobEntry>,
    next_seq: u64,
}

/// Bounded in-memory job store. `max_jobs` caps retained jobs, so a client
/// that never stops creating gets a refusal instead of an unbounded table.
///
/// One job at a time: a submitted job waits in the queue until `admit_next`
/// hands it the render slot, because one device renders one request at a time.
///
/// There is no delete route in the local contract, so a job and its MP4 stay
/// until the host restarts; the output directory is the operator's to prune.
pub struct VideoStore {
    inner: Mutex<StoreInner>,
    max_jobs: usize,
    id_prefix: String,
}

impl VideoStore {
    pub fn new(max_jobs: usize) -> Self {
        static NEXT_STORE: AtomicU64 = AtomicU64::new(0);
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        Self {
            inner: Mutex::new(StoreInner {
                jobs: Vec::new(),
                next_seq: 1,
            }),
            max_jobs,
            // Restarting the host must not reuse completed output names.
            id_prefix: format!(
                "video_{epoch:x}_{:x}_{:x}",
                std::process::id(),
                NEXT_STORE.fetch_add(1, Ordering::Relaxed)
            ),
        }
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

    /// Insert a validated plan as a queued job. Returns the task record.
    pub fn submit(&self, plan: &Plan, now: u64) -> Result<Video, ApiError> {
        let mut inner = self.lock();
        if self.max_jobs == 0 || inner.jobs.len() >= self.max_jobs {
            return Err(ApiError::too_many(
                "queue_full",
                format!(
                    "The h3 host holds at most {} jobs; raise --max-jobs or restart it.",
                    self.max_jobs
                ),
            ));
        }
        let seq = inner.next_seq;
        inner.next_seq += 1;
        let video = Video {
            id: format!("{}_{seq}", self.id_prefix),
            status: VideoStatus::Queued,
            size: plan.size,
            seconds: plan.seconds,
            frames: plan.frames,
            seed: plan.seed,
            created_at: now,
            completed_at: None,
            error: None,
        };
        inner.jobs.push(JobEntry {
            video: video.clone(),
            plan: Some(plan.clone()),
            output: None,
        });
        Ok(video)
    }

    /// Admit the next queued job if the render slot is free. The caller
    /// renders what it gets and calls this again when it finishes.
    pub fn admit_next(&self) -> Option<Admitted> {
        let mut inner = self.lock();
        if inner
            .jobs
            .iter()
            .any(|job| job.video.status == VideoStatus::InProgress)
        {
            return None;
        }
        let job = inner
            .jobs
            .iter_mut()
            .find(|job| job.video.status == VideoStatus::Queued)?;
        let plan = job.plan.take()?;
        job.video.status = VideoStatus::InProgress;
        Some(Admitted {
            id: job.video.id.clone(),
            plan,
        })
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

    /// The record as the query route answers it, with its place in line.
    pub fn view(&self, id: &str) -> Option<VideoView> {
        let inner = self.lock();
        let queue_position = queued_position(&inner, id);
        inner
            .jobs
            .iter()
            .find(|job| job.video.id == id)
            .map(|job| VideoView {
                video: job.video.clone(),
                queue_position,
            })
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
}

/// A queued job's place in line, 1 for the job the engine takes next. `None`
/// for a job that holds the slot or has finished, and for an unknown id.
fn queued_position(inner: &StoreInner, id: &str) -> Option<usize> {
    let queued = inner
        .jobs
        .iter()
        .filter(|job| job.video.status == VideoStatus::Queued);
    for (position, job) in queued.enumerate() {
        if job.video.id == id {
            return Some(position + 1);
        }
    }
    None
}

pub fn unknown_video(id: &str) -> ApiError {
    ApiError::not_found(
        "video_not_found",
        Some("video_id"),
        format!("Video with id '{id}' not found."),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::local_api::{parse_create, Task};

    fn plan() -> Plan {
        Plan {
            task: Task::T2va,
            prompt: "a red cube".to_string(),
            size: VideoSize::new(1344, 768),
            frames: 243,
            seconds: 10,
            seed: 7,
            first_frame: None,
            last_frame: None,
            references: Vec::new(),
        }
    }

    #[test]
    fn the_record_carries_the_resolved_target_and_the_lifecycle() {
        let store = VideoStore::new(8);
        let video = store.submit(&plan(), 1_700_000_000).unwrap();
        let object = video.to_value(None);
        let keys: Vec<&str> = object
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "id",
                "status",
                "queue_position",
                "size",
                "seconds",
                "frames",
                "seed",
                "created_at",
                "completed_at",
                "error"
            ]
        );
        assert_eq!(object["id"], video.id);
        assert_eq!(object["status"], "queued");
        assert_eq!(object["queue_position"], Value::Null);
        assert_eq!(object["size"], "1344x768");
        assert_eq!(object["seconds"], 10);
        assert_eq!(object["frames"], 243);
        assert_eq!(object["seed"], 7);
        assert_eq!(object["completed_at"], Value::Null);
        assert_eq!(object["error"], Value::Null);
    }

    #[test]
    fn a_job_moves_queued_to_in_progress_to_completed() {
        let store = VideoStore::new(8);
        let video = store.submit(&plan(), 10).unwrap();
        assert_eq!(store.output_path(&video.id), None);
        assert_eq!(store.admit_next().unwrap().id, video.id);
        assert_eq!(
            store.get(&video.id).unwrap().status,
            VideoStatus::InProgress
        );
        assert!(store.complete(&video.id, 20, PathBuf::from("/tmp/x.mp4")));
        let done = store.get(&video.id).unwrap();
        assert_eq!(done.status, VideoStatus::Completed);
        assert_eq!(done.completed_at, Some(20));
        assert_eq!(
            store.output_path(&video.id),
            Some(PathBuf::from("/tmp/x.mp4"))
        );
        assert!(!store.is_empty());
    }

    #[test]
    fn a_failure_is_terminal_and_carries_its_code() {
        let store = VideoStore::new(8);
        let video = store.submit(&plan(), 10).unwrap();
        assert!(store.admit_next().is_some());
        assert!(store.fail(&video.id, 20, VideoError::new("render_failed", "exit 1")));
        assert!(
            store.admit_next().is_none(),
            "a failed job is not re-admitted"
        );
        assert!(!store.complete(&video.id, 30, PathBuf::from("/tmp/x.mp4")));
        let failed = store.get(&video.id).unwrap();
        assert_eq!(failed.status, VideoStatus::Failed);
        assert_eq!(failed.error.unwrap().code, "render_failed");
    }

    #[test]
    fn the_store_admits_one_job_and_queues_the_rest() {
        let store = VideoStore::new(8);
        let first = store.submit(&plan(), 10).unwrap();
        let second = store.submit(&plan(), 11).unwrap();
        assert_eq!(store.view(&first.id).unwrap().queue_position, Some(1));
        assert_eq!(store.view(&second.id).unwrap().queue_position, Some(2));

        let admitted = store.admit_next().expect("the slot is free");
        assert_eq!(admitted.id, first.id);
        assert_eq!(store.view(&first.id).unwrap().queue_position, None);
        assert_eq!(store.view(&second.id).unwrap().queue_position, Some(1));
        assert!(store.admit_next().is_none(), "one slot, one job");

        assert!(store.complete(&first.id, 20, PathBuf::from("/tmp/x.mp4")));
        assert_eq!(store.admit_next().expect("the slot is free").id, second.id);
        assert_eq!(store.view(&second.id).unwrap().queue_position, None);
    }

    #[test]
    fn the_store_refuses_more_jobs_than_its_bound() {
        let store = VideoStore::new(1);
        store.submit(&plan(), 10).unwrap();
        let error = store.submit(&plan(), 11).unwrap_err();
        assert_eq!(error.status, 429);
        assert_eq!(error.code, "queue_full");
        assert!(VideoStore::new(0).submit(&plan(), 12).is_err());
    }

    #[test]
    fn the_body_produces_the_plan_the_store_records() {
        // The one tandem check between the wire module and this one: the
        // scripts' own t2va body reaches a 1344x768, 243-frame task.
        let body = br#"{
            "task": "t2va",
            "prompt": "a red cube",
            "conditions": [],
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10},
            "seed": 0
        }"#;
        let plan = parse_create(body).unwrap().plan().unwrap();
        let video = VideoStore::new(8).submit(&plan, 1).unwrap();
        assert_eq!(video.size, VideoSize::new(1344, 768));
        assert_eq!(video.frames, 243);
    }
}
