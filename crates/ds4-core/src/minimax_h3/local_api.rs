//! The local v1 video contract for the h3 family.
//!
//! Authority: the official repository's own local clients at
//! `/data/MiniMax-H3/scripts/readme/*.sh`. Every route, field and value below
//! is one of those scripts sends; nothing is added. The contract page is
//! `docs/minimax-h3/local-v1-contract.md`.
//!
//! ```text
//! POST /v1/videos               -> {"id": "<id>"}
//! GET  /v1/videos/{id}          -> {"status": ...} and the task record
//! GET  /v1/videos/{id}/content  -> the MP4 bytes
//! ```
//!
//! Named deviations, each carried in the contract page:
//!
//! - a condition `uri` must be a data URI. The scripts pass https CDN URLs;
//!   this host neither fetches a URL from a request body nor reads a
//!   caller-named filesystem path;
//! - `task: ref2va` is refused until the reference-media units land, and the
//!   message names them (`reference`, `reference_video`, `reference_audio`,
//!   `base_video` are their roles);
//! - `target.short_edge` is 768, the one tier this checkpoint provides;
//! - `aspect_ratio` 21:9 is refused: the released 768x1344 area cannot express
//!   it without the 1.33x anamorphic stretch `presets::MAX_DISTORTION` bounds.
//!
//! No HTTP and no GPU live here: this module parses the body and maps it onto
//! the engine's request. The state machine and the store are
//! `minimax_h3::video_api`'s, unchanged.

use serde_json::{Map, Value};

use super::presets::{self, Ratio};
use super::render::MAX_PIXELS;
use super::video_api::{
    align_frames, frames_from_seconds, AnchorImage, ApiError, VideoSize, H3_MAX_FRAMES,
    H3_MIN_FRAMES, MAX_ANCHOR_BYTES, MAX_PROMPT_CHARS,
};

/// The checkpoint's tier: the short edge every canvas here uses.
pub const SHORT_EDGE: u32 = presets::TIER_SHORT_EDGE as u32;

/// The documented duration range. The scripts send 5, 8 and 10; the platform
/// document's enum is 4 through 15, and the engine's aligned frames stay
/// inside 22..362 across it.
pub const MIN_SECONDS: u32 = 4;
pub const MAX_SECONDS: u32 = 15;

/// `aspect_ratio` value that takes the aspect from the anchored image.
pub const AUTO_RATIO: &str = "auto";

/// A clip carries at most a first and a last frame.
pub const MAX_CONDITIONS: usize = 2;

/// The reference's cap on ordered references (`h3.c:588`).
pub const MAX_REFERENCES: usize = 12;

/// The five fields of the create body, and the only five.
pub const CREATE_FIELDS: [&str; 5] = ["task", "prompt", "conditions", "target", "seed"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    T2va,
    Fl2va,
    Ref2va,
}

impl Task {
    pub fn parse(text: &str) -> Option<Task> {
        match text {
            "t2va" => Some(Task::T2va),
            "fl2va" => Some(Task::Fl2va),
            "ref2va" => Some(Task::Ref2va),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Task::T2va => "t2va",
            Task::Fl2va => "fl2va",
            Task::Ref2va => "ref2va",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionType {
    Image,
    Audio,
    Video,
}

impl ConditionType {
    fn parse(text: &str) -> Option<ConditionType> {
        match text {
            "image" => Some(ConditionType::Image),
            "audio" => Some(ConditionType::Audio),
            "video" => Some(ConditionType::Video),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Keyframe,
    FirstFrame,
    BaseVideo,
    Reference,
    ReferenceVideo,
    ReferenceAudio,
}

impl Role {
    fn parse(text: &str) -> Option<Role> {
        match text {
            "keyframe" => Some(Role::Keyframe),
            "first_frame" => Some(Role::FirstFrame),
            "base_video" => Some(Role::BaseVideo),
            "reference" => Some(Role::Reference),
            "reference_video" => Some(Role::ReferenceVideo),
            "reference_audio" => Some(Role::ReferenceAudio),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Role::Keyframe => "keyframe",
            Role::FirstFrame => "first_frame",
            Role::BaseVideo => "base_video",
            Role::Reference => "reference",
            Role::ReferenceVideo => "reference_video",
            Role::ReferenceAudio => "reference_audio",
        }
    }

    /// The roles that condition the clip (an anchored frame), as opposed to
    /// the Ref2VA roles (reference media).
    fn is_keyframe(self) -> bool {
        matches!(self, Role::Keyframe | Role::FirstFrame)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub kind: ConditionType,
    pub role: Role,
    /// Only a keyframe carries one: 0 is the first frame, 1 the last.
    pub frame_index: Option<u32>,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub short_edge: u32,
    pub aspect_ratio: String,
    pub duration_seconds: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateBody {
    pub task: Task,
    pub prompt: String,
    pub conditions: Vec<Condition>,
    pub target: Target,
    pub seed: u64,
}

/// The class of one ordered reference (`h3_reference_kind`, `h3.h:30-34`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
}

/// One ordered reference the request carries, bytes still owned. The renderer
/// writes them to files; the engine reads those files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceMedia {
    pub kind: ReferenceKind,
    pub media_type: String,
    pub bytes: Vec<u8>,
    /// A video reference conditions with its own soundtrack (`--ref-video`,
    /// not `--ref-silent-video`).
    pub include_audio: bool,
}

impl ReferenceMedia {
    /// The file extension the temp copy gets. The media readers probe content,
    /// so the extension only names the file; an unknown type is refused before
    /// anything is written.
    pub fn extension(&self) -> Option<&'static str> {
        match self.media_type.as_str() {
            "image/png" => Some("png"),
            "image/jpeg" | "image/jpg" => Some("jpg"),
            "image/webp" => Some("webp"),
            "video/mp4" => Some("mp4"),
            "video/quicktime" => Some("mov"),
            "video/webm" => Some("webm"),
            "audio/mpeg" | "audio/mp3" => Some("mp3"),
            "audio/wav" | "audio/x-wav" => Some("wav"),
            "audio/ogg" => Some("ogg"),
            "audio/mp4" | "audio/m4a" => Some("m4a"),
            _ => None,
        }
    }
}

/// One create, validated and mapped onto the engine's request. `size` is both
/// the model canvas and the delivered frame: this contract has no separate
/// delivery frame, so the file is the canvas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub task: Task,
    pub prompt: String,
    pub size: VideoSize,
    pub frames: u32,
    pub seconds: u32,
    pub seed: u64,
    pub first_frame: Option<AnchorImage>,
    pub last_frame: Option<AnchorImage>,
    /// Reference media in request order; empty for t2va and fl2va.
    pub references: Vec<ReferenceMedia>,
}

/// Parse and validate a create body.
pub fn parse_create(body: &[u8]) -> Result<CreateBody, ApiError> {
    let root: Value = serde_json::from_slice(body).map_err(|error| {
        ApiError::bad_request(
            "invalid_json",
            None,
            format!("the body is not JSON: {error}"),
        )
    })?;
    let object = root.as_object().ok_or_else(|| {
        ApiError::bad_request("invalid_json", None, "the body must be a JSON object")
    })?;
    refuse_unknown_fields(object, &CREATE_FIELDS, "the body")?;

    let task = required_str(object, "task", "invalid_task")?;
    let task = Task::parse(task).ok_or_else(|| {
        ApiError::bad_request(
            "invalid_task",
            Some("task"),
            format!("task must be t2va, fl2va or ref2va, not {task}"),
        )
    })?;

    let prompt = required_str(object, "prompt", "invalid_prompt")?.to_string();
    if prompt.trim().is_empty() {
        return Err(ApiError::bad_request(
            "invalid_prompt",
            Some("prompt"),
            "prompt must not be empty",
        ));
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "invalid_prompt",
            Some("prompt"),
            format!("prompt must be at most {MAX_PROMPT_CHARS} characters"),
        ));
    }

    let conditions = match object.get("conditions") {
        Some(Value::Array(items)) => {
            // Two keyframes cover the clip's edges; ordered references go to
            // the reference's own cap (`h3.c:588`).
            let limit = match task {
                Task::Ref2va => MAX_REFERENCES,
                Task::T2va | Task::Fl2va => MAX_CONDITIONS,
            };
            if items.len() > limit {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!("a {} clip carries at most {limit} conditions", task.name()),
                ));
            }
            items
                .iter()
                .map(parse_condition)
                .collect::<Result<Vec<_>, _>>()?
        }
        Some(_) => {
            return Err(ApiError::bad_request(
                "invalid_condition",
                Some("conditions"),
                "conditions must be an array",
            ))
        }
        None => Vec::new(),
    };

    let target = object
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::bad_request(
                "invalid_target",
                Some("target"),
                "target must be an object with short_edge, aspect_ratio and duration_seconds",
            )
        })?;
    let target = parse_target(target)?;

    let seed = object.get("seed").and_then(Value::as_u64).ok_or_else(|| {
        ApiError::bad_request(
            "invalid_seed",
            Some("seed"),
            "seed must be a non-negative integer",
        )
    })?;

    Ok(CreateBody {
        task,
        prompt,
        conditions,
        target,
        seed,
    })
}

impl CreateBody {
    /// Validate against what this host can render and map onto the engine.
    ///
    /// `aspect_ratio: auto` over a video reference needs the media's pixel
    /// size, which only a file can be probed for; a caller that has written
    /// the bytes to disk plugs that in through [`CreateBody::plan_with`].
    pub fn plan(&self) -> Result<Plan, ApiError> {
        self.plan_with(&|media| image_size(&media.bytes))
    }

    /// The same validation with a caller-supplied reference probe. It is asked
    /// only for the first visual reference when the ratio is `auto`.
    pub fn plan_with(
        &self,
        probe: &dyn Fn(&ReferenceMedia) -> Option<(u32, u32)>,
    ) -> Result<Plan, ApiError> {
        if self.target.short_edge != SHORT_EDGE {
            return Err(ApiError::bad_request(
                "unsupported_short_edge",
                Some("target.short_edge"),
                format!(
                    "short_edge {} is not served: the open checkpoint renders at {SHORT_EDGE}",
                    self.target.short_edge
                ),
            ));
        }

        if !(MIN_SECONDS..=MAX_SECONDS).contains(&self.target.duration_seconds) {
            return Err(ApiError::bad_request(
                "invalid_duration",
                Some("target.duration_seconds"),
                format!(
                    "duration_seconds must be {MIN_SECONDS} through {MAX_SECONDS}, not {}",
                    self.target.duration_seconds
                ),
            ));
        }

        // A request carries keyframes or ordered references, never both
        // (`h3.c:597-600`).
        let (first_frame, last_frame, references) = match self.task {
            Task::T2va | Task::Fl2va => {
                let (first, last) = self.anchors()?;
                (first, last, Vec::new())
            }
            Task::Ref2va => (None, None, self.references()?),
        };

        let (width, height) = self.canvas(
            first_frame.as_ref().or(last_frame.as_ref()),
            &references,
            probe,
        )?;

        let frames = align_frames(frames_from_seconds(f64::from(self.target.duration_seconds)));
        if !(H3_MIN_FRAMES..=H3_MAX_FRAMES).contains(&frames) {
            return Err(ApiError::bad_request(
                "invalid_duration",
                Some("target.duration_seconds"),
                format!(
                    "{} s aligns to {frames} frames, outside the engine's range",
                    self.target.duration_seconds
                ),
            ));
        }

        Ok(Plan {
            task: self.task,
            prompt: self.prompt.clone(),
            size: VideoSize::new(width as u32, height as u32),
            frames,
            seconds: self.target.duration_seconds,
            seed: self.seed,
            first_frame,
            last_frame,
            references,
        })
    }

    /// The ordered references a ref2va request carries, in request order.
    fn references(&self) -> Result<Vec<ReferenceMedia>, ApiError> {
        if self.conditions.len() > MAX_REFERENCES {
            return Err(ApiError::bad_request(
                "invalid_condition",
                Some("conditions"),
                format!("ref2va carries at most {MAX_REFERENCES} references"),
            ));
        }

        let mut references = Vec::with_capacity(self.conditions.len());
        for condition in &self.conditions {
            if condition.role.is_keyframe() {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!(
                        "ref2va carries reference media, not a {} anchor; \
                         use fl2va for frame anchors",
                        condition.role.name()
                    ),
                ));
            }
            if condition.frame_index.is_some() {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    "frame_index belongs to a keyframe, not a reference",
                ));
            }

            let kind = match (condition.role, condition.kind) {
                (Role::Reference, ConditionType::Image) => ReferenceKind::Image,
                (Role::Reference | Role::ReferenceVideo, ConditionType::Video) => {
                    ReferenceKind::Video
                }
                (Role::Reference | Role::ReferenceAudio, ConditionType::Audio) => {
                    ReferenceKind::Audio
                }
                (Role::BaseVideo, _) => {
                    return Err(ApiError::bad_request(
                        "unsupported_condition",
                        Some("conditions"),
                        "base_video is the platform's source-video edit role, which this host \
                         does not carry; reference, reference_video and reference_audio are served",
                    ))
                }
                _ => {
                    return Err(ApiError::bad_request(
                        "invalid_condition",
                        Some("conditions"),
                        format!(
                            "a {} role does not carry {}",
                            condition.role.name(),
                            kind_name(condition.kind)
                        ),
                    ))
                }
            };

            let prefix = match kind {
                ReferenceKind::Image => "image/",
                ReferenceKind::Video => "video/",
                ReferenceKind::Audio => "audio/",
            };
            if !condition.media_type.starts_with(prefix) {
                return Err(ApiError::bad_request(
                    "invalid_uri",
                    Some("conditions"),
                    format!(
                        "a {} reference needs a {prefix} data URI, not {}",
                        match kind {
                            ReferenceKind::Image => "image",
                            ReferenceKind::Video => "video",
                            ReferenceKind::Audio => "audio",
                        },
                        condition.media_type
                    ),
                ));
            }

            references.push(ReferenceMedia {
                kind,
                media_type: condition.media_type.clone(),
                bytes: condition.bytes.clone(),
                include_audio: kind == ReferenceKind::Video,
            });
        }

        if references.is_empty() {
            return Err(ApiError::bad_request(
                "invalid_condition",
                Some("conditions"),
                "ref2va needs at least one image or video reference",
            ));
        }
        if references
            .iter()
            .all(|media| media.kind == ReferenceKind::Audio)
        {
            return Err(ApiError::bad_request(
                "invalid_condition",
                Some("conditions"),
                "ordered audio needs an image or video reference",
            ));
        }

        Ok(references)
    }

    /// Exactly one anchor per `frame_index`, and only a keyframe role.
    fn anchors(&self) -> Result<(Option<AnchorImage>, Option<AnchorImage>), ApiError> {
        let mut first: Option<AnchorImage> = None;
        let mut last: Option<AnchorImage> = None;

        for condition in &self.conditions {
            if !condition.role.is_keyframe() {
                return Err(reference_refusal(condition.role));
            }
            if condition.kind != ConditionType::Image {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!(
                        "a {} role conditions an image, not {}",
                        condition.role.name(),
                        kind_name(condition.kind)
                    ),
                ));
            }
            let index = condition.frame_index.ok_or_else(|| {
                ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!(
                        "a {} condition carries frame_index: 0 for the first frame, 1 for the last",
                        condition.role.name()
                    ),
                )
            })?;
            let slot = if index == 0 {
                &mut first
            } else if index == 1 {
                &mut last
            } else {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!("frame_index must be 0 (first frame) or 1 (last frame), not {index}"),
                ));
            };
            if slot.is_some() {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    format!("frame_index {index} is carried twice"),
                ));
            }
            *slot = Some(AnchorImage {
                media_type: condition.media_type.clone(),
                bytes: condition.bytes.clone(),
            });
        }

        match self.task {
            Task::T2va if first.is_some() || last.is_some() => {
                return Err(ApiError::bad_request(
                    "invalid_condition",
                    Some("conditions"),
                    "t2va is text to video and carries no keyframe; use fl2va",
                ))
            }
            Task::Fl2va if first.is_none() && last.is_none() => return Err(ApiError::bad_request(
                "invalid_condition",
                Some("conditions"),
                "fl2va needs at least one keyframe: frame_index 0 (first frame) or 1 (last frame)",
            )),
            _ => {}
        }

        Ok((first, last))
    }

    /// The canvas: the named ratio's, or the media's own aspect.
    ///
    /// `auto` uses the FL2VA keyframe or the first visual Ref2VA reference.
    fn canvas(
        &self,
        anchor: Option<&AnchorImage>,
        references: &[ReferenceMedia],
        probe: &dyn Fn(&ReferenceMedia) -> Option<(u32, u32)>,
    ) -> Result<(i32, i32), ApiError> {
        let aspect = self.target.aspect_ratio.as_str();
        if aspect == AUTO_RATIO {
            let (width, height) = match anchor {
                Some(anchor) => image_size(&anchor.bytes).ok_or_else(|| {
                    ApiError::bad_request(
                        "invalid_condition",
                        Some("conditions"),
                        "cannot read the keyframe's pixel size; auto needs a PNG, JPEG or WebP image",
                    )
                })?,
                None => {
                    let media = references
                        .iter()
                        .find(|media| media.kind != ReferenceKind::Audio)
                        .ok_or_else(|| {
                            ApiError::bad_request(
                                "invalid_aspect_ratio",
                                Some("target.aspect_ratio"),
                                "aspect_ratio auto takes the aspect from the keyframe, and there is none",
                            )
                        })?;
                    probe(media).ok_or_else(|| {
                        ApiError::bad_request(
                            "invalid_aspect_ratio",
                            Some("target.aspect_ratio"),
                            "aspect_ratio auto takes the aspect from the first image or video \
                             reference, and its pixel size cannot be read",
                        )
                    })?
                }
            };
            let canvas = presets::canvas_for(width, height);
            check_stretch(
                canvas,
                f64::from(width.max(height)) / f64::from(width.min(height)),
                AUTO_RATIO,
            )?;
            return Ok(canvas);
        }

        let ratio = Ratio::parse(aspect).ok_or_else(|| {
            ApiError::bad_request(
                "invalid_aspect_ratio",
                Some("target.aspect_ratio"),
                format!("aspect_ratio must be auto, 16:9, 9:16, 4:3, 3:4 or 1:1, not {aspect}"),
            )
        })?;
        let (width, height) = ratio.parts();
        let canvas = presets::tier_canvas(ratio);
        // Compare long edge over short edge on both sides: the ratio name is
        // stated as width/height, and 9:16 is tall.
        check_stretch(
            canvas,
            f64::from(width.max(height)) / f64::from(width.min(height)),
            ratio.name(),
        )?;
        Ok(canvas)
    }
}

/// Refuse a canvas whose aspect is not the requested one beyond the stretch
/// `presets::MAX_DISTORTION` bounds. The one case at this tier is 21:9, whose
/// 1792 px width the released area refuses.
fn check_stretch(canvas: (i32, i32), want: f64, label: &str) -> Result<(), ApiError> {
    let (long, short) = (canvas.0.max(canvas.1), canvas.0.min(canvas.1));
    let got = f64::from(long) / f64::from(short);
    let distortion = (got / want).max(want / got);
    if distortion <= presets::MAX_DISTORTION {
        return Ok(());
    }

    Err(ApiError::bad_request(
        "invalid_aspect_ratio",
        Some("target.aspect_ratio"),
        format!(
            "{label} needs {:.0}x{short} at this tier, which exceeds the {}x{} released area; \
             the widest canvas is {long}x{short}, a {distortion:.3}x stretch. Use 16:9",
            want * f64::from(short),
            MAX_PIXELS / i64::from(presets::TIER_SHORT_EDGE),
            presets::TIER_SHORT_EDGE,
        ),
    ))
}

fn reference_refusal(role: Role) -> ApiError {
    ApiError::bad_request(
        "unsupported_condition",
        Some("conditions"),
        format!(
            "the {} role belongs to ref2va; a t2va or fl2va clip carries only keyframes \
             (keyframe or first_frame with frame_index) and no references",
            role.name()
        ),
    )
}

fn kind_name(kind: ConditionType) -> &'static str {
    match kind {
        ConditionType::Image => "an image",
        ConditionType::Audio => "an audio clip",
        ConditionType::Video => "a video clip",
    }
}

fn parse_target(target: &Map<String, Value>) -> Result<Target, ApiError> {
    refuse_unknown_fields(
        target,
        &["short_edge", "aspect_ratio", "duration_seconds"],
        "target",
    )?;
    let short_edge = target
        .get("short_edge")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            ApiError::bad_request(
                "invalid_target",
                Some("target.short_edge"),
                "short_edge must be a positive integer",
            )
        })?;
    let aspect_ratio = target
        .get("aspect_ratio")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::bad_request(
                "invalid_target",
                Some("target.aspect_ratio"),
                "aspect_ratio must be a string: auto or a ratio name",
            )
        })?
        .to_string();
    let duration_seconds = target
        .get("duration_seconds")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            ApiError::bad_request(
                "invalid_target",
                Some("target.duration_seconds"),
                "duration_seconds must be a positive integer",
            )
        })?;

    Ok(Target {
        short_edge,
        aspect_ratio,
        duration_seconds,
    })
}

fn parse_condition(value: &Value) -> Result<Condition, ApiError> {
    let object = value.as_object().ok_or_else(|| {
        ApiError::bad_request(
            "invalid_condition",
            Some("conditions"),
            "a condition must be an object",
        )
    })?;
    refuse_unknown_fields(
        object,
        &["type", "uri", "role", "frame_index"],
        "a condition",
    )?;

    let kind = required_str(object, "type", "invalid_condition")?;
    let kind = ConditionType::parse(kind).ok_or_else(|| {
        ApiError::bad_request(
            "invalid_condition",
            Some("conditions"),
            format!("a condition type must be image, audio or video, not {kind}"),
        )
    })?;
    let role = required_str(object, "role", "invalid_condition")?;
    let role = Role::parse(role).ok_or_else(|| {
        ApiError::bad_request(
            "invalid_condition",
            Some("conditions"),
            format!(
                "a condition role must be keyframe, first_frame, base_video, reference, \
                 reference_video or reference_audio, not {role}"
            ),
        )
    })?;
    let frame_index = match object.get("frame_index") {
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    ApiError::bad_request(
                        "invalid_condition",
                        Some("conditions"),
                        "frame_index must be a non-negative integer",
                    )
                })?,
        ),
        None => None,
    };
    let uri = required_str(object, "uri", "invalid_uri")?;
    let (media_type, bytes) = decode_data_uri(uri)?;
    if bytes.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_uri",
            Some("conditions"),
            "the data URI carries no bytes",
        ));
    }
    // Only an image is capped here. A keyframe and a reference image are one
    // picture each; a video or audio reference is bounded by the request body
    // (`MAX_BODY`), whose base64 inflates by 4/3.
    if kind == ConditionType::Image && bytes.len() > MAX_ANCHOR_BYTES {
        return Err(ApiError::bad_request(
            "invalid_uri",
            Some("conditions"),
            format!("the decoded image exceeds {MAX_ANCHOR_BYTES} bytes"),
        ));
    }

    Ok(Condition {
        kind,
        role,
        frame_index,
        media_type,
        bytes,
    })
}

fn required_str<'a>(
    object: &'a Map<String, Value>,
    name: &'static str,
    code: &'static str,
) -> Result<&'a str, ApiError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request(code, Some(name), format!("{name} must be a string")))
}

/// Unknown fields are refused, not ignored: a typo in a local contract should
/// fail loudly rather than silently render something else.
fn refuse_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    what: &str,
) -> Result<(), ApiError> {
    for name in object.keys() {
        if !allowed.contains(&name.as_str()) {
            return Err(ApiError::bad_request(
                "unknown_field",
                None,
                format!("{what} carries {name}, which this contract does not define"),
            ));
        }
    }
    Ok(())
}

/// `data:<media-type>;base64,<payload>`. The scripts' https URIs are refused
/// here with a reason instead of being fetched.
fn decode_data_uri(uri: &str) -> Result<(String, Vec<u8>), ApiError> {
    if uri.starts_with("https://") || uri.starts_with("http://") {
        return Err(ApiError::bad_request(
            "unsupported_uri",
            Some("conditions"),
            "a condition uri must be a data URI; this host does not fetch URLs from a request body",
        ));
    }
    let rest = uri.strip_prefix("data:").ok_or_else(|| {
        ApiError::bad_request(
            "invalid_uri",
            Some("conditions"),
            "a condition uri must be a data URI",
        )
    })?;
    let (media_type, payload) = rest.split_once(";base64,").ok_or_else(|| {
        ApiError::bad_request(
            "invalid_uri",
            Some("conditions"),
            "a data URI must be data:<media-type>;base64,<payload>",
        )
    })?;
    let bytes = base64(payload).map_err(|detail| {
        ApiError::bad_request(
            "invalid_uri",
            Some("conditions"),
            format!("the data URI payload is not base64: {detail}"),
        )
    })?;
    Ok((media_type.to_ascii_lowercase(), bytes))
}

/// Standard-alphabet base64 with the trailing bits required to be zero, so two
/// encodings never name different bytes.
fn base64(source: &str) -> Result<Vec<u8>, String> {
    let source = source.as_bytes();
    if !source.len().is_multiple_of(4) {
        return Err("length is not a multiple of four".to_string());
    }
    let padding = usize::from(source.last() == Some(&b'='))
        + usize::from(source.len() > 1 && source[source.len() - 2] == b'=');
    let decoded = source.len() / 4 * 3 - padding;
    let mut out = Vec::with_capacity(decoded);

    for (index, chunk) in source.as_chunks::<4>().0.iter().enumerate() {
        let last = index + 1 == source.len() / 4;
        let a = value(chunk[0]);
        let b = value(chunk[1]);
        let c = (chunk[2] != b'=').then(|| value(chunk[2])).flatten();
        let d = (chunk[3] != b'=').then(|| value(chunk[3])).flatten();
        let malformed = a.is_none()
            || b.is_none()
            || (chunk[2] != b'=' && c.is_none())
            || (chunk[3] != b'=' && d.is_none())
            || (!last && (c.is_none() || d.is_none()))
            || (chunk[2] == b'=' && chunk[3] != b'=')
            || (chunk[2] == b'=' && b.unwrap() & 15 != 0)
            || (chunk[3] == b'=' && c.is_some_and(|value| value & 3 != 0));
        if malformed {
            return Err("payload is malformed".to_string());
        }

        let value = a.unwrap() << 18 | b.unwrap() << 12 | c.unwrap_or(0) << 6 | d.unwrap_or(0);
        for shift in [16u32, 8, 0] {
            if out.len() < decoded {
                out.push((value >> shift) as u8);
            }
        }
    }
    if out.len() != decoded {
        return Err("payload has the wrong length".to_string());
    }
    Ok(out)
}

fn value(character: u8) -> Option<u32> {
    match character {
        b'A'..=b'Z' => Some((character - b'A') as u32),
        b'a'..=b'z' => Some((26 + character - b'a') as u32),
        b'0'..=b'9' => Some((52 + character - b'0') as u32),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// The anchor's pixel size, from the container header alone:
///
/// - PNG: signature, then IHDR as the first chunk (PNG spec 11.2.2), width
///   and height as big-endian u32 at bounds 16 and 20;
/// - JPEG: SOFn segments carry height then width (ITU T.81 B.2.2); SOF0..15
///   except DHT (C4), JPG (C8) and DAC (CC);
/// - WebP: VP8X canvas size is 24-bit minus one, VP8's is 14-bit after the
///   9D 01 2A start code, VP8L packs both as 14-bit fields (WebP container
///   spec, "Simple File Format").
pub fn image_size(bytes: &[u8]) -> Option<(u32, u32)> {
    png_size(bytes)
        .or_else(|| jpeg_size(bytes))
        .or_else(|| webp_size(bytes))
}

fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() < 24 || bytes[..8] != SIGNATURE || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn jpeg_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut index = 2;
    while index + 3 < bytes.len() {
        if bytes[index] != 0xFF {
            return None;
        }
        // A run of 0xFF bytes is legal padding before a marker.
        let mut marker = bytes[index + 1];
        while marker == 0xFF && index + 2 < bytes.len() {
            index += 1;
            marker = bytes[index + 1];
        }
        match marker {
            0xD8 | 0x01 => {
                index += 2;
                continue;
            }
            0xD9 | 0xDA => return None,
            _ => {}
        }
        let length = usize::from(u16::from_be_bytes([bytes[index + 2], bytes[index + 3]]));
        if length < 2 || index + 2 + length > bytes.len() {
            return None;
        }
        let is_sof = (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
        if is_sof {
            if index + 9 > bytes.len() {
                return None;
            }
            let height = u32::from(u16::from_be_bytes([bytes[index + 5], bytes[index + 6]]));
            let width = u32::from(u16::from_be_bytes([bytes[index + 7], bytes[index + 8]]));
            return (width > 0 && height > 0).then_some((width, height));
        }
        index += 2 + length;
    }
    None
}

fn webp_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 30 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    match &bytes[12..16] {
        b"VP8X" => {
            let width =
                1 + (u32::from(bytes[24]) | u32::from(bytes[25]) << 8 | u32::from(bytes[26]) << 16);
            let height =
                1 + (u32::from(bytes[27]) | u32::from(bytes[28]) << 8 | u32::from(bytes[29]) << 16);
            (width > 0 && height > 0).then_some((width, height))
        }
        b"VP8 " => {
            if bytes[23..26] != [0x9D, 0x01, 0x2A] {
                return None;
            }
            let width = u32::from(u16::from_le_bytes([bytes[26], bytes[27]]) & 0x3FFF);
            let height = u32::from(u16::from_le_bytes([bytes[28], bytes[29]]) & 0x3FFF);
            (width > 0 && height > 0).then_some((width, height))
        }
        b"VP8L" => {
            if bytes[20] != 0x2F {
                return None;
            }
            let packed = u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]);
            let width = (packed & 0x3FFF) + 1;
            let height = ((packed >> 14) & 0x3FFF) + 1;
            (width > 0 && height > 0).then_some((width, height))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        out.extend_from_slice(&13u32.to_be_bytes());
        out.extend_from_slice(b"IHDR");
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&[8, 6, 0, 0, 0]);
        out
    }

    fn jpeg_header(width: u16, height: u16) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        out
    }

    fn webp_vp8x(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&30u32.to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(b"VP8X");
        out.extend_from_slice(&10u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        let (w, h) = (width - 1, height - 1);
        out.extend_from_slice(&[w as u8, (w >> 8) as u8, (w >> 16) as u8]);
        out.extend_from_slice(&[h as u8, (h >> 8) as u8, (h >> 16) as u8]);
        out
    }

    fn data_uri(media_type: &str, bytes: &[u8]) -> String {
        format!("data:{media_type};base64,{}", encode(bytes))
    }

    fn encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                chunk.get(1).copied().unwrap_or(0),
                chunk.get(2).copied().unwrap_or(0),
            ];
            let value = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
            out.push(ALPHABET[(value >> 18) as usize & 63] as char);
            out.push(ALPHABET[(value >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(value >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[value as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    fn keyframe(index: u32, width: u32, height: u32) -> Value {
        json!({
            "type": "image",
            "uri": data_uri("image/png", &png_header(width, height)),
            "role": "keyframe",
            "frame_index": index,
        })
    }

    fn body(task: &str, conditions: Value, ratio: &str, seconds: u32) -> Value {
        json!({
            "task": task,
            "prompt": "a red cube",
            "conditions": conditions,
            "target": { "short_edge": 768, "aspect_ratio": ratio, "duration_seconds": seconds },
            "seed": 7,
        })
    }

    fn planned(value: Value) -> Result<Plan, ApiError> {
        parse_create(value.to_string().as_bytes())?.plan()
    }

    #[test]
    fn the_five_fields_are_the_contract() {
        let root = body("t2va", json!([]), "16:9", 8);
        let parsed = parse_create(root.to_string().as_bytes()).unwrap();
        assert_eq!(parsed.task, Task::T2va);
        assert_eq!(parsed.seed, 7);
        assert_eq!(parsed.target.short_edge, 768);

        // A sixth field is refused rather than ignored.
        let mut extra = root.clone();
        extra["model"] = json!("MiniMax-H3");
        let error = parse_create(extra.to_string().as_bytes()).unwrap_err();
        assert_eq!(error.code, "unknown_field");
        assert!(error.message.contains("model"));
    }

    #[test]
    fn t2va_renders_the_named_canvas() {
        let plan = planned(body("t2va", json!([]), "16:9", 8)).unwrap();
        assert_eq!((plan.size.width, plan.size.height), (1344, 768));
        assert_eq!(plan.frames, 192);
        assert_eq!(plan.seconds, 8);
        assert!(plan.first_frame.is_none());
        assert!(plan.last_frame.is_none());
    }

    #[test]
    fn every_named_ratio_reaches_its_canvas() {
        for (ratio, want) in [
            ("16:9", (1344, 768)),
            ("9:16", (768, 1344)),
            ("4:3", (1024, 768)),
            ("3:4", (768, 1024)),
            ("1:1", (768, 768)),
        ] {
            let plan = planned(body("t2va", json!([]), ratio, 8)).unwrap();
            assert_eq!((plan.size.width, plan.size.height), want, "{ratio}");
        }
    }

    #[test]
    fn twenty_one_by_nine_is_refused_by_name() {
        let error = planned(body("t2va", json!([]), "21:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_aspect_ratio");
        assert!(error.message.contains("21:9"));
        assert!(error.message.contains("1.333x"));
        assert!(error.message.contains("Use 16:9"));
    }

    #[test]
    fn only_the_768_tier_is_served() {
        let mut root = body("t2va", json!([]), "16:9", 8);
        root["target"]["short_edge"] = json!(480);
        let error = parse_create(root.to_string().as_bytes())
            .unwrap()
            .plan()
            .unwrap_err();
        assert_eq!(error.code, "unsupported_short_edge");
        assert!(error.message.contains("768"));
    }

    #[test]
    fn duration_is_the_documented_range() {
        assert_eq!(
            planned(body("t2va", json!([]), "16:9", 4)).unwrap().frames,
            107
        );
        assert_eq!(
            planned(body("t2va", json!([]), "16:9", 5)).unwrap().frames,
            124
        );
        assert_eq!(
            planned(body("t2va", json!([]), "16:9", 10)).unwrap().frames,
            243
        );
        assert_eq!(
            planned(body("t2va", json!([]), "16:9", 15)).unwrap().frames,
            362
        );

        for seconds in [3, 16] {
            let error = planned(body("t2va", json!([]), "16:9", seconds)).unwrap_err();
            assert_eq!(error.code, "invalid_duration");
        }
    }

    #[test]
    fn fl2va_maps_a_first_frame_and_a_last_frame() {
        let plan = planned(body(
            "fl2va",
            json!([keyframe(0, 1920, 1080), keyframe(1, 1920, 1080)]),
            "auto",
            8,
        ))
        .unwrap();
        assert_eq!((plan.size.width, plan.size.height), (1344, 768));
        assert!(plan.first_frame.is_some());
        assert!(plan.last_frame.is_some());
        assert_eq!(plan.first_frame.unwrap().media_type, "image/png");
    }

    #[test]
    fn auto_takes_the_aspect_from_the_anchor() {
        let plan = planned(body("fl2va", json!([keyframe(0, 1080, 1920)]), "auto", 8)).unwrap();
        assert_eq!((plan.size.width, plan.size.height), (768, 1344));

        let square = planned(body("fl2va", json!([keyframe(0, 1024, 1024)]), "auto", 8)).unwrap();
        assert_eq!((square.size.width, square.size.height), (768, 768));
    }

    #[test]
    fn auto_needs_a_keyframe_or_a_visual_reference() {
        let error = planned(body("t2va", json!([]), "auto", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_aspect_ratio");
        assert!(error.message.contains("there is none"));

        let error = planned(body(
            "ref2va",
            json!([{ "type": "audio", "uri": data_uri("audio/mpeg", b"x"), "role": "reference" }]),
            "auto",
            8,
        ))
        .unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("needs an image or video reference"));
    }

    #[test]
    fn t2va_carries_no_keyframe_and_fl2va_needs_one() {
        let error = planned(body("t2va", json!([keyframe(0, 640, 480)]), "16:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("use fl2va"));

        let error = planned(body("fl2va", json!([]), "16:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("at least one keyframe"));
    }

    #[test]
    fn a_frame_index_must_be_zero_or_one_and_unique() {
        let error = planned(body("fl2va", json!([keyframe(2, 640, 480)]), "16:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_condition");

        let error = planned(body(
            "fl2va",
            json!([keyframe(0, 640, 480), keyframe(0, 640, 480)]),
            "16:9",
            8,
        ))
        .unwrap_err();
        assert!(error.message.contains("carried twice"));
    }

    /// The official local Ref2VA body: a video reference, an audio reference,
    /// both `role: reference`, at 5 seconds.
    #[test]
    fn ref2va_maps_references_in_request_order() {
        let plan = planned(body(
            "ref2va",
            json!([
                { "type": "video", "uri": data_uri("video/mp4", b"video"), "role": "reference" },
                { "type": "audio", "uri": data_uri("audio/mpeg", b"audio"), "role": "reference" }
            ]),
            "16:9",
            5,
        ))
        .unwrap();

        assert_eq!(plan.references.len(), 2);
        assert_eq!(plan.references[0].kind, ReferenceKind::Video);
        assert!(plan.references[0].include_audio);
        assert_eq!(plan.references[1].kind, ReferenceKind::Audio);
        assert_eq!(plan.frames, 124);
        assert!(plan.first_frame.is_none() && plan.last_frame.is_none());
    }

    /// `reference_video` and `reference_audio` are the same roles under their
    /// platform names; an image reference is `role: reference`.
    #[test]
    fn ref2va_accepts_the_platform_role_names() {
        for (kind, role, media) in [
            ("image", "reference", "image/png"),
            ("video", "reference_video", "video/mp4"),
            ("audio", "reference_audio", "audio/wav"),
        ] {
            let uri = match kind {
                "image" => data_uri(media, &png_header(64, 64)),
                _ => data_uri(media, b"x"),
            };
            let plan = planned(body(
                "ref2va",
                json!([
                    { "type": "image", "uri": data_uri("image/png", &png_header(64, 64)), "role": "reference" },
                    { "type": kind, "uri": uri, "role": role }
                ]),
                "16:9",
                4,
            ))
            .unwrap();
            assert_eq!(plan.references.len(), 2);
        }
    }

    #[test]
    fn ref2va_refuses_roles_it_cannot_carry() {
        // `base_video` is the platform's source-video edit role.
        let error = planned(body(
            "ref2va",
            json!([
                { "type": "image", "uri": data_uri("image/png", &png_header(64, 64)), "role": "reference" },
                { "type": "video", "uri": data_uri("video/mp4", b"x"), "role": "base_video" }
            ]),
            "16:9",
            8,
        ))
        .unwrap_err();
        assert_eq!(error.code, "unsupported_condition");
        assert!(error.message.contains("base_video"));

        // A keyframe in ref2va is refused by name.
        let error = planned(body("ref2va", json!([keyframe(0, 640, 480)]), "16:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("not a"));

        // Audio alone conditions nothing.
        let error = planned(body(
            "ref2va",
            json!([{ "type": "audio", "uri": data_uri("audio/mpeg", b"x"), "role": "reference" }]),
            "16:9",
            8,
        ))
        .unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("needs an image or video reference"));
    }

    /// `h3.c:588`: at most twelve ordered references.
    #[test]
    fn ref2va_stops_at_twelve_references() {
        let mut conditions = vec![
            json!({ "type": "image", "uri": data_uri("image/png", &png_header(64, 64)), "role": "reference" }),
        ];
        for _ in 0..MAX_REFERENCES {
            conditions.push(
                json!({ "type": "audio", "uri": data_uri("audio/mpeg", b"x"), "role": "reference" }),
            );
        }
        let error = planned(body("ref2va", json!(conditions), "16:9", 8)).unwrap_err();
        assert_eq!(error.code, "invalid_condition");
        assert!(error.message.contains("at most 12"));
    }

    /// A reference's data URI must declare the class its condition names.
    #[test]
    fn a_reference_media_type_matches_its_condition() {
        let error = planned(body(
            "ref2va",
            json!([{ "type": "video", "uri": data_uri("image/png", &png_header(64, 64)), "role": "reference" }]),
            "16:9",
            8,
        ))
        .unwrap_err();
        assert_eq!(error.code, "invalid_uri");
        assert!(error.message.contains("video/"));
    }

    /// In fl2va, a reference role still names the path it would need.
    #[test]
    fn fl2va_refuses_reference_roles() {
        let error = planned(
            body(
                "fl2va",
                json!([{ "type": "image", "uri": data_uri("image/png", &png_header(64, 64)), "role": "reference" }]),
                "16:9",
                8,
            ),
        )
        .unwrap_err();
        assert_eq!(error.code, "unsupported_condition");
        assert!(error.message.contains("reference"));
    }

    #[test]
    fn a_condition_uri_is_a_data_uri() {
        let error = planned(
            body(
                "fl2va",
                json!([{ "type": "image", "uri": "https://cdn.hailuoai.com/x.png", "role": "keyframe", "frame_index": 0 }]),
                "16:9",
                8,
            ),
        )
        .unwrap_err();
        assert_eq!(error.code, "unsupported_uri");
        assert!(error.message.contains("does not fetch"));

        let error = planned(body(
            "fl2va",
            json!([{ "type": "image", "uri": "/tmp/x.png", "role": "keyframe", "frame_index": 0 }]),
            "16:9",
            8,
        ))
        .unwrap_err();
        assert_eq!(error.code, "invalid_uri");
    }

    #[test]
    fn the_anchor_size_reads_three_containers() {
        assert_eq!(image_size(&png_header(1344, 768)), Some((1344, 768)));
        assert_eq!(image_size(&jpeg_header(1920, 1080)), Some((1920, 1080)));
        assert_eq!(image_size(&webp_vp8x(1080, 1920)), Some((1080, 1920)));
        assert_eq!(image_size(b"not an image"), None);
        assert_eq!(image_size(&png_header(0, 0)), None);
    }

    #[test]
    fn the_body_must_be_json_with_a_task() {
        let error = parse_create(b"not json").unwrap_err();
        assert_eq!(error.code, "invalid_json");

        let mut root = body("t2va", json!([]), "16:9", 8);
        root["task"] = json!("r2v");
        let error = parse_create(root.to_string().as_bytes()).unwrap_err();
        assert_eq!(error.code, "invalid_task");

        let mut root = body("t2va", json!([]), "16:9", 8);
        root["prompt"] = json!("   ");
        let error = parse_create(root.to_string().as_bytes()).unwrap_err();
        assert_eq!(error.code, "invalid_prompt");
    }

    #[test]
    fn a_missing_field_is_named() {
        let mut root = body("t2va", json!([]), "16:9", 8);
        root.as_object_mut().unwrap().remove("seed");
        let error = parse_create(root.to_string().as_bytes()).unwrap_err();
        assert_eq!(error.code, "invalid_seed");
        assert!(error.message.contains("seed"));

        // A missing field carries its own code, so a client can tell which
        // one it forgot without reading the sentence.
        for (name, code) in [("task", "invalid_task"), ("prompt", "invalid_prompt")] {
            let mut root = body("t2va", json!([]), "16:9", 8);
            root.as_object_mut().unwrap().remove(name);
            let error = parse_create(root.to_string().as_bytes()).unwrap_err();
            assert_eq!(error.code, code, "{name}");
            assert_eq!(error.param, Some(name));
        }
    }
}
