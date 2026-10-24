//! Blocking OpenAI Videos API surface for the additive h3 family.
//!
//! This module owns HTTP only: the wire types and the job store live in
//! `ds4_core::minimax_h3::video_api`, and generation is delegated to an
//! operator-configured [`Renderer`]. It is a separate binary
//! (`h3-server-rs`), so no existing family's route table changes.
//!
//! Wire reference (verified 2026-10-24):
//! <https://developers.openai.com/api/reference/resources/videos>. Routes:
//! `POST /v1/videos`, `GET /v1/videos`, `GET /v1/videos/{id}`,
//! `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}`, and the Models
//! surface `GET /v1/models`.
//!
//! Security boundary: the request body supplies a prompt, model id, size and
//! duration only. The executable, its fixed arguments, the output directory
//! and the working directory are fixed at startup, and every request value is
//! passed as its own argv entry with no shell. The argv spells the reference
//! engine's CLI (`-p`, `-o`, `--width`, `--height`, `--seconds`), so the
//! oracle binary runs without an adapter.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ds4_core::minimax_h3::video_api::{
    parse_request, ApiError, ImageRef, ListQuery, RenderSpec, VideoError, VideoRequest,
    VideoStatus, VideoStore,
};
use serde_json::json;

use crate::error::http_head;
use crate::http::{content_length, header_chunked, header_end, MAX_BODY, MAX_HEADER};

/// Stderr kept for a failed renderer, matching the reference's practice of
/// surfacing the tail of the failing program's output.
pub const STDERR_TAIL_BYTES: usize = 4096;

/// How often the renderer wait loop checks for exit or timeout.
pub const RENDER_POLL: Duration = Duration::from_millis(20);

/// Bounds for form decoding. The whole body is already capped by
/// `http::MAX_BODY`; these stop pathological part lists and headers.
pub const MAX_FORM_FIELDS: usize = 64;
pub const MAX_PART_HEADER_BYTES: usize = 8 * 1024;

/// RFC 2046 caps a boundary at 70 characters.
pub const MAX_BOUNDARY_BYTES: usize = 70;

/// Which downloadable asset `GET /v1/videos/{id}/content` may return. The
/// reference lists these three; h3 produces only the MP4.
pub const CONTENT_TYPE_MP4: &str = "video/mp4";
pub const CONTENT_TYPE_JSON: &str = "application/json";

/// A renderer either accepts work or the bridge returns 503 on create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererState {
    Ready,
    Disabled,
}

/// A failed render, before it becomes a `VideoCreateError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderError {
    pub code: &'static str,
    pub message: String,
}

impl RenderError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Generation boundary. Implementations must be thread-safe: the server runs
/// one job thread per submitted video.
pub trait Renderer: Send + Sync {
    fn state(&self) -> RendererState;
    fn render(&self, spec: &RenderSpec, output: &Path) -> Result<(), RenderError>;
}

pub type SharedRenderer = Arc<dyn Renderer>;

/// No renderer configured: create answers 503 without touching the store.
pub struct DisabledRenderer;

impl Renderer for DisabledRenderer {
    fn state(&self) -> RendererState {
        RendererState::Disabled
    }

    fn render(&self, _spec: &RenderSpec, _output: &Path) -> Result<(), RenderError> {
        Err(RenderError::new(
            "renderer_disabled",
            "no h3 renderer is configured",
        ))
    }
}

/// Runs an operator-supplied program. The fixed program path and args are
/// chosen at startup; request values are appended as separate argv entries.
///
/// The spellings are the reference engine's own, verified in
/// `/data/h3.c/main.c` at eae5d54: `-p/--prompt` and `-o/--output` are the
/// short forms (`main.c:22-24`, option table `main.c:292-296`, switch
/// `main.c:361-381`), and `--width`, `--height`, `--seconds` are the long
/// options. `--seconds` is passed unchanged; the reference converts it at
/// 24 fps itself (`frames_from_seconds`, `main.c:78-91`), so the bridge never
/// invents a frames value. A fixed `-d <model-dir>` may precede the request
/// arguments.
///
/// `program [fixed args...] -p P --width W --height H --seconds S -o PATH`
pub struct CommandRenderer {
    program: PathBuf,
    fixed_args: Vec<String>,
    timeout: Duration,
}

impl CommandRenderer {
    pub fn new(program: impl Into<PathBuf>, fixed_args: Vec<String>, timeout: Duration) -> Self {
        Self {
            program: program.into(),
            fixed_args,
            timeout,
        }
    }

    /// The complete argv, program first. Request values never alter its
    /// length or positions, so a prompt cannot become an option or a command.
    pub fn argv(&self, spec: &RenderSpec, output: &Path) -> Vec<OsString> {
        let mut argv: Vec<OsString> = Vec::with_capacity(self.fixed_args.len() + 10);
        argv.push(self.program.clone().into_os_string());
        argv.extend(self.fixed_args.iter().map(OsString::from));
        argv.push(OsString::from("-p"));
        argv.push(OsString::from(&spec.prompt));
        argv.push(OsString::from("--width"));
        argv.push(OsString::from(spec.size.width.to_string()));
        argv.push(OsString::from("--height"));
        argv.push(OsString::from(spec.size.height.to_string()));
        argv.push(OsString::from("--seconds"));
        argv.push(OsString::from(&spec.seconds));
        argv.push(OsString::from("-o"));
        argv.push(output.as_os_str().to_os_string());
        argv
    }
}

impl Renderer for CommandRenderer {
    fn state(&self) -> RendererState {
        RendererState::Ready
    }

    fn render(&self, spec: &RenderSpec, output: &Path) -> Result<(), RenderError> {
        let argv = self.argv(spec, output);
        let mut command = Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(dir) = output.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            command.current_dir(dir);
        }
        let mut child = command.spawn().map_err(|error| {
            RenderError::new(
                "renderer_spawn_failed",
                format!(
                    "could not start renderer {}: {error}",
                    self.program.display()
                ),
            )
        })?;

        // Drain stderr on its own thread; a full pipe would block the child
        // and turn every verbose renderer into a timeout.
        let reader = child
            .stderr
            .take()
            .map(|pipe| thread::spawn(move || read_tail(pipe, STDERR_TAIL_BYTES)));

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        break None;
                    }
                    thread::sleep(RENDER_POLL);
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RenderError::new(
                        "renderer_spawn_failed",
                        format!("could not wait for renderer: {error}"),
                    ));
                }
            }
        };
        let stderr = reader
            .and_then(|handle| handle.join().ok())
            .map(|tail| String::from_utf8_lossy(&tail).trim().to_string())
            .unwrap_or_default();
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };

        let Some(status) = status else {
            return Err(RenderError::new(
                "render_timeout",
                format!(
                    "renderer exceeded the {}s timeout{detail}",
                    self.timeout.as_secs().max(1)
                ),
            ));
        };
        if !status.success() {
            let code = status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string());
            return Err(RenderError::new(
                "render_failed",
                format!("renderer exited with status {code}{detail}"),
            ));
        }
        Ok(())
    }
}

fn read_tail(mut stream: impl Read, keep: usize) -> Vec<u8> {
    let mut tail = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                tail.extend_from_slice(&buffer[..count]);
                if tail.len() > keep {
                    let cut = tail.len() - keep;
                    tail.drain(..cut);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    tail
}

/// Parsed HTTP/1.1 request. The query string is kept because the house
/// `HttpRequest` truncates it at `?`, and list/content both need it.
/// `content_type` selects the create-body decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub content_type: Option<String>,
    pub body: Vec<u8>,
}

/// Response before the status line is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Response {
    pub status: u16,
    pub content_type: Option<&'static str>,
    pub body: Vec<u8>,
    pub head_only: bool,
}

impl H3Response {
    pub fn json(status: u16, body: String) -> Self {
        Self {
            status,
            content_type: Some(CONTENT_TYPE_JSON),
            body: body.into_bytes(),
            head_only: false,
        }
    }

    pub fn error(error: &ApiError) -> Self {
        Self::json(error.status, error.to_json())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = http_head(
            self.status as i32,
            self.content_type,
            None,
            false,
            self.body.len(),
        )
        .into_bytes();
        if !self.head_only {
            out.extend_from_slice(&self.body);
        }
        out
    }
}

/// The Videos bridge. One instance owns the store, the renderer and the
/// output directory; routes run on per-connection threads.
pub struct VideoServer {
    store: Arc<VideoStore>,
    renderer: SharedRenderer,
    out_dir: PathBuf,
    model: String,
    started_at: u64,
}

impl VideoServer {
    pub fn new(store: Arc<VideoStore>, renderer: SharedRenderer, out_dir: PathBuf) -> Self {
        Self {
            model: store.model().to_string(),
            store,
            renderer,
            out_dir,
            started_at: unix_now(),
        }
    }

    pub fn store(&self) -> &Arc<VideoStore> {
        &self.store
    }

    /// Route one parsed request. Kept free of socket code so the tests can
    /// drive full create/poll/content cycles without a listener.
    pub fn handle(&self, request: &H3Request) -> H3Response {
        match (request.method.as_str(), route(&request.path)) {
            ("POST", Route::Videos) => self.create(request),
            ("GET", Route::Videos) => self.list(&request.query),
            ("GET", Route::Models) => self.models(),
            ("GET", Route::Video(id)) => self.retrieve(&id),
            ("DELETE", Route::Video(id)) => self.delete(&id),
            ("GET" | "HEAD", Route::Content(id)) => {
                self.content(&id, &request.query, request.method.as_str() == "HEAD")
            }
            _ => H3Response::error(&ApiError::not_found(
                "unknown_url",
                None,
                format!("Invalid URL ({} {})", request.method, request.path),
            )),
        }
    }

    fn create(&self, http: &H3Request) -> H3Response {
        if self.renderer.state() == RendererState::Disabled {
            return H3Response::error(&ApiError::unavailable(
                "renderer_disabled",
                "No h3 renderer is configured. Start h3-server-rs with --renderer <program>.",
            ));
        }
        let request = match decode_create_body(http.content_type.as_deref(), &http.body) {
            Ok(request) => request,
            Err(error) => return H3Response::error(&error),
        };
        let (video, spec) = match self.store.submit(&request, unix_now()) {
            Ok(created) => created,
            Err(error) => return H3Response::error(&error),
        };
        let id = video.id.clone();
        let output = self.out_dir.join(format!("{id}.mp4"));
        self.store.start(&id);
        self.spawn_render(id, spec, output);
        H3Response::json(200, video.to_json())
    }

    fn spawn_render(&self, id: String, spec: RenderSpec, output: PathBuf) {
        let store = Arc::clone(&self.store);
        let renderer = Arc::clone(&self.renderer);
        let fail_id = id.clone();
        let spawn = thread::Builder::new()
            .name(format!("h3-render-{id}"))
            .spawn(move || {
                let result = renderer.render(&spec, &output);
                let now = unix_now();
                match result {
                    Ok(()) => match std::fs::metadata(&output) {
                        Ok(meta) if meta.len() > 0 => {
                            store.complete(&id, now, output);
                        }
                        Ok(_) => {
                            store.fail(
                                &id,
                                now,
                                VideoError::new(
                                    "render_output_missing",
                                    format!("renderer wrote an empty file at {}", output.display()),
                                ),
                            );
                        }
                        Err(error) => {
                            store.fail(
                                &id,
                                now,
                                VideoError::new(
                                    "render_output_missing",
                                    format!(
                                        "renderer produced no readable output at {}: {error}",
                                        output.display()
                                    ),
                                ),
                            );
                        }
                    },
                    Err(error) => {
                        store.fail(&id, now, VideoError::new(error.code, error.message));
                    }
                }
            });
        if spawn.is_err() {
            self.store.fail(
                &fail_id,
                unix_now(),
                VideoError::new(
                    "renderer_spawn_failed",
                    "could not start the render thread".to_string(),
                ),
            );
        }
    }

    fn retrieve(&self, id: &str) -> H3Response {
        match self.store.get(id) {
            Some(video) => H3Response::json(200, video.to_json()),
            None => H3Response::error(&unknown_video(id)),
        }
    }

    fn list(&self, query: &str) -> H3Response {
        let query = parse_list_query(query);
        match self.store.list(&query) {
            Ok(page) => H3Response::json(200, page.to_json()),
            Err(error) => H3Response::error(&error),
        }
    }

    fn delete(&self, id: &str) -> H3Response {
        match self.store.delete(id) {
            Ok((deleted, output)) => {
                if let Some(path) = output {
                    // Only a path this server built from the job id reaches
                    // here; a request never supplies a filesystem path.
                    let _ = std::fs::remove_file(path);
                }
                H3Response::json(200, deleted.to_json())
            }
            Err(error) => H3Response::error(&error),
        }
    }

    fn content(&self, id: &str, query: &str, head_only: bool) -> H3Response {
        if let Some(variant) = query_value(query, "variant") {
            if variant != "video" {
                return H3Response::error(&ApiError::bad_request(
                    "unsupported_parameter",
                    Some("variant"),
                    format!(
                        "Variant '{variant}' is not produced by the h3 bridge; only 'video' is available."
                    ),
                ));
            }
        }
        let Some(video) = self.store.get(id) else {
            return H3Response::error(&unknown_video(id));
        };
        if video.status != VideoStatus::Completed {
            return H3Response::error(&ApiError::not_found(
                "video_not_ready",
                Some("video_id"),
                format!(
                    "Video '{id}' is {}; its content is not ready.",
                    video.status.as_str()
                ),
            ));
        }
        let Some(path) = self.store.output_path(id) else {
            return H3Response::error(&content_missing(id));
        };
        let body = match std::fs::read(&path) {
            Ok(body) => body,
            Err(_) => return H3Response::error(&content_missing(id)),
        };
        H3Response {
            status: 200,
            content_type: Some(CONTENT_TYPE_MP4),
            body,
            head_only,
        }
    }

    fn models(&self) -> H3Response {
        // `GET /v1/models` response: object, data[] with id, object, created,
        // owned_by and optional shutdown_date.
        let body = json!({
            "object": "list",
            "data": [{
                "id": self.model,
                "object": "model",
                "created": self.started_at,
                "owned_by": "ds4",
                "shutdown_date": null,
            }],
        })
        .to_string();
        H3Response::json(200, body)
    }
}

fn unknown_video(id: &str) -> ApiError {
    ApiError::not_found(
        "video_not_found",
        Some("video_id"),
        format!("Video with id '{id}' not found."),
    )
}

fn content_missing(id: &str) -> ApiError {
    ApiError::not_found(
        "content_not_found",
        Some("video_id"),
        format!("Video '{id}' has no downloadable content."),
    )
}

enum Route {
    Videos,
    Video(String),
    Content(String),
    Models,
    Unknown,
}

fn route(path: &str) -> Route {
    if path == "/v1/videos" {
        return Route::Videos;
    }
    if path == "/v1/models" {
        return Route::Models;
    }
    let Some(rest) = path.strip_prefix("/v1/videos/") else {
        return Route::Unknown;
    };
    if let Some(id) = rest.strip_suffix("/content") {
        if !id.is_empty() && !id.contains('/') {
            return Route::Content(id.to_string());
        }
        return Route::Unknown;
    }
    if rest.is_empty() || rest.contains('/') {
        return Route::Unknown;
    }
    Route::Video(rest.to_string())
}

fn parse_list_query(query: &str) -> ListQuery {
    ListQuery {
        limit: query_value(query, "limit"),
        order: query_value(query, "order"),
        after: query_value(query, "after"),
    }
}

fn query_value(query: &str, name: &str) -> Option<String> {
    let mut found = None;
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        if key == name {
            found = Some(percent_decode(value));
        }
    }
    found
}

/// Minimal query decoding; ids and enum values are ASCII, but a client may
/// still percent-encode them.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
                match hex.and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                    Some(byte) => {
                        out.push(byte);
                        index += 3;
                    }
                    None => {
                        out.push(b'%');
                        index += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| text.to_string())
}

/// Decode a create body in any encoding the official client sends, into the
/// one `VideoRequest` that the shared validation path consumes.
///
/// - `application/json` (and an absent Content-Type, the original behaviour):
///   parsed by `video_api::parse_request`; unknown JSON keys are ignored.
/// - `application/x-www-form-urlencoded`: percent-decoded with `+` as space;
///   malformed escapes are 400.
/// - `multipart/form-data; boundary=...`: RFC 7578 parts, quoted or unquoted
///   boundary, CRLF framing, preamble and epilogue ignored.
///
/// Form fields are limited to `prompt`, `model`, `seconds`, `size` and
/// `input_reference`; an unknown form field is 400, and a repeated form field
/// keeps its last value, the same rule `GET /v1/videos` uses for query keys.
pub fn decode_create_body(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<VideoRequest, ApiError> {
    if body.len() > MAX_BODY {
        return Err(ApiError::bad_request(
            "request_too_large",
            None,
            "Request body exceeds the server limit.",
        ));
    }
    match create_encoding(content_type)? {
        CreateEncoding::Json => parse_request(body),
        CreateEncoding::Urlencoded => request_from_form(parse_urlencoded(body)?),
        CreateEncoding::Multipart(boundary) => request_from_form(parse_multipart(body, &boundary)?),
    }
}

enum CreateEncoding {
    Json,
    Urlencoded,
    Multipart(String),
}

fn create_encoding(content_type: Option<&str>) -> Result<CreateEncoding, ApiError> {
    let Some(raw) = content_type else {
        return Ok(CreateEncoding::Json);
    };
    let (kind, params) = split_params(raw);
    match kind.to_ascii_lowercase().as_str() {
        "application/json" => Ok(CreateEncoding::Json),
        "application/x-www-form-urlencoded" => Ok(CreateEncoding::Urlencoded),
        "multipart/form-data" => {
            let boundary = params
                .iter()
                .find(|(name, _)| name == "boundary")
                .map(|(_, value)| value.clone())
                .filter(|value| !value.is_empty())
                .ok_or_else(missing_boundary)?;
            check_boundary(&boundary)?;
            Ok(CreateEncoding::Multipart(boundary))
        }
        other => Err(ApiError::bad_request(
            "unsupported_content_type",
            None,
            format!(
                "Unsupported Content-Type '{other}'. Use application/json, \
                 application/x-www-form-urlencoded or multipart/form-data."
            ),
        )),
    }
}

fn missing_boundary() -> ApiError {
    ApiError::bad_request(
        "invalid_content_type",
        None,
        "multipart/form-data requires a boundary parameter, for example \
         'multipart/form-data; boundary=----formdata'.",
    )
}

/// Reject boundaries that could never form a valid delimiter line.
fn check_boundary(boundary: &str) -> Result<(), ApiError> {
    let invalid = boundary.len() > MAX_BOUNDARY_BYTES
        || boundary.bytes().any(|byte| !(0x20..=0x7e).contains(&byte));
    if invalid {
        return Err(ApiError::bad_request(
            "invalid_content_type",
            None,
            format!(
                "Invalid multipart boundary: at most {MAX_BOUNDARY_BYTES} printable ASCII bytes."
            ),
        ));
    }
    Ok(())
}

/// One decoded form field. `filename` marks a file part (RFC 7578).
struct FormField {
    name: String,
    value: Vec<u8>,
    filename: Option<String>,
}

impl FormField {
    fn into_text(self, name: &str) -> Result<String, ApiError> {
        if self.filename.is_some() {
            return Err(ApiError::bad_request(
                "invalid_type",
                None,
                format!("Parameter '{name}' must be a text field, not a file."),
            ));
        }
        String::from_utf8(self.value).map_err(|_| {
            ApiError::bad_request(
                "invalid_encoding",
                None,
                format!("Parameter '{name}' is not valid UTF-8."),
            )
        })
    }

    fn into_image_ref(mut self) -> ImageRef {
        match self.filename.take() {
            Some(filename) => ImageRef {
                file_id: Some(filename),
                image_url: None,
            },
            None => ImageRef {
                file_id: None,
                image_url: Some(String::from_utf8_lossy(&self.value).into_owned()),
            },
        }
    }
}

fn request_from_form(fields: Vec<FormField>) -> Result<VideoRequest, ApiError> {
    let mut request = VideoRequest::default();
    for field in fields {
        match field.name.as_str() {
            "prompt" => request.prompt = Some(field.into_text("prompt")?),
            "model" => request.model = Some(field.into_text("model")?),
            "seconds" => request.seconds = Some(field.into_text("seconds")?),
            "size" => request.size = Some(field.into_text("size")?),
            // Decoded into the reference's object shape so `validate` refuses
            // it with the same unsupported_parameter error as a JSON body.
            "input_reference" => request.input_reference = Some(field.into_image_ref()),
            other => {
                return Err(ApiError::bad_request(
                    "unknown_parameter",
                    None,
                    format!("Unknown parameter: '{other}'."),
                ))
            }
        }
    }
    Ok(request)
}

fn parse_urlencoded(body: &[u8]) -> Result<Vec<FormField>, ApiError> {
    let text = std::str::from_utf8(body).map_err(|_| {
        ApiError::bad_request(
            "invalid_urlencoded",
            None,
            "The urlencoded body is not valid UTF-8.",
        )
    })?;
    let mut fields = Vec::new();
    for pair in text.split('&') {
        if pair.is_empty() {
            continue;
        }
        if fields.len() >= MAX_FORM_FIELDS {
            return Err(ApiError::bad_request(
                "invalid_urlencoded",
                None,
                format!("At most {MAX_FORM_FIELDS} form fields are accepted."),
            ));
        }
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        fields.push(FormField {
            name: percent_decode_strict(key)?,
            value: percent_decode_strict(value)?.into_bytes(),
            filename: None,
        });
    }
    Ok(fields)
}

/// Strict form decoding: `+` is a space and every `%` needs two hex digits.
fn percent_decode_strict(text: &str) -> Result<String, ApiError> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' => {
                if index + 2 >= bytes.len() {
                    return Err(malformed_escape());
                }
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3])
                    .ok()
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
                    .ok_or_else(malformed_escape)?;
                out.push(hex);
                index += 3;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| {
        ApiError::bad_request(
            "invalid_urlencoded",
            None,
            "A urlencoded value does not decode to valid UTF-8.",
        )
    })
}

fn malformed_escape() -> ApiError {
    ApiError::bad_request(
        "invalid_urlencoded",
        None,
        "Malformed percent escape: every '%' must be followed by two hex digits.",
    )
}

fn parse_multipart(body: &[u8], boundary: &str) -> Result<Vec<FormField>, ApiError> {
    // A leading CRLF makes the first boundary look like every later one, so
    // a preamble before it is skipped by the same search.
    let mut stream = Vec::with_capacity(body.len() + 2);
    stream.extend_from_slice(b"\r\n");
    stream.extend_from_slice(body);
    let marker = [b"\r\n--".as_slice(), boundary.as_bytes()].concat();

    let first = find_delimiter(&stream, &marker, 0)
        .ok_or_else(|| multipart_error("body has no initial boundary line"))?;
    let mut fields = Vec::new();
    let mut position = first + marker.len();
    loop {
        if stream[position..].starts_with(b"--") {
            return Ok(fields);
        }
        if !stream[position..].starts_with(b"\r\n") {
            return Err(multipart_error("malformed boundary line"));
        }
        position += 2;

        let next = find_delimiter(&stream, &marker, position)
            .ok_or_else(|| multipart_error("truncated body: the closing boundary is missing"))?;
        let part = &stream[position..next];
        let headers_end = find_subslice(part, b"\r\n\r\n")
            .ok_or_else(|| multipart_error("truncated part: the header terminator is missing"))?;
        let (name, filename) = part_disposition(&part[..headers_end])?;
        if fields.len() >= MAX_FORM_FIELDS {
            return Err(multipart_error("too many parts"));
        }
        fields.push(FormField {
            name,
            value: part[headers_end + 4..].to_vec(),
            filename,
        });
        position = next + marker.len();
    }
}

fn multipart_error(reason: &str) -> ApiError {
    ApiError::bad_request(
        "invalid_multipart",
        None,
        format!("Malformed multipart/form-data body: {reason}."),
    )
}

/// Find `marker` where it is a real delimiter: followed by CRLF (another
/// part) or `--` (the closing boundary). A bare prefix match inside a part
/// body is skipped.
fn find_delimiter(stream: &[u8], marker: &[u8], from: usize) -> Option<usize> {
    let mut index = from;
    while let Some(offset) = find_subslice(&stream[index..], marker) {
        let at = index + offset;
        let after = at + marker.len();
        if stream[after..].starts_with(b"\r\n") || stream[after..].starts_with(b"--") {
            return Some(at);
        }
        index = at + 1;
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Read the part's `Content-Disposition: form-data; name="..."; filename="..."`.
fn part_disposition(headers: &[u8]) -> Result<(String, Option<String>), ApiError> {
    if headers.len() > MAX_PART_HEADER_BYTES {
        return Err(multipart_error("part headers are too large"));
    }
    let text = std::str::from_utf8(headers)
        .map_err(|_| multipart_error("part headers are not valid UTF-8"))?;
    for line in text.split("\r\n") {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("content-disposition") {
            continue;
        }
        let (kind, params) = split_params(value.trim());
        if !kind.eq_ignore_ascii_case("form-data") {
            return Err(multipart_error("only form-data parts are accepted"));
        }
        let field = params
            .iter()
            .find(|(name, _)| name == "name")
            .map(|(_, value)| value.clone())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| multipart_error("a part is missing its field name"))?;
        let filename = params
            .iter()
            .find(|(name, _)| name == "filename")
            .map(|(_, value)| value.clone());
        return Ok((field, filename));
    }
    Err(multipart_error(
        "a part is missing Content-Disposition: form-data",
    ))
}

/// Split `type; name=value; name="quoted value"` and lowercase the parameter
/// names. Quoted values may contain `;` and use backslash escapes.
fn split_params(text: &str) -> (String, Vec<(String, String)>) {
    let (kind, rest) = match text.find(';') {
        Some(index) => (&text[..index], &text[index + 1..]),
        None => (text, ""),
    };
    (kind.trim().to_string(), parse_param_list(rest))
}

fn parse_param_list(text: &str) -> Vec<(String, String)> {
    let bytes = text.as_bytes();
    let mut params = Vec::new();
    let mut index = 0;
    loop {
        while index < bytes.len() && (bytes[index] == b';' || bytes[index].is_ascii_whitespace()) {
            index += 1;
        }
        if index >= bytes.len() {
            return params;
        }
        let name_start = index;
        while index < bytes.len() && bytes[index] != b'=' && bytes[index] != b';' {
            index += 1;
        }
        let name = text[name_start..index].trim().to_ascii_lowercase();
        if index >= bytes.len() || bytes[index] != b'=' {
            params.push((name, String::new()));
            continue;
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let value = if index < bytes.len() && bytes[index] == b'"' {
            index += 1;
            let mut value = String::new();
            while index < bytes.len() {
                match bytes[index] {
                    b'\\' if index + 1 < bytes.len() => {
                        value.push(bytes[index + 1] as char);
                        index += 2;
                    }
                    b'"' => {
                        index += 1;
                        break;
                    }
                    byte => {
                        value.push(byte as char);
                        index += 1;
                    }
                }
            }
            value
        } else {
            let value_start = index;
            while index < bytes.len() && bytes[index] != b';' {
                index += 1;
            }
            text[value_start..index].trim().to_string()
        };
        params.push((name, value));
    }
}

/// Case-insensitive header lookup over the raw header block, returning the
/// trimmed value. Only Content-Type is read here.
fn header_value(head: &[u8], name: &[u8]) -> Option<String> {
    for line in head.split(|&byte| byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.len() < name.len() || !line[..name.len()].eq_ignore_ascii_case(name) {
            continue;
        }
        let value = std::str::from_utf8(&line[name.len()..]).ok()?;
        return Some(value.trim().to_string());
    }
    None
}

/// Read one request, keeping the query string. Content-Length only: the
/// bridge serves JSON and file bodies, and explicit refusal is safer than a
/// half-supported chunked path.
pub fn read_request<R: Read>(reader: &mut R) -> Result<H3Request, ReadError> {
    let mut buffer = Vec::new();
    let head_len = loop {
        if let Some(end) = header_end(&buffer) {
            break end;
        }
        if buffer.len() >= MAX_HEADER {
            return Err(ReadError::Malformed);
        }
        let mut chunk = [0u8; 4096];
        match reader.read(&mut chunk) {
            Ok(0) => return Err(ReadError::Eof),
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(ReadError::Eof),
        }
    };

    // Copy the request line out before the body loop: extending `buffer`
    // would otherwise keep the header borrow alive.
    let (method, path, query, content_type, length) = {
        let head = &buffer[..head_len];
        let line_end = head
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or(ReadError::Malformed)?;
        let line = std::str::from_utf8(&head[..line_end]).map_err(|_| ReadError::Malformed)?;
        let line = line.trim_end_matches('\r');
        let mut parts = line.splitn(3, ' ');
        let method = parts.next().ok_or(ReadError::Malformed)?;
        let target = parts.next().ok_or(ReadError::Malformed)?;
        if method.is_empty() || target.is_empty() {
            return Err(ReadError::Malformed);
        }
        let (path, query) = target.split_once('?').unwrap_or((target, ""));

        if header_chunked(head) {
            return Err(ReadError::Chunked);
        }
        let length = content_length(head);
        if length < 0 || length as usize > MAX_BODY {
            return Err(ReadError::TooLarge);
        }
        (
            method.to_string(),
            path.to_string(),
            query.to_string(),
            header_value(head, b"Content-Type:"),
            length,
        )
    };
    let total = head_len + length as usize;
    while buffer.len() < total {
        let mut chunk = [0u8; 8192];
        match reader.read(&mut chunk) {
            Ok(0) => return Err(ReadError::Eof),
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(ReadError::Eof),
        }
    }
    Ok(H3Request {
        method,
        path,
        query,
        content_type,
        body: buffer[head_len..total].to_vec(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    Eof,
    Malformed,
    Chunked,
    TooLarge,
}

/// Accept and answer exactly one connection. Used by the socket test.
pub fn serve_one(listener: &TcpListener, server: &VideoServer) -> io::Result<()> {
    let (mut stream, _) = listener.accept()?;
    answer(server, &mut stream)
}

/// Accept connections forever, one thread per connection.
pub fn serve(listener: TcpListener, server: Arc<VideoServer>) -> io::Result<()> {
    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                let server = Arc::clone(&server);
                thread::spawn(move || {
                    let _ = answer(&server, &mut stream);
                });
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

fn answer(server: &VideoServer, stream: &mut std::net::TcpStream) -> io::Result<()> {
    let response = match read_request(stream) {
        Ok(request) => server.handle(&request),
        Err(ReadError::Eof) => return Ok(()),
        Err(ReadError::Chunked) => H3Response::error(&ApiError::bad_request(
            "unsupported_encoding",
            None,
            "Chunked request bodies are not supported by the h3 bridge.",
        )),
        Err(ReadError::TooLarge) => H3Response::error(&ApiError::bad_request(
            "request_too_large",
            None,
            "Request body exceeds the server limit.",
        )),
        Err(ReadError::Malformed) => H3Response::error(&ApiError::bad_request(
            "invalid_request",
            None,
            "Malformed HTTP request.",
        )),
    };
    stream.write_all(&response.to_bytes())?;
    stream.flush()
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_core::minimax_h3::video_api::{
        VideoSize, DEFAULT_MODEL, DEFAULT_SIZE, MAX_PROMPT_CHARS,
    };
    use serde_json::Value;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    struct FnRenderer<F> {
        func: F,
    }

    impl<F> Renderer for FnRenderer<F>
    where
        F: Fn(&RenderSpec, &Path) -> Result<(), RenderError> + Send + Sync,
    {
        fn state(&self) -> RendererState {
            RendererState::Ready
        }

        fn render(&self, spec: &RenderSpec, output: &Path) -> Result<(), RenderError> {
            (self.func)(spec, output)
        }
    }

    fn renderer<F>(func: F) -> SharedRenderer
    where
        F: Fn(&RenderSpec, &Path) -> Result<(), RenderError> + Send + Sync + 'static,
    {
        Arc::new(FnRenderer { func })
    }

    fn write_output(output: &Path) -> Result<(), RenderError> {
        std::fs::write(output, b"fake-mp4")
            .map_err(|error| RenderError::new("render_failed", error.to_string()))
    }

    fn test_server(renderer: SharedRenderer, max_jobs: usize) -> (Arc<VideoServer>, PathBuf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "h3-video-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let store = Arc::new(VideoStore::new(DEFAULT_MODEL, max_jobs));
        (
            Arc::new(VideoServer::new(store, renderer, dir.clone())),
            dir,
        )
    }

    fn request(method: &str, path: &str) -> H3Request {
        H3Request {
            method: method.to_string(),
            path: path.to_string(),
            query: String::new(),
            content_type: None,
            body: Vec::new(),
        }
    }

    fn with_query(method: &str, path: &str, query: &str) -> H3Request {
        H3Request {
            query: query.to_string(),
            ..request(method, path)
        }
    }

    fn post(server: &VideoServer, body: &str) -> H3Response {
        post_encoded(server, "application/json", body.as_bytes())
    }

    fn post_encoded(server: &VideoServer, content_type: &str, body: &[u8]) -> H3Response {
        server.handle(&H3Request {
            content_type: Some(content_type.to_string()),
            body: body.to_vec(),
            ..request("POST", "/v1/videos")
        })
    }

    fn multipart_body(boundary: &str, parts: &[(&str, &str)]) -> String {
        let mut body = String::new();
        for (name, value) in parts {
            body.push_str(&format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{boundary}--\r\n"));
        body
    }

    fn json_of(response: &H3Response) -> Value {
        serde_json::from_slice(&response.body).unwrap_or(Value::Null)
    }

    fn poll_until(server: &VideoServer, id: &str, want: &str) -> Value {
        for _ in 0..600 {
            let response = server.handle(&request("GET", &format!("/v1/videos/{id}")));
            let json = json_of(&response);
            if json["status"] == want {
                return json;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("job {id} never reached {want}");
    }

    fn create(server: &VideoServer, prompt: &str) -> String {
        let response = post(server, &format!(r#"{{"prompt":"{prompt}"}}"#));
        assert_eq!(response.status, 200);
        json_of(&response)["id"].as_str().unwrap().to_string()
    }

    #[test]
    fn create_poll_complete_then_download_content() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let response = post(&server, r#"{"prompt":"a red cube"}"#);
        assert_eq!(response.status, 200);
        let created = json_of(&response);
        assert_eq!(created["status"], "queued");
        assert_eq!(created["object"], "video");
        assert_eq!(created["size"], DEFAULT_SIZE);
        assert_eq!(created["seconds"], "4");
        let id = created["id"].as_str().unwrap().to_string();

        let done = poll_until(&server, &id, "completed");
        assert_eq!(done["progress"], 100);
        assert!(!done["completed_at"].is_null());

        let content = server.handle(&request("GET", &format!("/v1/videos/{id}/content")));
        assert_eq!(content.status, 200);
        assert_eq!(content.content_type, Some(CONTENT_TYPE_MP4));
        assert_eq!(content.body, b"fake-mp4");
        let wire = String::from_utf8_lossy(&content.to_bytes()).to_string();
        assert!(wire.contains("Content-Length: 8\r\n"), "{wire}");
        assert!(wire.contains("Content-Type: video/mp4\r\n"), "{wire}");

        let head = server.handle(&request("HEAD", &format!("/v1/videos/{id}/content")));
        assert_eq!(head.status, 200);
        assert_eq!(head.body, b"fake-mp4");
        let wire = head.to_bytes();
        assert!(wire.ends_with(b"\r\n\r\n"), "HEAD carries no body");
    }

    #[test]
    fn in_progress_is_visible_before_the_renderer_finishes() {
        let release = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&release);
        let (server, _dir) = test_server(
            renderer(move |_, output| {
                while !flag.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(2));
                }
                write_output(output)
            }),
            8,
        );
        let id = create(&server, "slow");
        let running = poll_until(&server, &id, "in_progress");
        assert_eq!(running["progress"], 0);
        release.store(true, Ordering::SeqCst);
        poll_until(&server, &id, "completed");
    }

    #[test]
    fn create_validates_every_field_and_names_its_param() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        for (body, param, code) in [
            (r#"{"prompt":""}"#, "prompt", "string_below_min_length"),
            (
                r#"{"prompt":"p","size":"720x1280"}"#,
                "size",
                "invalid_value",
            ),
            (
                r#"{"prompt":"p","seconds":"16"}"#,
                "seconds",
                "invalid_value",
            ),
            (
                r#"{"prompt":"p","model":"sora-2"}"#,
                "model",
                "model_not_found",
            ),
            (
                r#"{"prompt":"p","input_reference":{"image_url":"data:image/png;base64,AA"}}"#,
                "input_reference",
                "unsupported_parameter",
            ),
        ] {
            let response = post(&server, body);
            assert_eq!(response.status, 400, "{body}");
            let json = json_of(&response);
            assert_eq!(json["error"]["param"], param, "{body}");
            assert_eq!(json["error"]["code"], code, "{body}");
            assert_eq!(json["error"]["type"], "invalid_request_error");
        }

        let response = post(&server, "{bad json");
        assert_eq!(response.status, 400);
        assert_eq!(json_of(&response)["error"]["code"], "invalid_json");
        assert!(json_of(&response)["error"]["param"].is_null());

        let long = "x".repeat(MAX_PROMPT_CHARS + 1);
        let response = post(&server, &format!(r#"{{"prompt":"{long}"}}"#));
        assert_eq!(json_of(&response)["error"]["param"], "prompt");
    }

    #[test]
    fn all_three_encodings_reach_the_same_job() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let fields = [
            ("model", DEFAULT_MODEL),
            ("prompt", "A calico cat playing a piano on stage"),
            ("seconds", "8"),
            ("size", "768x768"),
        ];

        let json = post(
            &server,
            r#"{"model":"minimax-h3","prompt":"A calico cat playing a piano on stage","seconds":"8","size":"768x768"}"#,
        );
        assert_eq!(json.status, 200);

        let urlencoded = post_encoded(
            &server,
            "application/x-www-form-urlencoded",
            b"model=minimax-h3&prompt=A+calico+cat+playing+a+piano+on+stage&seconds=8&size=768x768",
        );
        assert_eq!(urlencoded.status, 200);

        // The official curl example is multipart with a quoted boundary.
        // A preamble and an epilogue are legal RFC 7578 framing and skipped.
        let body = format!(
            "preamble\r\n{}epilogue\r\n",
            multipart_body("----formdata", &fields)
        );
        let multipart = post_encoded(
            &server,
            "multipart/form-data; boundary=\"----formdata\"",
            body.as_bytes(),
        );
        assert_eq!(multipart.status, 200);

        for response in [&json, &urlencoded, &multipart] {
            let video = json_of(response);
            assert_eq!(video["model"], DEFAULT_MODEL);
            assert_eq!(video["prompt"], "A calico cat playing a piano on stage");
            assert_eq!(video["seconds"], "8");
            assert_eq!(video["size"], "768x768");
        }
    }

    #[test]
    fn urlencoded_decodes_plus_and_rejects_malformed_escapes() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let ok = post_encoded(
            &server,
            "application/x-www-form-urlencoded; charset=utf-8",
            b"prompt=one&prompt=two+words",
        );
        assert_eq!(ok.status, 200);
        assert_eq!(json_of(&ok)["prompt"], "two words", "last field wins");

        for body in [
            "prompt=%ZZ".as_bytes(),
            b"prompt=%2",
            b"prompt=%",
            b"prompt=%ff",
        ] {
            let response = post_encoded(&server, "application/x-www-form-urlencoded", body);
            assert_eq!(response.status, 400, "{:?}", String::from_utf8_lossy(body));
            assert_eq!(json_of(&response)["error"]["code"], "invalid_urlencoded");
        }

        let unknown = post_encoded(
            &server,
            "application/x-www-form-urlencoded",
            b"prompt=p&colour=red",
        );
        assert_eq!(unknown.status, 400);
        let json = json_of(&unknown);
        assert_eq!(json["error"]["code"], "unknown_parameter");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("colour"));
    }

    #[test]
    fn multipart_rejects_file_input_reference_and_unknown_fields() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let file_body = concat!(
            "--ref\r\n",
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n",
            "a cube\r\n",
            "--ref\r\n",
            "Content-Disposition: form-data; name=\"input_reference\"; filename=\"ref.png\"\r\n",
            "Content-Type: image/png\r\n\r\n",
            "PNGDATA\r\n",
            "--ref--\r\n"
        );
        let response = post_encoded(
            &server,
            "multipart/form-data; boundary=ref",
            file_body.as_bytes(),
        );
        assert_eq!(response.status, 400);
        let json = json_of(&response);
        assert_eq!(json["error"]["code"], "unsupported_parameter");
        assert_eq!(json["error"]["param"], "input_reference");

        let unknown = post_encoded(
            &server,
            "multipart/form-data; boundary=ref",
            multipart_body("ref", &[("prompt", "p"), ("colour", "red")]).as_bytes(),
        );
        assert_eq!(unknown.status, 400);
        let json = json_of(&unknown);
        assert_eq!(json["error"]["code"], "unknown_parameter");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("colour"));
    }

    #[test]
    fn multipart_reports_missing_boundary_and_truncated_parts() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let response = post_encoded(
            &server,
            "multipart/form-data",
            multipart_body("b", &[("prompt", "p")]).as_bytes(),
        );
        assert_eq!(response.status, 400);
        assert_eq!(json_of(&response)["error"]["code"], "invalid_content_type");

        // No closing boundary.
        let truncated = concat!(
            "--b\r\n",
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n",
            "abc"
        );
        let response = post_encoded(
            &server,
            "multipart/form-data; boundary=b",
            truncated.as_bytes(),
        );
        assert_eq!(response.status, 400);
        assert_eq!(json_of(&response)["error"]["code"], "invalid_multipart");

        // Part headers without the blank line that ends them.
        let truncated = concat!(
            "--b\r\n",
            "Content-Disposition: form-data; name=\"prompt\"\r\n",
            "--b--\r\n"
        );
        let response = post_encoded(
            &server,
            "multipart/form-data; boundary=b",
            truncated.as_bytes(),
        );
        assert_eq!(response.status, 400);
        assert_eq!(json_of(&response)["error"]["code"], "invalid_multipart");
    }

    #[test]
    fn multipart_empty_values_feed_the_shared_validation() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let response = post_encoded(
            &server,
            "multipart/form-data; boundary=b",
            multipart_body("b", &[("prompt", "")]).as_bytes(),
        );
        assert_eq!(response.status, 400);
        let json = json_of(&response);
        assert_eq!(json["error"]["code"], "string_below_min_length");
        assert_eq!(json["error"]["param"], "prompt");
    }

    #[test]
    fn unknown_content_type_is_refused() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let response = post_encoded(&server, "text/plain", b"prompt=p");
        assert_eq!(response.status, 400);
        assert_eq!(
            json_of(&response)["error"]["code"],
            "unsupported_content_type"
        );
    }

    #[test]
    fn unknown_ids_and_unready_content_are_404() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        for (method, path) in [
            ("GET", "/v1/videos/video_99"),
            ("DELETE", "/v1/videos/video_99"),
            ("GET", "/v1/videos/video_99/content"),
            ("HEAD", "/v1/videos/video_99/content"),
        ] {
            let response = server.handle(&request(method, path));
            assert_eq!(response.status, 404, "{method} {path}");
            assert_eq!(json_of(&response)["error"]["code"], "video_not_found");
        }

        let release = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&release);
        let (server, _dir) = test_server(
            renderer(move |_, output| {
                while !flag.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(2));
                }
                write_output(output)
            }),
            8,
        );
        let id = create(&server, "pending");
        let response = server.handle(&request("GET", &format!("/v1/videos/{id}/content")));
        assert_eq!(response.status, 404);
        assert_eq!(json_of(&response)["error"]["code"], "video_not_ready");
        release.store(true, Ordering::SeqCst);
        poll_until(&server, &id, "completed");
    }

    #[test]
    fn list_paginates_through_the_query_string() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        for index in 0..3 {
            let id = create(&server, &format!("p{index}"));
            poll_until(&server, &id, "completed");
        }

        let first = server.handle(&with_query("GET", "/v1/videos", "limit=2"));
        assert_eq!(first.status, 200);
        let json = json_of(&first);
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["first_id"], "video_3");
        assert_eq!(json["last_id"], "video_2");
        assert_eq!(json["has_more"], true);

        let second = server.handle(&with_query("GET", "/v1/videos", "limit=2&after=video_2"));
        let json = json_of(&second);
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["first_id"], "video_1");
        assert_eq!(json["has_more"], false);

        let ascending = json_of(&server.handle(&with_query("GET", "/v1/videos", "order=asc")));
        assert_eq!(ascending["first_id"], "video_1");
        assert_eq!(ascending["last_id"], "video_3");

        let bad = server.handle(&with_query("GET", "/v1/videos", "limit=101"));
        assert_eq!(bad.status, 400);
        assert_eq!(json_of(&bad)["error"]["param"], "limit");
        let bad = server.handle(&with_query("GET", "/v1/videos", "after=video_77"));
        assert_eq!(json_of(&bad)["error"]["param"], "after");
    }

    #[test]
    fn delete_removes_the_job_and_its_file() {
        let (server, dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let id = create(&server, "gone");
        poll_until(&server, &id, "completed");
        let path = dir.join(format!("{id}.mp4"));
        assert!(path.exists());

        let response = server.handle(&request("DELETE", &format!("/v1/videos/{id}")));
        assert_eq!(response.status, 200);
        let json = json_of(&response);
        assert_eq!(json["object"], "video.deleted");
        assert_eq!(json["deleted"], true);
        assert_eq!(json["id"], id);
        assert!(!path.exists());
        assert_eq!(
            server
                .handle(&request("GET", &format!("/v1/videos/{id}")))
                .status,
            404
        );
    }

    #[test]
    fn two_jobs_render_concurrently() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (a, b) = (Arc::clone(&active), Arc::clone(&peak));
        let (server, _dir) = test_server(
            renderer(move |_, output| {
                let now = a.fetch_add(1, Ordering::SeqCst) + 1;
                b.fetch_max(now, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(120));
                a.fetch_sub(1, Ordering::SeqCst);
                write_output(output)
            }),
            8,
        );
        let first = create(&server, "one");
        let second = create(&server, "two");
        poll_until(&server, &first, "completed");
        poll_until(&server, &second, "completed");
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "both render threads should overlap"
        );
    }

    #[test]
    fn disabled_renderer_answers_503() {
        let (server, _dir) = test_server(Arc::new(DisabledRenderer), 8);
        let response = post(&server, r#"{"prompt":"p"}"#);
        assert_eq!(response.status, 503);
        let json = json_of(&response);
        assert_eq!(json["error"]["type"], "service_unavailable_error");
        assert_eq!(json["error"]["code"], "renderer_disabled");
        assert_eq!(server.store().len(), 0, "nothing was enqueued");
    }

    #[test]
    fn renderer_failure_and_missing_output_become_failed_jobs() {
        let (server, _dir) = test_server(
            renderer(|_, _| Err(RenderError::new("render_failed", "boom"))),
            8,
        );
        let id = create(&server, "fails");
        let failed = poll_until(&server, &id, "failed");
        assert_eq!(failed["error"]["code"], "render_failed");
        assert_eq!(failed["error"]["message"], "boom");

        let (server, _dir) = test_server(renderer(|_, _| Ok(())), 8);
        let id = create(&server, "silent");
        let failed = poll_until(&server, &id, "failed");
        assert_eq!(failed["error"]["code"], "render_output_missing");
    }

    #[test]
    fn queue_full_answers_429() {
        let release = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&release);
        let (server, _dir) = test_server(
            renderer(move |_, output| {
                while !flag.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(2));
                }
                write_output(output)
            }),
            1,
        );
        let id = create(&server, "first");
        let response = post(&server, r#"{"prompt":"second"}"#);
        assert_eq!(response.status, 429);
        let json = json_of(&response);
        assert_eq!(json["error"]["type"], "rate_limit_error");
        assert_eq!(json["error"]["code"], "queue_full");
        release.store(true, Ordering::SeqCst);
        poll_until(&server, &id, "completed");
    }

    #[test]
    fn content_variant_is_limited_to_video() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let id = create(&server, "p");
        poll_until(&server, &id, "completed");
        let ok = server.handle(&with_query(
            "GET",
            &format!("/v1/videos/{id}/content"),
            "variant=video",
        ));
        assert_eq!(ok.status, 200);
        let bad = server.handle(&with_query(
            "GET",
            &format!("/v1/videos/{id}/content"),
            "variant=thumbnail",
        ));
        assert_eq!(bad.status, 400);
        assert_eq!(json_of(&bad)["error"]["param"], "variant");
    }

    #[test]
    fn models_lists_the_configured_model() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let response = server.handle(&request("GET", "/v1/models"));
        assert_eq!(response.status, 200);
        let json = json_of(&response);
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], DEFAULT_MODEL);
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "ds4");
    }

    #[test]
    fn unknown_routes_and_methods_answer_404() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        for (method, path) in [
            ("GET", "/v1/videos/video_1/remix"),
            ("PUT", "/v1/videos"),
            ("GET", "/v1/files"),
        ] {
            let response = server.handle(&request(method, path));
            assert_eq!(response.status, 404, "{method} {path}");
            assert_eq!(json_of(&response)["error"]["code"], "unknown_url");
        }
    }

    #[test]
    fn command_renderer_keeps_request_values_as_separate_argv() {
        let renderer = CommandRenderer::new(
            "/home/leandro/h3-oracle/h3",
            vec![
                "-d".to_string(),
                "/mnt/h3-artifacts/MiniMax-H3".to_string(),
                "--steps".to_string(),
                "20".to_string(),
            ],
            Duration::from_secs(60),
        );
        let spec = RenderSpec {
            prompt: "a cube; rm -rf / --width 9".to_string(),
            model: DEFAULT_MODEL.to_string(),
            size: VideoSize::new(512, 768),
            seconds: "4".to_string(),
            frames: 96,
        };
        let argv: Vec<String> = renderer
            .argv(&spec, Path::new("/out/video_1.mp4"))
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect();
        // Exact reference spellings: `-p`, `-o`, `--width`, `--height`,
        // `--seconds` (`/data/h3.c/main.c:22-29`, `:292-296`, `:361-381`).
        assert_eq!(
            argv,
            vec![
                "/home/leandro/h3-oracle/h3",
                "-d",
                "/mnt/h3-artifacts/MiniMax-H3",
                "--steps",
                "20",
                "-p",
                "a cube; rm -rf / --width 9",
                "--width",
                "512",
                "--height",
                "768",
                "--seconds",
                "4",
                "-o",
                "/out/video_1.mp4",
            ]
        );
        assert_eq!(argv.len(), 15, "a prompt never adds argv entries");
        assert!(
            !argv.iter().any(|arg| arg == "--frames"),
            "the bridge must not invent a frames conversion"
        );
    }

    #[test]
    fn command_renderer_maps_nonzero_exit_and_timeout() {
        if !Path::new("/bin/false").exists() {
            return;
        }
        let renderer = CommandRenderer::new("/bin/false", Vec::new(), Duration::from_secs(5));
        let spec = RenderSpec {
            prompt: "p".to_string(),
            model: DEFAULT_MODEL.to_string(),
            size: VideoSize::new(512, 512),
            seconds: "4".to_string(),
            frames: 96,
        };
        let error = renderer
            .render(&spec, Path::new("/tmp/h3-video-never.mp4"))
            .unwrap_err();
        assert_eq!(error.code, "render_failed");
        assert!(error.message.contains("status 1"), "{}", error.message);

        // `yes --` ends option parsing and then loops forever; guard on its
        // presence. The `--` lives in the fixed args, so the request-derived
        // argv is still just operands.
        if Path::new("/usr/bin/yes").exists() {
            let renderer = CommandRenderer::new(
                "/usr/bin/yes",
                vec!["--".to_string()],
                Duration::from_millis(60),
            );
            let error = renderer
                .render(&spec, Path::new("/tmp/h3-video-never.mp4"))
                .unwrap_err();
            assert_eq!(error.code, "render_timeout", "{}", error.message);
        }
    }

    #[test]
    fn socket_serves_models_over_tcp() {
        use std::io::Read;
        use std::net::TcpStream;

        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let thread = thread::spawn(move || serve_one(&listener, &server).unwrap());

        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: h3\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        thread.join().unwrap();

        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("\"object\":\"list\""), "{text}");
        assert!(text.contains(DEFAULT_MODEL), "{text}");
    }

    #[test]
    fn request_reader_keeps_the_query_and_reads_the_body() {
        let wire = b"POST /v1/videos?limit=2 HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 4\r\n\r\nbody".to_vec();
        let request = read_request(&mut wire.as_slice()).unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/videos");
        assert_eq!(request.query, "limit=2");
        assert_eq!(
            request.content_type.as_deref(),
            Some("multipart/form-data; boundary=x")
        );
        assert_eq!(request.body, b"body");

        let chunked =
            b"POST /v1/videos HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n".to_vec();
        assert_eq!(
            read_request(&mut chunked.as_slice()),
            Err(ReadError::Chunked)
        );

        let malformed = b"garbage\r\n\r\n".to_vec();
        assert_eq!(
            read_request(&mut malformed.as_slice()),
            Err(ReadError::Malformed)
        );
    }
}
