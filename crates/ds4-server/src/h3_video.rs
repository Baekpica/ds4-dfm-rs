//! HTTP surface for the h3 family's local v1 contract.
//!
//! The contract is `docs/minimax-h3/local-v1-contract.md`, taken from the
//! official repository's own local clients
//! (`/data/MiniMax-H3/scripts/readme/*.sh`):
//!
//! ```text
//! POST /v1/videos               -> {"id": "<id>"}
//! GET  /v1/videos/{id}          -> the task record, whose "status" is polled
//! GET  /v1/videos/{id}/content  -> the MP4 bytes
//! ```
//!
//! No other route exists: not the hosted platform's `/v2` paths, not a list
//! or delete route, and not `/v1/models`.
//!
//! This module owns HTTP only: the body and its mapping live in
//! `ds4_core::minimax_h3::local_api`, the task record and store in
//! `ds4_core::minimax_h3::video_api`. It is a separate binary
//! (`h3-server-rs`), so no existing family's route table changes.
//!
//! Security boundary: the request body supplies a prompt, a target, a seed
//! and at most two data-URI anchors. The executable, its fixed arguments, the
//! output directory and the working directory are fixed at startup, and every
//! request value is passed as its own argv entry with no shell. The argv
//! spells the reference engine's CLI (`-p`, `-o`, `--width`, `--height`,
//! `--seconds`, `--first-frame`, `--last-frame`, `--seed`), so the oracle
//! binary runs without an adapter.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ds4_core::minimax_h3::local_api::{
    image_size, parse_create, Plan, ReferenceKind, ReferenceMedia,
};
use ds4_core::minimax_h3::media::probe_visual_size;
use ds4_core::minimax_h3::video_api::{
    unknown_video, Admitted, AnchorImage, ApiError, VideoError, VideoStatus, VideoStore,
};
use serde_json::json;

use crate::error::http_head;
use crate::http::{content_length, header_chunked, header_end, MAX_BODY, MAX_HEADER};

/// Stderr kept for a failed renderer, matching the reference's practice of
/// surfacing the tail of the failing program's output.
pub const STDERR_TAIL_BYTES: usize = 4096;

/// How often the renderer wait loop checks for exit or timeout.
pub const RENDER_POLL: Duration = Duration::from_millis(20);

const CLIENT_LIMIT: usize = 64;
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

pub const CONTENT_TYPE_MP4: &str = "video/mp4";
pub const CONTENT_TYPE_JSON: &str = "application/json";

/// A renderer either accepts work or the bridge returns 503 on create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererState {
    Ready,
    Disabled,
}

/// A failed render, before it becomes a `VideoError`.
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
    fn render(&self, spec: &Plan, output: &Path) -> Result<(), RenderError>;
}

pub type SharedRenderer = Arc<dyn Renderer>;

/// No renderer configured: create answers 503 without touching the store.
pub struct DisabledRenderer;

impl Renderer for DisabledRenderer {
    fn state(&self) -> RendererState {
        RendererState::Disabled
    }

    fn render(&self, _spec: &Plan, _output: &Path) -> Result<(), RenderError> {
        Err(RenderError::new(
            "renderer_disabled",
            "no h3 renderer is configured",
        ))
    }
}

/// One conditioning image on disk, for the duration of one render.
///
/// Both renderers take a path, as the reference does (`h3.h:71-73`), so the
/// decoded wire bytes are written beside the output — the server's only
/// writable state — and removed when the render returns. The name is derived
/// from the output, which the store already keeps unique per request.
pub(crate) struct MediaFile {
    path: PathBuf,
}

impl MediaFile {
    pub(crate) fn write(
        anchor: Option<&AnchorImage>,
        output: &Path,
        tag: &str,
    ) -> Result<Option<MediaFile>, RenderError> {
        let Some(anchor) = anchor else {
            return Ok(None);
        };
        let extension = anchor.extension().ok_or_else(|| {
            RenderError::new(
                "unsupported_parameter",
                format!(
                    "anchor media type '{}' has no file extension",
                    anchor.media_type
                ),
            )
        })?;
        let path = output.with_extension(format!("{tag}.{extension}"));
        std::fs::write(&path, &anchor.bytes).map_err(|error| {
            RenderError::new(
                "render_failed",
                format!("could not write the anchor at {}: {error}", path.display()),
            )
        })?;
        Ok(Some(MediaFile { path }))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Write one ordered reference beside the output. The tag keeps files
    /// apart within a job; the extension follows the declared media type.
    pub(crate) fn write_reference(
        reference: &ReferenceMedia,
        output: &Path,
        index: usize,
    ) -> Result<MediaFile, RenderError> {
        let extension = reference.extension().ok_or_else(|| {
            RenderError::new(
                "unsupported_parameter",
                format!(
                    "reference media type '{}' has no file extension",
                    reference.media_type
                ),
            )
        })?;
        let path = output.with_extension(format!("ref{index}.{extension}"));
        std::fs::write(&path, &reference.bytes).map_err(|error| {
            RenderError::new(
                "render_failed",
                format!(
                    "could not write reference {index} at {}: {error}",
                    path.display()
                ),
            )
        })?;
        Ok(MediaFile { path })
    }
}

impl Drop for MediaFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Runs an operator-supplied program. The fixed program path and args are
/// chosen at startup; request values are appended as separate argv entries.
///
/// The spellings are the reference engine's own, verified in
/// `/data/h3.c/main.c` at eae5d54: `-p/--prompt` and `-o/--output` are the
/// short forms (`main.c:22-24`, option table `main.c:292-296`, switch
/// `main.c:361-381`), and `--width`, `--height`, `--seconds`, `--seed`,
/// `--first-frame` and `--last-frame` are the long options (`main.c:298`,
/// `:330-333`). `--seconds` is passed unchanged; the reference converts it at
/// 24 fps itself (`frames_from_seconds`, `main.c:78-91`), so the bridge never
/// invents a frames value. A fixed `-d <model-dir>` may precede the request
/// arguments.
///
/// `program [fixed args...] -p P --width W --height H --seconds S
///  [--first-frame P] [--last-frame P] --seed N -o PATH`
/// One reference file and the flag that carries it to the reference CLI
/// (`main.c:438-490`).
pub struct ReferenceArg<'a> {
    pub path: &'a Path,
    pub kind: ReferenceKind,
    pub include_audio: bool,
}

impl ReferenceArg<'_> {
    /// `--ref-image`, `--ref-video`, `--ref-silent-video` or `--ref-audio`.
    fn flag(&self) -> &'static str {
        match self.kind {
            ReferenceKind::Image => "--ref-image",
            ReferenceKind::Video if self.include_audio => "--ref-video",
            ReferenceKind::Video => "--ref-silent-video",
            ReferenceKind::Audio => "--ref-audio",
        }
    }
}

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

    /// The complete argv, program first. Every request value is one argv entry
    /// behind a fixed flag, so a value can never become an option or a
    /// command. Anchor paths are server-generated (`MediaFile`), never caller
    /// text.
    pub fn argv(
        &self,
        spec: &Plan,
        output: &Path,
        first: Option<&Path>,
        last: Option<&Path>,
        references: &[ReferenceArg<'_>],
    ) -> Vec<OsString> {
        let mut argv: Vec<OsString> = Vec::with_capacity(self.fixed_args.len() + 16);
        argv.push(self.program.clone().into_os_string());
        argv.extend(self.fixed_args.iter().map(OsString::from));
        argv.push(OsString::from("-p"));
        argv.push(OsString::from(&spec.prompt));
        argv.push(OsString::from("--width"));
        argv.push(OsString::from(spec.size.width.to_string()));
        argv.push(OsString::from("--height"));
        argv.push(OsString::from(spec.size.height.to_string()));
        argv.push(OsString::from("--seconds"));
        argv.push(OsString::from(spec.seconds.to_string()));
        if let Some(first) = first {
            argv.push(OsString::from("--first-frame"));
            argv.push(first.as_os_str().to_os_string());
        }
        if let Some(last) = last {
            argv.push(OsString::from("--last-frame"));
            argv.push(last.as_os_str().to_os_string());
        }
        // References follow the anchors in request order, one flag each.
        for reference in references {
            argv.push(OsString::from(reference.flag()));
            argv.push(reference.path.as_os_str().to_os_string());
        }
        argv.push(OsString::from("--seed"));
        argv.push(OsString::from(spec.seed.to_string()));
        argv.push(OsString::from("-o"));
        argv.push(output.as_os_str().to_os_string());
        argv
    }
}

impl Renderer for CommandRenderer {
    fn state(&self) -> RendererState {
        RendererState::Ready
    }

    fn render(&self, spec: &Plan, output: &Path) -> Result<(), RenderError> {
        let first = MediaFile::write(spec.first_frame.as_ref(), output, "first")?;
        let last = MediaFile::write(spec.last_frame.as_ref(), output, "last")?;
        let files = spec
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| MediaFile::write_reference(reference, output, index))
            .collect::<Result<Vec<_>, _>>()?;
        let references = files
            .iter()
            .zip(&spec.references)
            .map(|(file, media)| ReferenceArg {
                path: file.path(),
                kind: media.kind,
                include_audio: media.include_audio,
            })
            .collect::<Vec<_>>();
        let argv = self.argv(
            spec,
            output,
            first.as_ref().map(MediaFile::path),
            last.as_ref().map(MediaFile::path),
            &references,
        );
        let mut command = Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
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
                        break None;
                    }
                    thread::sleep(RENDER_POLL);
                }
                Err(error) => {
                    stop_renderer(&mut child);
                    return Err(RenderError::new(
                        "renderer_spawn_failed",
                        format!("could not wait for renderer: {error}"),
                    ));
                }
            }
        };
        // Descendants must release stderr even if their parent exited first.
        stop_renderer(&mut child);
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

fn stop_renderer(child: &mut Child) {
    #[cfg(unix)]
    // The child leads the process group created for this render only.
    unsafe {
        libc::killpg(child.id() as i32, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = child.kill();
    let _ = child.wait();
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

/// Parsed HTTP/1.1 request. A query string is not part of the contract, so it
/// is stripped from `path` and never carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Request {
    pub method: String,
    pub path: String,
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
/// Probe a visual reference for the `auto` ratio. An image carries its size in
/// its header; a video's needs FFprobe, so the bytes are written beside the
/// output and removed at once. No job exists yet, so the name is unique per
/// call rather than per id.
fn probe_visual(media: &ReferenceMedia, out_dir: &Path) -> Option<(u32, u32)> {
    if media.kind == ReferenceKind::Image {
        return image_size(&media.bytes);
    }

    static PROBES: AtomicU64 = AtomicU64::new(0);
    let extension = media.extension()?;
    let tag = PROBES.fetch_add(1, Ordering::Relaxed);
    let path = out_dir.join(format!(".probe-{tag}.{extension}"));
    std::fs::write(&path, &media.bytes).ok()?;
    let size = probe_visual_size(&path).ok();
    let _ = std::fs::remove_file(&path);

    size
}

pub struct VideoServer {
    store: Arc<VideoStore>,
    renderer: SharedRenderer,
    out_dir: PathBuf,
}

impl VideoServer {
    pub fn new(store: Arc<VideoStore>, renderer: SharedRenderer, out_dir: PathBuf) -> Self {
        Self {
            store,
            renderer,
            out_dir,
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
            ("GET", Route::Video(id)) => self.retrieve(&id),
            ("GET" | "HEAD", Route::Content(id)) => {
                self.content(&id, request.method.as_str() == "HEAD")
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
        // `aspect_ratio: auto` over a reference needs that reference's pixel
        // size, which the body does not state; the probe reads it here.
        let probe = |media: &ReferenceMedia| probe_visual(media, &self.out_dir);
        let plan = match parse_create(&http.body).and_then(|body| body.plan_with(&probe)) {
            Ok(plan) => plan,
            Err(error) => return H3Response::error(&error),
        };
        let video = match self.store.submit(&plan, unix_now()) {
            Ok(video) => video,
            Err(error) => return H3Response::error(&error),
        };
        let id = video.id.clone();
        // The store admits one job at a time. A create that does not get the
        // slot waits, and its record says so: `queued` with its position.
        pump(
            Arc::clone(&self.store),
            Arc::clone(&self.renderer),
            self.out_dir.clone(),
        );
        // The contract's create answer is the id alone
        // (`full-2k-t2va-h3-base.sh:24`); the task record is the query route's.
        H3Response::json(200, json!({ "id": id }).to_string())
    }

    fn retrieve(&self, id: &str) -> H3Response {
        match self.store.view(id) {
            Some(view) => H3Response::json(200, view.to_json()),
            None => H3Response::error(&unknown_video(id)),
        }
    }

    fn content(&self, id: &str, head_only: bool) -> H3Response {
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
}

/// Admit the next queued job when the render slot is free, and render it on its
/// own thread; that thread admits the next one when it finishes. One device
/// renders one request at a time, so the store hands out a single slot and a
/// create that finds it taken leaves its job `queued` with a position.
///
/// Called at create and at every render's end, which is what keeps the queue
/// moving without a scheduler thread.
fn pump(store: Arc<VideoStore>, renderer: SharedRenderer, out_dir: PathBuf) {
    loop {
        let Some(Admitted { id, plan }) = store.admit_next() else {
            return;
        };
        let output = out_dir.join(format!("{id}.mp4"));
        let (job_store, job_renderer, job_dir) =
            (Arc::clone(&store), Arc::clone(&renderer), out_dir.clone());
        let thread_id = id.clone();
        let spawned = thread::Builder::new()
            .name(format!("h3-render-{id}"))
            .spawn(move || {
                // Reserve the name even if a clock rollback ever repeats an id.
                let result = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&output)
                    .map_err(|error| {
                        let code = match error.kind() {
                            io::ErrorKind::AlreadyExists => "render_output_exists",
                            _ => "render_failed",
                        };
                        RenderError::new(code, format!("cannot reserve render output: {error}"))
                    })
                    .and_then(|file| {
                        drop(file);
                        job_renderer.render(&plan, &output)
                    });
                finish(&job_store, &thread_id, &output, result);
                pump(job_store, job_renderer, job_dir);
            });
        if spawned.is_ok() {
            return;
        }
        // The thread never started, so nothing will free the slot it took;
        // fail the job here and let the loop admit the next one.
        store.fail(
            &id,
            unix_now(),
            VideoError::new(
                "renderer_spawn_failed",
                "could not start the render thread".to_string(),
            ),
        );
    }
}

/// Record a render's outcome: a written file completes the job, anything else
/// fails it under the code the renderer named.
fn finish(store: &VideoStore, id: &str, output: &Path, result: Result<(), RenderError>) {
    let missing = |detail: String| VideoError::new("render_output_missing", detail);

    match result {
        Ok(()) => match std::fs::metadata(output) {
            Ok(meta) if meta.len() > 0 => {
                store.complete(id, unix_now(), output.to_path_buf());
            }
            Ok(_) => {
                store.fail(
                    id,
                    unix_now(),
                    missing(format!(
                        "renderer wrote an empty file at {}",
                        output.display()
                    )),
                );
            }
            Err(error) => {
                store.fail(
                    id,
                    unix_now(),
                    missing(format!(
                        "renderer produced no readable output at {}: {error}",
                        output.display()
                    )),
                );
            }
        },
        Err(error) => {
            store.fail(id, unix_now(), VideoError::new(error.code, error.message));
        }
    }
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
    Unknown,
}

fn route(path: &str) -> Route {
    if path == "/v1/videos" {
        return Route::Videos;
    }
    let Some(rest) = path.strip_prefix("/v1/videos/") else {
        return Route::Unknown;
    };
    if rest.is_empty() {
        return Route::Unknown;
    }
    if let Some(id) = rest.strip_suffix("/content") {
        if id.is_empty() || id.contains('/') {
            return Route::Unknown;
        }
        return Route::Content(id.to_string());
    }
    if rest.contains('/') {
        return Route::Unknown;
    }
    Route::Video(rest.to_string())
}

/// The server's clock, in whole seconds since the epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Read one request. A query string is dropped: the contract has no query
/// parameters. Content-Length only: the bridge serves JSON and file bodies,
/// and explicit refusal is safer than a half-supported chunked path.
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
    let (method, path, length) = {
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
        let path = target.split('?').next().unwrap_or(target);

        if header_chunked(head) {
            return Err(ReadError::Chunked);
        }
        let length = content_length(head);
        if length < 0 || length as usize > MAX_BODY {
            return Err(ReadError::TooLarge);
        }
        (method.to_string(), path.to_string(), length)
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

/// Bound connection threads separately from retained generation jobs.
pub fn serve(listener: TcpListener, server: Arc<VideoServer>) -> io::Result<()> {
    let clients = Arc::new(AtomicUsize::new(0));
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let _ = start_client(
                    stream,
                    Arc::clone(&server),
                    Arc::clone(&clients),
                    CLIENT_LIMIT,
                    CLIENT_TIMEOUT,
                );
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

struct ClientLease(Arc<AtomicUsize>);

impl Drop for ClientLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn start_client(
    mut stream: TcpStream,
    server: Arc<VideoServer>,
    clients: Arc<AtomicUsize>,
    limit: usize,
    timeout: Duration,
) -> io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    if clients
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
            (count < limit).then_some(count + 1)
        })
        .is_err()
    {
        return stream.write_all(
            &H3Response::error(&ApiError::unavailable(
                "server_busy",
                "Too many active connections.",
            ))
            .to_bytes(),
        );
    }
    let lease = ClientLease(clients);
    thread::Builder::new().spawn(move || {
        let _lease = lease;
        let _ = answer(&server, &mut stream);
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_core::minimax_h3::local_api::Task;
    use ds4_core::minimax_h3::video_api::VideoSize;
    use serde_json::Value;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;

    #[test]
    fn idle_connections_are_bounded() {
        let (server, dir) = test_server(Arc::new(DisabledRenderer), 1);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let clients = Arc::new(AtomicUsize::new(0));
        let connect = || {
            let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let (stream, _) = listener.accept().unwrap();
            start_client(
                stream,
                Arc::clone(&server),
                Arc::clone(&clients),
                1,
                Duration::from_millis(60),
            )
            .unwrap();
            client
        };
        let mut idle = connect();
        let mut excess = connect();
        let mut response = String::new();
        excess.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"), "{response}");

        idle.read_to_end(&mut Vec::new()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while clients.load(Ordering::Relaxed) != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(clients.load(Ordering::Relaxed), 0);
        let mut next = connect();
        next.write_all(b"GET /missing HTTP/1.1\r\nHost: local\r\n\r\n")
            .unwrap();
        response.clear();
        next.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The official t2va body (`full-2k-t2va-h3-base.sh:6-21`), byte for
    /// byte in shape: five fields, no conditions, 768P 16:9, ten seconds.
    const T2VA: &str = r#"{
        "task": "t2va",
        "prompt": "a red cube on a wooden table",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10},
        "seed": 0
    }"#;

    /// One 1x1 PNG, base64, as an FL2VA keyframe.
    const PNG_1PX: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    struct FnRenderer<F> {
        func: F,
    }

    impl<F> Renderer for FnRenderer<F>
    where
        F: Fn(&Plan, &Path) -> Result<(), RenderError> + Send + Sync,
    {
        fn state(&self) -> RendererState {
            RendererState::Ready
        }

        fn render(&self, spec: &Plan, output: &Path) -> Result<(), RenderError> {
            (self.func)(spec, output)
        }
    }

    fn renderer<F>(func: F) -> SharedRenderer
    where
        F: Fn(&Plan, &Path) -> Result<(), RenderError> + Send + Sync + 'static,
    {
        Arc::new(FnRenderer { func })
    }

    fn write_output(output: &Path) -> Result<(), RenderError> {
        std::fs::write(output, b"fake-mp4")
            .map_err(|error| RenderError::new("render_failed", error.to_string()))
    }

    /// A renderer that stays in `running` until the test releases it. A check
    /// for "content before completion" is otherwise a race against the render
    /// thread: the stub writes its 8-byte file faster than the next request.
    fn gated_renderer() -> (SharedRenderer, std::sync::mpsc::Sender<()>) {
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let wait = std::sync::Mutex::new(wait);
        let renderer = renderer(move |_, output| {
            let _ = wait.lock().unwrap().recv();
            write_output(output)
        });
        (renderer, release)
    }

    fn test_server(renderer: SharedRenderer, max_jobs: usize) -> (Arc<VideoServer>, PathBuf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "h3-video-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let store = Arc::new(VideoStore::new(max_jobs));
        (
            Arc::new(VideoServer::new(store, renderer, dir.clone())),
            dir,
        )
    }

    fn request(method: &str, path: &str) -> H3Request {
        H3Request {
            method: method.to_string(),
            path: path.to_string(),
            body: Vec::new(),
        }
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

    fn post(server: &VideoServer, body: &str) -> H3Response {
        server.handle(&H3Request {
            body: body.as_bytes().to_vec(),
            ..request("POST", "/v1/videos")
        })
    }

    fn get(server: &VideoServer, path: &str) -> H3Response {
        server.handle(&request("GET", path))
    }

    /// A `data:` URI with the bytes base64-encoded, as the contract requires.
    fn data_uri(media_type: &str, bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);

        for chunk in bytes.chunks(3) {
            let value = (u32::from(chunk[0]) << 16)
                | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
                | u32::from(*chunk.get(2).unwrap_or(&0));
            text.push(ALPHABET[(value >> 18) as usize & 63] as char);
            text.push(ALPHABET[(value >> 12) as usize & 63] as char);
            text.push(match chunk.len() > 1 {
                true => ALPHABET[(value >> 6) as usize & 63] as char,
                false => '=',
            });
            text.push(match chunk.len() > 2 {
                true => ALPHABET[value as usize & 63] as char,
                false => '=',
            });
        }

        format!("data:{media_type};base64,{text}")
    }

    /// A PNG signature and IHDR, which is all the size reader needs.
    fn png_of(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes
    }

    fn create(server: &VideoServer, body: &str) -> String {
        let response = post(server, body);
        assert_eq!(response.status, 200, "{}", response.status);
        json_of(&response)["id"].as_str().unwrap().to_string()
    }

    fn plan() -> Plan {
        Plan {
            task: Task::T2va,
            prompt: "p".to_string(),
            size: VideoSize::new(1344, 768),
            frames: 243,
            seconds: 10,
            seed: 0,
            first_frame: None,
            last_frame: None,
            references: Vec::new(),
        }
    }

    #[test]
    fn existing_outputs_are_refused() {
        let (server, dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let video = server.store.submit(&plan(), unix_now()).unwrap();
        let output = dir.join(format!("{}.mp4", video.id));
        std::fs::write(&output, b"saved clip").unwrap();

        pump(
            Arc::clone(&server.store),
            Arc::clone(&server.renderer),
            dir.clone(),
        );
        let failed = poll_until(&server, &video.id, "failed");
        assert_eq!(failed["error"]["code"], "render_output_exists");
        assert_eq!(std::fs::read(&output).unwrap(), b"saved clip");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn restart_preserves_outputs() {
        let (server, dir) = test_server(
            renderer(|_, output| {
                std::fs::write(output, b"first clip").unwrap();
                Ok(())
            }),
            8,
        );
        let first = create(&server, T2VA);
        poll_until(&server, &first, "completed");
        drop(server);

        let server = VideoServer::new(
            Arc::new(VideoStore::new(8)),
            renderer(|_, output| {
                std::fs::write(output, b"second clip").unwrap();
                Ok(())
            }),
            dir.clone(),
        );
        let second = create(&server, T2VA);
        poll_until(&server, &second, "completed");

        assert_eq!(
            std::fs::read(dir.join(format!("{first}.mp4"))).unwrap(),
            b"first clip"
        );
        assert_ne!(first, second);
        assert_eq!(
            std::fs::read(dir.join(format!("{second}.mp4"))).unwrap(),
            b"second clip"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_three_routes_serve_create_poll_and_content() {
        let (renderer, release) = gated_renderer();
        let (server, _dir) = test_server(renderer, 8);

        let response = post(&server, T2VA);
        assert_eq!(response.status, 200);
        let created = json_of(&response);
        assert_eq!(created.as_object().unwrap().len(), 1, "the id alone");
        let id = created["id"].as_str().unwrap().to_string();

        let early = server.handle(&request("GET", &format!("/v1/videos/{id}/content")));
        assert_eq!(early.status, 404);
        assert_eq!(json_of(&early)["error"]["code"], "video_not_ready");

        // Only now may the render finish; until this point the job cannot be
        // completed, so the 404 above is a decision, not a scheduling accident.
        release.send(()).expect("the renderer is still waiting");

        let done = poll_until(&server, &id, "completed");
        assert!(!done["completed_at"].is_null());
        assert_eq!(done["size"], "1344x768");
        assert_eq!(done["seconds"], 10);
        assert_eq!(done["frames"], 243);

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
        assert!(
            head.to_bytes().ends_with(b"\r\n\r\n"),
            "HEAD carries no body"
        );
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
        let id = create(&server, T2VA);
        poll_until(&server, &id, "in_progress");
        release.store(true, Ordering::SeqCst);
        poll_until(&server, &id, "completed");
    }

    #[test]
    fn a_second_job_waits_in_the_queue_with_its_position() {
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
        let record = |id: &str| json_of(&get(&server, &format!("/v1/videos/{id}")));

        let first = create(&server, T2VA);
        poll_until(&server, &first, "in_progress");

        // One device renders one job, so the next two wait, in arrival order.
        let second = create(&server, T2VA);
        let third = create(&server, T2VA);
        assert_eq!(record(&first)["queue_position"], Value::Null);
        assert_eq!(record(&second)["status"], "queued");
        assert_eq!(record(&second)["queue_position"], 1);
        assert_eq!(record(&third)["queue_position"], 2);

        // Releasing the renderer moves the queue one job at a time.
        release.store(true, Ordering::SeqCst);
        assert_eq!(
            poll_until(&server, &first, "completed")["queue_position"],
            Value::Null
        );
        poll_until(&server, &second, "completed");
        assert_eq!(
            poll_until(&server, &third, "completed")["queue_position"],
            Value::Null
        );
    }

    #[test]
    fn command_renderer_carries_references_in_order() {
        let dir = std::env::temp_dir().join(format!("h3-ref-argv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("video_7.mp4");
        let mut spec = plan();
        spec.task = Task::Ref2va;
        spec.references = vec![
            ReferenceMedia {
                kind: ReferenceKind::Image,
                media_type: "image/png".to_string(),
                bytes: b"image".to_vec(),
                include_audio: false,
            },
            ReferenceMedia {
                kind: ReferenceKind::Video,
                media_type: "video/mp4".to_string(),
                bytes: b"video".to_vec(),
                include_audio: true,
            },
            ReferenceMedia {
                kind: ReferenceKind::Audio,
                media_type: "audio/mpeg".to_string(),
                bytes: b"audio".to_vec(),
                include_audio: false,
            },
        ];

        let files = spec
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| MediaFile::write_reference(reference, &output, index))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(std::fs::read(files[1].path()).unwrap(), b"video");
        let references = files
            .iter()
            .zip(&spec.references)
            .map(|(file, media)| ReferenceArg {
                path: file.path(),
                kind: media.kind,
                include_audio: media.include_audio,
            })
            .collect::<Vec<_>>();

        let renderer = CommandRenderer::new("/bin/true", Vec::new(), Duration::from_secs(5));
        let argv: Vec<String> = renderer
            .argv(&spec, &output, None, None, &references)
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect();

        // The reference CLI's own spellings (`main.c:438-490`).
        let flags: Vec<(usize, &str)> = argv
            .iter()
            .enumerate()
            .filter(|(_, arg)| arg.starts_with("--ref-"))
            .map(|(index, arg)| (index, arg.as_str()))
            .collect();
        assert_eq!(
            flags.iter().map(|(_, flag)| *flag).collect::<Vec<_>>(),
            vec!["--ref-image", "--ref-video", "--ref-audio"]
        );
        for (index, _) in &flags {
            assert_eq!(
                argv[index + 1],
                files[flags.iter().position(|(at, _)| at == index).unwrap()]
                    .path()
                    .to_string_lossy()
            );
        }

        let paths: Vec<PathBuf> = files.iter().map(|file| file.path().to_path_buf()).collect();
        drop(files);
        for path in paths {
            assert!(!path.exists(), "the reference outlives its render");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_ref2va_body_reaches_the_renderer_with_its_references() {
        let seen: Arc<Mutex<Option<Plan>>> = Arc::new(Mutex::new(None));
        let captured = Arc::clone(&seen);
        let (server, _dir) = test_server(
            renderer(move |spec, output| {
                *captured.lock().unwrap() = Some(spec.clone());
                write_output(output)
            }),
            8,
        );

        let body = format!(
            r#"{{"task":"ref2va","prompt":"p","conditions":[
                {{"type":"video","uri":"{video}","role":"reference"}},
                {{"type":"audio","uri":"{audio}","role":"reference"}}],
                "target":{{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5}},
                "seed":1}}"#,
            video = data_uri("video/mp4", b"video"),
            audio = data_uri("audio/mpeg", b"audio"),
        );
        let id = create(&server, &body);

        let record = poll_until(&server, &id, "completed");
        assert_eq!(record["frames"], 124);
        assert_eq!(record["size"], "1344x768");

        let plan = seen.lock().unwrap().clone().expect("the render ran");
        assert_eq!(plan.task, Task::Ref2va);
        assert_eq!(plan.references.len(), 2);
        assert_eq!(plan.references[0].kind, ReferenceKind::Video);
        assert!(plan.references[0].include_audio);
        assert_eq!(plan.references[1].kind, ReferenceKind::Audio);
        assert!(plan.first_frame.is_none());
    }

    #[test]
    fn an_auto_ref2va_takes_the_canvas_from_the_first_image() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let body = format!(
            r#"{{"task":"ref2va","prompt":"p","conditions":[
                {{"type":"image","uri":"{image}","role":"reference"}}],
                "target":{{"short_edge":768,"aspect_ratio":"auto","duration_seconds":4}},
                "seed":0}}"#,
            image = data_uri("image/png", &png_of(1, 1)),
        );
        let id = create(&server, &body);

        let response = get(&server, &format!("/v1/videos/{id}"));
        // A 1x1 reference is square, so auto resolves to the square tier.
        assert_eq!(json_of(&response)["size"], "768x768");
    }

    #[test]
    fn the_renderer_receives_the_resolved_plan() {
        let seen: Arc<Mutex<Option<Plan>>> = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&seen);
        let (server, _dir) = test_server(
            renderer(move |spec, output| {
                *slot.lock().unwrap() = Some(spec.clone());
                write_output(output)
            }),
            8,
        );
        let id = create(&server, T2VA);
        poll_until(&server, &id, "completed");
        let plan = seen.lock().unwrap().clone().expect("one render");
        assert_eq!(plan.task, Task::T2va);
        assert_eq!(plan.prompt, "a red cube on a wooden table");
        assert_eq!(plan.size, VideoSize::new(1344, 768));
        assert_eq!(plan.frames, 243);
        assert_eq!(plan.seconds, 10);
        assert_eq!(plan.seed, 0);
        assert!(plan.first_frame.is_none());
        assert!(plan.last_frame.is_none());
    }

    #[test]
    fn a_fl2va_keyframe_reaches_the_renderer_as_an_anchor() {
        let seen: Arc<Mutex<Option<Plan>>> = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&seen);
        let (server, _dir) = test_server(
            renderer(move |spec, output| {
                *slot.lock().unwrap() = Some(spec.clone());
                write_output(output)
            }),
            8,
        );
        let body = format!(
            r#"{{"task":"fl2va","prompt":"p","conditions":[{{"type":"image",
               "uri":"data:image/png;base64,{PNG_1PX}","role":"keyframe","frame_index":0}}],
               "target":{{"short_edge":768,"aspect_ratio":"auto","duration_seconds":8}},"seed":3}}"#
        );
        let id = create(&server, &body);
        poll_until(&server, &id, "completed");
        let plan = seen.lock().unwrap().clone().expect("one render");
        let anchor = plan.first_frame.expect("first frame");
        assert_eq!(anchor.media_type, "image/png");
        assert_eq!(plan.size, VideoSize::new(768, 768), "a square anchor");
        assert_eq!(plan.frames, 192);
        assert_eq!(plan.seed, 3);
    }

    #[test]
    fn a_refused_body_is_named_on_the_wire() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let image = format!("data:image/png;base64,{PNG_1PX}");
        let target = r#""target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":8}"#;
        for (body, code) in [
            (r#"not json"#.to_string(), "invalid_json"),
            (r#"{"prompt":"p"}"#.to_string(), "invalid_task"),
            (r#"{"task":"t2va"}"#.to_string(), "invalid_prompt"),
            (
                r#"{"task":"t2va","prompt":"","conditions":[],"seed":0}"#.to_string(),
                "invalid_prompt",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],"seed":0}"#.to_string(),
                "invalid_target",
            ),
            (
                format!(r#"{{"task":"t2va","prompt":"p","conditions":[],{target}}}"#),
                "invalid_seed",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],"target":
                   {"short_edge":768,"aspect_ratio":"16:9","duration_seconds":10},
                   "seed":0,"size":"1x1"}"#
                    .to_string(),
                "unknown_field",
            ),
            // ref2va with no reference media conditions nothing.
            (
                r#"{"task":"ref2va","prompt":"p","conditions":[],
                   "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":8},
                   "seed":0}"#
                    .to_string(),
                "invalid_condition",
            ),
            // A keyframe belongs to fl2va, not to t2va.
            (
                format!(
                    r#"{{"task":"t2va","prompt":"p","conditions":[{{"type":"image",
                       "uri":"{image}","role":"keyframe","frame_index":0}}],{target},"seed":0}}"#
                ),
                "invalid_condition",
            ),
            // Two keyframes on index 0 name the same frame twice.
            (
                format!(
                    r#"{{"task":"fl2va","prompt":"p","conditions":[
                       {{"type":"image","uri":"{image}","role":"keyframe","frame_index":0}},
                       {{"type":"image","uri":"{image}","role":"keyframe","frame_index":0}}],
                       {target},"seed":0}}"#
                ),
                "invalid_condition",
            ),
            // A reference role needs ref2va's reference-media path.
            (
                format!(
                    r#"{{"task":"fl2va","prompt":"p","conditions":[{{"type":"image",
                       "uri":"{image}","role":"reference"}}],{target},"seed":0}}"#
                ),
                "unsupported_condition",
            ),
            (
                format!(
                    r#"{{"task":"fl2va","prompt":"p","conditions":[{{"type":"image",
                       "uri":"https://cdn.example/x.png","role":"keyframe","frame_index":0}}],
                       {target},"seed":0}}"#
                ),
                "unsupported_uri",
            ),
            (
                format!(
                    r#"{{"task":"fl2va","prompt":"p","conditions":[{{"type":"image",
                       "uri":"/tmp/x.png","role":"keyframe","frame_index":0}}],{target},"seed":0}}"#
                ),
                "invalid_uri",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],
                   "target":{"short_edge":768,"aspect_ratio":"21:9","duration_seconds":8},
                   "seed":0}"#
                    .to_string(),
                "invalid_aspect_ratio",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],
                   "target":{"short_edge":768,"aspect_ratio":"5:4","duration_seconds":8},
                   "seed":0}"#
                    .to_string(),
                "invalid_aspect_ratio",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],
                   "target":{"short_edge":480,"aspect_ratio":"16:9","duration_seconds":8},
                   "seed":0}"#
                    .to_string(),
                "unsupported_short_edge",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],
                   "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":16},
                   "seed":0}"#
                    .to_string(),
                "invalid_duration",
            ),
            (
                r#"{"task":"t2va","prompt":"p","conditions":[],
                   "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":0},
                   "seed":0}"#
                    .to_string(),
                "invalid_duration",
            ),
        ] {
            let response = post(&server, &body);
            assert_eq!(response.status, 400, "{code}: {body}");
            assert_eq!(json_of(&response)["error"]["code"], code, "{body}");
        }
    }

    #[test]
    fn only_the_three_contract_routes_exist() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let id = create(&server, T2VA);
        poll_until(&server, &id, "completed");
        for (method, path) in [
            ("GET", "/v1/videos".to_string()),
            ("DELETE", format!("/v1/videos/{id}")),
            ("GET", "/v1/models".to_string()),
            ("GET", "/v2/video_generation".to_string()),
            ("GET", "/v1/videos/".to_string()),
        ] {
            let response = server.handle(&request(method, &path));
            assert_eq!(response.status, 404, "{method} {path}");
            assert_eq!(json_of(&response)["error"]["code"], "unknown_url");
        }
        let missing = server.handle(&request("GET", "/v1/videos/video_99"));
        assert_eq!(missing.status, 404);
        assert_eq!(json_of(&missing)["error"]["code"], "video_not_found");
    }

    #[test]
    fn renderer_failure_and_missing_output_become_failed_jobs() {
        let (server, _dir) = test_server(
            renderer(|_, _| Err(RenderError::new("render_failed", "exit 1"))),
            8,
        );
        let id = create(&server, T2VA);
        let failed = poll_until(&server, &id, "failed");
        assert_eq!(failed["error"]["code"], "render_failed");
        let content = server.handle(&request("GET", &format!("/v1/videos/{id}/content")));
        assert_eq!(content.status, 404);

        let (server, _dir) = test_server(renderer(|_, _| Ok(())), 8);
        let id = create(&server, T2VA);
        let failed = poll_until(&server, &id, "failed");
        assert_eq!(failed["error"]["code"], "render_output_missing");
    }

    #[test]
    fn a_disabled_renderer_answers_503_and_stores_nothing() {
        let (server, _dir) = test_server(Arc::new(DisabledRenderer), 8);
        let response = post(&server, T2VA);
        assert_eq!(response.status, 503);
        assert_eq!(json_of(&response)["error"]["code"], "renderer_disabled");
        assert!(server.store().is_empty());
    }

    #[test]
    fn the_store_bound_is_reported_as_queue_full() {
        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 1);
        create(&server, T2VA);
        let response = post(&server, T2VA);
        assert_eq!(response.status, 429);
        assert_eq!(json_of(&response)["error"]["code"], "queue_full");
    }

    #[test]
    fn command_renderer_keeps_request_values_as_separate_argv() {
        let renderer = CommandRenderer::new(
            "/path/to/h3",
            vec![
                "-d".to_string(),
                "/path/to/MiniMax-H3".to_string(),
                "--steps".to_string(),
                "20".to_string(),
            ],
            Duration::from_secs(60),
        );
        let mut spec = plan();
        spec.prompt = "a cube; rm -rf / --width 9".to_string();
        spec.size = VideoSize::new(512, 768);
        spec.seconds = 4;
        spec.frames = 107;
        spec.seed = 9;
        let argv: Vec<String> = renderer
            .argv(&spec, Path::new("/out/video_1.mp4"), None, None, &[])
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect();
        // Exact reference spellings: `-p`, `-o`, `--width`, `--height`,
        // `--seconds`, `--seed` (`/data/h3.c/main.c:22-29`, `:292-296`,
        // `:330`, `:361-381`).
        assert_eq!(
            argv,
            vec![
                "/path/to/h3",
                "-d",
                "/path/to/MiniMax-H3",
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
                "--seed",
                "9",
                "-o",
                "/out/video_1.mp4",
            ]
        );
        assert_eq!(argv.len(), 17, "a prompt never adds argv entries");
        assert!(
            !argv.iter().any(|arg| arg == "--frames"),
            "the bridge must not invent a frames conversion"
        );
    }

    #[test]
    fn command_renderer_carries_both_anchors_and_removes_them() {
        let dir = std::env::temp_dir().join(format!("h3-anchor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("video_1.mp4");
        let mut spec = plan();
        for (tag, frame) in [
            ("first", &mut spec.first_frame),
            ("last", &mut spec.last_frame),
        ] {
            *frame = Some(AnchorImage {
                media_type: "image/png".to_string(),
                bytes: tag.as_bytes().to_vec(),
            });
        }

        let first = MediaFile::write(spec.first_frame.as_ref(), &output, "first")
            .unwrap()
            .expect("first anchor");
        let last = MediaFile::write(spec.last_frame.as_ref(), &output, "last")
            .unwrap()
            .expect("last anchor");
        assert_eq!(std::fs::read(first.path()).unwrap(), b"first");
        assert_eq!(first.path().file_stem().unwrap(), "video_1.first");
        assert_eq!(last.path().extension().unwrap(), "png");

        let renderer = CommandRenderer::new("/bin/true", Vec::new(), Duration::from_secs(5));
        let argv: Vec<String> = renderer
            .argv(&spec, &output, Some(first.path()), Some(last.path()), &[])
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect();
        let flag = argv.iter().position(|arg| arg == "--first-frame").unwrap();
        assert_eq!(argv[flag + 1], first.path().to_string_lossy());
        let flag = argv.iter().position(|arg| arg == "--last-frame").unwrap();
        assert_eq!(argv[flag + 1], last.path().to_string_lossy());

        let first_path = first.path().to_path_buf();
        let last_path = last.path().to_path_buf();
        drop((first, last));
        assert!(!first_path.exists(), "the anchor outlives its render");
        assert!(!last_path.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn relative_render_paths_work() {
        let dir = PathBuf::from(format!("h3-relative-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("clip.mp4");
        let mut spec = plan();
        spec.first_frame = Some(AnchorImage {
            media_type: "image/png".to_string(),
            bytes: b"anchor bytes".to_vec(),
        });
        let script = "while [ $# -gt 0 ]; do case $1 in -o) output=$2; shift 2;; --first-frame) anchor=$2; shift 2;; *) shift;; esac; done; test -s \"$anchor\" || exit 77; printf rendered > \"$output\"";
        let renderer = CommandRenderer::new(
            "sh",
            vec!["-c".to_string(), script.to_string(), "--".to_string()],
            Duration::from_secs(5),
        );
        let result = renderer.render(&spec, &output);
        let bytes = std::fs::read(&output);
        std::fs::remove_dir_all(dir).unwrap();

        result.unwrap();
        assert_eq!(bytes.unwrap(), b"rendered");
    }

    #[test]
    fn command_renderer_maps_nonzero_exit_and_timeout() {
        if !Path::new("/bin/false").exists() {
            return;
        }
        let renderer = CommandRenderer::new("/bin/false", Vec::new(), Duration::from_secs(5));
        let error = renderer
            .render(&plan(), Path::new("/tmp/h3-video-never.mp4"))
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
                .render(&plan(), Path::new("/tmp/h3-video-never.mp4"))
                .unwrap_err();
            assert_eq!(error.code, "render_timeout", "{}", error.message);
        }
    }

    #[cfg(unix)]
    #[test]
    fn renderer_stops_descendants() {
        for (ending, expected) in [("wait", Some("render_timeout")), ("exit 0", None)] {
            let dir =
                std::env::temp_dir().join(format!("h3-descendant-{}-{ending}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let pid_file = dir.join("child.pid");
            let script = format!("sleep 60 & echo $! > \"$1\"; {ending}");
            let renderer = CommandRenderer::new(
                "/bin/sh",
                vec![
                    "-c".to_string(),
                    script,
                    "--".to_string(),
                    pid_file.display().to_string(),
                ],
                Duration::from_millis(60),
            );
            let output = dir.join("out.mp4");
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = thread::spawn(move || {
                sender.send(renderer.render(&plan(), &output)).unwrap();
            });
            let result = receiver.recv_timeout(Duration::from_secs(2));
            // Bound a regressed run without leaving its sleeping child behind.
            if result.is_err() {
                if let Ok(pid) = std::fs::read_to_string(&pid_file) {
                    unsafe {
                        libc::kill(pid.trim().parse().unwrap(), libc::SIGKILL);
                    }
                }
            }
            worker.join().unwrap();
            std::fs::remove_dir_all(dir).unwrap();
            let result = result.expect("renderer left a descendant holding stderr");
            match expected {
                Some(code) => assert_eq!(result.unwrap_err().code, code),
                None => result.unwrap(),
            }
        }
    }

    #[test]
    fn socket_serves_the_three_routes_over_tcp() {
        use std::io::Read;
        use std::net::TcpStream;

        let (server, _dir) = test_server(renderer(|_, output| write_output(output)), 8);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let serving = Arc::clone(&server);
        let thread = thread::spawn(move || serve_one(&listener, &serving).unwrap());

        let mut stream = TcpStream::connect(address).unwrap();
        let body = T2VA.as_bytes();
        let head = format!(
            "POST /v1/videos HTTP/1.1\r\nHost: h3\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(body).unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        thread.join().unwrap();

        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        let (_, body) = text.split_once("\r\n\r\n").unwrap();
        let created: Value = serde_json::from_str(body).unwrap();
        let id = created["id"].as_str().unwrap();
        assert!(server.store.get(id).is_some());
    }

    #[test]
    fn the_reader_drops_the_query_and_reads_the_body() {
        let wire = b"POST /v1/videos?limit=2 HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody".to_vec();
        let request = read_request(&mut wire.as_slice()).unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/videos");
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
