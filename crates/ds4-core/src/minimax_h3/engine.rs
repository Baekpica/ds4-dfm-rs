//! The resident render engine: the reference's interactive session.
//!
//! Caches the prompt embedding (`h3.c:143-166`, `h3.c:223-309`) and the DiT
//! prepared for that conditioning and request shape (`h3.c:168-190`). A
//! matching request avoids reading the text encoder and DiT weights again.
//!
//! The device handles are `!Send` (`ds4_sys::h3_dit::{Gpu, Tensor}` hold
//! `NonNull`), so the engine owns a thread and the host reaches it over a
//! channel. That also makes admission serial by construction: one render at a
//! time, which is what one GPU allows.
//!
//! Both caches identify the prompt, ordered media and their presentation.
//! The identity hashes decoded content, so equivalent files may share it;
//! names or VAE rows alone cannot identify what the vision encoder consumes.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::dit::{Dit, LoadOptions};
use super::host::{reuse_schedule, temporal, Rng, Schedule};
use super::media::{self, Fit};
use super::partition::{Catalog, Partition, TOKENIZER};
use super::reference::{self, fnv_f32};
use super::render::{check_frames, mux_mp4, Canvas, RgbFrames};
use super::sequence::{self, Spec};
use super::text_forward::{self, Embedding};
use super::tokenizer::{EmptyPrompt, Tokenizer};
use super::{condition, vae_audio, vae_encoder, vae_video};

/// How the delivered frame relates to the model canvas.
///
/// The reference serves the first two (`h3.c:1644-1660`) and refuses a frame
/// of a different aspect (`h3.c:505-520`), so `Anamorphic` is the named
/// deviation: each axis takes its own scale and the report carries both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presentation {
    /// The file is the model canvas; the resize is the identity.
    Native,
    /// A different pixel count of the same aspect: one uniform scale.
    Uniform,
    /// A different aspect: an anamorphic scale, per axis.
    Anamorphic,
}

/// One render request. The canvas is the reference's `h3_params` canvas; the
/// output canvas is native, so the resize step is the identity.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    pub prompt: String,
    /// The model canvas: denoised and decoded.
    pub width: i32,
    pub height: i32,
    /// The delivered frame. Equal to the canvas unless a presentation preset
    /// scales it (`presets.rs`); the render canvas stays the model canvas.
    pub output_width: i32,
    pub output_height: i32,
    /// How the frame relates to the canvas. The caller names it, so an
    /// anamorphic stretch is never inferred.
    pub presentation: Presentation,
    /// Requested frames, before `5 + 17k` alignment (`h3.c:867-870`).
    pub frames: i32,
    pub steps: usize,
    pub reuse: usize,
    pub seed: u64,
    /// FL2VA anchors: the first and last frame the clip must start and end on
    /// (`h3.c:1231-1257`). They condition the DiT; they are not delivered.
    pub first_frame: Option<PathBuf>,
    pub last_frame: Option<PathBuf>,
    /// Ref2VA references, in request order (`h3.c:1027-1050`). Mutually
    /// exclusive with the anchors (`h3.c:597-600`) and served from the
    /// Ref2VA partition (`h3.c:872-875`).
    pub references: Vec<reference::Reference>,
    pub output: PathBuf,
}

/// What the render did, for the operator's log and the gate.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderReport {
    pub frames: i32,
    pub evaluated: usize,
    /// The delivery scale per axis; `(1.0, 1.0)` when the file is the canvas.
    pub scale: (f64, f64),
    pub text_hit: bool,
    pub prepared_hit: bool,
    pub video_hash: u64,
    pub audio_hash: u64,
    pub frames_hash: u64,
    pub text_seconds: f64,
    pub load_seconds: f64,
    pub denoise_seconds: f64,
    pub audio_seconds: f64,
    pub video_seconds: f64,
    pub mux_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(String);

impl EngineError {
    fn new(detail: impl Into<String>) -> Self {
        Self(detail.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

/// `h3_prepared_key` (`h3.c:168-190`) reduced to the fields the port serves:
/// the conditioning, the shape and every setting that changes the prepared
/// graph. `cells` is `canvas.rope_scale()`; a different scale is a different
/// rope table and must not be reused.
///
/// The two upstream `slow*` and `reduce` fields do not exist here: the port
/// serves exactly one execution path, so they cannot vary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedKey {
    conditioning: [u8; 32],
    width: i32,
    height: i32,
    frames: i32,
    steps: usize,
    reuse: usize,
    rope: u32,
}

struct Prepared {
    key: PreparedKey,
    dit: Dit,
}

pub struct Engine {
    /// `h3_create`'s checkpoint check (`h3.c:425-452`), kept for the
    /// partition a request reads.
    catalog: Catalog,
    tokenizer: Tokenizer,
    /// `h3.c:223-309`: one conditioning entry is enough for a server; the
    /// reference keeps a small table and counts hits and misses.
    embedding: Option<([u8; 32], Embedding)>,
    prepared: Option<Prepared>,
}

impl Engine {
    /// Open the checkpoint metadata. No device state yet: the text encoder is
    /// loaded per encode, as the reference does (it frees the embedding after
    /// the DiT is prepared, `h3.c:1539`).
    pub fn open(model_dir: &Path) -> Result<Engine, EngineError> {
        let catalog =
            Catalog::open(model_dir).map_err(|error| EngineError::new(error.message()))?;
        let path = catalog.component(Partition::Fl2va, TOKENIZER);
        let tokenizer = Tokenizer::load(&path)
            .map_err(|error| EngineError::new(format!("tokenizer: {}", error.token())))?;

        Ok(Engine {
            catalog,
            tokenizer,
            embedding: None,
            prepared: None,
        })
    }

    fn component(&self, name: &str) -> PathBuf {
        self.catalog.component(Partition::Fl2va, name)
    }

    /// Which partition the request reads (`h3.c:872-875`): ordered references
    /// need the Ref2VA transformer, everything else the FL2VA one.
    pub fn partition(&self, references: usize) -> Result<Partition, EngineError> {
        self.catalog
            .partition(references)
            .map_err(|error| EngineError::new(error.message()))
    }

    pub fn render(&mut self, request: &RenderRequest) -> Result<RenderReport, EngineError> {
        // The presentation is the caller's claim about the delivery, so it is
        // validated rather than re-derived: a mismatch would either stretch a
        // frame silently or skip a promised resize.
        let delivered =
            request.output_width != request.width || request.output_height != request.height;
        let canvas = match request.presentation {
            Presentation::Native if delivered => {
                return Err(EngineError::new(
                    "native presentation requires the model canvas as the output size",
                ));
            }
            Presentation::Native => Canvas::new(request.width, request.height, 0, 0),
            Presentation::Uniform => Canvas::new_delivery(
                request.output_width,
                request.output_height,
                request.width,
                request.height,
            ),
            Presentation::Anamorphic => Canvas::new_anamorphic(
                request.output_width,
                request.output_height,
                request.width,
                request.height,
            ),
        }
        .map_err(|error| EngineError::new(format!("canvas: {error}")))?;
        let scale = canvas.scale();
        let frame_count = check_frames(request.frames).map_err(EngineError::new)?;
        let schedule = Schedule::serving(request.steps)
            .ok_or_else(|| EngineError::new("--steps is outside 2..=1000"))?;
        let selected = reuse_schedule(request.steps, request.reuse)
            .ok_or_else(|| EngineError::new("--reuse is outside 1..=32"))?;
        let evaluated = selected.iter().filter(|flag| **flag).count();

        let shape = temporal(frame_count);
        let (latent_w, latent_h) = canvas.latent();
        let partition = self.partition(request.references.len())?;

        // `h3.c:597-600`: an ordered-reference request carries no anchors.
        if !request.references.is_empty()
            && (request.first_frame.is_some() || request.last_frame.is_some())
        {
            return Err(EngineError::new(
                "full references cannot be combined with frame anchors",
            ));
        }

        // Anchors (`h3.c:1231-1345`): each frame is read at the canvas, encoded
        // by the video VAE, patchified and jittered before the layout is built,
        // because the cond row count is part of the packed signature. The same
        // pixels are the prompt's vision spans, so they are read once.
        let anchors = self.read_anchors(request, frame_count, latent_h, latent_w)?;

        // References (`h3.c:1027-1200`): decoded at their own canvases,
        // encoded and jittered in request order. Their rows join the anchors'
        // only in a request that cannot carry both.
        let references = self.read_references(request, shape.frame_count)?;
        let conditioning = conditioning_identity(&request.prompt, &anchors, &references);
        let (keyframes, mut video_rows) = (anchors.keyframes, anchors.rows);
        video_rows.extend_from_slice(&references.video_rows);
        let audio_rows = references.audio_rows.clone();

        // Conditioning: the reference memoizes the prompt embedding
        // (`h3.c:223-309`) because the encoder is the largest read of the run.
        // Position, presentation and original vision pixels belong to the key
        // along with the VAE rows; both consumers must see the same request.
        let text_started = Instant::now();
        let cached = self
            .embedding
            .as_ref()
            .and_then(|(key, text)| (*key == conditioning).then(|| text.clone()));
        let text_hit = cached.is_some();
        let text = match cached {
            Some(text) => text,
            None => {
                // `h3.c:1345-1436`: references take the Ref2VA presentation,
                // anchors the FL2VA one, and neither takes the plain text path.
                let text = if !references.pixels.is_empty() {
                    condition::reference_embedding(
                        &self.component("text_encoder"),
                        &self.tokenizer,
                        &request.prompt,
                        &references.pixels,
                    )
                    .map_err(EngineError::new)?
                } else if !anchors.pixels.is_empty() {
                    condition::anchor_embedding(
                        &self.component("text_encoder"),
                        &self.tokenizer,
                        &request.prompt,
                        &anchors.pixels,
                    )
                    .map_err(EngineError::new)?
                } else {
                    let ids = self
                        .tokenizer
                        .encode(&request.prompt, EmptyPrompt::Pad)
                        .map_err(|error| {
                            EngineError::new(format!("tokenizer: {}", error.token()))
                        })?;
                    text_forward::encode(&self.component("text_encoder"), &ids).map_err(
                        |error| EngineError::new(format!("text encoder: {}", error.token())),
                    )?
                };
                self.embedding = Some((conditioning, text.clone()));
                text
            }
        };
        let text_seconds = text_started.elapsed().as_secs_f64();

        let key = PreparedKey {
            conditioning,
            width: request.width,
            height: request.height,
            frames: frame_count,
            steps: request.steps,
            reuse: request.reuse,
            rope: canvas.rope_scale(false).to_bits(),
        };

        // Prepared DiT (`h3.c:168-190`, `h3.c:1505-1516`): a hit skips the
        // 61.7 GiB read and the whole preparation, including the AdaLN tree
        // and the rope tables. A miss frees the old graph first so both are
        // never resident.
        let load_started = Instant::now();
        let prepared_hit = self.prepared.as_ref().is_some_and(|held| held.key == key);
        if !prepared_hit {
            self.prepared = None;
            let spec = Spec {
                text_len: text.tokens(),
                latent_t: shape.video_t,
                latent_h,
                latent_w,
                audio_t: shape.audio_t,
                frame_count: shape.frame_count,
                keyframes: &keyframes,
                references: &references.layout,
            };
            let sequence = sequence::build(&spec)
                .map_err(|error| EngineError::new(format!("layout: {error:?}")))?;
            let dit = Dit::load_with(
                &self.catalog.component(partition, "transformer"),
                sequence,
                schedule,
                &text,
                canvas.rope_scale(false),
                LoadOptions::resident(),
            )
            .map_err(|error| EngineError::new(format!("transformer: {error}")))?;
            self.prepared = Some(Prepared { key, dit });
        }
        let load_seconds = load_started.elapsed().as_secs_f64();

        let dit = &mut self
            .prepared
            .as_mut()
            .expect("the prepared DiT was just inserted")
            .dit;

        // The leading spans are refreshed on every call, cached graph or fresh
        // (`h3_dit_reset_run`, `h3_dit.c:2408-2440`). With no media both counts
        // are zero and this only validates.
        dit.set_conditions(&video_rows, &audio_rows)
            .map_err(|error| EngineError::new(format!("conditions: {error}")))?;

        let mut video = vec![0.0f32; dit.video_elements()];
        let mut audio = vec![0.0f32; dit.audio_elements()];
        Rng::seed(request.seed).fill_normal(&mut video);
        Rng::seed(request.seed).fill_normal(&mut audio);

        let denoise_started = Instant::now();
        // The reference reports progress through `on_progress(phase, done,
        // total)` (`h3.c:649-676`); without a sink an in-process render is
        // silent until the MP4 lands, which reads as a stall. One line per
        // evaluated step, with the seconds since the previous one.
        let mut step_started = Instant::now();
        let mut progress = |step: usize, _latent: &[f32]| {
            eprintln!(
                "STEP {}/{} seconds={:.3} total={:.3}",
                step + 1,
                request.steps,
                step_started.elapsed().as_secs_f64(),
                denoise_started.elapsed().as_secs_f64()
            );
            step_started = Instant::now();
            Ok(())
        };
        dit.denoise_with(&mut video, &mut audio, request.reuse, Some(&mut progress))
            .map_err(|error| EngineError::new(format!("denoise: {error}")))?;
        let denoise_seconds = denoise_started.elapsed().as_secs_f64();

        let video_hash = fnv_f32(&video);
        let audio_hash = fnv_f32(&audio);

        // The prepared DiT stays resident across the decode, as in the
        // reference session. The one-shot CLI drops it here to lower the peak;
        // that is the documented difference between the two lifetimes
        // (`h3.h:180-184`).
        let audio_started = Instant::now();
        let waveform =
            vae_audio::decode(&self.component("audio_vae"), &audio, audio.len() / (32 * 2))
                .map_err(|error| EngineError::new(format!("audio VAE: {error}")))?;
        let audio_seconds = audio_started.elapsed().as_secs_f64();

        let video_started = Instant::now();
        let decoded = vae_video::decode(
            &self.component("video_vae/source"),
            &video,
            shape.video_t as usize,
            latent_h as usize,
            latent_w as usize,
        )
        .map_err(|error| EngineError::new(format!("video VAE: {error}")))?;
        let video_seconds = video_started.elapsed().as_secs_f64();

        let decoded_frames = small(decoded.frames)?;
        let decoded_width = small(decoded.width)?;
        let decoded_height = small(decoded.height)?;
        let frames_hash = fnv_f32(&decoded.rgb);
        let frames =
            RgbFrames::from_f32(&decoded.rgb, decoded_frames, decoded_width, decoded_height)
                .map_err(|error| EngineError::new(format!("rgb24: {error}")))?;

        // Delivery preset: the picture is the model canvas, the file is the
        // requested frame. The caller logs the scale as a named deviation.
        let frames = if delivered {
            frames
                .resized_to(request.output_width, request.output_height)
                .map_err(|error| EngineError::new(format!("delivery resize: {error}")))?
        } else {
            frames
        };

        let mux_started = Instant::now();
        let program = std::env::var_os("H3_FFMPEG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("ffmpeg"));
        mux_mp4(
            &program,
            &request.output,
            &frames,
            &waveform.pcm,
            small(waveform.samples)?,
            small(waveform.channels)?,
            small(waveform.sample_rate)?,
        )
        .map_err(|error| EngineError::new(format!("mux: {error}")))?;
        let mux_seconds = mux_started.elapsed().as_secs_f64();

        if !output_is_present(&request.output) {
            return Err(EngineError::new(format!(
                "mux wrote no output at {}",
                request.output.display()
            )));
        }

        Ok(RenderReport {
            frames: shape.frame_count,
            evaluated,
            scale,
            text_hit,
            prepared_hit,
            video_hash,
            audio_hash,
            frames_hash,
            text_seconds,
            load_seconds,
            denoise_seconds,
            audio_seconds,
            video_seconds,
            mux_seconds,
        })
    }

    /// Read, encode and jitter the ordered references (`h3.c:1027-1200`).
    ///
    /// A reference is decoded at its own canvas, never the render canvas, and
    /// a video is truncated to the request's frame count so its soundtrack and
    /// its rows describe the same clip. The twelve cap (`h3.c:588`) and the
    /// 15-second budget over the whole set (`h3.c:1171-1176`) live in
    /// `reference::prepare_set`, which the CLI renderer calls too.
    fn read_references(
        &self,
        request: &RenderRequest,
        frame_count: i32,
    ) -> Result<reference::Set, EngineError> {
        let video_vae = self.component("video_vae/source");
        let audio_vae = self.component("audio_vae");
        let paths = reference::Paths {
            video_vae: &video_vae,
            audio_vae: &audio_vae,
        };

        reference::prepare_set(
            &paths,
            &request.references,
            reference::ImageSize::Match,
            (request.width as u32, request.height as u32),
            frame_count as usize,
            request.seed,
        )
        .map_err(|error| EngineError::new(error.message()))
    }

    /// Read, encode, patchify and jitter the anchor frames (`h3.c:1231-1345`).
    ///
    /// The reference fits the first frame by stretching and the last by
    /// covering (`h3.c:1233-1255`), and the keyframe index is the clip edge, not
    /// the media's. Each anchor contributes the target canvas' own row span, so
    /// a latent of any other geometry is an error rather than a short span
    /// (`h3.c:1338-1341`).
    fn read_anchors(
        &self,
        request: &RenderRequest,
        frame_count: i32,
        latent_h: i32,
        latent_w: i32,
    ) -> Result<Anchors, EngineError> {
        let anchors = [
            (0, Fit::Stretch, request.first_frame.as_deref()),
            (frame_count - 1, Fit::Cover, request.last_frame.as_deref()),
        ];
        let mut out = Anchors {
            keyframes: Vec::new(),
            rows: Vec::new(),
            pixels: Vec::new(),
        };

        for (frame, fit, path) in anchors {
            let Some(path) = path else { continue };
            let values =
                media::read_image_f32(path, request.width as u32, request.height as u32, fit)
                    .map_err(|error| {
                        EngineError::new(format!("anchor {}: {}", path.display(), error.detail()))
                    })?;
            let latent = vae_encoder::encode(
                &self.component("video_vae/source"),
                &values,
                1,
                request.height as usize,
                request.width as usize,
                |_, _| {},
            )
            .map_err(|error| EngineError::new(format!("anchor {}: {error}", path.display())))?;

            if latent.time != 1
                || latent.height != latent_h as usize
                || latent.width != latent_w as usize
            {
                return Err(EngineError::new(
                    "anchor VAE produced unexpected latent geometry",
                ));
            }

            out.keyframes.push(frame);
            out.rows.extend(
                condition::anchor_rows(&latent, request.seed)
                    .map_err(|error| EngineError::new(format!("anchor rows: {error}")))?,
            );
            out.pixels.push(condition::AnchorPixels {
                values,
                height: request.height as u32,
                width: request.width as u32,
            });
        }

        Ok(out)
    }
}

/// One request's anchors: the keyframe indices the layout places, the rows the
/// DiT consumes, and the pixels the prompt's vision spans borrow
/// (`h3.c:1231-1252`).
struct Anchors {
    keyframes: Vec<i32>,
    rows: Vec<f32>,
    pixels: Vec<condition::AnchorPixels>,
}

fn conditioning_identity(prompt: &str, anchors: &Anchors, references: &reference::Set) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ds4-h3-conditioning-v1");
    hash.update((prompt.len() as u64).to_le_bytes());
    hash.update(prompt.as_bytes());
    hash.update((anchors.keyframes.len() as u64).to_le_bytes());
    for frame in &anchors.keyframes {
        hash.update(frame.to_le_bytes());
    }
    hash_f32(&mut hash, &anchors.rows);
    hash.update((anchors.pixels.len() as u64).to_le_bytes());
    for pixels in &anchors.pixels {
        hash.update(pixels.height.to_le_bytes());
        hash.update(pixels.width.to_le_bytes());
        hash_f32(&mut hash, &pixels.values);
    }

    // Equal VAE rows can have different positions, prefixes or vision pixels.
    hash.update((references.layout.len() as u64).to_le_bytes());
    for reference in &references.layout {
        hash.update(match reference.kind {
            sequence::ReferenceKind::Image => b"i",
            sequence::ReferenceKind::Video => b"v",
            sequence::ReferenceKind::Audio => b"a",
        });
        for dimension in [
            reference.latent_t,
            reference.latent_h,
            reference.latent_w,
            reference.audio_t,
        ] {
            hash.update(dimension.to_le_bytes());
        }
    }
    hash_f32(&mut hash, &references.video_rows);
    hash_f32(&mut hash, &references.audio_rows);
    hash.update((references.pixels.len() as u64).to_le_bytes());
    for pixels in &references.pixels {
        hash.update(match pixels.presentation {
            super::multimodal::ReferenceKind::Image => b"i",
            super::multimodal::ReferenceKind::Video => b"v",
            super::multimodal::ReferenceKind::Audio => b"a",
        });
        hash.update([u8::from(pixels.has_audio)]);
        hash.update(pixels.height.to_le_bytes());
        hash.update(pixels.width.to_le_bytes());
        hash.update((pixels.frames as u64).to_le_bytes());
        hash.update((pixels.timestamps.len() as u64).to_le_bytes());
        for timestamp in &pixels.timestamps {
            hash.update(timestamp.to_le_bytes());
        }
        hash_f32(&mut hash, &pixels.pixels);
    }

    hash.finalize().into()
}

fn hash_f32(hash: &mut Sha256, values: &[f32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

/// The engine owns a thread; the host holds this handle. Commands are served
/// in arrival order, so a second render waits for the first instead of
/// contending for the device.
pub struct EngineHandle {
    calls: Sender<Call>,
}

enum Call {
    Render(RenderRequest, Sender<Result<RenderReport, EngineError>>),
    Stop,
}

impl EngineHandle {
    pub fn spawn(model_dir: impl Into<PathBuf>) -> Result<EngineHandle, EngineError> {
        let model_dir = model_dir.into();
        let (calls, receiver) = mpsc::channel();
        let (ready, opened) = mpsc::sync_channel(0);

        thread::Builder::new()
            .name("h3-engine".to_string())
            .spawn(move || match Engine::open(&model_dir) {
                Ok(mut engine) => {
                    let _ = ready.send(Ok(()));
                    serve(&mut engine, &receiver);
                }
                Err(error) => {
                    let _ = ready.send(Err(error));
                }
            })
            .map_err(|error| EngineError::new(format!("engine thread: {error}")))?;

        match opened.recv() {
            Ok(Ok(())) => Ok(EngineHandle { calls }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(EngineError::new("engine thread stopped while opening")),
        }
    }

    /// Blocks until the engine answers. The wait is the queue.
    pub fn render(&self, request: RenderRequest) -> Result<RenderReport, EngineError> {
        let (reply, answer) = mpsc::channel();
        self.calls
            .send(Call::Render(request, reply))
            .map_err(|_| EngineError::new("engine thread is gone"))?;
        answer
            .recv()
            .map_err(|_| EngineError::new("engine thread stopped mid-render"))?
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        let _ = self.calls.send(Call::Stop);
    }
}

fn serve(engine: &mut Engine, receiver: &Receiver<Call>) {
    while let Ok(call) = receiver.recv() {
        match call {
            Call::Render(request, reply) => {
                let _ = reply.send(engine.render(&request));
            }
            Call::Stop => break,
        }
    }
}

fn output_is_present(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.len() > 0)
}

fn small(value: impl TryInto<i32>) -> Result<i32, EngineError> {
    value
        .try_into()
        .map_err(|_| EngineError::new("a decoded dimension does not fit an i32"))
}

/// Kept for the operator's log line; the server prints it per request.
impl RenderReport {
    pub fn summary(&self) -> String {
        let (x, y) = self.scale;
        // A non-uniform scale is the named anamorphic delivery, so the log
        // carries it instead of leaving the stretch invisible.
        let scale = if (x - y).abs() < 1e-9 {
            format!(" scale={x:.4}")
        } else {
            format!(" scale={x:.4}x{y:.4} anamorphic")
        };

        format!(
            "frames={} evaluated={} text={:.3}s{} load={:.3}s{} denoise={:.3}s \
             audio={:.3}s video={:.3}s mux={:.3}s{}",
            self.frames,
            self.evaluated,
            self.text_seconds,
            if self.text_hit { " (hit)" } else { "" },
            self.load_seconds,
            if self.prepared_hit { " (hit)" } else { "" },
            self.denoise_seconds,
            self.audio_seconds,
            self.video_seconds,
            self.mux_seconds,
            scale,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(frames: i32, conditioning: [u8; 32]) -> PreparedKey {
        PreparedKey {
            conditioning,
            width: 512,
            height: 192,
            frames,
            steps: 20,
            reuse: 1,
            rope: 16,
        }
    }

    fn anchors(frame: i32) -> Anchors {
        Anchors {
            keyframes: vec![frame],
            rows: vec![0.5; 96],
            pixels: vec![condition::AnchorPixels {
                values: vec![0.5; 3 * 32 * 32],
                height: 32,
                width: 32,
            }],
        }
    }

    fn references() -> reference::Set {
        reference::Set {
            layout: vec![
                sequence::Reference {
                    kind: sequence::ReferenceKind::Image,
                    latent_t: 1,
                    latent_h: 2,
                    latent_w: 2,
                    audio_t: 0,
                },
                sequence::Reference {
                    kind: sequence::ReferenceKind::Audio,
                    latent_t: 0,
                    latent_h: 0,
                    latent_w: 0,
                    audio_t: 2,
                },
            ],
            video_rows: vec![0.25; 96],
            audio_rows: vec![0.5; 128],
            pixels: vec![
                condition::ReferencePixels {
                    presentation: super::super::multimodal::ReferenceKind::Image,
                    has_audio: false,
                    pixels: vec![0.25; 3 * 32 * 32],
                    height: 32,
                    width: 32,
                    frames: 1,
                    timestamps: Vec::new(),
                },
                condition::ReferencePixels {
                    presentation: super::super::multimodal::ReferenceKind::Audio,
                    has_audio: true,
                    pixels: Vec::new(),
                    height: 0,
                    width: 0,
                    frames: 0,
                    timestamps: Vec::new(),
                },
            ],
            ..reference::Set::default()
        }
    }

    #[test]
    fn anchor_role_invalidates_cache() {
        let first = anchors(0);
        let last = anchors(4);
        let references = reference::Set::default();
        assert_eq!(first.rows, last.rows);

        let spec = Spec {
            text_len: 3,
            latent_t: 2,
            latent_h: 2,
            latent_w: 2,
            audio_t: 8,
            frame_count: 5,
            keyframes: &first.keyframes,
            references: &[],
        };
        let first_sequence = sequence::build(&spec).unwrap();
        let last_sequence = sequence::build(&Spec {
            keyframes: &last.keyframes,
            ..spec
        })
        .unwrap();
        assert_ne!(first_sequence.positions, last_sequence.positions);

        let first_key = conditioning_identity("p", &first, &references);
        let last_key = conditioning_identity("p", &last, &references);
        assert_ne!(first_key, last_key);
        assert_ne!(key(5, first_key), key(5, last_key));
        assert_eq!(
            first_key,
            conditioning_identity("p", &anchors(0), &references)
        );
        assert_ne!(key(5, first_key), key(22, first_key));
    }

    #[test]
    fn ref_order_invalidates_cache() {
        let anchors = Anchors {
            keyframes: Vec::new(),
            rows: Vec::new(),
            pixels: Vec::new(),
        };
        let image_first = references();
        let mut audio_first = image_first.clone();
        audio_first.layout.swap(0, 1);
        audio_first.pixels.swap(0, 1);
        assert_eq!(image_first.video_rows, audio_first.video_rows);
        assert_eq!(image_first.audio_rows, audio_first.audio_rows);

        let spec = Spec {
            text_len: 3,
            latent_t: 2,
            latent_h: 2,
            latent_w: 2,
            audio_t: 8,
            frame_count: 5,
            keyframes: &[],
            references: &image_first.layout,
        };
        let a = sequence::build(&spec).unwrap();
        let b = sequence::build(&Spec {
            references: &audio_first.layout,
            ..spec
        })
        .unwrap();
        assert_ne!(a.positions, b.positions);

        let a = conditioning_identity("p", &anchors, &image_first);
        let b = conditioning_identity("p", &anchors, &audio_first);
        assert_ne!(a, b);
        assert_ne!(key(5, a), key(5, b));
        assert_eq!(
            a,
            conditioning_identity("p", &anchors, &image_first.clone())
        );
    }

    #[test]
    fn vision_pixels_are_in_the_key() {
        let mut anchors = anchors(0);
        let refs = reference::Set::default();
        let original = conditioning_identity("p", &anchors, &refs);
        anchors.pixels[0].values[0] = 0.0;
        assert_ne!(original, conditioning_identity("p", &anchors, &refs));

        let anchors = Anchors {
            keyframes: Vec::new(),
            rows: Vec::new(),
            pixels: Vec::new(),
        };
        let mut refs = references();
        let original = conditioning_identity("p", &anchors, &refs);
        refs.pixels[0].pixels[0] = 0.0;
        assert_ne!(original, conditioning_identity("p", &anchors, &refs));
    }

    #[test]
    fn ref_time_invalidates_cache() {
        let anchors = Anchors {
            keyframes: Vec::new(),
            rows: Vec::new(),
            pixels: Vec::new(),
        };
        let mut refs = references();
        refs.layout[0].kind = sequence::ReferenceKind::Video;
        refs.layout[0].latent_t = 2;
        refs.video_rows.resize(2 * 96, 0.25);
        refs.pixels[0].presentation = super::super::multimodal::ReferenceKind::Video;
        refs.pixels[0].frames = 2;
        refs.pixels[0].pixels.resize(2 * 3 * 32 * 32, 0.25);
        refs.pixels[0].timestamps = vec![0.0];
        let original = conditioning_identity("p", &anchors, &refs);

        refs.pixels[0].timestamps[0] = 0.5;
        assert_ne!(original, conditioning_identity("p", &anchors, &refs));
        refs.pixels[0].timestamps[0] = 0.0;
        assert_ne!(
            original,
            conditioning_identity("different prompt", &anchors, &refs)
        );
    }
}
