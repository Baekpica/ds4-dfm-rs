//! The resident render engine: the reference's interactive session.
//!
//! With its cache enabled the reference keeps three things across calls
//! (`h3_cache_info`, `/data/h3.c/h3.h:21-27`, cleared in `h3.c:53-64`): the
//! prompt embedding (`h3.c:143-166`, `h3.c:223-309`), the prepared DiT keyed on
//! the conditioning and the request shape (`h3.c:168-190`) and the video
//! decoder (`h3.c:697-723`). A one-shot caller keeps the original
//! phase-by-phase lifetime instead (`h3.h:180-184`), which is what the CLI
//! does. This module mirrors the session mechanism for the port's pipeline, so
//! a second OpenAI request does not re-read the 62 GiB text encoder or the
//! 61.7 GiB DiT.
//!
//! The device handles are `!Send` (`ds4_sys::h3_dit::{Gpu, Tensor}` hold
//! `NonNull`), so the engine owns a thread and the host reaches it over a
//! channel. That also makes admission serial by construction: one render at a
//! time, which is what one GPU allows.
//!
//! Scope: the text-only path. The conditioning key collapses to the prompt
//! while no anchor or reference media is served (`h3.c:145-149` returns the
//! prompt-only key in exactly that case); P12 extends both keys.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use super::dit::{Dit, LoadOptions};
use super::host::{reuse_schedule, temporal, Rng, Schedule};
use super::media::{self, Fit};
use super::render::{check_frames, mux_mp4, Canvas, RgbFrames};
use super::sequence::{self, Spec};
use super::text_forward::{self, Embedding};
use super::tokenizer::{EmptyPrompt, Tokenizer};
use super::{condition, vae_audio, vae_encoder, vae_video};

/// The component layout `h3.c:425-436` resolves under the model directory.
const FL2VA: &str = "FL2VA";

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
    prompt: String,
    width: i32,
    height: i32,
    frames: i32,
    steps: usize,
    reuse: usize,
    rope: u32,
    /// The anchor rows themselves, as `h3_conditioning_key` folds the media
    /// pixels into its key (`h3.c:143-166`): identical pads over other pixels
    /// must prefill again.
    anchors: u64,
}

struct Prepared {
    key: PreparedKey,
    dit: Dit,
}

pub struct Engine {
    model_dir: PathBuf,
    tokenizer: Tokenizer,
    /// `h3.c:223-309`: one conditioning entry is enough for a server; the
    /// reference keeps a small table and counts hits and misses. The anchor
    /// rows are part of the key because they are vision spans in the prompt.
    embedding: Option<(String, u64, Embedding)>,
    prepared: Option<Prepared>,
}

impl Engine {
    /// Open the checkpoint metadata. No device state yet: the text encoder is
    /// loaded per encode, as the reference does (it frees the embedding after
    /// the DiT is prepared, `h3.c:1539`).
    pub fn open(model_dir: &Path) -> Result<Engine, EngineError> {
        let path = model_dir
            .join(FL2VA)
            .join("tokenizer")
            .join("tokenizer.json");
        let tokenizer = Tokenizer::load(&path)
            .map_err(|error| EngineError::new(format!("tokenizer: {}", error.token())))?;

        Ok(Engine {
            model_dir: model_dir.to_path_buf(),
            tokenizer,
            embedding: None,
            prepared: None,
        })
    }

    fn component(&self, name: &str) -> PathBuf {
        self.model_dir.join(FL2VA).join(name)
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

        // Anchors (`h3.c:1231-1345`): each frame is read at the canvas, encoded
        // by the video VAE, patchified and jittered before the layout is built,
        // because the cond row count is part of the packed signature. The same
        // pixels are the prompt's vision spans, so they are read once.
        let anchors = self.read_anchors(request, frame_count, latent_h, latent_w)?;
        let anchor_hash = fnv_f32(&anchors.rows);
        let (keyframes, video_rows) = (anchors.keyframes, anchors.rows);

        // Conditioning: the reference memoizes the prompt embedding
        // (`h3.c:223-309`) because the encoder is the largest read of the run.
        // An anchor changes the embedding (it is a vision span), so the anchor
        // rows are part of the memo key, as `h3.c:143-166` folds the media in.
        let text_started = Instant::now();
        let cached = self.embedding.as_ref().and_then(|(prompt, anchors, text)| {
            (prompt == &request.prompt && *anchors == anchor_hash).then(|| text.clone())
        });
        let text_hit = cached.is_some();
        let text = match cached {
            Some(text) => text,
            None => {
                let text = match anchors.pixels.is_empty() {
                    true => {
                        let ids = self
                            .tokenizer
                            .encode(&request.prompt, EmptyPrompt::Pad)
                            .map_err(|error| {
                                EngineError::new(format!("tokenizer: {}", error.token()))
                            })?;
                        text_forward::encode(&self.component("text_encoder"), &ids).map_err(
                            |error| EngineError::new(format!("text encoder: {}", error.token())),
                        )
                    }
                    false => condition::anchor_embedding(
                        &self.component("text_encoder"),
                        &self.tokenizer,
                        &request.prompt,
                        &anchors.pixels,
                    )
                    .map_err(EngineError::new),
                }?;
                self.embedding = Some((request.prompt.clone(), anchor_hash, text.clone()));
                text
            }
        };
        let text_seconds = text_started.elapsed().as_secs_f64();

        let key = PreparedKey {
            prompt: request.prompt.clone(),
            width: request.width,
            height: request.height,
            frames: frame_count,
            steps: request.steps,
            reuse: request.reuse,
            rope: canvas.rope_scale(false).to_bits(),
            anchors: anchor_hash,
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
            };
            let sequence = sequence::build(&spec)
                .map_err(|error| EngineError::new(format!("layout: {error:?}")))?;
            let dit = Dit::load_with(
                &self.component("transformer"),
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
        // (`h3_dit_reset_run`, `h3_dit.c:2408-2440`). With no anchors both
        // counts are zero and this only validates.
        dit.set_conditions(&video_rows, &[])
            .map_err(|error| EngineError::new(format!("conditions: {error}")))?;

        let mut video = vec![0.0f32; dit.video_elements()];
        let mut audio = vec![0.0f32; dit.audio_elements()];
        Rng::seed(request.seed).fill_normal(&mut video);
        Rng::seed(request.seed).fill_normal(&mut audio);

        let denoise_started = Instant::now();
        dit.denoise(&mut video, &mut audio, request.reuse)
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

/// The gate hash convention: FNV-1a over little-endian bytes, the same one the
/// probe family uses (`text_forward::fnv1a`).
fn fnv_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    text_forward::fnv1a(&bytes)
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

    fn key(frames: i32, anchors: u64) -> PreparedKey {
        PreparedKey {
            prompt: "p".to_string(),
            width: 512,
            height: 192,
            frames,
            steps: 20,
            reuse: 1,
            rope: 16,
            anchors,
        }
    }

    /// `h3_conditioning_key` folds the anchor media into the key
    /// (`h3.c:143-166`). The port folds the rows, not the path, so the same
    /// pads saved elsewhere hit; one changed value misses, and so does a
    /// different frame count.
    #[test]
    fn the_prepared_key_folds_the_anchor_rows() {
        let rows = vec![0.5f32; 96 * 96];
        let mut other = rows.clone();
        other[0] = 0.25;

        assert_eq!(key(5, fnv_f32(&rows)), key(5, fnv_f32(&rows)));
        assert_ne!(key(5, fnv_f32(&rows)), key(5, fnv_f32(&other)));
        assert_ne!(key(5, fnv_f32(&rows)), key(6, fnv_f32(&rows)));
    }
}
