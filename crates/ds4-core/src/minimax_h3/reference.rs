//! Ref2VA reference media: from a reference file to condition rows.
//!
//! Source of truth is `/data/h3.c` at `eae5d54`:
//!
//! - A reference is `{kind, path, audio_path, include_embedded_audio}`
//!   (`h3.h:30-42`); at most 12 are accepted (`h3.c:588`, `main.c:107-110`).
//! - Visual references are decoded at their OWN canvas, never the render
//!   canvas (`h3.c:1052-1094`): `h3_ffprobe_visual_size`, then
//!   `h3_reference_image_canvas` (cap 2048 for `max`) or
//!   `h3_reference_video_canvas`, then the image reader (STRETCH) or the video
//!   reader at `temporal.frame_count`.
//! - The video VAE encodes those pixels, and the latent is patchified into
//!   96-wide rows (`h3.c:1296-1328`, `h3_dit.c:3077-3097`). The jitter is
//!   applied later, per span, by `h3_augment_conditions` (`h3.c:323-368`,
//!   `:1446`): a fresh generator per span, seeded with the request seed.
//! - Reference videos carry Qwen vision timestamps: `samples = (frames + 11) /
//!   12`, `blocks = (samples + 1) / 2`, and block b's timestamp is
//!   `(2b + min(2b + 1, samples - 1)) / 4` (`h3.c:1096-1114`).
//! - Audio: an audio reference reads its own file; `video+audio` reads
//!   `audio_path`; a video with `include_embedded_audio` reads the video's own
//!   track. Both video forms truncate to `frames * 32000 / 24` samples and
//!   need at least 64000 (`h3.c:1126-1170`). The encoder's channel-major
//!   `[32,2,t]` output is repacked to row-major `[2,t,32]` (`h3.c:1182-1197`)
//!   and jittered with `seed + 1` (`h3.c:344`).
//! - The 15-second total audio budget (`h3.c:1171-1176`) spans the whole
//!   reference set, so it is the caller's rule, not this module's.

use std::fmt;
use std::path::{Path, PathBuf};

use super::condition::{augment_span, VIDEO_ROW_WIDTH};
use super::host::{reference_image_canvas, reference_video_canvas, FPS};
use super::media::{self, Fit};
use super::{vae_audio, vae_encoder};

/// The reference's cap on ordered references (`h3.c:588`).
pub const MAX_REFERENCES: usize = 12;
/// 32 kHz stereo, 15 s (`h3.c:1129`, `:1172`).
pub const SAMPLE_RATE: usize = 32000;
pub const MAX_AUDIO_SAMPLES: usize = SAMPLE_RATE * 15;
/// The shortest soundtrack a truncated video reference can carry
/// (`h3.c:1150`).
pub const MIN_VIDEO_AUDIO_SAMPLES: usize = 64_000;
/// The `max` policy's short-edge cap (`h3.c:1051`).
const MAX_SHORT_EDGE: i32 = 2048;
/// The audio VAE's released geometry (`h3.c:1177-1181`).
const AUDIO_CHANNELS: usize = 32;
const AUDIO_STEREO: usize = 2;

/// `h3_reference_kind` (`h3.h:30-34`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Image,
    Video,
    Audio,
    VideoAudio,
}

/// `h3_reference_image_size` (`h3.h:44-47`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSize {
    /// Match the render canvas pixel area.
    Match,
    /// Cap the short edge at 2048 px.
    Max,
}

impl ImageSize {
    fn max_short_edge(self) -> i32 {
        match self {
            ImageSize::Match => 0,
            ImageSize::Max => MAX_SHORT_EDGE,
        }
    }
}

/// One ordered reference (`h3_reference`, `h3.h:36-42`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub kind: Kind,
    pub path: PathBuf,
    /// The separate soundtrack of a `video+audio` reference.
    pub audio_path: Option<PathBuf>,
    /// Read a plain video's own track as its soundtrack.
    pub include_embedded_audio: bool,
}

/// The presentation class a reference gets in the prompt
/// (`h3_presentation_kind`, `h3.c:1034-1036`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presentation {
    Image,
    Video,
    Audio,
}

/// What one reference contributes to the conditioning.
#[derive(Debug, Clone, Default)]
pub struct Prepared {
    pub presentation: Option<Presentation>,
    /// The media canvas the reference was decoded at; zero for audio.
    pub width: u32,
    pub height: u32,
    /// Decoded video frames; one for an image, zero for audio.
    pub frames: usize,
    /// `h3_layout_ref`'s latent geometry (`h3.c:1116-1124`).
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    /// Patchified, jittered condition rows (`h3.c:1320-1328`, `:344-349`).
    pub video_rows: Vec<f32>,
    /// The repacked audio rows and their latent length (`h3.c:1196-1197`).
    pub audio_t: usize,
    pub audio_rows: Vec<f32>,
    /// Source samples the soundtrack produced, for the 15 s set budget.
    pub audio_samples: usize,
    /// Qwen vision blocks and their timestamps (`h3.c:1096-1114`): one block
    /// for an image, `ceil(samples / 2)` for a video, none for audio.
    pub vision_blocks: usize,
    pub timestamps: Vec<f64>,
    /// The decoded pixels the vision tower reads, in channel-major RGB.
    pub pixels: Vec<f32>,
}

/// The two VAE weight directories a reference encode needs.
pub struct Paths<'a> {
    pub video_vae: &'a Path,
    pub audio_vae: &'a Path,
}

/// One request's ordered references: the layout the sequence places, the rows
/// the DiT consumes, the pixels the prompt's vision spans read, and the row
/// hash the embedding memo folds in (`h3.c:143-166`). Shared by the resident
/// engine and the CLI so `h3.c:1027-1200` has one port, not two.
#[derive(Debug, Clone, Default)]
pub struct Set {
    pub layout: Vec<super::sequence::Reference>,
    pub video_rows: Vec<f32>,
    pub audio_rows: Vec<f32>,
    pub pixels: Vec<super::condition::ReferencePixels>,
    /// Source samples over the whole set, against `MAX_AUDIO_SAMPLES`.
    pub samples: usize,
    pub hash: u64,
}

/// The gate hash convention: FNV-1a over little-endian bytes, the same one the
/// probe family uses (`text_forward::fnv1a`).
pub(crate) fn fnv_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    super::text_forward::fnv1a(&bytes)
}

/// Fold two hashes into one. The memo needs a single value for the media it
/// folds (`h3.c:143-166`); moving a byte moves the mix's input.
pub(crate) fn combine(first: u64, second: u64) -> u64 {
    first.rotate_left(17) ^ second.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Prepare every ordered reference (`h3.c:1027-1200`). The twelve cap
/// (`h3.c:588`) and the 15-second soundtrack budget (`h3.c:1171-1176`) are the
/// reference's own set rules, so they are enforced here for every caller.
pub fn prepare_set(
    paths: &Paths<'_>,
    references: &[Reference],
    image_size: ImageSize,
    canvas: (u32, u32),
    frame_count: usize,
    seed: u64,
) -> Result<Set, ReferenceError> {
    let mut out = Set {
        layout: Vec::with_capacity(references.len()),
        video_rows: Vec::new(),
        audio_rows: Vec::new(),
        pixels: Vec::with_capacity(references.len()),
        samples: 0,
        hash: 0,
    };

    // `h3.c:588`: the reference refuses a thirteenth before any decode.
    if references.len() > MAX_REFERENCES {
        return Err(ReferenceError(format!(
            "Ref2VA supports at most {MAX_REFERENCES} references"
        )));
    }

    for (index, media) in references.iter().enumerate() {
        let prepared = prepare(paths, media, image_size, canvas, frame_count, seed, index)?;
        out.samples = out.samples.saturating_add(prepared.audio_samples);
        if out.samples > MAX_AUDIO_SAMPLES {
            return Err(ReferenceError(
                "ordered reference audio exceeds 15 seconds in total".to_string(),
            ));
        }

        layout_one(&mut out, media, prepared);
    }

    out.hash = combine(fnv_f32(&out.video_rows), fnv_f32(&out.audio_rows));

    Ok(out)
}

/// Fold one prepared reference into the set: its layout span, its rows and its
/// vision span, in request order (`h3_host.c:367-433`).
fn layout_one(out: &mut Set, media: &Reference, prepared: Prepared) {
    let presentation = prepared
        .presentation
        .expect("every prepared reference names its class");
    // A video's own soundtrack counts only when the caller kept it
    // (`h3.c:1126-1151`).
    let has_audio = match (media.kind, media.include_embedded_audio) {
        (Kind::Audio, _) | (Kind::VideoAudio, _) => true,
        (Kind::Video, embedded) => embedded,
        (Kind::Image, _) => false,
    };

    out.layout.push(match presentation {
        Presentation::Image => super::sequence::Reference {
            kind: super::sequence::ReferenceKind::Image,
            latent_t: 1,
            latent_h: prepared.latent_h as i32,
            latent_w: prepared.latent_w as i32,
            audio_t: 0,
        },
        Presentation::Audio => super::sequence::Reference {
            kind: super::sequence::ReferenceKind::Audio,
            latent_t: 0,
            latent_h: 0,
            latent_w: 0,
            audio_t: prepared.audio_t as i32,
        },
        Presentation::Video => super::sequence::Reference {
            kind: super::sequence::ReferenceKind::Video,
            latent_t: prepared.latent_t as i32,
            latent_h: prepared.latent_h as i32,
            latent_w: prepared.latent_w as i32,
            audio_t: prepared.audio_t as i32,
        },
    });

    out.video_rows.extend_from_slice(&prepared.video_rows);
    out.audio_rows.extend_from_slice(&prepared.audio_rows);
    out.pixels.push(super::condition::ReferencePixels {
        presentation: match presentation {
            Presentation::Image => super::multimodal::ReferenceKind::Image,
            Presentation::Video => super::multimodal::ReferenceKind::Video,
            Presentation::Audio => super::multimodal::ReferenceKind::Audio,
        },
        has_audio,
        pixels: prepared.pixels,
        height: prepared.height,
        width: prepared.width,
        frames: prepared.frames,
        timestamps: prepared.timestamps,
    });
}

#[derive(Debug)]
pub struct ReferenceError(String);

impl ReferenceError {
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ReferenceError {}

fn fail(detail: impl Into<String>) -> Result<Prepared, ReferenceError> {
    Err(ReferenceError(detail.into()))
}

/// Prepare one ordered reference. `index` is zero-based and only names the
/// reference in the reference's own messages.
pub fn prepare(
    paths: &Paths,
    reference: &Reference,
    size: ImageSize,
    render: (u32, u32),
    frame_count: usize,
    seed: u64,
    index: usize,
) -> Result<Prepared, ReferenceError> {
    let mut prepared = match reference.kind {
        Kind::Audio => Prepared {
            presentation: Some(Presentation::Audio),
            ..Prepared::default()
        },
        _ => prepare_visual(paths, reference, size, render, frame_count, seed, index)?,
    };

    if let Some((path, truncate)) = audio_source(reference) {
        let samples = match truncate {
            true => {
                // `llround(frames * 32000 / 24)` (`h3.c:1144-1147`).
                let limit =
                    (prepared.frames as f64 * SAMPLE_RATE as f64 / f64::from(FPS)).round() as usize;
                if limit < MIN_VIDEO_AUDIO_SAMPLES {
                    return fail(format!(
                        "video soundtrack {} requires at least 2 seconds; \
                         request at least 56 output frames",
                        index + 1
                    ));
                }
                limit
            }
            false => MAX_AUDIO_SAMPLES,
        };
        let pcm = media::read_audio_f32(path, samples as u32, truncate)
            .map_err(|error| ReferenceError(error.detail().to_string()))?;
        let count = pcm.len() / AUDIO_STEREO;
        let latent = vae_audio::encode(paths.audio_vae, &pcm, count)
            .map_err(|error| ReferenceError(format!("reference audio: {error}")))?;
        if latent.channels != AUDIO_CHANNELS || latent.stereo != AUDIO_STEREO || latent.length < 1 {
            return fail("reference audio encoder produced invalid geometry");
        }

        let mut rows = repack_audio(&latent);
        augment_span(&mut rows, seed.wrapping_add(1));
        prepared.audio_t = latent.length;
        prepared.audio_rows = rows;
        prepared.audio_samples = count;
    }

    Ok(prepared)
}

/// Decode and encode the visual half (`h3.c:1052-1128`).
fn prepare_visual(
    paths: &Paths,
    reference: &Reference,
    size: ImageSize,
    render: (u32, u32),
    frame_count: usize,
    seed: u64,
    index: usize,
) -> Result<Prepared, ReferenceError> {
    let (source_w, source_h) = media::probe_visual_size(&reference.path)
        .map_err(|error| ReferenceError(error.detail().to_string()))?;
    let (source_w, source_h) = (source_w as i32, source_h as i32);

    let (width, height, presentation) = match reference.kind {
        Kind::Image => {
            let (width, height) = reference_image_canvas(
                source_w,
                source_h,
                render.0 as i32,
                render.1 as i32,
                size.max_short_edge(),
            )
            .ok_or_else(|| {
                ReferenceError(format!(
                    "cannot resolve reference image {} canvas",
                    index + 1
                ))
            })?;
            (width, height, Presentation::Image)
        }
        Kind::Video | Kind::VideoAudio => {
            let (width, height) = reference_video_canvas(source_w, source_h).ok_or_else(|| {
                ReferenceError(format!(
                    "cannot resolve reference video {} canvas",
                    index + 1
                ))
            })?;
            (width, height, Presentation::Video)
        }
        Kind::Audio => unreachable!("audio references have no visual half"),
    };

    let (pixels, frames) = match reference.kind {
        Kind::Image => {
            let values =
                media::read_image_f32(&reference.path, width as u32, height as u32, Fit::Stretch)
                    .map_err(|error| ReferenceError(error.detail().to_string()))?;
            (values, 1)
        }
        _ => media::read_video_f32(
            &reference.path,
            width as u32,
            height as u32,
            frame_count as u32,
        )
        .map_err(|error| ReferenceError(error.detail().to_string()))?,
    };

    let latent = vae_encoder::encode(
        paths.video_vae,
        &pixels,
        frames,
        height as usize,
        width as usize,
        |_, _| {},
    )
    .map_err(|error| ReferenceError(format!("reference {}: {error}", index + 1)))?;

    let (latent_w, latent_h) = ((width as usize) / 16, (height as usize) / 16);
    if latent.time != vae_encoder::latent_time(frames)
        || latent.height != latent_h
        || latent.width != latent_w
    {
        return fail("visual condition VAE produced unexpected latent geometry");
    }

    let rows = latent.time * latent_h * latent_w / 4;
    let mut video_rows = vec![0.0f32; rows * VIDEO_ROW_WIDTH];
    super::dit::patchify_video(
        &latent.values,
        vae_encoder::LATENT_CHANNELS,
        latent.time,
        latent_h,
        latent_w,
        &mut video_rows,
    )
    .map_err(|error| ReferenceError(format!("patchify: {error:?}")))?;
    augment_span(&mut video_rows, seed);

    let (vision_blocks, timestamps) = match reference.kind {
        Kind::Image => (1, Vec::new()),
        _ => {
            let timestamps = vision_timestamps(frames);
            (timestamps.len(), timestamps)
        }
    };

    Ok(Prepared {
        presentation: Some(presentation),
        width: width as u32,
        height: height as u32,
        frames,
        latent_t: latent.time,
        latent_h,
        latent_w,
        video_rows,
        vision_blocks,
        timestamps,
        pixels,
        ..Prepared::default()
    })
}

/// `h3.c:832-855`: Qwen reads a reference video as time-major two-frame
/// blocks, while the media reader and the VAE keep channel-major
/// `[3, T, H, W]`. `first` and `second` are frame indices; a repeated index
/// pairs a frame with itself, which is what the last odd sample does.
pub fn vision_pair(
    pixels: &[f32],
    frames: usize,
    height: usize,
    width: usize,
    first: usize,
    second: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let area = height * width;
    if frames < 1 || height < 1 || width < 1 || first >= frames || second >= frames {
        return Err(ReferenceError(
            "reference vision pair is outside the decoded frames".to_string(),
        ));
    }

    let mut pair = vec![0.0f32; 2 * 3 * area];
    for (time, index) in [first, second].into_iter().enumerate() {
        for channel in 0..3 {
            let source = (channel * frames + index) * area;
            let destination = (time * 3 + channel) * area;
            pair[destination..destination + area]
                .copy_from_slice(&pixels[source..source + area]);
        }
    }

    Ok(pair)
}

/// `h3.c:1096-1114`: one timestamp per two-frame pair, `(first + second) / 4`.
pub fn vision_timestamps(frames: usize) -> Vec<f64> {
    let samples = frames.div_ceil(12);
    let blocks = samples.div_ceil(2);
    (0..blocks)
        .map(|block| {
            let first = 2 * block;
            let second = if first + 1 < samples {
                first + 1
            } else {
                first
            };
            (first as f64 + second as f64) / 4.0
        })
        .collect()
}

/// `h3.c:1126-1147`: which file carries a reference's soundtrack, and whether
/// it is cut to the decoded video length.
fn audio_source(reference: &Reference) -> Option<(&Path, bool)> {
    match reference.kind {
        Kind::Audio => Some((&reference.path, false)),
        Kind::VideoAudio => reference.audio_path.as_deref().map(|path| (path, true)),
        Kind::Video if reference.include_embedded_audio => Some((&reference.path, true)),
        Kind::Video | Kind::Image => None,
    }
}

/// `h3.c:1182-1197`: channel-major `[32,2,t]` to row-major `[2,t,32]`.
fn repack_audio(latent: &vae_audio::Latent) -> Vec<f32> {
    let mut rows = vec![0.0f32; latent.stereo * latent.length * AUDIO_CHANNELS];
    for stereo in 0..latent.stereo {
        for time in 0..latent.length {
            for channel in 0..AUDIO_CHANNELS {
                let source = (channel * latent.stereo + stereo) * latent.length + time;
                let destination = (stereo * latent.length + time) * AUDIO_CHANNELS + channel;
                rows[destination] = latent.values[source];
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `h3.c:1096-1114`: pairs of 12-frame samples, the odd tail repeated, and
    /// the timestamp is the pair's mean divided by two.
    #[test]
    fn video_timestamps_follow_the_reference_pairs() {
        assert_eq!(vision_timestamps(12), vec![0.0]);
        // 24 frames is two samples, one block, and its pair is (0, 1).
        assert_eq!(vision_timestamps(22), vec![0.25]);
        assert_eq!(vision_timestamps(24), vec![0.25]);
        // 36 frames is three samples: (0, 1) then the repeated 2.
        assert_eq!(vision_timestamps(36), vec![0.25, 1.0]);
        assert_eq!(vision_timestamps(362).len(), 16);
    }

    /// `h3.c:1182-1197`: `channel * 2 + stereo` becomes `(stereo, time,
    /// channel)`, so a constant channel keeps its value in its own slot.
    #[test]
    fn audio_rows_are_repacked_channel_major_to_row_major() {
        let latent = vae_audio::Latent {
            channels: AUDIO_CHANNELS,
            stereo: AUDIO_STEREO,
            length: 2,
            values: (0..AUDIO_CHANNELS * AUDIO_STEREO * 2)
                .map(|index| index as f32)
                .collect(),
        };
        let rows = repack_audio(&latent);
        assert_eq!(rows.len(), AUDIO_STEREO * 2 * AUDIO_CHANNELS);
        for channel in 0..AUDIO_CHANNELS {
            for stereo in 0..AUDIO_STEREO {
                for time in 0..2 {
                    let source = (channel * AUDIO_STEREO + stereo) * 2 + time;
                    let destination = (stereo * 2 + time) * AUDIO_CHANNELS + channel;
                    assert_eq!(rows[destination], source as f32);
                }
            }
        }
    }

    /// `h3.c:1129-1147`: only a truncated video form has a soundtrack, and a
    /// `video+audio` reference without its file contributes none.
    #[test]
    fn audio_sources_follow_the_kind() {
        let reference = |kind, audio: Option<&str>, embedded| Reference {
            kind,
            path: PathBuf::from("clip.mp4"),
            audio_path: audio.map(PathBuf::from),
            include_embedded_audio: embedded,
        };

        assert_eq!(
            audio_source(&reference(Kind::Audio, None, false)),
            Some((Path::new("clip.mp4"), false))
        );
        assert_eq!(
            audio_source(&reference(Kind::VideoAudio, None, false)),
            None
        );
        assert_eq!(
            audio_source(&reference(Kind::VideoAudio, Some("track.wav"), false)),
            Some((Path::new("track.wav"), true))
        );
        assert_eq!(
            audio_source(&reference(Kind::Video, None, true)),
            Some((Path::new("clip.mp4"), true))
        );
        assert_eq!(audio_source(&reference(Kind::Video, None, false)), None);
        assert_eq!(audio_source(&reference(Kind::Image, None, false)), None);
    }
}
