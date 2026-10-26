//! Host-side media read for the conditioning surface.
//!
//! Source of truth `/data/h3.c` at `eae5d54`, `h3_ffmpeg.c` and `h3_ffmpeg.h`.
//! The reference shells out to FFmpeg and FFprobe and converts the raw output
//! itself; nothing here touches the device.
//!
//! - `h3_ffmpeg.h:20-40` the three readers and their shapes: one image as
//!   channel-major F32 `[3,H,W]`, a clip as `[3,T,H,W]` at 24 fps, a
//!   soundtrack as channel-major stereo F32 at 32 kHz.
//! - `h3_ffmpeg.c:21-29` `H3_FFMPEG` / `H3_FFPROBE` select the programs.
//! - `h3_ffmpeg.c:77-140` `h3_ffprobe_visual_size`: `csv=p=0:s=x` of
//!   `stream=width,height`, `WxH` only.
//! - `h3_ffmpeg.c:143-283` `h3_ffmpeg_read_image_f32`: `scale` with Lanczos,
//!   `force_original_aspect_ratio=increase` plus `crop` for `COVER`, exactly
//!   `W*H*3` bytes, no trailing byte, exit zero, then `value / 255` as F32.
//! - `h3_ffmpeg.c:285-393` `h3_ffmpeg_read_video_f32`: `fps=24` then Lanczos
//!   then `setsar=1`, at most `max_frames`, at least five, trimmed down to the
//!   released `5 + 17k` cadence.
//! - `h3_ffmpeg.c:395-521` `h3_ffmpeg_read_audio_f32`: stereo 32 kHz F32,
//!   two seconds minimum, fifteen maximum, and a clip's own soundtrack may be
//!   truncated at the limit while a standalone reference reports it.

use std::fmt;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

/// `AUDIO_RATE` (`h3_ffmpeg.c:396`).
pub const SAMPLE_RATE: u32 = 32_000;

/// `AUDIO_CHANNELS` (`h3_ffmpeg.c:396`).
pub const CHANNELS: usize = 2;

/// `MIN_SAMPLES` (`h3_ffmpeg.c:396`), two seconds.
pub const MIN_SAMPLES: usize = 64_000;

/// `max_samples > AUDIO_RATE * 15` is refused (`h3_ffmpeg.c:401`).
pub const MAX_SECONDS: u32 = 15;

/// The reference decodes visual media at 24 fps (`h3_ffmpeg.c:299`).
pub const FRAME_RATE: u32 = 24;

/// The released request cadence is `5 + 17k` (`h3_ffmpeg.c:372`, `h3.c:867`).
pub const MIN_FRAMES: usize = 5;

pub const FRAME_STEP: usize = 17;

/// `h3_image_fit` (`h3_ffmpeg.h:9-12`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Scale to the requested frame, changing the aspect.
    Stretch,
    /// Scale until both edges cover the frame, then crop.
    Cover,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaError(String);

impl MediaError {
    fn new(detail: String) -> MediaError {
        MediaError(detail)
    }

    pub fn detail(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MediaError {}

/// `ffmpeg_program` / `ffprobe_program` (`h3_ffmpeg.c:21-29`).
fn program(variable: &str, default: &str) -> String {
    match std::env::var(variable) {
        Ok(value) if !value.is_empty() => value,
        _ => default.to_string(),
    }
}

/// Run a program, read exactly `expected` bytes from stdout, and require EOF
/// and a zero exit status. That is the reference's read loop and its trailing
/// byte check (`h3_ffmpeg.c:232-262`) in one place.
fn capture(program: &str, arguments: &[String], expected: usize) -> Result<Vec<u8>, MediaError> {
    let (output, overflow) = capture_up_to(program, arguments, expected)?;
    if overflow {
        return Err(MediaError::new(format!(
            "{program} wrote more than the {expected} bytes the shape allows"
        )));
    }
    if output.len() != expected {
        return Err(MediaError::new(format!(
            "{program} produced {} bytes, expected {expected}",
            output.len()
        )));
    }

    Ok(output)
}

/// `h3_ffprobe_visual_size` (`h3_ffmpeg.c:77-140`): the first visual stream's
/// coded size, without decoding it.
pub fn probe_visual_size(path: &Path) -> Result<(u32, u32), MediaError> {
    let arguments = [
        "-v".to_string(),
        "error".to_string(),
        "-select_streams".to_string(),
        "v:0".to_string(),
        "-show_entries".to_string(),
        "stream=width,height".to_string(),
        "-of".to_string(),
        "csv=p=0:s=x".to_string(),
        path.display().to_string(),
    ];
    let output = Command::new(program("H3_FFPROBE", "ffprobe"))
        .args(&arguments)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| MediaError::new(format!("cannot start FFprobe: {error}")))?;

    if !output.status.success() {
        return Err(MediaError::new(format!(
            "FFprobe could not inspect visual stream {}",
            path.display()
        )));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let text = text.trim_end();
    let (width, height) = text.split_once('x').ok_or_else(|| {
        MediaError::new(format!(
            "FFprobe returned an invalid visual size {:?}",
            text.trim()
        ))
    })?;

    let width = width.trim().parse::<u32>().map_err(|_| {
        MediaError::new(format!(
            "FFprobe returned an invalid visual size {:?}",
            text.trim()
        ))
    })?;
    let height = height.trim().parse::<u32>().map_err(|_| {
        MediaError::new(format!(
            "FFprobe returned an invalid visual size {:?}",
            text.trim()
        ))
    })?;

    if width < 1 || height < 1 {
        return Err(MediaError::new(format!(
            "visual stream has invalid dimensions {width}x{height}"
        )));
    }

    Ok((width, height))
}

fn pixels(values: u32) -> Result<usize, MediaError> {
    usize::try_from(values).map_err(|_| MediaError::new("media size overflows".to_string()))
}

/// `h3_ffmpeg_read_image_f32` (`h3_ffmpeg.c:143-283`): one frame scaled to the
/// requested shape and returned channel-major.
pub fn read_image_f32(
    path: &Path,
    width: u32,
    height: u32,
    fit: Fit,
) -> Result<Vec<f32>, MediaError> {
    let width = pixels(width)?;
    let height = pixels(height)?;
    if width < 1 || height < 1 {
        return Err(MediaError::new("invalid image shape".to_string()));
    }

    let area = width
        .checked_mul(height)
        .ok_or_else(|| MediaError::new("decoded image size overflows".to_string()))?;
    let bytes = area
        .checked_mul(3)
        .ok_or_else(|| MediaError::new("decoded image size overflows".to_string()))?;

    // The two filters are the reference's (`h3_ffmpeg.c:190-199`).
    let filter = match fit {
        Fit::Stretch => format!("scale={width}:{height}:flags=lanczos"),
        Fit::Cover => format!(
            "scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,\
             crop={width}:{height}"
        ),
    };

    let arguments = [
        "-v".to_string(),
        "error".to_string(),
        "-i".to_string(),
        path.display().to_string(),
        "-frames:v".to_string(),
        "1".to_string(),
        "-vf".to_string(),
        filter,
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "pipe:1".to_string(),
    ];
    let rgb = capture(&program("H3_FFMPEG", "ffmpeg"), &arguments, bytes)?;

    Ok(channel_major(&rgb, area, 1))
}

/// `h3_ffmpeg_read_video_f32` (`h3_ffmpeg.c:285-393`): a 24 fps clip resampled
/// to the requested shape, trimmed to the released cadence.
pub fn read_video_f32(
    path: &Path,
    width: u32,
    height: u32,
    max_frames: u32,
) -> Result<(Vec<f32>, usize), MediaError> {
    let width = pixels(width)?;
    let height = pixels(height)?;
    let max_frames = pixels(max_frames)?;
    if width < 1 || height < 1 || max_frames < MIN_FRAMES {
        return Err(MediaError::new("invalid video arguments".to_string()));
    }

    let area = width
        .checked_mul(height)
        .ok_or_else(|| MediaError::new("decoded video size overflows".to_string()))?;
    let frame_bytes = area
        .checked_mul(3)
        .ok_or_else(|| MediaError::new("decoded video size overflows".to_string()))?;
    let capacity = max_frames
        .checked_mul(frame_bytes)
        .ok_or_else(|| MediaError::new("decoded video size overflows".to_string()))?;

    let filter = format!("fps={FRAME_RATE},scale={width}:{height}:flags=lanczos,setsar=1");
    let arguments = [
        "-v".to_string(),
        "error".to_string(),
        "-i".to_string(),
        path.display().to_string(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-an".to_string(),
        "-vf".to_string(),
        filter,
        "-frames:v".to_string(),
        max_frames.to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "pipe:1".to_string(),
    ];

    // A short clip is legal here: the reader takes what it gets, then trims.
    // `capture` wants an exact length, so the frame count is discovered from
    // the byte count instead, through the same trailing-byte discipline.
    let (rgb, overflow) = capture_up_to(&program("H3_FFMPEG", "ffmpeg"), &arguments, capacity)?;
    if overflow {
        return Err(MediaError::new(
            "the clip exceeds the bounded frame count".to_string(),
        ));
    }

    if rgb.len() % frame_bytes != 0 {
        return Err(MediaError::new(format!(
            "FFmpeg decoded {} bytes, not whole {}x{} frames",
            rgb.len(),
            width,
            height
        )));
    }

    let mut frames = rgb.len() / frame_bytes;
    if frames < MIN_FRAMES {
        return Err(MediaError::new(
            "reference videos require at least 5 decoded frames".to_string(),
        ));
    }

    while frames % FRAME_STEP != MIN_FRAMES {
        frames -= 1;
    }

    Ok((
        channel_major(&rgb[..frames * frame_bytes], area, frames),
        frames,
    ))
}

/// `h3_ffmpeg_read_audio_f32` (`h3_ffmpeg.c:395-521`): stereo 32 kHz F32 in
/// channel-major order.
pub fn read_audio_f32(
    path: &Path,
    max_samples: u32,
    truncate_at_limit: bool,
) -> Result<Vec<f32>, MediaError> {
    let max_samples = pixels(max_samples)?;
    if max_samples < MIN_SAMPLES || max_samples > SAMPLE_RATE as usize * MAX_SECONDS as usize {
        return Err(MediaError::new(
            "audio limit must be between 2 and 15 seconds at 32 kHz".to_string(),
        ));
    }

    let mut seconds = max_samples as f64 / f64::from(SAMPLE_RATE);
    if !truncate_at_limit {
        seconds += 1.0 / f64::from(SAMPLE_RATE);
    }

    let arguments = [
        "-v".to_string(),
        "error".to_string(),
        "-i".to_string(),
        path.display().to_string(),
        "-map".to_string(),
        "0:a:0".to_string(),
        "-vn".to_string(),
        "-ac".to_string(),
        CHANNELS.to_string(),
        "-ar".to_string(),
        SAMPLE_RATE.to_string(),
        "-t".to_string(),
        format!("{seconds:.9}"),
        "-f".to_string(),
        "f32le".to_string(),
        "pipe:1".to_string(),
    ];

    let capacity = max_samples * CHANNELS * std::mem::size_of::<f32>();
    let (raw, overflow) = capture_up_to(&program("H3_FFMPEG", "ffmpeg"), &arguments, capacity)?;
    let frame_bytes = CHANNELS * std::mem::size_of::<f32>();
    if raw.len() % frame_bytes != 0 {
        return Err(MediaError::new(
            "FFmpeg decoded a partial stereo frame".to_string(),
        ));
    }

    let samples = raw.len() / frame_bytes;
    if samples < MIN_SAMPLES {
        return Err(MediaError::new(
            "reference audio requires at least 2 seconds at 32 kHz".to_string(),
        ));
    }

    // A clip's own soundtrack may be cut at the bound; a standalone
    // reference that reaches it is an error (`h3_ffmpeg.c:459-478`).
    if !truncate_at_limit && overflow {
        return Err(MediaError::new(
            "reference audio exceeds the 15 second total limit".to_string(),
        ));
    }

    let values: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();

    let mut channel_major = vec![0.0f32; samples * CHANNELS];
    for sample in 0..samples {
        for channel in 0..CHANNELS {
            channel_major[channel * samples + sample] = values[sample * CHANNELS + channel];
        }
    }

    Ok(channel_major)
}

/// Read up to `capacity` bytes of stdout and report whether the stream went
/// past it. The reference asks for one byte more than the bound to tell a
/// full clip from an over-long one (`h3_ffmpeg.c:441-478`).
fn capture_up_to(
    program: &str,
    arguments: &[String],
    capacity: usize,
) -> Result<(Vec<u8>, bool), MediaError> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| MediaError::new(format!("cannot start {program}: {error}")))?;

    // Both pipes must keep moving, even when the decoder exceeds a bound.
    let stderr = child.stderr.take().expect("piped decoder stderr");
    let errors = std::thread::spawn(move || error_tail(stderr));
    let mut output = Vec::new();
    let read = (|| {
        let mut stdout = child.stdout.take().expect("piped decoder stdout");
        stdout
            .by_ref()
            .take(capacity as u64 + 1)
            .read_to_end(&mut output)?;
        std::io::copy(&mut stdout, &mut std::io::sink())
    })();
    if read.is_err() {
        let _ = child.kill();
    }

    let status = child.wait();
    let tail = errors.join().unwrap_or_default();
    read.map_err(|error| MediaError::new(format!("{program} output failed: {error}")))?;
    let status =
        status.map_err(|error| MediaError::new(format!("cannot wait for {program}: {error}")))?;

    if !status.success() {
        let detail = String::from_utf8_lossy(&tail);
        return Err(MediaError::new(format!(
            "{program} failed ({}): {}",
            status,
            detail.trim()
        )));
    }

    let overflow = output.len() > capacity;
    output.truncate(capacity);

    Ok((output, overflow))
}

fn error_tail(mut stream: impl Read) -> Vec<u8> {
    const KEEP: usize = 4096;
    let mut tail = Vec::new();
    let mut buffer = [0; KEEP];

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                tail.extend_from_slice(&buffer[..count]);
                if tail.len() > KEEP {
                    tail.drain(..tail.len() - KEEP);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }

    tail
}

/// `value / 255` into channel-major `[3,T,H,W]` (`h3_ffmpeg.c:271-280`,
/// `:377-390`).
fn channel_major(rgb: &[u8], area: usize, frames: usize) -> Vec<f32> {
    const SCALE: f32 = 1.0 / 255.0;

    let mut values = vec![0.0f32; frames * 3 * area];
    for time in 0..frames {
        for pixel in 0..area {
            for channel in 0..3 {
                let source = (time * area + pixel) * 3 + channel;
                values[(channel * frames + time) * area + pixel] = f32::from(rgb[source]) * SCALE;
            }
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_pipes_never_block() {
        const CHILD: &str = "DS4_H3_MEDIA_PIPE_TEST";

        if std::env::var_os(CHILD).is_some() {
            let args = [
                "-c".to_string(),
                "dd if=/dev/zero bs=65536 count=4 >&2 2>/dev/null; printf decoder-failed >&2; printf x; exit 1".to_string(),
            ];
            let exact = capture("sh", &args, 1).unwrap_err();
            let bounded = capture_up_to("sh", &args, 1).unwrap_err();

            for error in [exact, bounded] {
                assert!(error.detail().contains("decoder-failed"), "{error}");
                assert!(error.detail().len() < 8192);
            }

            let args = [
                "-c".to_string(),
                "dd if=/dev/zero bs=65536 count=4 2>/dev/null".to_string(),
            ];
            let (bytes, overflow) = capture_up_to("sh", &args, 4).unwrap();
            assert_eq!(bytes, [0; 4]);
            assert!(overflow);
            let exact = capture("sh", &args, 4).unwrap_err();
            assert!(exact.detail().contains("more than the 4 bytes"));
            return;
        }

        // A pipe regression must fail promptly rather than hang the test job.
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "minimax_h3::media::tests::decoder_pipes_never_block",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);

        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "decoder pipe checks failed: {status}");
                break;
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("decoder blocked on a full stdout or stderr pipe");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn pattern() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/h3/test-pattern.png")
    }

    fn fnv(values: &[f32]) -> u64 {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        super::super::text_forward::fnv1a(&bytes)
    }

    #[test]
    fn image_probes_at_coded_size() {
        let (width, height) = probe_visual_size(&pattern()).expect("ffprobe the pattern");

        assert_eq!((width, height), (64, 32));
    }

    #[test]
    fn a_stretched_image_is_channel_major_and_normalized() {
        let values = read_image_f32(&pattern(), 512, 192, Fit::Stretch).expect("stretch");

        assert_eq!(values.len(), 3 * 512 * 192);
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));

        // Channel-major: the red plane is first and it is not empty. The
        // reference's bytes are compared on the Spark, where its own FFmpeg
        // build decodes the same file; a hash pinned here would track this
        // machine's scaler, not the port.
        assert!(values[..512 * 192].iter().any(|value| *value > 0.0));
    }

    #[test]
    fn cover_crops_instead_of_stretching() {
        let stretched = read_image_f32(&pattern(), 512, 192, Fit::Stretch).expect("stretch");
        let covered = read_image_f32(&pattern(), 512, 192, Fit::Cover).expect("cover");

        assert_eq!(covered.len(), stretched.len());
        assert_ne!(fnv(&stretched), fnv(&covered));
    }

    /// Build a probe clip from the pattern through the same FFmpeg the reader
    /// uses, exactly as a reference video would arrive.
    fn clip(name: &str, frames: u32, size: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("h3-media-{name}-{}.mp4", std::process::id()));
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                &pattern().display().to_string(),
                "-frames:v",
                &frames.to_string(),
                "-vf",
                size,
                "-r",
                "24",
                "-pix_fmt",
                "yuv420p",
                path.to_str().expect("a UTF-8 temp path"),
            ])
            .status()
            .expect("ffmpeg builds the probe clip");

        assert!(status.success(), "ffmpeg could not build {name}");

        path
    }

    #[test]
    fn a_clip_is_trimmed_to_the_released_cadence() {
        // 30 decoded frames at 24 fps, bounded at 48: 30 is not 5 + 17k, so
        // the reader drops down to 22 (`h3_ffmpeg.c:372-374`).
        let path = clip("clip", 30, "scale=512:192");

        let (values, frames) = read_video_f32(&path, 512, 192, 48).expect("the clip");

        assert_eq!(frames, 22);
        assert_eq!(values.len(), 3 * 22 * 512 * 192);
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_short_clip_is_refused() {
        let path = clip("short", 4, "scale=256:96");

        let error = read_video_f32(&path, 256, 96, 34).expect_err("four frames");

        assert!(error.detail().contains("at least 5"), "{error}");

        let _ = std::fs::remove_file(&path);
    }

    /// A stereo 32 kHz probe soundtrack.
    fn soundtrack(name: &str, seconds: f32) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("h3-media-{name}-{}.wav", std::process::id()));
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("sine=frequency=440:duration={seconds}"),
                "-ac",
                "2",
                "-ar",
                "32000",
                path.to_str().expect("a UTF-8 temp path"),
            ])
            .status()
            .expect("ffmpeg builds the probe audio");

        assert!(status.success(), "ffmpeg could not build {name}");

        path
    }

    #[test]
    fn audio_comes_back_channel_major_at_32_khz() {
        let path = soundtrack("tone", 2.5);

        let pcm = read_audio_f32(&path, 96_000, false).expect("the tone");

        assert_eq!(pcm.len(), 2 * 80_000);
        // The two channels are the same tone, so they agree sample for sample.
        assert_eq!(pcm[..80_000], pcm[80_000..]);
        // FFmpeg's sine source is fixed at an amplitude of 0.0881.
        let peak = pcm.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        assert!((0.05..0.2).contains(&peak), "peak {peak}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_standalone_reference_may_not_be_truncated() {
        // Four seconds against a two-second bound: a clip's soundtrack would
        // be cut here, a standalone reference is refused (`h3_ffmpeg.c:459-478`).
        let path = soundtrack("long", 4.0);

        let error = read_audio_f32(&path, 64_000, false).expect_err("four seconds");
        assert!(error.detail().contains("15 second"), "{error}");

        let pcm = read_audio_f32(&path, 64_000, true).expect("truncated");
        assert_eq!(pcm.len(), 2 * 64_000);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_absent_image_is_refused_by_name() {
        let missing = std::env::temp_dir().join("h3-media-absent.png");
        let error = read_image_f32(&missing, 64, 64, Fit::Stretch).expect_err("absent");

        assert!(error.detail().contains("ffmpeg failed"), "{error}");
    }
}
