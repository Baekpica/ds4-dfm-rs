//! Concurrent FFmpeg video/PCM pipes: mux, probe, then read every stream back.
//!
//! Reference: `/data/h3.c` at `eae5d54`, `tests/test_av_mux.c:18-110`.
//! The fixture is the reference's own: 32x32, 8 frames, a red/green ramp with a
//! blue frame ramp, and two 220/330 Hz tones at 32 kHz. Assertions, in the
//! reference's order:
//!
//! - the container exists and holds more than 1000 bytes;
//! - FFprobe reports the coded size (32x32);
//! - one decoded image is channel-major, finite, in [0,1], red and green sums
//!   above 100 and blue below 80, so the ramp arrives in the right planes;
//! - a cover fit at twice the height decodes without error;
//! - the clip is trimmed to the released `5 + 17k` cadence: 8 input frames
//!   decode as 5, all finite and in range;
//! - the soundtrack returns exactly 64000 samples, both channels carry
//!   energy, and the peak stays under the anti-clipping ceiling.
//!
//! Model-free, but it needs `ffmpeg` and `ffprobe` on PATH (`H3_FFMPEG` and
//! `H3_FFPROBE` override them, as in the port's readers). Run with
//! `make tests/h3-test_av_mux`; the optional argument is the output path.

use std::path::Path;
use std::process::ExitCode;

use ds4_core::minimax_h3::media::{
    probe_visual_size, read_audio_f32, read_image_f32, read_video_f32, Fit,
};
use ds4_core::minimax_h3::render::{mux_mp4, RgbFrames};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;
const FRAMES: u32 = 8;
const SAMPLES: u32 = 64_000;
const CHANNELS: u32 = 2;
const SAMPLE_RATE: u32 = 32_000;

/// The reference's default output path (`tests/test_av_mux.c:26`).
const DEFAULT_PATH: &str = "/tmp/h3-av-mux-test.mp4";

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PATH.to_string());

    match run(Path::new(&path)) {
        Ok(bytes) => {
            println!("ok: concurrent FFmpeg video/PCM pipes created {path} ({bytes} bytes)");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("FAIL tests/h3/test_av_mux.rs: {message}");
            ExitCode::from(1)
        }
    }
}

fn run(path: &Path) -> Result<u64, String> {
    let frames = RgbFrames::from_f32(&fixture_rgb(), FRAMES as i32, WIDTH as i32, HEIGHT as i32)?;
    let pcm = fixture_pcm();

    mux_mp4(
        Path::new(&program("H3_FFMPEG", "ffmpeg")),
        path,
        &frames,
        &pcm,
        SAMPLES as i32,
        CHANNELS as i32,
        SAMPLE_RATE as i32,
    )?;

    let bytes = std::fs::metadata(path)
        .map_err(|error| format!("cannot stat {}: {error}", path.display()))?
        .len();

    if bytes <= 1000 {
        return Err(format!(
            "FFmpeg wrote {bytes} bytes, expected more than 1000"
        ));
    }

    let (width, height) = probe_visual_size(path).map_err(|error| error.to_string())?;

    if (width, height) != (WIDTH, HEIGHT) {
        return Err(format!(
            "FFprobe reported {width}x{height}, expected {WIDTH}x{HEIGHT}"
        ));
    }

    check_still(path)?;
    check_cover(path)?;
    check_clip(path)?;
    check_soundtrack(path)?;

    Ok(bytes)
}

/// `h3_ffmpeg_read_image_f32` with a stretch fit (`tests/test_av_mux.c:55-74`).
fn check_still(path: &Path) -> Result<(), String> {
    let values =
        read_image_f32(path, WIDTH, HEIGHT, Fit::Stretch).map_err(|error| error.to_string())?;
    let area = (WIDTH * HEIGHT) as usize;

    if values.len() != 3 * area {
        return Err(format!(
            "decoded image holds {} values, expected {}",
            values.len(),
            3 * area
        ));
    }

    let mut sums = [0.0f64; 3];

    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 || *value > 1.0 {
            return Err(format!("decoded image value {index} is {value}"));
        }

        sums[index / area] += f64::from(*value);
    }

    if sums[0] < 100.0 || sums[1] < 100.0 || sums[2] > 80.0 {
        return Err(format!(
            "decoded image has unexpected channel sums {sums:?}"
        ));
    }

    Ok(())
}

/// The cover arm only has to decode: the aspect that fits is the taller frame.
fn check_cover(path: &Path) -> Result<(), String> {
    let values =
        read_image_f32(path, WIDTH, HEIGHT * 2, Fit::Cover).map_err(|error| error.to_string())?;
    let expected = ((WIDTH * HEIGHT * 2) * 3) as usize;

    if values.len() != expected {
        return Err(format!(
            "cover image holds {} values, expected {expected}",
            values.len()
        ));
    }

    Ok(())
}

/// `h3_ffmpeg_read_video_f32`: an 8-frame clip trims to 5, the released cadence.
fn check_clip(path: &Path) -> Result<(), String> {
    let (values, frames) =
        read_video_f32(path, WIDTH, HEIGHT, FRAMES).map_err(|error| error.to_string())?;

    if frames != 5 {
        return Err(format!(
            "video decoded {frames} frames, expected the 5+17k trim to 5"
        ));
    }

    if values.len() != (frames * (WIDTH * HEIGHT * 3) as usize) {
        return Err(format!(
            "video holds {} values for {frames} frames",
            values.len()
        ));
    }

    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 || *value > 1.0 {
            return Err(format!("decoded video value {index} is {value}"));
        }
    }

    Ok(())
}

/// `h3_ffmpeg_read_audio_f32`: 2 s at 32 kHz, both channels alive, no clip.
fn check_soundtrack(path: &Path) -> Result<(), String> {
    let values = read_audio_f32(path, SAMPLES, true).map_err(|error| error.to_string())?;
    let samples = SAMPLES as usize;

    if values.len() != 2 * samples {
        return Err(format!(
            "audio holds {} values, expected {}",
            values.len(),
            2 * samples
        ));
    }

    let mut energy = [0.0f64; 2];
    let mut peak = 0.0f32;

    for sample in 0..samples {
        for channel in 0..2 {
            let value = values[channel * samples + sample];

            if !value.is_finite() {
                return Err(format!("decoded audio sample {sample} is {value}"));
            }

            energy[channel] += f64::from(value) * f64::from(value);
            peak = peak.max(value.abs());
        }
    }

    if energy[0] < 1.0 || energy[1] < 1.0 {
        return Err(format!(
            "decoded audio has no stereo signal: energies {energy:?}"
        ));
    }

    if peak > 1.0 {
        return Err(format!("decoded audio peak {peak} exceeds 1.0"));
    }

    Ok(())
}

/// `tests/test_av_mux.c:29-42`: frame-major RGB in [0,1].
fn fixture_rgb() -> Vec<f32> {
    let frame_values = (HEIGHT * WIDTH * 3) as usize;
    let mut rgb = vec![0.0f32; FRAMES as usize * frame_values];

    for frame in 0..FRAMES {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let pixel = frame as usize * frame_values + (y * WIDTH + x) as usize * 3;
                rgb[pixel] = f32::from((x * 8) as u8) / 255.0;
                rgb[pixel + 1] = f32::from((y * 8) as u8) / 255.0;
                rgb[pixel + 2] = f32::from((frame * 32) as u8) / 255.0;
            }
        }
    }

    rgb
}

/// `tests/test_av_mux.c:43-51`: channel-major 220/330 Hz tones above full scale,
/// which the mux's clamp and limiter must bring inside the ceiling.
fn fixture_pcm() -> Vec<f32> {
    let samples = SAMPLES as usize;
    let mut pcm = vec![0.0f32; samples * CHANNELS as usize];

    for channel in 0..CHANNELS as usize {
        for sample in 0..samples {
            let frequency = 220.0 + channel as f32 * 110.0;
            let phase = 2.0 * std::f32::consts::PI * frequency * sample as f32 / SAMPLE_RATE as f32;
            pcm[channel * samples + sample] = 1.5 * phase.sin();
        }
    }

    pcm
}

/// `ffmpeg_program` (`h3_ffmpeg.c:21-24`): the environment override, or the name.
fn program(variable: &str, default: &str) -> String {
    match std::env::var(variable) {
        Ok(value) if !value.is_empty() => value,
        _ => default.to_string(),
    }
}
