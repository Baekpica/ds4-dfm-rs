//! The P12.4 gate: the port's VideoVAE encoder against the reference's own
//! latent, hashed byte for byte.
//!
//! Usage: h3_video_encoder_gate <FL2VA/video_vae/source> <media>
//!        [--height H] [--width W] [--frames N] [--expect HEX]
//!
//! `<media>` is decoded by the same reader the reference uses
//! (`h3_ffmpeg.c:143-283` for one image, `:285-393` for a clip), so both
//! sides see the same channel-major pixels when they are given the same file
//! and the same extent. `--frames 1` takes the image path; five frames and up
//! take the clip path and are trimmed to the released `5 + 17k` cadence, so
//! 2..=4 is refused instead of silently resampled.
//!
//! Output, in order:
//!
//!   PIXELS frames= height= width= fnv= first= last=
//!   LATENT t= h= w= values= fnv= first= last=
//!   SECONDS total= encode=
//!
//! The hash covers the latent as `h3_video_latent.values` carries it, which
//! is channel-major `[24,t,h,w]` after the released normalization
//! (`h3_video_encoder.c:722-741`, means and deviations from
//! `<dir>/../config.json`). It is the same FNV-1a convention as the rest of
//! the h3 gates, so the numbers printed here and by
//! `tools/minimax_h3/video_encoder_probe.c` are directly comparable.
//!
//! With `--expect` the process exits 3 on a mismatch.
//!
//! Gated on the Spark on 2026-10-24 against `video_encoder_probe.c` with the
//! pilot image at 512x192, the fox clip's five frames at 512x192 and the
//! pilot image at 512x320 (two tile rows, so a vertical 64 px blend), all
//! three equal to the reference byte for byte:
//!
//!   LATENT fnv=ca995ec30255b666  t=1 h=12 w=32   3 tiles  512x192 frames 1
//!   LATENT fnv=d039bbcf2a108277  t=2 h=12 w=32   3 tiles  512x192 frames 5
//!   LATENT fnv=e67374beb0e376bc  t=1 h=20 w=32   6 tiles  512x320 frames 1
//!
//! Encoder wall time, port against reference: 13.460/12.824 s, 45.258/45.195 s
//! and 34.540/34.240 s.

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::media::{self, Fit};
use ds4_core::minimax_h3::text_forward;
use ds4_core::minimax_h3::vae_encoder;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match run(&args) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let (positional, options) = split(args)?;
    if positional.len() != 2 {
        return Err(usage("expected two positional arguments"));
    }

    let directory = std::path::Path::new(&positional[0]);
    let media_path = std::path::Path::new(&positional[1]);

    let started = Instant::now();
    let (pixels, frames) = read_pixels(media_path, &options)?;
    let read_seconds = started.elapsed().as_secs_f64();
    println!(
        "PIXELS frames={} height={} width={} fnv={:016x} first={:.9} last={:.9}",
        frames,
        options.height,
        options.width,
        fnv_f32(&pixels),
        pixels[0],
        pixels[pixels.len() - 1]
    );

    let started = Instant::now();
    let latent = vae_encoder::encode(
        directory,
        &pixels,
        frames,
        options.height as usize,
        options.width as usize,
        |completed, total| {
            if completed == total || completed % 8 == 0 {
                eprintln!("tile {completed}/{total}");
            }
        },
    )
    .map_err(|error| error.to_string())?;
    let encode_seconds = started.elapsed().as_secs_f64();

    let hash = fnv_f32(&latent.values);
    println!(
        "LATENT t={} h={} w={} values={} fnv={:016x} first={:.9} last={:.9}",
        latent.time,
        latent.height,
        latent.width,
        latent.values.len(),
        hash,
        latent.values[0],
        latent.values[latent.values.len() - 1]
    );
    println!(
        "SECONDS read={read_seconds:.3} encode={encode_seconds:.3} total={:.3}",
        read_seconds + encode_seconds
    );

    if let Some(expect) = options.expect {
        if hash != expect {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// The reference decodes one image for a single frame and a clip at the
/// released 24 fps otherwise (`h3_ffmpeg.c:285-393`).
fn read_pixels(path: &std::path::Path, options: &Options) -> Result<(Vec<f32>, usize), String> {
    if options.frames == 1 {
        let pixels = media::read_image_f32(path, options.width, options.height, Fit::Stretch)
            .map_err(|error| error.detail().to_string())?;
        return Ok((pixels, 1));
    }

    if options.frames < media::MIN_FRAMES {
        return Err(usage("--frames must be 1 or at least 5"));
    }

    let (pixels, frames) =
        media::read_video_f32(path, options.width, options.height, options.frames as u32)
            .map_err(|error| error.detail().to_string())?;
    Ok((pixels, frames))
}

fn fnv_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    text_forward::fnv1a(&bytes)
}

fn parse_hash(text: &str) -> Option<u64> {
    if text.len() != 16 {
        return None;
    }
    u64::from_str_radix(text, 16).ok()
}

#[derive(Debug)]
struct Options {
    height: u32,
    width: u32,
    frames: usize,
    expect: Option<u64>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        height: 192,
        width: 512,
        frames: 1,
        expect: None,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];
        let value = || {
            args.get(index + 1)
                .ok_or_else(|| usage(&format!("{argument} needs a value")))
        };

        match argument.as_str() {
            "--height" => options.height = number(value()?)?,
            "--width" => options.width = number(value()?)?,
            "--frames" => options.frames = number(value()?)?,
            "--expect" => {
                options.expect = Some(
                    parse_hash(value()?).ok_or_else(|| usage("--expect needs 16 hex digits"))?,
                );
            }
            _ if argument.starts_with("--") => {
                return Err(usage(&format!("unknown option {argument}")));
            }
            _ => {
                positional.push(argument.clone());
                index += 1;
                continue;
            }
        }
        index += 2;
    }

    Ok((positional, options))
}

fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse::<T>()
        .map_err(|_| usage(&format!("{text} is not a number")))
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_video_encoder_gate <FL2VA/video_vae/source> <media> \
         [--height H] [--width W] [--frames N] [--expect HEX]"
    )
}
