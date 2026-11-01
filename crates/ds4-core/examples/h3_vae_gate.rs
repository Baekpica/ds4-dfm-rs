//! What does the H3 video VAE decoder cost at one canvas, and what does its
//! arithmetic arm change?
//!
//! Usage: h3_vae_gate --latent T,H,W [--shape PIXEL_H,PIXEL_W] [--latent-file
//!        FILE] [--seed N] [--repeat N] [--expect HEX]
//!
//! The decoder alone: no DiT, no prompt, no mux. `--latent T,H,W` is the
//! latent's own shape in the layout the product route writes, `[channel]
//! [time][height][width]`, and the canvas follows from it at 16 pixels per
//! latent edge; `--shape` only checks that derivation. `--latent-file` reads a
//! BF16 dump of that shape (`h3-rs --save-latent`, `h3-sol-rs --save-upscaled`);
//! without it the gate draws a deterministic latent from `--seed`, which is
//! enough to time an arm and not enough to judge one.
//!
//! The arms are the decoder's own environment switches, so the gate reads none
//! of them itself: `H3_VAE_GEMM` (fp32 | tf32 | bf16 | bf16i), `H3_VAE_ATTENTION`
//! and `H3_VAE_TILE_PIXELS` are the released surface (`h3_video_vae.c:666-690`,
//! `docs/minimax-h3/*`), and `H3_VAE_TIMING=1` splits each chunk.
//!
//! Live gate: the weights come from `H3_VAE`, never from a path in this file.
//! Unset, or absent on disk, it exits 3, which is the convention `make
//! test-h3` reads.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::vae_video::{self, LATENT_CHANNELS};

/// Pixels per latent edge (`h3_video_vae.c:28`).
const SPATIAL_RATIO: usize = 16;

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut latent_shape: Option<[usize; 3]> = None;
    let mut pixel_shape: Option<[usize; 2]> = None;
    let mut latent_file: Option<PathBuf> = None;
    let mut seed = 1u64;
    let mut repeat = 1usize;
    let mut expect: Option<String> = None;
    let mut dump: Option<PathBuf> = None;

    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let mut value = || -> Result<&String, String> {
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--latent" => latent_shape = Some(parse_triple(value()?)?),
            "--shape" => pixel_shape = Some(parse_pair(value()?)?),
            "--latent-file" => latent_file = Some(PathBuf::from(value()?)),
            "--seed" => seed = parse(&value()?, "integer")?,
            "--repeat" => repeat = parse(&value()?, "integer")?,
            "--expect" => expect = Some(value()?.clone()),
            "--dump" => dump = Some(PathBuf::from(value()?)),
            "-h" | "--help" => {
                println!(
                    "usage: h3_vae_gate --latent T,H,W [--shape H,W] [--latent-file FILE] \
                     [--seed N] [--repeat N] [--expect HEX] [--dump FILE]"
                );
                return Ok(ExitCode::SUCCESS);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    let [time, latent_h, latent_w] = latent_shape.ok_or("--latent T,H,W is required")?;
    let pixel = [latent_h * SPATIAL_RATIO, latent_w * SPATIAL_RATIO];
    if let Some(named) = pixel_shape {
        if named != pixel {
            return Err(format!(
                "a {time}x{latent_h}x{latent_w} latent decodes to {}x{}, not {}x{}",
                pixel[0], pixel[1], named[0], named[1]
            ));
        }
    }

    // No default path: the weights are the caller's setting.
    let directory = match std::env::var_os("H3_VAE") {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => {
            println!("SKIP H3_VAE is not set");
            return Ok(ExitCode::from(3));
        }
    };
    if !directory.is_dir() {
        println!("SKIP missing video VAE directory {}", directory.display());
        return Ok(ExitCode::from(3));
    }

    let (pixels, tiles_y, tiles_x) =
        vae_video::tile_plan(pixel[0], pixel[1]).map_err(|error| format!("tile plan: {error}"))?;
    let (chunks, frames) = vae_video::frame_plan(time).map_err(|error| error.to_string())?;
    println!(
        "PLAN canvas={}x{} latent={}x{}x{} tile={} tiles={}x{} chunks={} frames={}",
        pixel[1], pixel[0], time, latent_h, latent_w, pixels, tiles_x, tiles_y, chunks, frames
    );

    let latent = match &latent_file {
        Some(path) => read_bf16(path, time * latent_h * latent_w * LATENT_CHANNELS)?,
        None => draw(seed, time * latent_h * latent_w * LATENT_CHANNELS),
    };

    for attempt in 0..repeat {
        let started = Instant::now();
        let decoded = vae_video::decode(
            &directory,
            &latent,
            time,
            latent_h,
            latent_w,
            // The gate's arms are the environment's, which is the point: each
            // run of the harness is one word on one binary.
            vae_video::Arms::default(),
        )
        .map_err(|error| format!("decode: {error}"))?;
        let seconds = started.elapsed().as_secs_f64();
        let hash = fnv_f32(&decoded.rgb);
        println!(
            "RUN {attempt} seconds={seconds:.3} frames={} height={} width={} hash={hash:016x} first={:.9} last={:.9}",
            decoded.frames,
            decoded.height,
            decoded.width,
            decoded.rgb[0],
            decoded.rgb[decoded.rgb.len() - 1]
        );

        if let Some(path) = &dump {
            write_rgb8(path, &decoded.rgb)?;
            println!("DUMP path={} bytes={}", path.display(), decoded.rgb.len());
        }

        if let Some(wanted) = &expect {
            let wanted = wanted.trim().trim_start_matches("0x").to_lowercase();
            let found = format!("{hash:016x}");
            let same = found == wanted;
            println!("MATCH expect={wanted} found={found} identical={same}");
            if !same {
                return Ok(ExitCode::from(3));
            }
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// The frames as raw 8-bit RGB, frame-major, for an outside comparison: the
/// decoder's own output through the product's own fold
/// (`render::rgb_f32_to_u8`), so a pair of dumps is the phase's deviation and
/// not the mux's.
fn write_rgb8(path: &std::path::Path, rgb: &[f32]) -> Result<(), String> {
    let bytes = ds4_core::minimax_h3::render::rgb_f32_to_u8(rgb);
    std::fs::write(path, &bytes).map_err(|error| format!("{}: {error}", path.display()))
}

/// A deterministic latent in `[-1, 1]`: two splitmix64 draws per value, so a
/// rerun of an arm is the same input and an arm's timing is comparable.
fn draw(seed: u64, elements: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    };

    (0..elements).map(|_| (2.0 * next() - 1.0) as f32).collect()
}

/// A BF16 dump in `[channel][time][height][width]`, as `write_latent` writes.
fn read_bf16(path: &Path, elements: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != elements * 2 {
        return Err(format!(
            "{} holds {} bytes, the shape needs {}",
            path.display(),
            bytes.len(),
            elements * 2
        ));
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect())
}

fn fnv_f32(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn parse<T: std::str::FromStr>(text: &str, what: &str) -> Result<T, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("{what} expected, got {text}"))
}

fn parse_triple(text: &str) -> Result<[usize; 3], String> {
    let values: Vec<usize> = text
        .split(',')
        .map(|part| parse(part, "integer"))
        .collect::<Result<_, _>>()?;
    <[usize; 3]>::try_from(values).map_err(|_| format!("three values expected, got {text}"))
}

fn parse_pair(text: &str) -> Result<[usize; 2], String> {
    let values: Vec<usize> = text
        .split(',')
        .map(|part| parse(part, "integer"))
        .collect::<Result<_, _>>()?;
    <[usize; 2]>::try_from(values).map_err(|_| format!("two values expected, got {text}"))
}
