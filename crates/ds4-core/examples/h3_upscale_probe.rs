//! Run the latent upscaler on a real saved latent.
//!
//! Usage: h3_upscale_probe --latent-file IN.bf16 --latent T,H,W
//!        [--out-shape HO,WO] [--out OUT.bf16] [--dir DIR]
//!
//! `h3-rs --save-latent` writes the denoised video latent as raw BF16 in the
//! product route's own layout, `[channel][time][height][width]`, and
//! `h3_vae_gate --latent-file` decodes one. This probe is the step between
//! them: it applies the released resizer to a latent the model actually
//! produced, prints per-channel statistics of both sides, and writes the
//! result in the same layout so a decoder can be pointed at it.
//!
//! Why it exists: `h3_upscale_gate` checks the resizer against a host
//! transcription at 2 x 2 x 2 and *times* it at 32 x 34 x 58, so no instrument
//! said what the network does to a real latent at a real shape until the
//! two-stage route rendered a lattice.
//!
//! The checkpoint's directory comes from `--dir`, else `H3_UPSCALER_DIR`; a
//! path in this file would be an assumption about someone else's disk.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ds4_core::minimax_h3::latent_upscale::{Upscaler, LATENT_CHANNELS};
use ds4_core::minimax_h3::vision::f32_to_bf16;
use ds4_sys::h3_dit::Gpu;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(detail) => {
            eprintln!("FAIL {detail}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut dir: Option<PathBuf> = None;
    let mut latent_file: Option<PathBuf> = None;
    let mut out_file: Option<PathBuf> = None;
    let mut shape: Option<(usize, usize, usize)> = None;
    let mut out_shape: Option<(usize, usize)> = None;

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].clone();
        let mut value = || -> Result<String, String> {
            index += 1;
            arguments
                .get(index)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--dir" => dir = Some(PathBuf::from(value()?)),
            "--latent-file" => latent_file = Some(PathBuf::from(value()?)),
            "--out" => out_file = Some(PathBuf::from(value()?)),
            "--latent" => {
                let text = value()?;
                let parts: Vec<usize> = text
                    .split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect();
                if parts.len() != 3 {
                    return Err("--latent wants T,H,W".to_string());
                }
                shape = Some((parts[0], parts[1], parts[2]));
            }
            "--out-shape" => {
                let text = value()?;
                let parts: Vec<usize> = text
                    .split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect();
                if parts.len() != 2 {
                    return Err("--out-shape wants HO,WO".to_string());
                }
                out_shape = Some((parts[0], parts[1]));
            }
            "-h" | "--help" => {
                println!(
                    "usage: h3_upscale_probe --latent-file IN.bf16 --latent T,H,W \
                     [--out-shape HO,WO] [--out OUT.bf16] [--dir DIR]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    let shape = shape.ok_or("--latent T,H,W is required")?;
    let path = latent_file.ok_or("--latent-file is required")?;
    let dir = match dir {
        Some(dir) => dir,
        None => std::env::var_os("H3_UPSCALER_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or("--dir or H3_UPSCALER_DIR is required")?,
    };
    let (time, height, width) = shape;
    let (out_height, out_width) = out_shape.unwrap_or((height * 2, width * 2));

    let voxels = time * height * width;
    let latent = read_bf16(&path, LATENT_CHANNELS * voxels)?;
    println!(
        "IN latent={time}x{height}x{width} channels={LATENT_CHANNELS} \
         statistics={}",
        statistics(&latent, voxels)
    );

    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;
    let upscaler = Upscaler::open(&dir, gpu).map_err(|detail| detail.to_string())?;
    let started = std::time::Instant::now();
    let output = upscaler
        .upscale(&latent, time, height, width, out_height, out_width)
        .map_err(|detail| detail.to_string())?;
    let elapsed = started.elapsed();

    let out_voxels = output.time * output.height * output.width;
    println!(
        "OUT latent={}x{}x{} arm={} seconds={:.3} statistics={}",
        output.time,
        output.height,
        output.width,
        upscaler.arm(),
        elapsed.as_secs_f64(),
        statistics(&output.values, out_voxels)
    );

    if let Some(path) = out_file {
        write_bf16(&path, &output.values)?;
        println!("WROTE {} values={}", path.display(), output.values.len());
    }

    Ok(())
}

/// Mean and standard deviation of each channel, folded into one line: a
/// degenerate upscale (a constant field, a saturated one, a lattice) shows up
/// here before it is decoded.
fn statistics(values: &[f32], voxels: usize) -> String {
    let mut text = String::new();
    let channels = values.len() / voxels.max(1);
    for channel in 0..channels.min(24) {
        let slice = &values[channel * voxels..(channel + 1) * voxels];
        let mean = slice.iter().map(|value| *value as f64).sum::<f64>() / voxels as f64;
        let variance = slice
            .iter()
            .map(|value| (*value as f64 - mean).powi(2))
            .sum::<f64>()
            / voxels as f64;
        text.push_str(&format!("{channel}:{mean:.3}/{:.3} ", variance.sqrt()));
    }
    text.trim_end().to_string()
}

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

fn write_bf16(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f32_to_bf16(*value).to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}
