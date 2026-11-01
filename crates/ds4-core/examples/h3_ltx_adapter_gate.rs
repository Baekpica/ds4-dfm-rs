//! The P2 gate: the port's latent path against the reference's own tensors.
//!
//! Usage: h3_ltx_adapter_gate <golden-dir> <upscaler.safetensors> <adapter-dir>
//!        [--arm f32|bf16] [--relative N]
//!
//! `<golden-dir>` is what `tools/minimax_h3/ltx/latent_oracle.py` writes: a
//! `latents.safetensors` (or the older `bridge.safetensors`) plus the
//! `manifest.json` that pins the geometry, the pinned node and both
//! checkpoints. The four tensors in it are the reference's own float32 outputs
//! at the release's geometry, produced by the profile's Python — the latent
//! path has no C reference, so that Python is the authority and this is its
//! frozen vector.
//!
//! The port is run twice over the same input:
//!
//!   LATENT  h3_input  24 x 37 x 24 x 42   the node's `N(model(D(x)))`
//!   LATENT  h3_upscaled       48 x 84     the alignment plus the 22 blocks
//!   LATENT  refiner_input     16 frames   `adapter_output[:, :, :16]`
//!
//! The f32 arm is the structural gate: it must reproduce the reference to
//! within a few parts in 10^4. The bf16 arm is the released arithmetic and is
//! reported against the same float32 vector at the noise floor bf16 storage
//! implies, which is a statement about precision, not about the structure the
//! f32 arm already proves.
//!
//! Exit 3 when a stage exceeds its tolerance.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::ltx_adapter::{
    geometry, temporal_plan, Adapter, Arm, Geometry, TemporalPlan, Upscaler, H3_LATENT_CHANNELS,
    REFINER_FRAMES, TEMPORAL_PACK_SLOTS,
};
use ds4_core::minimax_h3::safetensors;
use ds4_sys::h3_dit::{Dtype, Gpu};

/// The relative-RMS the f32 arm must stay inside: it is the same arithmetic as
/// the reference's own float32 run, so only the summation order differs.
const F32_RELATIVE: f64 = 1e-4;
/// What bf16 storage implies: an 8-bit mantissa is 4e-3 per element, and the
/// two models together are 46 convolutions deep.
const BF16_RELATIVE: f64 = 3e-2;

struct Options {
    golden: PathBuf,
    upscaler: PathBuf,
    adapter: PathBuf,
    arm: Arm,
    relative: f64,
    /// Where the port's own refiner-shaped latent goes when it is asked for:
    /// the `--latent` file `h3_ltx_refiner_gate --stage` reads.
    refiner_latent: Option<PathBuf>,
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(arguments: &[String]) -> Result<ExitCode, String> {
    let options = parse(arguments)?;
    let (values, geometry) = read_golden(&options.golden)?;
    let h3_input = values
        .get("h3_input")
        .ok_or_else(|| "golden has no h3_input".to_string())?;
    let expected_upscaled = values
        .get("h3_upscaled")
        .ok_or_else(|| "golden has no h3_upscaled".to_string())?;
    let expected_adapter = values
        .get("adapter_output")
        .ok_or_else(|| "golden has no adapter_output".to_string())?;
    let expected_refiner = values
        .get("refiner_input")
        .ok_or_else(|| "golden has no refiner_input".to_string())?;

    let relative = if options.relative > 0.0 {
        options.relative
    } else {
        match options.arm {
            Arm::F32 => F32_RELATIVE,
            Arm::Bf16 => BF16_RELATIVE,
        }
    };
    println!(
        "GOLDEN frames={} draft={}x{} canvas={}x{} arm={} relative={relative:e}",
        geometry.pixel_frames,
        geometry.draft_height,
        geometry.draft_width,
        geometry.canvas_height,
        geometry.canvas_width,
        options.arm.name()
    );

    // The input the reference itself fed, rounded the way the reference rounds
    // it (`h3_upscale.py:82`'s cast to the compute dtype).
    let gpu = Gpu::create()?;
    let input = upload(&gpu, options.arm, h3_input)?;

    let mut failures = 0usize;

    println!("=== upscaler: the node's N(model(D(x))) ===");
    let started = Instant::now();
    let upscaler = Upscaler::load(&options.upscaler, options.arm, &geometry)
        .map_err(|error| error.to_string())?;
    let load_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let upscaled = upscaler
        .forward(&input)
        .map_err(|error| error.to_string())?;
    let forward_seconds = started.elapsed().as_secs_f64();
    let produced = download(&gpu, &upscaled)?;
    if !compare("h3_upscaled", &produced, expected_upscaled, relative)? {
        failures += 1;
    }
    println!("SECONDS upscaler_load={load_seconds:.3} upscaler_forward={forward_seconds:.3}");

    println!("=== adapter: alignment, 22 blocks, the 2x shuffle ===");
    let started = Instant::now();
    let adapter = Adapter::load(&options.adapter, options.arm, &geometry)
        .map_err(|error| error.to_string())?;
    let load_seconds = started.elapsed().as_secs_f64();

    // The alignment on its own, against the reference's own upscaled latent:
    // a host implementation of `geometry.py:81-112`, so the model is not in
    // the way of the answer.
    let plan = temporal_plan(geometry.pixel_frames).map_err(|error| error.to_string())?;
    let reference_input = upload(&gpu, options.arm, expected_upscaled)?;
    let aligned = adapter
        .align(&reference_input)
        .map_err(|error| error.to_string())?;
    let host_aligned = host_align(expected_upscaled, &geometry, &plan);
    if !compare(
        "aligned_h3",
        &download(&gpu, aligned)?,
        &host_aligned,
        relative,
    )? {
        failures += 1;
    }

    // The model alone, on the host's own alignment, so an alignment
    // disagreement cannot mask a model one. This is a gate in the f32 arm,
    // where the input is the same tensor the device computed; in the bf16 arm
    // the host alignment is an f32 computation rounded once, which is a
    // different bf16 input — so there it is reported as the model's input
    // sensitivity, not judged.
    let aligned_input = upload(&gpu, options.arm, &host_aligned)?;
    let from_host = adapter
        .model(&aligned_input)
        .map_err(|error| error.to_string())?;
    let produced = download(&gpu, &from_host)?;
    let judged = options.arm == Arm::F32;
    if !judged {
        print!("INFO (not judged at bf16) ");
    }
    if !compare("adapter_from_host", &produced, expected_adapter, relative)? && judged {
        failures += 1;
    }

    let started = Instant::now();
    let latent = adapter
        .convert(&upscaled)
        .map_err(|error| error.to_string())?;
    let convert_seconds = started.elapsed().as_secs_f64();
    let produced = download(&gpu, &latent)?;
    if !compare("adapter_output", &produced, expected_adapter, relative)? {
        profile(
            "adapter_output",
            &produced,
            expected_adapter,
            geometry.ltx[0],
            geometry.ltx[1],
            geometry.ltx[2] * geometry.ltx[3],
        );
        failures += 1;
    }
    // `latent[:, :, :16]` (`stage2.py:76`) is a crop of the temporal axis, so
    // it is strided per channel, not a prefix of the flat tensor.
    let channels = geometry.ltx[0];
    let frames = geometry.ltx[1];
    let plane = geometry.ltx[2] * geometry.ltx[3];
    let cropped: Vec<f32> = (0..channels)
        .flat_map(|channel| {
            let begin = channel * frames * plane;
            produced[begin..begin + 16 * plane].to_vec()
        })
        .collect();
    if !compare("refiner_input", &cropped, expected_refiner, relative)? {
        failures += 1;
    }
    println!("SECONDS adapter_load={load_seconds:.3} adapter_convert={convert_seconds:.3}");

    // The port's own latent for the refiner, in the shape and dtype the refiner
    // gate's `--latent` reads. The reference's vector above is a comparison;
    // this is the file a stage-2 evaluation is driven with.
    if let Some(path) = &options.refiner_latent {
        write_refiner_latent(&gpu, &latent, &geometry, path)?;
    }

    if failures != 0 {
        println!("GATE FAIL stages={failures}");
        return Ok(ExitCode::from(3));
    }
    println!("GATE PASS");
    Ok(ExitCode::SUCCESS)
}

fn parse(arguments: &[String]) -> Result<Options, String> {
    if arguments.len() < 3 {
        return Err(usage());
    }
    let mut options = Options {
        golden: PathBuf::from(&arguments[0]),
        upscaler: PathBuf::from(&arguments[1]),
        adapter: PathBuf::from(&arguments[2]),
        arm: Arm::from_env(),
        relative: 0.0,
        refiner_latent: None,
    };
    let mut index = 3;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--arm" => {
                index += 1;
                let value = arguments.get(index).ok_or_else(usage)?;
                options.arm = match value.as_str() {
                    "f32" | "fp32" => Arm::F32,
                    "bf16" => Arm::Bf16,
                    other => return Err(format!("unknown arm {other}")),
                };
            }
            "--relative" => {
                index += 1;
                let value = arguments.get(index).ok_or_else(usage)?;
                options.relative = value.parse().map_err(|_| format!("bad {value}"))?;
            }
            "--dump-latent" => {
                index += 1;
                let value = arguments.get(index).ok_or_else(usage)?;
                options.refiner_latent = Some(PathBuf::from(value));
            }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
        index += 1;
    }
    Ok(options)
}

fn usage() -> String {
    "usage: h3_ltx_adapter_gate <golden-dir> <upscaler.safetensors> <adapter-dir> \
     [--arm f32|bf16] [--relative N] [--dump-latent FILE]"
        .to_string()
}

/// The golden's tensors and the geometry its manifest pins.
fn read_golden(
    directory: &Path,
) -> Result<(std::collections::HashMap<String, Vec<f32>>, Geometry), String> {
    let manifest = directory.join("manifest.json");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("{}: {error}", manifest.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("manifest: {error}"))?;
    let number = |field: &str, key: &str| -> Result<usize, String> {
        value
            .get(field)
            .and_then(|entry| entry.get(key))
            .and_then(|entry| entry.as_u64())
            .map(|entry| entry as usize)
            .ok_or_else(|| format!("manifest has no {field}.{key}"))
    };
    let frames = number("geometry", "pixel_frames")?;
    let geometry_value = value
        .get("geometry")
        .and_then(|entry| entry.get("draft_canvas"))
        .and_then(|entry| entry.as_array())
        .ok_or_else(|| "manifest has no geometry.draft_canvas".to_string())?;
    let height = geometry_value
        .first()
        .and_then(|entry| entry.as_u64())
        .ok_or_else(|| "manifest draft_canvas has no height".to_string())?
        as usize;
    let width = geometry_value
        .get(1)
        .and_then(|entry| entry.as_u64())
        .ok_or_else(|| "manifest draft_canvas has no width".to_string())? as usize;
    let geometry = geometry(frames, height, width).map_err(|error| error.to_string())?;

    // `latents.safetensors` after the rename, `bridge.safetensors` for a
    // vector produced before it.
    let mut shard = None;
    for name in ["latents.safetensors", "bridge.safetensors"] {
        let path = directory.join(name);
        if path.exists() {
            shard = Some(safetensors::read_shard(&path).map_err(|error| format!("{error:?}"))?);
            break;
        }
    }
    let shard = shard.ok_or_else(|| format!("no golden tensor file in {}", directory.display()))?;

    let mut values = std::collections::HashMap::new();
    for tensor in &shard.tensors {
        let mut bytes = vec![0u8; tensor.bytes() as usize];
        shard
            .read_data(tensor, &mut bytes)
            .map_err(|error| format!("{error:?}"))?;
        if tensor.dtype != safetensors::Dtype::F32 {
            return Err(format!("{} is {:?}, not F32", tensor.name, tensor.dtype));
        }
        values.insert(
            tensor.name.clone(),
            bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect(),
        );
    }
    Ok((values, geometry))
}

fn upload(gpu: &Gpu, arm: Arm, values: &[f32]) -> Result<ds4_sys::h3_dit::Tensor, String> {
    let bytes: Vec<u8> = match arm {
        Arm::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Arm::Bf16 => values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                ((rounded >> 16) as u16).to_le_bytes()
            })
            .collect(),
    };
    let tensor = gpu.alloc(arm_dtype(arm), values.len())?;
    gpu.upload(&tensor, &bytes)?;
    Ok(tensor)
}

/// The port's own `refiner_input` (`runtime/stage2.py:76`, `latent[:, :, :16]`),
/// written as the raw BF16 file `h3_ltx_refiner_gate --latent` reads: one run
/// of 16 frames per channel, `[channels, rows]`, which is the `(1, 128, 16,
/// 24, 42)` grid `patchify_video` takes its rows from.
fn write_refiner_latent(
    gpu: &Gpu,
    latent: &ds4_sys::h3_dit::Tensor,
    geometry: &Geometry,
    path: &Path,
) -> Result<(), String> {
    if latent.dtype() != Dtype::Bf16 {
        return Err("--dump-latent is the bf16 arm's file; the f32 arm is not a latent".into());
    }

    let [channels, frames, height, width] = geometry.ltx;
    let plane = height * width;

    // The crop is per channel, not a prefix: `[:, :, :16]` keeps the first 16
    // of the 17 frames inside each channel's own run.
    let (stride, keep) = (frames * plane * 2, REFINER_FRAMES * plane * 2);
    let mut payload = vec![0u8; latent.bytes()];
    gpu.read(latent, &mut payload)?;

    let mut cropped = Vec::with_capacity(channels * keep);
    for channel in 0..channels {
        cropped.extend_from_slice(&payload[channel * stride..channel * stride + keep]);
    }
    std::fs::write(path, &cropped).map_err(|error| format!("{}: {error}", path.display()))?;

    println!(
        "REFINER-LATENT path={} channels={channels} frames={REFINER_FRAMES} plane={plane} \
         bytes={}",
        path.display(),
        cropped.len()
    );
    Ok(())
}

fn download(gpu: &Gpu, tensor: &ds4_sys::h3_dit::Tensor) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; tensor.elements() * tensor.dtype().size()];
    gpu.read(tensor, &mut bytes)?;
    Ok(match tensor.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect(),
        _ => bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
            .collect(),
    })
}

fn arm_dtype(arm: Arm) -> Dtype {
    match arm {
        Arm::F32 => Dtype::F32,
        Arm::Bf16 => Dtype::Bf16,
    }
}

/// `align_h3_to_ltx` (`geometry.py:81-112`) on the host: the temporal
/// resample beside the nearest pack, then the 2x pixel shuffle.
fn host_align(h3: &[f32], geometry: &Geometry, plan: &TemporalPlan) -> Vec<f32> {
    let [_, frames, height, width] = geometry.h3;
    let [_, ltx_frames, ltx_height, ltx_width] = geometry.ltx;
    let plane = height * width;
    let aligned = [
        H3_LATENT_CHANNELS * (1 + TEMPORAL_PACK_SLOTS),
        ltx_frames,
        height,
        width,
    ];

    // The linear half.
    let mut buffer = vec![0.0f32; aligned.iter().product::<usize>()];
    for channel in 0..H3_LATENT_CHANNELS {
        for target in 0..ltx_frames {
            let left = plan.linear_left[target] as usize;
            let right = plan.linear_right[target] as usize;
            let weight = plan.linear_weight[target];
            for point in 0..plane {
                let near = h3[(channel * frames + left) * plane + point];
                let far = h3[(channel * frames + right) * plane + point];
                buffer[(channel * ltx_frames + target) * plane + point] =
                    near + weight * (far - near);
            }
        }
    }

    // The packed half, at `slot * channels + channel`.
    for slot in 0..TEMPORAL_PACK_SLOTS {
        for target in 0..ltx_frames {
            let source = plan.pack[target * TEMPORAL_PACK_SLOTS + slot];
            for channel in 0..H3_LATENT_CHANNELS {
                let destination = H3_LATENT_CHANNELS + slot * H3_LATENT_CHANNELS + channel;
                for point in 0..plane {
                    buffer[(destination * ltx_frames + target) * plane + point] =
                        if source == u32::MAX {
                            0.0
                        } else {
                            h3[(channel * frames + source as usize) * plane + point]
                        };
                }
            }
        }
    }

    // The 2x pixel shuffle.
    let channels = aligned[0];
    let mut out = vec![0.0f32; channels * 4 * ltx_frames * ltx_height * ltx_width];
    for channel in 0..channels {
        for sub in 0..4usize {
            for frame in 0..ltx_frames {
                for y in 0..ltx_height {
                    for x in 0..ltx_width {
                        let y_source = 2 * y + sub / 2;
                        let x_source = 2 * x + sub % 2;
                        out[(((channel * 4 + sub) * ltx_frames + frame) * ltx_height + y)
                            * ltx_width
                            + x] = buffer
                            [(channel * ltx_frames + frame) * plane + y_source * width + x_source];
                    }
                }
            }
        }
    }
    out
}

/// Where a failing tensor is wrong: the relative RMS of each frame and of each
/// channel, worst first. A localized failure names the op that produced it.
fn profile(
    name: &str,
    produced: &[f32],
    expected: &[f32],
    channels: usize,
    frames: usize,
    plane: usize,
) {
    let mut worst_frame = (0usize, 0.0f64);
    for frame in 0..frames {
        let mut error = 0.0f64;
        let mut reference = 0.0f64;
        for channel in 0..channels {
            for point in 0..plane {
                let index = (channel * frames + frame) * plane + point;
                let difference = (produced[index] - expected[index]) as f64;
                error += difference * difference;
                reference += (expected[index] as f64) * (expected[index] as f64);
            }
        }
        let ratio = (error / reference.max(1e-30)).sqrt();
        if ratio > worst_frame.1 {
            worst_frame = (frame, ratio);
        }
        if frame < 3 || frame + 2 >= frames {
            println!("  {name} frame {frame} relative {ratio:.4}");
        }
    }
    let mut worst_channel = (0usize, 0.0f64);
    for channel in 0..channels {
        let mut error = 0.0f64;
        let mut reference = 0.0f64;
        for frame in 0..frames {
            for point in 0..plane {
                let index = (channel * frames + frame) * plane + point;
                let difference = (produced[index] - expected[index]) as f64;
                error += difference * difference;
                reference += (expected[index] as f64) * (expected[index] as f64);
            }
        }
        let ratio = (error / reference.max(1e-30)).sqrt();
        if ratio > worst_channel.1 {
            worst_channel = (channel, ratio);
        }
    }
    println!(
        "  {name} worst frame {} relative {:.4}, worst channel {} relative {:.4}",
        worst_frame.0, worst_frame.1, worst_channel.0, worst_channel.1
    );
}

/// Max absolute error, the RMS of the error against the reference's own RMS,
/// and how many elements sit beyond ten times the reported scale.
fn compare(stage: &str, produced: &[f32], expected: &[f32], relative: f64) -> Result<bool, String> {
    if produced.len() != expected.len() {
        return Err(format!(
            "{stage}: produced {} values, the reference has {}",
            produced.len(),
            expected.len()
        ));
    }
    let mut worst = 0.0f64;
    let mut error = 0.0f64;
    let mut reference = 0.0f64;
    let mut outliers = 0usize;
    for (left, right) in produced.iter().zip(expected) {
        let difference = (*left as f64 - *right as f64).abs();
        worst = worst.max(difference);
        error += difference * difference;
        reference += (*right as f64) * (*right as f64);
        if difference > 1e-2 * (right.abs() as f64).max(1.0) {
            outliers += 1;
        }
    }
    let rms = (error / produced.len() as f64).sqrt();
    let reference_rms = (reference / produced.len() as f64).sqrt();
    let ratio = rms / reference_rms.max(1e-30);
    let pass = ratio <= relative;
    println!(
        "{stage} values={} reference_rms={reference_rms:.6} rms={rms:.6e} relative={ratio:.3e} \
         max={worst:.6e} beyond_1e-2={outliers} {}",
        produced.len(),
        if pass { "ok" } else { "OVER" }
    );
    Ok(pass)
}
