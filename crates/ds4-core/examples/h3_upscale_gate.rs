//! Gate: the H3 latent upscaler against an independent CPU reference.
//!
//! The reference here is written from the released model's own definition
//! (`minimax_h3_latent_upscaler_3d.py` and `config.json`), in f64, over the
//! checkpoint's tensors read by name. It shares no code with the device path
//! beyond the constants the architecture fixes, so an indexing, padding,
//! normalization or block-order mistake shows up as a number rather than as a
//! soft frame.
//!
//! ```text
//! cargo run --release -p ds4-core --example h3_upscale_gate -- \
//!     --dir /mnt/h3-artifacts/models/latent_upscale
//! ```
//!
//! The latent is deterministic noise, small enough that the f64 reference runs
//! in seconds: 2 x 2 x 2 voxels in, 4 x 4 out.

use std::path::Path;
use std::path::PathBuf;

use ds4_core::minimax_h3::h3_dit::Gpu;
use ds4_core::minimax_h3::latent_upscale::{
    half_to_f32, temporal_at, Upscaler, BLOCKS, EPSILON, GROUPS, LATENTS_MEAN, LATENTS_STD,
    LATENT_CHANNELS, TEMPORAL_KERNEL, WIDTH,
};
use ds4_core::minimax_h3::vae_store::Store;

const TIME: usize = 2;
const HEIGHT: usize = 2;
const WIDTH_IN: usize = 2;
const HEIGHT_OUT: usize = 4;
const WIDTH_OUT: usize = 4;

/// The shape the transcription is asked about. The released resizer is not
/// shape-fixed, and the engine calls it at shapes this gate's 2 x 2 x 2 never
/// covered; `--shape` is how a shape gets checked instead of only timed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Dims {
    time: usize,
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
}

impl Dims {
    const RELEASED: Dims = Dims {
        time: TIME,
        height: HEIGHT,
        width: WIDTH_IN,
        out_height: HEIGHT_OUT,
        out_width: WIDTH_OUT,
    };
}

fn main() {
    let mut dir: Option<PathBuf> = None;
    let mut bench: Option<(usize, usize, usize)> = None;
    let mut dims = Dims::RELEASED;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--dir" => dir = argv.next().map(PathBuf::from),
            "--bench" => {
                let mut shape = [0usize; 3];
                for slot in shape.iter_mut() {
                    *slot = argv
                        .next()
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("--bench wants time height width");
                            std::process::exit(2);
                        });
                }
                bench = Some((shape[0], shape[1], shape[2]));
            }
            "--shape" => {
                let values: Vec<usize> = argv
                    .by_ref()
                    .take(5)
                    .filter_map(|value| value.parse().ok())
                    .collect();
                if values.len() != 5 {
                    eprintln!("--shape wants time height width out_height out_width");
                    std::process::exit(2);
                }
                dims = Dims {
                    time: values[0],
                    height: values[1],
                    width: values[2],
                    out_height: values[3],
                    out_width: values[4],
                };
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let dir = dir.unwrap_or_else(|| PathBuf::from("/mnt/h3-artifacts/models/latent_upscale"));
    let outcome = match bench {
        Some((time, height, width)) => bench_run(&dir, time, height, width),
        None => run(&dir, dims),
    };
    match outcome {
        Ok(()) => {}
        Err(detail) => {
            eprintln!("FAIL {detail}");
            std::process::exit(1);
        }
    }
}

/// Real-size timing: the same resizer, a latent of the given shape, doubled in
/// height and width. No reference here; this is the cost of the stage.
fn bench_run(dir: &Path, time: usize, height: usize, width: usize) -> Result<(), String> {
    let voxels = time * height * width;
    let latent: Vec<f32> = (0..LATENT_CHANNELS * voxels)
        .map(|index| {
            let channel = index / voxels;
            LATENTS_MEAN[channel] + ((index % 97) as f32 / 97.0 - 0.5) * LATENTS_STD[channel]
        })
        .collect();
    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;
    let started = std::time::Instant::now();
    // The bench reads the arm from the environment so a shell can A/B it; the
    // gate below pins F32 on the row-major orientation.
    let upscaler = Upscaler::open(dir, gpu).map_err(|detail| detail.to_string())?;
    let load = started.elapsed().as_secs_f64();

    let mut best = f64::MAX;
    let mut stages: Vec<(&'static str, f64)> = Vec::new();
    for attempt in 0..2 {
        // The last attempt is the one whose phases are reported: it runs on a
        // warm device, which is the product's case.
        let last = attempt == 1;
        if last {
            upscaler.timing(true);
            upscaler.timing_reset();
        }
        let started = std::time::Instant::now();
        let (output, laps) = if last {
            upscaler.upscale_timed(&latent, time, height, width, height * 2, width * 2)
        } else {
            upscaler
                .upscale(&latent, time, height, width, height * 2, width * 2)
                .map(|output| (output, Vec::new()))
        }
        .map_err(|detail| detail.to_string())?;
        let elapsed = started.elapsed().as_secs_f64();
        if last {
            upscaler.timing(false);
            stages = laps;
        }
        best = best.min(elapsed);
        let mean = output.values.iter().sum::<f32>() / output.values.len() as f32;
        println!(
            "run {}: {time} x {height} x {width} -> {} x {} x {} in {:.3}s (mean {mean:.4})",
            attempt + 1,
            output.time,
            output.height,
            output.width,
            elapsed
        );
    }
    let [columns, gemm, bias] = upscaler.timing_read();
    let parts = columns + gemm + bias;
    println!(
        "phases {parts:.3}s of the timed run: im2col {columns:.3}s, gemm {gemm:.3}s, \
         bias {bias:.3}s ({:.0}% im2col)",
        if parts > 0.0 {
            100.0 * columns / parts
        } else {
            0.0
        }
    );
    if !stages.is_empty() {
        let lapped: f64 = stages.iter().map(|(_, seconds)| seconds).sum();
        let mut line = String::new();
        for (name, seconds) in &stages {
            line.push_str(&format!("{name} {seconds:.3}s, "));
        }
        println!(
            "stages {lapped:.3}s of the timed run: {}",
            line.trim_end_matches(", ")
        );
    }
    println!(
        "load {load:.2}s, best {best:.3}s for {voxels} latent voxels, arm {}",
        upscaler.arm()
    );
    // The kernel each shape actually ran on. Without this a GEMM number cannot
    // be read: the same source dispatched a SIMT kernel in one build and a
    // tensor-core one in the next while the algorithm was left to the
    // heuristic.
    println!("algo {}", Gpu::lt_algo_report());
    Ok(())
}

fn fail<T>(detail: impl Into<String>) -> Result<T, String> {
    Err(detail.into())
}

/// The output's overall deviation over the input's: one for a graph that
/// returns a latent of the input's own scale, and far from one for a graph
/// that has lost it.
fn deviation_ratio(input: &[f32], output: &[f32]) -> f64 {
    let spread = |values: &[f32]| -> f64 {
        let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (*value as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt()
    };
    let denominator = spread(input);
    if denominator == 0.0 {
        return f64::INFINITY;
    }
    spread(output) / denominator
}

fn run(dir: &Path, d: Dims) -> Result<(), String> {
    let store = Store::open(dir, "latent upscaler")?;

    // Deterministic input, channel-major and already in the model's own value
    // range: the mean plus a small offset on the standard deviation.
    let voxels = d.time * d.height * d.width;
    let mut latent = vec![0.0f32; LATENT_CHANNELS * voxels];
    for channel in 0..LATENT_CHANNELS {
        for voxel in 0..voxels {
            let noise = ((channel * 31 + voxel * 17) % 23) as f32 / 23.0 - 0.5;
            latent[channel * voxels + voxel] = LATENTS_MEAN[channel] + noise * LATENTS_STD[channel];
        }
    }

    let reference = reference(&store, &latent, d)?;
    if reference.len() != LATENT_CHANNELS * d.time * d.out_height * d.out_width {
        return fail("reference length differs");
    }
    let scale = reference
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()));

    // Every arm the resizer can run, each against the same reference (name,
    // tf32, columns, half, bound). The two F32 arms are the arithmetic the port
    // was verified on; the TF32 arms are the reference implementation's own
    // class (it computes in F16, ten mantissa bits) and only reach a tensor-core
    // kernel on the column-major orientation; the F16 arm stores the operands
    // the way the checkpoint does. A bound per arm, so a faster arm has to say
    // what it costs.
    let arms: [(&str, bool, bool, bool, f64); 5] = [
        ("fp32 rows", false, false, false, 5e-4),
        ("fp32 columns", false, true, false, 5e-4),
        ("tf32 rows", true, false, false, 5e-4),
        ("tf32 columns", true, true, false, 5e-4),
        ("fp16 columns", true, true, true, 5e-4),
    ];
    let mut failed = false;
    for (name, tf32, columns, half, bound) in arms {
        // One context per arm: `Gpu` owns its stream and the resizer owns the
        // context, so an arm that shares one cannot hold it across the loop.
        let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;
        let upscaler = Upscaler::open_with(dir, gpu, tf32, columns, half)
            .map_err(|detail| detail.to_string())?;
        let started = std::time::Instant::now();
        let output = upscaler
            .upscale(
                &latent,
                d.time,
                d.height,
                d.width,
                d.out_height,
                d.out_width,
            )
            .map_err(|detail| detail.to_string())?;
        let elapsed = started.elapsed();
        if output.time != d.time || output.height != d.out_height || output.width != d.out_width {
            return fail(format!(
                "geometry {}x{}x{}, expected {}x{}x{}",
                output.time, output.height, output.width, d.time, d.out_height, d.out_width
            ));
        }
        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        for (index, (candidate, target)) in output.values.iter().zip(reference.iter()).enumerate() {
            let error = (*candidate as f64 - target).abs();
            if error > worst {
                worst = error;
                worst_at = index;
            }
        }
        let relative = if scale > 0.0 { worst / scale } else { worst };
        let verdict = if relative > bound { "FAIL" } else { "ok" };
        println!(
            "{name:<13} {:>8?}  max |device - reference| {worst:.3e}, relative {relative:.3e} \
             (worst channel {})  {verdict}",
            elapsed,
            worst_at / (d.time * d.out_height * d.out_width)
        );
        // Each arm names the algorithm it ran on, so a difference between two
        // arms is a difference in arithmetic or in kernel, never in doubt.
        println!("  algo {}", Gpu::lt_algo_report());

        // A check the transcription cannot make, because a misreading shared
        // by the transcription and the device would pass the comparison above.
        // The resizer normalizes per channel and denormalizes with the same
        // constants, so it returns a latent of the input's own scale: a
        // deviation ratio far from one is a graph that is not returning a
        // latent at all, however well it agrees with its own transcription.
        //
        // Measured: 3.0 on this gate's own near-constant input (detail
        // synthesis dominates a latent with little structure), 1.1 on a real
        // 22 x 22 draft, and 8 to 20 when the blocks' modulation was wrong.
        let spread = deviation_ratio(&latent, &output.values);
        let plausible = (0.25..=6.0).contains(&spread);
        println!(
            "  scale output/input deviation {spread:.3}  {}",
            if plausible {
                "ok"
            } else {
                "FAIL: not a latent's own scale"
            }
        );

        if relative > bound || !plausible {
            failed = true;
        }
    }
    if failed {
        return fail("an arm is outside its bound");
    }
    println!("PASS {voxels} voxels, {} arms", arms.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// The reference: f64, channels-last, from the model definition.
// ---------------------------------------------------------------------------

type Volume = Vec<f64>;

fn shape(time: usize, height: usize, width: usize, channels: usize) -> usize {
    time * height * width * channels
}

fn load(store: &Store, name: &str) -> Result<Volume, String> {
    let shape = store
        .shape(name)
        .ok_or_else(|| format!("checkpoint is missing {name}"))?;
    let bytes = store.raw(name, &shape)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| half_to_f32(u16::from_le_bytes(*word)) as f64)
        .collect())
}

fn conv(
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    time: usize,
    height: usize,
    width: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
) -> Volume {
    let pad = kernel / 2;
    let mut output = vec![0.0f64; shape(time, height, width, output_channels)];
    for t in 0..time {
        for y in 0..height {
            for x in 0..width {
                for out in 0..output_channels {
                    let mut sum = bias[out];
                    for inside in 0..input_channels {
                        for kt in 0..kernel {
                            for ky in 0..kernel {
                                for kx in 0..kernel {
                                    let it = t as isize + kt as isize - pad as isize;
                                    let iy = y as isize + ky as isize - pad as isize;
                                    let ix = x as isize + kx as isize - pad as isize;
                                    if it < 0
                                        || iy < 0
                                        || ix < 0
                                        || it >= time as isize
                                        || iy >= height as isize
                                        || ix >= width as isize
                                    {
                                        continue;
                                    }
                                    let source = (((it as usize * height + iy as usize) * width
                                        + ix as usize)
                                        * input_channels)
                                        + inside;
                                    let index = ((((out * input_channels + inside) * kernel + kt)
                                        * kernel
                                        + ky)
                                        * kernel)
                                        + kx;
                                    sum += input[source] * weight[index];
                                }
                            }
                        }
                    }
                    output[((t * height + y) * width + x) * output_channels + out] = sum;
                }
            }
        }
    }
    output
}

fn group_norm(
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    time: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Volume {
    let per_group = channels / GROUPS as usize;
    let voxels = time * height * width;
    let mut output = vec![0.0f64; input.len()];
    for group in 0..GROUPS as usize {
        let first = group * per_group;
        let mut sum = 0.0f64;
        let mut squares = 0.0f64;
        for voxel in 0..voxels {
            for channel in 0..per_group {
                let value = input[voxel * channels + first + channel];
                sum += value;
                squares += value * value;
            }
        }
        let count = (voxels * per_group) as f64;
        let mean = sum / count;
        let variance = squares / count - mean * mean;
        let inverse = 1.0 / (variance + EPSILON as f64).sqrt();
        for voxel in 0..voxels {
            for channel in 0..per_group {
                let index = voxel * channels + first + channel;
                output[index] = (input[index] - mean) * inverse * weight[first + channel]
                    + bias[first + channel];
            }
        }
    }
    output
}

fn silu(input: &[f64]) -> Volume {
    input
        .iter()
        .map(|value| value / (1.0 + (-*value).exp()))
        .collect()
}

fn affine(input: &[f64], scale: &[f64], shift: &[f64], channels: usize) -> Volume {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| value * scale[index % channels] + shift[index % channels])
        .collect()
}

fn temporal_conv(
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    time: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Volume {
    let pad = TEMPORAL_KERNEL as isize / 2;
    let mut output = vec![0.0f64; input.len()];
    for t in 0..time {
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let mut sum = bias[channel];
                    for tap in 0..TEMPORAL_KERNEL as isize {
                        let it = t as isize + tap - pad;
                        if it < 0 || it >= time as isize {
                            continue;
                        }
                        let source =
                            (((it as usize * height + y) * width + x) * channels) + channel;
                        sum += input[source]
                            * weight[channel * TEMPORAL_KERNEL as usize + tap as usize];
                    }
                    output[((t * height + y) * width + x) * channels + channel] = sum;
                }
            }
        }
    }
    output
}

fn linear(
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Volume {
    let mut output = vec![0.0f64; rows * output_dim];
    for row in 0..rows {
        for out in 0..output_dim {
            let mut sum = bias[out];
            for inside in 0..input_dim {
                sum += input[row * input_dim + inside] * weight[out * input_dim + inside];
            }
            output[row * output_dim + out] = sum;
        }
    }
    output
}

/// `F.interpolate(mode="trilinear", align_corners=False)`, height and width
/// only: `(out + 0.5) * in / out - 0.5`, clamped at zero.
fn resize(
    input: &[f64],
    time: usize,
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
    channels: usize,
) -> Volume {
    let mut output = vec![0.0f64; shape(time, out_height, out_width, channels)];
    for t in 0..time {
        for oy in 0..out_height {
            let sy = ((oy as f64 + 0.5) * (height as f64 / out_height as f64) - 0.5).max(0.0);
            let y0 = (sy.floor() as usize).min(height - 1);
            let ly = sy - y0 as f64;
            let y1 = (y0 + 1).min(height - 1);
            for ox in 0..out_width {
                let sx = ((ox as f64 + 0.5) * (width as f64 / out_width as f64) - 0.5).max(0.0);
                let x0 = (sx.floor() as usize).min(width - 1);
                let lx = sx - x0 as f64;
                let x1 = (x0 + 1).min(width - 1);
                for channel in 0..channels {
                    let at = |y: usize, x: usize| {
                        input[(((t * height + y) * width + x) * channels) + channel]
                    };
                    let top = at(y0, x0) * (1.0 - lx) + at(y0, x1) * lx;
                    let bottom = at(y1, x0) * (1.0 - lx) + at(y1, x1) * lx;
                    output[(((t * out_height + oy) * out_width + ox) * channels) + channel] =
                        top * (1.0 - ly) + bottom * ly;
                }
            }
        }
    }
    output
}

fn reference(store: &Store, latent: &[f32], d: Dims) -> Result<Volume, String> {
    let width = WIDTH as usize;
    let embed_in_w = load(store, "embed.0.weight")?;
    let embed_in_b = load(store, "embed.0.bias")?;
    let embed_out_w = load(store, "embed.2.weight")?;
    let embed_out_b = load(store, "embed.2.bias")?;

    // Normalize on the host, exactly as the node does, then go time-major.
    let voxels = d.time * d.height * d.width;
    let mut current = vec![0.0f64; latent.len()];
    for channel in 0..LATENT_CHANNELS {
        for voxel in 0..voxels {
            current[voxel * LATENT_CHANNELS + channel] = (latent[channel * voxels + voxel] as f64
                - LATENTS_MEAN[channel] as f64)
                / LATENTS_STD[channel] as f64;
        }
    }

    let scale = (d.out_width as f64 / d.width as f64 + d.out_height as f64 / d.height as f64) / 2.0;
    let hidden = linear(&[scale - 1.0], &embed_in_w, &embed_in_b, 1, 1, 64);
    let hidden = silu(&hidden);
    let embed = linear(&hidden, &embed_out_w, &embed_out_b, 1, 64, 64);

    current = conv(
        &current,
        &load(store, "conv_in.weight")?,
        &load(store, "conv_in.bias")?,
        d.time,
        d.height,
        d.width,
        LATENT_CHANNELS,
        width,
        3,
    );

    let mut run_blocks =
        |side: &str, input: Volume, height: usize, width_in: usize| -> Result<Volume, String> {
            let mut volume = input;
            for index in 0..BLOCKS {
                let prefix = format!("{side}.{index}");
                if temporal_at(index) {
                    let norm = group_norm(
                        &volume,
                        &load(store, &format!("{prefix}.norm.weight"))?,
                        &load(store, &format!("{prefix}.norm.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                    );
                    let activated = silu(&norm);
                    let depthwise = temporal_conv(
                        &activated,
                        &load(store, &format!("{prefix}.dwconv.weight"))?,
                        &load(store, &format!("{prefix}.dwconv.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                    );
                    let pointwise = linear(
                        &depthwise,
                        &load(store, &format!("{prefix}.pwconv.weight"))?,
                        &load(store, &format!("{prefix}.pwconv.bias"))?,
                        d.time * height * width_in,
                        width,
                        width,
                    );
                    volume = volume
                        .iter()
                        .zip(pointwise.iter())
                        .map(|(left, right)| left + right)
                        .collect();
                } else {
                    let normalized = group_norm(
                        &volume,
                        &load(store, &format!("{prefix}.in_layers.0.weight"))?,
                        &load(store, &format!("{prefix}.in_layers.0.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                    );
                    let activated = silu(&normalized);
                    let convolved = conv(
                        &activated,
                        &load(store, &format!("{prefix}.in_layers.2.weight"))?,
                        &load(store, &format!("{prefix}.in_layers.2.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                        width,
                        3,
                    );
                    let normalized = group_norm(
                        &convolved,
                        &load(store, &format!("{prefix}.out_norm.weight"))?,
                        &load(store, &format!("{prefix}.out_norm.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                    );
                    // `emb_layers` is `SiLU()` then `Linear` and the SiLU is
                    // the block's own (`:168-170`): the shared `embed` is not
                    // activated for it.
                    let projected = linear(
                        &silu(&embed),
                        &load(store, &format!("{prefix}.emb_layers.1.weight"))?,
                        &load(store, &format!("{prefix}.emb_layers.1.bias"))?,
                        1,
                        64,
                        2 * width,
                    );
                    let modulation_scale: Vec<f64> =
                        projected[..width].iter().map(|value| 1.0 + value).collect();
                    let modulation_shift: Vec<f64> = projected[width..].to_vec();
                    let modulated =
                        affine(&normalized, &modulation_scale, &modulation_shift, width);
                    let activated = silu(&modulated);
                    let residual = conv(
                        &activated,
                        &load(store, &format!("{prefix}.out_layers.2.weight"))?,
                        &load(store, &format!("{prefix}.out_layers.2.bias"))?,
                        d.time,
                        height,
                        width_in,
                        width,
                        width,
                        3,
                    );
                    volume = volume
                        .iter()
                        .zip(residual.iter())
                        .map(|(left, right)| left + right)
                        .collect();
                }
            }
            Ok(volume)
        };

    current = run_blocks("in_blocks", current, d.height, d.width)?;
    current = resize(
        &current,
        d.time,
        d.height,
        d.width,
        d.out_height,
        d.out_width,
        width,
    );
    current = run_blocks("out_blocks", current, d.out_height, d.out_width)?;

    let normalized = group_norm(
        &current,
        &load(store, "norm_out.weight")?,
        &load(store, "norm_out.bias")?,
        d.time,
        d.out_height,
        d.out_width,
        width,
    );
    let activated = silu(&normalized);
    let output = conv(
        &activated,
        &load(store, "conv_out.weight")?,
        &load(store, "conv_out.bias")?,
        d.time,
        d.out_height,
        d.out_width,
        width,
        LATENT_CHANNELS,
        3,
    );

    // Back to channel-major and denormalized.
    let out_voxels = d.time * d.out_height * d.out_width;
    let mut values = vec![0.0f64; output.len()];
    for channel in 0..LATENT_CHANNELS {
        for voxel in 0..out_voxels {
            values[channel * out_voxels + voxel] = output[voxel * LATENT_CHANNELS + channel]
                * LATENTS_STD[channel] as f64
                + LATENTS_MEAN[channel] as f64;
        }
    }
    Ok(values)
}
