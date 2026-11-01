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

fn main() {
    let mut dir: Option<PathBuf> = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--dir" => dir = argv.next().map(PathBuf::from),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let dir = dir.unwrap_or_else(|| PathBuf::from("/mnt/h3-artifacts/models/latent_upscale"));
    match run(&dir) {
        Ok(()) => {}
        Err(detail) => {
            eprintln!("FAIL {detail}");
            std::process::exit(1);
        }
    }
}

fn fail<T>(detail: impl Into<String>) -> Result<T, String> {
    Err(detail.into())
}

fn run(dir: &Path) -> Result<(), String> {
    let store = Store::open(dir, "latent upscaler")?;

    // Deterministic input, channel-major and already in the model's own value
    // range: the mean plus a small offset on the standard deviation.
    let voxels = TIME * HEIGHT * WIDTH_IN;
    let mut latent = vec![0.0f32; LATENT_CHANNELS * voxels];
    for channel in 0..LATENT_CHANNELS {
        for voxel in 0..voxels {
            let noise = ((channel * 31 + voxel * 17) % 23) as f32 / 23.0 - 0.5;
            latent[channel * voxels + voxel] = LATENTS_MEAN[channel] + noise * LATENTS_STD[channel];
        }
    }

    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;
    let upscaler = Upscaler::open(dir, gpu).map_err(|detail| detail.to_string())?;
    let started = std::time::Instant::now();
    let output = upscaler
        .upscale(&latent, TIME, HEIGHT, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT)
        .map_err(|detail| detail.to_string())?;
    let elapsed = started.elapsed();
    if output.time != TIME || output.height != HEIGHT_OUT || output.width != WIDTH_OUT {
        return fail(format!(
            "geometry {}x{}x{}, expected {}x{}x{}",
            output.time, output.height, output.width, TIME, HEIGHT_OUT, WIDTH_OUT
        ));
    }

    let reference = reference(&store, &latent)?;
    if reference.len() != output.values.len() {
        return fail("reference length differs");
    }
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    let mut scale = 0.0f64;
    for (index, (candidate, target)) in output.values.iter().zip(reference.iter()).enumerate() {
        scale = scale.max(target.abs());
        let error = (*candidate as f64 - target).abs();
        if error > worst {
            worst = error;
            worst_at = index;
        }
    }
    let relative = if scale > 0.0 { worst / scale } else { worst };
    println!(
        "upscale {:?} for {voxels} voxels: max |device - reference| {worst:.3e}, \
         relative {relative:.3e} (scale {scale:.3e}, worst channel {})",
        elapsed,
        worst_at / (TIME * HEIGHT_OUT * WIDTH_OUT)
    );
    // f32 device arithmetic against an f64 reference over twenty-four
    // convolutions: a correct port lands far inside this, and any indexing or
    // normalization mistake lands orders of magnitude outside it.
    if relative > 5e-4 {
        return fail(format!("relative error {relative:.3e} above 5e-4"));
    }
    println!("PASS");
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

fn reference(store: &Store, latent: &[f32]) -> Result<Volume, String> {
    let width = WIDTH as usize;
    let embed_in_w = load(store, "embed.0.weight")?;
    let embed_in_b = load(store, "embed.0.bias")?;
    let embed_out_w = load(store, "embed.2.weight")?;
    let embed_out_b = load(store, "embed.2.bias")?;

    // Normalize on the host, exactly as the node does, then go time-major.
    let voxels = TIME * HEIGHT * WIDTH_IN;
    let mut current = vec![0.0f64; latent.len()];
    for channel in 0..LATENT_CHANNELS {
        for voxel in 0..voxels {
            current[voxel * LATENT_CHANNELS + channel] = (latent[channel * voxels + voxel] as f64
                - LATENTS_MEAN[channel] as f64)
                / LATENTS_STD[channel] as f64;
        }
    }

    let scale = (WIDTH_OUT as f64 / WIDTH_IN as f64 + HEIGHT_OUT as f64 / HEIGHT as f64) / 2.0;
    let hidden = linear(&[scale - 1.0], &embed_in_w, &embed_in_b, 1, 1, 64);
    let hidden = silu(&hidden);
    let embed = linear(&hidden, &embed_out_w, &embed_out_b, 1, 64, 64);

    current = conv(
        &current,
        &load(store, "conv_in.weight")?,
        &load(store, "conv_in.bias")?,
        TIME,
        HEIGHT,
        WIDTH_IN,
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
                        TIME,
                        height,
                        width_in,
                        width,
                    );
                    let activated = silu(&norm);
                    let depthwise = temporal_conv(
                        &activated,
                        &load(store, &format!("{prefix}.dwconv.weight"))?,
                        &load(store, &format!("{prefix}.dwconv.bias"))?,
                        TIME,
                        height,
                        width_in,
                        width,
                    );
                    let pointwise = linear(
                        &depthwise,
                        &load(store, &format!("{prefix}.pwconv.weight"))?,
                        &load(store, &format!("{prefix}.pwconv.bias"))?,
                        TIME * height * width_in,
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
                        TIME,
                        height,
                        width_in,
                        width,
                    );
                    let activated = silu(&normalized);
                    let convolved = conv(
                        &activated,
                        &load(store, &format!("{prefix}.in_layers.2.weight"))?,
                        &load(store, &format!("{prefix}.in_layers.2.bias"))?,
                        TIME,
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
                        TIME,
                        height,
                        width_in,
                        width,
                    );
                    let projected = linear(
                        &embed,
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
                        TIME,
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

    current = run_blocks("in_blocks", current, HEIGHT, WIDTH_IN)?;
    current = resize(
        &current, TIME, HEIGHT, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT, width,
    );
    current = run_blocks("out_blocks", current, HEIGHT_OUT, WIDTH_OUT)?;

    let normalized = group_norm(
        &current,
        &load(store, "norm_out.weight")?,
        &load(store, "norm_out.bias")?,
        TIME,
        HEIGHT_OUT,
        WIDTH_OUT,
        width,
    );
    let activated = silu(&normalized);
    let output = conv(
        &activated,
        &load(store, "conv_out.weight")?,
        &load(store, "conv_out.bias")?,
        TIME,
        HEIGHT_OUT,
        WIDTH_OUT,
        width,
        LATENT_CHANNELS,
        3,
    );

    // Back to channel-major and denormalized.
    let out_voxels = TIME * HEIGHT_OUT * WIDTH_OUT;
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
