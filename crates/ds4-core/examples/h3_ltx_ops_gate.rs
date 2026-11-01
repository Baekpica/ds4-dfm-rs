//! The latent path's primitives, each against a host implementation of the
//! same arithmetic.
//!
//! Usage: h3_ltx_ops_gate
//!
//! The end-to-end gate (`h3_ltx_adapter_gate`) says only that something is
//! wrong; this one says which op. Every case below builds a small tensor whose
//! dimensions all differ, uploads it, runs the device op, and compares against
//! the same computation written out on the host in the layout the reference
//! uses. Shapes are deliberately not square and not powers of two, so a
//! transposed or mis-strided index cannot pass by symmetry.
//!
//! Exit 3 on any mismatch.

use std::process::ExitCode;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

const TOLERANCE: f32 = 1e-5;

/// A deterministic sequence, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn new() -> Rng {
        Rng(0x2545f4914f6cdd1d)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8388608.0) - 1.0
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let gpu = Gpu::create()?;
    let mut rng = Rng::new();
    let mut failures = 0usize;

    if !pad3d(&gpu, &mut rng)? {
        failures += 1;
    }
    if !im2col(&gpu, &mut rng)? {
        failures += 1;
    }
    if !planar(&gpu, &mut rng)? {
        failures += 1;
    }
    if !pack(&gpu, &mut rng)? {
        failures += 1;
    }
    if !matmul_planar(&gpu, &mut rng)? {
        failures += 1;
    }
    if !matmul(&gpu, &mut rng)? {
        failures += 1;
    }
    if !matmul_channels(&gpu, &mut rng)? {
        failures += 1;
    }
    if !epilogue(&gpu, &mut rng)? {
        failures += 1;
    }
    if !norm(&gpu, &mut rng)? {
        failures += 1;
    }
    if !silu_add(&gpu, &mut rng)? {
        failures += 1;
    }
    if !depthwise(&gpu, &mut rng)? {
        failures += 1;
    }
    if !resize(&gpu, &mut rng)? {
        failures += 1;
    }
    if !time_ops(&gpu, &mut rng)? {
        failures += 1;
    }
    if !unshuffle(&gpu, &mut rng)? {
        failures += 1;
    }
    if !accumulate(&gpu, &mut rng)? {
        failures += 1;
    }
    if !gate_mul(&gpu, &mut rng)? {
        failures += 1;
    }
    if !scale(&gpu, &mut rng)? {
        failures += 1;
    }

    if failures != 0 {
        println!("OPS FAIL failures={failures}");
        return Ok(ExitCode::from(3));
    }
    println!("OPS PASS");
    Ok(ExitCode::SUCCESS)
}

fn upload(gpu: &Gpu, dtype: Dtype, values: &[f32]) -> Result<Tensor, String> {
    let bytes: Vec<u8> = match dtype {
        Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
    };
    let tensor = gpu.alloc(dtype, values.len())?;
    gpu.upload(&tensor, &bytes)?;
    Ok(tensor)
}

fn upload_u32(gpu: &Gpu, values: &[u32]) -> Result<Tensor, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = gpu.alloc(Dtype::U32, values.len())?;
    gpu.upload(&tensor, &bytes)?;
    Ok(tensor)
}

fn download(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
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

fn check(name: &str, produced: &[f32], expected: &[f32], tolerance: f32) -> bool {
    if produced.len() != expected.len() {
        println!(
            "{name} FAIL length {} vs {}",
            produced.len(),
            expected.len()
        );
        return false;
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (index, (left, right)) in produced.iter().zip(expected).enumerate() {
        let difference = (left - right).abs();
        if difference > worst {
            worst = difference;
            at = index;
        }
    }
    let pass = worst <= tolerance;
    println!(
        "{name} values={} max={worst:.3e} at={at} {}",
        produced.len(),
        if pass { "ok" } else { "FAIL" }
    );
    if !pass {
        let low = at.saturating_sub(2);
        let high = (at + 3).min(produced.len());
        println!("  device {:?}", &produced[low..high]);
        println!("  host   {:?}", &expected[low..high]);
    }
    pass
}

fn pad3d(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, depth, height, width) = (2usize, 3usize, 4usize, 5usize);
    let pad = (1usize, 2usize, 1usize);
    let input = rng.values(channels * depth * height * width);
    let tensor = upload(gpu, Dtype::F32, &input)?;
    let (out_t, out_h, out_w) = (depth + 2 * pad.0, height + 2 * pad.1, width + 2 * pad.2);
    let out = gpu.alloc(Dtype::F32, channels * out_t * out_h * out_w)?;
    gpu.latent_pad3d(
        &out,
        &tensor,
        Dtype::F32,
        channels as u32,
        depth as u32,
        height as u32,
        width as u32,
        pad.0 as u32,
        pad.1 as u32,
        pad.2 as u32,
    )?;

    let mut expected = vec![0.0f32; channels * out_t * out_h * out_w];
    for c in 0..channels {
        for t in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let source = ((c * depth + t) * height + y) * width + x;
                    let destination =
                        ((c * out_t + t + pad.0) * out_h + y + pad.1) * out_w + x + pad.2;
                    expected[destination] = input[source];
                }
            }
        }
    }
    Ok(check("pad3d", &download(gpu, &out)?, &expected, 0.0))
}

fn im2col(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, depth, height, width) = (3usize, 4usize, 5usize, 3usize);
    let kernel = [3usize, 3, 3];
    let pad = [1usize, 1, 1];
    let padded = rng.values(channels * depth * height * width);
    let tensor = upload(gpu, Dtype::F32, &padded)?;

    let (out_t, out_h, out_w) = (depth - 2 * pad[0], height - 2 * pad[1], width - 2 * pad[2]);
    let columns = channels * kernel[0] * kernel[1] * kernel[2];
    let out = gpu.alloc(Dtype::F32, out_t * out_h * out_w * columns)?;
    gpu.latent_im2col(
        &out,
        &tensor,
        Dtype::F32,
        channels as u32,
        [depth as u32, height as u32, width as u32],
        [kernel[0] as u32, kernel[1] as u32, kernel[2] as u32],
        [out_t as u32, out_h as u32, out_w as u32],
        0,
    )?;

    let mut expected = vec![0.0f32; out_t * out_h * out_w * columns];
    for t in 0..out_t {
        for y in 0..out_h {
            for x in 0..out_w {
                let row = (t * out_h + y) * out_w + x;
                for c in 0..channels {
                    for kt in 0..kernel[0] {
                        for ky in 0..kernel[1] {
                            for kx in 0..kernel[2] {
                                let column =
                                    ((c * kernel[0] + kt) * kernel[1] + ky) * kernel[2] + kx;
                                let source =
                                    ((c * depth + t + kt) * height + y + ky) * width + x + kx;
                                expected[row * columns + column] = padded[source];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(check("im2col", &download(gpu, &out)?, &expected, 0.0))
}

/// The planar buffer: the padded activation in channel-last form, once per
/// column tap, which is the operand the convolution's dense matmuls read.
fn planar(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, depth, height, width) = (3usize, 3usize, 4usize, 5usize);
    let kw = 3usize;
    let pad = [1usize, 1, 1];
    let input = rng.values(channels * depth * height * width);
    let tensor = upload(gpu, Dtype::F32, &input)?;

    let (pad_depth, pad_height) = (depth + 2 * pad[0], height + 2 * pad[1]);
    let elements = channels * pad_depth * pad_height * width * kw;
    let out = gpu.alloc(Dtype::F32, elements)?;
    gpu.latent_planar(
        &out,
        &tensor,
        Dtype::F32,
        channels as u32,
        depth as u32,
        height as u32,
        width as u32,
        kw as u32,
        pad[0] as u32,
        pad[1] as u32,
        pad[2] as u32,
        width as u32,
        false,
    )?;

    let mut expected = vec![0.0f32; elements];
    for t in 0..pad_depth {
        for y in 0..pad_height {
            for x in 0..width {
                for s in 0..kw {
                    for c in 0..channels {
                        let st = t as isize - pad[0] as isize;
                        let sy = y as isize - pad[1] as isize;
                        let sx = (x + s) as isize - pad[2] as isize;
                        let inside = st >= 0
                            && (st as usize) < depth
                            && sy >= 0
                            && (sy as usize) < height
                            && sx >= 0
                            && (sx as usize) < width;
                        let value = if inside {
                            input[((c * depth + st as usize) * height + sy as usize) * width
                                + sx as usize]
                        } else {
                            0.0
                        };
                        let index =
                            ((t * pad_height + y) * width + x) * kw * channels + s * channels + c;
                        expected[index] = value;
                    }
                }
            }
        }
    }
    Ok(check("planar", &download(gpu, &out)?, &expected, 0.0))
}

/// The weights in the pass order the planar matmul reads: `[kt*kh + ky][out]
/// [s*in + c]`, a permuted copy of `[out][in][kd][kh][kw]`.
fn pack(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (out_channels, in_channels) = (2usize, 3usize);
    let kernel = [2usize, 3, 4];
    let taps = kernel[0] * kernel[1] * kernel[2];
    let passes = kernel[0] * kernel[1];
    let weight = rng.values(out_channels * in_channels * taps);
    let tensor = upload(gpu, Dtype::F32, &weight)?;
    let out = gpu.alloc(Dtype::F32, weight.len())?;
    gpu.latent_weight_pack(
        &out,
        &tensor,
        Dtype::F32,
        out_channels as u32,
        in_channels as u32,
        taps as u32,
        kernel[2] as u32,
    )?;

    let mut expected = vec![0.0f32; weight.len()];
    for pass in 0..passes {
        for o in 0..out_channels {
            for s in 0..kernel[2] {
                for c in 0..in_channels {
                    let index = ((pass * out_channels + o) * kernel[2] + s) * in_channels + c;
                    expected[index] = weight[(o * in_channels + c) * taps + pass * kernel[2] + s];
                }
            }
        }
    }
    Ok(check("pack", &download(gpu, &out)?, &expected, 0.0))
}

/// One tap as a dense matmul over a window of the planar buffer, twice, the
/// second time accumulating — which is how `kd*kh` of them sum into one
/// convolution.
fn matmul_planar(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, depth, height, width) = (3usize, 3usize, 4usize, 5usize);
    let (kw, out_channels) = (3usize, 2usize);
    let pad = [1usize, 1, 1];
    let input = rng.values(channels * depth * height * width);
    let tensor = upload(gpu, Dtype::F32, &input)?;

    let (pad_depth, pad_height) = (depth + 2 * pad[0], height + 2 * pad[1]);
    let planar = gpu.alloc(Dtype::F32, channels * pad_depth * pad_height * width * kw)?;
    gpu.latent_planar(
        &planar,
        &tensor,
        Dtype::F32,
        channels as u32,
        depth as u32,
        height as u32,
        width as u32,
        kw as u32,
        pad[0] as u32,
        pad[1] as u32,
        pad[2] as u32,
        width as u32,
        false,
    )?;

    let columns = kw * channels;
    let out_plane = height * width;
    let (tap_t, tap_h) = (1usize, 1usize);
    let offset = ((tap_t * pad_height) + tap_h) * width * columns;
    let first = rng.values(out_channels * columns);
    let second = rng.values(out_channels * columns);
    let first_tensor = upload(gpu, Dtype::F32, &first)?;
    let second_tensor = upload(gpu, Dtype::F32, &second)?;
    let out = gpu.alloc(Dtype::F32, out_plane * out_channels)?;

    // The device A window, read back, so the host sum uses the same operand:
    // the planar case above is what checks the window itself.
    let window = download(gpu, &planar)?[offset..offset + out_plane * columns].to_vec();
    let sum = |weights: &[f32]| -> Vec<f32> {
        let mut values = vec![0.0f32; out_plane * out_channels];
        for row in 0..out_plane {
            for o in 0..out_channels {
                let mut accumulator = 0.0f32;
                for k in 0..columns {
                    accumulator += window[row * columns + k] * weights[o * columns + k];
                }
                values[row * out_channels + o] = accumulator;
            }
        }
        values
    };

    gpu.latent_matmul_planar(
        &out,
        &planar,
        offset,
        &first_tensor,
        0,
        0,
        out_plane as u32,
        columns as u32,
        out_channels as u32,
        false,
        Dtype::F32,
    )?;
    let mut pass = check("matmul_planar", &download(gpu, &out)?, &sum(&first), 1e-5);

    gpu.latent_matmul_planar(
        &out,
        &planar,
        offset,
        &second_tensor,
        0,
        0,
        out_plane as u32,
        columns as u32,
        out_channels as u32,
        true,
        Dtype::F32,
    )?;
    let both: Vec<f32> = sum(&first)
        .iter()
        .zip(sum(&second))
        .map(|(left, right)| left + right)
        .collect();
    pass &= check(
        "matmul_planar_accumulate",
        &download(gpu, &out)?,
        &both,
        1e-5,
    );
    Ok(pass)
}

fn matmul(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (rows, input_dim, output_dim) = (5usize, 3usize, 7usize);
    let left = rng.values(rows * input_dim);
    let right = rng.values(output_dim * input_dim);
    let left_tensor = upload(gpu, Dtype::F32, &left)?;
    let right_tensor = upload(gpu, Dtype::F32, &right)?;
    let out = gpu.alloc(Dtype::F32, rows * output_dim)?;
    gpu.latent_matmul(
        &out,
        &left_tensor,
        &right_tensor,
        rows as u32,
        input_dim as u32,
        output_dim as u32,
        Dtype::F32,
    )?;

    let mut expected = vec![0.0f32; rows * output_dim];
    for row in 0..rows {
        for column in 0..output_dim {
            let mut sum = 0.0f32;
            for k in 0..input_dim {
                sum += left[row * input_dim + k] * right[column * input_dim + k];
            }
            expected[row * output_dim + column] = sum;
        }
    }
    Ok(check("matmul", &download(gpu, &out)?, &expected, 1e-5))
}

fn matmul_channels(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (spatial, input_dim, output_dim) = (6usize, 3usize, 4usize);
    let input = rng.values(input_dim * spatial);
    let weight = rng.values(output_dim * input_dim);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let weight_tensor = upload(gpu, Dtype::F32, &weight)?;
    let out = gpu.alloc(Dtype::F32, spatial * output_dim)?;
    gpu.latent_matmul_channels(
        &out,
        &input_tensor,
        &weight_tensor,
        spatial as u32,
        input_dim as u32,
        output_dim as u32,
        Dtype::F32,
    )?;

    let mut expected = vec![0.0f32; spatial * output_dim];
    for point in 0..spatial {
        for column in 0..output_dim {
            let mut sum = 0.0f32;
            for k in 0..input_dim {
                sum += input[k * spatial + point] * weight[column * input_dim + k];
            }
            expected[point * output_dim + column] = sum;
        }
    }
    Ok(check(
        "matmul_channels",
        &download(gpu, &out)?,
        &expected,
        1e-5,
    ))
}

fn epilogue(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, area) = (7usize, 3usize, 5usize);
    let plane = frames * area;
    // The matrix is written as if only the middle frame had been computed, at
    // an offset inside a larger destination: both the stride and the offset
    // have to be right for this to pass.
    let rows = 1usize * area;
    let matrix = rng.values(rows * channels);
    let bias = rng.values(channels);
    let matrix_tensor = upload(gpu, Dtype::F32, &matrix)?;
    let bias_tensor = upload(gpu, Dtype::F32, &bias)?;
    let out = upload(gpu, Dtype::F32, &vec![0.0f32; channels * plane])?;
    gpu.latent_epilogue(
        &out,
        &matrix_tensor,
        Some(&bias_tensor),
        Dtype::F32,
        channels as u32,
        rows as u32,
        plane,
        area,
    )?;

    let produced = download(gpu, &out)?;
    for channel in 0..channels {
        for point in 0..area {
            let expected = matrix[point * channels + channel] + bias[channel];
            let got = produced[channel * plane + area + point];
            if (got - expected).abs() > 0.0 {
                println!("epilogue FAIL at channel {channel} point {point}: {got} vs {expected}");
                return Ok(false);
            }
            // Everything outside the written frame must be untouched.
            if produced[channel * plane + point] != 0.0
                || produced[channel * plane + 2 * area + point] != 0.0
            {
                println!("epilogue FAIL wrote outside its frame at channel {channel}");
                return Ok(false);
            }
        }
    }
    println!("epilogue values={} frames={frames} ok", produced.len());
    Ok(true)
}

/// `nn.GroupNorm` over the whole volume: one mean and rstd per group, then
/// `norm * weight + bias`, the optional embedding's scale/shift, the optional
/// SiLU.
fn norm(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, groups, frames, area) = (4usize, 2usize, 3usize, 5usize);
    let plane = frames * area;
    let epsilon = 1.0e-5f32;
    let input = rng.values(channels * plane);
    let weight = rng.values(channels);
    let bias = rng.values(channels);
    let embedding = rng.values(2 * channels);

    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let weight_tensor = upload(gpu, Dtype::F32, &weight)?;
    let bias_tensor = upload(gpu, Dtype::F32, &bias)?;
    let embedding_tensor = upload(gpu, Dtype::F32, &embedding)?;
    let out = gpu.alloc(Dtype::F32, channels * plane)?;
    let per_group = channels / groups;
    let chunks = plane.div_ceil(8192);
    let partial = gpu.alloc(Dtype::F32, groups * per_group * chunks * 2)?;
    let mean = gpu.alloc(Dtype::F32, groups)?;
    let rstd = gpu.alloc(Dtype::F32, groups)?;
    gpu.latent_norm(
        &out,
        &input_tensor,
        &weight_tensor,
        &bias_tensor,
        Some(&embedding_tensor),
        &partial,
        &mean,
        &rstd,
        Dtype::F32,
        channels as u32,
        groups as u32,
        frames as u32,
        area as u32,
        epsilon,
        true,
    )?;

    let mut expected = vec![0.0f32; channels * plane];
    for group in 0..groups {
        let mut sum = 0.0f64;
        let mut squares = 0.0f64;
        let mut count = 0usize;
        for local in 0..per_group {
            let channel = group * per_group + local;
            for index in 0..plane {
                let value = input[channel * plane + index] as f64;
                sum += value;
                squares += value * value;
                count += 1;
            }
        }
        let average = sum / count as f64;
        let variance = squares / count as f64 - average * average;
        let inverse = 1.0 / (variance + epsilon as f64).sqrt();
        for local in 0..per_group {
            let channel = group * per_group + local;
            for index in 0..plane {
                let mut value = ((input[channel * plane + index] as f64 - average)
                    * inverse
                    * weight[channel] as f64)
                    + bias[channel] as f64;
                value = value * (1.0 + embedding[channel] as f64)
                    + embedding[channels + channel] as f64;
                let value = value as f32;
                expected[channel * plane + index] = value / (1.0 + (-value).exp());
            }
        }
    }
    Ok(check("norm", &download(gpu, &out)?, &expected, 1e-5))
}

fn silu_add(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let count = 11usize;
    let input = rng.values(count);
    let other = rng.values(count);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let other_tensor = upload(gpu, Dtype::F32, &other)?;
    let silu_out = gpu.alloc(Dtype::F32, count)?;
    let add_out = gpu.alloc(Dtype::F32, count)?;
    gpu.latent_silu(&silu_out, &input_tensor, Dtype::F32, count)?;
    gpu.latent_add(&add_out, &input_tensor, &other_tensor, Dtype::F32, count)?;

    let expected_silu: Vec<f32> = input
        .iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect();
    let expected_add: Vec<f32> = input
        .iter()
        .zip(&other)
        .map(|(left, right)| left + right)
        .collect();
    let mut pass = check("silu", &download(gpu, &silu_out)?, &expected_silu, 1e-6);
    pass &= check("add", &download(gpu, &add_out)?, &expected_add, 1e-6);
    Ok(pass)
}

fn depthwise(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, area, kernel) = (3usize, 5usize, 4usize, 3usize);
    let pad = kernel / 2;
    let input = rng.values(channels * frames * area);
    let weight = rng.values(channels * kernel);
    let bias = rng.values(channels);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let weight_tensor = upload(gpu, Dtype::F32, &weight)?;
    let bias_tensor = upload(gpu, Dtype::F32, &bias)?;
    let out = gpu.alloc(Dtype::F32, channels * frames * area)?;
    gpu.latent_depthwise_temporal(
        &out,
        &input_tensor,
        &weight_tensor,
        &bias_tensor,
        Dtype::F32,
        channels as u32,
        frames as u32,
        area as u32,
        kernel as u32,
    )?;

    let mut expected = vec![0.0f32; channels * frames * area];
    for c in 0..channels {
        for t in 0..frames {
            for point in 0..area {
                let mut sum = bias[c];
                for tap in 0..kernel {
                    let source = t as isize + tap as isize - pad as isize;
                    if source < 0 || source >= frames as isize {
                        continue;
                    }
                    sum += input[(c * frames + source as usize) * area + point]
                        * weight[c * kernel + tap];
                }
                expected[(c * frames + t) * area + point] = sum;
            }
        }
    }
    Ok(check("depthwise", &download(gpu, &out)?, &expected, 1e-6))
}

fn resize(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, height, width) = (2usize, 2usize, 3usize, 4usize);
    let (out_frames, out_height, out_width) = (2usize, 6usize, 8usize);
    let input = rng.values(channels * frames * height * width);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let out = gpu.alloc(Dtype::F32, channels * out_frames * out_height * out_width)?;
    gpu.latent_resize3d(
        &out,
        &input_tensor,
        Dtype::F32,
        channels as u32,
        [frames as u32, height as u32, width as u32],
        [out_frames as u32, out_height as u32, out_width as u32],
    )?;

    let axis = |in_size: usize, out_size: usize, index: usize| -> (usize, usize, f32) {
        let scale = in_size as f32 / out_size as f32;
        let mut source = scale * (index as f32 + 0.5) - 0.5;
        if source < 0.0 {
            source = 0.0;
        }
        let low = source as usize;
        let low = low.min(in_size - 1);
        let high = (low + 1).min(in_size - 1);
        (low, high, source - low as f32)
    };

    let mut expected = vec![0.0f32; channels * out_frames * out_height * out_width];
    for c in 0..channels {
        for t in 0..out_frames {
            let (t0, t1, ft) = axis(frames, out_frames, t);
            for y in 0..out_height {
                let (y0, y1, fy) = axis(height, out_height, y);
                for x in 0..out_width {
                    let (x0, x1, fx) = axis(width, out_width, x);
                    let mut sum = 0.0f32;
                    for (ts, wt) in [(t0, 1.0 - ft), (t1, ft)] {
                        for (ys, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                            for (xs, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                                sum += wt
                                    * wy
                                    * wx
                                    * input[((c * frames + ts) * height + ys) * width + xs];
                            }
                        }
                    }
                    expected[((c * out_frames + t) * out_height + y) * out_width + x] = sum;
                }
            }
        }
    }
    Ok(check("resize3d", &download(gpu, &out)?, &expected, 1e-6))
}

fn time_ops(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, sources, targets, area) = (2usize, 5usize, 3usize, 4usize);
    let input = rng.values(channels * sources * area);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;

    // The nearest pack, three slots, with one target unoccupied.
    let index = vec![0u32, 1, 2, 3, 4, u32::MAX, 0, u32::MAX, 2];
    let index_tensor = upload_u32(gpu, &index)?;
    let packed = gpu.alloc(Dtype::F32, 3 * channels * targets * area)?;
    gpu.latent_time_gather(
        &packed,
        &input_tensor,
        &index_tensor,
        Dtype::F32,
        channels as u32,
        sources as u32,
        targets as u32,
        area as u32,
        3,
    )?;

    let produced = download(gpu, &packed)?;
    let mut pass = true;
    for slot in 0..3usize {
        for target in 0..targets {
            let source = index[target * 3 + slot];
            for c in 0..channels {
                for point in 0..area {
                    let got = produced[((slot * channels + c) * targets + target) * area + point];
                    let want = if source == u32::MAX {
                        0.0
                    } else {
                        input[(c * sources + source as usize) * area + point]
                    };
                    if (got - want).abs() > 0.0 {
                        println!("gather FAIL slot {slot} target {target} c {c} point {point}");
                        pass = false;
                    }
                }
            }
        }
    }
    println!("time_gather ok={pass}");

    // The resample: `a + w * (b - a)`.
    let left = vec![0u32, 1, 4];
    let right = vec![1u32, 2, 4];
    let weight = vec![0.25f32, 0.5, 1.0];
    let left_tensor = upload_u32(gpu, &left)?;
    let right_tensor = upload_u32(gpu, &right)?;
    let weight_tensor = upload(gpu, Dtype::F32, &weight)?;
    let lerped = gpu.alloc(Dtype::F32, channels * targets * area)?;
    gpu.latent_time_lerp(
        &lerped,
        &input_tensor,
        &left_tensor,
        &right_tensor,
        &weight_tensor,
        Dtype::F32,
        channels as u32,
        sources as u32,
        targets as u32,
        area as u32,
    )?;

    let mut expected = vec![0.0f32; channels * targets * area];
    for c in 0..channels {
        for target in 0..targets {
            for point in 0..area {
                let near = input[(c * sources + left[target] as usize) * area + point];
                let far = input[(c * sources + right[target] as usize) * area + point];
                expected[(c * targets + target) * area + point] =
                    near + weight[target] * (far - near);
            }
        }
    }
    pass &= check("time_lerp", &download(gpu, &lerped)?, &expected, 1e-6);
    Ok(pass)
}

fn unshuffle(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, height, width) = (3usize, 2usize, 4usize, 6usize);
    let input = rng.values(channels * frames * height * width);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let out = gpu.alloc(
        Dtype::F32,
        channels * 4 * frames * (height / 2) * (width / 2),
    )?;
    gpu.latent_unshuffle2(
        &out,
        &input_tensor,
        Dtype::F32,
        channels as u32,
        frames as u32,
        height as u32,
        width as u32,
    )?;

    let mut expected = vec![0.0f32; channels * 4 * frames * (height / 2) * (width / 2)];
    for c in 0..channels {
        for sub in 0..4usize {
            for t in 0..frames {
                for y in 0..height / 2 {
                    for x in 0..width / 2 {
                        let destination =
                            (((c * 4 + sub) * frames + t) * (height / 2) + y) * (width / 2) + x;
                        let source = ((c * frames + t) * height + (2 * y + sub / 2)) * width
                            + (2 * x + sub % 2);
                        expected[destination] = input[source];
                    }
                }
            }
        }
    }
    Ok(check("unshuffle2", &download(gpu, &out)?, &expected, 0.0))
}

fn accumulate(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, area) = (2usize, 5usize, 3usize);
    let in_frames = 4usize;
    let weights = vec![0.5f32, 1.0, 2.0];
    let segment = rng.values(channels * in_frames * area);
    let segment_tensor = upload(gpu, Dtype::F32, &segment)?;
    let weight_tensor = upload(gpu, Dtype::F32, &weights)?;
    let accumulator = upload(gpu, Dtype::F32, &vec![0.0f32; channels * frames * area])?;
    // Two segments, the second reading at an offset and writing elsewhere.
    gpu.latent_accumulate(
        &accumulator,
        &segment_tensor,
        &weight_tensor,
        Dtype::F32,
        channels as u32,
        frames as u32,
        3,
        area as u32,
        in_frames as u32,
        0,
        0,
    )?;
    gpu.latent_accumulate(
        &accumulator,
        &segment_tensor,
        &weight_tensor,
        Dtype::F32,
        channels as u32,
        frames as u32,
        3,
        area as u32,
        in_frames as u32,
        1,
        2,
    )?;

    let mut expected = vec![0.0f32; channels * frames * area];
    for (in_begin, out_begin) in [(0usize, 0usize), (1, 2)] {
        for c in 0..channels {
            for frame in 0..3 {
                for point in 0..area {
                    expected[(c * frames + out_begin + frame) * area + point] +=
                        segment[(c * in_frames + in_begin + frame) * area + point] * weights[frame];
                }
            }
        }
    }
    let mut pass = check("accumulate", &download(gpu, &accumulator)?, &expected, 1e-6);

    // The average, with the guard `weight_full.clamp(min=1e-8)`.
    let totals = upload(gpu, Dtype::F32, &[1.0f32, 1.0, 1.0, 2.0, 1.0])?;
    let out = gpu.alloc(Dtype::F32, channels * frames * area)?;
    gpu.latent_blend_finish(
        &out,
        &accumulator,
        &totals,
        Dtype::F32,
        channels as u32,
        frames as u32,
        area as u32,
    )?;
    let mut averaged = expected.clone();
    for c in 0..channels {
        for frame in 0..frames {
            for point in 0..area {
                averaged[(c * frames + frame) * area + point] /= totals_value(frame);
            }
        }
    }
    pass &= check("blend_finish", &download(gpu, &out)?, &averaged, 1e-6);
    Ok(pass)
}

fn totals_value(frame: usize) -> f32 {
    [1.0f32, 1.0, 1.0, 2.0, 1.0][frame]
}

fn gate_mul(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (half, area) = (3usize, 4usize);
    let input = rng.values(2 * half * area);
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let out = gpu.alloc(Dtype::F32, half * area)?;
    gpu.latent_gate_mul(&out, &input_tensor, Dtype::F32, half as u32, area)?;

    let expected: Vec<f32> = (0..half * area)
        .map(|index| {
            let gate = input[half * area + index];
            input[index] * (gate / (1.0 + (-gate).exp()))
        })
        .collect();
    Ok(check("gate_mul", &download(gpu, &out)?, &expected, 1e-6))
}

fn scale(gpu: &Gpu, rng: &mut Rng) -> Result<bool, String> {
    let (channels, frames, area) = (3usize, 2usize, 4usize);
    let plane = frames * area;
    let input = rng.values(channels * plane);
    let center: Vec<f32> = (0..channels).map(|c| c as f32 * 0.25).collect();
    let factor: Vec<f32> = (0..channels).map(|c| 1.0 + c as f32 * 0.5).collect();
    let input_tensor = upload(gpu, Dtype::F32, &input)?;
    let center_tensor = upload(gpu, Dtype::F32, &center)?;
    let factor_tensor = upload(gpu, Dtype::F32, &factor)?;
    let normalized = gpu.alloc(Dtype::F32, channels * plane)?;
    let restored = gpu.alloc(Dtype::F32, channels * plane)?;
    gpu.latent_scale(
        &normalized,
        &input_tensor,
        &center_tensor,
        &factor_tensor,
        Dtype::F32,
        channels as u32,
        plane,
        false,
    )?;
    gpu.latent_scale(
        &restored,
        &normalized,
        &center_tensor,
        &factor_tensor,
        Dtype::F32,
        channels as u32,
        plane,
        true,
    )?;

    let mut expected = vec![0.0f32; channels * plane];
    for c in 0..channels {
        for index in 0..plane {
            expected[c * plane + index] = (input[c * plane + index] - center[c]) / factor[c];
        }
    }
    let mut pass = check("scale_N", &download(gpu, &normalized)?, &expected, 1e-6);
    // D undoes N channel by channel: a wrong channel stride cannot round-trip.
    pass &= check("scale_D", &download(gpu, &restored)?, &input, 1e-5);
    Ok(pass)
}
