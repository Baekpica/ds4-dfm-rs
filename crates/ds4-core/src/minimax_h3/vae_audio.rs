//! The released BigVGAN AudioVAE (`h3_audio_vae.c:1-1343`).
//!
//! Decode (`:1-715`) and encode (`:708-1343`). The decoder serves every
//! render; the encoder serves Ref2VA audio references and is gated on its own
//! (`h3_audio_encoder_gate`). The graph, the stage geometry, the weight-norm
//! reconstruction and the alias-free Snake activation are transcribed in the
//! reference's order, over the F32 primitives the two halves call
//! (`h3d_conv1d`, `h3d_conv_transpose1d`, `h3d_weight_norm`,
//! `h3d_alias_free_snake`, `h3d_snake1d_f32`, `h3d_add_scaled`,
//! `h3d_copy_f32`, `h3d_clip`, `h3d_layer_norm_f32`, `h3d_linear`,
//! `h3d_geglu_f32`, `h3d_audio_qkv_split_f32`, `h3d_sdpa_causal_f32`,
//! `h3d_audio_attention_pool_f32`).
//!
//! Layouts are the reference's: activations are [batch, time, channels] with
//! the decoder's mono weights run over a batch of two (stereo folded into the
//! batch, `h3_audio_vae.h:47-52`); the latent is [32, 2, T], transposed and
//! normalized in `prepare_input` (`:351-379`) on the way out of the decoder
//! and de-normalized in `encoder_output` (`:1253-1268`) on the way out of the
//! encoder.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::vae_store::{f1, f2, f3, Store};

pub const LATENT_CHANNELS: usize = 32;
const LATENT_DIM: usize = 2048;
const DECODER_DIM: usize = 1024;
const STEREO: u32 = 2;
const STAGES: usize = 7;
const RESBLOCKS: usize = 3;
const RESIDUAL_PAIRS: usize = 3;
const FILTER_SIZE: u32 = 12;
pub const SAMPLE_RATE: u32 = 32000;
/// `h3_audio_vae.h:44`: 800 samples per latent step.
pub const HOP_LENGTH: usize = 800;
const UPSAMPLE_RATES: [u32; STAGES] = [5, 5, 2, 2, 2, 2, 2];
const UPSAMPLE_KERNELS: [u32; STAGES] = [9, 9, 4, 4, 4, 4, 4];
const RESIDUAL_KERNELS: [u32; RESBLOCKS] = [3, 7, 11];
const RESIDUAL_DILATIONS: [u32; RESIDUAL_PAIRS] = [1, 3, 5];
const DEVICE_ERROR: &str = "audio VAE device";

/// What `h3_audio_vae_decode` returns (`h3_audio_vae.h:16-24`).
pub struct Waveform {
    pub channels: usize,
    pub samples: usize,
    pub sample_rate: u32,
    /// Channel-major F32, clipped to [-1, 1].
    pub pcm: Vec<f32>,
}

#[derive(Debug)]
pub struct VaeError(String);

impl fmt::Display for VaeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VaeError {}

impl From<String> for VaeError {
    fn from(detail: String) -> VaeError {
        VaeError(detail)
    }
}

fn fail<T>(detail: impl Into<String>) -> Result<T, VaeError> {
    Err(VaeError(detail.into()))
}

fn device<T>(result: Result<T, String>) -> Result<T, VaeError> {
    result.map_err(|detail| VaeError(format!("{DEVICE_ERROR}: {detail}")))
}

/// One convolution with its weight already reconstructed (`audio_conv`,
/// `h3_audio_vae.c:31-45`). `weight` is the normalized tensor for a
/// weight-norm convolution.
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    padding: u32,
    dilation: u32,
    stride: u32,
    transpose: bool,
}

/// `audio_activation` (`:47-50`); alpha and beta are stored as logs.
struct Activation {
    alpha: Tensor,
    beta: Tensor,
}

struct Resblock {
    activations: Vec<Activation>,
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

struct Stage {
    upsample: Conv,
    blocks: Vec<Resblock>,
}

/// `load_plain_conv` (`:155-177`).
#[allow(clippy::too_many_arguments)]
fn load_plain_conv(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    padding: u32,
    dilation: u32,
    has_bias: bool,
) -> Result<Conv, VaeError> {
    let weight = f3(
        store,
        gpu,
        &format!("{prefix}.weight"),
        output_channels as u64,
        input_channels as u64,
        kernel as u64,
    )?;
    let bias = match has_bias {
        true => Some(f1(
            store,
            gpu,
            &format!("{prefix}.bias"),
            output_channels as u64,
        )?),
        false => None,
    };

    Ok(Conv {
        weight,
        bias,
        input_channels,
        output_channels,
        kernel,
        padding,
        dilation,
        stride: 1,
        transpose: false,
    })
}

/// `load_normalized_conv` (`:179-216`) plus `normalize_conv` (`:218-232`):
/// the checkpoint stores `weight_v`/`weight_g`, and the device rebuilds
/// `w = v * g / ||v||` with the reference's own fmaf/rsqrtf kernel.
#[allow(clippy::too_many_arguments)]
fn load_normalized_conv(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    padding: u32,
    dilation: u32,
    stride: u32,
    transpose: bool,
    has_bias: bool,
) -> Result<Conv, VaeError> {
    let outer = if transpose {
        input_channels
    } else {
        output_channels
    };
    let inner_channels = if transpose {
        output_channels
    } else {
        input_channels
    };
    let vector = f3(
        store,
        gpu,
        &format!("{prefix}.weight_v"),
        outer as u64,
        inner_channels as u64,
        kernel as u64,
    )?;
    let magnitude = f3(
        store,
        gpu,
        &format!("{prefix}.weight_g"),
        outer as u64,
        1,
        1,
    )?;

    let elements = outer as usize * inner_channels as usize * kernel as usize;
    let weight = device(gpu.alloc(Dtype::F32, elements))?;
    device(gpu.weight_norm(&weight, &vector, &magnitude, outer, inner_channels * kernel))?;
    // `retire_normalization_inputs` (`:234-237`): the pair is not needed once
    // the weight exists.
    drop(vector);
    drop(magnitude);

    let bias = match has_bias {
        true => Some(f1(
            store,
            gpu,
            &format!("{prefix}.bias"),
            output_channels as u64,
        )?),
        false => None,
    };

    Ok(Conv {
        weight,
        bias,
        input_channels,
        output_channels,
        kernel,
        padding,
        dilation,
        stride,
        transpose,
    })
}

/// `load_activation` (`:239-249`).
fn load_activation(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    channels: u32,
) -> Result<Activation, VaeError> {
    Ok(Activation {
        alpha: f1(store, gpu, &format!("{prefix}.act.alpha"), channels as u64)?,
        beta: f1(store, gpu, &format!("{prefix}.act.beta"), channels as u64)?,
    })
}

/// `load_stage` (`:423-469`).
fn load_stage(store: &Store, gpu: &Gpu, index: usize) -> Result<Stage, VaeError> {
    let input_channels = (DECODER_DIM >> index) as u32;
    let output_channels = (DECODER_DIM >> (index + 1)) as u32;
    let rate = UPSAMPLE_RATES[index];
    let kernel = UPSAMPLE_KERNELS[index];
    let padding = (kernel - rate) / 2;
    let prefix = format!("decoder.ups.{index}.0");
    let upsample = load_normalized_conv(
        store,
        gpu,
        &prefix,
        input_channels,
        output_channels,
        kernel,
        padding,
        1,
        rate,
        true,
        true,
    )?;

    let mut blocks = Vec::with_capacity(RESBLOCKS);
    for (block_index, residual_kernel) in RESIDUAL_KERNELS.iter().enumerate().take(RESBLOCKS) {
        let global = index * RESBLOCKS + block_index;
        let residual_kernel = *residual_kernel;
        let mut activations = Vec::with_capacity(RESIDUAL_PAIRS * 2);
        let mut convs1 = Vec::with_capacity(RESIDUAL_PAIRS);
        let mut convs2 = Vec::with_capacity(RESIDUAL_PAIRS);

        for (pair, dilation) in RESIDUAL_DILATIONS.iter().enumerate().take(RESIDUAL_PAIRS) {
            for which in 0..2 {
                let activation = pair * 2 + which;
                let prefix = format!("decoder.resblocks.{global}.activations.{activation}");
                activations.push(load_activation(store, gpu, &prefix, output_channels)?);
            }

            let dilation = *dilation;
            let dilated_padding = dilation * (residual_kernel - 1) / 2;
            convs1.push(load_normalized_conv(
                store,
                gpu,
                &format!("decoder.resblocks.{global}.convs1.{pair}"),
                output_channels,
                output_channels,
                residual_kernel,
                dilated_padding,
                dilation,
                1,
                false,
                true,
            )?);
            convs2.push(load_normalized_conv(
                store,
                gpu,
                &format!("decoder.resblocks.{global}.convs2.{pair}"),
                output_channels,
                output_channels,
                residual_kernel,
                (residual_kernel - 1) / 2,
                1,
                1,
                false,
                true,
            )?);
        }

        blocks.push(Resblock {
            activations,
            convs1,
            convs2,
        });
    }

    Ok(Stage { upsample, blocks })
}

/// `run_conv` (`:504-517`).
fn run_conv(
    gpu: &Gpu,
    output: &Tensor,
    input: &Tensor,
    conv: &Conv,
    length: u32,
) -> Result<(), VaeError> {
    let result = match conv.transpose {
        true => gpu.conv_transpose1d(
            output,
            input,
            &conv.weight,
            conv.bias.as_ref(),
            STEREO,
            length,
            conv.input_channels,
            conv.output_channels,
            conv.kernel,
            conv.stride,
            conv.padding,
        ),
        false => gpu.conv1d(
            output,
            input,
            &conv.weight,
            conv.bias.as_ref(),
            STEREO,
            length,
            conv.input_channels,
            conv.output_channels,
            conv.kernel,
            conv.stride,
            conv.padding,
            conv.dilation,
        ),
    };
    device(result)
}

/// `run_activation` (`:493-502`).
fn run_activation(
    gpu: &Gpu,
    output: &Tensor,
    input: &Tensor,
    activation: &Activation,
    filters: &(Tensor, Tensor),
    length: u32,
    channels: u32,
) -> Result<(), VaeError> {
    device(gpu.alias_free_snake(
        output,
        input,
        &activation.alpha,
        &activation.beta,
        &filters.0,
        &filters.1,
        STEREO,
        length,
        channels,
    ))
}

fn f32_alloc(gpu: &Gpu, elements: usize) -> Result<Tensor, VaeError> {
    device(gpu.alloc(Dtype::F32, elements))
}

/// `h3_gpu_tensor_read_f32` for the encoder's final readback.
fn read_f32(gpu: &Gpu, tensor: &Tensor, elements: usize) -> Result<Vec<f32>, VaeError> {
    let mut bytes = vec![0u8; elements * 4];
    device(gpu.read(tensor, &mut bytes))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// The whole decoder, one call: `h3_audio_vae_decode` (`:660-705`).
///
/// `latent` is normalized posterior means in [32, 2, T] with
/// `latent_length == T`, exactly what `Dit::forward` leaves in `audio_input`
/// (`h3_audio_vae.h:31-38`).
pub fn decode(
    weight_directory: &Path,
    latent: &[f32],
    latent_length: usize,
) -> Result<Waveform, VaeError> {
    if latent_length == 0 {
        return fail("audio VAE needs at least one latent step");
    }
    let expected = LATENT_CHANNELS * STEREO as usize * latent_length;
    if latent.len() != expected {
        return fail(format!(
            "audio VAE latent is {} values, expected {expected}",
            latent.len()
        ));
    }

    let (mean, deviation) = latent_normalization(weight_directory)?;
    let store = Store::open(weight_directory, "audio VAE")?;
    let gpu = device(Gpu::create())?;

    // `prepare_input` (`:351-421`).
    let mut rows = vec![0.0f32; expected];
    let mut destination = 0usize;
    for stereo in 0..STEREO as usize {
        for time in 0..latent_length {
            for channel in 0..LATENT_CHANNELS {
                let source = (channel * STEREO as usize + stereo) * latent_length + time;
                rows[destination] = latent[source] * deviation[channel] + mean[channel];
                destination += 1;
            }
        }
    }

    let latent_tensor = f32_alloc(&gpu, expected)?;
    device(gpu.upload(&latent_tensor, bytemuck_f32(&rows)))?;

    let projection = load_plain_conv(
        &store,
        &gpu,
        "dec_in_proj",
        LATENT_CHANNELS as u32,
        LATENT_DIM as u32,
        1,
        0,
        1,
        true,
    )?;
    let pre = load_normalized_conv(
        &store,
        &gpu,
        "decoder.conv_pre",
        LATENT_DIM as u32,
        DECODER_DIM as u32,
        7,
        3,
        1,
        1,
        false,
        true,
    )?;
    let projected = f32_alloc(&gpu, STEREO as usize * latent_length * LATENT_DIM)?;
    let mut hidden = Some(f32_alloc(
        &gpu,
        STEREO as usize * latent_length * DECODER_DIM,
    )?);

    run_conv(
        &gpu,
        &projected,
        &latent_tensor,
        &projection,
        latent_length as u32,
    )?;
    run_conv(
        &gpu,
        hidden.as_ref().unwrap(),
        &projected,
        &pre,
        latent_length as u32,
    )?;
    drop(latent_tensor);
    drop(projected);
    drop(projection);
    drop(pre);

    // `load_filters` (`:251-260`): both alias-free filters are shared by every
    // activation.
    let filters = (
        f3(
            &store,
            &gpu,
            "decoder.activation_post.upsample.filter",
            1,
            1,
            FILTER_SIZE as u64,
        )?,
        f3(
            &store,
            &gpu,
            "decoder.activation_post.downsample.lowpass.filter",
            1,
            1,
            FILTER_SIZE as u64,
        )?,
    );

    let mut length = latent_length as u32;
    for index in 0..STAGES {
        let stage = load_stage(&store, &gpu, index)?;
        let channels = (DECODER_DIM >> (index + 1)) as u32;
        let kernel = UPSAMPLE_KERNELS[index];
        let stride = UPSAMPLE_RATES[index];
        let padding = (kernel - stride) / 2;
        let expanded = (length as u64 - 1) * stride as u64 + kernel as u64 - 2 * padding as u64;
        let output_length = u32::try_from(expanded)
            .map_err(|_| VaeError("audio VAE stage length overflows".into()))?;
        let elements = STEREO as usize * output_length as usize * channels as usize;

        let upsampled = f32_alloc(&gpu, elements)?;
        let sum = f32_alloc(&gpu, elements)?;
        let work = f32_alloc(&gpu, elements)?;
        let activated = f32_alloc(&gpu, elements)?;
        let branch = f32_alloc(&gpu, elements)?;

        run_conv(
            &gpu,
            &upsampled,
            hidden.as_ref().unwrap(),
            &stage.upsample,
            length,
        )?;

        // `run_stage` (`:519-603`): three ResBlocks, each three dilation
        // pairs; block 1 adds into the sum, block 2 averages.
        for (block_index, block) in stage.blocks.iter().enumerate() {
            let target = if block_index == 0 { &sum } else { &work };
            device(gpu.copy_f32(target, 0, &upsampled, 0, elements))?;
            for pair in 0..RESIDUAL_PAIRS {
                run_activation(
                    &gpu,
                    &activated,
                    target,
                    &block.activations[pair * 2],
                    &filters,
                    output_length,
                    channels,
                )?;
                run_conv(
                    &gpu,
                    &branch,
                    &activated,
                    &block.convs1[pair],
                    output_length,
                )?;
                run_activation(
                    &gpu,
                    &activated,
                    &branch,
                    &block.activations[pair * 2 + 1],
                    &filters,
                    output_length,
                    channels,
                )?;
                run_conv(
                    &gpu,
                    &branch,
                    &activated,
                    &block.convs2[pair],
                    output_length,
                )?;
                device(gpu.add_scaled(target, target, &branch, 1.0, 1.0))?;
            }
            if block_index == 1 {
                device(gpu.add_scaled(&sum, &sum, &work, 1.0, 1.0))?;
            }
            if block_index == 2 {
                device(gpu.add_scaled(&sum, &sum, &work, 1.0 / 3.0, 1.0 / 3.0))?;
            }
        }

        drop(upsampled);
        drop(work);
        drop(activated);
        drop(branch);
        hidden = Some(sum);
        length = output_length;
    }

    // `decode_output` (`:605-658`).
    let post_activation = load_activation(&store, &gpu, "decoder.activation_post", 8)?;
    let post = load_normalized_conv(
        &store,
        &gpu,
        "decoder.conv_post",
        8,
        1,
        7,
        3,
        1,
        1,
        false,
        false,
    )?;
    let activated = f32_alloc(&gpu, STEREO as usize * length as usize * 8)?;
    let waveform = f32_alloc(&gpu, STEREO as usize * length as usize)?;
    run_activation(
        &gpu,
        &activated,
        hidden.as_ref().unwrap(),
        &post_activation,
        &filters,
        length,
        8,
    )?;
    run_conv(&gpu, &waveform, &activated, &post, length)?;
    device(gpu.clip(&waveform, &waveform, -1.0, 1.0))?;
    device(gpu.sync())?;

    let elements = STEREO as usize * length as usize;
    let mut bytes = vec![0u8; elements * 4];
    device(gpu.read(&waveform, &mut bytes))?;
    let pcm: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();

    Ok(Waveform {
        channels: STEREO as usize,
        samples: length as usize,
        sample_rate: SAMPLE_RATE,
        pcm,
    })
}

/// `load_latent_normalization` (`:300-349`): `latents_mean` and `latents_std`
/// from `config.json`, parsed with `strtof`'s correctly-rounded semantics
/// (`str::parse::<f32>`), not through an f64 intermediate.
fn latent_normalization(dir: &Path) -> Result<(Vec<f32>, Vec<f32>), VaeError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| VaeError(format!("cannot read {}: {error}", path.display())))?;
    let mean = float_array(&text, "latents_mean", LATENT_CHANNELS)?;
    let deviation = float_array(&text, "latents_std", LATENT_CHANNELS)?;
    if deviation.iter().any(|value| *value <= 0.0) {
        return fail("audio VAE latent standard deviation is invalid");
    }
    Ok((mean, deviation))
}

fn float_array(text: &str, key: &str, count: usize) -> Result<Vec<f32>, VaeError> {
    let pattern = format!("\"{key}\"");
    let start = text
        .find(&pattern)
        .ok_or_else(|| VaeError(format!("audio VAE config is missing {key}")))?;
    let open = text[start..]
        .find('[')
        .map(|offset| start + offset)
        .ok_or_else(|| VaeError(format!("audio VAE config is missing {key}")))?;
    let close = text[open..]
        .find(']')
        .map(|offset| open + offset)
        .ok_or_else(|| VaeError(format!("audio VAE config has malformed {key}")))?;

    let mut values = Vec::with_capacity(count);
    for token in text[open + 1..close].split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: f32 = token
            .parse()
            .map_err(|_| VaeError(format!("audio VAE config has malformed {key}")))?;
        if !value.is_finite() {
            return fail(format!("audio VAE config has malformed {key}"));
        }
        values.push(value);
    }
    if values.len() != count {
        return fail(format!(
            "audio VAE config {key} has {} values, expected {count}",
            values.len()
        ));
    }
    Ok(values)
}

fn bytemuck_f32(values: &[f32]) -> &[u8] {
    // No `bytemuck` dependency: F32 has no padding, so this is just a view.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_arrays_parse_like_strtof() {
        let text = r#"{"latents_mean":[-1.5, 0.25],"latents_std":[1.0, 2.5e-1]}"#;
        let mean = float_array(text, "latents_mean", 2).expect("mean");
        let deviation = float_array(text, "latents_std", 2).expect("std");
        assert_eq!(mean, vec![-1.5, 0.25]);
        assert_eq!(deviation, vec![1.0, 0.25]);
    }

    #[test]
    fn config_arrays_reject_a_short_or_malformed_list() {
        assert!(float_array(r#"{"latents_mean":[1.0]}"#, "latents_mean", 2).is_err());
        assert!(float_array(r#"{"latents_mean":[x]}"#, "latents_mean", 1).is_err());
        assert!(float_array(r#"{"latents_mean":[NaN]}"#, "latents_mean", 1).is_err());
    }

    #[test]
    fn geometry_matches_the_released_decoder() {
        // 8 latents x 5 x 5 x 2^5 = 6400 samples, the CLI's 5-frame case.
        let mut length = 8u32;
        for (index, rate) in UPSAMPLE_RATES.iter().enumerate() {
            length = (length - 1) * rate + UPSAMPLE_KERNELS[index]
                - 2 * ((UPSAMPLE_KERNELS[index] - rate) / 2);
        }
        assert_eq!(length as usize, 6400);
        assert_eq!(length as usize, 8 * HOP_LENGTH);
    }
}

// --- the encode half (`h3_audio_vae.c:708-1343`) -------------------------

/// `ENCODER_STAGES`/`ENCODER_RESIDUALS`/`ENCODER_HEADS`/`ENCODER_DIM`
/// (`h3_audio_vae.c:708-713`).
const ENCODER_STAGES: usize = 5;
const ENCODER_RESIDUALS: usize = 3;
const ENCODER_HEADS: u32 = 8;
const ENCODER_DIM: u32 = 64;
/// `encoder_strides` / `encoder_dilations` (`:715-716`).
const ENCODER_STRIDES: [u32; ENCODER_STAGES] = [2, 4, 4, 5, 5];
const ENCODER_DILATIONS: [u32; ENCODER_RESIDUALS] = [1, 3, 9];
/// `1e-5f` in every encoder LayerNorm (`:1040`, `:1081`, `:1163`, `:1199`).
const ENCODER_EPSILON: f32 = 1e-5;
const ENCODER_RESIDUAL_KERNEL: u32 = 7;

/// What `h3_audio_vae_encode` returns (`h3_audio_vae.h:16-30`):
/// normalized posterior means in [32, 2, T], channel-major.
pub struct Latent {
    pub channels: usize,
    pub stereo: usize,
    pub length: usize,
    pub values: Vec<f32>,
}

/// `encoder_conv_length` (`h3_audio_vae.c:746-755`).
fn conv_length(length: u32, kernel: u32, stride: u32, padding: u32, dilation: u32) -> Option<u32> {
    let effective = u64::from(dilation) * u64::from(kernel - 1) + 1;
    let padded = u64::from(length) + 2 * u64::from(padding);

    if padded < effective || stride == 0 {
        return None;
    }

    let output = (padded - effective) / u64::from(stride) + 1;

    if output == 0 || output > u64::from(u32::MAX) {
        return None;
    }

    Some(output as u32)
}

/// `encoder_alpha` (`:757-761`): the checkpoint stores the log alpha as a
/// [1, channels, 1] tensor.
fn encoder_alpha(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    channels: u32,
) -> Result<Tensor, VaeError> {
    Ok(f3(
        store,
        gpu,
        &format!("{prefix}.alpha"),
        1,
        channels as u64,
        1,
    )?)
}

/// `load_linear` (`:995-1009`) for the pre-block: a 2-D weight and one bias.
#[allow(clippy::too_many_arguments)]
fn load_linear(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    input_dim: u32,
    output_dim: u32,
    has_bias: bool,
) -> Result<(Tensor, Option<Tensor>), VaeError> {
    let weight = f2(
        store,
        gpu,
        &format!("{prefix}.weight"),
        output_dim as u64,
        input_dim as u64,
    )?;
    let bias = match has_bias {
        true => Some(f1(
            store,
            gpu,
            &format!("{prefix}.bias"),
            output_dim as u64,
        )?),
        false => None,
    };

    Ok((weight, bias))
}

/// `load_norm` (`:978-993`).
fn load_norm(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    width: u32,
) -> Result<(Tensor, Tensor), VaeError> {
    let weight = f1(store, gpu, &format!("{prefix}.weight"), width as u64)?;
    let bias = f1(store, gpu, &format!("{prefix}.bias"), width as u64)?;

    Ok((weight, bias))
}

#[allow(clippy::too_many_arguments)]
fn run_linear(
    gpu: &Gpu,
    output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    rows: u32,
    input_dim: u32,
    output_dim: u32,
) -> Result<(), VaeError> {
    device(gpu.linear(
        output,
        input,
        weight,
        bias,
        rows,
        input_dim,
        output_dim,
        Dtype::F32,
    ))
}

fn run_layer_norm(
    gpu: &Gpu,
    output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    rows: u32,
    width: u32,
) -> Result<(), VaeError> {
    device(gpu.layer_norm_f32(output, input, weight, bias, rows, width, ENCODER_EPSILON))
}

fn run_snake(
    gpu: &Gpu,
    output: &Tensor,
    input: &Tensor,
    alpha: &Tensor,
    length: u32,
    channels: u32,
) -> Result<(), VaeError> {
    device(gpu.snake1d_f32(output, input, alpha, STEREO, length, channels))
}

/// `encoder_initial` (`:764-810`): zero-pad the waveform to a whole hop, then
/// one 7-tap convolution from the two channels to 64.
fn encoder_initial(
    store: &Store,
    gpu: &Gpu,
    pcm: &[f32],
    samples: usize,
) -> Result<(Tensor, u32), VaeError> {
    let padded = samples.div_ceil(HOP_LENGTH) * HOP_LENGTH;
    let mut input_rows = vec![0.0f32; STEREO as usize * padded];

    for channel in 0..STEREO as usize {
        input_rows[channel * padded..channel * padded + samples]
            .copy_from_slice(&pcm[channel * samples..(channel + 1) * samples]);
    }

    let input = f32_alloc(gpu, input_rows.len())?;
    device(gpu.upload(&input, bytemuck_f32(&input_rows)))?;

    let hidden = f32_alloc(gpu, STEREO as usize * padded * ENCODER_DIM as usize)?;
    let conv = load_normalized_conv(
        store,
        gpu,
        "encoder.block.0",
        1,
        ENCODER_DIM,
        7,
        3,
        1,
        1,
        false,
        true,
    )?;
    run_conv(gpu, &hidden, &input, &conv, padded as u32)?;

    Ok((hidden, padded as u32))
}

/// `encoder_residual` (`:812-879`): Snake → dilated 7-tap → Snake → 1-tap,
/// added into the running hidden state.
fn encoder_residual(
    store: &Store,
    gpu: &Gpu,
    stage: usize,
    residual: usize,
    channels: u32,
    hidden: &Tensor,
    length: u32,
) -> Result<(), VaeError> {
    let dilation = ENCODER_DILATIONS[residual];
    let alpha1 = encoder_alpha(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.{residual}.block.0"),
        channels,
    )?;
    let conv1 = load_normalized_conv(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.{residual}.block.1"),
        channels,
        channels,
        ENCODER_RESIDUAL_KERNEL,
        3 * dilation,
        dilation,
        1,
        false,
        true,
    )?;
    let alpha2 = encoder_alpha(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.{residual}.block.2"),
        channels,
    )?;
    let conv2 = load_normalized_conv(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.{residual}.block.3"),
        channels,
        channels,
        1,
        0,
        1,
        1,
        false,
        true,
    )?;

    let elements = STEREO as usize * length as usize * channels as usize;
    let activated = f32_alloc(gpu, elements)?;
    let branch = f32_alloc(gpu, elements)?;

    run_snake(gpu, &activated, hidden, &alpha1, length, channels)?;
    run_conv(gpu, &branch, &activated, &conv1, length)?;
    run_snake(gpu, &activated, &branch, &alpha2, length, channels)?;
    run_conv(gpu, &branch, &activated, &conv2, length)?;
    device(gpu.add_scaled(hidden, hidden, &branch, 1.0, 1.0))?;

    Ok(())
}

/// `encoder_downsample` (`:881-938`): Snake, then a `2*stride`-tap strided
/// convolution that doubles the channel count.
fn encoder_downsample(
    store: &Store,
    gpu: &Gpu,
    stage: usize,
    channels: u32,
    hidden: &Tensor,
    length: u32,
) -> Result<(Tensor, u32, u32), VaeError> {
    let stride = ENCODER_STRIDES[stage - 1];
    let kernel = stride * 2;
    let padding = stride.div_ceil(2);
    let output_channels = channels * 2;
    let output_length = conv_length(length, kernel, stride, padding, 1)
        .ok_or_else(|| VaeError("AudioVAE encoder downsample geometry".to_string()))?;
    let alpha = encoder_alpha(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.3"),
        channels,
    )?;
    let conv = load_normalized_conv(
        store,
        gpu,
        &format!("encoder.block.{stage}.block.4"),
        channels,
        output_channels,
        kernel,
        padding,
        1,
        stride,
        false,
        true,
    )?;

    let activated = f32_alloc(gpu, STEREO as usize * length as usize * channels as usize)?;
    let down = f32_alloc(
        gpu,
        STEREO as usize * output_length as usize * output_channels as usize,
    )?;

    run_snake(gpu, &activated, hidden, &alpha, length, channels)?;
    run_conv(gpu, &down, &activated, &conv, length)?;

    Ok((down, output_length, output_channels))
}

/// `encoder_final_conv` (`:940-976`): Snake at 2048, then a 3-tap
/// convolution inside the same width.
fn encoder_final_conv(
    store: &Store,
    gpu: &Gpu,
    hidden: &Tensor,
    length: u32,
) -> Result<Tensor, VaeError> {
    let channels = LATENT_DIM as u32;
    let alpha = encoder_alpha(store, gpu, "encoder.block.6", channels)?;
    let conv = load_normalized_conv(
        store,
        gpu,
        "encoder.block.7",
        channels,
        channels,
        3,
        1,
        1,
        1,
        false,
        true,
    )?;

    let elements = STEREO as usize * length as usize * channels as usize;
    let activated = f32_alloc(gpu, elements)?;
    let output = f32_alloc(gpu, elements)?;

    run_snake(gpu, &activated, hidden, &alpha, length, channels)?;
    run_conv(gpu, &output, &activated, &conv, length)?;

    Ok(output)
}

/// `encoder_projection_branch` (`:1011-1056`): LayerNorm at 2048, then the
/// linear down to the 32-channel base.
fn encoder_projection_branch(
    store: &Store,
    gpu: &Gpu,
    hidden: &Tensor,
    rows: u32,
    length: u32,
) -> Result<Tensor, VaeError> {
    let (norm_weight, norm_bias) = load_norm(store, gpu, "pre_block.norm3", LATENT_DIM as u32)?;
    let (proj_weight, proj_bias) = load_linear(
        store,
        gpu,
        "pre_block.proj",
        LATENT_DIM as u32,
        LATENT_CHANNELS as u32,
        true,
    )?;
    let bias = proj_bias.ok_or_else(|| VaeError("pre_block.proj has no bias".to_string()))?;

    let normalized = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let projected = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;

    run_layer_norm(
        gpu,
        &normalized,
        hidden,
        &norm_weight,
        &norm_bias,
        rows,
        LATENT_DIM as u32,
    )?;
    let _ = length;
    run_linear(
        gpu,
        &projected,
        &normalized,
        &proj_weight,
        Some(&bias),
        rows,
        LATENT_DIM as u32,
        LATENT_CHANNELS as u32,
    )?;

    Ok(projected)
}

/// `encoder_attention_branch` (`:1058-1145`): the causal multi-head attention
/// over the 2048-wide hidden state, pooled to 32 channels and added into the
/// base. The q/k/v biases are separate tensors, and k's is the reference's
/// `zero_k_bias`.
fn encoder_attention_branch(
    store: &Store,
    gpu: &Gpu,
    hidden: &Tensor,
    base: &Tensor,
    rows: u32,
    length: u32,
) -> Result<(), VaeError> {
    let width = LATENT_DIM as u32;
    let head_dim = width / ENCODER_HEADS;
    let (norm_weight, norm_bias) = load_norm(store, gpu, "pre_block.norm1", width)?;
    let (qkv_weight, unused) =
        load_linear(store, gpu, "pre_block.attn.qkv", width, width * 3, false)?;
    if unused.is_some() {
        return fail("pre_block.attn.qkv must not carry a bias");
    }
    let q_bias = f1(store, gpu, "pre_block.attn.q_bias", width as u64)?;
    let k_bias = f1(store, gpu, "pre_block.attn.zero_k_bias", width as u64)?;
    let v_bias = f1(store, gpu, "pre_block.attn.v_bias", width as u64)?;
    let (proj_weight, proj_bias) = load_linear(
        store,
        gpu,
        "pre_block.attn.proj",
        LATENT_CHANNELS as u32,
        LATENT_CHANNELS as u32,
        true,
    )?;
    let proj_bias =
        proj_bias.ok_or_else(|| VaeError("pre_block.attn.proj has no bias".to_string()))?;

    let normalized = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let qkv = f32_alloc(gpu, rows as usize * LATENT_DIM * 3)?;
    let query = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let key = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let value = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let attended = f32_alloc(gpu, rows as usize * LATENT_DIM)?;
    let pooled = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;
    let projected = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;

    run_layer_norm(
        gpu,
        &normalized,
        hidden,
        &norm_weight,
        &norm_bias,
        rows,
        width,
    )?;
    run_linear(
        gpu,
        &qkv,
        &normalized,
        &qkv_weight,
        None,
        rows,
        width,
        width * 3,
    )?;
    device(gpu.audio_qkv_split_f32(
        &query,
        &key,
        &value,
        &qkv,
        &q_bias,
        &k_bias,
        &v_bias,
        STEREO,
        length,
        ENCODER_HEADS,
        head_dim,
    ))?;
    device(gpu.sdpa_causal_f32(
        &attended,
        &query,
        &key,
        &value,
        STEREO,
        length,
        ENCODER_HEADS,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
    ))?;
    device(gpu.audio_attention_pool_f32(
        &pooled,
        &attended,
        STEREO,
        length,
        ENCODER_HEADS,
        head_dim,
        LATENT_CHANNELS as u32,
    ))?;
    run_linear(
        gpu,
        &projected,
        &pooled,
        &proj_weight,
        Some(&proj_bias),
        rows,
        LATENT_CHANNELS as u32,
        LATENT_CHANNELS as u32,
    )?;
    device(gpu.add_scaled(base, base, &projected, 1.0, 1.0))?;

    Ok(())
}

/// `encoder_mlp` (`:1147-1221`): two LayerNorms at 32, a 32→64 gate and a
/// 32→64 linear through GEGLU, then 64→32 added into the base.
fn encoder_mlp(store: &Store, gpu: &Gpu, base: &Tensor, rows: u32) -> Result<(), VaeError> {
    let width = LATENT_CHANNELS as u32;
    let (n2w, n2b) = load_norm(store, gpu, "pre_block.norm2", width)?;
    let (nmw, nmb) = load_norm(store, gpu, "pre_block.mlp.norm", width)?;
    let (w0, b0) = load_linear(store, gpu, "pre_block.mlp.w0", width, width * 2, true)?;
    let (w1, b1) = load_linear(store, gpu, "pre_block.mlp.w1", width, width * 2, true)?;
    let (w2, b2) = load_linear(store, gpu, "pre_block.mlp.w2", width * 2, width, true)?;
    let b0 = b0.ok_or_else(|| VaeError("pre_block.mlp.w0 has no bias".to_string()))?;
    let b1 = b1.ok_or_else(|| VaeError("pre_block.mlp.w1 has no bias".to_string()))?;
    let b2 = b2.ok_or_else(|| VaeError("pre_block.mlp.w2 has no bias".to_string()))?;

    let norm2 = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;
    let normm = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;
    let gate = f32_alloc(gpu, rows as usize * LATENT_CHANNELS * 2)?;
    let linear = f32_alloc(gpu, rows as usize * LATENT_CHANNELS * 2)?;
    let geglu = f32_alloc(gpu, rows as usize * LATENT_CHANNELS * 2)?;
    let branch = f32_alloc(gpu, rows as usize * LATENT_CHANNELS)?;

    run_layer_norm(gpu, &norm2, base, &n2w, &n2b, rows, width)?;
    run_layer_norm(gpu, &normm, &norm2, &nmw, &nmb, rows, width)?;
    run_linear(gpu, &gate, &normm, &w0, Some(&b0), rows, width, width * 2)?;
    run_linear(gpu, &linear, &normm, &w1, Some(&b1), rows, width, width * 2)?;
    device(gpu.geglu_f32(&geglu, &gate, &linear, rows as usize * LATENT_CHANNELS * 2))?;
    run_linear(gpu, &branch, &geglu, &w2, Some(&b2), rows, width * 2, width)?;
    device(gpu.add_scaled(base, base, &branch, 1.0, 1.0))?;

    Ok(())
}

/// `encoder_output` (`:1223-1273`): the 1-tap mean projection, then the
/// inverse latent normalization into [32, 2, T].
fn encoder_output(
    store: &Store,
    gpu: &Gpu,
    base: &Tensor,
    length: u32,
    mean: &[f32],
    deviation: &[f32],
) -> Result<Latent, VaeError> {
    let channels = LATENT_CHANNELS as u32;
    let conv = load_plain_conv(store, gpu, "mean_proj", channels, channels, 1, 0, 1, true)?;
    let elements = STEREO as usize * length as usize * LATENT_CHANNELS;
    let latent = f32_alloc(gpu, elements)?;

    run_conv(gpu, &latent, base, &conv, length)?;

    let rows = read_f32(gpu, &latent, elements)?;

    let mut values = vec![0.0f32; elements];
    for channel in 0..LATENT_CHANNELS {
        for stereo in 0..STEREO as usize {
            for time in 0..length as usize {
                let source = (stereo * length as usize + time) * LATENT_CHANNELS + channel;
                let destination = (channel * STEREO as usize + stereo) * length as usize + time;
                values[destination] = (rows[source] - mean[channel]) / deviation[channel];
            }
        }
    }

    Ok(Latent {
        channels: LATENT_CHANNELS,
        stereo: STEREO as usize,
        length: length as usize,
        values,
    })
}

/// The whole encoder, one call: `h3_audio_vae_encode` (`:1275-1343`).
///
/// `pcm` is channel-major F32 stereo at [`SAMPLE_RATE`], at most 15 seconds
/// (`:1285`); the returned latent is [32, 2, ceil(samples / 800)].
pub fn encode(weight_directory: &Path, pcm: &[f32], samples: usize) -> Result<Latent, VaeError> {
    if samples == 0 || samples > SAMPLE_RATE as usize * 15 {
        return fail("audio VAE encoder needs 1..=480000 samples");
    }
    if pcm.len() != samples * STEREO as usize {
        return fail(format!(
            "audio VAE encoder input is {} values, expected {}",
            pcm.len(),
            samples * STEREO as usize
        ));
    }

    let (mean, deviation) = latent_normalization(weight_directory)?;
    let store = Store::open(weight_directory, "audio VAE")?;
    let gpu = device(Gpu::create())?;

    let (mut hidden, mut length) = encoder_initial(&store, &gpu, pcm, samples)?;
    let mut channels = ENCODER_DIM;

    for stage in 1..=ENCODER_STAGES {
        for residual in 0..ENCODER_RESIDUALS {
            encoder_residual(&store, &gpu, stage, residual, channels, &hidden, length)?;
        }

        let (down, next_length, next_channels) =
            encoder_downsample(&store, &gpu, stage, channels, &hidden, length)?;
        hidden = down;
        length = next_length;
        channels = next_channels;
    }

    let expected_length = samples.div_ceil(HOP_LENGTH);
    if channels != LATENT_DIM as u32 || length as usize != expected_length {
        return fail(format!(
            "AudioVAE encoder produced {channels} channels over {length} steps, \
             expected {LATENT_DIM} over {expected_length}"
        ));
    }

    let hidden = encoder_final_conv(&store, &gpu, &hidden, length)?;
    let rows = STEREO * length;
    let base = encoder_projection_branch(&store, &gpu, &hidden, rows, length)?;

    encoder_attention_branch(&store, &gpu, &hidden, &base, rows, length)?;
    encoder_mlp(&store, &gpu, &base, rows)?;

    encoder_output(&store, &gpu, &base, length, &mean, &deviation)
}
