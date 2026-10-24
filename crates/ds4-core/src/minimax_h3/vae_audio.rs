//! The released BigVGAN AudioVAE decoder (`h3_audio_vae.c:1-715`).
//!
//! Decode only: the reference's encode half serves [keyframe and reference
//! conditioning] and is a separate unit. The graph, the stage geometry, the
//! weight-norm reconstruction and the alias-free Snake activation are
//! transcribed in the reference's order, over the seven F32 primitives the
//! decoder calls (`h3d_conv1d`, `h3d_conv_transpose1d`, `h3d_weight_norm`,
//! `h3d_alias_free_snake`, `h3d_add_scaled`, `h3d_copy_f32`, `h3d_clip`).
//!
//! Layouts are the reference's: activations are [batch, time, channels] with
//! the decoder's mono weights run over a batch of two (stereo folded into the
//! batch, `h3_audio_vae.h:47-52`); the input latent is [32, 2, T] and is
//! transposed and de-normalized in `prepare_input` (`:351-379`).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use crate::mapped::MappedFile;
use crate::minimax_h3::safetensors::{self, Dtype as StDtype};

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

fn fail<T>(detail: impl Into<String>) -> Result<T, VaeError> {
    Err(VaeError(detail.into()))
}

fn device<T>(result: Result<T, String>) -> Result<T, VaeError> {
    result.map_err(|detail| VaeError(format!("{DEVICE_ERROR}: {detail}")))
}

/// One F32 weight, resolved to its shard and byte range.
#[derive(Clone, Copy)]
struct Site {
    shard: usize,
    begin: usize,
    bytes: usize,
}

/// The audio VAE's checkpoint: one shard in practice, indexed by name and
/// mapped on first use, like `h3_weight_store_open` (`h3_weights.c`).
struct Store {
    shards: Vec<PathBuf>,
    sites: HashMap<String, Site>,
    maps: Vec<OnceLock<MappedFile>>,
}

impl Store {
    fn open(dir: &Path) -> Result<Store, VaeError> {
        let store = safetensors::open_dir(dir)
            .map_err(|error| VaeError(format!("audio VAE reader: {error:?}")))?;
        let mut shards = Vec::new();
        let mut sites = HashMap::new();
        for (index, shard) in store.shards().iter().enumerate() {
            shards.push(shard.path.clone());
            for tensor in &shard.tensors {
                if tensor.dtype != StDtype::F32 {
                    return fail(format!("audio VAE tensor {} is not F32", tensor.name));
                }
                let begin = usize::try_from(tensor.file_offset)
                    .map_err(|_| VaeError("audio VAE offset too wide".into()))?;
                let bytes = usize::try_from(tensor.bytes())
                    .map_err(|_| VaeError("audio VAE range too wide".into()))?;
                sites.insert(
                    tensor.name.clone(),
                    Site {
                        shard: index,
                        begin,
                        bytes,
                    },
                );
            }
        }
        let maps = (0..shards.len()).map(|_| OnceLock::new()).collect();

        Ok(Store {
            shards,
            sites,
            maps,
        })
    }

    fn map(&self, shard: usize) -> Result<&MappedFile, VaeError> {
        let cell = &self.maps[shard];
        if let Some(map) = cell.get() {
            return Ok(map);
        }
        let map = MappedFile::open_ro(&self.shards[shard])
            .map_err(|error| VaeError(format!("map audio VAE shard: {error}")))?;
        let _ = cell.set(map);
        cell.get()
            .ok_or_else(|| VaeError("audio VAE shard map lost".into()))
    }

    /// Upload `name` as F32 after checking its exact shape, the reference's
    /// `h3_weight_load_f32(..., ndim, shape, ...)` (`h3_audio_vae.c:128-133`).
    fn load(&self, gpu: &Gpu, name: &str, shape: &[u64], error: &str) -> Result<Tensor, VaeError> {
        let site = self
            .sites
            .get(name)
            .ok_or_else(|| VaeError(format!("audio VAE is missing {name}")))?;
        let elements = shape.iter().product::<u64>();
        if elements * 4 != site.bytes as u64 {
            return fail(format!(
                "{name}: plan {shape:?} needs {} bytes, checkpoint has {}",
                elements * 4,
                site.bytes
            ));
        }
        let map = self.map(site.shard)?;
        let bytes = map
            .as_slice()
            .get(site.begin..site.begin + site.bytes)
            .ok_or_else(|| VaeError(format!("{name}: range outside its shard")))?;

        let tensor = device(gpu.alloc(Dtype::F32, elements as usize))?;
        device(gpu.upload(&tensor, bytes))
            .map_err(|detail| VaeError(format!("{error}: {detail}")))?;
        Ok(tensor)
    }
}

fn f1(store: &Store, gpu: &Gpu, name: &str, width: u64) -> Result<Tensor, VaeError> {
    store.load(gpu, name, &[width], name)
}

fn f3(
    store: &Store,
    gpu: &Gpu,
    name: &str,
    first: u64,
    second: u64,
    third: u64,
) -> Result<Tensor, VaeError> {
    store.load(gpu, name, &[first, second, third], name)
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
    let store = Store::open(weight_directory)?;
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
