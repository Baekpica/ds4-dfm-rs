//! The H3 latent upscaler (`LatentResizer3D`, `minimax_h3_latent_upscaler_3d_conv_v1`).
//!
//! This is the learned resizer the two-stage recipe upscales a draft with, and
//! it is *not* a latent-space translator: it takes H3's own 24-channel latent
//! and returns the same 24-channel latent at twice the spatial size, so no
//! adapter sits between two model families. That is the whole difference from
//! the LTX stage this port used to carry, which had to convert an H3 latent into
//! an LTX one before an LTX refiner could read it.
//!
//! Geometry (`config.json`, verified against the checkpoint's 322 tensors):
//! `conv_in` 24 -> 512, twelve `ResBlockEmb3D` alternating with six
//! `TemporalConv`, a trilinear resize in the middle, twelve more of each,
//! `norm_out`, `conv_out` 512 -> 24. GroupNorm has 32 groups and the embedding
//! is a 1 -> 64 -> 64 MLP over `scale - 1`.
//!
//! The upscale is the trilinear resize; the convolutions either side of it
//! re-synthesize the detail the resize cannot invent. TemporalConv is depthwise
//! over time with a five-tap kernel and zero padding, so a frame's context is
//! two frames either side and temporal chunking would be exact.
//!
//! Latents are normalized per channel before the network and denormalized after
//! (`latents_mean` / `latents_std`, the released constants), exactly as the
//! node does. Tensors go to the device channels-last, `[t, h, w, c]`, which is
//! the layout `h3d_conv3d` takes; the caller sees channel-major
//! `[24, t, h, w]`, the layout the VAE and the engine use.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::vae_store::Store;

/// The released latent width, which is also the output width.
pub const LATENT_CHANNELS: usize = 24;
/// The resizer's internal width.
pub const WIDTH: u32 = 512;
pub const GROUPS: u32 = 32;
/// `nn.GroupNorm`'s default epsilon, which the node leaves alone.
pub const EPSILON: f32 = 1e-5;
pub const TEMPORAL_KERNEL: u32 = 5;
/// Output rows per im2col tile: `8192 * 512 * 27 * 4` bytes of scratch, which
/// is the largest tile this resizer asks for.
pub const SCRATCH_TILE: u32 = 8192;
const RES_BLOCKS: usize = 12;
/// Why this pairing: every block list in the checkpoint is `ResBlockEmb3D`
/// followed by `TemporalConv` when its index is `0 mod 3` and the next is not,
/// which is the `temporal_every = 2` layout: temporal blocks at 1, 4, 7, 10, 13
/// and 16 of eighteen.
pub const BLOCKS: usize = (RES_BLOCKS / 2) * 3;
const DEVICE_ERROR: &str = "latent upscaler device";

/// `latents_mean` / `latents_std` from the model card, per latent channel.
pub const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    0.858_090_34,
    -0.960_659_15,
    1.066_164,
    -0.509_032_55,
    -0.272_758_2,
    -1.367_541_2,
    -0.255_325_5,
    -0.269_075_54,
    -0.537_684_1,
    -0.046_409_73,
    0.665_737_03,
    0.196_901_28,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_341_99,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_44,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];

pub const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_84,
    0.799_680_95,
    0.449_889,
    0.719_739_97,
    0.693_629_3,
    2.961_095_2,
    2.769_42,
    3.049_618_5,
    2.108_805_3,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];

/// A resized latent, channel-major and denormalized: `[24, time, height, width]`.
pub struct Latent {
    pub time: usize,
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct UpscaleError(String);

impl fmt::Display for UpscaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UpscaleError {}

impl From<String> for UpscaleError {
    fn from(detail: String) -> UpscaleError {
        UpscaleError(detail)
    }
}

fn fail<T>(detail: impl Into<String>) -> Result<T, UpscaleError> {
    Err(UpscaleError(detail.into()))
}

fn device<T>(result: Result<T, String>) -> Result<T, UpscaleError> {
    result.map_err(|detail| UpscaleError(format!("{DEVICE_ERROR}: {detail}")))
}

fn f32_alloc(gpu: &Gpu, elements: usize) -> Result<Tensor, UpscaleError> {
    device(gpu.alloc(Dtype::F32, elements))
}

fn upload_f32(gpu: &Gpu, tensor: &Tensor, values: &[f32]) -> Result<(), UpscaleError> {
    // SAFETY: f32 has no padding or invalid bit patterns, so reinterpreting the
    // slice as bytes is sound; the bytes are only read.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    device(gpu.upload(tensor, bytes))
}

fn read_f32(gpu: &Gpu, tensor: &Tensor, elements: usize) -> Result<Vec<f32>, UpscaleError> {
    let mut bytes = vec![0u8; elements * 4];
    device(gpu.read(tensor, &mut bytes))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// IEEE half to f32. The checkpoint stores F16, and the reference loads it into
/// F32 before any arithmetic (`load_model`'s `to(dtype)`), so this is the same
/// widening with no rounding of its own: normals and subnormals are both exact.
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        let value = mantissa as f32 * 2.0f32.powi(-24);
        if sign == 1 {
            -value
        } else {
            value
        }
    } else if exponent == 0x1F {
        if mantissa == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(((sign as u32) << 31) | ((exponent as u32 + 112) << 23) | (mantissa << 13))
    }
}

/// One normalization: GroupNorm's per-channel weight and bias.
struct Norm {
    weight: Tensor,
    bias: Tensor,
}

/// One convolution, weights `[out, in, kt, ky, kx]` as the checkpoint has them.
struct Conv {
    weight: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
}

/// `ResBlockEmb3D` (`minimax_h3_latent_upscaler_3d.py:160-188`): GroupNorm and
/// SiLU, a 3x3x3 convolution, the embedding's scale and shift, another
/// GroupNorm and SiLU, a zero-initialized 3x3x3 convolution, and an identity
/// skip. `out_channels == channels`, so the skip is the input itself.
struct ResBlock {
    norm_in: Norm,
    conv_in: Conv,
    /// `emb_layers.1`, `[2 * 512, 64]`, kept on the host: it is the only part of
    /// the block that reads the shared 64-wide embedding, and its output is
    /// 1024 values per call.
    emb: Vec<f32>,
    emb_bias: Vec<f32>,
    norm_out: Norm,
    conv_out: Conv,
}

/// `TemporalConv` (`:190-207`): GroupNorm and SiLU, a depthwise five-tap
/// convolution over time, a 1x1x1 convolution, and a residual.
struct TemporalBlock {
    norm: Norm,
    depthwise: Conv,
    pointwise: Conv,
}

enum Block {
    Res(ResBlock),
    Temporal(TemporalBlock),
}

/// The loaded resizer.
pub struct Upscaler {
    gpu: Gpu,
    conv_in: Conv,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: Norm,
    conv_out: Conv,
    /// `embed` is `Linear(1, 64)`, SiLU, `Linear(64, 64)`; 64 values wide, so it
    /// is evaluated on the host once per call and uploaded as two vectors.
    embed_in: (Vec<f32>, Vec<f32>),
    embed_out: (Vec<f32>, Vec<f32>),
    /// `H3_UPSCALER_GEMM=tf32` asks for the FAST-TF32 GEMM, which is the
    /// reference implementation's own precision class: it computes in F16, and
    /// a TF32 product carries the same ten mantissa bits. `fp32` is the arm to
    /// compare against, and the gate's default.
    tf32: bool,
}

/// One block's position: `true` is a `TemporalConv`.
pub fn temporal_at(index: usize) -> bool {
    (index * 2) % 3 == 2
}

/// `[out, in, kt, ky, kx]` to `[out, kt, ky, kx, in]`: the column order the
/// im2col emits, so the copy out of the padded volume is coalesced and the
/// GEMM reads the weight without a transpose. Done once, at load.
fn permute_kernel(
    weight: &[f32],
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let taps = kernel * kernel * kernel;
    let mut permuted = vec![0.0f32; weight.len()];
    for out in 0..output_channels {
        for inside in 0..input_channels {
            for tap in 0..taps {
                permuted[out * taps * input_channels + tap * input_channels + inside] =
                    weight[(out * input_channels + inside) * taps + tap];
            }
        }
    }
    permuted
}

impl Upscaler {
    /// Open the released checkpoint in `dir`. The store is the video VAE's: a
    /// directory of shards with exact shape checks per tensor.
    pub fn open(dir: &Path, gpu: Gpu) -> Result<Upscaler, UpscaleError> {
        let tf32 = std::env::var("H3_UPSCALER_GEMM")
            .map(|value| value.starts_with('t') || value.starts_with('T'))
            .unwrap_or(true);
        Self::open_with(dir, gpu, tf32)
    }

    /// The same load with the GEMM arm stated instead of read from the
    /// environment, so a gate can hold it fixed.
    pub fn open_with(dir: &Path, gpu: Gpu, tf32: bool) -> Result<Upscaler, UpscaleError> {
        let store = Store::open(dir, "latent upscaler")?;
        let host = |name: &str, shape: &[u64]| -> Result<Vec<f32>, UpscaleError> {
            let bytes = store.raw(name, shape)?;
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| half_to_f32(u16::from_le_bytes(*word)))
                .collect())
        };
        let load = |name: &str, shape: &[u64]| -> Result<Tensor, UpscaleError> {
            let values = host(name, shape)?;
            let tensor = f32_alloc(&gpu, values.len())?;
            upload_f32(&gpu, &tensor, &values)?;
            Ok(tensor)
        };

        let norm = |prefix: &str| -> Result<Norm, UpscaleError> {
            Ok(Norm {
                weight: load(&format!("{prefix}.weight"), &[WIDTH as u64])?,
                bias: load(&format!("{prefix}.bias"), &[WIDTH as u64])?,
            })
        };
        let conv =
            |prefix: &str, input: u32, output: u32, kernel: u32| -> Result<Conv, UpscaleError> {
                let name = format!("{prefix}.weight");
                let shape = [
                    output as u64,
                    input as u64,
                    kernel as u64,
                    kernel as u64,
                    kernel as u64,
                ];
                let mut values = host(&name, &shape)?;
                if kernel == 3 {
                    values = permute_kernel(&values, output as usize, input as usize, 3);
                }
                let weight = f32_alloc(&gpu, values.len())?;
                upload_f32(&gpu, &weight, &values)?;
                Ok(Conv {
                    weight,
                    bias: load(&format!("{prefix}.bias"), &[output as u64])?,
                    input_channels: input,
                    output_channels: output,
                    kernel,
                })
            };

        let conv_in = conv("conv_in", LATENT_CHANNELS as u32, WIDTH, 3)?;
        let conv_out = conv("conv_out", WIDTH, LATENT_CHANNELS as u32, 3)?;
        let norm_out = norm("norm_out")?;

        let build = |side: &str| -> Result<Vec<Block>, UpscaleError> {
            let mut blocks = Vec::with_capacity(BLOCKS);
            for index in 0..BLOCKS {
                let prefix = format!("{side}.{index}");
                if temporal_at(index) {
                    blocks.push(Block::Temporal(TemporalBlock {
                        norm: norm(&format!("{prefix}.norm"))?,
                        depthwise: load(
                            &format!("{prefix}.dwconv.weight"),
                            &[WIDTH as u64, 1, TEMPORAL_KERNEL as u64, 1, 1],
                        )
                        .and_then(|weight| {
                            Ok(Conv {
                                weight,
                                bias: load(&format!("{prefix}.dwconv.bias"), &[WIDTH as u64])?,
                                input_channels: WIDTH,
                                output_channels: WIDTH,
                                kernel: TEMPORAL_KERNEL,
                            })
                        })?,
                        pointwise: conv(&format!("{prefix}.pwconv"), WIDTH, WIDTH, 1)?,
                    }));
                } else {
                    blocks.push(Block::Res(ResBlock {
                        norm_in: norm(&format!("{prefix}.in_layers.0"))?,
                        conv_in: conv(&format!("{prefix}.in_layers.2"), WIDTH, WIDTH, 3)?,
                        emb: host(
                            &format!("{prefix}.emb_layers.1.weight"),
                            &[2 * WIDTH as u64, 64],
                        )?,
                        emb_bias: host(
                            &format!("{prefix}.emb_layers.1.bias"),
                            &[2 * WIDTH as u64],
                        )?,
                        norm_out: norm(&format!("{prefix}.out_norm"))?,
                        conv_out: conv(&format!("{prefix}.out_layers.2"), WIDTH, WIDTH, 3)?,
                    }));
                }
            }
            Ok(blocks)
        };

        let in_blocks = build("in_blocks")?;
        let out_blocks = build("out_blocks")?;

        let embed_in = (
            host("embed.0.weight", &[64, 1])?,
            host("embed.0.bias", &[64])?,
        );
        let embed_out = (
            host("embed.2.weight", &[64, 64])?,
            host("embed.2.bias", &[64])?,
        );

        Ok(Upscaler {
            gpu,
            tf32,
            conv_in,
            in_blocks,
            out_blocks,
            norm_out,
            conv_out,
            embed_in,
            embed_out,
        })
    }

    /// The embedding for a scale: `Linear(1, 64)`, SiLU, `Linear(64, 64)` over
    /// `scale - 1` on the host in F32, one `mul_add` per term in index order.
    fn embed(&self, scale: f32) -> Vec<f32> {
        let input = scale - 1.0;
        let mut hidden = self.embed_in.1.clone();
        for (row, value) in hidden.iter_mut().enumerate() {
            *value = self.embed_in.0[row].mul_add(input, *value);
            *value = *value / (1.0 + (-*value).exp());
        }
        let mut out = self.embed_out.1.clone();
        for (row, value) in out.iter_mut().enumerate() {
            let mut sum = *value;
            for (column, hidden_value) in hidden.iter().enumerate() {
                sum = self.embed_out.0[row * 64 + column].mul_add(*hidden_value, sum);
            }
            *value = sum;
        }
        out
    }

    /// Sample the convolution's phases with device events. `timing_read`
    /// returns the accumulated device seconds as `[im2col, gemm, bias]`, and
    /// `timing_reset` clears them: a host timer around `upscale` cannot see
    /// this split, because the native entries only report launch errors.
    pub fn timing(&self, enable: bool) -> bool {
        self.gpu.conv3d_timing(enable)
    }

    /// The accumulated phase seconds, over every `conv3d_gemm` call.
    pub fn timing_read(&self) -> [f64; 3] {
        Gpu::conv3d_timing_read()
    }

    /// Clear the phase accumulators.
    pub fn timing_reset(&self) {
        Gpu::conv3d_timing_reset()
    }

    /// Resize a channel-major, normalized latent `[24, time, height, width]` to
    /// `out_height` by `out_width`, keeping the temporal extent.
    pub fn upscale(
        &self,
        latent: &[f32],
        time: usize,
        height: usize,
        width: usize,
        out_height: usize,
        out_width: usize,
    ) -> Result<Latent, UpscaleError> {
        if time == 0 || height == 0 || width == 0 || out_height == 0 || out_width == 0 {
            return fail("upscale needs a non-empty latent");
        }
        if out_height < height || out_width < width {
            return fail(format!(
                "the resizer only upscales: {width}x{height} -> {out_width}x{out_height}"
            ));
        }
        let expected = LATENT_CHANNELS * time * height * width;
        if latent.len() != expected {
            return fail(format!(
                "latent has {} values, {expected} expected",
                latent.len()
            ));
        }
        if out_height == height && out_width == width {
            return Ok(Latent {
                time,
                height,
                width,
                values: latent.to_vec(),
            });
        }

        let voxels = time * height * width;
        let count = voxels * WIDTH as usize;
        let scale = (out_width as f32 / width as f32 + out_height as f32 / height as f32) / 2.0;
        let embed = self.embed(scale);
        let scratch = f32_alloc(&self.gpu, SCRATCH_TILE as usize * WIDTH as usize * 27)?;

        let normalized: Vec<f32> = latent
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let channel = index / voxels;
                (value - LATENTS_MEAN[channel]) / LATENTS_STD[channel]
            })
            .collect();
        let mut time_major = vec![0.0f32; normalized.len()];
        for channel in 0..LATENT_CHANNELS {
            for voxel in 0..voxels {
                time_major[voxel * LATENT_CHANNELS + channel] =
                    normalized[channel * voxels + voxel];
            }
        }

        let input = f32_alloc(&self.gpu, normalized.len())?;
        upload_f32(&self.gpu, &input, &time_major)?;

        let mut current = self.conv(&self.conv_in, &input, time, height, width, &scratch)?;

        for block in &self.in_blocks {
            current = match block {
                Block::Res(res) => {
                    self.res_block(res, &current, &embed, time, height, width, count, &scratch)?
                }
                Block::Temporal(temporal) => {
                    self.temporal_block(temporal, &current, time, height, width, voxels)?
                }
            };
        }

        let resized_elements = time * out_height * out_width * WIDTH as usize;
        let resized = f32_alloc(&self.gpu, resized_elements)?;
        device(self.gpu.resize_hw(
            &resized,
            &current,
            1,
            time as u32,
            height as u32,
            width as u32,
            out_height as u32,
            out_width as u32,
            WIDTH,
        ))?;
        drop(input);
        drop(current);

        let mut current = resized;
        let resized_count = time * out_height * out_width * WIDTH as usize;
        for block in &self.out_blocks {
            current = match block {
                Block::Res(res) => self.res_block(
                    res,
                    &current,
                    &embed,
                    time,
                    out_height,
                    out_width,
                    resized_count,
                    &scratch,
                )?,
                Block::Temporal(temporal) => self.temporal_block(
                    temporal,
                    &current,
                    time,
                    out_height,
                    out_width,
                    time * out_height * out_width,
                )?,
            };
        }

        let normed = f32_alloc(&self.gpu, resized_count)?;
        device(self.gpu.group_norm(
            &normed,
            &current,
            &self.norm_out.weight,
            &self.norm_out.bias,
            1,
            time as u32,
            out_height as u32,
            out_width as u32,
            WIDTH,
            GROUPS,
            EPSILON,
        ))?;
        let activated = f32_alloc(&self.gpu, resized_count)?;
        device(self.gpu.silu_f32(&activated, &normed))?;
        let output = self.conv(
            &self.conv_out,
            &activated,
            time,
            out_height,
            out_width,
            &scratch,
        )?;

        let out_voxels = time * out_height * out_width;
        let time_major_out = read_f32(&self.gpu, &output, out_voxels * LATENT_CHANNELS)?;
        let mut values = vec![0.0f32; time_major_out.len()];
        for channel in 0..LATENT_CHANNELS {
            for voxel in 0..out_voxels {
                values[channel * out_voxels + voxel] =
                    time_major_out[voxel * LATENT_CHANNELS + channel];
            }
        }
        for (index, value) in values.iter_mut().enumerate() {
            let channel = index / out_voxels;
            *value = *value * LATENTS_STD[channel] + LATENTS_MEAN[channel];
        }

        Ok(Latent {
            time,
            height: out_height,
            width: out_width,
            values,
        })
    }

    /// `scale, shift = torch.chunk(emb_layers(embed), 2, dim=1)`, then `1 +
    /// scale` folded in so the device affine is one multiply and one add. Each
    /// block has its own `emb_layers`, so this is per block; the 64-wide
    /// `embed` is the only part the blocks share.
    fn modulation(&self, block: &ResBlock, embed: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let width = WIDTH as usize;
        let mut projected = block.emb_bias.clone();
        for (row, value) in projected.iter_mut().enumerate() {
            for (column, hidden) in embed.iter().enumerate() {
                *value = block.emb[row * 64 + column].mul_add(*hidden, *value);
            }
        }
        let mut scale = vec![0.0f32; width];
        let mut shift = vec![0.0f32; width];
        for channel in 0..width {
            scale[channel] = 1.0 + projected[channel];
            shift[channel] = projected[width + channel];
        }
        (scale, shift)
    }

    /// A convolution over an already channels-last tensor, with the padding the
    /// module declares (`kernel / 2` on every axis).
    fn conv(
        &self,
        conv: &Conv,
        input: &Tensor,
        time: usize,
        height: usize,
        width: usize,
        scratch: &Tensor,
    ) -> Result<Tensor, UpscaleError> {
        let pad = conv.kernel / 2;
        if pad == 0 {
            let elements = time * height * width * conv.output_channels as usize;
            let output = f32_alloc(&self.gpu, elements)?;
            device(self.gpu.vae_linear(
                &output,
                input,
                &conv.weight,
                Some(&conv.bias),
                (time * height * width) as u32,
                conv.input_channels,
                conv.output_channels,
            ))?;
            return Ok(output);
        }
        let padded_elements = (time + 2 * pad as usize)
            * (height + 2 * pad as usize)
            * (width + 2 * pad as usize)
            * conv.input_channels as usize;
        let padded = f32_alloc(&self.gpu, padded_elements)?;
        device(self.gpu.zero_pad3d(
            &padded,
            input,
            1,
            time as u32,
            height as u32,
            width as u32,
            conv.input_channels,
            [pad, pad, pad],
        ))?;
        let output_elements = time * height * width * conv.output_channels as usize;
        let output = f32_alloc(&self.gpu, output_elements)?;
        device(self.gpu.conv3d_gemm(
            &output,
            &padded,
            &conv.weight,
            &conv.bias,
            (time + 2 * pad as usize) as u32,
            (height + 2 * pad as usize) as u32,
            (width + 2 * pad as usize) as u32,
            conv.input_channels,
            conv.output_channels,
            conv.kernel,
            SCRATCH_TILE,
            scratch,
            self.tf32,
        ))?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn res_block(
        &self,
        block: &ResBlock,
        input: &Tensor,
        embed: &[f32],
        time: usize,
        height: usize,
        width: usize,
        count: usize,
        scratch: &Tensor,
    ) -> Result<Tensor, UpscaleError> {
        let (scale_values, shift_values) = self.modulation(block, embed);
        let mod_scale = f32_alloc(&self.gpu, scale_values.len())?;
        upload_f32(&self.gpu, &mod_scale, &scale_values)?;
        let mod_shift = f32_alloc(&self.gpu, shift_values.len())?;
        upload_f32(&self.gpu, &mod_shift, &shift_values)?;
        let hidden = f32_alloc(&self.gpu, count)?;
        // GroupNorm over the whole volume, then SiLU in place: `group_norm_silu`
        // reduces per temporal slice, which is the encoder's shape (T = 1) and
        // not this layer's.
        device(self.gpu.group_norm(
            &hidden,
            input,
            &block.norm_in.weight,
            &block.norm_in.bias,
            1,
            time as u32,
            height as u32,
            width as u32,
            WIDTH,
            GROUPS,
            EPSILON,
        ))?;
        device(self.gpu.silu_f32(&hidden, &hidden))?;
        let convolved = self.conv(&block.conv_in, &hidden, time, height, width, scratch)?;
        drop(hidden);
        let normalized = f32_alloc(&self.gpu, count)?;
        device(self.gpu.group_norm(
            &normalized,
            &convolved,
            &block.norm_out.weight,
            &block.norm_out.bias,
            1,
            time as u32,
            height as u32,
            width as u32,
            WIDTH,
            GROUPS,
            EPSILON,
        ))?;
        drop(convolved);
        let modulated = f32_alloc(&self.gpu, count)?;
        device(self.gpu.channel_affine(
            &modulated,
            &normalized,
            &mod_scale,
            &mod_shift,
            WIDTH,
            count,
        ))?;
        drop(normalized);
        let activated = f32_alloc(&self.gpu, count)?;
        device(self.gpu.silu_f32(&activated, &modulated))?;
        drop(modulated);
        let residual = self.conv(&block.conv_out, &activated, time, height, width, scratch)?;
        let output = f32_alloc(&self.gpu, count)?;
        device(self.gpu.add_f32(&output, input, &residual))?;
        Ok(output)
    }

    fn temporal_block(
        &self,
        block: &TemporalBlock,
        input: &Tensor,
        time: usize,
        height: usize,
        width: usize,
        voxels: usize,
    ) -> Result<Tensor, UpscaleError> {
        let count = voxels * WIDTH as usize;
        let hidden = f32_alloc(&self.gpu, count)?;
        device(self.gpu.group_norm(
            &hidden,
            input,
            &block.norm.weight,
            &block.norm.bias,
            1,
            time as u32,
            height as u32,
            width as u32,
            WIDTH,
            GROUPS,
            EPSILON,
        ))?;
        device(self.gpu.silu_f32(&hidden, &hidden))?;
        let depthwise = f32_alloc(&self.gpu, count)?;
        device(self.gpu.temporal_dwconv(
            &depthwise,
            &hidden,
            &block.depthwise.weight,
            &block.depthwise.bias,
            1,
            time as u32,
            height as u32,
            width as u32,
            WIDTH,
            TEMPORAL_KERNEL,
        ))?;
        drop(hidden);
        let pointwise = f32_alloc(&self.gpu, count)?;
        device(self.gpu.vae_linear(
            &pointwise,
            &depthwise,
            &block.pointwise.weight,
            Some(&block.pointwise.bias),
            voxels as u32,
            WIDTH,
            WIDTH,
        ))?;
        let output = f32_alloc(&self.gpu, count)?;
        device(self.gpu.add_f32(&output, input, &pointwise))?;
        Ok(output)
    }
}
