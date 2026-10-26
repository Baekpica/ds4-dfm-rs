//! The released video VAE encoder (`h3_video_encoder.c:1-814`).
//!
//! Encode only: the decoder is [`super::vae_video`]. The graph is the
//! reference's 3D convolutional encoder over F32, `[3,T,H,W]` RGB in `[0,1]`
//! to a channel-major `[24,t,h,w]` latent (`h3_video_encoder.h:8-23`), and it
//! is the path anchors and Ref2VA references go through.
//!
//! Geometry (`:13-27`): 24 latent channels, 48 output moments, six levels of
//! two residual blocks, 32 normalization groups, a 16x spatial ratio, released
//! 256 px tiles with a 64 px minimum overlap, level channels
//! `{128,256,256,512,512,1024}`, space strides `{2,2,2,2,1,1}` and time
//! strides `{1,2,2,1,1,1}`. A tile is normalized with the image statistics,
//! reordered to time-major `[T,H,W,C]` on the host (`:487-507`), run through
//! the levels, then the 48 moments are reordered back to channel-major and
//! normalized with `latents_mean`/`latents_std` (`:571-589`).
//!
//! Two rules that are the encoder's own and not the decoder's:
//!
//! - The tile plan (`tile_axis_build`, `:611-651`) has no tile-size search.
//!   One tile when the extent is at most 256; otherwise the count is raised
//!   until `256*count - 64*(count-1)` reaches the extent, every gap starts at
//!   64 px, and the leftover is handed out 16 px at a time, first gap first.
//! - The stitched overlap (`:676-743`) blends the incoming tile's first rows
//!   with the previous tile's tail at weight `y / overlap`, and the
//!   destination is the tile start over 16.
//!
//! The moments are read back and reordered on the host: the reference reads
//! the whole tile (`:571-589`) and so does this port, which keeps the device
//! graph free of a layout kernel the reference does not have.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::vae_store::{f1, Store};

pub const LATENT_CHANNELS: usize = 24;
const RGB_CHANNELS: usize = 3;
const MOMENT_CHANNELS: usize = 48;
const LEVELS: usize = 6;
const BLOCKS: usize = 2;
const GROUPS: u32 = 32;
const SPATIAL_RATIO: usize = 16;
const TILE_PIXELS: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
/// The encoder's own epsilon (`h3_video_encoder.c:521`, `:554`).
const EPSILON: f32 = 1e-6;
const LEVEL_CHANNELS: [u32; LEVELS] = [128, 256, 256, 512, 512, 1024];
const SPACE_STRIDES: [u32; LEVELS] = [2, 2, 2, 2, 1, 1];
const TIME_STRIDES: [u32; LEVELS] = [1, 2, 2, 1, 1, 1];
/// The reference refuses an extent below this or not a multiple of 16.
const MIN_EXTENT: usize = 32;
/// Input normalization (`:487-492`), the image statistics the decoder undoes.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const DEVICE_ERROR: &str = "video VAE encoder device";

/// `h3_video_latent` (`h3_video_encoder.h:8-13`).
pub struct Latent {
    pub time: usize,
    pub height: usize,
    pub width: usize,
    /// Channel-major normalized F32: `[24, time, height, width]`.
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct EncoderError(String);

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EncoderError {}

impl From<String> for EncoderError {
    fn from(detail: String) -> EncoderError {
        EncoderError(detail)
    }
}

fn fail<T>(detail: impl Into<String>) -> Result<T, EncoderError> {
    Err(EncoderError(detail.into()))
}

fn device<T>(result: Result<T, String>) -> Result<T, EncoderError> {
    result.map_err(|detail| EncoderError(format!("{DEVICE_ERROR}: {detail}")))
}

fn f32_alloc(gpu: &Gpu, elements: usize) -> Result<Tensor, EncoderError> {
    device(gpu.alloc(Dtype::F32, elements))
}

/// The reference streams the normalized pixels as one F32 buffer, so the
/// upload is the slice's own bytes.
fn upload_f32(gpu: &Gpu, tensor: &Tensor, values: &[f32]) -> Result<(), EncoderError> {
    // SAFETY: f32 has no padding or invalid bit patterns, so reinterpreting
    // the slice as bytes is sound; the bytes are only read.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    device(gpu.upload(tensor, bytes))
}

fn read_f32(gpu: &Gpu, tensor: &Tensor, elements: usize) -> Result<Vec<f32>, EncoderError> {
    let mut bytes = vec![0u8; elements * 4];
    device(gpu.read(tensor, &mut bytes))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// `h3_host.c:30-32`: the temporal compression every caller sizes its spans
/// with, `(frames + 3) / 4`.
pub fn latent_time(frames: usize) -> usize {
    frames.div_ceil(4)
}

/// The spatial tile plan for one axis (`tile_axis_build`, `:611-651`).
struct TileAxis {
    starts: Vec<usize>,
    /// One entry fewer than `starts`: the last tile has no following overlap.
    overlaps: Vec<usize>,
    length: usize,
}

impl TileAxis {
    fn count(&self) -> usize {
        self.starts.len()
    }

    fn extent(&self) -> usize {
        self.starts[self.starts.len() - 1] + self.length
    }
}

/// `tile_axis_build` (`:611-651`). Note the encoder fixes the tile at 256 px;
/// the decoder's tile-size search is a different rule.
fn tile_axis(extent: usize) -> Result<TileAxis, EncoderError> {
    if extent < MIN_EXTENT || !extent.is_multiple_of(SPATIAL_RATIO) {
        return fail(format!(
            "visual encoder extent must be a multiple of {SPATIAL_RATIO}"
        ));
    }
    if extent <= TILE_PIXELS {
        return Ok(TileAxis {
            starts: vec![0],
            overlaps: Vec::new(),
            length: extent,
        });
    }

    let mut count = extent.div_ceil(TILE_PIXELS);
    while TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) < extent {
        count += 1;
    }

    let mut overlaps = vec![TILE_OVERLAP_MIN; count - 1];
    let covered = TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1);
    for unit in 0..(covered - extent) / SPATIAL_RATIO {
        overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    }
    let mut starts = Vec::with_capacity(count);
    starts.push(0);
    for index in 1..count {
        starts.push(starts[index - 1] + TILE_PIXELS - overlaps[index - 1]);
    }

    Ok(TileAxis {
        starts,
        overlaps,
        length: TILE_PIXELS,
    })
}

/// `extract_pixel_tile` (`:653-674`): a channel-major `[3, T, tile_h, tile_w]`
/// slab copied row by row out of the full image.
#[allow(clippy::too_many_arguments)]
fn extract_pixel_tile(
    pixels: &[f32],
    frames: usize,
    full_h: usize,
    full_w: usize,
    start_y: usize,
    start_x: usize,
    tile_h: usize,
    tile_w: usize,
) -> Vec<f32> {
    let mut tile = vec![0.0f32; RGB_CHANNELS * frames * tile_h * tile_w];

    for channel in 0..RGB_CHANNELS {
        for time in 0..frames {
            for y in 0..tile_h {
                let source = ((channel * frames + time) * full_h + start_y + y) * full_w + start_x;
                let destination = ((channel * frames + time) * tile_h + y) * tile_w;
                tile[destination..destination + tile_w]
                    .copy_from_slice(&pixels[source..source + tile_w]);
            }
        }
    }
    tile
}

/// One convolution and the geometry it was loaded with (`encoder_conv`,
/// `:35-50`). `pads` is `[depth_front, height_before, height_after,
/// width_before, width_after]`, the reference's causal padding order.
struct Conv {
    weight: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    stride: [u32; 3],
    pads: [u32; 5],
}

/// One group-norm pair (`encoder_norm`, `:52-55`).
struct Norm {
    weight: Tensor,
    bias: Tensor,
}

/// `encoder_block` (`:57-64`).
struct Block {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// `encoder_level` (`:66-70`).
struct Level {
    blocks: Vec<Block>,
    downsample: Option<Conv>,
}

/// The loaded encoder, reusable across tiles and clips.
pub struct Encoder {
    gpu: Gpu,
    conv_in: Conv,
    levels: Vec<Level>,
    norm_out: Norm,
    conv_out: Conv,
    quant: Conv,
    mean: Vec<f32>,
    deviation: Vec<f32>,
}

fn load_norm(store: &Store, gpu: &Gpu, prefix: &str, channels: u32) -> Result<Norm, EncoderError> {
    Ok(Norm {
        weight: f1(store, gpu, &format!("{prefix}.weight"), channels as u64)?,
        bias: f1(store, gpu, &format!("{prefix}.bias"), channels as u64)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_conv(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    stride: [u32; 3],
    pads: [u32; 5],
) -> Result<Conv, EncoderError> {
    let weight = store.load(
        gpu,
        &format!("{prefix}.weight"),
        &[
            output_channels as u64,
            input_channels as u64,
            kernel as u64,
            kernel as u64,
            kernel as u64,
        ],
        &format!("{prefix}.weight"),
    )?;

    Ok(Conv {
        weight,
        bias: f1(
            store,
            gpu,
            &format!("{prefix}.bias"),
            output_channels as u64,
        )?,
        input_channels,
        output_channels,
        kernel,
        stride,
        pads,
    })
}

/// `load_weights` (`:286-346`): the block at index 0 takes the previous
/// level's channel count, the second block its own level's.
fn load_block(
    store: &Store,
    gpu: &Gpu,
    prefix: &str,
    previous_channels: u32,
    channels: u32,
    first: bool,
) -> Result<Block, EncoderError> {
    let input_channels = if first { previous_channels } else { channels };
    let centred = [2, 1, 1, 1, 1];
    let stride = [1, 1, 1];

    let shortcut = (input_channels != channels).then(|| {
        load_conv(
            store,
            gpu,
            &format!("{prefix}.nin_shortcut"),
            input_channels,
            channels,
            1,
            stride,
            [0, 0, 0, 0, 0],
        )
    });

    Ok(Block {
        norm1: load_norm(store, gpu, &format!("{prefix}.norm1"), input_channels)?,
        conv1: load_conv(
            store,
            gpu,
            &format!("{prefix}.conv1"),
            input_channels,
            channels,
            3,
            stride,
            centred,
        )?,
        norm2: load_norm(store, gpu, &format!("{prefix}.norm2"), channels)?,
        conv2: load_conv(
            store,
            gpu,
            &format!("{prefix}.conv2"),
            channels,
            channels,
            3,
            stride,
            centred,
        )?,
        shortcut: shortcut.transpose()?,
    })
}

impl Encoder {
    /// Load every F32 tensor of the video VAE partition, plus the two 24-value
    /// normalization vectors from the config beside it (`:237-284`).
    pub fn load(directory: &Path) -> Result<Encoder, EncoderError> {
        let gpu = device(Gpu::create())?;
        let store = Store::open(directory, "video VAE")?;
        let (mean, deviation) = super::vae_video::latent_normalization(directory)
            .map_err(|error| EncoderError(error.to_string()))?;

        let conv_in = load_conv(
            &store,
            &gpu,
            "encoder.conv_in",
            RGB_CHANNELS as u32,
            LEVEL_CHANNELS[0],
            3,
            [1, 1, 1],
            [2, 1, 1, 1, 1],
        )?;

        let mut levels = Vec::with_capacity(LEVELS);
        let mut previous = LEVEL_CHANNELS[0];
        for (index, channels) in LEVEL_CHANNELS.iter().copied().enumerate() {
            let mut blocks = Vec::with_capacity(BLOCKS);
            for block in 0..BLOCKS {
                blocks.push(load_block(
                    &store,
                    &gpu,
                    &format!("encoder.down.{index}.block.{block}"),
                    previous,
                    channels,
                    block == 0,
                )?);
            }

            let downsample = (SPACE_STRIDES[index] * TIME_STRIDES[index] > 1)
                .then(|| {
                    let tail = u32::from(SPACE_STRIDES[index] == 2);
                    load_conv(
                        &store,
                        &gpu,
                        &format!("encoder.down.{index}.downsample.conv"),
                        channels,
                        channels,
                        3,
                        [
                            TIME_STRIDES[index],
                            SPACE_STRIDES[index],
                            SPACE_STRIDES[index],
                        ],
                        [2, 0, tail, 0, tail],
                    )
                })
                .transpose()?;

            levels.push(Level { blocks, downsample });
            previous = channels;
        }

        let last = LEVEL_CHANNELS[LEVELS - 1];
        Ok(Encoder {
            norm_out: load_norm(&store, &gpu, "encoder.norm_out", last)?,
            conv_out: load_conv(
                &store,
                &gpu,
                "encoder.conv_out",
                last,
                MOMENT_CHANNELS as u32,
                3,
                [1, 1, 1],
                [2, 1, 1, 1, 1],
            )?,
            quant: load_conv(
                &store,
                &gpu,
                "quant_conv",
                MOMENT_CHANNELS as u32,
                MOMENT_CHANNELS as u32,
                1,
                [1, 1, 1],
                [0, 0, 0, 0, 0],
            )?,
            conv_in,
            levels,
            gpu,
            mean,
            deviation,
        })
    }

    /// `conv_op` (`:454-485`) through `run_conv` (`:487-507`): pad, then a
    /// valid convolution, with the output shape the padded size implies.
    fn run_conv(
        &self,
        conv: &Conv,
        input: &Tensor,
        depth: u32,
        height: u32,
        width: u32,
    ) -> Result<(Tensor, u32, u32, u32), EncoderError> {
        let padded_depth = depth + conv.pads[0];
        let padded_height = height + conv.pads[1] + conv.pads[2];
        let padded_width = width + conv.pads[3] + conv.pads[4];
        if padded_depth < conv.kernel || padded_height < conv.kernel || padded_width < conv.kernel {
            return fail("visual encoder convolution is smaller than its kernel");
        }
        let output_depth = (padded_depth - conv.kernel) / conv.stride[0] + 1;
        let output_height = (padded_height - conv.kernel) / conv.stride[1] + 1;
        let output_width = (padded_width - conv.kernel) / conv.stride[2] + 1;

        let padded = (conv.pads.iter().any(|value| *value > 0))
            .then(|| {
                f32_alloc(
                    &self.gpu,
                    padded_depth as usize
                        * padded_height as usize
                        * padded_width as usize
                        * conv.input_channels as usize,
                )
            })
            .transpose()?;
        if let Some(padded) = &padded {
            device(self.gpu.vae_encoder_pad(
                padded,
                input,
                1,
                depth,
                height,
                width,
                conv.input_channels,
                conv.pads[0],
                conv.pads[1],
                conv.pads[2],
                conv.pads[3],
                conv.pads[4],
            ))?;
        }

        let output = f32_alloc(
            &self.gpu,
            output_depth as usize
                * output_height as usize
                * output_width as usize
                * conv.output_channels as usize,
        )?;
        device(self.gpu.conv3d(
            &output,
            padded.as_ref().unwrap_or(input),
            &conv.weight,
            Some(&conv.bias),
            1,
            padded_depth,
            padded_height,
            padded_width,
            conv.input_channels,
            conv.output_channels,
            [conv.kernel; 3],
            conv.stride,
        ))?;

        Ok((output, output_depth, output_height, output_width))
    }

    /// `run_block` (`:508-545`): group norm and SiLU, 3x3 convolution, again,
    /// then the scaled residual add against the identity or the 1x1 shortcut.
    fn run_block(
        &self,
        block: &Block,
        input: &Tensor,
        depth: u32,
        height: u32,
        width: u32,
    ) -> Result<(Tensor, u32, u32, u32), EncoderError> {
        let channels = block.conv1.input_channels;
        let count = depth as usize * height as usize * width as usize * channels as usize;
        let norm1 = f32_alloc(&self.gpu, count)?;
        device(self.gpu.group_norm_silu(
            &norm1,
            input,
            &block.norm1.weight,
            &block.norm1.bias,
            1,
            depth,
            height,
            width,
            channels,
            GROUPS,
            EPSILON,
        ))?;
        let (hidden, next_depth, next_height, next_width) =
            self.run_conv(&block.conv1, &norm1, depth, height, width)?;

        let norm2 = f32_alloc(
            &self.gpu,
            next_depth as usize
                * next_height as usize
                * next_width as usize
                * block.conv2.input_channels as usize,
        )?;
        device(self.gpu.group_norm_silu(
            &norm2,
            &hidden,
            &block.norm2.weight,
            &block.norm2.bias,
            1,
            next_depth,
            next_height,
            next_width,
            block.conv2.input_channels,
            GROUPS,
            EPSILON,
        ))?;
        let (output, out_depth, out_height, out_width) =
            self.run_conv(&block.conv2, &norm2, next_depth, next_height, next_width)?;

        let residual = match &block.shortcut {
            Some(shortcut) => Some(self.run_conv(shortcut, input, depth, height, width)?.0),
            None => None,
        };
        device(self.gpu.add_scaled(
            &output,
            residual.as_ref().unwrap_or(input),
            &output,
            1.0,
            1.0,
        ))?;

        Ok((output, out_depth, out_height, out_width))
    }

    /// `encode_tile` (`:487-589`): pixels in, one tile's normalized latent out,
    /// with the latent time the graph produced.
    fn encode_tile(
        &self,
        pixels: &[f32],
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize), EncoderError> {
        // `:494-505`: normalize with the image statistics and reorder from
        // channel-major `[3,T,H,W]` to time-major `[T,H,W,C]` for the first
        // convolution.
        let count = frames * height * width * RGB_CHANNELS;
        let mut normalized = vec![0.0f32; count];
        let mut destination = 0;
        for time in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    for (channel, (mean, deviation)) in MEAN.iter().zip(STD.iter()).enumerate() {
                        let source = ((channel * frames + time) * height + y) * width + x;
                        normalized[destination] = (pixels[source] - mean) / deviation;
                        destination += 1;
                    }
                }
            }
        }

        let hidden = f32_alloc(&self.gpu, count)?;
        upload_f32(&self.gpu, &hidden, &normalized)?;

        let (mut current, mut depth, mut h, mut w) = self.run_conv(
            &self.conv_in,
            &hidden,
            frames as u32,
            height as u32,
            width as u32,
        )?;
        for level in &self.levels {
            for block in &level.blocks {
                let (next, next_d, next_h, next_w) =
                    self.run_block(block, &current, depth, h, w)?;
                current = next;
                depth = next_d;
                h = next_h;
                w = next_w;
            }
            if let Some(downsample) = &level.downsample {
                let (next, next_d, next_h, next_w) =
                    self.run_conv(downsample, &current, depth, h, w)?;
                current = next;
                depth = next_d;
                h = next_h;
                w = next_w;
            }
        }

        // `:547-570`: the head, all three ops on the same activation shape.
        let norm = f32_alloc(
            &self.gpu,
            depth as usize * h as usize * w as usize * LEVEL_CHANNELS[LEVELS - 1] as usize,
        )?;
        device(self.gpu.group_norm_silu(
            &norm,
            &current,
            &self.norm_out.weight,
            &self.norm_out.bias,
            1,
            depth,
            h,
            w,
            LEVEL_CHANNELS[LEVELS - 1],
            GROUPS,
            EPSILON,
        ))?;
        let (moments, _, _, _) = self.run_conv(&self.conv_out, &norm, depth, h, w)?;
        let (quant, _, _, _) = self.run_conv(&self.quant, &moments, depth, h, w)?;

        // `:571-589`: the 48 moments are interleaved per position, so latent
        // channel `c` is moment `c` with the normalization undone.
        let moment_count = depth as usize * h as usize * w as usize * MOMENT_CHANNELS;
        device(self.gpu.sync())?;
        let raw = read_f32(&self.gpu, &quant, moment_count)?;
        let mut latent = vec![0.0f32; LATENT_CHANNELS * depth as usize * h as usize * w as usize];
        for channel in 0..LATENT_CHANNELS {
            for time in 0..depth as usize {
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let source =
                            ((time * h as usize + y) * w as usize + x) * MOMENT_CHANNELS + channel;
                        let target =
                            ((channel * depth as usize + time) * h as usize + y) * w as usize + x;
                        latent[target] =
                            (raw[source] - self.mean[channel]) / self.deviation[channel];
                    }
                }
            }
        }

        Ok((latent, depth as usize))
    }
}

/// `stitch_latents` (`:676-743`): the kept part of each tile, with the border
/// rows blended against the previous tile's tail at `y / overlap`.
fn stitch_latents(
    tiles: &[Vec<f32>],
    latent_time: usize,
    y_axis: &TileAxis,
    x_axis: &TileAxis,
) -> Latent {
    let tile_h = y_axis.length / SPATIAL_RATIO;
    let tile_w = x_axis.length / SPATIAL_RATIO;
    let full_h = y_axis.extent() / SPATIAL_RATIO;
    let full_w = x_axis.extent() / SPATIAL_RATIO;
    let mut values = vec![0.0f32; LATENT_CHANNELS * latent_time * full_h * full_w];

    for tile_y in 0..y_axis.count() {
        for tile_x in 0..x_axis.count() {
            let index = tile_y * x_axis.count() + tile_x;
            let current = &tiles[index];
            let above = (tile_y > 0).then(|| &tiles[index - x_axis.count()]);
            let left = (tile_x > 0).then(|| &tiles[index - 1]);
            let overlap_y = if tile_y > 0 {
                y_axis.overlaps[tile_y - 1] / SPATIAL_RATIO
            } else {
                0
            };
            let overlap_x = if tile_x > 0 {
                x_axis.overlaps[tile_x - 1] / SPATIAL_RATIO
            } else {
                0
            };
            let keep_h = tile_h
                - if tile_y + 1 < y_axis.count() {
                    y_axis.overlaps[tile_y] / SPATIAL_RATIO
                } else {
                    0
                };
            let keep_w = tile_w
                - if tile_x + 1 < x_axis.count() {
                    x_axis.overlaps[tile_x] / SPATIAL_RATIO
                } else {
                    0
                };
            let destination_y = y_axis.starts[tile_y] / SPATIAL_RATIO;
            let destination_x = x_axis.starts[tile_x] / SPATIAL_RATIO;

            for channel in 0..LATENT_CHANNELS {
                for time in 0..latent_time {
                    for y in 0..keep_h {
                        for x in 0..keep_w {
                            let local = ((channel * latent_time + time) * tile_h + y) * tile_w + x;
                            let mut value = current[local];
                            if let Some(above) = above {
                                if y < overlap_y {
                                    let source = ((channel * latent_time + time) * tile_h + tile_h
                                        - overlap_y
                                        + y)
                                        * tile_w
                                        + x;
                                    let weight = y as f32 / overlap_y as f32;
                                    value = above[source] * (1.0 - weight) + value * weight;
                                }
                            }
                            if let Some(left) = left {
                                if x < overlap_x {
                                    let source = ((channel * latent_time + time) * tile_h + y)
                                        * tile_w
                                        + tile_w
                                        - overlap_x
                                        + x;
                                    let weight = x as f32 / overlap_x as f32;
                                    value = left[source] * (1.0 - weight) + value * weight;
                                }
                            }
                            let destination =
                                ((channel * latent_time + time) * full_h + destination_y + y)
                                    * full_w
                                    + destination_x
                                    + x;
                            values[destination] = value;
                        }
                    }
                }
            }
        }
    }

    Latent {
        time: latent_time,
        height: full_h,
        width: full_w,
        values,
    }
}

/// `h3_video_vae_encode` (`:745-808`): every tile, then one stitched latent.
/// `progress` receives `(completed_tiles, total_tiles)` after each tile.
pub fn encode(
    directory: &Path,
    pixels: &[f32],
    frames: usize,
    height: usize,
    width: usize,
    mut progress: impl FnMut(usize, usize),
) -> Result<Latent, EncoderError> {
    if frames < 1 || height < MIN_EXTENT || width < MIN_EXTENT {
        return fail("invalid visual encoder arguments");
    }
    if !height.is_multiple_of(SPATIAL_RATIO) || !width.is_multiple_of(SPATIAL_RATIO) {
        return fail("visual encoder extents must be multiples of 16");
    }
    let expected = RGB_CHANNELS * frames * height * width;
    if pixels.len() != expected {
        return fail(format!(
            "visual encoder pixels have {} values, expected {expected}",
            pixels.len()
        ));
    }

    let y_axis = tile_axis(height)?;
    let x_axis = tile_axis(width)?;
    let tile_count = y_axis.count() * x_axis.count();
    let encoder = Encoder::load(directory)?;

    let mut tiles = Vec::with_capacity(tile_count);
    let mut time = 0;
    for (index, (start_y, start_x)) in (0..y_axis.count())
        .flat_map(|y| (0..x_axis.count()).map(move |x| (y, x)))
        .map(|(y, x)| (y_axis.starts[y], x_axis.starts[x]))
        .enumerate()
    {
        let tile = extract_pixel_tile(
            pixels,
            frames,
            height,
            width,
            start_y,
            start_x,
            y_axis.length,
            x_axis.length,
        );
        let (latent, produced) =
            encoder.encode_tile(&tile, frames, y_axis.length, x_axis.length)?;
        if index > 0 && produced != time {
            return fail(format!(
                "visual encoder tiles disagree on latent time: {time} and {produced}"
            ));
        }
        time = produced;
        tiles.push(latent);
        progress(index + 1, tile_count);
    }

    Ok(stitch_latents(&tiles, time, &y_axis, &x_axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_tile_when_the_extent_fits_the_released_tile() {
        for extent in [32, 48, 192, 256] {
            let axis = tile_axis(extent).expect("fits");
            assert_eq!(axis.count(), 1);
            assert_eq!(axis.length, extent);
            assert!(axis.overlaps.is_empty());
            assert_eq!(axis.extent(), extent);
        }
    }

    #[test]
    fn the_count_is_raised_until_the_minimum_overlap_reaches_the_extent() {
        // 272: ceil(272/256) = 2 and 256*2 - 64 = 448 >= 272, so two tiles.
        let axis = tile_axis(272).expect("two");
        assert_eq!(axis.count(), 2);
        assert_eq!(axis.length, TILE_PIXELS);
        // The leftover is 448 - 272 = 176, which is 11 units of 16 handed to
        // the single gap: 64 + 176 = 240.
        assert_eq!(axis.overlaps, vec![240]);
        assert_eq!(axis.starts, vec![0, 16]);
        assert_eq!(axis.extent(), 272);
    }

    #[test]
    fn surplus_is_spread_over_the_gaps_first_gap_first() {
        // 512 needs three tiles: two cover only 448 px, three cover 640.
        let three = tile_axis(512).expect("512");
        assert_eq!(three.count(), 3);
        // The surplus is 640 - 512 = 128, eight 16 px units over two gaps.
        assert_eq!(three.overlaps, vec![128, 128]);
        assert_eq!(three.starts, vec![0, 128, 256]);
        assert_eq!(three.extent(), 512);

        // 768: four tiles cover 4*256 - 3*64 = 832, surplus 64 = four units
        // over three gaps: 80, 80, 80 and one more unit on the first gap.
        let four = tile_axis(768).expect("768");
        assert_eq!(four.count(), 4);
        assert_eq!(four.overlaps, vec![96, 80, 80]);
        assert_eq!(four.extent(), 768);
        assert_eq!(four.starts, vec![0, 160, 336, 512]);
    }

    #[test]
    fn extents_below_the_reference_floor_or_unaligned_are_refused() {
        assert!(tile_axis(16).is_err());
        assert!(tile_axis(0).is_err());
        assert!(tile_axis(300).is_err());
    }

    #[test]
    fn latent_time_is_the_released_temporal_compression() {
        assert_eq!(latent_time(1), 1);
        assert_eq!(latent_time(5), 2);
        assert_eq!(latent_time(22), 6);
        assert_eq!(latent_time(39), 10);
        assert_eq!(latent_time(107), 27);
    }

    #[test]
    fn a_tile_is_cut_channel_major_with_its_own_stride() {
        let (frames, full_h, full_w) = (2usize, 4usize, 4usize);
        let pixels: Vec<f32> = (0..RGB_CHANNELS * frames * full_h * full_w)
            .map(|value| value as f32)
            .collect();
        let (tile_h, tile_w) = (2usize, 2usize);
        let tile = extract_pixel_tile(&pixels, frames, full_h, full_w, 1, 2, tile_h, tile_w);
        let full = |channel: usize, time: usize, y: usize, x: usize| {
            ((channel * frames + time) * full_h + 1 + y) * full_w + 2 + x
        };
        let local = |channel: usize, time: usize, y: usize, x: usize| {
            ((channel * frames + time) * tile_h + y) * tile_w + x
        };

        for channel in 0..RGB_CHANNELS {
            for time in 0..frames {
                for y in 0..tile_h {
                    for x in 0..tile_w {
                        assert_eq!(
                            tile[local(channel, time, y, x)],
                            pixels[full(channel, time, y, x)]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_stitched_border_blends_the_previous_tiles_tail() {
        // Two tiles 256 px wide over a 272 px extent: overlap 240 px, which is
        // 15 latent columns, and the first tile keeps 256 - 240 = 1 column.
        let y_axis = tile_axis(32).expect("one row");
        let x_axis = tile_axis(272).expect("two columns");
        let latent_time = 1;
        let tile_h = y_axis.length / SPATIAL_RATIO;
        let tile_w = x_axis.length / SPATIAL_RATIO;
        let first = vec![1.0f32; LATENT_CHANNELS * latent_time * tile_h * tile_w];
        let second = vec![3.0f32; LATENT_CHANNELS * latent_time * tile_h * tile_w];
        let latent = stitch_latents(&[first, second], latent_time, &y_axis, &x_axis);

        assert_eq!(latent.height, 2);
        assert_eq!(latent.width, 17);
        // The first tile owns column 0 outright.
        assert_eq!(latent.values[0], 1.0);
        // Column 1 is the second tile's row 0 at alpha 0/15, so the first
        // tile's tail value 1.0 wins.
        assert_eq!(latent.values[1], 1.0);
        // Column 2 is alpha 1/15, and the last column is the second tile's own.
        let expected = 1.0 * (1.0 - 1.0 / 15.0) + 3.0 * (1.0 / 15.0);
        assert!((latent.values[2] - expected).abs() < 1e-6);
        assert_eq!(latent.values[16], 3.0);
    }
}
