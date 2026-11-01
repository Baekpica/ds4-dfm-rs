//! The LTX-2.5 audio VAE encoder: the state the stage-2 refiner holds clean.
//!
//! Source of truth: `/data/ltx2-pin/packages/ltx-core` (pin
//! `d1511477` = tag v1.2.0), `model/audio_vae/`, and the artifact's own header
//! (`ltx-2.5-audio-vae-bf16.safetensors`: 1329 BF16 tensors and a `config`
//! whose `ddconfig` is `ch 128`, `ch_mult [1, 2, 4]`, `num_res_blocks 2`,
//! `double_z true`, `z_channels 8`, `mel_bins 64`, `in_channels 2`,
//! `causality_axis height`, `norm_type pixel`, no mid-block attention).
//!
//! The release reaches it from `runtime/stage2.py:78-89`
//! (`encode_source_audio`): the stage-1 PCM at 32 kHz through
//! `AudioProcessor.waveform_to_mel` (`ops.py:11-55`: sinc resample to 16 kHz,
//! `MelSpectrogram(16000, n_fft=1024, hop=160, n_mels=64, power=1.0,
//! window_fn=hann, center=True, pad_mode="reflect", mel_scale="slaney",
//! norm="slaney")`, then `log(clamp(x, 1e-5))`), then the encoder body
//! (`audio_vae.py:172-247`):
//!
//! ```text
//! mel [1, 2, T=518, 64]                        (channels, time, mel bins)
//!   -> conv_in 2 -> 128, 3x3 causal             (causal_conv_2d.py:47-57)
//!   -> 3 levels x 2 ResnetBlock(128, 256, 512)  (resnet.py:98-170)
//!        norm1(PixelNorm) -> SiLU -> conv1 -> norm2 -> SiLU -> conv2, + x
//!      levels 0 and 1 end in Downsample(3x3, stride 2)  (downsample.py:36-57)
//!   -> mid.block_1, mid.block_2 (this artifact has no mid attention)
//!   -> norm_out(PixelNorm) -> SiLU -> conv_out 512 -> 16   (double_z)
//!   -> chunk(2, dim=1)[0]: the means -> 8 channels
//!   -> patchify "b c t f -> b t (c f)" -> per_channel_statistics.normalize
//!   -> unpatchify: [1, 8, 130, 16]
//! ```
//!
//! Every pad is zeros: `F.pad(x, (1, 1, 2, 0))` for the causal 3x3 and
//! `(0, 1, 2, 0)` for the stride-2 downsample (`downsample.py:47-56`), time
//! padded on top only. The graph therefore is the F32 valid convolution over
//! an explicitly padded activation (`h3d_conv3d`, channels-last), and the two
//! kernels this module needed are `h3a_pad_zero` and `h3a_pixel_norm`.
//!
//! Named deviations, both measured, both in
//! `docs/minimax-h3/ltx-audio-encoder-2026-10-31.md`:
//!
//! - **F32 arithmetic over widened BF16 weights.** The release builds this
//!   encoder at `models.dtype` = BF16 (`stage2_ops/models.py:110-114`), so its
//!   own latent carries one BF16 rounding per layer. The port widens the
//!   checkpoint's BF16 weights (exact) and computes F32, which is what the
//!   reference golden it is gated against was produced with
//!   (`AudioEncoderConfigurator` builds F32 there; its `load_state_dict`
//!   widens the BF16 tensors). The port's one BF16 rounding happens at the
//!   refiner boundary, where `AudioPatchifier.patchify` takes a BF16 latent.
//! - **The convolution is the direct F32 valid convolution**, not cuDNN: same
//!   expression, same `fmaf` accumulation; the gate reports its rel-RMS on the
//!   golden.
//!
//! The front end stays on the host: it is tens of MFLOP per clip and its
//! values are the reference's own (`torchaudio.functional.resample`'s
//! `sinc_interp_hann` kernel and `melscale_fbanks`'s slaney grid, both
//! transcribed below and each gated on its own).
//!
//! Frames: this stack divides the time axis by four, so 518 mel frames become
//! 130 latent frames. The refiner's shape is 126 (`AUDIO_TOKENS`) and the
//! release truncates at that boundary (`runtime/stage2.py:88-89`,
//! `conformed[:, :, :frames]`); [`AudioLatent::conformed`] is that step, and
//! the encoder itself returns its own 130.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::gemma::bf16_round;
use super::safetensors::{self, Dtype as ShardDtype, Shard};
use super::vision::bf16_to_f32;

/// The stage-1 PCM rate (`h3_audio_vae.h:46`, `vae_audio.rs`).
pub const SOURCE_RATE: u32 = 32_000;
/// `AudioProcessor`'s `target_sample_rate` (`ops.py:20`).
pub const SAMPLE_RATE: u32 = 16_000;
/// `MelSpectrogram(n_mels=64)` (`ops.py:27`).
pub const MEL_BINS: usize = 64;
/// The artifact's `in_channels`: the mel is stereo.
pub const MEL_CHANNELS: usize = 2;
/// `n_fft` / `win_length` (`ops.py:22-23`).
const N_FFT: usize = 1024;
/// `hop_length` (`ops.py:24`).
const HOP: usize = 160;
/// The artifact's `ch`.
const CHANNELS: u32 = 128;
/// The artifact's `ch_mult`: one level per entry.
const CH_MULT: [u32; 3] = [1, 2, 4];
/// The artifact's `num_res_blocks`.
const RES_BLOCKS: usize = 2;
/// `z_channels` (`double_z` doubles `conv_out` to 16, `audio_vae.py:184-188`).
pub const LATENT_CHANNELS: u32 = 8;
/// The artifact's `mel_bins` after two halvings on each axis.
pub const LATENT_BINS: u32 = 16;
/// `2 * z_channels`.
const CONV_OUT_CHANNELS: u32 = LATENT_CHANNELS * 2;
/// `build_normalization_layer(normtype=PIXEL)` (`common/normalization.py:46`).
const PIXEL_NORM_EPSILON: f32 = 1e-6;
/// `CausalConv2d`'s kernel (`resnet.py:154-168`).
const KERNEL: u32 = 3;
/// The shortcut's `1x1` (`resnet.py:166-170`).
const SHORTCUT_KERNEL: u32 = 1;
/// `Downsample`'s stride (`downsample.py:44`).
const DOWNSAMPLE_STRIDE: u32 = 2;
/// `torch.log(torch.clamp(mel, min=1e-5))` (`ops.py:52`).
const LOG_FLOOR: f32 = 1e-5;
/// The resampler's width and roll-off (`torchaudio`'s `sinc_interp_hann`
/// defaults, `lowpass_filter_width=6` and `rolloff=0.99`; `functional.py`
/// `_get_sinc_resample_kernel`).
const RESAMPLE_WIDTH: f64 = 6.0;
const RESAMPLE_ROLLOFF: f64 = 0.99;
/// `slaney`'s mel scale (`torchaudio.functional.melscale_fbanks`).
const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
/// `ln(6.4) / 27`.
const SLANEY_LOG_STEP: f64 = 0.068_751_730_871_196_29;

const ENCODER_PREFIX: &str = "audio_vae.encoder.";
const STATS_PREFIX: &str = "audio_vae.per_channel_statistics.";
const MEAN_OF_MEANS: &str = "mean-of-means";
const STD_OF_MEANS: &str = "std-of-means";

/// Why the encoder could not be loaded or run.
#[derive(Debug)]
pub enum AudioError {
    /// The checkpoint could not be read.
    Io(String),
    /// A tensor the graph needs is absent.
    Missing(String),
    /// A tensor's dtype or shape is not the one the config derives.
    Shape(String),
    /// The safetensors reader refused the file.
    Reader(String),
    /// A device call failed.
    Run(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Io(detail) => write!(formatter, "audio vae io: {detail}"),
            AudioError::Missing(name) => write!(formatter, "audio vae checkpoint has no {name}"),
            AudioError::Shape(detail) => write!(formatter, "audio vae shape: {detail}"),
            AudioError::Reader(detail) => write!(formatter, "audio vae reader: {detail}"),
            AudioError::Run(detail) => write!(formatter, "audio vae run: {detail}"),
        }
    }
}

impl std::error::Error for AudioError {}

fn run(label: &str) -> impl Fn(String) -> AudioError + '_ {
    move |detail| AudioError::Run(format!("{label}: {detail}"))
}

/// The log-mel the encoder consumes: channels-last `[frames][bins][channels]`,
/// the layout the valid convolution reads.
pub struct Mel {
    pub channels: usize,
    pub frames: usize,
    pub bins: usize,
    pub values: Vec<f32>,
}

/// The encoder's output in the checkpoint's own `[B][C][T][F]` layout.
pub struct AudioLatent {
    pub channels: u32,
    pub frames: u32,
    pub bins: u32,
    /// Channel-major: `((channel * frames) + frame) * bins + bin`.
    pub values: Vec<f32>,
}

impl AudioLatent {
    /// The BF16 bytes `AudioPatchifier.patchify` takes (`[C][T][F]`, one
    /// rounding): the refiner's audio state is a BF16 tensor.
    pub fn bf16_bytes(&self) -> Vec<u8> {
        self.values
            .iter()
            .flat_map(|value| (bf16_round(*value).to_bits() >> 16).to_le_bytes())
            .collect()
    }

    /// `conformed[:, :, :frames]` (`runtime/stage2.py:88-89`): the first
    /// `frames` frames of every channel, zero-filled when the encoder
    /// produced fewer.
    pub fn conformed(&self, frames: u32) -> AudioLatent {
        let bins = self.bins as usize;
        let copy = frames.min(self.frames) as usize * bins;
        let mut values = vec![0.0f32; self.channels as usize * frames as usize * bins];

        for channel in 0..self.channels as usize {
            let source = channel * self.frames as usize * bins;
            let destination = channel * frames as usize * bins;
            values[destination..destination + copy]
                .copy_from_slice(&self.values[source..source + copy]);
        }

        AudioLatent {
            channels: self.channels,
            frames,
            bins: self.bins,
            values,
        }
    }
}

/// One convolution: the checkpoint's `[out][in][kh][kw]`, which is also the
/// `[out][in][1][kh][kw]` order the F32 valid convolution reads, plus its bias.
pub struct Conv {
    weight: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
    kernel: u32,
    stride: u32,
}

/// One `ResnetBlock` (`resnet.py:98-170`): the artifact carries no timestep
/// conditioning (`temb_channels == 0`) and no attention.
struct Block {
    conv1: Conv,
    conv2: Conv,
    /// `nin_shortcut` when the block changes width (`resnet.py:162-170`).
    shortcut: Option<Conv>,
}

/// One downsampling level: its resblocks and, except at the last level, the
/// stride-2 convolution (`downsample.py:172-186`).
struct Level {
    blocks: Vec<Block>,
    downsample: Option<Conv>,
}

/// `PerChannelStatistics` (`ops.py:58-76`), `channels * mel_bins` wide.
struct Stats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// A shape on the encoder's `(channels, time, mel bins)` grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub channels: u32,
    pub frames: u32,
    pub bins: u32,
}

impl Shape {
    fn positions(&self) -> usize {
        self.frames as usize * self.bins as usize
    }

    fn elements(&self) -> usize {
        self.positions() * self.channels as usize
    }
}

/// The shipped encoder, weights resident.
pub struct Encoder {
    conv_in: Conv,
    levels: Vec<Level>,
    mid: Vec<Block>,
    conv_out: Conv,
    stats: Stats,
    tensors: usize,
    bytes: u64,
}

impl Encoder {
    /// Read the shipped `ltx-2.5-audio-vae-bf16.safetensors`: `conv_in`, three
    /// levels of two resblocks with a downsample after the first two, the two
    /// mid blocks, `conv_out` and the per-channel statistics.
    pub fn open(gpu: &Gpu, path: &Path) -> Result<Encoder, AudioError> {
        let shard = safetensors::read_shard(path)
            .map_err(|error| AudioError::Io(format!("{}: {error:?}", path.display())))?;
        let mut reader = Reader {
            gpu,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };

        let conv_in = reader.conv(&shard, "conv_in", MEL_CHANNELS as u32, CHANNELS, KERNEL, 1)?;
        let mut levels = Vec::with_capacity(CH_MULT.len());

        for (index, mult) in CH_MULT.iter().enumerate() {
            let width = CHANNELS * mult;
            let mut previous = if index == 0 {
                CHANNELS
            } else {
                CHANNELS * CH_MULT[index - 1]
            };
            let mut blocks = Vec::with_capacity(RES_BLOCKS);

            for block in 0..RES_BLOCKS {
                let name = format!("down.{index}.block.{block}");
                let shortcut = if previous != width {
                    Some(reader.conv(
                        &shard,
                        &format!("{name}.nin_shortcut"),
                        previous,
                        width,
                        SHORTCUT_KERNEL,
                        1,
                    )?)
                } else {
                    None
                };
                blocks.push(Block {
                    conv1: reader.conv(
                        &shard,
                        &format!("{name}.conv1"),
                        previous,
                        width,
                        KERNEL,
                        1,
                    )?,
                    conv2: reader.conv(
                        &shard,
                        &format!("{name}.conv2"),
                        width,
                        width,
                        KERNEL,
                        1,
                    )?,
                    shortcut,
                });
                previous = width;
            }

            let downsample = if index + 1 == CH_MULT.len() {
                None
            } else {
                Some(reader.conv(
                    &shard,
                    &format!("down.{index}.downsample"),
                    width,
                    width,
                    KERNEL,
                    DOWNSAMPLE_STRIDE,
                )?)
            };
            levels.push(Level { blocks, downsample });
        }

        let last = CHANNELS * CH_MULT[CH_MULT.len() - 1];
        let mut mid = Vec::with_capacity(2);
        for index in 1..=2 {
            let name = format!("mid.block_{index}");
            mid.push(Block {
                conv1: reader.conv(&shard, &format!("{name}.conv1"), last, last, KERNEL, 1)?,
                conv2: reader.conv(&shard, &format!("{name}.conv2"), last, last, KERNEL, 1)?,
                shortcut: None,
            });
        }

        let conv_out = reader.conv(&shard, "conv_out", last, CONV_OUT_CHANNELS, KERNEL, 1)?;
        let stats = Stats {
            mean: reader.scalar(&shard, MEAN_OF_MEANS, CHANNELS)?,
            std: reader.scalar(&shard, STD_OF_MEANS, CHANNELS)?,
        };
        let (tensors, bytes) = (reader.tensors, reader.bytes);

        Ok(Encoder {
            conv_in,
            levels,
            mid,
            conv_out,
            stats,
            tensors,
            bytes,
        })
    }

    /// How many tensors the graph holds and how many checkpoint bytes they
    /// came from: the artifact's encoder half is 46 tensors.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// The graph's stage shapes for a mel of `frames` rows, in the order they
    /// are visited: one per level, after that level's resblocks and downsample.
    pub fn shapes(frames: u32) -> Vec<Shape> {
        let mut shapes = vec![Shape {
            channels: CHANNELS,
            frames,
            bins: MEL_BINS as u32,
        }];
        let mut shape = shapes[0];

        for index in 0..CH_MULT.len() {
            shape.channels = CHANNELS * CH_MULT[index];
            if index + 1 != CH_MULT.len() {
                shape.frames = (shape.frames + 2 - KERNEL) / DOWNSAMPLE_STRIDE + 1;
                shape.bins = (shape.bins + 1 - KERNEL) / DOWNSAMPLE_STRIDE + 1;
            }
            shapes.push(shape);
        }

        shapes
    }

    /// The whole encoder: the mel is uploaded channels-last, the convolution
    /// stack runs on the device, and the few thousand latent values come back
    /// for the per-channel normalisation.
    pub fn encode(&self, gpu: &Gpu, mel: &Mel) -> Result<AudioLatent, AudioError> {
        if mel.bins != MEL_BINS || mel.channels != MEL_CHANNELS {
            return Err(AudioError::Shape(format!(
                "the encoder takes {MEL_CHANNELS} channels of {MEL_BINS} bins, not {} of {}",
                mel.channels, mel.bins
            )));
        }

        let input = Shape {
            channels: mel.channels as u32,
            frames: mel.frames as u32,
            bins: mel.bins as u32,
        };
        let mel_bytes: Vec<u8> = mel
            .values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let source = gpu
            .alloc(Dtype::F32, mel.values.len())
            .map_err(run("mel"))?;
        gpu.upload(&source, &mel_bytes).map_err(run("mel"))?;

        // `conv_in`'s own pad scratch: two channels, which no level needs.
        let padded = pad_buffer(gpu, input.channels, input)?;
        let (mut state, mut shape) =
            self.conv(gpu, Some(&padded), &source, &self.conv_in, input)?;

        for level in &self.levels {
            let widest = level
                .blocks
                .iter()
                .flat_map(|block| [block.conv1.input_channels, block.conv2.input_channels])
                .chain(level.downsample.iter().map(|down| down.input_channels))
                .max()
                .unwrap_or(shape.channels);
            let mut padded = pad_buffer(gpu, widest, shape)?;

            for block in &level.blocks {
                let (next, next_shape) = self.block(gpu, &mut padded, &state, block, shape)?;
                state = next;
                shape = next_shape;
            }

            if let Some(down) = &level.downsample {
                let (out, out_shape) = self.conv(gpu, Some(&padded), &state, down, shape)?;
                let expected = downsampled(shape);
                if out_shape != expected {
                    return Err(AudioError::Shape(format!(
                        "the downsample produced {out_shape:?}, not {expected:?}"
                    )));
                }
                state = out;
                shape = expected;
            }
        }

        // The mid blocks and the head (`audio_vae.py:200-220`).
        let widest = self
            .mid
            .iter()
            .flat_map(|block| [block.conv1.input_channels, block.conv2.input_channels])
            .chain([self.conv_out.input_channels])
            .max()
            .unwrap_or(shape.channels);
        let mut padded = pad_buffer(gpu, widest, shape)?;

        for block in &self.mid {
            let (next, next_shape) = self.block(gpu, &mut padded, &state, block, shape)?;
            state = next;
            shape = next_shape;
        }

        let normed = gpu
            .alloc(Dtype::F32, shape.elements())
            .map_err(run("norm_out"))?;
        gpu.audio_pixel_norm(
            &normed,
            &state,
            shape.channels,
            shape.positions(),
            PIXEL_NORM_EPSILON,
        )
        .map_err(run("norm_out"))?;
        gpu.silu_f32(&normed, &normed)
            .map_err(run("norm_out silu"))?;
        let (output, out_shape) = self.conv(gpu, Some(&padded), &normed, &self.conv_out, shape)?;

        // `double_z` keeps the first half; the statistics then normalise each
        // `(channel, bin)` pair, which is what patchify's `(c, f)` fold means.
        let mut host = vec![0.0f32; output.elements()];
        gpu.read_f32(&output, &mut host)
            .map_err(run("latent read"))?;

        Ok(self.normalise(&host, out_shape))
    }

    /// `_normalize_latents` (`audio_vae.py:223-246`) on the host, where the
    /// tensor is 33,280 values: `chunk(2, dim=1)[0]`, patchify's `(c, f)`
    /// fold, `(x - mean) / std`, and the channel-major result.
    fn normalise(&self, host: &[f32], shape: Shape) -> AudioLatent {
        let (channels, frames, bins) = (
            LATENT_CHANNELS as usize,
            shape.frames as usize,
            shape.bins as usize,
        );
        let mut values = vec![0.0f32; channels * frames * bins];

        for channel in 0..channels {
            for frame in 0..frames {
                for bin in 0..bins {
                    let at = (frame * bins + bin) * shape.channels as usize + channel;
                    let stat = channel * bins + bin;
                    values[(channel * frames + frame) * bins + bin] =
                        (host[at] - self.stats.mean[stat]) / self.stats.std[stat];
                }
            }
        }

        AudioLatent {
            channels: LATENT_CHANNELS,
            frames: shape.frames,
            bins: shape.bins,
            values,
        }
    }

    /// One resblock (`resnet.py:140-170`). Every intermediate is an exact-size
    /// buffer, because the F32 elementwise ops take the tensor's whole extent.
    fn block(
        &self,
        gpu: &Gpu,
        padded: &mut Tensor,
        state: &Tensor,
        block: &Block,
        shape: Shape,
    ) -> Result<(Tensor, Shape), AudioError> {
        // norm1 -> SiLU -> conv1.
        let normed = gpu
            .alloc(Dtype::F32, shape.elements())
            .map_err(run("norm1"))?;
        gpu.audio_pixel_norm(
            &normed,
            state,
            shape.channels,
            shape.positions(),
            PIXEL_NORM_EPSILON,
        )
        .map_err(run("norm1"))?;
        gpu.silu_f32(&normed, &normed).map_err(run("silu1"))?;
        let (first, first_shape) = self.conv(gpu, Some(padded), &normed, &block.conv1, shape)?;

        // norm2 -> SiLU -> dropout(0) -> conv2.
        let normed = gpu
            .alloc(Dtype::F32, first_shape.elements())
            .map_err(run("norm2"))?;
        gpu.audio_pixel_norm(
            &normed,
            &first,
            first_shape.channels,
            first_shape.positions(),
            PIXEL_NORM_EPSILON,
        )
        .map_err(run("norm2"))?;
        gpu.silu_f32(&normed, &normed).map_err(run("silu2"))?;
        let (second, out_shape) =
            self.conv(gpu, Some(padded), &normed, &block.conv2, first_shape)?;

        // The shortcut (`resnet.py:158-170`): `x` itself, or its 1x1.
        let output = gpu
            .alloc(Dtype::F32, out_shape.elements())
            .map_err(run("residual"))?;
        match &block.shortcut {
            Some(shortcut) => {
                let (projected, projected_shape) = self.conv(gpu, None, state, shortcut, shape)?;
                if projected_shape != out_shape {
                    return Err(AudioError::Shape(format!(
                        "the shortcut produced {projected_shape:?}, the block {out_shape:?}"
                    )));
                }
                gpu.audio_add(&output, &projected, &second, out_shape.elements())
                    .map_err(run("shortcut add"))?;
            }
            None => {
                gpu.audio_add(&output, state, &second, out_shape.elements())
                    .map_err(run("residual add"))?;
            }
        }

        Ok((output, out_shape))
    }

    /// One convolution: the layer's zero pad (`h3a_pad_zero` when the kernel
    /// needs a halo) and the F32 valid convolution. `padded` is the level's
    /// scratch; a 1x1 passes `None` and reads its input directly.
    fn conv(
        &self,
        gpu: &Gpu,
        padded: Option<&Tensor>,
        input: &Tensor,
        conv: &Conv,
        shape: Shape,
    ) -> Result<(Tensor, Shape), AudioError> {
        let pads = pads(conv.kernel, conv.stride);
        let padded_shape = Shape {
            channels: shape.channels,
            frames: shape.frames + pads[0] + pads[1],
            bins: shape.bins + pads[2] + pads[3],
        };
        let out_shape = output_shape(conv, padded_shape);
        let output = gpu
            .alloc(Dtype::F32, out_shape.elements())
            .map_err(run("conv output"))?;

        if padded_shape == shape {
            self.valid(gpu, &output, input, conv, shape, out_shape)?;
            return Ok((output, out_shape));
        }

        let padded = padded.ok_or_else(|| {
            AudioError::Shape(format!(
                "the {}x{} stride-{} convolution needs a padded buffer",
                conv.kernel, conv.kernel, conv.stride
            ))
        })?;
        if padded.elements() < padded_shape.elements() {
            return Err(AudioError::Shape(format!(
                "the pad buffer holds {} values, the convolution needs {}",
                padded.elements(),
                padded_shape.elements()
            )));
        }
        gpu.audio_pad_zero(
            padded,
            input,
            shape.frames,
            shape.bins,
            shape.channels,
            pads[0],
            pads[1],
            pads[2],
            pads[3],
        )
        .map_err(run("pad"))?;
        self.valid(gpu, &output, padded, conv, padded_shape, out_shape)?;

        Ok((output, out_shape))
    }

    fn valid(
        &self,
        gpu: &Gpu,
        output: &Tensor,
        padded: &Tensor,
        conv: &Conv,
        padded_shape: Shape,
        out_shape: Shape,
    ) -> Result<(), AudioError> {
        gpu.conv3d(
            output,
            padded,
            &conv.weight,
            Some(&conv.bias),
            1,
            1,
            padded_shape.frames,
            padded_shape.bins,
            conv.input_channels,
            conv.output_channels,
            [1, conv.kernel, conv.kernel],
            [1, conv.stride, conv.stride],
        )
        .map_err(run("conv"))?;

        let landed = output_shape(conv, padded_shape);
        if landed != out_shape {
            return Err(AudioError::Shape(format!(
                "the convolution lands on {landed:?}, the plan expects {out_shape:?}"
            )));
        }

        Ok(())
    }
}

/// The pad scratch one stage needs: `channels` wide over `(frames + 2,
/// bins + 2)`, the largest plane a 3x3 convolution of that stage can read.
fn pad_buffer(gpu: &Gpu, channels: u32, shape: Shape) -> Result<Tensor, AudioError> {
    let elements = channels as usize
        * (shape.frames as usize + KERNEL as usize - 1)
        * (shape.bins as usize + KERNEL as usize - 1);

    gpu.alloc(Dtype::F32, elements.max(1))
        .map_err(run("pad buffer"))
}

/// `CausalConv2d`'s and `Downsample`'s pad, `(top, bottom, left, right)`:
/// `causal_conv_2d.py:47-57` is `(pad_h, 0, pad_w / 2, pad_w - pad_w / 2)` and
/// `downsample.py:47-56` is `(2, 0, 0, 1)`.
fn pads(kernel: u32, stride: u32) -> [u32; 4] {
    if stride == DOWNSAMPLE_STRIDE {
        return [KERNEL - 1, 0, 0, 1];
    }

    let halo = kernel - 1;
    [halo, 0, halo / 2, halo - halo / 2]
}

fn output_shape(conv: &Conv, padded: Shape) -> Shape {
    Shape {
        channels: conv.output_channels,
        frames: (padded.frames - conv.kernel) / conv.stride + 1,
        bins: (padded.bins - conv.kernel) / conv.stride + 1,
    }
}

fn downsampled(shape: Shape) -> Shape {
    let pads = pads(KERNEL, DOWNSAMPLE_STRIDE);
    let padded = Shape {
        frames: shape.frames + pads[0] + pads[1],
        bins: shape.bins + pads[2] + pads[3],
        ..shape
    };

    Shape {
        channels: shape.channels,
        frames: (padded.frames - KERNEL) / DOWNSAMPLE_STRIDE + 1,
        bins: (padded.bins - KERNEL) / DOWNSAMPLE_STRIDE + 1,
    }
}

/// Reads the artifact's tensors into F32 device buffers. The checkpoint is
/// BF16 throughout, so a read checks its dtype and shape before widening.
struct Reader<'a> {
    gpu: &'a Gpu,
    buffer: Vec<u8>,
    tensors: usize,
    bytes: u64,
}

impl Reader<'_> {
    fn pull(&mut self, shard: &Shard, name: &str, shape: &[usize]) -> Result<Tensor, AudioError> {
        let tensor = shard
            .find(name)
            .ok_or_else(|| AudioError::Missing(name.to_string()))?;
        let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

        if tensor.dtype != ShardDtype::Bf16 || tensor.shape != want {
            return Err(AudioError::Shape(format!(
                "{name}: checkpoint has {} {:?}, the graph needs {} {:?}",
                tensor.dtype.name(),
                tensor.shape,
                ShardDtype::Bf16.name(),
                want
            )));
        }

        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| AudioError::Reader(error.token()))?;

        // BF16 to F32 is exact, so widening changes no value; the reference
        // golden this port is gated against widened the same way.
        let values = self.widen();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let device = self
            .gpu
            .alloc(Dtype::F32, values.len())
            .map_err(run(name))?;
        self.gpu.upload(&device, &bytes).map_err(run(name))?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }

    fn widen(&self) -> Vec<f32> {
        self.buffer
            .chunks_exact(2)
            .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect()
    }

    fn conv(
        &mut self,
        shard: &Shard,
        name: &str,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        stride: u32,
    ) -> Result<Conv, AudioError> {
        let weight = self.pull(
            shard,
            &format!("{ENCODER_PREFIX}{name}.conv.weight"),
            &[
                output_channels as usize,
                input_channels as usize,
                kernel as usize,
                kernel as usize,
            ],
        )?;
        let bias = self.pull(
            shard,
            &format!("{ENCODER_PREFIX}{name}.conv.bias"),
            &[output_channels as usize],
        )?;

        Ok(Conv {
            weight,
            bias,
            input_channels,
            output_channels,
            kernel,
            stride,
        })
    }

    fn scalar(&mut self, shard: &Shard, name: &str, width: u32) -> Result<Vec<f32>, AudioError> {
        let tensor = shard
            .find(&format!("{STATS_PREFIX}{name}"))
            .ok_or_else(|| AudioError::Missing(format!("{STATS_PREFIX}{name}")))?;
        if tensor.dtype != ShardDtype::Bf16 || tensor.shape != [width as u64] {
            return Err(AudioError::Shape(format!(
                "{STATS_PREFIX}{name}: {} {:?}, the graph needs {} x BF16",
                tensor.dtype.name(),
                tensor.shape,
                width
            )));
        }

        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| AudioError::Reader(error.token()))?;
        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(self.widen())
    }
}

/// The release's log mel for one clip: `pcm` is interleaved little-endian F32
/// at [`SOURCE_RATE`]. `AudioProcessor.waveform_to_mel` (`ops.py:44-55`)
/// resamples, takes the STFT, the slaney mel bank, the log and the
/// `[b][c][t][f]` to `[b][c][f][t]` permutation; this returns the
/// channels-last plane the convolution reads.
pub fn log_mel(pcm: &[f32], channels: usize, rate: u32) -> Result<Mel, AudioError> {
    if channels != MEL_CHANNELS {
        return Err(AudioError::Shape(format!(
            "the audio VAE takes stereo, not {channels} channels"
        )));
    }
    if rate != SOURCE_RATE {
        return Err(AudioError::Shape(format!(
            "the stage-1 PCM rate is {SOURCE_RATE} Hz, not {rate}; the release resamples from the clip's own rate, which this port does not"
        )));
    }
    if pcm.len() % channels != 0 {
        return Err(AudioError::Shape(
            "the PCM does not hold whole frames".into(),
        ));
    }

    let frames = pcm.len() / channels;
    let mut planes = Vec::with_capacity(channels);
    for channel in 0..channels {
        let samples: Vec<f32> = pcm
            .iter()
            .skip(channel)
            .step_by(channels)
            .copied()
            .collect();
        if samples.len() != frames {
            return Err(AudioError::Shape("short channel".into()));
        }
        planes.push(mel_spectrogram(&resample(
            &samples,
            rate as usize,
            SAMPLE_RATE as usize,
        )));
    }

    let mel_frames = planes[0].len() / MEL_BINS;
    if planes
        .iter()
        .any(|plane| plane.len() / MEL_BINS != mel_frames)
    {
        return Err(AudioError::Shape("the channels disagree on frames".into()));
    }

    // `[channel][frame][bin]` to the convolution's `[frame][bin][channel]`,
    // with the reference's `log(clamp(x, 1e-5))` on the way (`ops.py:52`).
    let mut values = vec![0.0f32; mel_frames * MEL_BINS * channels];
    for (channel, plane) in planes.iter().enumerate() {
        for frame in 0..mel_frames {
            for bin in 0..MEL_BINS {
                let at = (frame * MEL_BINS + bin) * channels + channel;
                values[at] = plane[frame * MEL_BINS + bin].max(LOG_FLOOR).ln();
            }
        }
    }

    Ok(Mel {
        channels,
        frames: mel_frames,
        bins: MEL_BINS,
        values,
    })
}

fn gcd(left: usize, right: usize) -> usize {
    if right == 0 {
        left
    } else {
        gcd(right, left % right)
    }
}

/// `torchaudio.functional.resample` (`functional.py` `resample`):
/// `sinc_interp_hann`'s kernel, one output channel per phase, and the
/// interleave `_apply_sinc_resample_kernel` performs on its concatenated
/// output.
pub fn resample(input: &[f32], source: usize, target: usize) -> Vec<f32> {
    let divisor = gcd(source, target);
    let (orig, new) = (source / divisor, target / divisor);
    let width =
        (RESAMPLE_WIDTH * orig as f64 / (orig.min(new) as f64 * RESAMPLE_ROLLOFF)).ceil() as usize;
    let taps = 2 * width + orig;
    let lowpass = orig.min(new) as f64 * RESAMPLE_ROLLOFF;

    // `_get_sinc_resample_kernel`: a Hann window over the sinc, one kernel per
    // phase, each scaled by `lowpass / orig`.
    let mut kernel = vec![0.0f64; new * taps];
    for phase in 0..new {
        for tap in 0..taps {
            let offset = (tap as f64 - width as f64) / orig as f64;
            let mut time = (phase as f64) / new as f64 + offset;
            time *= lowpass;
            time = time.clamp(-RESAMPLE_WIDTH, RESAMPLE_WIDTH);
            let window = (time * std::f64::consts::PI / RESAMPLE_WIDTH / 2.0)
                .cos()
                .powi(2);
            let scaled = time * std::f64::consts::PI;
            let sinc = if scaled == 0.0 {
                1.0
            } else {
                scaled.sin() / scaled
            };
            kernel[phase * taps + tap] = sinc * window * (lowpass / orig as f64);
        }
    }

    let mut padded = vec![0.0f32; width + input.len() + width + orig];
    padded[width..width + input.len()].copy_from_slice(input);

    let out_len = (new as f64 * input.len() as f64 / orig as f64).ceil() as usize;
    let mut out = vec![0.0f32; out_len];
    for index in 0..out_len {
        let (position, phase) = (index / new, index % new);
        let mut acc = 0.0f64;
        for tap in 0..taps {
            acc += padded[position * orig + tap] as f64 * kernel[phase * taps + tap];
        }
        out[index] = acc as f32;
    }

    out
}

fn hz_to_mel(frequency: f64) -> f64 {
    let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
    if frequency >= SLANEY_MIN_LOG_HZ {
        min_log_mel + (frequency / SLANEY_MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    } else {
        frequency / SLANEY_F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
    if mel >= min_log_mel {
        SLANEY_MIN_LOG_HZ * (SLANEY_LOG_STEP * (mel - min_log_mel)).exp()
    } else {
        mel * SLANEY_F_SP
    }
}

/// `torchaudio.functional.melscale_fbanks(..., mel_scale="slaney",
/// norm="slaney")` over `[0, 8 kHz]`.
fn mel_bank() -> Vec<Vec<f64>> {
    let bins = N_FFT / 2 + 1;
    let freqs: Vec<f64> = (0..bins)
        .map(|index| (SAMPLE_RATE as f64 / 2.0) * index as f64 / (bins as f64 - 1.0))
        .collect();
    let low = hz_to_mel(0.0);
    let high = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| low + (high - low) * index as f64 / (MEL_BINS as f64 + 1.0))
        .collect();
    let hz: Vec<f64> = points.iter().map(|mel| mel_to_hz(*mel)).collect();

    let mut bank = vec![vec![0.0f64; MEL_BINS]; bins];
    for (bin, frequency) in freqs.iter().enumerate() {
        for mel in 0..MEL_BINS {
            let (left, centre, right) = (hz[mel], hz[mel + 1], hz[mel + 2]);
            let rising = (frequency - left) / (centre - left);
            let falling = (right - frequency) / (right - centre);
            bank[bin][mel] = rising.min(falling).max(0.0) * (2.0 / (right - left));
        }
    }

    bank
}

/// Radix-2 in-place FFT, the `torch.stft` transform for a 1024-point window.
pub fn fft(real: &mut [f64], imaginary: &mut [f64]) {
    let length = real.len();
    let mut j = 0usize;
    for i in 1..length {
        let mut bit = length >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            real.swap(i, j);
            imaginary.swap(i, j);
        }
    }

    let mut span = 2;
    while span <= length {
        let angle = -2.0 * std::f64::consts::PI / span as f64;
        let (cosine, sine) = (angle.cos(), angle.sin());
        let mut start = 0;
        while start < length {
            let (mut wr, mut wi) = (1.0f64, 0.0f64);
            for k in 0..span / 2 {
                let (ur, ui) = (real[start + k], imaginary[start + k]);
                let half = start + k + span / 2;
                let (vr, vi) = (
                    real[half] * wr - imaginary[half] * wi,
                    real[half] * wi + imaginary[half] * wr,
                );
                real[start + k] = ur + vr;
                imaginary[start + k] = ui + vi;
                real[half] = ur - vr;
                imaginary[half] = ui - vi;
                let next = wr * cosine - wi * sine;
                wi = wr * sine + wi * cosine;
                wr = next;
            }
            start += span;
        }
        span <<= 1;
    }
}

/// `MelSpectrogram(center=True, pad_mode="reflect", power=1.0)` over a
/// 1024-point periodic Hann window, returning the linear mel in
/// `[frame][bin]` order.
pub fn mel_spectrogram(input: &[f32]) -> Vec<f32> {
    let half = N_FFT / 2;
    let mut padded = vec![0.0f32; input.len() + 2 * half];
    // Reflect: `F.pad(mode="reflect")` mirrors the edge sample's neighbour, so
    // index `i` walks back to `x[i + 1]` and the tail to `x[len - 2 - i]`.
    for i in 0..half {
        padded[half - 1 - i] = input[i + 1];
        padded[half + input.len() + i] = input[input.len() - 2 - i];
    }
    padded[half..half + input.len()].copy_from_slice(input);

    let window: Vec<f64> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
        .collect();
    let bank = mel_bank();
    let bins = N_FFT / 2 + 1;
    let frames = 1 + input.len() / HOP;
    let mut out = vec![0.0f32; frames * MEL_BINS];
    let (mut real, mut imaginary) = (vec![0.0f64; N_FFT], vec![0.0f64; N_FFT]);
    let mut magnitude = vec![0.0f64; bins];

    for frame in 0..frames {
        for n in 0..N_FFT {
            real[n] = padded[frame * HOP + n] as f64 * window[n];
            imaginary[n] = 0.0;
        }
        fft(&mut real, &mut imaginary);
        for bin in 0..bins {
            magnitude[bin] = (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]).sqrt();
        }
        for mel in 0..MEL_BINS {
            let mut acc = 0.0f64;
            for bin in 0..bins {
                acc += magnitude[bin] * bank[bin][mel];
            }
            out[frame * MEL_BINS + mel] = acc as f32;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encoder_grid_is_the_release_shape() {
        let shapes = Encoder::shapes(518);
        let last = shapes.last().copied().unwrap();

        assert_eq!(shapes[0].channels, 128);
        assert_eq!(shapes[1].frames, 259);
        assert_eq!(shapes[1].bins, 32);
        assert_eq!(last.channels, 512);
        assert_eq!(last.frames, 130);
        assert_eq!(last.bins, 16);
    }

    #[test]
    fn the_pads_are_the_reference_ones() {
        assert_eq!(pads(KERNEL, 1), [2, 0, 1, 1]);
        assert_eq!(pads(KERNEL, DOWNSAMPLE_STRIDE), [2, 0, 0, 1]);
        assert_eq!(pads(SHORTCUT_KERNEL, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn the_downsamples_land_on_the_golden_frames() {
        let top = Shape {
            channels: 128,
            frames: 518,
            bins: 64,
        };
        let first = downsampled(top);
        let second = downsampled(Shape {
            channels: 256,
            ..first
        });

        assert_eq!((first.frames, first.bins), (259, 32));
        assert_eq!((second.frames, second.bins), (130, 16));
    }

    #[test]
    fn the_release_keeps_the_first_126_frames() {
        let latent = AudioLatent {
            channels: 2,
            frames: 4,
            bins: 2,
            values: (0..16).map(|value| value as f32).collect(),
        };
        let conformed = latent.conformed(3);

        assert_eq!(conformed.frames, 3);
        assert_eq!(&conformed.values[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&conformed.values[6..], &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

        let longer = latent.conformed(6);
        assert_eq!(
            &longer.values[..8],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(&longer.values[8..12], &[0.0; 4]);
        assert_eq!(
            &longer.values[12..20],
            &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        );
        assert_eq!(&longer.values[20..], &[0.0; 4]);
    }

    #[test]
    fn the_resampler_reduces_the_rate_by_two() {
        let input: Vec<f32> = (0..64).map(|index| (index as f32 * 0.1).sin()).collect();
        let output = resample(&input, SOURCE_RATE as usize, SAMPLE_RATE as usize);

        assert_eq!(output.len(), input.len() / 2);
    }
}
