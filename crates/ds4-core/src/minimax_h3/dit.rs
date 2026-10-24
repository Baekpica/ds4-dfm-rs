//! The MiniMax-H3 FL2VA DiT forward.
//!
//! One host loop over the device primitives in `native/h3/h3_dit_gpu.h`, plus
//! the AdaLN precompute, the RoPE tables and the packed-sequence glue. Source
//! of truth is `/data/h3.c` at `eae5d54`:
//!
//! - `h3_dit.c:1559-1700` `load_dit`: the order of the load (text refinement,
//!   AdaLN precompute, RoPE, row maps, core, activations) and the flags that
//!   select the CUDA BF16 path.
//! - `h3_dit.c:804-881` `refine_text`: `condition_proj`, two token-refiner
//!   blocks and a final RMS norm. The refiner blocks have no AdaLN; they use
//!   plain RMS norms, full attention and residual adds.
//! - `h3_dit_schedule.c:235-331` `h3_dit_schedule_precompute`: the time
//!   features, the F32 timestep embedder, then one BF16 `adaln_proj` per block
//!   and the final projection.
//! - `h3_dit.c:2050-2396` `encode_forward`: segment packing, the 50-block loop
//!   and the final heads, including the cross-block AdaLN fusion the CUDA path
//!   uses by default.
//! - `h3_dit.c:2442-2505` `h3_dit_forward`: patchify/pack on the host, one
//!   `encode_forward`, then unpatchify/unpack.
//!
//! ```text
//! load:  text -> refiner -> refined_text
//!        sigmas -> time features -> time embedder -> adaln_proj[50] + final
//!        positions + inv_freq -> cos/sin
//!        row maps per step
//!        blocks[50] weights, patch/final weights, activations
//! step:  pack text | patch video | patch audio        (one contiguous hidden)
//!        block 0..49: adaln, qkv+rope, attention, out, gate+adaln, mlp, gate
//!        final: adaln(norm) + audio head + video head
//! ```
//!
//! Deviations, named and measured where they can be:
//!
//! - The reference's fused kernels (`h3_gpu_gate_adaln_bf16`,
//!   `h3_gpu_adaln_linear_bf16`, `h3_gpu_mlp_bf16`) are sequences of the same
//!   kernels this file calls; `h3_dit.c:3385-3391` is `gate` then `adaln` and
//!   `h3_gpu_cuda.cu:2203-2218` is `linear`, `swiglu`, `linear`. The port calls
//!   the parts, so the arithmetic is identical and the fusion is a launch
//!   count, not a numerical choice.
//! - `h3_gpu_adaln_linear_bf16` (`:1195-1226`) also writes an `inverse` buffer
//!   that no later call reads; the effective math is `adaln` then `linear`,
//!   which is what the final head here does.
//! - int8, NAX and SSD-streaming variants are not ported: CUDA reports
//!   `h3_gpu_has_int8_mlp`/`h3_gpu_has_nax_mlp` as 0 (`h3_gpu_cuda.cu:332-334`)
//!   and the serving path passes `ssd_streaming = 0`, so the reference's own
//!   CUDA run takes exactly the path implemented here.
//! - The weight stream is load-everything-then-run. The reference can stream
//!   matrix slots from SSD; that and the hot/cold AdaLN split are P9 items.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use crate::minimax_h3::host::Schedule;
use crate::minimax_h3::layout::{Config, Role};
use crate::minimax_h3::sequence::{SegmentKind, Sequence};
use crate::minimax_h3::text_forward::{fnv1a, Embedding};
use crate::minimax_h3::weights::Weights;

/// `H3_DIT_MODALITIES` (`h3_dit_schedule.h:13`).
pub const MODALITIES: u32 = 3;
/// `H3_DIT_ADALN_SLOTS` (`h3_dit_schedule.h:14`).
pub const SLOTS: u32 = 6;
/// `FINAL_SLOTS` (`h3_dit.c:29`): the final head has a shift and a scale only.
pub const FINAL_SLOTS: u32 = 2;
/// `SLOTS` as the block AdaLN kernel receives it (`h3_dit.c:28`).
pub const BLOCK_SLOTS: u32 = 6;
/// `AUDIO_STREAMS` (`h3_dit.c:26`).
pub const AUDIO_STREAMS: u32 = 2;
/// `TIME_INPUT`: the sinusoid width (`h3_dit_schedule.c:12`).
pub const TIME_INPUT: usize = 256;
/// `ROPE_FREQS` (`h3_dit.c:27`): 16 frequencies over three axes.
pub const ROPE_FREQS: usize = 16;
/// `ROPE_HALF` (`h3_dit.c:28`): `3 * ROPE_FREQS` rotated pairs per row.
pub const ROPE_HALF: usize = 3 * ROPE_FREQS;
/// The RMS norm epsilon every DiT call passes (`h3_dit.c:1901`).
pub const EPSILON: f32 = 1e-5;

/// `ROPE_FREQS` in the checkpoint's `rope.inv_freq`.
const INV_FREQ_ELEMENTS: usize = 16;

#[derive(Debug)]
pub enum DitError {
    /// A role is missing or unreadable in the checkpoint.
    Weights(String),
    /// A device call failed; the text is the native message.
    Gpu(String),
    /// The packed sequence and the checkpoint geometry disagree.
    Geometry { detail: String },
    /// A layout this unit does not implement was requested.
    Unsupported { detail: String },
}

impl fmt::Display for DitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DitError::Weights(detail) => write!(formatter, "weights: {detail}"),
            DitError::Gpu(detail) => write!(formatter, "device: {detail}"),
            DitError::Geometry { detail } => write!(formatter, "geometry: {detail}"),
            DitError::Unsupported { detail } => write!(formatter, "unsupported: {detail}"),
        }
    }
}

impl std::error::Error for DitError {}

fn device<T>(result: Result<T, String>) -> Result<T, DitError> {
    result.map_err(DitError::Gpu)
}

/// One block's eight BF16 matrices plus its norm weights
/// (`h3_dit.c:502-525`).
struct Block {
    norm1: Tensor,
    norm2: Tensor,
    qkv: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    out: Tensor,
    fc1: Tensor,
    fc2: Tensor,
}

/// `final_layer` (`h3_dit.c:1295-1311`). The released heads are F32 and the
/// reference converts them to BF16 once at load (`h3_dit.c:1312-1354`).
struct Final {
    norm: Tensor,
    video_weight: Tensor,
    video_bias: Tensor,
    audio_weight: Tensor,
    audio_bias: Tensor,
}

/// The F32 patch projections (`h3_dit.c:1290-1299`).
struct Patches {
    video_weight: Tensor,
    video_bias: Tensor,
    audio_weight: Tensor,
    audio_bias: Tensor,
}

/// The AdaLN row bookkeeping `h3_dit_schedule.c:79-160` builds before any
/// tensor exists.
#[derive(Debug, Clone, PartialEq)]
struct TimeRows {
    steps: usize,
    video: Vec<u32>,
    audio: Vec<u32>,
    visual_condition: Option<Vec<u32>>,
    audio_condition: Option<Vec<u32>>,
    times: Vec<f32>,
}

/// `h3_dit_schedule.c:79-160`. The video and audio timesteps share a row when
/// they are equal, otherwise the smaller one takes the lower row; condition
/// rows are appended after the per-step rows.
fn time_rows(schedule: &Schedule, visual_condition: bool, audio_condition: bool) -> TimeRows {
    let steps = schedule.steps;
    let mut video = vec![0u32; steps];
    let mut audio = vec![0u32; steps];
    let mut count = 0u32;

    for step in 0..steps {
        let video_time = 1.0f32 - schedule.video[step];
        let audio_time = 1.0f32 - schedule.audio[step];
        if video_time == audio_time {
            video[step] = count;
            audio[step] = count;
            count += 1;
        } else if video_time < audio_time {
            video[step] = count;
            count += 1;
            audio[step] = count;
            count += 1;
        } else {
            audio[step] = count;
            count += 1;
            video[step] = count;
            count += 1;
        }
    }

    let mut visual_condition_row = u32::MAX;
    let mut audio_condition_row = u32::MAX;
    if visual_condition {
        visual_condition_row = count;
        count += 1;
    }
    if audio_condition {
        audio_condition_row = count;
        count += 1;
    }

    let visual_rows = visual_condition.then(|| {
        (0..steps)
            .map(|step| {
                let video_time = 1.0f32 - schedule.video[step];
                if video_time >= 0.999f32 {
                    video[step]
                } else {
                    visual_condition_row
                }
            })
            .collect()
    });
    let audio_rows = audio_condition.then(|| {
        (0..steps)
            .map(|step| {
                let audio_time = 1.0f32 - schedule.audio[step];
                if audio_time >= 1.0f32 {
                    audio[step]
                } else {
                    audio_condition_row
                }
            })
            .collect()
    });

    let mut times = vec![0.0f32; count as usize];
    for step in 0..steps {
        times[video[step] as usize] = 1.0f32 - schedule.video[step];
        times[audio[step] as usize] = 1.0f32 - schedule.audio[step];
    }
    if visual_condition {
        times[visual_condition_row as usize] = 0.999f32;
    }
    if audio_condition {
        times[audio_condition_row as usize] = 1.0f32;
    }

    TimeRows {
        steps,
        video,
        audio,
        visual_condition: visual_rows,
        audio_condition: audio_rows,
        times,
    }
}

/// `h3_dit_schedule.c:147-157`: cos on the first half, sin on the second, at
/// `expf(-logf(10000) * index / 128)`.
fn time_features(times: &[f32]) -> Vec<f32> {
    let half = TIME_INPUT / 2;
    let mut features = vec![0.0f32; times.len() * TIME_INPUT];

    for (row, time) in times.iter().enumerate() {
        for index in 0..half {
            let frequency = (-(10_000.0f32).ln()) * (index as f32) / (half as f32);
            let frequency = frequency.exp();
            let angle = *time * frequency;
            features[row * TIME_INPUT + index] = angle.cos();
            features[row * TIME_INPUT + half + index] = angle.sin();
        }
    }

    features
}

/// `h3_dit_schedule.c:380-445` `h3_dit_schedule_row_map`: one modulation row
/// selector per sequence row, `time_row * MODALITIES + tag`.
fn row_map(tree: &TimeRows, step: usize, sequence: &Sequence) -> Result<Vec<u32>, DitError> {
    let mut rows = vec![0u32; sequence.rows()];
    let mut text_index = 0usize;

    for segment in &sequence.segments {
        let (time_row, tag) = match segment.kind {
            SegmentKind::Text => {
                let time_row = tree.video[step];
                for cell in &mut rows[segment.start..segment.stop] {
                    *cell = time_row * MODALITIES + 1;
                    text_index += 1;
                }
                continue;
            }
            SegmentKind::Cond | SegmentKind::RefImage => (
                tree.visual_condition
                    .as_ref()
                    .ok_or_else(|| DitError::Geometry {
                        detail: "a condition segment needs the visual condition row".into(),
                    })?[step],
                0,
            ),
            SegmentKind::RefAudio => (
                tree.audio_condition
                    .as_ref()
                    .ok_or_else(|| DitError::Geometry {
                        detail: "a reference-audio segment needs the audio condition row".into(),
                    })?[step],
                2,
            ),
            SegmentKind::Audio => (tree.audio[step], 2),
            SegmentKind::Video => (tree.video[step], 0),
        };

        let modulation = time_row * MODALITIES + tag;
        for cell in &mut rows[segment.start..segment.stop] {
            *cell = modulation;
        }
    }

    if text_index != sequence.signature[0] as usize {
        return Err(DitError::Geometry {
            detail: "text segment rows disagree with the sequence signature".into(),
        });
    }
    Ok(rows)
}

/// `h3_dit.c:3064-3097`: `[C,T,H,W]` latent to `[T,H/2,W/2,C,2,2]` rows.
pub fn patchify_video(
    latent: &[f32],
    channels: usize,
    time: usize,
    height: usize,
    width: usize,
    rows: &mut [f32],
) -> Result<(), DitError> {
    let expected = latent_shape(channels, time, height, width)?;
    if latent.len() != expected || rows.len() != expected {
        return Err(DitError::Geometry {
            detail: format!(
                "patchify of {} values into {} rows, expected {expected}",
                latent.len(),
                rows.len()
            ),
        });
    }

    let mut output = 0usize;
    for t in 0..time {
        for h in (0..height).step_by(2) {
            for w in (0..width).step_by(2) {
                for c in 0..channels {
                    for dh in 0..2 {
                        for dw in 0..2 {
                            let source = ((c * time + t) * height + h + dh) * width + w + dw;
                            rows[output] = latent[source];
                            output += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// `h3_dit.c:3099-3119`: the exact inverse of `patchify_video`.
pub fn unpatchify_video(
    rows: &[f32],
    channels: usize,
    time: usize,
    height: usize,
    width: usize,
    latent: &mut [f32],
) -> Result<(), DitError> {
    let expected = latent_shape(channels, time, height, width)?;
    if rows.len() != expected || latent.len() != expected {
        return Err(DitError::Geometry {
            detail: format!(
                "unpatchify of {} rows into a {} value latent, expected {expected}",
                rows.len(),
                latent.len()
            ),
        });
    }

    let mut input = 0usize;
    for t in 0..time {
        for h in (0..height).step_by(2) {
            for w in (0..width).step_by(2) {
                for c in 0..channels {
                    for dh in 0..2 {
                        for dw in 0..2 {
                            let destination = ((c * time + t) * height + h + dh) * width + w + dw;
                            latent[destination] = rows[input];
                            input += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// `h3_dit.c:3121-3136`: `[C,2,T]` audio latent to `[stream,T,C]` rows.
pub fn pack_audio(
    latent: &[f32],
    channels: usize,
    time: usize,
    rows: &mut [f32],
) -> Result<(), DitError> {
    let expected = audio_shape(channels, time)?;
    if latent.len() != expected || rows.len() != expected {
        return Err(DitError::Geometry {
            detail: format!(
                "pack_audio of {} values into {} rows, expected {expected}",
                latent.len(),
                rows.len()
            ),
        });
    }

    let mut output = 0usize;
    for stream in 0..AUDIO_STREAMS as usize {
        for t in 0..time {
            for channel in 0..channels {
                rows[output] = latent[(channel * 2 + stream) * time + t];
                output += 1;
            }
        }
    }
    Ok(())
}

/// `h3_dit.c:3137-3151`: the exact inverse of `pack_audio`.
pub fn unpack_audio(
    rows: &[f32],
    channels: usize,
    time: usize,
    latent: &mut [f32],
) -> Result<(), DitError> {
    let expected = audio_shape(channels, time)?;
    if rows.len() != expected || latent.len() != expected {
        return Err(DitError::Geometry {
            detail: format!(
                "unpack_audio of {} rows into a {} value latent, expected {expected}",
                rows.len(),
                latent.len()
            ),
        });
    }

    let mut input = 0usize;
    for stream in 0..AUDIO_STREAMS as usize {
        for t in 0..time {
            for channel in 0..channels {
                latent[(channel * 2 + stream) * time + t] = rows[input];
                input += 1;
            }
        }
    }
    Ok(())
}

/// `h3_dit.c:3064-3076` `video_shape`.
fn latent_shape(
    channels: usize,
    time: usize,
    height: usize,
    width: usize,
) -> Result<usize, DitError> {
    if channels < 1
        || time < 1
        || height < 2
        || width < 2
        || !height.is_multiple_of(2)
        || !width.is_multiple_of(2)
    {
        return Err(DitError::Geometry {
            detail: format!("invalid video latent shape {channels}x{time}x{height}x{width}"),
        });
    }
    Ok(channels * time * height * width)
}

/// `h3_dit.c:3121-3126`, the audio element count check.
fn audio_shape(channels: usize, time: usize) -> Result<usize, DitError> {
    if channels < 1 || time < 1 {
        return Err(DitError::Geometry {
            detail: format!("invalid audio latent shape {channels}x2x{time}"),
        });
    }
    Ok(channels * 2 * time)
}

/// The row counts `validate_layout` derives before any tensor is allocated
/// (`h3_dit.c:292-377`).
struct Geometry {
    text_rows: u32,
    video_cond_rows: u32,
    audio_cond_rows: u32,
    video_rows: u32,
    audio_rows: u32,
    video_total_rows: u32,
    audio_total_rows: u32,
    video_target_start: u32,
    audio_target_start: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    audio_t: u32,
}

impl Geometry {
    /// `h3_dit.c:292-377` `validate_layout`, with the columns the forward
    /// indexes. It refuses anything the segment walk cannot account for rather
    /// than trusting the caller's counts.
    fn check(
        config: &Config,
        sequence: &Sequence,
        text_tokens: usize,
    ) -> Result<Geometry, DitError> {
        let signature = sequence.signature;
        if text_tokens as i32 != signature[0]
            || signature[1] < 1
            || signature[2] < 2
            || signature[3] < 2
            || signature[2] % 2 != 0
            || signature[3] % 2 != 0
            || sequence.segments.is_empty()
        {
            return Err(DitError::Geometry {
                detail: "the packed sequence signature is not usable".into(),
            });
        }
        if sequence.segments[0].kind != SegmentKind::Text
            || sequence.segments[sequence.segments.len() - 1].kind != SegmentKind::Video
        {
            return Err(DitError::Geometry {
                detail: "a packed sequence must start with text and end with video".into(),
            });
        }
        if config.head_dim != 128 {
            return Err(DitError::Unsupported {
                detail: format!(
                    "head_dim {} selects the reference's generic attention; this port \
                     implements the tiled 128-wide kernel the H3 checkpoint uses",
                    config.head_dim
                ),
            });
        }

        let mut cursor = 0usize;
        let mut text_rows = 0u32;
        let mut video_condition = 0u32;
        let mut audio_condition = 0u32;
        let mut video_target = 0u32;
        let mut audio_target = 0u32;
        let mut video_target_start = 0u32;
        let mut audio_target_start = 0u32;
        let mut video_segments = 0;
        let mut audio_segments = 0;

        for segment in &sequence.segments {
            if segment.start != cursor
                || segment.stop < segment.start
                || segment.stop > sequence.rows()
            {
                return Err(DitError::Geometry {
                    detail: "packed sequence segments are not contiguous".into(),
                });
            }
            let rows = (segment.stop - segment.start) as u32;
            match segment.kind {
                SegmentKind::Text => text_rows += rows,
                SegmentKind::Cond | SegmentKind::RefImage => video_condition += rows,
                SegmentKind::RefAudio => audio_condition += rows,
                SegmentKind::Audio => {
                    audio_target += rows;
                    audio_target_start = segment.start as u32;
                    audio_segments += 1;
                }
                SegmentKind::Video => {
                    video_target += rows;
                    video_target_start = segment.start as u32;
                    video_segments += 1;
                }
            }
            cursor = segment.stop;
        }

        if cursor != sequence.rows()
            || text_rows as usize != text_tokens
            || video_condition as usize != sequence.img_cond_rows
            || audio_condition as usize != sequence.audio_cond_rows
            || video_target as usize != sequence.img_target_rows
            || audio_target as usize != sequence.audio_target_rows
            || video_segments != 1
            || audio_segments != 1
        {
            return Err(DitError::Geometry {
                detail: "packed sequence row sources are inconsistent".into(),
            });
        }

        Ok(Geometry {
            text_rows,
            video_cond_rows: video_condition,
            audio_cond_rows: audio_condition,
            video_rows: video_target,
            audio_rows: audio_target,
            video_total_rows: video_condition + video_target,
            audio_total_rows: audio_condition + audio_target,
            video_target_start,
            audio_target_start,
            latent_t: signature[1] as u32,
            latent_h: signature[2] as u32,
            latent_w: signature[3] as u32,
            audio_t: signature[4] as u32,
        })
    }
}

/// The FL2VA DiT with its weights resident: one device context, the loaded
/// core, and the per-step tensors the forward indexes.
pub struct Dit {
    gpu: Gpu,
    config: Config,
    sequence: Sequence,
    schedule: Schedule,
    hidden_width: u32,
    inner: u32,
    video_width: u32,
    audio_width: u32,

    rows: u32,
    video_cond_rows: u32,
    audio_cond_rows: u32,
    video_rows: u32,
    audio_rows: u32,
    video_total_rows: u32,
    audio_total_rows: u32,
    video_target_start: u32,
    audio_target_start: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    audio_t: u32,

    blocks: Vec<Block>,
    final_layer: Final,
    patches: Patches,

    modulation: Vec<Tensor>,
    final_modulation: Tensor,
    row_maps: Vec<Tensor>,
    final_audio_maps: Vec<Tensor>,
    final_video_maps: Vec<Tensor>,
    video_projection_map: Option<Tensor>,
    audio_projection_map: Option<Tensor>,

    refined_text: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,

    hidden: Tensor,
    mod_attention: Tensor,
    qkv: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_output: Tensor,
    fc1: Tensor,
    activated: Tensor,
    final_norm_out: Tensor,
    video_input: Tensor,
    audio_input: Tensor,
    video_output: Tensor,
    audio_output: Tensor,
}

impl Dit {
    /// Load and prepare every tensor `h3_dit.c:1559-1700` prepares, in the
    /// reference's order.
    ///
    /// `spatial_rope_scale` is the reference's own parameter; the CLI halves it
    /// for 256x256 canvases unless `--reference-rope` is passed
    /// (`h3.c:1457-1459`).
    pub fn load(
        dir: &Path,
        sequence: Sequence,
        schedule: Schedule,
        text: &Embedding,
        spatial_rope_scale: f32,
    ) -> Result<Dit, DitError> {
        let weights = Weights::open(dir).map_err(|error| DitError::Weights(error.to_string()))?;
        let config = weights.config().clone();
        let gpu = device(Gpu::create())?;

        let geometry = Geometry::check(&config, &sequence, text.tokens())?;
        let hidden_width = config.hidden as u32;
        let inner = config.inner() as u32;
        let video_width = config.video_patch() as u32;
        let audio_width = config.audio_latents as u32;

        // `h3_dit.c:1655-1665`: text refinement precedes the schedule.
        let refined_text = refine_text(&gpu, &weights, &config, text, geometry.text_rows)?;

        // `h3_dit.c:1666-1684`: AdaLN for this schedule, then RoPE, then maps.
        let tree = time_rows(
            &schedule,
            sequence.img_cond_rows != 0,
            sequence.audio_cond_rows != 0,
        );
        let (_, modulation, final_modulation) = adaln_projections(&gpu, &weights, &config, &tree)?;
        let (rope_cos, rope_sin) =
            rope_tables(&gpu, &weights, &config, &sequence, spatial_rope_scale)?;
        let (row_maps, final_audio_maps, final_video_maps) =
            modulation_maps(&gpu, &tree, &sequence)?;

        // `h3_dit.c:1685-1688`: the core, then the activation arena.
        let blocks = load_blocks(&gpu, &weights, &config)?;
        let final_layer = load_final(&gpu, &weights)?;
        let patches = load_patches(&gpu, &weights)?;
        let (video_projection_map, audio_projection_map) = projection_maps(&gpu, &sequence)?;

        let bf16 = |elements: usize| gpu_alloc(&gpu, Dtype::Bf16, elements);
        let f32t = |elements: usize| gpu_alloc(&gpu, Dtype::F32, elements);
        let rows = sequence.rows();

        // `h3_dit.c:1356-1553` with `activation_aliases = 1`: `attention_heads`
        // is `qkv`, `mod_mlp` is `qkv` and `mlp_output` is `attention_output`,
        // which is why this file passes those two buffers under both names.
        let hidden = bf16(rows * hidden_width as usize)?;
        let mod_attention = bf16(rows * hidden_width as usize)?;
        let qkv = bf16(rows * inner as usize * 3)?;
        let query = bf16(rows * inner as usize)?;
        let key = bf16(rows * inner as usize)?;
        let value = bf16(rows * inner as usize)?;
        let attention_output = bf16(rows * hidden_width as usize)?;
        let fc1 = bf16(rows * config.ffn as usize * 2)?;
        let activated = bf16(rows * config.ffn as usize)?;
        let final_norm_out =
            bf16(sequence.img_target_rows.max(sequence.audio_target_rows) * hidden_width as usize)?;
        let video_input = f32t(sequence.video_total_rows() * video_width as usize)?;
        let audio_input = f32t(sequence.audio_total_rows() * audio_width as usize)?;
        let video_output = bf16(sequence.img_target_rows * video_width as usize)?;
        let audio_output = bf16(sequence.audio_target_rows * audio_width as usize)?;

        device(gpu.sync())?;

        Ok(Dit {
            gpu,
            config,
            hidden_width,
            inner,
            video_width,
            audio_width,
            rows: rows as u32,
            video_cond_rows: geometry.video_cond_rows,
            audio_cond_rows: geometry.audio_cond_rows,
            video_rows: geometry.video_rows,
            audio_rows: geometry.audio_rows,
            video_total_rows: geometry.video_total_rows,
            audio_total_rows: geometry.audio_total_rows,
            video_target_start: geometry.video_target_start,
            audio_target_start: geometry.audio_target_start,
            latent_t: geometry.latent_t,
            latent_h: geometry.latent_h,
            latent_w: geometry.latent_w,
            audio_t: geometry.audio_t,
            sequence,
            schedule,
            blocks,
            final_layer,
            patches,
            modulation,
            final_modulation,
            row_maps,
            final_audio_maps,
            final_video_maps,
            video_projection_map,
            audio_projection_map,
            refined_text,
            rope_cos,
            rope_sin,
            hidden,
            mod_attention,
            qkv,
            query,
            key,
            value,
            attention_output,
            fc1,
            activated,
            final_norm_out,
            video_input,
            audio_input,
            video_output,
            audio_output,
        })
    }

    /// `h3_dit_video_elements` (`h3_dit.c:2398-2401`).
    pub fn video_elements(&self) -> usize {
        24 * self.latent_t as usize * self.latent_h as usize * self.latent_w as usize
    }

    /// `h3_dit_audio_elements` (`h3_dit.c:2403-2406`).
    pub fn audio_elements(&self) -> usize {
        32 * AUDIO_STREAMS as usize * self.audio_t as usize
    }

    /// The modulation tensors, exposed so a caller can gate them on their own:
    /// a defect here moves every block and the final head at once.
    pub fn modulation_hashes(&self) -> Result<(u64, u64), DitError> {
        let block = self.modulation.first().ok_or_else(|| DitError::Geometry {
            detail: "no modulation".into(),
        })?;
        Ok((
            hash_bf16(&self.gpu, block)?,
            hash_bf16(&self.gpu, &self.final_modulation)?,
        ))
    }

    /// `h3_dit_forward` (`h3_dit.c:2442-2505`): pack the host latents, run the
    /// graph once at `step`, and return both velocities in the reference's
    /// `[24,T,H,W]` / `[32,2,T]` layouts.
    pub fn forward(
        &self,
        step: usize,
        video_latent: &[f32],
        audio_latent: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), DitError> {
        if step >= self.schedule.steps {
            return Err(DitError::Geometry {
                detail: format!("step {step} outside 0..{}", self.schedule.steps),
            });
        }

        let mut video_rows = vec![0.0f32; self.video_rows as usize * self.video_width as usize];
        let mut audio_rows = vec![0.0f32; self.audio_rows as usize * self.audio_width as usize];
        patchify_video(
            video_latent,
            24,
            self.latent_t as usize,
            self.latent_h as usize,
            self.latent_w as usize,
            &mut video_rows,
        )?;
        pack_audio(audio_latent, 32, self.audio_t as usize, &mut audio_rows)?;

        device(self.gpu.upload_at(
            &self.video_input,
            self.video_cond_rows as usize * self.video_width as usize,
            &pack_f32(&video_rows),
        ))?;
        device(self.gpu.upload_at(
            &self.audio_input,
            self.audio_cond_rows as usize * self.audio_width as usize,
            &pack_f32(&audio_rows),
        ))?;

        self.encode_forward(step)?;

        let mut video_out = vec![0u8; video_rows.len() * 2];
        let mut audio_out = vec![0u8; audio_rows.len() * 2];
        device(self.gpu.read(&self.video_output, &mut video_out))?;
        device(self.gpu.read(&self.audio_output, &mut audio_out))?;

        let video_f32 = bf16_to_f32(&video_out);
        let audio_f32 = bf16_to_f32(&audio_out);

        let mut video_velocity = vec![0.0f32; self.video_elements()];
        unpatchify_video(
            &video_f32,
            24,
            self.latent_t as usize,
            self.latent_h as usize,
            self.latent_w as usize,
            &mut video_velocity,
        )?;
        let mut audio_velocity = vec![0.0f32; self.audio_elements()];
        unpack_audio(&audio_f32, 32, self.audio_t as usize, &mut audio_velocity)?;

        Ok((video_velocity, audio_velocity))
    }

    /// `h3_dit.c:2050-2396` `encode_forward` with `core_reuse_interval = 1`,
    /// `token_reduction = 0`, `ssd_streaming = 0` and the CUDA flag set of
    /// `h3_dit.c:1601-1665`.
    fn encode_forward(&self, step: usize) -> Result<(), DitError> {
        let hidden = self.hidden_width;

        // `h3_dit.c:2058-2158`: the fused patch projection writes each segment
        // straight into `hidden`, so no packed intermediate exists. A layout
        // with more than one video or audio segment projects all rows first and
        // scatters them through the row map instead.
        if let Some(map) = &self.video_projection_map {
            device(self.gpu.patch_linear_map(
                &self.hidden,
                &self.video_input,
                &self.patches.video_weight,
                &self.patches.video_bias,
                map,
                self.rows,
                self.video_total_rows,
                self.video_width,
                hidden,
            ))?;
        }
        if let Some(map) = &self.audio_projection_map {
            device(self.gpu.patch_linear_map(
                &self.hidden,
                &self.audio_input,
                &self.patches.audio_weight,
                &self.patches.audio_bias,
                map,
                self.rows,
                self.audio_total_rows,
                self.audio_width,
                hidden,
            ))?;
        }

        let mut video_offset = 0usize;
        let mut audio_offset = 0usize;
        for segment in &self.sequence.segments {
            let segment_rows = (segment.stop - segment.start) as u32;
            let destination = segment.start * hidden as usize;

            match segment.kind {
                SegmentKind::Text => {
                    device(self.gpu.copy(
                        &self.hidden,
                        destination,
                        &self.refined_text,
                        0,
                        segment_rows as usize * hidden as usize,
                    ))?;
                }
                SegmentKind::Cond | SegmentKind::RefImage | SegmentKind::Video => {
                    if self.video_projection_map.is_none() {
                        device(self.gpu.patch_linear(
                            &self.hidden,
                            destination,
                            &self.video_input,
                            video_offset * self.video_width as usize,
                            &self.patches.video_weight,
                            &self.patches.video_bias,
                            segment_rows,
                            self.video_width,
                            hidden,
                        ))?;
                    }
                    video_offset += segment_rows as usize;
                }
                SegmentKind::RefAudio | SegmentKind::Audio => {
                    if self.audio_projection_map.is_none() {
                        device(self.gpu.patch_linear(
                            &self.hidden,
                            destination,
                            &self.audio_input,
                            audio_offset * self.audio_width as usize,
                            &self.patches.audio_weight,
                            &self.patches.audio_bias,
                            segment_rows,
                            self.audio_width,
                            hidden,
                        ))?;
                    }
                    audio_offset += segment_rows as usize;
                }
            }
        }

        if video_offset != self.video_total_rows as usize
            || audio_offset != self.audio_total_rows as usize
        {
            return Err(DitError::Geometry {
                detail: "segment packing did not consume the row sources".into(),
            });
        }

        // `h3_dit.c:2181-2306`: every block, with the next block's attention
        // AdaLN fused into this block's MLP gate.
        let mut carried = false;
        for block in 0..self.blocks.len() {
            let fuse_next = block + 1 < self.blocks.len();
            self.run_block(block, step, fuse_next, &mut carried)?;
        }

        // `h3_dit.c:2316-2337`: the fused final head, AdaLN then projection,
        // once per modality slice.
        device(self.gpu.adaln(
            &self.final_norm_out,
            &self.hidden,
            self.audio_target_start as usize * hidden as usize,
            &self.final_layer.norm,
            &self.final_modulation,
            &self.final_audio_maps[step],
            self.audio_rows,
            hidden,
            FINAL_SLOTS,
            0,
            1,
            EPSILON,
        ))?;
        device(self.gpu.linear(
            &self.audio_output,
            &self.final_norm_out,
            &self.final_layer.audio_weight,
            Some(&self.final_layer.audio_bias),
            self.audio_rows,
            hidden,
            self.audio_width,
            Dtype::Bf16,
        ))?;

        device(self.gpu.adaln(
            &self.final_norm_out,
            &self.hidden,
            self.video_target_start as usize * hidden as usize,
            &self.final_layer.norm,
            &self.final_modulation,
            &self.final_video_maps[step],
            self.video_rows,
            hidden,
            FINAL_SLOTS,
            0,
            1,
            EPSILON,
        ))?;
        device(self.gpu.linear(
            &self.video_output,
            &self.final_norm_out,
            &self.final_layer.video_weight,
            Some(&self.final_layer.video_bias),
            self.video_rows,
            hidden,
            self.video_width,
            Dtype::Bf16,
        ))?;

        device(self.gpu.sync())
    }

    /// `h3_dit.c:1876-2047` `run_block` with the int8 and NAX branches off.
    ///
    /// `carried` is the `attention_adaln_ready` flag the previous block left
    /// behind (`h3_dit.c:2184-2226`): when set, this block's attention AdaLN
    /// already ran and only the QKV projection is left to do.
    fn run_block(
        &self,
        block: usize,
        step: usize,
        fuse_next_attention: bool,
        carried: &mut bool,
    ) -> Result<(), DitError> {
        let rows = self.rows;
        let hidden = self.hidden_width;
        let inner = self.inner;
        let modulation = &self.modulation[block];
        let row_map = &self.row_maps[step];
        let weight = &self.blocks[block];
        let attention_adaln_ready = *carried;
        *carried = false;

        if !attention_adaln_ready {
            device(self.gpu.adaln(
                &self.mod_attention,
                &self.hidden,
                0,
                &weight.norm1,
                modulation,
                row_map,
                rows,
                hidden,
                BLOCK_SLOTS,
                0,
                1,
                EPSILON,
            ))?;
        }

        // `h3_dit.c:1902-1919`: grouped QKV projection, then the Q/K norm and
        // RoPE boundary.
        device(self.gpu.linear(
            &self.qkv,
            &self.mod_attention,
            &weight.qkv,
            None,
            rows,
            hidden,
            inner * 3,
            Dtype::Bf16,
        ))?;
        device(self.gpu.qkv_rope(
            &self.query,
            &self.key,
            &self.value,
            &self.qkv,
            &weight.q_norm,
            &weight.k_norm,
            &self.rope_cos,
            &self.rope_sin,
            rows,
            self.config.heads as u32,
            self.config.head_dim as u32,
            ROPE_HALF as u32,
            EPSILON,
        ))?;

        // `h3_dit.c:1921-1956`: full attention into the qkv buffer (the
        // reference's `attention_heads` alias), then the output projection.
        device(self.gpu.sdpa(
            &self.qkv,
            &self.query,
            &self.key,
            &self.value,
            rows,
            self.config.heads as u32,
            self.config.head_dim as u32,
            1.0 / ((self.config.head_dim as f32).sqrt()),
        ))?;
        device(self.gpu.linear(
            &self.attention_output,
            &self.qkv,
            &weight.out,
            None,
            rows,
            inner,
            hidden,
            Dtype::Bf16,
        ))?;

        // `h3_dit.c:1958-1982`: the attention gate, then the MLP AdaLN, which
        // reuses the qkv buffer as `mod_mlp`.
        device(self.gpu.gate(
            &self.hidden,
            &self.hidden,
            &self.attention_output,
            modulation,
            row_map,
            rows,
            hidden,
            BLOCK_SLOTS,
            2,
        ))?;
        device(self.gpu.adaln(
            &self.qkv,
            &self.hidden,
            0,
            &weight.norm2,
            modulation,
            row_map,
            rows,
            hidden,
            BLOCK_SLOTS,
            3,
            4,
            EPSILON,
        ))?;

        // `h3_dit.c:1984-2013`: `h3_gpu_mlp_bf16` is linear, SwiGLU, linear.
        device(self.gpu.linear(
            &self.fc1,
            &self.qkv,
            &weight.fc1,
            None,
            rows,
            hidden,
            self.config.ffn as u32 * 2,
            Dtype::Bf16,
        ))?;
        device(
            self.gpu
                .swiglu(&self.activated, &self.fc1, rows, self.config.ffn as u32),
        )?;
        device(self.gpu.linear(
            &self.attention_output,
            &self.activated,
            &weight.fc2,
            None,
            rows,
            self.config.ffn as u32,
            hidden,
            Dtype::Bf16,
        ))?;

        // `h3_dit.c:2015-2046`: the MLP gate, and, when another block follows,
        // the next block's attention AdaLN in the same step.
        device(self.gpu.gate(
            &self.hidden,
            &self.hidden,
            &self.attention_output,
            modulation,
            row_map,
            rows,
            hidden,
            BLOCK_SLOTS,
            5,
        ))?;

        if !fuse_next_attention {
            return Ok(());
        }

        let next = &self.blocks[block + 1];
        device(self.gpu.adaln(
            &self.mod_attention,
            &self.hidden,
            0,
            &next.norm1,
            &self.modulation[block + 1],
            row_map,
            rows,
            hidden,
            BLOCK_SLOTS,
            0,
            1,
            EPSILON,
        ))?;
        *carried = true;
        Ok(())
    }
}

/// `h3_dit.c:800-881` `refine_text`: `condition_proj`, two refiner blocks, and
/// a final RMS norm, all over the prompt rows.
fn refine_text(
    gpu: &Gpu,
    weights: &Weights,
    config: &Config,
    text: &Embedding,
    text_rows: u32,
) -> Result<Tensor, DitError> {
    let hidden = config.hidden as u32;
    let inner = config.inner() as u32;
    let ffn = config.ffn as u32;
    let rows = text_rows;

    let source = gpu_alloc(gpu, Dtype::Bf16, text.tokens() * config.text_dim as usize)?;
    device(gpu.upload(&source, text.values()))?;

    let condition_weight = tensor_from_role(gpu, weights, Role::ConditionWeight, Dtype::Bf16)?;
    let condition_bias = tensor_from_role(gpu, weights, Role::ConditionBias, Dtype::Bf16)?;
    let block0 = load_refiner_block(gpu, weights, 0)?;
    let block1 = load_refiner_block(gpu, weights, 1)?;
    let final_norm = tensor_from_role(gpu, weights, Role::RefinerFinalNorm, Dtype::Bf16)?;

    let refined = gpu_alloc(gpu, Dtype::Bf16, rows as usize * hidden as usize)?;
    let norm = gpu_alloc(gpu, Dtype::Bf16, rows as usize * hidden as usize)?;
    let qkv = gpu_alloc(gpu, Dtype::Bf16, rows as usize * inner as usize * 3)?;
    let query = gpu_alloc(gpu, Dtype::Bf16, rows as usize * inner as usize)?;
    let key = gpu_alloc(gpu, Dtype::Bf16, rows as usize * inner as usize)?;
    let value = gpu_alloc(gpu, Dtype::Bf16, rows as usize * inner as usize)?;
    // `h3_dit.c:810`: the attention rows are `rows x INNER`, not `rows x HIDDEN`.
    let heads = gpu_alloc(gpu, Dtype::Bf16, rows as usize * inner as usize)?;
    let branch = gpu_alloc(gpu, Dtype::Bf16, rows as usize * hidden as usize)?;
    let fc1 = gpu_alloc(gpu, Dtype::Bf16, rows as usize * ffn as usize * 2)?;
    let activated = gpu_alloc(gpu, Dtype::Bf16, rows as usize * ffn as usize)?;

    // `h3_dit.c:855-860`.
    device(gpu.linear(
        &refined,
        &source,
        &condition_weight,
        Some(&condition_bias),
        rows,
        config.text_dim as u32,
        hidden,
        Dtype::Bf16,
    ))?;

    // `h3_dit.c:763-801` `run_refiner_block`: no AdaLN, no RoPE (the call
    // passes `rope_half = 0`), residuals in BF16.
    for block in [&block0, &block1] {
        device(gpu.rms_norm(&norm, &refined, &block.norm1, rows, hidden, EPSILON))?;
        device(gpu.linear(
            &qkv,
            &norm,
            &block.qkv,
            None,
            rows,
            hidden,
            inner * 3,
            Dtype::Bf16,
        ))?;
        device(gpu.qkv_rope(
            &query,
            &key,
            &value,
            &qkv,
            &block.q_norm,
            &block.k_norm,
            &block.q_norm,
            &block.k_norm,
            rows,
            config.heads as u32,
            config.head_dim as u32,
            0,
            EPSILON,
        ))?;
        device(gpu.sdpa(
            &heads,
            &query,
            &key,
            &value,
            rows,
            config.heads as u32,
            config.head_dim as u32,
            1.0 / ((config.head_dim as f32).sqrt()),
        ))?;
        device(gpu.linear(
            &branch,
            &heads,
            &block.out,
            None,
            rows,
            inner,
            hidden,
            Dtype::Bf16,
        ))?;
        device(gpu.add(&refined, &refined, &branch))?;

        device(gpu.rms_norm(&norm, &refined, &block.norm2, rows, hidden, EPSILON))?;
        device(gpu.linear(
            &fc1,
            &norm,
            &block.fc1,
            None,
            rows,
            hidden,
            ffn * 2,
            Dtype::Bf16,
        ))?;
        device(gpu.swiglu(&activated, &fc1, rows, ffn))?;
        device(gpu.linear(
            &branch,
            &activated,
            &block.fc2,
            None,
            rows,
            ffn,
            hidden,
            Dtype::Bf16,
        ))?;
        device(gpu.add(&refined, &refined, &branch))?;
    }

    // `h3_dit.c:866-869`: in place, as the reference writes it.
    device(gpu.rms_norm(&refined, &refined, &final_norm, rows, hidden, EPSILON))?;

    Ok(refined)
}

/// The AdaLN projections alone, hashed the way `dit_probe --adaln-only` does
/// it: block 0 and the final layer. A defect here moves every block, so this
/// is the cheapest gate that can separate modulation from the core.
pub fn adaln_hashes(dir: &Path, schedule: Schedule) -> Result<AdalnHashes, DitError> {
    let weights = Weights::open(dir).map_err(|error| DitError::Weights(error.to_string()))?;
    let config = weights.config().clone();
    let gpu = device(Gpu::create())?;

    // T2VA: no condition rows, exactly as the probe runs it.
    let tree = time_rows(&schedule, false, false);
    let (embedding, modulation, final_modulation) =
        adaln_projections(&gpu, &weights, &config, &tree)?;
    let block = modulation.first().ok_or_else(|| DitError::Geometry {
        detail: "the schedule produced no block modulation".into(),
    })?;

    let embed = hash_bf16(&gpu, &embedding)?;
    let block = hash_bf16(&gpu, block)?;
    let final_layer = hash_bf16(&gpu, &final_modulation)?;

    Ok(AdalnHashes {
        embed,
        block,
        final_layer,
    })
}

/// The three AdaLN stages the probe can hash separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdalnHashes {
    pub embed: u64,
    pub block: u64,
    pub final_layer: u64,
}

/// BF16 tensor to the reference's F32 readback, then FNV.
fn hash_bf16(gpu: &Gpu, tensor: &Tensor) -> Result<u64, DitError> {
    let mut bytes = vec![0u8; tensor.bytes()];
    device(gpu.read(tensor, &mut bytes))?;
    Ok(fnv1a(
        &bf16_to_f32(&bytes)
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
    ))
}

/// `h3_dit_schedule.c:164-232`: the F32 timestep embedder, then every
/// `adaln_proj`. The embeddings are BF16 and the projections are BF16; only
/// `time_embedder` is F32 in the checkpoint.
fn adaln_projections(
    gpu: &Gpu,
    weights: &Weights,
    config: &Config,
    tree: &TimeRows,
) -> Result<(Tensor, Vec<Tensor>, Tensor), DitError> {
    let rows = tree.times.len();
    let time_input = config.time_input as usize;
    let time_hidden = config.time_hidden as u32;
    let time_dim = config.time_dim as u32;

    let features = time_features(&tree.times);
    let input = gpu_alloc(gpu, Dtype::F32, rows * time_input)?;
    device(gpu.upload(&input, &pack_f32(&features)))?;

    let in_weight = tensor_from_role(gpu, weights, Role::TimeInWeight, Dtype::F32)?;
    let in_bias = tensor_from_role(gpu, weights, Role::TimeInBias, Dtype::F32)?;
    let out_weight = tensor_from_role(gpu, weights, Role::TimeOutWeight, Dtype::F32)?;
    let out_bias = tensor_from_role(gpu, weights, Role::TimeOutBias, Dtype::F32)?;

    let pre = gpu_alloc(gpu, Dtype::F32, rows * time_hidden as usize)?;
    let activated = gpu_alloc(gpu, Dtype::F32, rows * time_hidden as usize)?;
    let projected = gpu_alloc(gpu, Dtype::F32, rows * time_dim as usize)?;
    let rounded = gpu_alloc(gpu, Dtype::Bf16, rows * time_dim as usize)?;
    let embedding = gpu_alloc(gpu, Dtype::Bf16, rows * time_dim as usize)?;

    // `h3_dit_schedule.c:200-221`.
    device(gpu.linear(
        &pre,
        &input,
        &in_weight,
        Some(&in_bias),
        rows as u32,
        time_input as u32,
        time_hidden,
        Dtype::F32,
    ))?;
    device(gpu.silu_f32(&activated, &pre))?;
    device(gpu.linear(
        &projected,
        &activated,
        &out_weight,
        Some(&out_bias),
        rows as u32,
        time_hidden,
        time_dim,
        Dtype::F32,
    ))?;
    device(gpu.cast_f32_to_bf16(&rounded, &projected))?;
    device(gpu.silu_bf16(&embedding, &rounded))?;

    // `h3_dit_schedule.c:263-298`: one projection per block.
    let mut modulation = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        let weight = tensor_from_role(gpu, weights, Role::BlockAdalnWeight(layer), Dtype::Bf16)?;
        let bias = tensor_from_role(gpu, weights, Role::BlockAdalnBias(layer), Dtype::Bf16)?;
        let out = gpu_alloc(gpu, Dtype::Bf16, rows * config.adaln_out as usize)?;
        device(gpu.linear(
            &out,
            &embedding,
            &weight,
            Some(&bias),
            rows as u32,
            time_dim,
            config.adaln_out as u32,
            Dtype::Bf16,
        ))?;
        modulation.push(out);
    }

    // `h3_dit_schedule.c:303-330`: the final projection.
    let final_weight = tensor_from_role(gpu, weights, Role::FinalAdalnWeight, Dtype::Bf16)?;
    let final_bias = tensor_from_role(gpu, weights, Role::FinalAdalnBias, Dtype::Bf16)?;
    let final_modulation = gpu_alloc(gpu, Dtype::Bf16, rows * config.final_adaln_out as usize)?;
    device(gpu.linear(
        &final_modulation,
        &embedding,
        &final_weight,
        Some(&final_bias),
        rows as u32,
        time_dim,
        config.final_adaln_out as u32,
        Dtype::Bf16,
    ))?;

    Ok((embedding, modulation, final_modulation))
}

/// `h3_dit.c:883-1001` `prepare_rope`: `{t, h * scale, w * scale}` times
/// `rope.inv_freq`, three axes over 16 frequencies, then a BF16 cast.
fn rope_tables(
    gpu: &Gpu,
    weights: &Weights,
    config: &Config,
    sequence: &Sequence,
    spatial_scale: f32,
) -> Result<(Tensor, Tensor), DitError> {
    if config.rope_freqs as usize != ROPE_FREQS {
        return Err(DitError::Geometry {
            detail: format!("rope_freqs {} is not {ROPE_FREQS}", config.rope_freqs),
        });
    }
    let inverse = read_f32(weights, Role::RopeFreq)?;
    if inverse.len() != INV_FREQ_ELEMENTS {
        return Err(DitError::Geometry {
            detail: format!("rope.inv_freq holds {} values", inverse.len()),
        });
    }

    let rows = sequence.rows();
    let mut cosines = vec![0.0f32; rows * ROPE_HALF];
    let mut sines = vec![0.0f32; rows * ROPE_HALF];

    for (row, position) in sequence.positions.iter().enumerate() {
        let axes = [
            position.t as f32,
            (position.h * spatial_scale as f64) as f32,
            (position.w * spatial_scale as f64) as f32,
        ];
        for (axis, coordinate) in axes.iter().enumerate() {
            for (frequency, inverse_frequency) in inverse.iter().enumerate() {
                let index = row * ROPE_HALF + axis * ROPE_FREQS + frequency;
                let angle = *coordinate * inverse_frequency;
                cosines[index] = angle.cos();
                sines[index] = angle.sin();
            }
        }
    }

    let cos_f32 = gpu_alloc(gpu, Dtype::F32, cosines.len())?;
    let sin_f32 = gpu_alloc(gpu, Dtype::F32, sines.len())?;
    device(gpu.upload(&cos_f32, &pack_f32(&cosines)))?;
    device(gpu.upload(&sin_f32, &pack_f32(&sines)))?;
    let cos_bf16 = gpu_alloc(gpu, Dtype::Bf16, cosines.len())?;
    let sin_bf16 = gpu_alloc(gpu, Dtype::Bf16, sines.len())?;
    device(gpu.cast_f32_to_bf16(&cos_bf16, &cos_f32))?;
    device(gpu.cast_f32_to_bf16(&sin_bf16, &sin_f32))?;

    Ok((cos_bf16, sin_bf16))
}

/// Per step: the sequence row map, the audio head map and the video head map.
type ModulationMaps = (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>);

/// `h3_dit.c:1003-1068` `prepare_maps`: the per-step modulation row map, plus
/// the final-head maps, which are constant per step.
fn modulation_maps(
    gpu: &Gpu,
    tree: &TimeRows,
    sequence: &Sequence,
) -> Result<ModulationMaps, DitError> {
    let mut row_maps = Vec::with_capacity(tree.steps);
    let mut final_audio = Vec::with_capacity(tree.steps);
    let mut final_video = Vec::with_capacity(tree.steps);

    for step in 0..tree.steps {
        let rows = row_map(tree, step, sequence)?;
        row_maps.push(u32_tensor(gpu, &rows)?);

        let audio: Vec<u32> = vec![tree.audio[step]; sequence.audio_target_rows];
        let video: Vec<u32> = vec![tree.video[step]; sequence.img_target_rows];
        final_audio.push(u32_tensor(gpu, &audio)?);
        final_video.push(u32_tensor(gpu, &video)?);
    }

    Ok((row_maps, final_audio, final_video))
}

/// `h3_dit.c:1070-1126` `prepare_projection_maps`: only a layout with more than
/// one segment of a modality needs a scatter map; otherwise each segment is
/// projected in place at its own offset.
fn projection_maps(
    gpu: &Gpu,
    sequence: &Sequence,
) -> Result<(Option<Tensor>, Option<Tensor>), DitError> {
    let mut video_segments = 0;
    let mut audio_segments = 0;
    for segment in &sequence.segments {
        match segment.kind {
            SegmentKind::Cond | SegmentKind::RefImage | SegmentKind::Video => video_segments += 1,
            SegmentKind::Text => {}
            _ => audio_segments += 1,
        }
    }

    let mut video_map = None;
    let mut audio_map = None;
    if video_segments > 1 {
        let mut rows = Vec::with_capacity(sequence.video_total_rows());
        for segment in &sequence.segments {
            if matches!(
                segment.kind,
                SegmentKind::Cond | SegmentKind::RefImage | SegmentKind::Video
            ) {
                rows.extend((segment.start..segment.stop).map(|row| row as u32));
            }
        }
        video_map = Some(u32_tensor(gpu, &rows)?);
    }
    if audio_segments > 1 {
        let mut rows = Vec::with_capacity(sequence.audio_total_rows());
        for segment in &sequence.segments {
            if matches!(segment.kind, SegmentKind::RefAudio | SegmentKind::Audio) {
                rows.extend((segment.start..segment.stop).map(|row| row as u32));
            }
        }
        audio_map = Some(u32_tensor(gpu, &rows)?);
    }

    Ok((video_map, audio_map))
}

fn load_blocks(gpu: &Gpu, weights: &Weights, config: &Config) -> Result<Vec<Block>, DitError> {
    let mut blocks = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        blocks.push(load_block(gpu, weights, layer)?);
    }
    Ok(blocks)
}

/// `h3_dit.c:502-525` `load_block`.
fn load_block(gpu: &Gpu, weights: &Weights, layer: u32) -> Result<Block, DitError> {
    load_role_block(
        gpu,
        weights,
        [
            Role::BlockNorm1(layer),
            Role::BlockNorm2(layer),
            Role::BlockQkv(layer),
            Role::BlockQNorm(layer),
            Role::BlockKNorm(layer),
            Role::BlockOut(layer),
            Role::BlockFc1(layer),
            Role::BlockFc2(layer),
        ],
    )
}

fn load_refiner_block(gpu: &Gpu, weights: &Weights, layer: u32) -> Result<Block, DitError> {
    load_role_block(
        gpu,
        weights,
        [
            Role::RefinerNorm1(layer),
            Role::RefinerNorm2(layer),
            Role::RefinerQkv(layer),
            Role::RefinerQNorm(layer),
            Role::RefinerKNorm(layer),
            Role::RefinerOut(layer),
            Role::RefinerFc1(layer),
            Role::RefinerFc2(layer),
        ],
    )
}

/// One block's eight BF16 tensors, in the order `load_role_block` receives
/// them.
fn load_role_block(gpu: &Gpu, weights: &Weights, roles: [Role; 8]) -> Result<Block, DitError> {
    Ok(Block {
        norm1: tensor_from_role(gpu, weights, roles[0], Dtype::Bf16)?,
        norm2: tensor_from_role(gpu, weights, roles[1], Dtype::Bf16)?,
        qkv: tensor_from_role(gpu, weights, roles[2], Dtype::Bf16)?,
        q_norm: tensor_from_role(gpu, weights, roles[3], Dtype::Bf16)?,
        k_norm: tensor_from_role(gpu, weights, roles[4], Dtype::Bf16)?,
        out: tensor_from_role(gpu, weights, roles[5], Dtype::Bf16)?,
        fc1: tensor_from_role(gpu, weights, roles[6], Dtype::Bf16)?,
        fc2: tensor_from_role(gpu, weights, roles[7], Dtype::Bf16)?,
    })
}

/// `h3_dit.c:1290-1354`: the patch projections stay F32, the final heads are
/// converted to BF16 once.
fn load_final(gpu: &Gpu, weights: &Weights) -> Result<Final, DitError> {
    Ok(Final {
        norm: tensor_from_role(gpu, weights, Role::FinalNorm, Dtype::Bf16)?,
        video_weight: cast_role_to_bf16(gpu, weights, Role::FinalVideoWeight)?,
        video_bias: cast_role_to_bf16(gpu, weights, Role::FinalVideoBias)?,
        audio_weight: cast_role_to_bf16(gpu, weights, Role::FinalAudioWeight)?,
        audio_bias: cast_role_to_bf16(gpu, weights, Role::FinalAudioBias)?,
    })
}

fn load_patches(gpu: &Gpu, weights: &Weights) -> Result<Patches, DitError> {
    Ok(Patches {
        video_weight: tensor_from_role(gpu, weights, Role::VideoPatchWeight, Dtype::F32)?,
        video_bias: tensor_from_role(gpu, weights, Role::VideoPatchBias, Dtype::F32)?,
        audio_weight: tensor_from_role(gpu, weights, Role::AudioPatchWeight, Dtype::F32)?,
        audio_bias: tensor_from_role(gpu, weights, Role::AudioPatchBias, Dtype::F32)?,
    })
}

/// Upload one role as `dtype`, refusing a checkpoint whose stored dtype is not
/// the one the reference loads it as.
fn tensor_from_role(
    gpu: &Gpu,
    weights: &Weights,
    role: Role,
    dtype: Dtype,
) -> Result<Tensor, DitError> {
    let span = weights
        .span(role)
        .map_err(|error| DitError::Weights(format!("{:?}: {error}", role)))?;
    let bytes = weights
        .bytes(role)
        .map_err(|error| DitError::Weights(format!("{:?}: {error}", role)))?;

    let stored = span.dtype.size() as usize;
    if stored != dtype.size() {
        return Err(DitError::Weights(format!(
            "{:?} is stored in {stored}-byte elements, the reference loads it as {}",
            role,
            dtype.size()
        )));
    }

    let tensor = gpu_alloc(gpu, dtype, span.elements() as usize)?;
    device(gpu.upload(&tensor, bytes))?;
    Ok(tensor)
}

/// `h3_dit.c:1312-1342`: an F32 final-head tensor converted to BF16 on the
/// device, one rounding, exactly as the reference does it.
fn cast_role_to_bf16(gpu: &Gpu, weights: &Weights, role: Role) -> Result<Tensor, DitError> {
    let source = tensor_from_role(gpu, weights, role, Dtype::F32)?;
    let target = gpu_alloc(gpu, Dtype::Bf16, source.elements())?;
    device(gpu.cast_f32_to_bf16(&target, &source))?;
    Ok(target)
}

/// Read an F32 role straight from the mapped shard: `rope.inv_freq` is 16
/// values the reference uploads only to read back on the host
/// (`h3_dit.c:884-893`).
fn read_f32(weights: &Weights, role: Role) -> Result<Vec<f32>, DitError> {
    let span = weights
        .span(role)
        .map_err(|error| DitError::Weights(format!("{:?}: {error}", role)))?;
    if span.dtype.size() != 4 {
        return Err(DitError::Weights(format!("{:?} is not F32", role)));
    }
    let bytes = weights
        .bytes(role)
        .map_err(|error| DitError::Weights(format!("{:?}: {error}", role)))?;

    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn gpu_alloc(gpu: &Gpu, dtype: Dtype, elements: usize) -> Result<Tensor, DitError> {
    if elements == 0 {
        return Err(DitError::Geometry {
            detail: "an empty tensor was requested".into(),
        });
    }
    device(gpu.alloc(dtype, elements))
}

fn u32_tensor(gpu: &Gpu, values: &[u32]) -> Result<Tensor, DitError> {
    let tensor = gpu_alloc(gpu, Dtype::U32, values.len().max(1))?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if !values.is_empty() {
        device(gpu.upload_at(&tensor, 0, &bytes))?;
    }
    Ok(tensor)
}

/// Little-endian F32 bytes for an upload.
fn pack_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// BF16 words to F32 the way `h3_dit.c:2474-2483` does it: low half zero.
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::host::Schedule;
    use crate::minimax_h3::sequence::{Position, Segment};

    /// `h3_dit_schedule.c:79-128` at the E5 clip's two evaluations: step 0
    /// shares one row, step 1 needs two, so `time_rows = 3` (the probe's
    /// `ADALN rows=3`).
    #[test]
    fn timestep_rows_follow_the_sigma_order() {
        let schedule = Schedule::serving(2).expect("two steps");
        let rows = time_rows(&schedule, false, false);

        assert_eq!(rows.steps, 2);
        assert_eq!(rows.video, vec![0, 1]);
        assert_eq!(rows.audio, vec![0, 2]);
        assert_eq!(rows.times.len(), 3);
        assert_eq!(rows.times[0], 0.0);
        assert_eq!(rows.times[1], 1.0 - schedule.video[1]);
        assert_eq!(rows.times[2], 1.0 - schedule.audio[1]);
    }

    /// `h3_dit_schedule.c:100-118`: a condition row is appended, and the row
    /// selection falls back to it until the modality time saturates.
    #[test]
    fn condition_rows_are_appended_and_selected() {
        let schedule = Schedule::serving(2).expect("two steps");
        let rows = time_rows(&schedule, true, false);

        assert_eq!(rows.times.len(), 4);
        let visual = rows.visual_condition.expect("visual condition rows");
        // `h3_dit_schedule.c:120-124`: the condition row is used whenever the
        // video time is below 0.999, which at sigma 1 and 0.923 is both steps.
        assert_eq!(visual, vec![3, 3]);
        assert_eq!(rows.times[3], 0.999);
    }

    /// `h3_dit_schedule_row_map` (`:380-445`): text rows carry tag 1, video and
    /// condition rows tag 0, audio tag 2.
    #[test]
    fn row_map_tags_by_segment_kind() {
        let schedule = Schedule::serving(2).expect("two steps");
        let sequence = Sequence {
            positions: vec![
                Position {
                    t: 0.0,
                    h: 0.0,
                    w: 0.0
                };
                4
            ],
            segments: vec![
                Segment {
                    start: 0,
                    stop: 2,
                    kind: SegmentKind::Text,
                },
                Segment {
                    start: 2,
                    stop: 3,
                    kind: SegmentKind::Audio,
                },
                Segment {
                    start: 3,
                    stop: 4,
                    kind: SegmentKind::Video,
                },
            ],
            img_cond_rows: 0,
            img_target_rows: 1,
            audio_cond_rows: 0,
            audio_target_rows: 1,
            signature: [2, 2, 16, 16, 8],
        };
        let tree = time_rows(&schedule, false, false);
        let rows = row_map(&tree, 1, &sequence).expect("row map");

        assert_eq!(
            rows,
            vec![
                tree.video[1] * MODALITIES + 1,
                tree.video[1] * MODALITIES + 1,
                tree.audio[1] * MODALITIES + 2,
                tree.video[1] * MODALITIES
            ]
        );
    }

    /// `h3_dit_schedule.c:147-157` against the reference's own table:
    /// `dit_probe --features --steps 2` prints
    /// `FEATURES rows=3 fnv=419a597fdbeceadd times=00000000.3d9d89d8.3e800000`.
    #[test]
    fn time_features_match_the_reference_table() {
        let schedule = Schedule::serving(2).expect("two steps");
        let rows = time_rows(&schedule, false, false);

        let bits: Vec<u32> = rows.times.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, vec![0x0000_0000, 0x3d9d_89d8, 0x3e80_0000]);

        let features = time_features(&rows.times);
        let bytes: Vec<u8> = features
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(fnv1a(&bytes), 0x419a_597f_dbec_eadd);
        assert_eq!(features[0], 1.0f32);
    }

    /// `h3_dit.c:3064-3119`: patchify and unpatchify round trip exactly, and
    /// the row order is `[t, h/2, w/2, c, dh, dw]`.
    #[test]
    fn video_patchify_round_trips_in_row_order() {
        let (channels, time, height, width) = (3usize, 2usize, 4usize, 4usize);
        let count = channels * time * height * width;
        let latent: Vec<f32> = (0..count).map(|value| value as f32).collect();
        let mut rows = vec![0.0f32; count];

        patchify_video(&latent, channels, time, height, width, &mut rows).expect("patchify");
        // Row 0 is t=0, h=0, w=0, c=0, dh=0, dw=0..1 -> latent 0 and 1.
        assert_eq!(&rows[0..2], &[0.0, 1.0]);
        // Row 1 is dh=1 -> latent 4 and 5.
        assert_eq!(&rows[2..4], &[4.0, 5.0]);

        let mut restored = vec![0.0f32; count];
        unpatchify_video(&rows, channels, time, height, width, &mut restored).expect("unpatchify");
        assert_eq!(restored, latent);
    }

    /// `h3_dit.c:3121-3151`: audio rows are `[stream, t, channel]`.
    #[test]
    fn audio_pack_round_trips_in_stream_order() {
        let (channels, time) = (2usize, 3usize);
        let latent: Vec<f32> = (0..channels * 2 * time).map(|value| value as f32).collect();
        let mut rows = vec![0.0f32; latent.len()];

        pack_audio(&latent, channels, time, &mut rows).expect("pack");
        // stream 0, t 0, channels 0..1 -> latent[0] and latent[2 * time].
        assert_eq!(&rows[0..2], &[0.0, 6.0]);

        let mut restored = vec![0.0f32; latent.len()];
        unpack_audio(&rows, channels, time, &mut restored).expect("unpack");
        assert_eq!(restored, latent);
    }

    /// `h3_dit.c:3071-3072`: odd spatial extents are refused.
    #[test]
    fn odd_latent_shapes_are_refused() {
        let mut rows = vec![0.0f32; 4];
        assert!(patchify_video(&[0.0; 4], 1, 1, 3, 2, &mut rows).is_err());
        assert!(pack_audio(&[0.0; 4], 0, 2, &mut rows).is_err());
    }

    /// `h3_dit.c:20-29`: the time features are cos then sin at
    /// `exp(-ln(10000) * i / 128)`.
    #[test]
    fn time_features_are_cos_then_sin() {
        let features = time_features(&[0.0f32]);
        assert_eq!(features.len(), TIME_INPUT);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[TIME_INPUT / 2], 0.0);
    }

    /// The seed and prime are the probe family's, so a golden taken from the
    /// reference can be compared with `fnv1a` directly.
    #[test]
    fn fnv_constants_match_the_probe_family() {
        use crate::minimax_h3::text_forward::{FNV_OFFSET, FNV_PRIME};

        assert_eq!(FNV_OFFSET, 1_469_598_103_934_665_603);
        assert_eq!(FNV_PRIME, 1_099_511_628_211);
        assert_eq!(fnv1a(b""), 0x1465_0fb0_739d_0383);
    }
}
