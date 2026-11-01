//! The profile's stage-2 latent path: the x2 latent upscaler, the frozen
//! alignment, and the H3-to-LTX Conv3D adapter. Both models run natively here;
//! there is no intervening layer and nothing of this path is Python at run
//! time. The two ops are separate types, [`Upscaler`] and [`Adapter`], called
//! in the order `runtime/stage2.py:58-76` calls them.
//!
//! This path has no C reference. Its authority is the Sol-H3-Spark profile's
//! own Python plus the pinned ComfyUI node:
//!
//! - `runtime/stage2.py:58-76` `video_encode`: the draft latent is upscaled,
//!   the adapter is then told the *doubled* canvas, and the refiner input is
//!   the adapter's output cropped to 16 latent frames.
//! - `runtime/stage2_ops/h3_upscale.py:112-126` `upscale_comfy_normalized_h3`:
//!   the author node's own `N(model(D(x)))` around an already-normalized H3
//!   latent, in bf16, with `scale=2.0` and `enable_chunking=True`.
//! - `runtime/stage2_ops/h3_ltx_adapter/{constants,geometry,model,adapter}.py`:
//!   the alignment (temporal resample plus nearest pack, then a 2x pixel
//!   unshuffle) and the 22-block factorized Conv3D adapter.
//! - `nodes/minimax_h3_latent_upscaler_3d.py` at `d7c01b90`
//!   (sha256 `744063b4…5cb89e2`): `LatentResizer3D`, the temporal chunking
//!   blend, and the 24-channel `LATENTS_MEAN`/`LATENTS_STD` literals.
//!
//! The record is `docs/minimax-h3/ltx-reference-map-2026-10-28.md`; the frozen
//! gate vector is produced offline by `tools/minimax_h3/ltx/latent_oracle.py`
//! from those same references — it is a test input, not a runtime component —
//! and is compared against by `examples/h3_ltx_adapter_gate.rs`.
//!
//! Layouts. Every activation here is channel-major `[channel][time][height]
//! [width]`, the H3 family's own convention, and every convolution weight is
//! PyTorch's `[out][in][kd][kh][kw]` as it sits in the checkpoint. A 3x3x3
//! convolution runs as im2col plus `cublasLt`; a 1x1 convolution is a matmul
//! over the channel-major operands with no copy at all.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use crate::mapped::MappedFile;
use crate::minimax_h3::safetensors::{self, Dtype as StDtype};
use crate::minimax_h3::vision::{bf16_to_f32, f32_to_bf16};

/// `constants.py:3-7`: the H3 latent's own geometry.
pub const H3_LATENT_CHANNELS: usize = 24;
pub const H3_SPATIAL_COMPRESSION: usize = 16;
pub const H3_TEMPORAL_COMPRESSION: usize = 4;
pub const H3_CLIP_LENGTH: usize = 17;
pub const H3_TOKEN_DROP: usize = 3;

/// `constants.py:9-11`: the LTX latent the adapter produces.
pub const LTX_LATENT_CHANNELS: usize = 128;
pub const LTX_SPATIAL_COMPRESSION: usize = 32;
pub const LTX_TEMPORAL_COMPRESSION: usize = 8;

/// `adapter.py:50-54`: `geometry.temporal_pack_slots`, and the frozen
/// `spatial_mode: pixel_unshuffle` / `temporal_mode: linear_plus_nearest_pack`.
pub const TEMPORAL_PACK_SLOTS: usize = 3;
pub const PIXEL_UNSHUFFLE: usize = 2;

/// `constants.py:13-66` `H3_LATENTS_MEAN` / `H3_LATENTS_STD`, identical to the
/// node's `LATENTS_MEAN` / `LATENTS_STD` (`minimax_h3_latent_upscaler_3d.py:
/// 83-99`) to within 5.6e-17, which is the bf16 constant they were fitted
/// against. Checked by `the_node_and_adapter_statistics_agree`.
pub const H3_LATENTS_MEAN: [f32; H3_LATENT_CHANNELS] = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
];

pub const H3_LATENTS_STD: [f32; H3_LATENT_CHANNELS] = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
];

/// `LatentResizer3D.__init__` defaults (`minimax_h3_latent_upscaler_3d.py:
/// 215-243`), confirmed against the checkpoint by `_detect_arch` (`:379-411`).
pub const UPSCALER_IN_CHANNELS: usize = H3_LATENT_CHANNELS;
pub const UPSCALER_CHANNELS: usize = 512;
pub const UPSCALER_BLOCKS: usize = 12;
pub const UPSCALER_TEMPORAL_EVERY: usize = 2;
pub const UPSCALER_TEMPORAL_KERNEL: usize = 5;
pub const UPSCALER_EMBED_DIM: usize = 64;
/// `normalization(channels)` (`:135-136`): `nn.GroupNorm(32, channels)` at
/// PyTorch's default epsilon 1e-5. The adapter's is 1e-6 — see below.
pub const UPSCALER_GROUPS: usize = 32;
pub const UPSCALER_EPSILON: f32 = 1e-5;

/// `LatentResizer3D.forward` (`:264-268`): `chunk = 32`, `overlap = tk`, the
/// depthwise temporal kernel width.
///
/// The chunk is *not* a VRAM-only knob, whatever its tooltip says: the model's
/// temporal blocks downsample and upsample in time, so the segment's frame
/// count decides the intermediate grid, and a single 37-frame segment lands a
/// different latent (`docs/minimax-h3/planar-conv-2026-10-29.md`: 3.8e-1
/// relative against the reference, both arms, both convolution paths). It is
/// part of the arithmetic and the port keeps the node's value.
pub const UPSCALER_CHUNK: usize = 32;

/// `model.py:28-42` and `config.json`: in 384 (24 linear + 24x3 packed, x4
/// from the unshuffle), out 128, width 752, 22 blocks, expansion 2, 16 groups.
pub const ADAPTER_IN_CHANNELS: usize = 384;
pub const ADAPTER_OUT_CHANNELS: usize = LTX_LATENT_CHANNELS;
pub const ADAPTER_WIDTH: usize = 752;
pub const ADAPTER_BLOCKS: usize = 22;
pub const ADAPTER_EXPANSION: usize = 2;
pub const ADAPTER_GROUPS: usize = 16;
/// `model.py:13` `nn.GroupNorm(groups, channels, eps=1e-6)`.
pub const ADAPTER_EPSILON: f32 = 1e-6;

/// The refiner's latent frames after `stage2.py:76`'s `latent[:, :, :16]`.
pub const REFINER_FRAMES: usize = 16;

/// A `[(slot, target)] -> source frame` entry that writes zero, matching
/// `geometry.py:72-78`'s `occupied` mask.
pub const PACK_UNOCCUPIED: u32 = u32::MAX;

#[derive(Debug)]
pub enum LatentError {
    Io(String),
    Shape(String),
    Device(String),
    Plan(String),
}

impl std::fmt::Display for LatentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatentError::Io(detail) => write!(formatter, "ltx latent io: {detail}"),
            LatentError::Shape(detail) => write!(formatter, "ltx latent shape: {detail}"),
            LatentError::Device(detail) => write!(formatter, "ltx latent device: {detail}"),
            LatentError::Plan(detail) => write!(formatter, "ltx latent plan: {detail}"),
        }
    }
}

impl std::error::Error for LatentError {}

fn device<T>(result: Result<T, String>) -> Result<T, LatentError> {
    result.map_err(LatentError::Device)
}

// ---------------------------------------------------------------- geometry --

/// `geometry.py:13-24` `h3_temporal_positions`: the pixel frame each H3 latent
/// frame stands for, `ceil(17/4)` tokens per 17-frame clip, clamped to the
/// last pixel frame, then the last `H3_TOKEN_DROP` entries dropped.
pub fn h3_temporal_positions(pixel_frames: usize) -> Vec<f32> {
    assert!(pixel_frames > 0, "num_pixel_frames must be positive");
    let tokens_per_chunk = H3_CLIP_LENGTH.div_ceil(H3_TEMPORAL_COMPRESSION);
    let chunks = pixel_frames.div_ceil(H3_CLIP_LENGTH);
    let mut positions: Vec<f32> = (0..chunks)
        .flat_map(|chunk| {
            (0..tokens_per_chunk).map(move |token| {
                (chunk * H3_CLIP_LENGTH + token * H3_TEMPORAL_COMPRESSION).min(pixel_frames - 1)
                    as f32
            })
        })
        .collect();
    positions.truncate(positions.len() - H3_TOKEN_DROP);
    positions
}

/// `geometry.py:27-31` `padded_ltx_pixel_frames`.
pub fn padded_ltx_pixel_frames(pixel_frames: usize) -> usize {
    assert!(pixel_frames > 0, "num_pixel_frames must be positive");
    (pixel_frames - 1).div_ceil(LTX_TEMPORAL_COMPRESSION) * LTX_TEMPORAL_COMPRESSION + 1
}

/// `geometry.py:34-37` `ltx_temporal_positions`.
pub fn ltx_temporal_positions(pixel_frames: usize) -> Vec<f32> {
    let latent_frames = (pixel_frames - 1) / LTX_TEMPORAL_COMPRESSION + 1;
    (0..latent_frames)
        .map(|index| (index * LTX_TEMPORAL_COMPRESSION).min(pixel_frames - 1) as f32)
        .collect()
}

/// The three index/weight vectors the alignment needs, resolved once on the
/// host: `geometry.py:39-79`. The device kernels take these as they are, so
/// the `searchsorted`/`argmin` semantics are torch's own and are unit-tested
/// against hand-computed cases rather than re-derived on the device.
pub struct TemporalPlan {
    /// Number of latent frames the source carries.
    pub source_frames: usize,
    /// Target latent frames: `ltx_temporal_positions(padded)`.
    pub target_frames: usize,
    /// `temporal_resample`'s left/right source frames and interpolation
    /// weight, one entry per target frame (`geometry.py:44-51`).
    pub linear_left: Vec<u32>,
    pub linear_right: Vec<u32>,
    pub linear_weight: Vec<f32>,
    /// `temporal_nearest_pack`'s source frame per `(target, slot)`, or
    /// `PACK_UNOCCUPIED` (`geometry.py:64-78`).
    pub pack: Vec<u32>,
    pub slots: usize,
}

/// `geometry.py:39-79`: the resample and pack plans for one `pixel_frames`.
pub fn temporal_plan(pixel_frames: usize) -> Result<TemporalPlan, LatentError> {
    let source_positions = h3_temporal_positions(pixel_frames);
    let padded = padded_ltx_pixel_frames(pixel_frames);
    let target_positions = ltx_temporal_positions(padded);
    let source_frames = source_positions.len();
    let target_frames = target_positions.len();

    // temporal_resample: right is searchsorted(source, target, right=False),
    // clamped to the last source index; left is right-1 clamped at zero; the
    // weight is the position ratio, and zero where the two coincide.
    let mut linear_left = Vec::with_capacity(target_frames);
    let mut linear_right = Vec::with_capacity(target_frames);
    let mut linear_weight = Vec::with_capacity(target_frames);
    for target in &target_positions {
        let right = source_positions
            .partition_point(|source| source < target)
            .min(source_frames - 1);
        let left = right.saturating_sub(1);
        let span = source_positions[right] - source_positions[left];
        let weight = if span > 0.0 {
            ((target - source_positions[left]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        linear_left.push(left as u32);
        linear_right.push(right as u32);
        linear_weight.push(weight);
    }

    // temporal_nearest_pack: every source frame belongs to its nearest target
    // (torch.argmin takes the first minimum), and each target's slots hold its
    // assigned source frames in ascending order.
    let mut assigned: Vec<Vec<u32>> = vec![Vec::new(); target_frames];
    for (index, source) in source_positions.iter().enumerate() {
        let mut best = 0usize;
        let mut best_distance = f32::INFINITY;
        for (target_index, target) in target_positions.iter().enumerate() {
            let distance = (source - target).abs();
            if distance < best_distance {
                best_distance = distance;
                best = target_index;
            }
        }
        assigned[best].push(index as u32);
    }

    let required = assigned.iter().map(Vec::len).max().unwrap_or(0);
    if required > TEMPORAL_PACK_SLOTS {
        return Err(LatentError::Plan(format!(
            "temporal packing needs {required} slots, the checkpoint provides {TEMPORAL_PACK_SLOTS}"
        )));
    }

    let mut pack = vec![PACK_UNOCCUPIED; target_frames * TEMPORAL_PACK_SLOTS];
    for (target_index, sources) in assigned.iter().enumerate() {
        for (slot, source) in sources.iter().enumerate() {
            pack[target_index * TEMPORAL_PACK_SLOTS + slot] = *source;
        }
    }

    Ok(TemporalPlan {
        source_frames,
        target_frames,
        linear_left,
        linear_right,
        linear_weight,
        pack,
        slots: TEMPORAL_PACK_SLOTS,
    })
}

/// `adapter.py:16-24` `expected_shapes`, for the doubled canvas: the upscaled
/// H3 grid the adapter is handed, and the LTX latent it returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub pixel_frames: usize,
    /// The draft's own canvas, which the upscaler doubles.
    pub draft_height: usize,
    pub draft_width: usize,
    /// The adapter's canvas: `pixel_height // 32` is the LTX latent height.
    pub canvas_height: usize,
    pub canvas_width: usize,
    /// `[channels, time, height, width]` of the H3 latent the upscaler reads.
    pub upscaler_input: [usize; 4],
    /// …of the upscaled H3 latent, which is the adapter's input.
    pub h3: [usize; 4],
    /// `[384, t, h, w]`: the aligned, pixel-unshuffled H3 grid.
    pub aligned: [usize; 4],
    /// `[128, t, h, w]`: the adapter's normalized LTX latent.
    pub ltx: [usize; 4],
}

impl Geometry {
    /// The latent's element count and the plane count a channels-last matrix
    /// over the interior grid has, both used to size scratch.
    pub fn upscaler_input_elements(&self) -> usize {
        self.upscaler_input.iter().product()
    }

    pub fn upscaled_elements(&self) -> usize {
        self.h3.iter().product()
    }

    pub fn refiner_elements(&self) -> usize {
        self.ltx[0] * REFINER_FRAMES * self.ltx[2] * self.ltx[3]
    }
}

/// `expected_shapes` plus the upscaler's own input: the draft canvas at 16x
/// with the H3 latent's frame count.
pub fn geometry(
    pixel_frames: usize,
    draft_height: usize,
    draft_width: usize,
) -> Result<Geometry, LatentError> {
    let canvas_height = draft_height * 2;
    let canvas_width = draft_width * 2;
    if canvas_height % 32 != 0 || canvas_width % 32 != 0 {
        return Err(LatentError::Shape(format!(
            "adapter canvas {canvas_height}x{canvas_width} must be divisible by 32"
        )));
    }
    let time = h3_temporal_positions(pixel_frames).len();
    let ltx_time = ltx_temporal_positions(padded_ltx_pixel_frames(pixel_frames)).len();

    Ok(Geometry {
        pixel_frames,
        draft_height,
        draft_width,
        canvas_height,
        canvas_width,
        upscaler_input: [
            H3_LATENT_CHANNELS,
            time,
            draft_height / H3_SPATIAL_COMPRESSION,
            draft_width / H3_SPATIAL_COMPRESSION,
        ],
        h3: [
            H3_LATENT_CHANNELS,
            time,
            canvas_height / H3_SPATIAL_COMPRESSION,
            canvas_width / H3_SPATIAL_COMPRESSION,
        ],
        aligned: [
            ADAPTER_IN_CHANNELS,
            ltx_time,
            canvas_height / LTX_SPATIAL_COMPRESSION,
            canvas_width / LTX_SPATIAL_COMPRESSION,
        ],
        ltx: [
            LTX_LATENT_CHANNELS,
            ltx_time,
            canvas_height / LTX_SPATIAL_COMPRESSION,
            canvas_width / LTX_SPATIAL_COMPRESSION,
        ],
    })
}

/// `LatentResizer3D.forward` (`:268-219`, the chunking branch): the segments
/// the model is run over, each with the output range it covers and the ramp
/// weight applied to that range before the weighted average.
///
/// Frame weights come in pairs — one ramp up at the segment's leading overlap,
/// one down at its trailing one — and `chunk` frames of stride. With the
/// release's `T = 37`, `chunk = 32` and `overlap = 5` this is two segments,
/// the second covering frames 27..37 and the first 0..37.
pub struct ChunkPlan {
    /// `(segment input range, segment output range, output weight)`.
    pub segments: Vec<Segment>,
    /// Total frames: `T`.
    pub frames: usize,
}

pub struct Segment {
    /// The frames this segment reads, as `[begin, end)` **of the
    /// replicate-padded temporal axis** — the single-segment case included,
    /// which is why the unchunked branch still reads at the overlap offset.
    pub input: (usize, usize),
    /// The output frames it writes, as `[begin, end)`.
    pub output: (usize, usize),
    /// Where this segment's own output tensor holds the frames it contributes:
    /// `s0 = (out_start + overlap) - lo` (`:290`), which is not zero because
    /// the segment is run over the padded window, not over its output range.
    pub source_begin: usize,
    /// The ramp weight per output frame it writes.
    pub weights: Vec<f32>,
}

/// `forward` (`:273-311`): `overlap` frames of replicate padding, a `chunk`
/// stride, and the two ramps `1/(n+1) .. n/(n+1)` and `n/(n+1) .. 1/(n+1)`.
pub fn chunk_plan(frames: usize, overlap: usize, chunk: usize) -> ChunkPlan {
    if frames <= chunk {
        // `:266-267` returns the segment unreached by the chunk machinery, so
        // the padding it reads is the identity offset into the padded axis.
        return ChunkPlan {
            segments: vec![Segment {
                input: (overlap, overlap + frames),
                output: (0, frames),
                source_begin: 0,
                weights: vec![1.0; frames],
            }],
            frames,
        };
    }

    let mut segments = Vec::new();
    let mut start = 0;
    while start < frames {
        let segment_start = start;
        let segment_end = (start + chunk).min(frames);
        let output_start = segment_start.saturating_sub(overlap);
        let output_end = (segment_end + overlap).min(frames);
        let low = output_start.saturating_sub(overlap);
        let high = (output_end + overlap).min(frames + 2 * overlap);

        let count = output_end - output_start;
        let mut weights = vec![1.0f32; count];
        if segment_start > output_start {
            let blend = segment_start - output_start;
            for index in 0..blend {
                weights[index] = (index + 1) as f32 / (blend + 1) as f32;
            }
        }
        if output_end > segment_end {
            let blend = output_end - segment_end;
            for index in 0..blend {
                weights[count - blend + index] = (blend - index) as f32 / (blend + 1) as f32;
            }
        }

        segments.push(Segment {
            input: (low, high),
            output: (output_start, output_end),
            source_begin: (output_start + overlap) - low,
            weights,
        });
        start += chunk;
    }

    ChunkPlan { segments, frames }
}

/// The replicate padding the chunk plan needs on the temporal axis, and the
/// index of each padded frame in the unpadded tensor (`:271`,
/// `F.pad(x, (0,0,0,0,overlap,overlap), mode='replicate')`).
pub fn replicate_frames(frames: usize, overlap: usize) -> Vec<u32> {
    let last = frames - 1;
    (0..frames + 2 * overlap)
        .map(|index| index.saturating_sub(overlap).min(last) as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_geometry_is_the_profiles_own() {
        // H3_INPUT/H3_UPSCALED/ADAPTER_OUTPUT/REFINER_INPUT,
        // `runtime/stage2_ops/h3_upscale.py:24-27`, and the golden manifest's
        // own `expected_shapes`.
        let plan = geometry(124, 384, 672).unwrap();
        assert_eq!(plan.upscaler_input, [24, 37, 24, 42]);
        assert_eq!(plan.h3, [24, 37, 48, 84]);
        assert_eq!(plan.aligned, [384, 17, 24, 42]);
        assert_eq!(plan.ltx, [128, 17, 24, 42]);
    }

    #[test]
    fn the_h3_frame_count_is_the_reference_sequence() {
        let positions = h3_temporal_positions(124);
        assert_eq!(positions.len(), 37);
        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[4], 16.0);
        assert_eq!(positions[5], 17.0);
        // The last clip clamps every token past frame 123 onto it — 119, 123,
        // 123, 123, 123 — and the three-frame drop then removes exactly those
        // three clamped repeats, so the tail is 118, 119, 123.
        assert_eq!(&positions[32..], &[110.0, 114.0, 118.0, 119.0, 123.0]);
    }

    #[test]
    fn the_padded_ltx_length_and_positions_are_the_references() {
        assert_eq!(padded_ltx_pixel_frames(124), 129);
        assert_eq!(
            ltx_temporal_positions(129),
            (0..17)
                .map(|index| (index * 8) as f32)
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn a_short_clip_drops_to_two_frames() {
        // 17 pixel frames: one 17-frame clip, five positions, three dropped.
        let positions = h3_temporal_positions(17);
        assert_eq!(positions, vec![0.0, 4.0]);
        assert_eq!(padded_ltx_pixel_frames(17), 17);
        assert_eq!(ltx_temporal_positions(17), vec![0.0, 8.0, 16.0]);
        // The upscaler does not chunk at or below 32 latent frames; the
        // segment still reads through the padded axis' overlap offset.
        let single = chunk_plan(2, 5, 32);
        assert_eq!(single.segments.len(), 1);
        assert_eq!(single.segments[0].input, (5, 7));
        assert_eq!(single.segments[0].output, (0, 2));
    }

    #[test]
    fn the_resample_plan_matches_hand_computed_indices() {
        let plan = temporal_plan(124).unwrap();
        assert_eq!(plan.source_frames, 37);
        assert_eq!(plan.target_frames, 17);
        // Target 8 lies exactly on source frame 8 (positions 0,4,8,12,16,...):
        // searchsorted finds index 2, left is 1 at 4.0, and the weight is 1.
        assert_eq!(plan.linear_left[1], 1);
        assert_eq!(plan.linear_right[1], 2);
        assert_eq!(plan.linear_weight[1], 1.0);
        // Target 128 is past every source position, so searchsorted clamps to
        // the last index; the span is 123-119 = 4 and the ratio is 2.25, which
        // `weight.clamp(0, 1)` pins at 1 — the last target frame reads the
        // last source frame exactly.
        assert_eq!(plan.linear_left[16], 35);
        assert_eq!(plan.linear_right[16], 36);
        assert_eq!(plan.linear_weight[16], 1.0);
    }

    #[test]
    fn the_pack_assigns_every_source_frame_and_fits_three_slots() {
        let plan = temporal_plan(124).unwrap();
        assert_eq!(plan.pack.len(), 17 * 3);
        let mut seen: Vec<u32> = plan
            .pack
            .iter()
            .copied()
            .filter(|slot| *slot != PACK_UNOCCUPIED)
            .collect();
        seen.sort_unstable();
        assert_eq!(seen.len(), 37, "every source frame lands once");
        assert_eq!(seen, (0..37).collect::<Vec<u32>>());
        // Source frame 1 sits at position 4, exactly between targets 0 and 8;
        // torch's argmin takes the first minimum, so it lands on target 0 and
        // becomes that target's second slot.
        assert_eq!(plan.pack[0], 0);
        assert_eq!(plan.pack[1], 1);
    }

    #[test]
    fn the_chunk_plan_is_the_nodes_own_segments() {
        // T = 37 > chunk = 32 with overlap 5: two segments, overlapping on
        // frames 27..32, each with a five-frame ramp.
        let plan = chunk_plan(37, 5, 32);
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].input, (0, 42));
        assert_eq!(plan.segments[0].output, (0, 37));
        assert_eq!(plan.segments[0].weights[0], 1.0);
        assert_eq!(
            &plan.segments[0].weights[32..],
            &[5.0, 4.0, 3.0, 2.0, 1.0].map(|w| w / 6.0)
        );
        assert_eq!(plan.segments[1].input, (22, 42));
        assert_eq!(plan.segments[1].output, (27, 37));
        // s0 = (27 + 5) - 22: the frames it contributes sit ten frames into its
        // own output, which is why the accumulator needs the offset.
        assert_eq!(plan.segments[1].source_begin, 10);
        assert_eq!(plan.segments[0].source_begin, 5);
        assert_eq!(
            &plan.segments[1].weights[..5],
            &[1.0, 2.0, 3.0, 4.0, 5.0].map(|w| w / 6.0)
        );
        assert_eq!(plan.segments[1].weights[5], 1.0);
    }

    #[test]
    fn replicate_padding_holds_the_first_and_last_frame() {
        // T = 4, overlap 2: the padded axis is -2,-1,0,1,2,3,4,5 -> 0,0,0,1,2,3,3,3.
        assert_eq!(replicate_frames(4, 2), vec![0, 0, 0, 1, 2, 3, 3, 3]);
    }

    #[test]
    fn the_node_and_adapter_statistics_agree() {
        // The two literals the profile keeps: the node's `LATENTS_MEAN` at
        // `minimax_h3_latent_upscaler_3d.py:83-99` and the adapter's
        // `H3_LATENTS_MEAN` at `constants.py:13-66`. Both files' sums are
        // -2.05874753370881 and 41.0599799752235, so the port carries one copy.
        assert_eq!(H3_LATENTS_MEAN.len(), H3_LATENT_CHANNELS);
        assert_eq!(H3_LATENTS_STD.len(), H3_LATENT_CHANNELS);
        let mean_sum: f32 = H3_LATENTS_MEAN.iter().sum();
        assert!((mean_sum - -2.0587475).abs() < 1e-4, "mean sum {mean_sum}");
        let std_sum: f32 = H3_LATENTS_STD.iter().sum();
        assert!((std_sum - 41.059979).abs() < 1e-3, "std sum {std_sum}");
        assert_eq!(H3_LATENTS_MEAN[0], 0.858090341091156);
        assert_eq!(H3_LATENTS_STD[23], 2.6127843856811523);
    }
}

// ------------------------------------------------------------- execution --

/// Which arithmetic the latent path runs in. `Bf16` is what the reference
/// itself runs (`h3_upscale.py:82` casts the upscaler to bfloat16 and the
/// adapter's `from_pretrained` defaults to it); `F32` widens the same bf16
/// weights and computes in f32, which is the arm the gate compares tightly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Bf16,
    F32,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Bf16 => "bf16",
            Arm::F32 => "f32",
        }
    }

    fn dtype(self) -> Dtype {
        match self {
            Arm::Bf16 => Dtype::Bf16,
            Arm::F32 => Dtype::F32,
        }
    }

    /// `H3_LTX_ARM`, the reference's own bf16 unless it names the diagnostic
    /// f32 arm, so a binary that sets nothing is the released path.
    pub fn from_env() -> Arm {
        match std::env::var("H3_LTX_ARM") {
            Ok(value) if value.eq_ignore_ascii_case("f32") => Arm::F32,
            _ => Arm::Bf16,
        }
    }
}

/// One tensor's place in the checkpoint: which shard, which byte range, and
/// what the header says it is.
struct Site {
    shard: usize,
    begin: usize,
    bytes: usize,
    shape: Vec<u64>,
    dtype: StDtype,
}

/// The two checkpoints, read the way `vae_store` reads the VAE's: shards by
/// path, the first shard holding a name winning, each mapped on first use.
/// Unlike the VAE store these hold bf16, so a load either keeps the stored
/// bytes (the bf16 arm) or widens them to f32 (the diagnostic arm).
struct Store {
    shards: Vec<PathBuf>,
    sites: HashMap<String, Site>,
    maps: Vec<OnceLock<MappedFile>>,
    label: &'static str,
}

impl Store {
    fn open(path: &Path, label: &'static str) -> Result<Store, LatentError> {
        let shards: Vec<PathBuf> = if path.is_dir() {
            let store = safetensors::open_dir(path)
                .map_err(|error| LatentError::Io(format!("{label} reader: {error:?}")))?;
            store
                .shards()
                .iter()
                .map(|shard| shard.path.clone())
                .collect()
        } else {
            vec![path.to_path_buf()]
        };
        if shards.is_empty() {
            return Err(LatentError::Io(format!(
                "{label}: no safetensors shard under {}",
                path.display()
            )));
        }

        let mut sites = HashMap::new();
        for (index, shard_path) in shards.iter().enumerate() {
            let shard = safetensors::read_shard(shard_path).map_err(|error| {
                LatentError::Io(format!("{label} shard {}: {error:?}", shard_path.display()))
            })?;
            for tensor in &shard.tensors {
                sites.entry(tensor.name.clone()).or_insert_with(|| Site {
                    shard: index,
                    begin: usize::try_from(tensor.file_offset).unwrap_or(usize::MAX),
                    bytes: usize::try_from(tensor.bytes()).unwrap_or(usize::MAX),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                });
            }
        }

        let maps = (0..shards.len()).map(|_| OnceLock::new()).collect();
        Ok(Store {
            shards,
            sites,
            maps,
            label,
        })
    }

    fn map(&self, shard: usize) -> Result<&MappedFile, LatentError> {
        let cell = &self.maps[shard];
        if let Some(map) = cell.get() {
            return Ok(map);
        }
        let map = MappedFile::open_ro(&self.shards[shard])
            .map_err(|error| LatentError::Io(format!("map {} shard: {error}", self.label)))?;
        let _ = cell.set(map);
        cell.get()
            .ok_or_else(|| LatentError::Io(format!("{} shard map lost", self.label)))
    }

    /// The checkpoint's own shape for `name`, which is where every dimension
    /// in this module comes from: `_detect_arch` (`:379-411`) reads the same
    /// headers rather than trusting the constructor's defaults.
    fn shape(&self, name: &str) -> Result<&[u64], LatentError> {
        Ok(&self
            .sites
            .get(name)
            .ok_or_else(|| LatentError::Shape(format!("{} is missing {name}", self.label)))?
            .shape)
    }

    fn has(&self, name: &str) -> bool {
        self.sites.contains_key(name)
    }

    fn bytes(&self, name: &str, shape: &[u64]) -> Result<(&[u8], StDtype), LatentError> {
        let site = self
            .sites
            .get(name)
            .ok_or_else(|| LatentError::Shape(format!("{} is missing {name}", self.label)))?;
        if shape != site.shape.as_slice() {
            return Err(LatentError::Shape(format!(
                "{name}: wanted {shape:?}, checkpoint has {:?}",
                site.shape
            )));
        }
        let map = self.map(site.shard)?;
        let slice = map
            .as_slice()
            .get(site.begin..site.begin + site.bytes)
            .ok_or_else(|| LatentError::Shape(format!("{name}: range outside its shard")))?;
        Ok((slice, site.dtype))
    }

    fn load(&self, gpu: &Gpu, arm: Arm, name: &str, shape: &[u64]) -> Result<Tensor, LatentError> {
        let (bytes, stored) = self.bytes(name, shape)?;
        let elements: usize = shape.iter().product::<u64>() as usize;
        let tensor = device(gpu.alloc(arm.dtype(), elements))?;

        match (stored, arm) {
            (StDtype::Bf16, Arm::Bf16) | (StDtype::F32, Arm::F32) => {
                device(gpu.upload(&tensor, bytes))?;
            }
            (StDtype::Bf16, Arm::F32) => {
                // The exact widening: a bf16 value is its bits shifted into the
                // top half of an f32, so this loses nothing.
                let widened: Vec<u8> = bytes
                    .chunks_exact(2)
                    .flat_map(|pair| {
                        let bits = u16::from_le_bytes([pair[0], pair[1]]) as u32;
                        f32::from_bits(bits << 16).to_le_bytes()
                    })
                    .collect();
                device(gpu.upload(&tensor, &widened))?;
            }
            (other, _) => {
                return Err(LatentError::Shape(format!(
                    "{name}: checkpoint dtype {other:?} is not bf16"
                )));
            }
        }

        Ok(tensor)
    }
}

/// One convolution as the checkpoint stores it. `weight` is PyTorch's
/// `[out][in][kd][kh][kw]`, whose memory order *is* the `(out, in*kd*kh*kw)`
/// matrix the im2col matmul wants, so the classic path never permutes a weight.
///
/// `packed` is the same values in the planar path's order —
/// `[kt*kh + ky][out][s*in + c]`, one `(out, kw*in)` matrix per `(kt, ky)` tap —
/// built once at load by `h3b_weight_pack`. The classic path's `[out][in][kd]
/// [kh][kw]` order cannot express it, because there the `kw` triple is
/// contiguous and the channel stride is `kd*kh*kw`; the copy changes no value.
struct Conv {
    weight: Tensor,
    packed: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
    kernel: [u32; 3],
    padding: [u32; 3],
}

/// Which convolution the latent path runs. The planar one (`h3b_planar` plus
/// `h3b_matmul_planar`) never materializes an im2col matrix; the classic one
/// does, and is the A/B and the fallback. `H3_LTX_CONV=classic` selects it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvPath {
    Planar,
    Classic,
}

fn conv_path() -> ConvPath {
    static PATH: OnceLock<ConvPath> = OnceLock::new();
    *PATH.get_or_init(|| match std::env::var("H3_LTX_CONV").as_deref() {
        Ok("classic") => ConvPath::Classic,
        _ => ConvPath::Planar,
    })
}

/// The planar convolution: `h3b_planar` writes the padded activation once per
/// kernel column tap, then each of the `kd*kh` tap pairs is one dense matmul
/// whose A operand is a window of that buffer — leading dimension
/// `kw*channels`, the element (row `m`, column `k`) at `base + m*columns + k` —
/// accumulating in f32, and the epilogue transposes the frame's result into
/// channel-major order.
///
/// One frame at a time, and that is the point of the form: the A window a frame
/// reads is `kd` images of `kh + out_height` rows of `kw*channels` halfwords,
/// which is small enough that the taps' nine passes over it are served by L2
/// rather than by DRAM. The materialized im2col moved 104 GiB per conversion.
///
/// What is *not* worth doing is blocking the frame's rows so the accumulator
/// fits L2 by itself: at 1,024 rows the call has 32 CTAs on a 48-SM part, and
/// the measured cost of losing the waves is larger than the accumulator's DRAM
/// walk (2,283 -> 3,158 us on the workstation, 2026-10-29). The taps stay whole
/// plane, which is 2.6 waves at the widest convolution.
fn convolve_planar(
    gpu: &Gpu,
    arm: Arm,
    planar: &Tensor,
    temp: &Tensor,
    out: &Tensor,
    input: &Tensor,
    conv: &Conv,
    shape: Shape,
) -> Result<(), LatentError> {
    let dtype = arm.dtype();
    let [kt, kh, kw] = conv.kernel;
    let [pt, ph, pw] = conv.padding;
    let frames = shape.frames;
    let out_height = shape.height + 2 * ph - kh + 1;
    let out_width = shape.width + 2 * pw - kw + 1;
    let out_plane = out_height as usize * out_width as usize;
    let padded_height = (shape.height + 2 * ph) as usize;
    let columns = conv.input_channels as usize * kw as usize;

    // Fail loud rather than let a mis-sized scratch truncate the buffer: the
    // A windows walk `frames + kd - 1` images of `height + kh - 1` rows.
    let needed = planar_elements(
        conv.input_channels as u64,
        frames as u64,
        shape.height as u64,
        out_width as u64,
        [kt as u64, kh as u64, kw as u64],
    ) as usize;
    if planar.elements() < needed {
        return Err(LatentError::Shape(format!(
            "planar scratch holds {} elements, this convolution reads {needed}",
            planar.elements()
        )));
    }
    if temp.elements() < out_plane * conv.output_channels as usize {
        return Err(LatentError::Shape(format!(
            "accumulator holds {} elements, this convolution writes {}",
            temp.elements(),
            out_plane * conv.output_channels as usize
        )));
    }

    device(gpu.latent_planar(
        planar,
        input,
        dtype,
        conv.input_channels,
        frames,
        shape.height,
        shape.width,
        kw,
        pt,
        ph,
        pw,
        out_width,
        false,
    ))?;

    for frame in 0..frames {
        for tap_t in 0..kt {
            for tap_h in 0..kh {
                let pass = (tap_t * kh + tap_h) as usize;
                // The tap's window: image row `(frame + tap_t, tap_h)` of the
                // planar buffer, which is `m*columns` from the window's start
                // for the output row `m` — the planar layout puts consecutive
                // image positions exactly one matrix row apart.
                let offset = ((frame + tap_t) as usize * padded_height + tap_h as usize)
                    * out_width as usize
                    * columns;
                device(gpu.latent_matmul_planar(
                    temp,
                    planar,
                    offset,
                    &conv.packed,
                    pass * conv.output_channels as usize * columns,
                    0,
                    out_plane as u32,
                    columns as u32,
                    conv.output_channels,
                    tap_t != 0 || tap_h != 0,
                    dtype,
                ))?;
            }
        }
        device(gpu.latent_epilogue(
            out,
            temp,
            Some(&conv.bias),
            dtype,
            conv.output_channels,
            out_plane as u32,
            frames as usize * out_plane,
            frame as usize * out_plane,
        ))?;
    }
    Ok(())
}

/// A normalization and its fixed axes: `channels` per group and the epsilon
/// the module was built with (the upscaler's GroupNorm defaults to 1e-5, the
/// adapter passes 1e-6).
struct Norm {
    weight: Tensor,
    bias: Tensor,
    channels: u32,
    groups: u32,
    epsilon: f32,
}

/// GroupNorm's scratch, sized from the largest plane the model normalizes.
struct NormScratch {
    partial: Tensor,
    mean: Tensor,
    rstd: Tensor,
}

/// `nn.GroupNorm` with the ResBlock epilogues folded in — one call site shape
/// for every normalization in both models.
#[allow(clippy::too_many_arguments)]
fn normalise(
    gpu: &Gpu,
    arm: Arm,
    scratch: &NormScratch,
    out: &Tensor,
    input: &Tensor,
    norm: &Norm,
    embedding: Option<&Tensor>,
    silu: bool,
    shape: Shape,
) -> Result<(), LatentError> {
    device(gpu.latent_norm(
        out,
        input,
        &norm.weight,
        &norm.bias,
        embedding,
        &scratch.partial,
        &scratch.mean,
        &scratch.rstd,
        arm.dtype(),
        norm.channels,
        norm.groups,
        shape.frames,
        shape.area() as u32,
        norm.epsilon,
        silu,
    ))
}

/// A 1x1 convolution: a matmul over the channel-major activation, no copy.
struct Pointwise {
    weight: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
}

/// The shapes one op sees, in the family's channel-major order.
#[derive(Clone, Copy, Debug)]
struct Shape {
    channels: u32,
    frames: u32,
    height: u32,
    width: u32,
}

impl Shape {
    /// The elements one channel holds: what a matmul's `spatial` argument wants.
    fn plane(&self) -> usize {
        self.frames as usize * self.height as usize * self.width as usize
    }

    /// `height * width`: what a normalization's `spatial` argument wants, since
    /// the op multiplies it by `rows` itself.
    fn area(&self) -> usize {
        self.height as usize * self.width as usize
    }

    fn elements(&self) -> usize {
        self.channels as usize * self.plane()
    }
}

fn load_conv(
    store: &Store,
    gpu: &Gpu,
    arm: Arm,
    name: &str,
    kernel: [u32; 3],
    padding: [u32; 3],
) -> Result<Conv, LatentError> {
    let shape = store.shape(&format!("{name}.weight"))?;
    if shape.len() != 5 {
        return Err(LatentError::Shape(format!("{name}.weight is not 5-D")));
    }
    let stored = [shape[2] as u32, shape[3] as u32, shape[4] as u32];
    if stored != kernel {
        return Err(LatentError::Shape(format!(
            "{name}.weight kernel {stored:?}, the architecture says {kernel:?}"
        )));
    }

    let weight = store.load(gpu, arm, &format!("{name}.weight"), shape)?;
    let output_channels = shape[0] as u32;
    let input_channels = shape[1] as u32;
    let taps = kernel[0] * kernel[1] * kernel[2];
    let packed = device(gpu.alloc(arm.dtype(), weight.elements()))?;
    device(gpu.latent_weight_pack(
        &packed,
        &weight,
        arm.dtype(),
        output_channels,
        input_channels,
        taps,
        kernel[2],
    ))?;

    Ok(Conv {
        weight,
        packed,
        bias: store.load(
            gpu,
            arm,
            &format!("{name}.bias"),
            store.shape(&format!("{name}.bias"))?,
        )?,
        input_channels,
        output_channels,
        kernel,
        padding,
    })
}

fn load_pointwise(
    store: &Store,
    gpu: &Gpu,
    arm: Arm,
    name: &str,
) -> Result<Pointwise, LatentError> {
    let shape = store.shape(&format!("{name}.weight"))?;
    let bias = store.shape(&format!("{name}.bias"))?.to_vec();
    if shape.len() != 5 || shape[2] != 1 || shape[3] != 1 || shape[4] != 1 {
        return Err(LatentError::Shape(format!("{name}.weight is not 1x1x1")));
    }
    if bias.len() != 1 || bias[0] != shape[0] {
        return Err(LatentError::Shape(format!("{name}.bias is not per output")));
    }

    Ok(Pointwise {
        weight: store.load(gpu, arm, &format!("{name}.weight"), shape)?,
        bias: store.load(gpu, arm, &format!("{name}.bias"), &bias)?,
        input_channels: shape[1] as u32,
        output_channels: shape[0] as u32,
    })
}

fn load_norm(
    store: &Store,
    gpu: &Gpu,
    arm: Arm,
    name: &str,
    channels: u32,
    groups: u32,
    epsilon: f32,
) -> Result<Norm, LatentError> {
    let shape = store.shape(&format!("{name}.weight"))?;
    if shape != [channels as u64] {
        return Err(LatentError::Shape(format!(
            "{name}.weight is {shape:?}, not {channels}"
        )));
    }

    Ok(Norm {
        weight: store.load(gpu, arm, &format!("{name}.weight"), shape)?,
        bias: store.load(gpu, arm, &format!("{name}.bias"), shape)?,
        channels,
        groups,
        epsilon,
    })
}

fn upload_u32(gpu: &Gpu, values: &[u32]) -> Result<Tensor, LatentError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = device(gpu.alloc(Dtype::U32, values.len()))?;
    device(gpu.upload(&tensor, &bytes))?;
    Ok(tensor)
}

fn upload_f32(gpu: &Gpu, values: &[f32]) -> Result<Tensor, LatentError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = device(gpu.alloc(Dtype::F32, values.len()))?;
    device(gpu.upload(&tensor, &bytes))?;
    Ok(tensor)
}

/// `constants.py:13-66`'s statistics as the node's own buffers, in the arm's
/// dtype: the reference builds them with `torch.tensor(..., dtype=dtype)`
/// (`:171-172`), so in bf16 the constants themselves are rounded and this
/// rounds them the same way.
fn statistics(gpu: &Gpu, arm: Arm) -> Result<(Tensor, Tensor), LatentError> {
    let center = device(gpu.alloc(arm.dtype(), H3_LATENT_CHANNELS))?;
    let scale = device(gpu.alloc(arm.dtype(), H3_LATENT_CHANNELS))?;
    let bytes = |values: &[f32]| -> Vec<u8> {
        match arm {
            Arm::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            // `torch.tensor(value, dtype=torch.bfloat16)` rounds to nearest
            // even; truncating the low 16 bits would be half an ulp off.
            Arm::Bf16 => values
                .iter()
                .flat_map(|v| {
                    let bits = v.to_bits();
                    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                    ((rounded >> 16) as u16).to_le_bytes()
                })
                .collect(),
        }
    };
    device(gpu.upload(&center, &bytes(&H3_LATENTS_MEAN)))?;
    device(gpu.upload(&scale, &bytes(&H3_LATENTS_STD)))?;
    Ok((center, scale))
}

/// A `nn.Linear` as the checkpoints store it: `(out, in)`.
struct Linear {
    weight: Tensor,
    bias: Tensor,
    input_dim: u32,
    output_dim: u32,
}

fn load_linear(store: &Store, gpu: &Gpu, arm: Arm, name: &str) -> Result<Linear, LatentError> {
    let shape = store.shape(&format!("{name}.weight"))?;
    let bias = store.shape(&format!("{name}.bias"))?;
    if shape.len() != 2 || bias != [shape[0]] {
        return Err(LatentError::Shape(format!("{name} is not a 2-D linear")));
    }

    Ok(Linear {
        weight: store.load(gpu, arm, &format!("{name}.weight"), shape)?,
        bias: store.load(gpu, arm, &format!("{name}.bias"), bias)?,
        input_dim: shape[1] as u32,
        output_dim: shape[0] as u32,
    })
}

/// `ResBlockEmb3D` (`minimax_h3_latent_upscaler_3d.py:160-189`): GroupNorm +
/// SiLU + 3x3x3, then the embedding's scale/shift on a second GroupNorm, then
/// SiLU and a zero-initialized 3x3x3, added to the block's input. `skip` is
/// `Identity` here because `out_channels == channels`.
struct ResBlock {
    in_norm: Norm,
    in_conv: Conv,
    embedding_weight: Tensor,
    embedding_bias: Tensor,
    out_norm: Norm,
    out_conv: Conv,
}

/// `TemporalConv` (`:191-213`): `x + pwconv(dwconv(silu(norm(x))))`.
struct TemporalBlock {
    norm: Norm,
    depth_weight: Tensor,
    depth_bias: Tensor,
    pointwise: Pointwise,
}

enum UpBlock {
    Res(ResBlock),
    Temporal(TemporalBlock),
}

struct UpModel {
    conv_in: Conv,
    embed_in: Linear,
    embed_hidden: Linear,
    in_blocks: Vec<UpBlock>,
    out_blocks: Vec<UpBlock>,
    norm_out: Norm,
    conv_out: Conv,
}

impl UpModel {
    fn load(store: &Store, gpu: &Gpu, arm: Arm) -> Result<UpModel, LatentError> {
        let channels = UPSCALER_CHANNELS as u32;
        let kernel = [3u32, 3, 3];
        let pad = [1u32, 1, 1];

        // The list is `in_blocks`/`out_blocks` as the constructor builds it
        // (`:228-238`, `:240`): a ResBlock, and a TemporalConv after every
        // `temporal_every`-th one, so the list has
        // `blocks + blocks/temporal_every` entries and the temporal ones sit at
        // every third index.
        let entries = UPSCALER_BLOCKS + UPSCALER_BLOCKS / UPSCALER_TEMPORAL_EVERY;
        let blocks = |side: &str| -> Result<Vec<UpBlock>, LatentError> {
            let mut blocks = Vec::new();
            for index in 0..entries {
                let prefix = format!("{side}.{index}");
                let temporal = format!("{prefix}.dwconv.weight");
                if store.has(&temporal) {
                    if index % 3 != 1 {
                        return Err(LatentError::Shape(format!(
                            "{prefix} is a TemporalConv at an unexpected index"
                        )));
                    }
                    let kernel_shape = store.shape(&temporal)?.to_vec();
                    let depth_weight = store.load(gpu, arm, &temporal, &kernel_shape)?;
                    let depth_bias = store.load(
                        gpu,
                        arm,
                        &format!("{prefix}.dwconv.bias"),
                        store.shape(&format!("{prefix}.dwconv.bias"))?,
                    )?;
                    blocks.push(UpBlock::Temporal(TemporalBlock {
                        norm: load_norm(
                            store,
                            gpu,
                            arm,
                            &format!("{prefix}.norm"),
                            channels,
                            UPSCALER_GROUPS as u32,
                            UPSCALER_EPSILON,
                        )?,
                        depth_weight,
                        depth_bias,
                        pointwise: load_pointwise(store, gpu, arm, &format!("{prefix}.pwconv"))?,
                    }));
                    continue;
                }

                blocks.push(UpBlock::Res(ResBlock {
                    in_norm: load_norm(
                        store,
                        gpu,
                        arm,
                        &format!("{prefix}.in_layers.0"),
                        channels,
                        UPSCALER_GROUPS as u32,
                        UPSCALER_EPSILON,
                    )?,
                    in_conv: load_conv(
                        store,
                        gpu,
                        arm,
                        &format!("{prefix}.in_layers.2"),
                        kernel,
                        pad,
                    )?,
                    embedding_weight: store.load(
                        gpu,
                        arm,
                        &format!("{prefix}.emb_layers.1.weight"),
                        store.shape(&format!("{prefix}.emb_layers.1.weight"))?,
                    )?,
                    embedding_bias: store.load(
                        gpu,
                        arm,
                        &format!("{prefix}.emb_layers.1.bias"),
                        store.shape(&format!("{prefix}.emb_layers.1.bias"))?,
                    )?,
                    out_norm: load_norm(
                        store,
                        gpu,
                        arm,
                        &format!("{prefix}.out_norm"),
                        channels,
                        UPSCALER_GROUPS as u32,
                        UPSCALER_EPSILON,
                    )?,
                    out_conv: load_conv(
                        store,
                        gpu,
                        arm,
                        &format!("{prefix}.out_layers.2"),
                        kernel,
                        pad,
                    )?,
                }));
            }
            Ok(blocks)
        };

        let model = UpModel {
            conv_in: load_conv(store, gpu, arm, "conv_in", kernel, pad)?,
            embed_in: load_linear(store, gpu, arm, "embed.0")?,
            embed_hidden: load_linear(store, gpu, arm, "embed.2")?,
            in_blocks: blocks("in_blocks")?,
            out_blocks: blocks("out_blocks")?,
            norm_out: load_norm(
                store,
                gpu,
                arm,
                "norm_out",
                channels,
                UPSCALER_GROUPS as u32,
                UPSCALER_EPSILON,
            )?,
            conv_out: load_conv(store, gpu, arm, "conv_out", [3, 3, 3], [1, 1, 1])?,
        };

        for (side, blocks) in [
            ("in_blocks", &model.in_blocks),
            ("out_blocks", &model.out_blocks),
        ] {
            let res = blocks
                .iter()
                .filter(|block| matches!(block, UpBlock::Res(_)))
                .count();
            if res != UPSCALER_BLOCKS {
                return Err(LatentError::Shape(format!(
                    "{side} has {res} ResBlocks, the architecture says {UPSCALER_BLOCKS}"
                )));
            }
        }

        Ok(model)
    }
}

/// Every convolution in the upscaler (`load` builds all of them from this one
/// kernel shape).
const UPSCALER_KERNEL: [u64; 3] = [3, 3, 3];

/// The planar buffer's element count for a convolution that reads
/// `(channels, frames, height, ·)`: the padded image in channel-last form, once
/// per column tap. `out_width` is the convolution's own output width, which for
/// a same-shape kernel is the input width.
fn planar_elements(
    channels: u64,
    frames: u64,
    height: u64,
    out_width: u64,
    kernel: [u64; 3],
) -> u64 {
    channels * (frames + kernel[0] - 1) * (height + kernel[1] - 1) * out_width * kernel[2]
}

/// The zero-padded element count of `[channels, frames + 2, height + 2,
fn padded_elements(channels: u64, frames: u64, height: u64, width: u64) -> u64 {
    channels * (frames + 2) * (height + 2) * (width + 2)
}

/// The frames one convolution slab may cover: what the im2col matrix holds,
/// what the f32 accumulator holds, and the tensor itself, whichever is least.
fn slab_frames(matrix: usize, accumulator: usize, frames: usize) -> u32 {
    matrix.min(accumulator).clamp(1, frames.max(1)) as u32
}

/// How many frames of the largest convolution are im2col'd at once. The
/// release's `out_blocks` grid would need 4.1 GiB for the whole thing; four
/// frames is a bounded setting that keeps the matrix at ~450 MiB and still
/// gives `cublasLt` a 16,128-row GEMM.
const SLAB_FRAMES: usize = 4;

/// Every device buffer one upscaler forward touches, sized once from the
/// geometry: the chunk plan fixes the largest segment, so nothing grows at run
/// time.
struct UpScratch {
    normalized: Tensor,
    replicated: Tensor,
    segment: Tensor,
    padded: Tensor,
    acts: [Tensor; 4],
    matrix: Tensor,
    /// The planar path's A operand: the padded activation in channel-last form,
    /// `kw` copies wide, sized for the largest convolution at the largest
    /// segment (`conv3d`'s classic side uses `matrix` instead).
    planar: Tensor,
    temp: Tensor,
    accumulator: Tensor,
    weight_total: Tensor,
    norm: NormScratch,
    pad_index: Tensor,
    /// One index map per segment: the frames it reads, in absolute coordinates
    /// of the padded axis. Reading from zero would silently slice the wrong
    /// window for every segment after the first.
    slice_indices: Vec<Tensor>,
    segment_weights: Vec<Tensor>,
    blended: Tensor,
}

impl UpScratch {
    fn new(
        gpu: &Gpu,
        arm: Arm,
        geometry: &Geometry,
        plan: &ChunkPlan,
    ) -> Result<UpScratch, LatentError> {
        let dtype = arm.dtype();
        let [channels, frames, height, width] = geometry.upscaler_input;
        let (frames, height, width) = (frames as u32, height as u32, width as u32);
        let overlap = UPSCALER_TEMPORAL_KERNEL;
        let padded_frames = frames as usize + 2 * overlap;
        let largest = plan
            .segments
            .iter()
            .map(|segment| segment.input.1 - segment.input.0)
            .max()
            .unwrap_or(0) as u32;
        let out_height = height * 2;
        let out_width = width * 2;
        let out_plane = out_height as usize * out_width as usize;
        let small_plane = height as usize * width as usize;

        let alloc =
            |elements: usize| -> Result<Tensor, LatentError> { device(gpu.alloc(dtype, elements)) };
        let alloc_f32 = |elements: usize| -> Result<Tensor, LatentError> {
            device(gpu.alloc(Dtype::F32, elements))
        };

        let activations = Shape {
            channels: UPSCALER_CHANNELS as u32,
            frames: largest,
            height: out_height,
            width: out_width,
        };
        // The largest planar buffer: the padded image in channel-last form,
        // once per column tap. The upscaler's three convolutions differ in
        // input channel count (24 for `conv_in`, 512 for the rest) and in
        // resolution, so all three are sized and the biggest wins.
        let planar = [
            planar_elements(
                UPSCALER_CHANNELS as u64,
                largest as u64,
                out_height as u64,
                out_width as u64,
                UPSCALER_KERNEL,
            ),
            planar_elements(
                UPSCALER_CHANNELS as u64,
                largest as u64,
                height as u64,
                width as u64,
                UPSCALER_KERNEL,
            ),
            planar_elements(
                UPSCALER_IN_CHANNELS as u64,
                largest as u64,
                height as u64,
                width as u64,
                UPSCALER_KERNEL,
            ),
        ]
        .into_iter()
        .max()
        .unwrap_or(0) as usize;
        // The largest zero-padded activation any convolution materializes: one
        // frame of halo on every axis, at whichever of the two resolutions is
        // bigger.
        let padded = [
            padded_elements(
                UPSCALER_CHANNELS as u64,
                largest as u64,
                out_height as u64,
                out_width as u64,
            ),
            padded_elements(
                UPSCALER_CHANNELS as u64,
                largest as u64,
                height as u64,
                width as u64,
            ),
        ]
        .into_iter()
        .max()
        .unwrap_or(0) as usize;
        // The im2col matrix holds `SLAB_FRAMES` output frames of the 3x3x3 at
        // the widest resolution; the convolution computes its own slab from
        // whatever this leaves. Only the classic path writes it, so under the
        // planar one it stays at a single element instead of holding half a
        // gibibyte nothing reads.
        let matrix = match conv_path() {
            ConvPath::Planar => 1,
            ConvPath::Classic => SLAB_FRAMES * out_plane * UPSCALER_CHANNELS * 3 * 3 * 3,
        };
        // The slabs differ per convolution: one that runs at the narrow
        // resolution fits the whole segment in the matrix, so its row count is
        // the segment's frames, not `SLAB_FRAMES`.
        let temp = SLAB_FRAMES
            .max(largest as usize)
            .max(largest as usize * small_plane / out_plane)
            * out_plane
            * UPSCALER_CHANNELS;

        // GroupNorm's scratch: one (sum, sum of squares) pair per 8,192-element
        // block of each group's channel plane.
        let max_plane = (largest as usize * out_plane).max(frames as usize * small_plane);
        let chunks = max_plane.div_ceil(8192);
        let partial = UPSCALER_GROUPS * (UPSCALER_CHANNELS / UPSCALER_GROUPS) * chunks * 2;

        let mut weights = Vec::new();
        let mut slices = Vec::new();
        for segment in &plan.segments {
            weights.push(upload_f32(gpu, &segment.weights)?);
            let frames: Vec<u32> = (segment.input.0..segment.input.1)
                .map(|frame| frame as u32)
                .collect();
            slices.push(upload_u32(gpu, &frames)?);
        }
        let totals = upload_f32(gpu, &weight_totals(plan))?;

        Ok(UpScratch {
            normalized: alloc(channels * frames as usize * small_plane)?,
            replicated: alloc(channels * padded_frames * small_plane)?,
            segment: alloc((channels as u32 * largest) as usize * small_plane)?,
            padded: alloc(padded)?,
            acts: [
                alloc(activations.elements())?,
                alloc(activations.elements())?,
                alloc(activations.elements())?,
                alloc(activations.elements())?,
            ],
            matrix: alloc(matrix)?,
            planar: alloc(planar)?,
            temp: alloc_f32(temp)?,
            accumulator: alloc_f32(H3_LATENT_CHANNELS * frames as usize * out_plane)?,
            weight_total: totals,
            norm: NormScratch {
                partial: alloc_f32(partial)?,
                mean: alloc_f32(UPSCALER_GROUPS)?,
                rstd: alloc_f32(UPSCALER_GROUPS)?,
            },
            pad_index: upload_u32(gpu, &replicate_frames(frames as usize, overlap))?,
            slice_indices: slices,
            segment_weights: weights,
            blended: alloc(H3_LATENT_CHANNELS * frames as usize * out_plane)?,
        })
    }
}

/// The per-frame weight total `weight_full` reaches: the sum of the segment
/// ramps that cover each output frame, in the order `forward` accumulates them
/// (`:306`), so the weighted average divides by exactly the same number.
fn weight_totals(plan: &ChunkPlan) -> Vec<f32> {
    let mut totals = vec![0.0f32; plan.frames];
    for segment in &plan.segments {
        for (index, weight) in segment.weights.iter().enumerate() {
            totals[segment.output.0 + index] += weight;
        }
    }
    totals
}

/// A single-precision scalar in the arm's dtype.
fn upload_scalar(gpu: &Gpu, arm: Arm, value: f32) -> Result<Tensor, LatentError> {
    let tensor = device(gpu.alloc(arm.dtype(), 1))?;
    let bits = value.to_bits();
    let bytes: [u8; 2] = ((bits >> 16) as u16).to_le_bytes();
    let payload: Vec<u8> = match arm {
        Arm::F32 => bits.to_le_bytes().to_vec(),
        Arm::Bf16 => bytes.to_vec(),
    };
    device(gpu.upload(&tensor, &payload))?;
    Ok(tensor)
}

/// The node's `scale` argument (`h3_upscale.py:122`), so the scalar the
/// embedding is built from is `scale - 1` = 1.
const UPSCALER_SCALE: f32 = 2.0;

/// `self.embed` (`:221-223`) then one `emb_layers` per ResBlock (`:168-170`):
/// the 64-vector is shared, the 2x channels projection is not.
///
/// Called once at load, because the scalar it starts from is a constant.
fn compute_embeddings(
    gpu: &Gpu,
    arm: Arm,
    model: &UpModel,
    temp: &Tensor,
) -> Result<(Vec<Tensor>, Vec<Tensor>), LatentError> {
    let dtype = arm.dtype();
    let scalar = upload_scalar(gpu, arm, UPSCALER_SCALE - 1.0)?;
    let hidden = device(gpu.alloc(dtype, UPSCALER_EMBED_DIM))?;
    let shared = device(gpu.alloc(dtype, UPSCALER_EMBED_DIM))?;

    // Linear(1, 64) -> SiLU -> Linear(64, 64).
    device(gpu.latent_matmul(
        temp,
        &scalar,
        &model.embed_in.weight,
        1,
        model.embed_in.input_dim,
        model.embed_in.output_dim,
        dtype,
    ))?;
    device(gpu.latent_epilogue(
        &hidden,
        temp,
        Some(&model.embed_in.bias),
        dtype,
        model.embed_in.output_dim,
        1,
        1,
        0,
    ))?;
    device(gpu.latent_silu(&hidden, &hidden, dtype, UPSCALER_EMBED_DIM))?;
    device(gpu.latent_matmul(
        temp,
        &hidden,
        &model.embed_hidden.weight,
        1,
        model.embed_hidden.input_dim,
        model.embed_hidden.output_dim,
        dtype,
    ))?;
    device(gpu.latent_epilogue(
        &shared,
        temp,
        Some(&model.embed_hidden.bias),
        dtype,
        model.embed_hidden.output_dim,
        1,
        1,
        0,
    ))?;

    // `emb_layers` is `Sequential(SiLU(), Linear(...))`, so the shared vector
    // is activated again before each block's own projection.
    let activated = device(gpu.alloc(dtype, UPSCALER_EMBED_DIM))?;
    device(gpu.latent_silu(&activated, &shared, dtype, UPSCALER_EMBED_DIM))?;

    let project = |side: &[UpBlock]| -> Result<Vec<Tensor>, LatentError> {
        let mut values = Vec::new();
        for block in side {
            let UpBlock::Res(res) = block else {
                continue;
            };
            let width = 2 * res.out_norm.channels;
            let value = device(gpu.alloc(dtype, width as usize))?;
            device(gpu.latent_matmul(
                temp,
                &activated,
                &res.embedding_weight,
                1,
                UPSCALER_EMBED_DIM as u32,
                width,
                dtype,
            ))?;
            device(gpu.latent_epilogue(
                &value,
                temp,
                Some(&res.embedding_bias),
                dtype,
                width,
                1,
                1,
                0,
            ))?;
            values.push(value);
        }
        Ok(values)
    };

    Ok((project(&model.in_blocks)?, project(&model.out_blocks)?))
}

/// The x2 latent upscaler: the pinned author's `LatentResizer3D` with the
/// profile's own `N` and `D` around it (`h3_upscale.py:112-126`).
pub struct Upscaler {
    gpu: Gpu,
    arm: Arm,
    geometry: Geometry,
    plan: ChunkPlan,
    model: UpModel,
    scratch: UpScratch,
    center: Tensor,
    scale: Tensor,
    in_embeddings: Vec<Tensor>,
    out_embeddings: Vec<Tensor>,
}

impl Upscaler {
    pub fn load(checkpoint: &Path, arm: Arm, geometry: &Geometry) -> Result<Upscaler, LatentError> {
        let gpu = device(Gpu::create())?;
        let store = Store::open(checkpoint, "latent upscaler")?;
        let model = UpModel::load(&store, &gpu, arm)?;
        if model.conv_in.input_channels as usize != H3_LATENT_CHANNELS
            || model.conv_out.output_channels as usize != H3_LATENT_CHANNELS
        {
            return Err(LatentError::Shape(format!(
                "upscaler channels {} -> {}, expected {H3_LATENT_CHANNELS}",
                model.conv_in.input_channels, model.conv_out.output_channels
            )));
        }

        let plan = chunk_plan(
            geometry.upscaler_input[1],
            UPSCALER_TEMPORAL_KERNEL,
            UPSCALER_CHUNK,
        );
        let scratch = UpScratch::new(&gpu, arm, geometry, &plan)?;
        let (center, scale) = statistics(&gpu, arm)?;
        let (in_embeddings, out_embeddings) = compute_embeddings(&gpu, arm, &model, &scratch.temp)?;
        device(gpu.sync())?;

        Ok(Upscaler {
            gpu,
            arm,
            geometry: geometry.clone(),
            plan,
            model,
            scratch,
            center,
            scale,
            in_embeddings,
            out_embeddings,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn arm(&self) -> Arm {
        self.arm
    }

    /// The node's `D(model(N(x)))` over one already-normalized H3 latent,
    /// returning the upscaled latent at the doubled canvas. The result is
    /// synchronized, so another device context may read it.
    pub fn forward(&self, normalized: &Tensor) -> Result<Tensor, LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let [_, frames, height, width] = self.geometry.upscaler_input;
        let (frames, height, width) = (frames as u32, height as u32, width as u32);
        let small_plane = height as usize * width as usize;
        if normalized.elements() < H3_LATENT_CHANNELS * frames as usize * small_plane {
            return Err(LatentError::Shape(format!(
                "latent has {} elements, the upscaler reads {}",
                normalized.elements(),
                H3_LATENT_CHANNELS * frames as usize * small_plane
            )));
        }

        // N, then the replicate padding the chunk loop reads.
        device(gpu.latent_scale(
            &self.scratch.normalized,
            normalized,
            &self.center,
            &self.scale,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames as usize * small_plane,
            false,
        ))?;
        let overlap = UPSCALER_TEMPORAL_KERNEL as u32;
        device(gpu.latent_time_gather(
            &self.scratch.replicated,
            &self.scratch.normalized,
            &self.scratch.pad_index,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames,
            frames + 2 * overlap,
            small_plane as u32,
            1,
        ))?;

        let out_height = height * 2;
        let out_width = width * 2;
        let out_plane = out_height as usize * out_width as usize;

        // `h3b_accumulate` adds into a persistent f32 sum, so it starts at zero
        // on every forward, not only when the scratch was allocated: a
        // resident process runs the upscaler once per request, and the second
        // request would otherwise add its segments to the first's.
        device(gpu.zero(&self.scratch.accumulator))?;

        for (index, segment) in self.plan.segments.iter().enumerate() {
            let count = (segment.input.1 - segment.input.0) as u32;

            // The segment's own slice of the padded axis: `x_padded[:, :, lo:hi]`.
            device(gpu.latent_time_gather(
                &self.scratch.segment,
                &self.scratch.replicated,
                &self.scratch.slice_indices[index],
                dtype,
                H3_LATENT_CHANNELS as u32,
                frames + 2 * overlap,
                count,
                small_plane as u32,
                1,
            ))?;

            let input = Shape {
                channels: H3_LATENT_CHANNELS as u32,
                frames: count,
                height,
                width,
            };
            self.conv3d(
                &self.scratch.acts[0],
                &self.scratch.segment,
                &self.model.conv_in,
                input,
            )?;

            let mut shape = Shape {
                channels: UPSCALER_CHANNELS as u32,
                ..input
            };
            let mut current =
                self.run_blocks(&self.model.in_blocks, &self.in_embeddings, 0, &mut shape)?;

            // `F.interpolate(x, size=(t, h_out, w_out), mode="trilinear")`.
            let next = (current + 1) % 4;
            device(gpu.latent_resize3d(
                &self.scratch.acts[next],
                &self.scratch.acts[current],
                dtype,
                UPSCALER_CHANNELS as u32,
                [count, height, width],
                [count, out_height, out_width],
            ))?;
            shape = Shape {
                channels: UPSCALER_CHANNELS as u32,
                frames: count,
                height: out_height,
                width: out_width,
            };
            current = self.run_blocks(
                &self.model.out_blocks,
                &self.out_embeddings,
                next,
                &mut shape,
            )?;

            // `norm_out` + SiLU, then `conv_out`.
            let next = (current + 1) % 4;
            normalise(
                gpu,
                self.arm,
                &self.scratch.norm,
                &self.scratch.acts[next],
                &self.scratch.acts[current],
                &self.model.norm_out,
                None,
                true,
                Shape {
                    channels: UPSCALER_CHANNELS as u32,
                    frames: count,
                    height: out_height,
                    width: out_width,
                },
            )?;
            let output = (next + 1) % 4;
            self.conv3d(
                &self.scratch.acts[output],
                &self.scratch.acts[next],
                &self.model.conv_out,
                shape,
            )?;

            device(gpu.latent_accumulate(
                &self.scratch.accumulator,
                &self.scratch.acts[output],
                &self.scratch.segment_weights[index],
                dtype,
                H3_LATENT_CHANNELS as u32,
                frames,
                segment.weights.len() as u32,
                out_plane as u32,
                count,
                segment.source_begin as u32,
                segment.output.0 as u32,
            ))?;
        }

        device(gpu.latent_blend_finish(
            &self.scratch.blended,
            &self.scratch.accumulator,
            &self.scratch.weight_total,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames,
            out_plane as u32,
        ))?;

        // D, then the result.
        let result = device(gpu.alloc(dtype, H3_LATENT_CHANNELS * frames as usize * out_plane))?;
        device(gpu.latent_scale(
            &result,
            &self.scratch.blended,
            &self.center,
            &self.scale,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames as usize * out_plane,
            true,
        ))?;
        device(gpu.sync())?;
        Ok(result)
    }

    /// The whole handoff in one call: a denoised F32 latent arrives as the
    /// engine holds it, is rounded at the BF16 boundary the BF16 arm reads,
    /// doubled, and comes back as F32. The rounding is the one the two-process
    /// route writes (`h3-rs --save-latent` stores BF16), so both routes feed
    /// the network the same bytes for the same latent.
    pub fn upscale(&self, latent: &[f32]) -> Result<Vec<f32>, LatentError> {
        let expected = self.geometry.upscaler_input_elements();
        if latent.len() != expected {
            return Err(LatentError::Shape(format!(
                "upscaler takes {expected} values, got {}",
                latent.len()
            )));
        }

        let mut bytes = Vec::with_capacity(latent.len() * 2);
        for value in latent {
            bytes.extend_from_slice(&f32_to_bf16(*value).to_le_bytes());
        }

        let input = device(self.gpu.alloc(Dtype::Bf16, latent.len()))?;
        device(self.gpu.upload(&input, &bytes))?;
        let doubled = self.forward(&input)?;

        let mut out = vec![0u8; doubled.bytes()];
        device(self.gpu.read(&doubled, &mut out))?;

        Ok(out
            .chunks_exact(2)
            .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect())
    }

    /// One block list, from the slot holding its input to the slot holding its
    /// output. Four activation slots rotate, so a block never writes the slot
    /// it is reading.
    fn run_blocks(
        &self,
        blocks: &[UpBlock],
        embeddings: &[Tensor],
        start: usize,
        shape: &mut Shape,
    ) -> Result<usize, LatentError> {
        let mut current = start;
        let mut index = 0usize;
        for block in blocks {
            let a = (current + 1) % 4;
            let b = (current + 2) % 4;
            let out = (current + 3) % 4;
            match block {
                UpBlock::Temporal(temporal) => {
                    self.temporal_block(temporal, current, out, a, b, *shape)?;
                }
                UpBlock::Res(res) => {
                    self.res_block(res, &embeddings[index], current, out, a, b, *shape)?;
                    index += 1;
                }
            }
            current = out;
        }
        Ok(current)
    }

    /// `ResBlockEmb3D.forward` (`:181-189`).
    #[allow(clippy::too_many_arguments)]
    fn res_block(
        &self,
        block: &ResBlock,
        embedding: &Tensor,
        input: usize,
        out: usize,
        a: usize,
        b: usize,
        shape: Shape,
    ) -> Result<(), LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let acts = &self.scratch.acts;

        normalise(
            gpu,
            self.arm,
            &self.scratch.norm,
            &acts[a],
            &acts[input],
            &block.in_norm,
            None,
            true,
            shape,
        )?;
        self.conv3d(&acts[b], &acts[a], &block.in_conv, shape)?;
        normalise(
            gpu,
            self.arm,
            &self.scratch.norm,
            &acts[a],
            &acts[b],
            &block.out_norm,
            Some(embedding),
            false,
            shape,
        )?;
        device(gpu.latent_silu(&acts[a], &acts[a], dtype, shape.elements()))?;
        self.conv3d(&acts[b], &acts[a], &block.out_conv, shape)?;
        device(gpu.latent_add(&acts[out], &acts[input], &acts[b], dtype, shape.elements()))?;
        Ok(())
    }

    /// `TemporalConv.forward` (`:205-210`): `x + pwconv(dwconv(silu(norm(x))))`.
    #[allow(clippy::too_many_arguments)]
    fn temporal_block(
        &self,
        block: &TemporalBlock,
        input: usize,
        out: usize,
        a: usize,
        b: usize,
        shape: Shape,
    ) -> Result<(), LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let acts = &self.scratch.acts;
        let plane = shape.plane();

        normalise(
            gpu,
            self.arm,
            &self.scratch.norm,
            &acts[a],
            &acts[input],
            &block.norm,
            None,
            true,
            shape,
        )?;
        device(gpu.latent_depthwise_temporal(
            &acts[b],
            &acts[a],
            &block.depth_weight,
            &block.depth_bias,
            dtype,
            shape.channels,
            shape.frames,
            shape.area() as u32,
            UPSCALER_TEMPORAL_KERNEL as u32,
        ))?;
        device(gpu.latent_matmul_channels(
            &self.scratch.temp,
            &acts[b],
            &block.pointwise.weight,
            plane as u32,
            block.pointwise.input_channels,
            block.pointwise.output_channels,
            dtype,
        ))?;
        device(gpu.latent_epilogue(
            &acts[a],
            &self.scratch.temp,
            Some(&block.pointwise.bias),
            dtype,
            block.pointwise.output_channels,
            plane as u32,
            plane,
            0,
        ))?;
        device(gpu.latent_add(&acts[out], &acts[input], &acts[a], dtype, shape.elements()))?;
        Ok(())
    }

    /// The convolution itself: whichever of the two forms `H3_LTX_CONV`
    /// selects — `convolve_planar` for the planar one, the im2col matrix for
    /// the classic one.
    fn conv3d(
        &self,
        out: &Tensor,
        input: &Tensor,
        conv: &Conv,
        input_shape: Shape,
    ) -> Result<(), LatentError> {
        if conv_path() == ConvPath::Planar {
            return convolve_planar(
                &self.gpu,
                self.arm,
                &self.scratch.planar,
                &self.scratch.temp,
                out,
                input,
                conv,
                input_shape,
            );
        }
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let [kt, kh, kw] = conv.kernel;
        let [pt, ph, pw] = conv.padding;
        let frames = input_shape.frames;
        let out_height = input_shape.height + 2 * ph - kh + 1;
        let out_width = input_shape.width + 2 * pw - kw + 1;
        let out_plane = out_height as usize * out_width as usize;
        let kernel_columns = (conv.input_channels * kt * kh * kw) as usize;

        device(gpu.latent_pad3d(
            &self.scratch.padded,
            input,
            dtype,
            conv.input_channels,
            frames,
            input_shape.height,
            input_shape.width,
            pt,
            ph,
            pw,
        ))?;

        let padded = [
            frames + 2 * pt,
            input_shape.height + 2 * ph,
            input_shape.width + 2 * pw,
        ];
        let per_frame = out_plane * kernel_columns;
        let slab = slab_frames(
            self.scratch.matrix.elements() / per_frame,
            self.scratch.temp.elements() / (conv.output_channels as usize * out_plane),
            frames as usize,
        );

        let mut begin = 0u32;
        while begin < frames {
            let count = (frames - begin).min(slab);
            let rows = count as usize * out_plane;
            device(gpu.latent_im2col(
                &self.scratch.matrix,
                &self.scratch.padded,
                dtype,
                conv.input_channels,
                padded,
                conv.kernel,
                [count, out_height, out_width],
                begin,
            ))?;
            device(gpu.latent_matmul(
                &self.scratch.temp,
                &self.scratch.matrix,
                &conv.weight,
                rows as u32,
                kernel_columns as u32,
                conv.output_channels,
                dtype,
            ))?;
            device(gpu.latent_epilogue(
                out,
                &self.scratch.temp,
                Some(&conv.bias),
                dtype,
                conv.output_channels,
                rows as u32,
                frames as usize * out_plane,
                begin as usize * out_plane,
            ))?;
            begin += count;
        }
        Ok(())
    }
}

/// `config.json`'s own contract, which `from_pretrained` checks before it
/// loads a weight (`adapter.py:41-58`). Kept here so a mismatched checkpoint is
/// refused by name rather than producing a plausible latent.
struct AdapterConfig {
    width: u32,
    blocks: u32,
    expansion: u32,
    groups: u32,
}

fn read_config(directory: &Path) -> Result<AdapterConfig, LatentError> {
    let path = directory.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| LatentError::Io(format!("{}: {error}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| LatentError::Io(format!("{}: {error}", path.display())))?;

    let field = |name: &str| -> Result<u64, LatentError> {
        value
            .get(name)
            .and_then(|field| field.as_u64())
            .ok_or_else(|| LatentError::Shape(format!("config.json has no {name}")))
    };
    let text_field = |name: &str| -> Result<String, LatentError> {
        value
            .get(name)
            .and_then(|field| field.as_str())
            .map(str::to_string)
            .ok_or_else(|| LatentError::Shape(format!("config.json has no {name}")))
    };

    if text_field("format")? != "h3_ltx_adapter_safetensors_v1" {
        return Err(LatentError::Shape("unexpected adapter format".into()));
    }
    let geometry = value
        .get("geometry")
        .ok_or_else(|| LatentError::Shape("config.json has no geometry".into()))?;
    let frozen = [
        ("spatial_mode", "pixel_unshuffle"),
        ("temporal_mode", "linear_plus_nearest_pack"),
    ];
    for (key, expected) in frozen {
        if geometry.get(key).and_then(|field| field.as_str()) != Some(expected) {
            return Err(LatentError::Shape(format!(
                "adapter geometry {key} is not {expected}"
            )));
        }
    }
    if geometry
        .get("temporal_pack_slots")
        .and_then(|field| field.as_u64())
        != Some(TEMPORAL_PACK_SLOTS as u64)
    {
        return Err(LatentError::Shape(format!(
            "adapter packs {TEMPORAL_PACK_SLOTS} slots at most"
        )));
    }

    let model = value
        .get("model_config")
        .ok_or_else(|| LatentError::Shape("config.json has no model_config".into()))?;
    let model_field = |name: &str| -> Result<u64, LatentError> {
        model
            .get(name)
            .and_then(|field| field.as_u64())
            .ok_or_else(|| LatentError::Shape(format!("model_config has no {name}")))
    };
    if model.get("model_type").and_then(|field| field.as_str()) != Some("tiny") {
        return Err(LatentError::Shape(
            "adapter is not the promoted model".into(),
        ));
    }
    // `model.py:47-53` refuses a checkpoint that enables any training variant.
    for variant in [
        "dense_temporal_blocks",
        "coordinate_channels",
        "output_refiner_blocks",
        "source_detail_blocks",
    ] {
        if model_field(variant).unwrap_or(0) != 0 {
            return Err(LatentError::Shape(format!(
                "adapter enables the unsupported {variant}"
            )));
        }
    }

    let config = AdapterConfig {
        width: model_field("width")? as u32,
        blocks: model_field("num_blocks")? as u32,
        expansion: model_field("expansion")? as u32,
        groups: model_field("groups")? as u32,
    };
    if config.width != ADAPTER_WIDTH as u32
        || config.blocks != ADAPTER_BLOCKS as u32
        || config.expansion != ADAPTER_EXPANSION as u32
        || config.groups != ADAPTER_GROUPS as u32
    {
        return Err(LatentError::Shape(format!(
            "adapter is {}x{}|{}|{}, the port implements {}x{}|{}|{}",
            config.width,
            config.blocks,
            config.expansion,
            config.groups,
            ADAPTER_WIDTH,
            ADAPTER_BLOCKS,
            ADAPTER_EXPANSION,
            ADAPTER_GROUPS
        )));
    }
    if model_field("in_channels")? != ADAPTER_IN_CHANNELS as u64
        || model_field("out_channels")? != ADAPTER_OUT_CHANNELS as u64
    {
        return Err(LatentError::Shape(format!(
            "adapter channels are not {ADAPTER_IN_CHANNELS} -> {ADAPTER_OUT_CHANNELS}"
        )));
    }
    let _ = field("parameters")?;
    let _ = text_field("dtype")?;

    Ok(config)
}

/// `FactorizedResidualBlock` (`model.py:9-22`).
struct AdBlock {
    norm: Norm,
    spatial: Conv,
    temporal_weight: Tensor,
    temporal_bias: Tensor,
    in_proj: Pointwise,
    out_proj: Pointwise,
}

struct AdModel {
    skip: Pointwise,
    stem: Conv,
    blocks: Vec<AdBlock>,
    final_norm: Norm,
    head: Pointwise,
}

impl AdModel {
    fn load(store: &Store, gpu: &Gpu, arm: Arm) -> Result<AdModel, LatentError> {
        let width = ADAPTER_WIDTH as u32;
        let mut blocks = Vec::new();
        for index in 0..ADAPTER_BLOCKS {
            let prefix = format!("blocks.{index}");
            let temporal = format!("{prefix}.temporal.weight");
            let temporal_shape = store.shape(&temporal)?.to_vec();
            blocks.push(AdBlock {
                norm: load_norm(
                    store,
                    gpu,
                    arm,
                    &format!("{prefix}.norm"),
                    width,
                    ADAPTER_GROUPS as u32,
                    ADAPTER_EPSILON,
                )?,
                spatial: load_conv(
                    store,
                    gpu,
                    arm,
                    &format!("{prefix}.spatial"),
                    [1, 3, 3],
                    [0, 1, 1],
                )?,
                temporal_weight: store.load(gpu, arm, &temporal, &temporal_shape)?,
                temporal_bias: store.load(
                    gpu,
                    arm,
                    &format!("{prefix}.temporal.bias"),
                    store.shape(&format!("{prefix}.temporal.bias"))?,
                )?,
                in_proj: load_pointwise(store, gpu, arm, &format!("{prefix}.in_proj"))?,
                out_proj: load_pointwise(store, gpu, arm, &format!("{prefix}.out_proj"))?,
            });
        }

        let model = AdModel {
            skip: load_pointwise(store, gpu, arm, "skip")?,
            stem: load_conv(store, gpu, arm, "stem", [3, 3, 3], [1, 1, 1])?,
            blocks,
            final_norm: load_norm(
                store,
                gpu,
                arm,
                "final_norm",
                width,
                ADAPTER_GROUPS as u32,
                ADAPTER_EPSILON,
            )?,
            head: load_pointwise(store, gpu, arm, "head")?,
        };

        if model.skip.input_channels as usize != ADAPTER_IN_CHANNELS
            || model.head.output_channels as usize != ADAPTER_OUT_CHANNELS
            || model.stem.input_channels as usize != ADAPTER_IN_CHANNELS
        {
            return Err(LatentError::Shape(format!(
                "adapter is {} -> {} with a {} stem, expected {ADAPTER_IN_CHANNELS} -> {ADAPTER_OUT_CHANNELS}",
                model.skip.input_channels, model.head.output_channels, model.stem.input_channels
            )));
        }
        // `FactorizedResidualBlock` (`model.py:11-18`): `in_proj` writes
        // `2 * channels * expansion` and the gate/value halves meet in
        // `out_proj`, which reads `channels * expansion`.
        let hidden = width * ADAPTER_EXPANSION as u32;
        for block in &model.blocks {
            if block.in_proj.output_channels != 2 * hidden
                || block.out_proj.input_channels != hidden
            {
                return Err(LatentError::Shape(format!(
                    "adapter block projects {width} -> {} -> {}, expected {width} -> {} -> {hidden}",
                    block.in_proj.output_channels,
                    block.out_proj.input_channels,
                    2 * hidden
                )));
            }
        }

        Ok(model)
    }
}

/// The adapter's own scratch: the alignment (24 + 72 channels of it) and the
/// 384-channel input the blocks read.
struct AdScratch {
    aligned: Tensor,
    packed: Tensor,
    stem_input: Tensor,
    pd: Tensor,
    acts: [Tensor; 4],
    projected: Tensor,
    residual: Tensor,
    skipped: Tensor,
    matrix: Tensor,
    /// The planar path's A operand, as in `UpScratch`.
    planar: Tensor,
    temp: Tensor,
    norm: NormScratch,
    left: Tensor,
    right: Tensor,
    weight: Tensor,
    pack_index: Tensor,
}

impl AdScratch {
    fn new(
        gpu: &Gpu,
        arm: Arm,
        geometry: &Geometry,
        plan: &TemporalPlan,
    ) -> Result<AdScratch, LatentError> {
        let dtype = arm.dtype();
        let [_, _frames, height, width] = geometry.h3;
        let (height, width) = (height as u32, width as u32);
        let [_, ltx_frames, ltx_height, ltx_width] = geometry.ltx;
        let aligned_plane = (height as usize) * (width as usize);
        let plane = (ltx_height * ltx_width) as usize;
        let grid = ltx_frames as u32 * ltx_height as u32 * ltx_width as u32;

        let alloc = |elements: usize| device(gpu.alloc(dtype, elements));
        let width_channels = ADAPTER_WIDTH as u32;

        // The largest matmul the adapter runs: `in_proj`'s 752 -> 3008 over the
        // whole latent grid, which is what sizes the f32 accumulator.
        let temp = (grid as usize) * (2 * ADAPTER_WIDTH * ADAPTER_EXPANSION);
        // The two convolutions the adapter runs differ in kernel and input
        // channel count: the stem is a 3x3x3 over 384 channels, the blocks a
        // 1x3x3 over 752.
        let planar = [
            planar_elements(
                ADAPTER_WIDTH as u64,
                ltx_frames as u64,
                ltx_height as u64,
                ltx_width as u64,
                [1, 3, 3],
            ),
            planar_elements(
                ADAPTER_IN_CHANNELS as u64,
                ltx_frames as u64,
                ltx_height as u64,
                ltx_width as u64,
                [3, 3, 3],
            ),
        ]
        .into_iter()
        .max()
        .unwrap_or(0) as usize;
        let matrix = match conv_path() {
            ConvPath::Planar => 1,
            ConvPath::Classic => {
                SLAB_FRAMES * plane * (ADAPTER_WIDTH * 3 * 3).max(ADAPTER_IN_CHANNELS * 3 * 3 * 3)
            }
        };
        let max_plane = ltx_frames as usize * plane;
        let chunks = max_plane.div_ceil(8192);
        let partial = ADAPTER_GROUPS * (ADAPTER_WIDTH / ADAPTER_GROUPS) * chunks * 2;

        Ok(AdScratch {
            aligned: alloc(ADAPTER_IN_CHANNELS * ltx_frames as usize * aligned_plane)?,
            packed: alloc(3 * H3_LATENT_CHANNELS * ltx_frames as usize * aligned_plane)?,
            stem_input: alloc(ADAPTER_IN_CHANNELS * ltx_frames as usize * plane)?,
            pd: alloc(
                [
                    padded_elements(
                        ADAPTER_WIDTH as u64,
                        ltx_frames as u64,
                        ltx_height as u64,
                        ltx_width as u64,
                    ),
                    padded_elements(
                        ADAPTER_IN_CHANNELS as u64,
                        ltx_frames as u64,
                        ltx_height as u64,
                        ltx_width as u64,
                    ),
                ]
                .into_iter()
                .max()
                .unwrap_or(0) as usize,
            )?,
            acts: [
                alloc(width_channels as usize * ltx_frames as usize * plane)?,
                alloc(width_channels as usize * ltx_frames as usize * plane)?,
                alloc(width_channels as usize * ltx_frames as usize * plane)?,
                alloc(width_channels as usize * ltx_frames as usize * plane)?,
            ],
            projected: alloc(
                (2 * ADAPTER_WIDTH * ADAPTER_EXPANSION) * ltx_frames as usize * plane,
            )?,
            residual: alloc(ADAPTER_OUT_CHANNELS * ltx_frames as usize * plane)?,
            skipped: alloc(ADAPTER_OUT_CHANNELS * ltx_frames as usize * plane)?,
            matrix: alloc(matrix)?,
            planar: alloc(planar)?,
            temp: device(gpu.alloc(Dtype::F32, temp))?,
            norm: NormScratch {
                partial: device(gpu.alloc(Dtype::F32, partial))?,
                mean: device(gpu.alloc(Dtype::F32, ADAPTER_GROUPS))?,
                rstd: device(gpu.alloc(Dtype::F32, ADAPTER_GROUPS))?,
            },
            left: upload_u32(gpu, &plan.linear_left)?,
            right: upload_u32(gpu, &plan.linear_right)?,
            weight: upload_f32(gpu, &plan.linear_weight)?,
            pack_index: upload_u32(gpu, &plan.pack)?,
        })
    }
}

/// The frozen H3-to-LTX Conv3D adapter (`stage2_ops/h3_ltx_adapter/`).
pub struct Adapter {
    gpu: Gpu,
    arm: Arm,
    geometry: Geometry,
    plan: TemporalPlan,
    model: AdModel,
    scratch: AdScratch,
}

impl Adapter {
    pub fn load(directory: &Path, arm: Arm, geometry: &Geometry) -> Result<Adapter, LatentError> {
        let _config = read_config(directory)?;
        let gpu = device(Gpu::create())?;
        let store = Store::open(&directory.join("model.safetensors"), "latent adapter")?;
        let model = AdModel::load(&store, &gpu, arm)?;
        let plan = temporal_plan(geometry.pixel_frames)?;
        let scratch = AdScratch::new(&gpu, arm, geometry, &plan)?;
        device(gpu.sync())?;

        Ok(Adapter {
            gpu,
            arm,
            geometry: geometry.clone(),
            plan,
            model,
            scratch,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn arm(&self) -> Arm {
        self.arm
    }

    /// `align_h3_to_ltx` (`geometry.py:81-112`): the upscaled H3 latent in at
    /// `(24, 37, 48, 84)`, the adapter's `(384, 17, 24, 42)` grid out. Its own
    /// method because the gate checks it separately from the model.
    pub fn align(&self, upscaled: &Tensor) -> Result<&Tensor, LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let [_, frames, height, width] = self.geometry.h3;
        let (height, width) = (height as u32, width as u32);
        let [_, ltx_frames, _, _] = self.geometry.ltx;
        // The aligned grid's own channel extent: the LTX latent is half this in
        // each spatial axis, and using its plane here would drop the packed
        // half a quarter of the way into the tensor.
        let aligned_plane = height as usize * width as usize;
        let grid = ltx_frames as usize * aligned_plane;

        if upscaled.elements() < H3_LATENT_CHANNELS * frames as usize * aligned_plane {
            return Err(LatentError::Shape(format!(
                "upscaled latent has {} elements, the adapter reads {}",
                upscaled.elements(),
                H3_LATENT_CHANNELS * frames as usize * aligned_plane
            )));
        }

        // `linear = temporal_resample(...)`, then the nearest pack beside it.
        device(gpu.latent_time_lerp(
            &self.scratch.aligned,
            upscaled,
            &self.scratch.left,
            &self.scratch.right,
            &self.scratch.weight,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames as u32,
            self.plan.target_frames as u32,
            aligned_plane as u32,
        ))?;
        device(gpu.latent_time_gather(
            &self.scratch.packed,
            upscaled,
            &self.scratch.pack_index,
            dtype,
            H3_LATENT_CHANNELS as u32,
            frames as u32,
            self.plan.target_frames as u32,
            aligned_plane as u32,
            self.plan.slots as u32,
        ))?;
        device(gpu.latent_copy_offset(
            &self.scratch.aligned,
            &self.scratch.packed,
            dtype,
            H3_LATENT_CHANNELS * grid,
            0,
            3 * H3_LATENT_CHANNELS * grid,
        ))?;

        // The 2x pixel shuffle is what makes the adapter's 384 channels.
        device(gpu.latent_unshuffle2(
            &self.scratch.stem_input,
            &self.scratch.aligned,
            dtype,
            (ADAPTER_IN_CHANNELS / 4) as u32,
            ltx_frames as u32,
            height,
            width,
        ))?;
        Ok(&self.scratch.stem_input)
    }

    /// The model itself (`model.py:39-42`), over an already aligned grid.
    pub fn model(&self, aligned: &Tensor) -> Result<Tensor, LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let [_, ltx_frames, ltx_height, ltx_width] = self.geometry.ltx;
        let plane = ltx_height as usize * ltx_width as usize;
        let grid = ltx_frames as usize * plane;

        let shape = Shape {
            channels: ADAPTER_WIDTH as u32,
            frames: ltx_frames as u32,
            height: ltx_height as u32,
            width: ltx_width as u32,
        };
        self.conv3d(
            &self.scratch.acts[0],
            aligned,
            &self.model.stem,
            Shape {
                channels: ADAPTER_IN_CHANNELS as u32,
                frames: ltx_frames as u32,
                height: ltx_height as u32,
                width: ltx_width as u32,
            },
        )?;

        let mut current = 0usize;
        for block in &self.model.blocks {
            let a = (current + 1) % 4;
            let b = (current + 2) % 4;
            let out = (current + 3) % 4;
            self.ad_block(block, current, out, a, b, shape)?;
            current = out;
        }

        // `head(final_activation(final_norm(residual)))` and the skip path.
        let next = (current + 1) % 4;
        normalise(
            gpu,
            self.arm,
            &self.scratch.norm,
            &self.scratch.acts[next],
            &self.scratch.acts[current],
            &self.model.final_norm,
            None,
            true,
            shape,
        )?;
        self.pointwise(
            &self.scratch.residual,
            &self.scratch.acts[next],
            &self.model.head,
            grid,
        )?;
        self.pointwise(&self.scratch.skipped, aligned, &self.model.skip, grid)?;

        let result = device(gpu.alloc(dtype, ADAPTER_OUT_CHANNELS * grid))?;
        device(gpu.latent_add(
            &result,
            &self.scratch.skipped,
            &self.scratch.residual,
            dtype,
            ADAPTER_OUT_CHANNELS * grid,
        ))?;
        device(gpu.sync())?;
        Ok(result)
    }

    /// `align_h3_to_ltx` then the model (`adapter.py:88-99`): the upscaled H3
    /// latent in, the normalized LTX latent out.
    pub fn convert(&self, upscaled: &Tensor) -> Result<Tensor, LatentError> {
        let aligned = self.align(upscaled)?;
        self.model(aligned)
    }

    /// `FactorizedResidualBlock.forward` (`model.py:20-22`).
    #[allow(clippy::too_many_arguments)]
    fn ad_block(
        &self,
        block: &AdBlock,
        input: usize,
        out: usize,
        a: usize,
        b: usize,
        shape: Shape,
    ) -> Result<(), LatentError> {
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let acts = &self.scratch.acts;
        let plane = shape.plane();

        normalise(
            gpu,
            self.arm,
            &self.scratch.norm,
            &acts[a],
            &acts[input],
            &block.norm,
            None,
            true,
            shape,
        )?;
        self.conv3d(&acts[b], &acts[a], &block.spatial, shape)?;
        device(gpu.latent_depthwise_temporal(
            &acts[a],
            &acts[b],
            &block.temporal_weight,
            &block.temporal_bias,
            dtype,
            shape.channels,
            shape.frames,
            shape.area() as u32,
            3,
        ))?;
        device(gpu.latent_silu(&acts[a], &acts[a], dtype, shape.elements()))?;

        self.pointwise(&self.scratch.projected, &acts[a], &block.in_proj, plane)?;
        device(gpu.latent_gate_mul(
            &self.scratch.projected,
            &self.scratch.projected,
            dtype,
            block.in_proj.output_channels / 2,
            plane,
        ))?;
        self.pointwise(&acts[a], &self.scratch.projected, &block.out_proj, plane)?;
        device(gpu.latent_add(&acts[out], &acts[input], &acts[a], dtype, shape.elements()))?;
        Ok(())
    }

    /// A 1x1 convolution: `matmul_channels` into the f32 accumulator, then the
    /// epilogue, which is where the bias and the single rounding happen.
    fn pointwise(
        &self,
        out: &Tensor,
        input: &Tensor,
        weights: &Pointwise,
        plane: usize,
    ) -> Result<(), LatentError> {
        let gpu = &self.gpu;
        device(gpu.latent_matmul_channels(
            &self.scratch.temp,
            input,
            &weights.weight,
            plane as u32,
            weights.input_channels,
            weights.output_channels,
            self.arm.dtype(),
        ))?;
        device(gpu.latent_epilogue(
            out,
            &self.scratch.temp,
            Some(&weights.bias),
            self.arm.dtype(),
            weights.output_channels,
            plane as u32,
            plane,
            0,
        ))?;
        Ok(())
    }

    /// The same two convolution forms the upscaler selects between.
    fn conv3d(
        &self,
        out: &Tensor,
        input: &Tensor,
        conv: &Conv,
        input_shape: Shape,
    ) -> Result<(), LatentError> {
        if conv_path() == ConvPath::Planar {
            return convolve_planar(
                &self.gpu,
                self.arm,
                &self.scratch.planar,
                &self.scratch.temp,
                out,
                input,
                conv,
                input_shape,
            );
        }
        let gpu = &self.gpu;
        let dtype = self.arm.dtype();
        let [kt, kh, kw] = conv.kernel;
        let [pt, ph, pw] = conv.padding;
        let frames = input_shape.frames;
        let out_height = input_shape.height + 2 * ph - kh + 1;
        let out_width = input_shape.width + 2 * pw - kw + 1;
        let out_plane = out_height as usize * out_width as usize;
        let kernel_columns = (conv.input_channels * kt * kh * kw) as usize;

        device(gpu.latent_pad3d(
            &self.scratch.pd,
            input,
            dtype,
            conv.input_channels,
            frames,
            input_shape.height,
            input_shape.width,
            pt,
            ph,
            pw,
        ))?;

        let padded = [
            frames + 2 * pt,
            input_shape.height + 2 * ph,
            input_shape.width + 2 * pw,
        ];
        let per_frame = out_plane * kernel_columns;
        let slab = slab_frames(
            self.scratch.matrix.elements() / per_frame,
            self.scratch.temp.elements() / (conv.output_channels as usize * out_plane),
            frames as usize,
        );

        let mut begin = 0u32;
        while begin < frames {
            let count = (frames - begin).min(slab);
            let rows = count as usize * out_plane;
            device(gpu.latent_im2col(
                &self.scratch.matrix,
                &self.scratch.pd,
                dtype,
                conv.input_channels,
                padded,
                conv.kernel,
                [count, out_height, out_width],
                begin,
            ))?;
            device(gpu.latent_matmul(
                &self.scratch.temp,
                &self.scratch.matrix,
                &conv.weight,
                rows as u32,
                kernel_columns as u32,
                conv.output_channels,
                dtype,
            ))?;
            device(gpu.latent_epilogue(
                out,
                &self.scratch.temp,
                Some(&conv.bias),
                dtype,
                conv.output_channels,
                rows as u32,
                frames as usize * out_plane,
                begin as usize * out_plane,
            ))?;
            begin += count;
        }
        Ok(())
    }
}
