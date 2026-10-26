//! The Qwen3-VL vision tower: tensor layout and host-side input preparation.
//!
//! Source of truth `/data/h3.c` at `eae5d54`:
//!
//! - `h3_vision_encoder.h:8-31`: the output, BF16 `[tokens, 5120]` plus three
//!   same-shaped deepstack additions; input F32 `[T,3,H,W]` in `[0,1]`, `T` in
//!   `{1,2}`, both spatial edges multiples of 32.
//! - `h3_vision_encoder.c:10-26`: the geometry constants below. The reference
//!   never reads them from a config; a vision `config.json` block is not part
//!   of its contract, so this module does not take one either.
//! - `h3_vision_encoder.c:250-287` one block's twelve tensors, `:353-368` the
//!   merger's six, `:468-470` the position table, `:507-511` the patch weight.
//!
//! Three host-side preparations live here, each transcribed from its
//! reference loop because the row order is part of the numerical contract:
//!
//! - `patch_rows` (`:135-162`): merge-block-major rows of the raw patch,
//!   `pixel * 2 - 1` rounded to BF16.
//! - `position_rows` (`:163-214`): bilinear interpolation of the 48x48 table
//!   with the reference's three BF16 roundings (coefficient, each term, sum).
//! - `rope_rows` (`:215-240`): two axes of 18 pairs in patch units, BF16.
//!
//! The merge-block-major order is what lets the merger view the hidden state
//! `[grid_h * grid_w, 1152]` as `[merged_rows, 4608]`: one merge block owns
//! four consecutive patch rows, in the inner-loop order the patch prep wrote
//! them (`h3_vision_encoder.c:355`).

use std::fmt;

/// `VISION_HIDDEN` (`h3_vision_encoder.c:11`).
pub const HIDDEN: u64 = 1152;

/// `VISION_INTERMEDIATE` (`h3_vision_encoder.c:12`).
pub const INTERMEDIATE: u64 = 4304;

/// `VISION_LAYERS` (`h3_vision_encoder.c:13`).
pub const LAYERS: u32 = 27;

/// `VISION_HEADS` (`h3_vision_encoder.c:14`).
pub const HEADS: u64 = 16;

/// `VISION_HEAD_DIM` (`h3_vision_encoder.c:15`).
pub const HEAD_DIM: u64 = 72;

/// `VISION_ROPE_HALF` (`h3_vision_encoder.c:16`).
pub const ROPE_HALF: u64 = 36;

/// `PATCH` (`h3_vision_encoder.c:17`).
pub const PATCH: u64 = 16;

/// `TEMPORAL_PATCH` (`h3_vision_encoder.c:18`).
pub const TEMPORAL_PATCH: u64 = 2;

/// `MERGE` (`h3_vision_encoder.c:19`).
pub const MERGE: u64 = 2;

/// Channels of the input image (`h3_vision_encoder.c:20`).
pub const CHANNELS: u64 = 3;

/// `PATCH_DIM` (`h3_vision_encoder.c:20`).
pub const PATCH_DIM: u64 = CHANNELS * TEMPORAL_PATCH * PATCH * PATCH;

/// `MERGE_DIM` (`h3_vision_encoder.c:21`).
pub const MERGE_DIM: u64 = HIDDEN * MERGE * MERGE;

/// `POSITION_SIDE` (`h3_vision_encoder.c:22`).
pub const POSITION_SIDE: u64 = 48;

/// `POSITION_COUNT` (`h3_vision_encoder.c:23`).
pub const POSITION_COUNT: u64 = POSITION_SIDE * POSITION_SIDE;

/// `OUTPUT_WIDTH` (`h3_vision_encoder.c:24`, `H3_VISION_OUTPUT_WIDTH`).
pub const OUTPUT_WIDTH: u64 = 5120;

/// `H3_VISION_DEEPSTACKS` (`h3_vision_encoder.h:9`).
pub const DEEPSTACKS: usize = 3;

/// The epsilon every layer norm passes (`h3_vision_encoder.c:300`, `:378`).
pub const LAYER_NORM_EPSILON: f32 = 1e-6;

/// The layers whose hidden state feeds a deepstack merger, 0-based
/// (`h3_vision_encoder.c:566`).
pub const DEEPSTACK_LAYERS: [u32; DEEPSTACKS] = [8, 16, 24];

/// The smallest accepted spatial edge (`h3_vision_encoder.c:452-455`).
pub const MIN_EDGE: u64 = 32;

/// The patch width of the position table: a merge of four patch columns.
pub const MERGE_ROWS: u64 = MERGE * MERGE;

const BLOCK_PREFIX: &str = "model.visual.blocks.";
const MERGER_PREFIX: &str = "model.visual.merger.";
const DEEPSTACK_PREFIX: &str = "model.visual.deepstack_merger_list.";

/// Which tensor of a vision block (`h3_vision_encoder.c:250-287`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Norm1Weight,
    Norm1Bias,
    QkvWeight,
    QkvBias,
    ProjWeight,
    ProjBias,
    Norm2Weight,
    Norm2Bias,
    Fc1Weight,
    Fc1Bias,
    Fc2Weight,
    Fc2Bias,
}

/// Which tensor of a merger (`h3_vision_encoder.c:353-368`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MergerPart {
    NormWeight,
    NormBias,
    Fc1Weight,
    Fc1Bias,
    Fc2Weight,
    Fc2Bias,
}

/// One place in the vision tower.
///
/// `Merger { deepstack: None }` is `model.visual.merger.`; `Some(index)` is
/// `model.visual.deepstack_merger_list.{index}.`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    PositionTable,
    PatchWeight,
    PatchBias,
    Block {
        layer: u32,
        part: Part,
    },
    Merger {
        deepstack: Option<u32>,
        part: MergerPart,
    },
}

/// One expected tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub role: Role,
    pub shape: Vec<u64>,
}

/// A borrowed tensor descriptor, so a synthetic set can be checked without
/// payloads. The vision tower lives in the text partition, so this is the
/// same descriptor `minimax_h3::text` checks its own path against.
pub use crate::minimax_h3::text::TensorRef;

/// The checked vision path: 351 tensors and their exact shapes.
///
/// The plan's first draft said 350 with eleven names per block; the reference
/// reads twelve (two norms, qkv with its bias, the projection with its bias,
/// two MLP matrices with their biases), so 3 + 27 * 12 + 6 + 3 * 6 = 351.
#[derive(Debug)]
pub struct VisionLayout {
    entries: Vec<Entry>,
}

impl VisionLayout {
    /// Check the expected names against an arbitrary tensor set
    /// (`h3_weight_load_bf16`, `h3_weights.c:144-197`: dtype and every
    /// dimension before a byte is read).
    pub fn build(tensors: &[TensorRef<'_>]) -> Result<VisionLayout, VisionError> {
        use std::collections::BTreeMap;

        let mut actual: BTreeMap<&str, &TensorRef<'_>> = BTreeMap::new();

        for tensor in tensors {
            if actual.insert(tensor.name, tensor).is_some() {
                return Err(VisionError::Duplicate {
                    name: tensor.name.to_string(),
                });
            }
        }

        let specs = specs();
        let mut entries = Vec::with_capacity(specs.len());

        for spec in specs {
            let found = actual
                .get(spec.name.as_str())
                .ok_or_else(|| VisionError::Missing {
                    name: spec.name.clone(),
                })?;

            if found.dtype != crate::minimax_h3::safetensors::Dtype::Bf16 {
                return Err(VisionError::Dtype {
                    name: spec.name.clone(),
                    got: found.dtype.name(),
                });
            }

            if found.shape != spec.shape.as_slice() {
                return Err(VisionError::Shape {
                    name: spec.name.clone(),
                    want: spec.shape.clone(),
                    got: found.shape.to_vec(),
                });
            }

            entries.push(spec);
        }

        Ok(VisionLayout { entries })
    }

    /// The 351 vision tensors, position table first.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

/// Every vision tensor, position table first, then patch embed, blocks,
/// final merger and the three deepstack mergers.
pub fn specs() -> Vec<Entry> {
    let mut specs = Vec::with_capacity(3 + LAYERS as usize * 12 + MERGER_COUNT * 6);

    specs.push(Entry {
        name: "model.visual.pos_embed.weight".to_string(),
        role: Role::PositionTable,
        shape: vec![POSITION_COUNT, HIDDEN],
    });
    specs.push(Entry {
        name: "model.visual.patch_embed.proj.weight".to_string(),
        role: Role::PatchWeight,
        shape: vec![HIDDEN, CHANNELS, TEMPORAL_PATCH, PATCH, PATCH],
    });
    specs.push(Entry {
        name: "model.visual.patch_embed.proj.bias".to_string(),
        role: Role::PatchBias,
        shape: vec![HIDDEN],
    });

    for layer in 0..LAYERS {
        for (part, suffix, shape) in block_parts() {
            specs.push(Entry {
                name: format!("{BLOCK_PREFIX}{layer}.{suffix}"),
                role: Role::Block { layer, part },
                shape,
            });
        }
    }

    for (deepstack, part, suffix, shape) in merger_parts() {
        let prefix = match deepstack {
            Some(index) => format!("{DEEPSTACK_PREFIX}{index}."),
            None => MERGER_PREFIX.to_string(),
        };

        specs.push(Entry {
            name: format!("{prefix}{suffix}"),
            role: Role::Merger { deepstack, part },
            shape,
        });
    }

    specs
}

/// The final merger plus one per deepstack.
const MERGER_COUNT: usize = 1 + DEEPSTACKS;

/// The twelve tensors of one block and their shapes (`:250-287`).
fn block_parts() -> Vec<(Part, &'static str, Vec<u64>)> {
    vec![
        (Part::Norm1Weight, "norm1.weight", vec![HIDDEN]),
        (Part::Norm1Bias, "norm1.bias", vec![HIDDEN]),
        (Part::QkvWeight, "attn.qkv.weight", vec![HIDDEN * 3, HIDDEN]),
        (Part::QkvBias, "attn.qkv.bias", vec![HIDDEN * 3]),
        (Part::ProjWeight, "attn.proj.weight", vec![HIDDEN, HIDDEN]),
        (Part::ProjBias, "attn.proj.bias", vec![HIDDEN]),
        (Part::Norm2Weight, "norm2.weight", vec![HIDDEN]),
        (Part::Norm2Bias, "norm2.bias", vec![HIDDEN]),
        (
            Part::Fc1Weight,
            "mlp.linear_fc1.weight",
            vec![INTERMEDIATE, HIDDEN],
        ),
        (Part::Fc1Bias, "mlp.linear_fc1.bias", vec![INTERMEDIATE]),
        (
            Part::Fc2Weight,
            "mlp.linear_fc2.weight",
            vec![HIDDEN, INTERMEDIATE],
        ),
        (Part::Fc2Bias, "mlp.linear_fc2.bias", vec![HIDDEN]),
    ]
}

/// The six tensors of every merger and their shapes (`:353-368`).
///
/// The final merger normalizes one hidden row at a time (width 1152); a
/// deepstack merger normalizes one merged row of four patch rows (width
/// 4608). The projection matrices are the same width either way, which is why
/// only the norm differs (`:357-359`).
fn merger_parts() -> Vec<(Option<u32>, MergerPart, &'static str, Vec<u64>)> {
    let mut parts = Vec::with_capacity(MERGER_COUNT * 6);
    let mut entries = vec![(None, HIDDEN)];
    entries.extend((0..DEEPSTACKS as u32).map(|index| (Some(index), MERGE_DIM)));

    for (deepstack, norm_width) in entries {
        parts.push((
            deepstack,
            MergerPart::NormWeight,
            "norm.weight",
            vec![norm_width],
        ));
        parts.push((
            deepstack,
            MergerPart::NormBias,
            "norm.bias",
            vec![norm_width],
        ));
        parts.push((
            deepstack,
            MergerPart::Fc1Weight,
            "linear_fc1.weight",
            vec![MERGE_DIM, MERGE_DIM],
        ));
        parts.push((
            deepstack,
            MergerPart::Fc1Bias,
            "linear_fc1.bias",
            vec![MERGE_DIM],
        ));
        parts.push((
            deepstack,
            MergerPart::Fc2Weight,
            "linear_fc2.weight",
            vec![OUTPUT_WIDTH, MERGE_DIM],
        ));
        parts.push((
            deepstack,
            MergerPart::Fc2Bias,
            "linear_fc2.bias",
            vec![OUTPUT_WIDTH],
        ));
    }

    parts
}

/// `f32_to_bf16` (`h3_vision_encoder.c:44-49`): round to nearest even by
/// adding half an ulp plus the retained bit's parity.
pub fn f32_to_bf16(value: f32) -> u16 {
    let mut bits = value.to_bits();

    bits = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (bits >> 16) as u16
}

/// `bf16_to_f32` (`h3_vision_encoder.c:51-56`).
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// The grid the patch loop and both preparations agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub height: usize,
    pub width: usize,
}

impl Grid {
    /// A spatial size the reference accepts (`:452-455`).
    pub fn new(frames: usize, height: usize, width: usize) -> Result<Grid, VisionError> {
        if frames != 1 && frames != 2 {
            return Err(VisionError::Frames(frames));
        }

        let edge = MIN_EDGE as usize;

        if height < edge
            || width < edge
            || !height.is_multiple_of(edge)
            || !width.is_multiple_of(edge)
        {
            return Err(VisionError::Edges { height, width });
        }

        Ok(Grid { height, width })
    }

    /// Patch rows along each axis.
    pub fn rows(&self) -> (usize, usize) {
        (self.height / PATCH as usize, self.width / PATCH as usize)
    }

    /// The patch sequence length, `grid_h * grid_w`.
    pub fn tokens(&self) -> usize {
        let (grid_h, grid_w) = self.rows();

        grid_h * grid_w
    }

    /// The merged sequence length, one row per 2x2 patch block.
    pub fn merged(&self) -> usize {
        self.tokens() / MERGE_ROWS as usize
    }
}

/// `prepare_patch_rows` (`h3_vision_encoder.c:135-162`).
///
/// Row order: `block_h, block_w, inner_h, inner_w`, then the 1536 columns in
/// `channel, temporal, patch_h, patch_w` order. A single frame fills both
/// temporal slots from frame 0 (`:151-152`).
pub fn patch_rows(pixels: &[f32], frames: usize, grid: Grid) -> Result<Vec<u16>, VisionError> {
    if frames != 1 && frames != 2 {
        return Err(VisionError::Frames(frames));
    }

    let height = grid.height;
    let width = grid.width;
    let expected = frames * CHANNELS as usize * height * width;

    if pixels.len() != expected {
        return Err(VisionError::Pixels {
            got: pixels.len(),
            want: expected,
        });
    }

    let (grid_h, grid_w) = grid.rows();
    let mut rows = vec![0u16; grid_h * grid_w * PATCH_DIM as usize];
    let mut output = 0;

    for block_h in 0..grid_h / MERGE as usize {
        for block_w in 0..grid_w / MERGE as usize {
            for inner_h in 0..MERGE as usize {
                for inner_w in 0..MERGE as usize {
                    for channel in 0..CHANNELS as usize {
                        for temporal in 0..TEMPORAL_PATCH as usize {
                            let frame = if frames == 1 { 0 } else { temporal };
                            for patch_h in 0..PATCH as usize {
                                let row =
                                    (block_h * MERGE as usize + inner_h) * PATCH as usize + patch_h;
                                for patch_w in 0..PATCH as usize {
                                    let x = (block_w * MERGE as usize + inner_w) * PATCH as usize
                                        + patch_w;
                                    let input = ((frame * CHANNELS as usize + channel) * height
                                        + row)
                                        * width
                                        + x;
                                    rows[output] = f32_to_bf16(pixels[input] * 2.0 - 1.0);
                                    output += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    debug_assert_eq!(output, rows.len());

    Ok(rows)
}

/// `prepare_position_rows` (`h3_vision_encoder.c:163-214`).
///
/// Bilinear from the 48x48 table, with the reference's three roundings: the
/// coefficient, each product, and the sum. Dropping any of them changes the
/// BF16 rows.
pub fn position_rows(table: &[u16], grid: Grid) -> Result<Vec<u16>, VisionError> {
    let hidden = HIDDEN as usize;
    let side = POSITION_SIDE as usize;
    let expected = POSITION_COUNT as usize * hidden;

    if table.len() != expected {
        return Err(VisionError::Table {
            got: table.len(),
            want: expected,
        });
    }

    let (grid_h, grid_w) = grid.rows();
    let mut rows = vec![0u16; grid_h * grid_w * hidden];
    let mut output_row = 0;

    for block_h in 0..grid_h / MERGE as usize {
        for block_w in 0..grid_w / MERGE as usize {
            for inner_h in 0..MERGE as usize {
                for inner_w in 0..MERGE as usize {
                    let row = block_h * MERGE as usize + inner_h;
                    let column = block_w * MERGE as usize + inner_w;
                    let hv = row as f64 * (side - 1) as f64 / (grid_h - 1) as f64;
                    let wv = column as f64 * (side - 1) as f64 / (grid_w - 1) as f64;
                    let h0 = hv as usize;
                    let w0 = wv as usize;
                    let h1 = if h0 + 1 < side { h0 + 1 } else { h0 };
                    let w1 = if w0 + 1 < side { w0 + 1 } else { w0 };
                    let dh = (hv - h0 as f64) as f32;
                    let dw = (wv - w0 as f64) as f32;
                    let coefficients = [
                        (1.0 - dh) * (1.0 - dw),
                        (1.0 - dh) * dw,
                        dh * (1.0 - dw),
                        dh * dw,
                    ];
                    let indices = [
                        h0 * side + w0,
                        h0 * side + w1,
                        h1 * side + w0,
                        h1 * side + w1,
                    ];

                    for channel in 0..hidden {
                        let mut sum = 0.0f32;

                        for corner in 0..4 {
                            let coefficient = bf16_to_f32(f32_to_bf16(coefficients[corner]));
                            let term = bf16_to_f32(table[indices[corner] * hidden + channel])
                                * coefficient;
                            sum += bf16_to_f32(f32_to_bf16(term));
                        }

                        rows[output_row * hidden + channel] = f32_to_bf16(sum);
                    }

                    output_row += 1;
                }
            }
        }
    }

    debug_assert_eq!(output_row * hidden, rows.len());

    Ok(rows)
}

/// The two BF16 tables `prepare_rope` writes (`:215-240`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionRope {
    rows: usize,
    cosines: Vec<u16>,
    sines: Vec<u16>,
}

impl VisionRope {
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `[rows, ROPE_HALF]`: axis 0 in 0..18, axis 1 in 18..36.
    pub fn cosines(&self) -> &[u16] {
        &self.cosines
    }

    pub fn sines(&self) -> &[u16] {
        &self.sines
    }
}

/// `prepare_rope` (`h3_vision_encoder.c:215-240`): two axes of 18 pairs,
/// `10000^(-2i/36)`, positions in patch units, cosines and sines rounded to
/// BF16 on the host.
pub fn rope_rows(grid: Grid) -> VisionRope {
    let (grid_h, grid_w) = grid.rows();
    let half = ROPE_HALF as usize;
    let rows = grid_h * grid_w;
    let mut cosines = vec![0u16; rows * half];
    let mut sines = vec![0u16; rows * half];
    let mut row = 0;

    for block_h in 0..grid_h / MERGE as usize {
        for block_w in 0..grid_w / MERGE as usize {
            for inner_h in 0..MERGE as usize {
                for inner_w in 0..MERGE as usize {
                    let positions = [
                        (block_h * MERGE as usize + inner_h) as f32,
                        (block_w * MERGE as usize + inner_w) as f32,
                    ];

                    for (axis, position) in positions.iter().enumerate() {
                        for index in 0..half / 2 {
                            let inverse = 10_000.0f32.powf(-((index * 2) as f32) / half as f32);
                            let angle = position * inverse;
                            let offset = row * half + axis * (half / 2) + index;

                            cosines[offset] = f32_to_bf16(angle.cos());
                            sines[offset] = f32_to_bf16(angle.sin());
                        }
                    }

                    row += 1;
                }
            }
        }
    }

    VisionRope {
        rows,
        cosines,
        sines,
    }
}

/// Why the vision path could not be built or prepared.
#[derive(Debug)]
pub enum VisionError {
    /// The tensor set holds a name twice.
    Duplicate { name: String },
    /// A tensor the vision tower reads is absent.
    Missing { name: String },
    /// A vision tensor is not BF16.
    Dtype { name: String, got: &'static str },
    /// A vision tensor has an unexpected shape.
    Shape {
        name: String,
        want: Vec<u64>,
        got: Vec<u64>,
    },
    /// A frame count the reference rejects.
    Frames(usize),
    /// Spatial edges that are not multiples of 32 of at least 32.
    Edges { height: usize, width: usize },
    /// A pixel buffer whose length disagrees with the geometry.
    Pixels { got: usize, want: usize },
    /// A position table whose length disagrees with 48x48x1152.
    Table { got: usize, want: usize },
}

impl VisionError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            VisionError::Duplicate { name } => format!("duplicate vision tensor {name}"),
            VisionError::Missing { name } => format!("missing vision tensor {name}"),
            VisionError::Dtype { name, got } => format!("vision tensor {name} is {got}"),
            VisionError::Shape { name, want, got } => {
                format!("vision tensor {name} is {got:?}, expected {want:?}")
            }
            VisionError::Frames(frames) => {
                format!("vision tower takes 1 or 2 frames, not {frames}")
            }
            VisionError::Edges { height, width } => {
                format!("vision tower needs edges that are multiples of 32, not {height}x{width}")
            }
            VisionError::Pixels { got, want } => {
                format!("pixel buffer holds {got} values, expected {want}")
            }
            VisionError::Table { got, want } => {
                format!("position table holds {got} values, expected {want}")
            }
        }
    }
}

impl fmt::Display for VisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for VisionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::safetensors::Dtype;

    #[test]
    fn the_table_is_the_three_hundred_fifty_one_reference_reads() {
        let specs = specs();

        assert_eq!(specs.len(), 351);
        assert_eq!(specs.len(), 3 + LAYERS as usize * 12 + MERGER_COUNT * 6);

        let names: std::collections::BTreeSet<&str> =
            specs.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names.len(), specs.len(), "a name appears twice");

        // Spot-check the two shapes the reference reads differently from a
        // plain matrix: the 5-D patch weight and the block qkv.
        let patch = specs
            .iter()
            .find(|entry| entry.role == Role::PatchWeight)
            .expect("patch weight");
        assert_eq!(patch.shape, vec![1152, 3, 2, 16, 16]);

        let qkv = specs
            .iter()
            .find(|entry| {
                entry.role
                    == Role::Block {
                        layer: 0,
                        part: Part::QkvWeight,
                    }
            })
            .expect("layer 0 qkv");
        assert_eq!(qkv.name, "model.visual.blocks.0.attn.qkv.weight");
        assert_eq!(qkv.shape, vec![3456, 1152]);

        let deepstack = specs
            .iter()
            .find(|entry| {
                entry.role
                    == Role::Merger {
                        deepstack: Some(2),
                        part: MergerPart::NormWeight,
                    }
            })
            .expect("deepstack 2 norm");
        assert_eq!(
            deepstack.name,
            "model.visual.deepstack_merger_list.2.norm.weight"
        );
        assert_eq!(deepstack.shape, vec![4608]);

        let merger = specs
            .iter()
            .find(|entry| {
                entry.role
                    == Role::Merger {
                        deepstack: None,
                        part: MergerPart::NormWeight,
                    }
            })
            .expect("merger norm");
        assert_eq!(merger.name, "model.visual.merger.norm.weight");
        assert_eq!(merger.shape, vec![1152]);
    }

    #[test]
    fn the_layout_accepts_its_own_table_and_rejects_a_bad_shape() {
        let specs = specs();
        let shapes: Vec<Vec<u64>> = specs.iter().map(|entry| entry.shape.clone()).collect();
        let mut tensors: Vec<TensorRef<'_>> = specs
            .iter()
            .zip(&shapes)
            .map(|(entry, shape)| TensorRef {
                name: &entry.name,
                dtype: Dtype::Bf16,
                shape,
            })
            .collect();

        let layout = VisionLayout::build(&tensors).expect("the table checks itself");
        assert_eq!(layout.entries().len(), 351);

        // A wrong dtype is refused by name.
        tensors[0].dtype = Dtype::F32;
        let error = VisionLayout::build(&tensors).expect_err("F32 is refused");
        assert!(
            error.token().contains("pos_embed.weight"),
            "{}",
            error.token()
        );
        tensors[0].dtype = Dtype::Bf16;

        // A wrong shape is refused by name.
        let wrong = vec![2304u64, 1151];
        tensors[0].shape = &wrong;
        let error = VisionLayout::build(&tensors).expect_err("1151 is refused");
        assert!(
            error.token().contains("expected [2304, 1152]"),
            "{}",
            error.token()
        );
    }

    #[test]
    fn a_grid_keeps_the_reference_limits() {
        assert!(Grid::new(1, 512, 192).is_ok());
        assert!(Grid::new(2, 512, 512).is_ok());
        assert!(matches!(
            Grid::new(3, 512, 512),
            Err(VisionError::Frames(3))
        ));
        assert!(matches!(
            Grid::new(1, 0, 512),
            Err(VisionError::Edges { .. })
        ));
        assert!(matches!(
            Grid::new(1, 496, 512),
            Err(VisionError::Edges { .. })
        ));

        let grid = Grid::new(1, 512, 192).expect("accepted");
        assert_eq!(grid.rows(), (32, 12));
        assert_eq!(grid.tokens(), 384);
        assert_eq!(grid.merged(), 96);
    }

    #[test]
    fn patch_rows_follow_the_merge_block_order() {
        // 32x64 is the smallest grid with two merge blocks in one axis:
        // grid 2x4, rows 8, and every row is one 16x16 patch.
        let height = 32;
        let width = 64;
        let grid = Grid::new(1, height, width).expect("accepted");
        let mut pixels = vec![0.0f32; 3 * height * width];

        for frame_channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let index = (frame_channel * height + y) * width + x;
                    pixels[index] = (index as f32) / 10_000.0;
                }
            }
        }

        let rows = patch_rows(&pixels, 1, grid).expect("prepared");

        assert_eq!(rows.len(), 8 * 1536);

        // Row 0 is merge block (0,0), inner (0,0), channel 0, both temporal
        // slots: the first 16 columns are pixels (0, 0..16) of channel 0.
        for patch_w in 0..16 {
            let want = f32_to_bf16(pixels[patch_w] * 2.0 - 1.0);
            assert_eq!(rows[patch_w], want, "row 0 column {patch_w}");
        }

        // Row 1 is the second patch of the same merge block: x 16..32.
        for patch_w in 0..16 {
            let want = f32_to_bf16(pixels[16 + patch_w] * 2.0 - 1.0);
            assert_eq!(rows[1536 + patch_w], want, "row 1 column {patch_w}");
        }

        // Temporal slot 1 repeats frame 0 for a single frame: offset 256 is
        // channel 0, temporal 1, patch (0,0).
        assert_eq!(rows[256], f32_to_bf16(pixels[0] * 2.0 - 1.0));

        // Row 4 is merge block (0,1): its inner (0,0) starts at x=32.
        for patch_w in 0..16 {
            let want = f32_to_bf16(pixels[32 + patch_w] * 2.0 - 1.0);
            assert_eq!(rows[4 * 1536 + patch_w], want, "row 4 column {patch_w}");
        }

        // Row 2 is the same merge block's inner (1,0): y=16.
        for patch_w in 0..16 {
            let want = f32_to_bf16(pixels[16 * width + patch_w] * 2.0 - 1.0);
            assert_eq!(rows[2 * 1536 + patch_w], want, "row 2 column {patch_w}");
        }
    }

    #[test]
    fn two_frames_take_their_own_temporal_slot() {
        let height = 32;
        let width = 32;
        let grid = Grid::new(2, height, width).expect("accepted");
        let mut pixels = vec![0.0f32; 2 * 3 * height * width];

        for frame in 0..2 {
            for channel in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let index = ((frame * 3 + channel) * height + y) * width + x;
                        pixels[index] = (frame as f32 + 0.5) / 4.0;
                    }
                }
            }
        }

        let rows = patch_rows(&pixels, 2, grid).expect("prepared");

        // Column 256 is temporal slot 1 of channel 0, patch (0,0).
        let first = f32_to_bf16(0.5 / 4.0 * 2.0 - 1.0);
        let second = f32_to_bf16(1.5 / 4.0 * 2.0 - 1.0);
        assert_eq!(rows[0], first);
        assert_eq!(rows[256], second);
    }

    #[test]
    fn a_pixel_buffer_that_disagrees_with_the_grid_is_refused() {
        let grid = Grid::new(1, 32, 32).expect("accepted");
        let error = patch_rows(&[0.0f32; 10], 1, grid).expect_err("short buffer");

        assert!(matches!(
            error,
            VisionError::Pixels {
                got: 10,
                want: 3072
            }
        ));
    }

    /// A table whose values are exact in BF16 and differ per cell and per
    /// channel, so a wrong index cannot hide behind rounding: the value is
    /// `((cell % 5) + 1) * (1 + (channel % 16) / 16)`, a multiple of 1/128 at
    /// most 9.6875 (the BF16 spacing there is 1/16).
    fn distinctive_table() -> Vec<u16> {
        let hidden = HIDDEN as usize;
        let mut table = vec![0u16; POSITION_COUNT as usize * hidden];

        for cell in 0..POSITION_COUNT as usize {
            for channel in 0..hidden {
                let value = (cell % 5 + 1) as f32 * (1.0 + (channel % 16) as f32 / 16.0);
                table[cell * hidden + channel] = f32_to_bf16(value);
            }
        }

        table
    }

    #[test]
    fn position_rows_round_coefficient_term_and_sum() {
        let grid = Grid::new(1, 64, 64).expect("accepted");
        let hidden = HIDDEN as usize;
        let table = distinctive_table();

        let rows = position_rows(&table, grid).expect("prepared");
        assert_eq!(rows.len(), 16 * hidden);

        // Row 1 is (block 0, inner (0,1)): h = 0, w = 1, and a 4x4 grid maps
        // w = 1 to 47/3 = 15.666.. in the table. h is 0, so dh = 0 and only
        // the two h0 corners carry weight.
        let wv = 47.0f64 / 3.0;
        let w0 = wv as usize;
        let dw = (wv - w0 as f64) as f32;
        let coefficients = [1.0 - dw, dw];
        let indices = [w0, w0 + 1];

        for channel in 0..hidden {
            let mut sum = 0.0f32;

            for corner in 0..2 {
                let coefficient = bf16_to_f32(f32_to_bf16(coefficients[corner]));
                let term = bf16_to_f32(table[indices[corner] * hidden + channel]) * coefficient;
                sum += bf16_to_f32(f32_to_bf16(term));
            }

            assert_eq!(
                rows[hidden + channel],
                f32_to_bf16(sum),
                "row 1 channel {channel}"
            );
        }

        // Row 4 is the second merge block's inner (0,0): h = 0, w = 2.
        let wv = 2.0f64 * 47.0 / 3.0;
        let w0 = wv as usize;
        let dw = (wv - w0 as f64) as f32;
        let coefficients = [1.0 - dw, dw];
        let indices = [w0, w0 + 1];

        for channel in 0..hidden {
            let mut sum = 0.0f32;

            for corner in 0..2 {
                let coefficient = bf16_to_f32(f32_to_bf16(coefficients[corner]));
                let term = bf16_to_f32(table[indices[corner] * hidden + channel]) * coefficient;
                sum += bf16_to_f32(f32_to_bf16(term));
            }

            assert_eq!(
                rows[4 * hidden + channel],
                f32_to_bf16(sum),
                "row 4 channel {channel}"
            );
        }
    }

    #[test]
    fn position_rows_walk_the_hidden_axis_too() {
        let grid = Grid::new(1, 64, 64).expect("accepted");
        let hidden = HIDDEN as usize;
        let side = POSITION_SIDE as usize;
        let table = distinctive_table();

        let rows = position_rows(&table, grid).expect("prepared");
        // Row 3 is inner (1,1): h = 1 and w = 1.
        let hv = 47.0f64 / 3.0;
        let h0 = hv as usize;
        let dh = (hv - h0 as f64) as f32;
        let coefficients = [
            (1.0 - dh) * (1.0 - dh),
            (1.0 - dh) * dh,
            dh * (1.0 - dh),
            dh * dh,
        ];
        let indices = [
            h0 * side + h0,
            h0 * side + h0 + 1,
            (h0 + 1) * side + h0,
            (h0 + 1) * side + h0 + 1,
        ];

        for channel in 0..hidden {
            let mut sum = 0.0f32;

            for corner in 0..4 {
                let coefficient = bf16_to_f32(f32_to_bf16(coefficients[corner]));
                let term = bf16_to_f32(table[indices[corner] * hidden + channel]) * coefficient;
                sum += bf16_to_f32(f32_to_bf16(term));
            }

            assert_eq!(
                rows[3 * hidden + channel],
                f32_to_bf16(sum),
                "row 3 channel {channel}"
            );
        }
    }

    #[test]
    fn rope_rows_carry_two_axes_of_eighteen_pairs() {
        let grid = Grid::new(1, 64, 64).expect("accepted");
        let rope = rope_rows(grid);

        assert_eq!(rope.rows(), 16);
        assert_eq!(rope.cosines().len(), 16 * 36);
        assert_eq!(rope.sines().len(), 16 * 36);

        // Row 1 is inner (0,1): axis 0 stands at 0 (cos 1, sin 0); axis 1
        // stands at 1.
        assert_eq!(rope.cosines()[36], f32_to_bf16(1.0));
        assert_eq!(rope.sines()[36], f32_to_bf16(0.0));

        let inverse = 10_000.0f32.powf(0.0);
        assert_eq!(rope.cosines()[36 + 18], f32_to_bf16(inverse.cos()));
        assert_eq!(rope.sines()[36 + 18], f32_to_bf16(inverse.sin()));

        // Row 3 is inner (1,1): both axes stand at 1, so the first pair is
        // the same as row 1's second axis.
        assert_eq!(rope.cosines()[3 * 36], rope.cosines()[36 + 18]);
        assert_eq!(rope.sines()[3 * 36], rope.sines()[36 + 18]);
    }

    #[test]
    fn bf16_rounding_matches_the_reference_expression() {
        // 1.0 carries 7 explicit mantissa bits, so the spacing above it is
        // 1/128 and a tie sits at 1/256.
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 1.0 / 128.0), 0x3f81);
        // Round to even: a tie goes to the neighbour with a zero last bit.
        assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(f32_to_bf16(1.0 + 5.0 / 256.0), 0x3f82);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(f32_to_bf16(0.0), 0);
    }
}
