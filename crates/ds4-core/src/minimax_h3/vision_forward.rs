//! The MiniMax-H3 vision-tower forward.
//!
//! One host loop over the device primitives in `native/h3/h3_vision_gpu.h`.
//! Source of truth `/data/h3.c` at `eae5d54`:
//!
//! - `h3_vision_encoder.c:548-585` the pipeline: patch embed (1536 -> 1152)
//!   plus the position rows, 27 blocks, a deepstack merger after layers 8, 16
//!   and 24 (0-based), then the final merger.
//! - `h3_vision_encoder.c:288-345` `run_block`: layer norm, qkv, vision RoPE,
//!   attention at `1/sqrt(72)`, output projection, residual, layer norm, fc1,
//!   GELU(approximate), fc2, residual.
//! - `h3_vision_encoder.c:347-426` `run_merger`: one layer norm over the patch
//!   grid seen as merged rows, fc1, GELU(exact), fc2 to 5120.
//! - `h3_gpu_cuda.cu:852-870` the projections are cuBLASLt BF16 matmuls with a
//!   separate bias launch, so the output is rounded twice exactly as the
//!   reference rounds it.
//!
//! Two reference hooks are not carried, and neither changes a number: the
//! `H3_DEBUG_VISION` / `H3_DEBUG_VISION_STEPS` dumps (`:58-100`, `:119-127`)
//! and the `h3_gpu_stats` counters (`:606`). The gate hashes the outputs
//! instead. The progress callback is carried as `progress`.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_vision::{VisionGpu, VisionTensor};

use crate::minimax_h3::text_forward::fnv1a;
use crate::minimax_h3::vision::{
    patch_rows, position_rows, rope_rows, Grid, MergerPart, Part, Role, VisionError, VisionRope,
    CHANNELS, DEEPSTACKS, DEEPSTACK_LAYERS, HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, LAYERS,
    LAYER_NORM_EPSILON, MERGE_DIM, MERGE_ROWS, OUTPUT_WIDTH, PATCH_DIM, ROPE_HALF,
};
use crate::minimax_h3::vision_weights::{VisionLoadError, VisionWeights};

/// The vision tower's output: BF16 Qwen presentation rows plus the three
/// deepstack additions (`h3_vision_encoder.h:12-19`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionOutput {
    grid: Grid,
    tokens: usize,
    merged: Vec<u8>,
    deepstack: [Vec<u8>; DEEPSTACKS],
}

impl VisionOutput {
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// Merged rows: `grid_h * grid_w / 4`.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// `H3_VISION_OUTPUT_WIDTH`.
    pub fn width(&self) -> usize {
        OUTPUT_WIDTH as usize
    }

    /// The final merger's BF16 payload.
    pub fn merged(&self) -> &[u8] {
        &self.merged
    }

    pub fn deepstack(&self, index: usize) -> &[u8] {
        &self.deepstack[index]
    }

    /// FNV-1a over `merged`, the probe's convention.
    pub fn fnv(&self) -> u64 {
        fnv1a(&self.merged)
    }

    /// FNV-1a over one deepstack payload.
    pub fn deepstack_fnv(&self, index: usize) -> u64 {
        fnv1a(&self.deepstack[index])
    }

    /// The first `count` BF16 words of `merged`, for the gate's first= field.
    pub fn first_words(&self, count: usize) -> Vec<u16> {
        let total = self.merged.len() / 2;
        let count = count.min(total);

        words(&self.merged, 0, count)
    }

    /// The last `count` BF16 words of `merged`.
    pub fn last_words(&self, count: usize) -> Vec<u16> {
        let total = self.merged.len() / 2;
        let count = count.min(total);

        words(&self.merged, total - count, count)
    }
}

/// Little-endian BF16 words of one span.
fn words(bytes: &[u8], start: usize, count: usize) -> Vec<u16> {
    bytes[start * 2..(start + count) * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// Encode one image, or one two-frame block, into the Qwen presentation rows.
pub fn encode(
    weights: &VisionWeights,
    gpu: &VisionGpu,
    pixels: &[f32],
    frames: usize,
    grid: Grid,
) -> Result<VisionOutput, ForwardError> {
    encode_with(weights, gpu, pixels, frames, grid, &mut |_, _| {})
}

/// [`encode`] with the reference's per-layer progress callback
/// (`h3_vision_encoder.c:576`).
pub fn encode_with(
    weights: &VisionWeights,
    gpu: &VisionGpu,
    pixels: &[f32],
    frames: usize,
    grid: Grid,
    progress: &mut dyn FnMut(u32, u32),
) -> Result<VisionOutput, ForwardError> {
    let rows = grid.tokens();
    let merged_rows = grid.merged();
    let hidden = rows * HIDDEN as usize;
    let patch_elements = rows * PATCH_DIM as usize;

    if rows > u32::MAX as usize / 3 {
        return Err(ForwardError::Gpu(
            "vision patch sequence is too large".to_string(),
        ));
    }

    // The three prepared inputs. Each is rounded on the host, exactly as the
    // reference rounds it before the upload (`:160`, `:241`, `:244`).
    let table = read_table(weights)?;
    let positions = position_rows(&table, grid).map_err(ForwardError::Host)?;
    let patches = patch_rows(pixels, frames, grid).map_err(ForwardError::Host)?;
    let rope = rope_rows(grid);
    drop(table);

    let patches = upload(gpu, &patches, patch_elements)?;
    let positions = upload(gpu, &positions, hidden)?;
    let (cosines, sines) = upload_rope(gpu, &rope, rows)?;

    // Patch embed plus the position rows (`h3_vision_encoder.c:513-526`).
    let patch_weight = weight(weights, gpu, Role::PatchWeight)?;
    let patch_bias = weight(weights, gpu, Role::PatchBias)?;
    let hidden_tensor = alloc(gpu, hidden)?;
    let rows_u32 = as_u32(rows)?;
    let hidden_u32 = as_u32(hidden)?;

    gpu.linear(
        &hidden_tensor,
        &patches,
        &patch_weight,
        Some(&patch_bias),
        rows_u32,
        as_u32(PATCH_DIM as usize)?,
        as_u32(HIDDEN as usize)?,
    )?;
    gpu.add(&hidden_tensor, &hidden_tensor, &positions, hidden_u32)?;
    drop((patches, positions, patch_weight, patch_bias));

    // One set of activations for every block, sized by the patch grid
    // (`h3_vision_encoder.c:536-546`).
    let activations = Activations::new(gpu, rows)?;
    let mut deepstack: [Option<VisionTensor>; DEEPSTACKS] = [None, None, None];

    for layer in 0..LAYERS {
        let block = Block::load(weights, gpu, layer)?;
        run_block(
            gpu,
            &block,
            &activations,
            &hidden_tensor,
            &cosines,
            &sines,
            rows,
            hidden,
        )?;
        drop(block);
        progress(layer + 1, LAYERS);

        if let Some(index) = DEEPSTACK_LAYERS.iter().position(|at| *at == layer) {
            deepstack[index] = Some(run_merger(
                weights,
                gpu,
                &hidden_tensor,
                rows,
                merged_rows,
                Some(index as u32),
            )?);
        }
    }

    let merged = run_merger(weights, gpu, &hidden_tensor, rows, merged_rows, None)?;
    let elements = merged_rows * OUTPUT_WIDTH as usize;

    let merged_bytes = read(gpu, &merged, elements)?;
    let mut stacks: [Vec<u8>; DEEPSTACKS] = std::array::from_fn(|_| Vec::new());

    for (index, tensor) in deepstack.iter().enumerate() {
        let tensor = tensor
            .as_ref()
            .ok_or_else(|| ForwardError::Gpu(format!("deepstack merger {index} did not run")))?;

        stacks[index] = read(gpu, tensor, elements)?;
    }

    drop((
        merged,
        deepstack,
        activations,
        cosines,
        sines,
        hidden_tensor,
    ));

    Ok(VisionOutput {
        grid,
        tokens: merged_rows,
        merged: merged_bytes,
        deepstack: stacks,
    })
}

/// The allocation sizes the forward reuses. A block and both merger shapes are
/// covered by the largest: `rows * INTERMEDIATE` for the block MLP,
/// `merged_rows * MERGE_DIM` for a merger (only ever smaller).
struct Activations {
    norm: VisionTensor,
    qkv: VisionTensor,
    query: VisionTensor,
    key: VisionTensor,
    value: VisionTensor,
    heads: VisionTensor,
    branch: VisionTensor,
    fc1: VisionTensor,
    activated: VisionTensor,
}

impl Activations {
    fn new(gpu: &VisionGpu, rows: usize) -> Result<Activations, ForwardError> {
        let hidden = rows * HIDDEN as usize;
        let wide = rows * INTERMEDIATE as usize;

        Ok(Activations {
            norm: alloc(gpu, hidden)?,
            qkv: alloc(gpu, hidden * 3)?,
            query: alloc(gpu, hidden)?,
            key: alloc(gpu, hidden)?,
            value: alloc(gpu, hidden)?,
            heads: alloc(gpu, hidden)?,
            branch: alloc(gpu, hidden)?,
            fc1: alloc(gpu, wide)?,
            activated: alloc(gpu, wide)?,
        })
    }
}

/// One block's twelve tensors.
struct Block {
    norm1_w: VisionTensor,
    norm1_b: VisionTensor,
    qkv_w: VisionTensor,
    qkv_b: VisionTensor,
    proj_w: VisionTensor,
    proj_b: VisionTensor,
    norm2_w: VisionTensor,
    norm2_b: VisionTensor,
    fc1_w: VisionTensor,
    fc1_b: VisionTensor,
    fc2_w: VisionTensor,
    fc2_b: VisionTensor,
}

impl Block {
    fn load(weights: &VisionWeights, gpu: &VisionGpu, layer: u32) -> Result<Block, ForwardError> {
        // Shapes are checked at open time, so the sizes here are the table's.
        let norm1_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Norm1Weight,
            },
        )?;
        let norm1_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Norm1Bias,
            },
        )?;
        let qkv_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::QkvWeight,
            },
        )?;
        let qkv_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::QkvBias,
            },
        )?;
        let proj_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::ProjWeight,
            },
        )?;
        let proj_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::ProjBias,
            },
        )?;
        let norm2_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Norm2Weight,
            },
        )?;
        let norm2_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Norm2Bias,
            },
        )?;
        let fc1_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Fc1Weight,
            },
        )?;
        let fc1_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Fc1Bias,
            },
        )?;
        let fc2_w = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Fc2Weight,
            },
        )?;
        let fc2_b = weight(
            weights,
            gpu,
            Role::Block {
                layer,
                part: Part::Fc2Bias,
            },
        )?;

        Ok(Block {
            norm1_w,
            norm1_b,
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
            norm2_w,
            norm2_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        })
    }
}

/// `run_block` (`h3_vision_encoder.c:288-345`).
#[allow(clippy::too_many_arguments)]
fn run_block(
    gpu: &VisionGpu,
    block: &Block,
    activations: &Activations,
    hidden: &VisionTensor,
    cosines: &VisionTensor,
    sines: &VisionTensor,
    rows: usize,
    elements: usize,
) -> Result<(), ForwardError> {
    let hidden_width = as_u32(HIDDEN as usize)?;
    let rows_u32 = as_u32(rows)?;
    let elements_u32 = as_u32(elements)?;

    gpu.layer_norm(
        &activations.norm,
        hidden,
        &block.norm1_w,
        &block.norm1_b,
        rows_u32,
        hidden_width,
        LAYER_NORM_EPSILON,
    )?;
    gpu.linear(
        &activations.qkv,
        &activations.norm,
        &block.qkv_w,
        Some(&block.qkv_b),
        rows_u32,
        hidden_width,
        as_u32(HIDDEN as usize * 3)?,
    )?;
    gpu.qkv_rope(
        &activations.query,
        &activations.key,
        &activations.value,
        &activations.qkv,
        cosines,
        sines,
        rows_u32,
        as_u32(HEADS as usize)?,
        as_u32(HEAD_DIM as usize)?,
        as_u32(ROPE_HALF as usize)?,
    )?;
    gpu.sdpa(
        &activations.heads,
        &activations.query,
        &activations.key,
        &activations.value,
        rows_u32,
        as_u32(HEADS as usize)?,
        as_u32(HEAD_DIM as usize)?,
        1.0 / (HEAD_DIM as f32).sqrt(),
    )?;
    gpu.linear(
        &activations.branch,
        &activations.heads,
        &block.proj_w,
        Some(&block.proj_b),
        rows_u32,
        hidden_width,
        hidden_width,
    )?;
    gpu.add(hidden, hidden, &activations.branch, elements_u32)?;

    gpu.layer_norm(
        &activations.norm,
        hidden,
        &block.norm2_w,
        &block.norm2_b,
        rows_u32,
        hidden_width,
        LAYER_NORM_EPSILON,
    )?;
    gpu.linear(
        &activations.fc1,
        &activations.norm,
        &block.fc1_w,
        Some(&block.fc1_b),
        rows_u32,
        hidden_width,
        as_u32(INTERMEDIATE as usize)?,
    )?;
    gpu.gelu(
        &activations.activated,
        &activations.fc1,
        as_u32(rows * INTERMEDIATE as usize)?,
        true,
    )?;
    gpu.linear(
        &activations.branch,
        &activations.activated,
        &block.fc2_w,
        Some(&block.fc2_b),
        rows_u32,
        as_u32(INTERMEDIATE as usize)?,
        hidden_width,
    )?;
    gpu.add(hidden, hidden, &activations.branch, elements_u32)?;

    Ok(())
}

/// `run_merger` (`h3_vision_encoder.c:347-426`).
///
/// The hidden state is `[patch_rows, 1152]`; a merger reads the same bytes as
/// `[merged_rows, 4608]`. That view is valid only because the patch prep
/// writes each 2x2 merge block's four rows consecutively
/// (`h3_vision_encoder.c:139-146`), which `minimax_h3::vision` keeps.
fn run_merger(
    weights: &VisionWeights,
    gpu: &VisionGpu,
    hidden: &VisionTensor,
    patch_rows: usize,
    merged_rows: usize,
    deepstack: Option<u32>,
) -> Result<VisionTensor, ForwardError> {
    // The final merger normalizes one patch row at a time; a deepstack merger
    // normalizes one merged row of four patch rows (`:357-359`).
    let (norm_width, norm_rows) = match deepstack {
        Some(_) => (MERGE_DIM as usize, merged_rows),
        None => (HIDDEN as usize, patch_rows),
    };
    let rows_u32 = as_u32(merged_rows)?;
    let merge_dim = as_u32(MERGE_DIM as usize)?;

    let norm_w = merger_weight(weights, gpu, deepstack, MergerPart::NormWeight)?;
    let norm_b = merger_weight(weights, gpu, deepstack, MergerPart::NormBias)?;
    let fc1_w = merger_weight(weights, gpu, deepstack, MergerPart::Fc1Weight)?;
    let fc1_b = merger_weight(weights, gpu, deepstack, MergerPart::Fc1Bias)?;
    let fc2_w = merger_weight(weights, gpu, deepstack, MergerPart::Fc2Weight)?;
    let fc2_b = merger_weight(weights, gpu, deepstack, MergerPart::Fc2Bias)?;

    let norm = alloc(gpu, patch_rows * HIDDEN as usize)?;
    let fc1 = alloc(gpu, merged_rows * MERGE_DIM as usize)?;
    let activated = alloc(gpu, merged_rows * MERGE_DIM as usize)?;
    let output = alloc(gpu, merged_rows * OUTPUT_WIDTH as usize)?;

    gpu.layer_norm(
        &norm,
        hidden,
        &norm_w,
        &norm_b,
        as_u32(norm_rows)?,
        as_u32(norm_width)?,
        LAYER_NORM_EPSILON,
    )?;
    gpu.linear(
        &fc1,
        &norm,
        &fc1_w,
        Some(&fc1_b),
        rows_u32,
        merge_dim,
        merge_dim,
    )?;
    gpu.gelu(
        &activated,
        &fc1,
        as_u32(merged_rows * MERGE_DIM as usize)?,
        false,
    )?;
    gpu.linear(
        &output,
        &activated,
        &fc2_w,
        Some(&fc2_b),
        rows_u32,
        merge_dim,
        as_u32(OUTPUT_WIDTH as usize)?,
    )?;

    Ok(output)
}

fn merger_weight(
    weights: &VisionWeights,
    gpu: &VisionGpu,
    deepstack: Option<u32>,
    part: MergerPart,
) -> Result<VisionTensor, ForwardError> {
    weight(weights, gpu, Role::Merger { deepstack, part })
}

/// Read the 48x48 position table into BF16 words.
fn read_table(weights: &VisionWeights) -> Result<Vec<u16>, ForwardError> {
    let bytes = read_role(weights, Role::PositionTable)?;

    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect())
}

/// Read one role and upload it as a device tensor.
fn weight(
    weights: &VisionWeights,
    gpu: &VisionGpu,
    role: Role,
) -> Result<VisionTensor, ForwardError> {
    let bytes = read_role(weights, role)?;
    let elements = bytes.len() / 2;
    let tensor = alloc(gpu, elements)?;

    gpu.upload(&tensor, &bytes).map_err(ForwardError::Gpu)?;

    Ok(tensor)
}

fn read_role(weights: &VisionWeights, role: Role) -> Result<Vec<u8>, ForwardError> {
    let mut bytes = Vec::new();

    weights
        .read_into(role, &mut bytes)
        .map_err(ForwardError::Load)?;

    Ok(bytes)
}

/// Upload BF16 words.
fn upload(gpu: &VisionGpu, values: &[u16], elements: usize) -> Result<VisionTensor, ForwardError> {
    if values.len() != elements {
        return Err(ForwardError::Gpu(format!(
            "prepared {} values for a {elements} element tensor",
            values.len()
        )));
    }

    let tensor = alloc(gpu, elements)?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    gpu.upload(&tensor, &bytes).map_err(ForwardError::Gpu)?;

    Ok(tensor)
}

fn upload_rope(
    gpu: &VisionGpu,
    rope: &VisionRope,
    rows: usize,
) -> Result<(VisionTensor, VisionTensor), ForwardError> {
    let elements = rows * ROPE_HALF as usize;

    Ok((
        upload(gpu, rope.cosines(), elements)?,
        upload(gpu, rope.sines(), elements)?,
    ))
}

fn alloc(gpu: &VisionGpu, elements: usize) -> Result<VisionTensor, ForwardError> {
    gpu.alloc(elements).map_err(ForwardError::Gpu)
}

fn read(gpu: &VisionGpu, tensor: &VisionTensor, elements: usize) -> Result<Vec<u8>, ForwardError> {
    let mut bytes = vec![0u8; elements * 2];

    gpu.read(tensor, &mut bytes).map_err(ForwardError::Gpu)?;

    Ok(bytes)
}

fn as_u32(value: usize) -> Result<u32, ForwardError> {
    u32::try_from(value).map_err(|_| ForwardError::Gpu(format!("{value} does not fit a u32 shape")))
}

/// Why the vision forward failed.
#[derive(Debug)]
pub enum ForwardError {
    /// A vision tensor could not be read.
    Load(VisionLoadError),
    /// The host preparation refused the input.
    Host(VisionError),
    /// A device call failed.
    Gpu(String),
}

impl ForwardError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            ForwardError::Load(error) => error.token(),
            ForwardError::Host(error) => error.token(),
            ForwardError::Gpu(detail) => format!("gpu: {detail}"),
        }
    }
}

impl fmt::Display for ForwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for ForwardError {}

/// Every `ds4_sys::h3_vision` wrapper reports device failures as one text
/// line, so a bare `?` on one belongs to [`ForwardError::Gpu`].
impl From<String> for ForwardError {
    fn from(detail: String) -> ForwardError {
        ForwardError::Gpu(detail)
    }
}

/// Open the text partition and run one image through the tower.
pub fn encode_dir(
    text_encoder: &Path,
    pixels: &[f32],
    frames: usize,
    grid: Grid,
) -> Result<VisionOutput, ForwardError> {
    let weights = VisionWeights::open(text_encoder).map_err(ForwardError::Load)?;
    let gpu = VisionGpu::create().map_err(ForwardError::Gpu)?;

    encode(&weights, &gpu, pixels, frames, grid)
}

/// The channels of an image, kept so a caller cannot pass the wrong count.
pub const FRAME_CHANNELS: u64 = CHANNELS;

/// One 2x2 merge block's four patch rows.
pub const MERGED_BLOCK: u64 = MERGE_ROWS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_output_words_are_little_endian_bf16() {
        let output = VisionOutput {
            grid: Grid::new(1, 32, 32).expect("accepted"),
            tokens: 2,
            merged: vec![0x00, 0x3f, 0x80, 0xbf],
            deepstack: std::array::from_fn(|_| Vec::new()),
        };

        assert_eq!(output.first_words(2), vec![0x3f00, 0xbf80]);
        assert_eq!(output.last_words(1), vec![0xbf80]);
        assert_eq!(output.first_words(8), vec![0x3f00, 0xbf80]);
        assert_ne!(output.fnv(), 0);
    }
}
