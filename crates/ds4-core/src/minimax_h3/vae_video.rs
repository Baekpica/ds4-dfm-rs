//! The released video VAE decoder (`h3_video_vae.c:1-1252`).
//!
//! Decode only: the encode half serves reference conditioning and is a
//! separate unit. The graph is the reference's 36-block transformer decoder
//! over F32, trimmed to the handful of primitives it calls: `h3d_linear`
//! (F32), `h3d_rms_norm_f32`, `h3d_video_qkv_rope_f32`, `h3d_sdpa_f32`,
//! `h3d_swiglu_f32`, `h3d_scale_add_f32`, `h3d_layer_norm_f32`,
//! `h3d_copy_f32`.
//!
//! Geometry (`:11-27`): 24 latent channels, a 7-latent chunk, 22 frames from
//! the first chunk and 17 new frames per later chunk over a 5-frame overlap,
//! hidden 2048, 36 layers, 32 heads of 64, FFN 8192, video RoPE over 24
//! halves (3 axes x 8 frequencies), and a 3072-value output patch per row
//! (`3 channels x 4 time x 16 x 16`). Pixels are 16 per latent edge.
//!
//! Layouts are the reference's. The latent arrives `[channel, time, height,
//! width]` (`h3_dit.h:93-98`); every decoder row packs `(time, height, width,
//! channel)` (`:320-346`); the block works on `[row, hidden]` and the video
//! QKV kernel writes head-major `[head, row, dim]` (`:3078-3136`); frames come
//! out frame-major interleaved RGB F32 in `[0, 1]` (`h3_video_vae.h:10-16`).
//!
//! Two deviations, both named:
//!
//! - The reference's single-chunk path streams one block at a time and frees
//!   it (`run_decoder`, `:458-531`); the port keeps all 36 blocks resident
//!   (`load_resident_weights`, `:518-531`), which the chunked path needs
//!   anyway. The arithmetic per row is unchanged and the E8 gate is bit-exact.
//! - `H3_VAE_TILE_PIXELS` (`:666-690`) is not read; the default search that
//!   the reference uses when the variable is absent is. Every released shape
//!   picks the reference's tile size.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::vae_store::{f1, f2, Store};

pub const LATENT_CHANNELS: usize = 24;
/// One trained chunk: seven latents in, twenty-two frames out.
const CHUNK_LATENT_TIME: usize = 7;
const FIRST_CHUNK_FRAMES: usize = 22;
/// Latent time 0 maps to output frame 3 (`:613-614`).
const FRAME_OFFSET: usize = 3;
const HIDDEN: usize = 2048;
const LAYERS: usize = 36;
const HEADS: u32 = 32;
const HEAD_DIM: u32 = 64;
const INNER: usize = HEADS as usize * HEAD_DIM as usize;
const FFN: usize = 8192;
/// Four learned register tokens plus one zero suffix token (`:12-13`).
const REGISTERS: usize = 4;
const SUFFIX: usize = 5;
const ROPE_HALF: u32 = 24;
const OUTPUT_PATCH: usize = 3 * 4 * 16 * 16;
const SPATIAL_RATIO: usize = 16;
const TILE_PIXELS: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
const EPSILON: f32 = 1e-5;
/// Pixels per latent edge, both axes.
const PATCH: usize = 16;
/// Frames each later chunk repeats before its own first new frame.
const CHUNK_OVERLAP: usize = 5;
/// New frames a later chunk contributes after the overlap (`:1092`).
const CHUNK_STRIDE: usize = 17;
/// The RGB normalization the output patches are decoded with (`:653-654`).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const DEVICE_ERROR: &str = "video VAE device";

/// What `h3_video_vae_decode` returns (`h3_video_vae.h:10-16`).
pub struct Frames {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    /// Frame-major interleaved RGB F32 in `[0, 1]`.
    pub rgb: Vec<f32>,
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

fn f32_alloc(gpu: &Gpu, elements: usize) -> Result<Tensor, VaeError> {
    device(gpu.alloc(Dtype::F32, elements))
}

/// Upload an f32 slice as its own bytes; the reference streams the same
/// buffer, so no conversion happens on either side.
fn upload_f32(gpu: &Gpu, tensor: &Tensor, values: &[f32]) -> Result<(), VaeError> {
    // SAFETY: f32 has no padding or invalid bit patterns, so reinterpreting
    // the slice as bytes is sound; the bytes are only read.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    device(gpu.upload(tensor, bytes))
}

/// The spatial tile plan for one axis (`tile_axis`, `h3_video_vae.c:692-731`).
struct TileAxis {
    starts: Vec<usize>,
    /// One entry fewer than `starts`; the last tile has no following overlap.
    overlaps: Vec<usize>,
    length: usize,
}

impl TileAxis {
    fn count(&self) -> usize {
        self.starts.len()
    }

    /// Pixels this axis covers after stitching.
    fn extent(&self) -> usize {
        self.starts[self.starts.len() - 1] + self.length
    }
}

/// `tile_count_for_extent` (`:658-664`): the minimum count whose tiles, minus
/// their minimum overlaps, still reach the extent.
fn tile_count_for_extent(extent: usize, tile_pixels: usize) -> usize {
    if extent <= tile_pixels {
        return 1;
    }

    let mut count = extent.div_ceil(tile_pixels);
    while tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent {
        count += 1;
    }
    count
}

/// `configured_tile_pixels` (`:666-690`): the smallest `tiles * pixels^2`
/// wins, searching 256..=320 in 16-pixel steps. The reference's environment
/// override is not read here.
fn configured_tile_pixels(height: usize, width: usize) -> usize {
    let mut best = TILE_PIXELS;
    let mut best_score = usize::MAX;

    let mut pixels = TILE_PIXELS;
    while pixels <= 320 {
        let tiles = tile_count_for_extent(height, pixels)
            .saturating_mul(tile_count_for_extent(width, pixels));
        let score = tiles.saturating_mul(pixels).saturating_mul(pixels);
        if score < best_score {
            best = pixels;
            best_score = score;
        }
        pixels += SPATIAL_RATIO;
    }
    best
}

/// `tile_axis_build` (`:692-731`): distribute the surplus over the overlaps in
/// 16-pixel units, first overlap first.
fn tile_axis(extent: usize, tile_pixels: usize) -> Result<TileAxis, VaeError> {
    if extent < 1 || !extent.is_multiple_of(SPATIAL_RATIO) {
        return fail(format!(
            "video VAE tile extent must be a multiple of {SPATIAL_RATIO}"
        ));
    }
    if extent <= tile_pixels {
        return Ok(TileAxis {
            starts: vec![0],
            overlaps: Vec::new(),
            length: extent,
        });
    }

    let count = tile_count_for_extent(extent, tile_pixels);
    let mut overlaps = vec![TILE_OVERLAP_MIN; count - 1];
    let covered = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1);
    for unit in 0..(covered - extent) / SPATIAL_RATIO {
        overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    }
    let mut starts = Vec::with_capacity(count);
    starts.push(0);
    for index in 1..count {
        starts.push(starts[index - 1] + tile_pixels - overlaps[index - 1]);
    }

    Ok(TileAxis {
        starts,
        overlaps,
        length: tile_pixels,
    })
}

/// `(latent_time - 2) / 5` chunks and `chunks * 17 + 5` frames
/// (`h3_video_vae.c:1092-1093`); latent time 2 is the two-token path.
fn chunk_plan(latent_time: usize) -> Result<(usize, usize), VaeError> {
    if latent_time == 2 {
        return Ok((0, 5));
    }
    if latent_time < CHUNK_LATENT_TIME || !(latent_time - 2).is_multiple_of(CHUNK_OVERLAP) {
        return fail(format!(
            "latent time {latent_time} is not a released chunk count"
        ));
    }
    let chunks = (latent_time - 2) / CHUNK_OVERLAP;
    Ok((chunks, chunks * CHUNK_STRIDE + CHUNK_OVERLAP))
}

/// One transformer block's tensors (`load_block`, `:169-203`).
struct Block {
    norm1: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    scale1: Tensor,
    norm2: Tensor,
    w1: Tensor,
    w1_b: Tensor,
    w2: Tensor,
    w2_b: Tensor,
    scale2: Tensor,
}

/// The projections around the blocks (`load_input_weights` `:205-220`,
/// `load_output_weights` `:222-233`).
struct Parts {
    post_w: Tensor,
    post_b: Tensor,
    embed_w: Tensor,
    embed_b: Tensor,
    registers: Tensor,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

/// The loaded decoder, reusable across chunks and tiles.
pub struct Decoder {
    gpu: Gpu,
    parts: Parts,
    blocks: Vec<Block>,
    patches: usize,
    sequence: usize,
    rope_cos: Tensor,
    rope_sin: Tensor,
    mean: Vec<f32>,
    deviation: Vec<f32>,
}

fn load_block(store: &Store, gpu: &Gpu, index: usize) -> Result<Block, VaeError> {
    let prefix = format!("decoder.transformer_blocks.{index}.");
    let name = |suffix: &str| format!("{prefix}{suffix}");

    Ok(Block {
        norm1: f1(store, gpu, &name("norm1.weight"), HIDDEN as u64)?,
        qkv_w: f2(
            store,
            gpu,
            &name("attn.to_qkv.weight"),
            (INNER * 3) as u64,
            HIDDEN as u64,
        )?,
        qkv_b: f1(store, gpu, &name("attn.to_qkv.bias"), (INNER * 3) as u64)?,
        out_w: f2(
            store,
            gpu,
            &name("attn.to_out.weight"),
            HIDDEN as u64,
            INNER as u64,
        )?,
        out_b: f1(store, gpu, &name("attn.to_out.bias"), HIDDEN as u64)?,
        scale1: f1(store, gpu, &name("scale1"), HIDDEN as u64)?,
        norm2: f1(store, gpu, &name("norm2.weight"), HIDDEN as u64)?,
        w1: f2(
            store,
            gpu,
            &name("ff.w1.weight"),
            (FFN * 2) as u64,
            HIDDEN as u64,
        )?,
        w1_b: f1(store, gpu, &name("ff.w1.bias"), (FFN * 2) as u64)?,
        w2: f2(store, gpu, &name("ff.w2.weight"), HIDDEN as u64, FFN as u64)?,
        w2_b: f1(store, gpu, &name("ff.w2.bias"), HIDDEN as u64)?,
        scale2: f1(store, gpu, &name("scale2"), HIDDEN as u64)?,
    })
}

/// `prepare_rope` (`:348-396`): one row per `(t, h, w)` in the same order the
/// latent rows take, 24 halves each; the register and suffix rows carry
/// cos 1 / sin 0.
fn rope_tables(sequence: usize, latent_h: usize, latent_w: usize) -> (Vec<f32>, Vec<f32>) {
    let halves = ROPE_HALF as usize;
    let mut cosines = vec![0.0f32; sequence * halves];
    let mut sines = vec![0.0f32; sequence * halves];
    let mut row = 0;

    for time in 0..CHUNK_LATENT_TIME {
        for height in 0..latent_h {
            for width in 0..latent_w {
                let axes = [
                    2.0f32 * ((time as f32 + 0.5) / CHUNK_LATENT_TIME as f32) - 1.0,
                    2.0f32 * ((height as f32 + 0.5) / latent_h as f32) - 1.0,
                    2.0f32 * ((width as f32 + 0.5) / latent_w as f32) - 1.0,
                ];
                for (axis, value) in axes.iter().enumerate() {
                    for frequency in 0..8usize {
                        let inverse = 1.0f32 / 100.0f32.powf(frequency as f32 * 0.125);
                        let angle = 2.0f32 * std::f32::consts::PI * value * inverse;
                        let offset = row * halves + axis * 8 + frequency;
                        cosines[offset] = angle.cos();
                        sines[offset] = angle.sin();
                    }
                }
                row += 1;
            }
        }
    }

    for value in &mut cosines[row * halves..sequence * halves] {
        *value = 1.0;
    }
    (cosines, sines)
}

/// `prepare_input` (`:320-346`): de-normalize each latent and pad time to the
/// chunk by repeating the last frame, in `(t, h, w, c)` row order. `tile` is
/// `[channel, latent_time, height, width]`.
///
/// The tile's own time extent sets the row stride. A chunk tile carries
/// `CHUNK_LATENT_TIME` steps even when the whole latent is longer, so taking
/// the caller's total latent time here would read past the tile: that is what
/// a 107-frame request (27 latent steps, 7-step tiles) exposed.
fn prepare_input_rows(
    tile: &[f32],
    latent_h: usize,
    latent_w: usize,
    mean: &[f32],
    deviation: &[f32],
) -> Vec<f32> {
    let patches = CHUNK_LATENT_TIME * latent_h * latent_w;
    let mut rows = vec![0.0f32; patches * LATENT_CHANNELS];
    let mut row = 0;
    let frame = LATENT_CHANNELS * latent_h * latent_w;
    assert!(
        frame > 0 && !tile.is_empty() && tile.len().is_multiple_of(frame),
        "video VAE tile is not a whole number of latent frames"
    );
    let latent_time = tile.len() / frame;

    for time in 0..CHUNK_LATENT_TIME {
        let source_t = time.min(latent_time - 1);
        for height in 0..latent_h {
            for width in 0..latent_w {
                for channel in 0..LATENT_CHANNELS {
                    let source =
                        ((channel * latent_time + source_t) * latent_h + height) * latent_w + width;
                    rows[row] = tile[source] * deviation[channel] + mean[channel];
                    row += 1;
                }
            }
        }
    }
    rows
}

impl Decoder {
    /// Load the whole decoder for one tile shape and build its RoPE tables.
    pub fn load(
        directory: &Path,
        latent_height: usize,
        latent_width: usize,
        mean: Vec<f32>,
        deviation: Vec<f32>,
    ) -> Result<Decoder, VaeError> {
        if latent_height < 1 || latent_width < 1 {
            return fail("invalid video VAE latent shape");
        }
        let gpu = device(Gpu::create())?;
        let store = Store::open(directory, "video VAE")?;

        let patches = CHUNK_LATENT_TIME * latent_height * latent_width;
        let sequence = patches + SUFFIX;
        let (cosines, sines) = rope_tables(sequence, latent_height, latent_width);

        let parts = Parts {
            post_w: store.load(
                &gpu,
                "post_quant_conv.weight",
                &[LATENT_CHANNELS as u64, LATENT_CHANNELS as u64, 1, 1, 1],
                "post_quant_conv.weight",
            )?,
            post_b: f1(&store, &gpu, "post_quant_conv.bias", LATENT_CHANNELS as u64)?,
            embed_w: f2(
                &store,
                &gpu,
                "decoder.x_embedder.weight",
                HIDDEN as u64,
                LATENT_CHANNELS as u64,
            )?,
            embed_b: f1(&store, &gpu, "decoder.x_embedder.bias", HIDDEN as u64)?,
            registers: store.load(
                &gpu,
                "decoder.register_tokens",
                &[1, REGISTERS as u64, HIDDEN as u64],
                "decoder.register_tokens",
            )?,
            norm_out_w: f1(&store, &gpu, "decoder.norm_out.weight", HIDDEN as u64)?,
            norm_out_b: f1(&store, &gpu, "decoder.norm_out.bias", HIDDEN as u64)?,
            proj_w: f2(
                &store,
                &gpu,
                "decoder.proj_out.weight",
                OUTPUT_PATCH as u64,
                HIDDEN as u64,
            )?,
            proj_b: f1(&store, &gpu, "decoder.proj_out.bias", OUTPUT_PATCH as u64)?,
        };
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            blocks.push(load_block(&store, &gpu, index)?);
        }
        drop(store);

        let rope_cos = f32_alloc(&gpu, sequence * ROPE_HALF as usize)?;
        upload_f32(&gpu, &rope_cos, &cosines)?;
        let rope_sin = f32_alloc(&gpu, sequence * ROPE_HALF as usize)?;
        upload_f32(&gpu, &rope_sin, &sines)?;

        Ok(Decoder {
            gpu,
            parts,
            blocks,
            patches,
            sequence,
            rope_cos,
            rope_sin,
            mean,
            deviation,
        })
    }

    /// One block (`run_block`, `:423-456`) in the reference's op order.
    fn run_block(
        &self,
        block: &Block,
        hidden: &Tensor,
        buffers: &Buffers,
        rows: u32,
    ) -> Result<(), VaeError> {
        let gpu = &self.gpu;

        device(gpu.rms_norm_f32(
            &buffers.norm,
            hidden,
            &block.norm1,
            rows,
            HIDDEN as u32,
            EPSILON,
        ))?;
        device(gpu.linear(
            &buffers.qkv,
            &buffers.norm,
            &block.qkv_w,
            Some(&block.qkv_b),
            rows,
            HIDDEN as u32,
            (INNER * 3) as u32,
            Dtype::F32,
        ))?;
        device(gpu.video_qkv_rope_f32(
            &buffers.query,
            &buffers.key,
            &buffers.value,
            &buffers.qkv,
            &self.rope_cos,
            &self.rope_sin,
            rows,
            HEADS,
            HEAD_DIM,
            ROPE_HALF,
            EPSILON,
        ))?;
        device(gpu.sdpa_f32(
            &buffers.heads,
            &buffers.query,
            &buffers.key,
            &buffers.value,
            rows,
            HEADS,
            HEAD_DIM,
            1.0f32 / (HEAD_DIM as f32).sqrt(),
        ))?;
        device(gpu.linear(
            &buffers.branch,
            &buffers.heads,
            &block.out_w,
            Some(&block.out_b),
            rows,
            INNER as u32,
            HIDDEN as u32,
            Dtype::F32,
        ))?;
        device(gpu.scale_add_f32(
            hidden,
            hidden,
            &buffers.branch,
            &block.scale1,
            rows,
            HIDDEN as u32,
        ))?;

        device(gpu.rms_norm_f32(
            &buffers.norm,
            hidden,
            &block.norm2,
            rows,
            HIDDEN as u32,
            EPSILON,
        ))?;
        device(gpu.linear(
            &buffers.ff1,
            &buffers.norm,
            &block.w1,
            Some(&block.w1_b),
            rows,
            HIDDEN as u32,
            (FFN * 2) as u32,
            Dtype::F32,
        ))?;
        device(gpu.swiglu_f32(&buffers.activated, &buffers.ff1, rows, FFN as u32))?;
        device(gpu.linear(
            &buffers.branch,
            &buffers.activated,
            &block.w2,
            Some(&block.w2_b),
            rows,
            FFN as u32,
            HIDDEN as u32,
            Dtype::F32,
        ))?;
        device(gpu.scale_add_f32(
            hidden,
            hidden,
            &buffers.branch,
            &block.scale2,
            rows,
            HIDDEN as u32,
        ))?;
        Ok(())
    }

    /// `run_resident_tile` (`:533-578`): one tile through every block, then
    /// the output layer norm and projection, read back as F32 rows.
    fn run_tile(&self, scratch: &TileScratch, rows_input: &[f32]) -> Result<Vec<f32>, VaeError> {
        let gpu = &self.gpu;
        let patches = self.patches;
        let rows = u32::try_from(self.sequence)
            .map_err(|_| VaeError("video VAE sequence overflows".into()))?;

        let latent = f32_alloc(gpu, patches * LATENT_CHANNELS)?;
        upload_f32(gpu, &latent, rows_input)?;

        let buffers = &scratch.buffers;
        let (post, patch_hidden, zero) = (&scratch.post, &scratch.patch_hidden, &scratch.zero);

        device(gpu.linear(
            post,
            &latent,
            &self.parts.post_w,
            Some(&self.parts.post_b),
            patches as u32,
            LATENT_CHANNELS as u32,
            LATENT_CHANNELS as u32,
            Dtype::F32,
        ))?;
        device(gpu.linear(
            patch_hidden,
            post,
            &self.parts.embed_w,
            Some(&self.parts.embed_b),
            patches as u32,
            LATENT_CHANNELS as u32,
            HIDDEN as u32,
            Dtype::F32,
        ))?;
        device(gpu.copy_f32(&buffers.hidden, 0, patch_hidden, 0, patches * HIDDEN))?;
        device(gpu.copy_f32(
            &buffers.hidden,
            patches * HIDDEN,
            &self.parts.registers,
            0,
            REGISTERS * HIDDEN,
        ))?;
        device(gpu.copy_f32(
            &buffers.hidden,
            (patches + REGISTERS) * HIDDEN,
            zero,
            0,
            HIDDEN,
        ))?;

        for block in &self.blocks {
            self.run_block(block, &buffers.hidden, buffers, rows)?;
        }

        device(gpu.layer_norm_f32(
            &buffers.norm,
            &buffers.hidden,
            &self.parts.norm_out_w,
            &self.parts.norm_out_b,
            rows,
            HIDDEN as u32,
            EPSILON,
        ))?;
        device(gpu.linear(
            &buffers.projected,
            &buffers.norm,
            &self.parts.proj_w,
            Some(&self.parts.proj_b),
            rows,
            HIDDEN as u32,
            OUTPUT_PATCH as u32,
            Dtype::F32,
        ))?;
        device(gpu.sync())?;

        let mut bytes = vec![0u8; self.sequence * OUTPUT_PATCH * 4];
        device(gpu.read(&buffers.projected, &mut bytes))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }
}

/// The per-tile activation buffers (`allocate_activations`, `:397-421`).
struct Buffers {
    hidden: Tensor,
    norm: Tensor,
    qkv: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: Tensor,
    branch: Tensor,
    ff1: Tensor,
    activated: Tensor,
    projected: Tensor,
}

impl Buffers {
    fn new(gpu: &Gpu, sequence: usize) -> Result<Buffers, VaeError> {
        Ok(Buffers {
            hidden: f32_alloc(gpu, sequence * HIDDEN)?,
            norm: f32_alloc(gpu, sequence * HIDDEN)?,
            qkv: f32_alloc(gpu, sequence * INNER * 3)?,
            query: f32_alloc(gpu, sequence * INNER)?,
            key: f32_alloc(gpu, sequence * INNER)?,
            value: f32_alloc(gpu, sequence * INNER)?,
            heads: f32_alloc(gpu, sequence * INNER)?,
            branch: f32_alloc(gpu, sequence * HIDDEN)?,
            ff1: f32_alloc(gpu, sequence * FFN * 2)?,
            activated: f32_alloc(gpu, sequence * FFN)?,
            projected: f32_alloc(gpu, sequence * OUTPUT_PATCH)?,
        })
    }
}

/// Everything one tile needs that does not depend on the tile's latent:
/// `allocate_activations` (`h3_video_vae.c:397-421`), called once per decode
/// at `:1089`. The reference never touches the allocator inside the tile loop;
/// this port allocated and freed the whole set per tile, a device-wide drain
/// per tile for no arithmetic difference. Only the tile's own latent stays per
/// tile (`:343`). Deviation, named: the reference's `zero` suffix row is a
/// per-tile local (`:533-539`), a constant here, so it is allocated and
/// uploaded once.
struct TileScratch {
    buffers: Buffers,
    post: Tensor,
    patch_hidden: Tensor,
    zero: Tensor,
}

impl TileScratch {
    fn new(gpu: &Gpu, patches: usize, sequence: usize) -> Result<TileScratch, VaeError> {
        let zero = f32_alloc(gpu, HIDDEN)?;
        upload_f32(gpu, &zero, &[0.0f32; HIDDEN])?;

        Ok(TileScratch {
            buffers: Buffers::new(gpu, sequence)?,
            post: f32_alloc(gpu, patches * LATENT_CHANNELS)?,
            patch_hidden: f32_alloc(gpu, patches * HIDDEN)?,
            zero,
        })
    }
}

/// `unpack_frames` (`:579-656`): the 3072 output values of one row are
/// `(channel, within_t, within_y, within_x)`; frames are cut out of them by
/// latent time 4 and 16-pixel blocks, de-normalized and clamped.
fn unpack_frames(
    projected: &[f32],
    latent_h: usize,
    latent_w: usize,
    total_frames: usize,
    first_frame: usize,
    frame_count: usize,
) -> Result<Vec<f32>, VaeError> {
    if frame_count < 1 || frame_count > total_frames || first_frame > total_frames - frame_count {
        return fail("invalid video VAE output frame range");
    }

    let pixel_h = latent_h * PATCH;
    let pixel_w = latent_w * PATCH;
    let mut rgb = vec![0.0f32; frame_count * pixel_h * pixel_w * 3];

    for output_frame in 0..frame_count {
        let frame = first_frame + output_frame;
        let mut decoded_t = frame + FRAME_OFFSET;
        if total_frames == FIRST_CHUNK_FRAMES && frame >= 17 {
            decoded_t += 3;
        }
        let patch_t = decoded_t / 4;
        let within_t = decoded_t % 4;

        for y in 0..pixel_h {
            let patch_y = y / PATCH;
            let within_y = y % PATCH;
            for x in 0..pixel_w {
                let patch_x = x / PATCH;
                let within_x = x % PATCH;
                let patch = (patch_t * latent_h + patch_y) * latent_w + patch_x;
                for channel in 0..3 {
                    let component =
                        ((channel * 4 + within_t) * PATCH + within_y) * PATCH + within_x;
                    let value =
                        projected[patch * OUTPUT_PATCH + component] * STD[channel] + MEAN[channel];
                    let destination = ((output_frame * pixel_h + y) * pixel_w + x) * 3 + channel;
                    rgb[destination] = value.clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(rgb)
}

/// `stitch_tiles` (`:761-830`): every tile keeps its pixels up to the overlap
/// of the next one, which is then blended in with `alpha = position / overlap`.
fn stitch_tiles(
    tiles: &[Vec<f32>],
    y_axis: &TileAxis,
    x_axis: &TileAxis,
    frame_count: usize,
) -> Frames {
    let full_h = y_axis.extent();
    let full_w = x_axis.extent();
    let mut rgb = vec![0.0f32; frame_count * full_h * full_w * 3];

    for tile_y in 0..y_axis.count() {
        for tile_x in 0..x_axis.count() {
            let index = tile_y * x_axis.count() + tile_x;
            let current = &tiles[index];
            let above = (tile_y > 0).then(|| &tiles[index - x_axis.count()]);
            let left = (tile_x > 0).then(|| &tiles[index - 1]);
            let overlap_y = if tile_y > 0 {
                y_axis.overlaps[tile_y - 1]
            } else {
                0
            };
            let overlap_x = if tile_x > 0 {
                x_axis.overlaps[tile_x - 1]
            } else {
                0
            };
            let keep_h = y_axis.length
                - if tile_y + 1 < y_axis.count() {
                    y_axis.overlaps[tile_y]
                } else {
                    0
                };
            let keep_w = x_axis.length
                - if tile_x + 1 < x_axis.count() {
                    x_axis.overlaps[tile_x]
                } else {
                    0
                };

            for frame in 0..frame_count {
                for y in 0..keep_h {
                    for x in 0..keep_w {
                        for channel in 0..3 {
                            let local =
                                ((frame * y_axis.length + y) * x_axis.length + x) * 3 + channel;
                            let mut value = current[local];
                            if let Some(above) = above {
                                if y < overlap_y {
                                    let top = ((frame * y_axis.length
                                        + (y_axis.length - overlap_y + y))
                                        * x_axis.length
                                        + x)
                                        * 3
                                        + channel;
                                    let alpha = y as f32 / overlap_y as f32;
                                    value = above[top] * (1.0 - alpha) + value * alpha;
                                }
                            }
                            if let Some(left) = left {
                                if x < overlap_x {
                                    let prior = ((frame * y_axis.length + y) * x_axis.length
                                        + (x_axis.length - overlap_x + x))
                                        * 3
                                        + channel;
                                    let alpha = x as f32 / overlap_x as f32;
                                    value = left[prior] * (1.0 - alpha) + value * alpha;
                                }
                            }
                            let destination = ((frame * full_h + (y_axis.starts[tile_y] + y))
                                * full_w
                                + (x_axis.starts[tile_x] + x))
                                * 3
                                + channel;
                            rgb[destination] = value;
                        }
                    }
                }
            }
        }
    }

    Frames {
        frames: frame_count,
        height: full_h,
        width: full_w,
        rgb,
    }
}

/// `extract_latent_tile` (`:733-759`): a `[7, tile_h, tile_w, 24]` slab out of
/// the full `[24, T, H, W]` latent, row by row.
#[allow(clippy::too_many_arguments)]
fn extract_latent_tile(
    latent: &[f32],
    latent_time: usize,
    latent_h: usize,
    latent_w: usize,
    start_t: usize,
    start_y: usize,
    start_x: usize,
    tile_h: usize,
    tile_w: usize,
) -> Vec<f32> {
    let mut tile = vec![0.0f32; LATENT_CHANNELS * CHUNK_LATENT_TIME * tile_h * tile_w];

    for channel in 0..LATENT_CHANNELS {
        for time in 0..CHUNK_LATENT_TIME {
            for y in 0..tile_h {
                let source = ((channel * latent_time + (start_t + time)) * latent_h
                    + (start_y + y))
                    * latent_w
                    + start_x;
                let destination = ((channel * CHUNK_LATENT_TIME + time) * tile_h + y) * tile_w;
                tile[destination..destination + tile_w]
                    .copy_from_slice(&latent[source..source + tile_w]);
            }
        }
    }
    tile
}

/// One chunk (`decoder_decode_chunk`, `:831-951`): every tile through the
/// graph, then one stitched frame set.
#[allow(clippy::too_many_arguments)]
fn decode_chunk(
    decoder: &Decoder,
    scratch: &TileScratch,
    latent: &[f32],
    latent_time: usize,
    chunk: usize,
    full_h: usize,
    full_w: usize,
    y_axis: &TileAxis,
    x_axis: &TileAxis,
    total_frames: usize,
    selected: Option<usize>,
) -> Result<Frames, VaeError> {
    let tile_h = y_axis.length / SPATIAL_RATIO;
    let tile_w = x_axis.length / SPATIAL_RATIO;
    let mut tiles = Vec::with_capacity(y_axis.count() * x_axis.count());
    // `h3_video_vae.c:850`: a preview asks for one frame, so only that frame
    // is unpacked per tile (`:874-877`) instead of the whole chunk.
    let (first_frame, frame_count) = match selected {
        Some(frame) => (frame, 1),
        None => (
            0,
            if total_frames == FIRST_CHUNK_FRAMES {
                FIRST_CHUNK_FRAMES
            } else {
                total_frames
            },
        ),
    };

    for tile_y in 0..y_axis.count() {
        for tile_x in 0..x_axis.count() {
            let input = extract_latent_tile(
                latent,
                latent_time,
                full_h,
                full_w,
                chunk * CHUNK_OVERLAP,
                y_axis.starts[tile_y] / SPATIAL_RATIO,
                x_axis.starts[tile_x] / SPATIAL_RATIO,
                tile_h,
                tile_w,
            );
            let rows =
                prepare_input_rows(&input, tile_h, tile_w, &decoder.mean, &decoder.deviation);
            let projected = decoder.run_tile(scratch, &rows)?;
            tiles.push(unpack_frames(
                &projected,
                tile_h,
                tile_w,
                total_frames,
                first_frame,
                frame_count,
            )?);
        }
    }

    Ok(stitch_tiles(&tiles, y_axis, x_axis, frame_count))
}

/// `load_latent_normalization` (`:271-318`): `latents_mean` and `latents_std`
/// from the parent of the shard directory, parsed at f32 precision. This is
/// the video VAE's own convention: the audio VAE reads them from its own
/// directory (`h3_audio_vae.c:303-309`).
pub fn latent_normalization(weight_directory: &Path) -> Result<(Vec<f32>, Vec<f32>), VaeError> {
    let parent = weight_directory
        .parent()
        .ok_or_else(|| VaeError("video VAE directory has no parent".into()))?;
    let path = parent.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| VaeError(format!("cannot read {}: {error}", path.display())))?;
    let mean = float_array(&text, "latents_mean", LATENT_CHANNELS)?;
    let deviation = float_array(&text, "latents_std", LATENT_CHANNELS)?;

    if deviation.iter().any(|value| *value <= 0.0) {
        return fail("video VAE latent standard deviation is invalid");
    }
    Ok((mean, deviation))
}

fn float_array(text: &str, key: &str, count: usize) -> Result<Vec<f32>, VaeError> {
    let pattern = format!("\"{key}\"");
    let start = text
        .find(&pattern)
        .ok_or_else(|| VaeError(format!("video VAE config is missing {key}")))?;
    let open = text[start..]
        .find('[')
        .map(|offset| start + offset)
        .ok_or_else(|| VaeError(format!("video VAE config is missing {key}")))?;
    let close = text[open..]
        .find(']')
        .map(|offset| open + offset)
        .ok_or_else(|| VaeError(format!("video VAE config has malformed {key}")))?;

    let mut values = Vec::with_capacity(count);
    for token in text[open + 1..close].split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: f32 = token
            .parse()
            .map_err(|_| VaeError(format!("video VAE config has malformed {key}")))?;
        if !value.is_finite() {
            return fail(format!("video VAE config has malformed {key}"));
        }
        values.push(value);
    }
    if values.len() != count {
        return fail(format!(
            "video VAE config has {} {key} values, expected {count}",
            values.len()
        ));
    }
    Ok(values)
}

/// The resident preview decoder (`h3_video_vae_decoder_load`,
/// `h3_video_vae.c:891-951`): loaded once and reused for every denoising
/// step, which is what the reference's `h3_acquire_video_decoder` cache gives
/// the preview.
pub struct PreviewDecoder {
    decoder: Decoder,
    latent_h: usize,
    latent_w: usize,
    y_axis: TileAxis,
    x_axis: TileAxis,
}

impl PreviewDecoder {
    pub fn load(
        weight_directory: &Path,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<PreviewDecoder, VaeError> {
        if latent_h < 1 || latent_w < 1 {
            return fail("invalid video VAE latent shape");
        }
        let pixel_h = latent_h * SPATIAL_RATIO;
        let pixel_w = latent_w * SPATIAL_RATIO;
        let tile_pixels = configured_tile_pixels(pixel_h, pixel_w);
        let y_axis = tile_axis(pixel_h, tile_pixels)?;
        let x_axis = tile_axis(pixel_w, tile_pixels)?;
        let (mean, deviation) = latent_normalization(weight_directory)?;
        let decoder = Decoder::load(
            weight_directory,
            y_axis.length / SPATIAL_RATIO,
            x_axis.length / SPATIAL_RATIO,
            mean,
            deviation,
        )?;

        Ok(PreviewDecoder {
            decoder,
            latent_h,
            latent_w,
            y_axis,
            x_axis,
        })
    }

    /// `h3_video_vae_decoder_preview` (`:953-975`): the middle chunk, its
    /// `FIRST_CHUNK_FRAMES / 2` frame, and that frame's index in the clip.
    pub fn decode(&self, latent: &[f32], latent_time: usize) -> Result<(Frames, usize), VaeError> {
        let (chunks, output_frames) = chunk_plan(latent_time)?;
        if latent_time < CHUNK_LATENT_TIME {
            return fail("video VAE preview needs a released chunk count");
        }
        let expected = LATENT_CHANNELS * latent_time * self.latent_h * self.latent_w;
        if latent.len() != expected {
            return fail(format!(
                "video VAE latent has {} values, expected {expected}",
                latent.len()
            ));
        }

        let chunk = chunks / 2;
        let local_frame = FIRST_CHUNK_FRAMES / 2;
        let mut global_frame = chunk * CHUNK_STRIDE + local_frame;
        if global_frame >= output_frames {
            global_frame = output_frames - 1;
        }
        let scratch = TileScratch::new(
            &self.decoder.gpu,
            self.decoder.patches,
            self.decoder.sequence,
        )?;
        let frames = decode_chunk(
            &self.decoder,
            &scratch,
            latent,
            latent_time,
            chunk,
            self.latent_h,
            self.latent_w,
            &self.y_axis,
            &self.x_axis,
            FIRST_CHUNK_FRAMES,
            Some(local_frame),
        )?;

        Ok((frames, global_frame))
    }
}

/// `h3_video_vae_decode` (`:1184-1252`): the whole video, one chunk at a time,
/// with the 5-frame temporal overlap blended by `frame / 5`.
pub fn decode(
    weight_directory: &Path,
    latent: &[f32],
    latent_time: usize,
    latent_h: usize,
    latent_w: usize,
) -> Result<Frames, VaeError> {
    let (chunks, output_frames) = chunk_plan(latent_time)?;
    if latent_h < 1 || latent_w < 1 {
        return fail("invalid video VAE latent shape");
    }
    let pixel_h = latent_h * SPATIAL_RATIO;
    let pixel_w = latent_w * SPATIAL_RATIO;
    let tile_pixels = configured_tile_pixels(pixel_h, pixel_w);
    let y_axis = tile_axis(pixel_h, tile_pixels)?;
    let x_axis = tile_axis(pixel_w, tile_pixels)?;
    if latent_time == 2 && (y_axis.count() > 1 || x_axis.count() > 1) {
        return fail("the two-token diagnostic VAE path supports one spatial tile");
    }
    let expected = LATENT_CHANNELS * latent_time * latent_h * latent_w;
    if latent.len() != expected {
        return fail(format!(
            "video VAE latent has {} values, expected {expected}",
            latent.len()
        ));
    }

    let (mean, deviation) = latent_normalization(weight_directory)?;
    let decoder = Decoder::load(
        weight_directory,
        y_axis.length / SPATIAL_RATIO,
        x_axis.length / SPATIAL_RATIO,
        mean,
        deviation,
    )?;

    // One activation set for every tile of every chunk (`:1089`).
    let scratch = TileScratch::new(&decoder.gpu, decoder.patches, decoder.sequence)?;

    // `:1226-1252`: the two-token path hands the whole latent, already padded
    // to the chunk, straight to the graph; one spatial tile, five frames, no
    // temporal blend and no tile extraction.
    if latent_time == 2 {
        let rows = prepare_input_rows(
            latent,
            latent_h,
            latent_w,
            &decoder.mean,
            &decoder.deviation,
        );
        let projected = decoder.run_tile(&scratch, &rows)?;
        let rgb = unpack_frames(
            &projected,
            latent_h,
            latent_w,
            output_frames,
            0,
            output_frames,
        )?;
        return Ok(Frames {
            frames: output_frames,
            height: pixel_h,
            width: pixel_w,
            rgb,
        });
    }

    let frame_elements = pixel_h * pixel_w * 3;
    let mut final_rgb = vec![0.0f32; output_frames * frame_elements];
    let mut overlap = vec![0.0f32; CHUNK_OVERLAP * frame_elements];

    for chunk in 0..chunks {
        let mut decoded = decode_chunk(
            &decoder,
            &scratch,
            latent,
            latent_time,
            chunk,
            latent_h,
            latent_w,
            &y_axis,
            &x_axis,
            FIRST_CHUNK_FRAMES,
            None,
        )?;
        // `:1105-1113`: the repeated 5 frames fade from the previous chunk's
        // tail into this chunk's own prediction.
        if chunk > 0 {
            for frame in 0..CHUNK_OVERLAP {
                let alpha = frame as f32 / CHUNK_OVERLAP as f32;
                let base = frame * frame_elements;
                for index in 0..frame_elements {
                    decoded.rgb[base + index] =
                        overlap[base + index] * (1.0 - alpha) + decoded.rgb[base + index] * alpha;
                }
            }
        }
        let base = chunk * CHUNK_STRIDE * frame_elements;
        final_rgb[base..base + CHUNK_STRIDE * frame_elements]
            .copy_from_slice(&decoded.rgb[..CHUNK_STRIDE * frame_elements]);
        overlap.copy_from_slice(
            &decoded.rgb
                [CHUNK_STRIDE * frame_elements..(CHUNK_STRIDE + CHUNK_OVERLAP) * frame_elements],
        );
    }
    let base = chunks * CHUNK_STRIDE * frame_elements;
    final_rgb[base..base + CHUNK_OVERLAP * frame_elements].copy_from_slice(&overlap);

    Ok(Frames {
        frames: output_frames,
        height: pixel_h,
        width: pixel_w,
        rgb: final_rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_plan_matches_the_reference_at_the_released_shapes() {
        let single = tile_axis(256, configured_tile_pixels(256, 256)).expect("256");
        assert_eq!(single.count(), 1);
        assert_eq!(single.length, 256);
        assert!(single.overlaps.is_empty());

        let wide = configured_tile_pixels(256, 512);
        assert_eq!(wide, 288);
        let axis = tile_axis(512, wide).expect("wide");
        assert_eq!(axis.count(), 2);
        assert_eq!(axis.overlaps, vec![64]);
        assert_eq!(axis.starts, vec![0, wide - 64]);
        assert_eq!(axis.extent(), 512);
    }

    #[test]
    fn overlap_surplus_is_spread_in_sixteen_pixel_units() {
        let axis = tile_axis(768, 256).expect("768");
        assert_eq!(axis.count(), 4);
        assert_eq!(axis.extent(), 768);
        assert_eq!(axis.overlaps.len(), 3);
        assert!(axis.overlaps.iter().all(|value| *value % 16 == 0));
        assert!(axis.overlaps.iter().all(|value| *value >= TILE_OVERLAP_MIN));
    }

    #[test]
    fn rope_rows_cover_the_patches_then_hold_identity() {
        let (cosines, sines) = rope_tables(7 * 2 * 2 + SUFFIX, 2, 2);
        let halves = ROPE_HALF as usize;
        assert_eq!(cosines.len(), (7 * 2 * 2 + SUFFIX) * halves);
        // Row 0 is (t, h, w) = (0, 0, 0), all three axes off centre, so its
        // angles are non-zero; only the register rows hold the identity.
        assert!(cosines[0] < 1.0);
        let suffix = 7 * 2 * 2 * halves;
        assert!(cosines[suffix..].iter().all(|value| *value == 1.0));
        assert!(sines[suffix..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn pads_time_by_repeating_the_last_latent_frame() {
        let input = vec![1.0f32; LATENT_CHANNELS * 2 * 1 * 1];
        let rows = prepare_input_rows(
            &input,
            1,
            1,
            &[0.0; LATENT_CHANNELS],
            &[1.0; LATENT_CHANNELS],
        );
        assert_eq!(rows.len(), CHUNK_LATENT_TIME * LATENT_CHANNELS);
        assert!(rows.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn a_chunk_tile_is_read_with_its_own_time_stride() {
        // Regression: a 107-frame request denoises to 27 latent steps and
        // decodes in 7-step chunk tiles. Reading a tile with the whole
        // latent's time stride indexed past its end.
        let (latent_time, height, width) = (12usize, 2usize, 2usize);
        let latent: Vec<f32> = (0..LATENT_CHANNELS * latent_time * height * width)
            .map(|value| value as f32)
            .collect();
        let tile = extract_latent_tile(&latent, latent_time, height, width, 0, 0, 0, height, width);
        let rows = prepare_input_rows(
            &tile,
            height,
            width,
            &[0.0; LATENT_CHANNELS],
            &[1.0; LATENT_CHANNELS],
        );

        assert_eq!(
            rows.len(),
            CHUNK_LATENT_TIME * height * width * LATENT_CHANNELS
        );
        // Row 0 is (t, h, w, c) = (0, 0, 0, 0), the tile's own first value.
        assert_eq!(rows[0], tile[0]);
        // The chunk's last step reads the tile's seventh latent frame, which
        // in this shape is source value 24, not the zero past the tile's end.
        let step = height * width * LATENT_CHANNELS;
        let last = (CHUNK_LATENT_TIME - 1) * step;
        assert_eq!(rows[last], latent[(CHUNK_LATENT_TIME - 1) * height * width]);
    }

    #[test]
    fn accepts_only_released_chunk_counts() {
        assert_eq!(chunk_plan(2).expect("two-token"), (0, 5));
        assert_eq!(chunk_plan(7).expect("one chunk"), (1, 22));
        assert_eq!(chunk_plan(12).expect("two chunks"), (2, 39));
        assert!(chunk_plan(3).is_err());
        assert!(chunk_plan(9).is_err());
    }

    #[test]
    fn patch_components_land_on_the_reference_offsets() {
        let mut projected = vec![0.0f32; 7 * OUTPUT_PATCH];
        for (index, value) in projected.iter_mut().enumerate() {
            *value = index as f32 * 0.001;
        }
        let rgb = unpack_frames(&projected, 1, 1, 1, 0, 1).expect("one frame");
        // frame 0 -> decoded_t 3 -> within_t 3, patch 0, pixel (0, 0):
        // component = ((0 * 4 + 3) * 16 + 0) * 16 + 0 = 768.
        let expected = 768.0f32 * 0.001 * STD[0] + MEAN[0];
        assert!((rgb[0] - expected).abs() < 1e-6);
    }
}
