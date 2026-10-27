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
//! - `h3_dit.c:2882-2985` `h3_dit_denoise_euler_preview`, host sampler: one
//!   forward per selected step, the reuse window (`:2537-2554`) and the
//!   velocity extrapolation (`:2512-2535`), with `h3_host.c:675-683` doing
//!   `sample += (sigma - sigma_next) * velocity`.
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
//! - Token reduction calls `enter_token_reduction`/`leave_token_reduction`
//!   and then the block's own AdaLN, which is exactly what the reference's
//!   fused `h3_gpu_token_pool_adaln_bf16`/`h3_gpu_token_expand_adaln_bf16`
//!   pair computes (`h3_dit.c:3400-3434`), so the reference's
//!   `H3_DISABLE_FUSED_TOKEN_POOL_ADALN`/`_ADALN` switches select between two
//!   spellings of the same arithmetic and have no analogue here.
//! - The reduced pass keeps its snapshot, baseline and reduced stream in their
//!   own tensors; the reference aliases them into `qkv` and
//!   `attention_output` (`h3_dit.c:1506-1527`). Buffer identity is a memory
//!   choice, not arithmetic, and the hashes are gated against the reference.
//! - `h3_gpu_adaln_linear_bf16` (`:1195-1226`) also writes an `inverse` buffer
//!   that no later call reads; the effective math is `adaln` then `linear`,
//!   which is what the final head here does.
//! - int8 and NAX variants are not ported: CUDA reports
//!   `h3_gpu_has_int8_mlp`/`h3_gpu_has_nax_mlp` as 0 (`h3_gpu_cuda.cu:332-334`).
//!   BF16 block weights use `Streamer`; the refiner remains resident.
//! - The device Euler sampler (`h3_dit.c:2613-2787` `denoise_euler_gpu`) is
//!   not ported. It is the reference's default only on M5
//!   (`h3_dit.c:2572-2578` returns `h3_gpu_is_m5`, 0 on CUDA per
//!   `h3_gpu_cuda.cu:332`); on this backend the reference runs the host
//!   sampler implemented here unless `H3_GPU_SAMPLER=1` overrides it.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use crate::minimax_h3::host::{self, Schedule};
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

/// `H3_DIT_BLOCKS` (`h3_dit.c:15-30`): the reference's block count. The
/// token-reduction window is validated against it even when fewer blocks load.
pub const DIT_BLOCKS: usize = 50;

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

/// A refiner block's eight BF16 matrices plus its norm weights
/// (`h3_dit.c:502-525`). The refiner is two blocks and never streams.
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

/// The four matrices that stream from the checkpoint, one slot per block
/// (`h3_dit.c:600-621`, `:2223-2240`).
struct BlockMatrices {
    qkv: Tensor,
    out: Tensor,
    fc1: Tensor,
    fc2: Tensor,
}

/// A core block's norms. Both residency modes keep them device-resident: they
/// are a few kilobytes each and every block needs them (`h3_dit.c:1251-1262`).
struct BlockNorms {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

/// One of the 50 core blocks. `matrices` is `None` when the block streams;
/// the two slots then hold whichever blocks are current and next.
struct CoreBlock {
    norms: BlockNorms,
    matrices: Option<BlockMatrices>,
}

/// How the core's 50 blocks live on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMode {
    /// Every block resident, as the reference's default does.
    Resident,
    /// Two matrix slots, read from the checkpoint per block.
    Stream,
}

/// Everything one reduced pass needs (`h3_dit.c:149-177`, `:440-470`): the
/// pair maps, the reduced AdaLN and RoPE tables, and the buffers the pool
/// snapshots into. The reference aliases these buffers with its activations
/// (`:1506-1527`); the port keeps them separate, which changes no arithmetic.
struct ReductionState {
    window: ReductionWindow,
    /// Reduced packed rows: text, condition and audio rows plus one row per
    /// video pair (`h3_dit.c:470`).
    rows: u32,
    /// Reduced video rows that pooled two rows, one baseline each (`:469`).
    baseline_rows: u32,
    pairs: Tensor,
    baseline_indices: Tensor,
    parents: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    row_maps: Vec<Tensor>,
    /// The reduced residual stream a pooled block reads and writes.
    hidden: Tensor,
    /// Every full row the pool replaced, and the pooled value it replaced
    /// (`:1506-1527`).
    original: Tensor,
    baseline: Tensor,
}

/// What one core block reads and writes: the full residual stream or the
/// reduced one inside the pairing window (`h3_dit.c:1893-1897`).
struct Active<'a> {
    rows: u32,
    hidden: &'a Tensor,
    row_map: &'a Tensor,
    rope_cos: &'a Tensor,
    rope_sin: &'a Tensor,
}

/// One checkpoint range, owned so the reader thread does not borrow `Weights`.
#[derive(Clone)]
struct StreamSource {
    path: std::path::PathBuf,
    offset: u64,
    bytes: usize,
}

/// A block being read by the prefetch thread.
struct PendingRead {
    layer: usize,
    slot: usize,
    join: std::thread::JoinHandle<Result<[Vec<u8>; 4], String>>,
}

/// The reference's two-slot stream ring (`h3_dit.c:625-690`, `:2223-2330`).
///
/// The current block sits in `active`; the other slot is free, so the next
/// block's four matrices can be read into it while the current block runs.
/// After the last block the ring wraps to block 0, which is how the next
/// denoise step starts without a read stall.
struct Streamer {
    slots: [BlockMatrices; 2],
    sources: Vec<[StreamSource; 4]>,
    active: usize,
    active_layer: usize,
    pending: Option<PendingRead>,
    slot_bytes: u64,
}

impl Streamer {
    fn new(gpu: &Gpu, weights: &Weights, config: &Config) -> Result<Streamer, DitError> {
        let hidden = config.hidden as usize;
        let inner = config.inner() as usize;
        let ffn = config.ffn as usize;
        let slot = || -> Result<BlockMatrices, DitError> {
            Ok(BlockMatrices {
                qkv: gpu_alloc(gpu, Dtype::Bf16, 3 * inner * hidden)?,
                out: gpu_alloc(gpu, Dtype::Bf16, hidden * inner)?,
                fc1: gpu_alloc(gpu, Dtype::Bf16, 2 * ffn * hidden)?,
                fc2: gpu_alloc(gpu, Dtype::Bf16, hidden * ffn)?,
            })
        };
        let slots = [slot()?, slot()?];

        let mut sources = Vec::with_capacity(config.layers as usize);
        let mut slot_bytes = 0u64;
        for layer in 0..config.layers {
            let four = [
                stream_source(weights, Role::BlockQkv(layer))?,
                stream_source(weights, Role::BlockOut(layer))?,
                stream_source(weights, Role::BlockFc1(layer))?,
                stream_source(weights, Role::BlockFc2(layer))?,
            ];
            // Every block carries the same four matrices.
            slot_bytes = four.iter().map(|source| source.bytes as u64).sum();
            sources.push(four);
        }

        Ok(Streamer {
            slots,
            sources,
            active: 0,
            // No block is loaded until `ensure(0)` reads one: 0 here would
            // make the priming call a no-op and run block 0 on a blank slot.
            active_layer: usize::MAX,
            pending: None,
            slot_bytes,
        })
    }

    /// Bytes one slot holds, four matrices.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn current(&self) -> &BlockMatrices {
        &self.slots[self.active]
    }

    /// Start reading `layer` into the free slot.
    fn start(&mut self, layer: usize) -> Result<(), DitError> {
        self.finish_pending()?;
        let slot = 1 - self.active;
        let sources = self.sources[layer].clone();
        self.pending = Some(PendingRead {
            layer,
            slot,
            join: std::thread::spawn(move || read_block(&sources)),
        });
        Ok(())
    }

    /// Join the reader, upload into its slot and make it current.
    fn finish(&mut self, gpu: &Gpu) -> Result<(), DitError> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| DitError::Gpu("stream finish without a read".into()))?;
        let data = pending
            .join
            .join()
            .map_err(|_| DitError::Gpu("stream reader panicked".into()))?
            .map_err(|detail| DitError::Gpu(format!("stream block {}: {detail}", pending.layer)))?;

        let slot = &self.slots[pending.slot];
        device(gpu.upload(&slot.qkv, &data[0]))?;
        device(gpu.upload(&slot.out, &data[1]))?;
        device(gpu.upload(&slot.fc1, &data[2]))?;
        device(gpu.upload(&slot.fc2, &data[3]))?;

        self.active = pending.slot;
        self.active_layer = pending.layer;
        Ok(())
    }

    fn finish_pending(&mut self) -> Result<(), DitError> {
        if self.pending.is_some() {
            return Err(DitError::Gpu(
                "stream reader still running when the ring advanced".into(),
            ));
        }
        Ok(())
    }

    /// Make `layer` the current block, reading it now if the ring does not
    /// already hold it.
    fn ensure(&mut self, gpu: &Gpu, layer: usize) -> Result<(), DitError> {
        if self.active_layer == layer && self.pending.is_none() {
            return Ok(());
        }
        self.start(layer)?;
        self.finish(gpu)
    }
}

/// One role as a checkpoint range, refusing a non-BF16 matrix.
fn stream_source(weights: &Weights, role: Role) -> Result<StreamSource, DitError> {
    let span = weights
        .span(role)
        .map_err(|error| DitError::Weights(error.to_string()))?;
    if span.dtype != crate::minimax_h3::safetensors::Dtype::Bf16 {
        return Err(DitError::Weights(format!(
            "{role:?} is not BF16 and cannot stream"
        )));
    }
    Ok(StreamSource {
        path: span.path.to_path_buf(),
        offset: span.file_offset,
        bytes: usize::try_from(span.bytes)
            .map_err(|_| DitError::Weights("span too wide".into()))?,
    })
}

/// Read one block's four matrices in ascending (path, offset) order — the
/// reference's `compare_stream_sources` (`h3_dit.c:570-577`) — and drop each
/// range from the page cache afterwards, as its streamed reader does
/// (`h3_gpu_cuda.cu:410-412`).
fn read_block(sources: &[StreamSource; 4]) -> Result<[Vec<u8>; 4], String> {
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|left, right| {
        sources[*left]
            .path
            .cmp(&sources[*right].path)
            .then(sources[*left].offset.cmp(&sources[*right].offset))
    });

    let mut data: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
    for index in order {
        let source = &sources[index];
        let file = std::fs::File::open(&source.path)
            .map_err(|error| format!("open {}: {error}", source.path.display()))?;
        let mut buffer = vec![0u8; source.bytes];
        read_at(&file, &mut buffer, source.offset)
            .map_err(|error| format!("read {}: {error}", source.path.display()))?;
        drop_cache(&file, source.offset, source.bytes);
        data[index] = buffer;
    }
    Ok(data)
}

fn read_at(file: &std::fs::File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buffer, offset)
    }
    #[cfg(not(unix))]
    {
        let _ = (file, buffer, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "streaming needs pread",
        ))
    }
}

fn drop_cache(file: &std::fs::File, offset: u64, bytes: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        // Advisory: the read data is on the device, the file's pages are not
        // needed, and on unified memory they would compete with the weights.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as i64,
                bytes as i64,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, bytes);
    }
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
/// selector per sequence row, `time_row * MODALITIES + tag`. The text rows take
/// the presentation's tags, because an anchored prompt carries vision spans
/// tagged 0 among prompt rows tagged 1 (`h3_multimodal.c:173`, `:182`), and the
/// reference feeds them to this map (`h3_dit.c:1028`).
fn row_map(
    tree: &TimeRows,
    step: usize,
    sequence: &Sequence,
    text_tags: Option<&[u8]>,
) -> Result<Vec<u32>, DitError> {
    // `h3_dit_schedule.c:440`: the tags cover the text rows exactly.
    let text_rows = sequence.signature[0] as usize;
    if matches!(text_tags, Some(tags) if tags.len() != text_rows) {
        return Err(DitError::Geometry {
            detail: format!("presentation tags do not cover the {text_rows} text rows"),
        });
    }

    let mut rows = vec![0u32; sequence.rows()];
    let mut text_index = 0usize;

    for segment in &sequence.segments {
        let (time_row, tag) = match segment.kind {
            SegmentKind::Text => {
                let time_row = tree.video[step];
                for cell in &mut rows[segment.start..segment.stop] {
                    // `h3_dit_schedule.c:452-454`: the presented tag, else text.
                    let tag = text_tags.map_or(1, |tags| tags[text_index]);
                    if u32::from(tag) >= MODALITIES {
                        return Err(DitError::Geometry {
                            detail: "a presentation tag is outside the modality count".into(),
                        });
                    }
                    *cell = time_row * MODALITIES + u32::from(tag);
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

/// Whether the reference's video-token pairing runs (`h3_dit.c:380-382`).
///
/// The reference's decision is `requested` OR the environment: `H3_TOKEN_REDUCTION`
/// set to anything but an empty string or "0" enables it on its own. The window
/// itself always comes from the environment (`:383-439`), never from a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// `configure_token_reduction(dit, 0, ...)`: the environment alone decides.
    Environment,
    /// `configure_token_reduction(dit, 1, ...)`: on unless a later
    /// `H3_DISABLE_TOKEN_REDUCTION` skips it for one forward.
    Enabled,
}

/// The resolved window `configure_token_reduction` keeps on the DiT
/// (`h3_dit.c:462-467`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReductionWindow {
    /// First block that runs on the reduced video grid (`:378`, default 4).
    pub begin: u32,
    /// First block back on the full grid (`:379`, default 30).
    pub end: u32,
    /// Steps that take the early window (`:413-415`, 10 below a 40-block end).
    pub early_steps: u32,
    /// `end` for those steps (`:414-415`, 40 below a 40-block end).
    pub early_end: u32,
    /// Scale on the restored delta (`:434`, default 1.0).
    pub scale: f32,
}

impl ReductionWindow {
    /// The `end` this step uses (`h3_dit.c:2168-2170`).
    pub fn end_for_step(&self, step: usize) -> u32 {
        if self.early_steps != 0 && (step as u32) < self.early_steps {
            self.early_end
        } else {
            self.end
        }
    }
}

/// The per-step preview hook (`h3_dit_preview`): the completed step index and
/// the video latent after that step's Euler update.
pub type PreviewHook<'a> = &'a mut dyn FnMut(usize, &[f32]) -> Result<(), DitError>;

/// `h3_dit.c:15-30`, `H3_DIT_BLOCKS`.
fn block_count() -> u32 {
    DIT_BLOCKS as u32
}

/// Parse one `BEGIN:END` environment value the way `:383-405` does: two
/// unsigned decimal fields and nothing else.
fn parse_pair(value: &str, message: &str) -> Result<(u32, u32), DitError> {
    let malformed = || DitError::Geometry {
        detail: message.to_string(),
    };
    let (first, second) = value.split_once(':').ok_or_else(malformed)?;
    let begin: u32 = first.parse().map_err(|_| malformed())?;
    let end: u32 = second.parse().map_err(|_| malformed())?;
    Ok((begin, end))
}

impl Reduction {
    /// `h3_dit.c:378-439`: the environment spelling, its defaults and its
    /// rejections, with the geometry check left to the caller.
    pub fn window(self) -> Result<Option<ReductionWindow>, DitError> {
        let environment = std::env::var("H3_TOKEN_REDUCTION").ok();
        let enabled = matches!(
            environment.as_deref(),
            Some(value) if !value.is_empty() && value != "0"
        );
        if self == Reduction::Environment && !enabled {
            return Ok(None);
        }

        let mut begin = 4u32;
        let mut end = 30u32;
        if let Some(range) = std::env::var("H3_TOKEN_REDUCTION_BLOCKS")
            .ok()
            .filter(|v| !v.is_empty())
        {
            let (parsed_begin, parsed_end) =
                parse_pair(&range, "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END")?;
            if parsed_begin >= parsed_end || parsed_end > block_count() {
                return Err(DitError::Geometry {
                    detail: "token-reduction block range must satisfy 0 <= BEGIN < END <= 50"
                        .into(),
                });
            }
            begin = parsed_begin;
            end = parsed_end;
        }

        // `:411-415`: coarse structure tolerates a deeper reduced stack while
        // the first noisy samples form; fine detail needs the full grid earlier.
        let mut early_steps = if end < 40 { 10 } else { 0 };
        let mut early_end = if end < 40 { 40 } else { end };
        if let Some(early) = std::env::var("H3_TOKEN_REDUCTION_EARLY")
            .ok()
            .filter(|value| !value.is_empty())
        {
            if early == "0" {
                early_steps = 0;
                early_end = end;
            } else {
                let (parsed_steps, parsed_end) =
                    parse_pair(&early, "H3_TOKEN_REDUCTION_EARLY must be STEPS:END")?;
                if parsed_steps == 0
                    || parsed_steps > 1000
                    || parsed_end <= end
                    || parsed_end > block_count()
                {
                    return Err(DitError::Geometry {
                        detail: "early token reduction requires STEPS > 0 and base END < END <= 50"
                            .into(),
                    });
                }
                early_steps = parsed_steps;
                early_end = parsed_end;
            }
        }

        let mut scale = 1.0f32;
        if let Some(text) = std::env::var("H3_TOKEN_REDUCTION_SCALE")
            .ok()
            .filter(|value| !value.is_empty())
        {
            let value: f32 = text.parse().map_err(|_| DitError::Geometry {
                detail: "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]".into(),
            })?;
            if !value.is_finite() || !(0.0..=2.0).contains(&value) {
                return Err(DitError::Geometry {
                    detail: "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]".into(),
                });
            }
            scale = value;
        }

        Ok(Some(ReductionWindow {
            begin,
            end,
            early_steps,
            early_end,
            scale,
        }))
    }
}

/// Which attention the DiT runs. `Tensor` is the default and is the tensor-core
/// kernel; `Scalar` is the byte-exact kernel and the only one that reproduces
/// the reference, so it is the one to ask for when a byte-identical output is
/// what is wanted (`docs/minimax-h3/tensor-default-2026-10-27.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Attention {
    #[default]
    Tensor,
    Scalar,
}

/// How the core runs: residency, active block count and token pairing
/// (`h3_dit.c:1601-1665`). The reference takes the same three choices as
/// separate loader arguments; one struct keeps the call sites readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadOptions {
    pub mode: LoadMode,
    /// Blocks that run, 25..=50 (`h3_dit.c:1590-1591`).
    pub layers: u32,
    pub reduction: Reduction,
    /// Core refresh interval, 1..=6 (`h3_dit.c:1591-1592`, `:2161-2163`).
    pub core_reuse: u32,
    /// Attention path; the tensor-core kernel unless `Scalar` is asked for.
    pub attention: Attention,
}

impl LoadOptions {
    /// Every block resident and every block active: the reference's defaults.
    pub fn resident() -> LoadOptions {
        LoadOptions {
            mode: LoadMode::Resident,
            layers: DIT_BLOCKS as u32,
            reduction: Reduction::Environment,
            core_reuse: 1,
            attention: Attention::Tensor,
        }
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions::resident()
    }
}

/// `configure_active_blocks` (`h3_dit.c:1181-1194`): the uniform starting
/// mask. Blocks 0 and 49 always stay; a skipped block is clamped off the ends.
fn uniform_blocks(layers: u32) -> Vec<bool> {
    let blocks = DIT_BLOCKS as u32;
    let mut active = vec![true; blocks as usize];
    let skipped = blocks - layers;
    if skipped == 0 {
        return active;
    }

    for index in 0..skipped {
        let mut block = ((2 * index + 1) * blocks) / (2 * skipped);
        if block == 0 {
            block = 1;
        }
        if block >= blocks - 1 {
            block = blocks - 2;
        }
        active[block as usize] = false;
    }

    active
}

/// `h3_dit_schedule_gate_score` (`h3_dit_schedule.c:388-418`): the mean
/// magnitude of the two gate slots (2 and 5) over every time row, modality and
/// column. A read failure is the reference's -1.
fn gate_score(
    gpu: &Gpu,
    modulation: &Tensor,
    time_rows: usize,
    hidden: usize,
) -> Result<f64, DitError> {
    let slots = BLOCK_SLOTS as usize;
    let modalities = MODALITIES as usize;
    let elements = time_rows * modalities * slots * hidden;
    let mut bytes = vec![0u8; elements * 2];
    if gpu.read(modulation, &mut bytes).is_err() {
        return Ok(-1.0);
    }

    let mut total = 0.0f64;
    for row in 0..time_rows {
        for modality in 0..modalities {
            for slot in [2usize, 5] {
                let base = ((row * modalities + modality) * slots + slot) * hidden;
                for column in 0..hidden {
                    let offset = 2 * (base + column);
                    let word = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                    // The reference widens BF16 with the low half zero
                    // (`h3_dit_schedule.c:409-411`).
                    total += f64::from(f32::from_bits(u32::from(word) << 16).abs());
                }
            }
        }
    }

    let samples = time_rows * modalities * 2 * hidden;
    Ok(if samples > 0 {
        total / samples as f64
    } else {
        -1.0
    })
}

/// `h3_dit.c:1196-1244` `configure_gate_ranked_blocks`: with fewer than 50
/// blocks the reference drops the lowest gate magnitudes outside blocks 0, 1
/// and 49, unless `H3_DIT_LAYER_POLICY=uniform`. Any unreadable score leaves
/// the uniform mask in place, as the reference does.
fn active_blocks(
    layers: u32,
    time_rows: usize,
    gpu: &Gpu,
    modulation: &[Tensor],
    hidden_width: u32,
) -> Result<Vec<bool>, DitError> {
    let blocks = DIT_BLOCKS as u32;
    let uniform = matches!(
        std::env::var("H3_DIT_LAYER_POLICY"),
        Ok(value) if value == "uniform"
    );
    if layers >= blocks || uniform {
        return Ok(uniform_blocks(layers));
    }

    let mut scores = Vec::with_capacity((blocks - 3) as usize);
    for block in 2..blocks - 1 {
        let score = gate_score(
            gpu,
            &modulation[block as usize],
            time_rows,
            hidden_width as usize,
        )?;
        if score < 0.0 {
            return Ok(uniform_blocks(layers));
        }
        scores.push((block, score));
    }

    // Ascending by magnitude: the `skipped` smallest are dropped.
    scores.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut active = vec![true; blocks as usize];
    for (block, _) in scores.iter().take((blocks - layers) as usize) {
        active[*block as usize] = false;
    }

    Ok(active)
}

/// The geometry the pairing maps need: the video target's first packed row and
/// the pool widths (`h3_dit.c:440-470`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PoolGeometry {
    start: u32,
    spatial_width: u32,
    reduced_width: u32,
    /// Reduced packed rows (`h3_dit.c:470`).
    rows: u32,
    /// Full video target rows, for the baseline count (`:469`).
    video_rows: u32,
}

/// `token_pool_sources` (`h3_dit.c:475-493`): the two full rows a reduced row
/// averages. A row before the video target, and the unpaired last column,
/// pools from itself.
fn pool_sources(geometry: PoolGeometry, reduced_row: u32) -> (u32, u32) {
    if reduced_row < geometry.start {
        return (reduced_row, reduced_row);
    }
    let local = reduced_row - geometry.start;
    let source = geometry.start
        + (local / geometry.reduced_width) * geometry.spatial_width
        + (local % geometry.reduced_width) * 2;
    let column = (source - geometry.start) % geometry.spatial_width;
    let second = source + u32::from(column + 1 < geometry.spatial_width);
    (source, second)
}

/// `token_reduced_parent` (`h3_dit.c:496-506`): the reduced row a full row
/// expands from.
fn reduced_parent(geometry: PoolGeometry, full_row: u32) -> u32 {
    if full_row < geometry.start {
        return full_row;
    }
    let local = full_row - geometry.start;
    geometry.start
        + (local / geometry.spatial_width) * geometry.reduced_width
        + (local % geometry.spatial_width) / 2
}

impl PoolGeometry {
    /// Rows the baseline buffer holds: one per reduced video row that pooled
    /// two rows, which is `video_rows - reduced_video_rows` (`h3_dit.c:469`).
    fn baseline_rows(&self) -> u32 {
        self.video_rows - (self.rows - self.start)
    }
}

/// `h3_dit.c:440-470`: the pairing geometry, refused unless the video target
/// ends the packed layout, which is what makes `sequence` an exact multiple of
/// the reduced row count.
fn pool_geometry(geometry: &Geometry, sequence_rows: usize) -> Result<PoolGeometry, DitError> {
    let spatial_height = geometry.latent_h / 2;
    let spatial_width = geometry.latent_w / 2;
    let reduced_width = spatial_width.div_ceil(2);
    let reduced_video = geometry.latent_t as u64 * spatial_height as u64 * reduced_width as u64;
    let inconsistent = spatial_height == 0
        || spatial_width == 0
        || geometry.latent_t as u64 * spatial_height as u64 * spatial_width as u64
            != geometry.video_rows as u64
        || geometry.video_target_start as u64 + geometry.video_rows as u64 != sequence_rows as u64
        || reduced_video > u32::MAX as u64
        || reduced_video > u32::MAX as u64 - geometry.video_target_start as u64;
    if inconsistent {
        return Err(DitError::Geometry {
            detail: "token reduction requires the target video to end the packed layout".into(),
        });
    }

    Ok(PoolGeometry {
        start: geometry.video_target_start,
        spatial_width,
        reduced_width,
        rows: geometry.video_target_start + reduced_video as u32,
        video_rows: geometry.video_rows,
    })
}

/// `h3_dit.c:1131-1178` `prepare_token_reduction_maps`: the pair list, the
/// baseline row each pooled video row lands in, and every full row's parent.
fn reduction_tables(
    gpu: &Gpu,
    sequence: &Sequence,
    pool: PoolGeometry,
) -> Result<(Tensor, Tensor, Tensor), DitError> {
    let mut pairs = Vec::with_capacity(pool.rows as usize * 2);
    let mut baseline_indices = vec![u32::MAX; pool.rows as usize];
    let mut baseline_row = 0u32;

    for row in 0..pool.rows {
        let (first, second) = pool_sources(pool, row);
        pairs.push(first);
        pairs.push(second);
        if row >= pool.start && first != second {
            baseline_indices[row as usize] = baseline_row;
            baseline_row += 1;
        }
    }

    if baseline_row != pool.baseline_rows() {
        return Err(DitError::Geometry {
            detail: "token-reduction baseline map is inconsistent".into(),
        });
    }

    let parents: Vec<u32> = (0..sequence.rows() as u32)
        .map(|row| reduced_parent(pool, row))
        .collect();

    Ok((
        u32_tensor(gpu, &pairs)?,
        u32_tensor(gpu, &baseline_indices)?,
        u32_tensor(gpu, &parents)?,
    ))
}

/// `h3_dit.c:1034-1041`: the per-step modulation map of the reduced rows,
/// which takes the first row of each pair.
fn reduced_maps(
    gpu: &Gpu,
    tree: &TimeRows,
    sequence: &Sequence,
    pool: PoolGeometry,
    text_tags: Option<&[u8]>,
) -> Result<Vec<Tensor>, DitError> {
    let mut maps = Vec::with_capacity(tree.steps);

    for step in 0..tree.steps {
        let full = row_map(tree, step, sequence, text_tags)?;
        let reduced: Vec<u32> = (0..pool.rows)
            .map(|row| full[pool_sources(pool, row).0 as usize])
            .collect();
        maps.push(u32_tensor(gpu, &reduced)?);
    }

    Ok(maps)
}

/// `h3_dit.c:896-948`: the reduced RoPE tables, whose positions are the mean
/// of the two pooled rows. The reference multiplies the averaged spatial
/// coordinate by the scale in F32 (`:940-941`); the same grouping is kept.
fn reduced_rope_tables(
    gpu: &Gpu,
    weights: &Weights,
    sequence: &Sequence,
    pool: PoolGeometry,
    spatial_scale: f32,
) -> Result<(Tensor, Tensor), DitError> {
    let inverse = read_f32(weights, Role::RopeFreq)?;
    if inverse.len() != INV_FREQ_ELEMENTS {
        return Err(DitError::Geometry {
            detail: format!("rope.inv_freq holds {} values", inverse.len()),
        });
    }

    let rows = pool.rows as usize;
    let mut cosines = vec![0.0f32; rows * ROPE_HALF];
    let mut sines = vec![0.0f32; rows * ROPE_HALF];

    for row in 0..rows {
        let (first, second) = pool_sources(pool, row as u32);
        let left = &sequence.positions[first as usize];
        let right = &sequence.positions[second as usize];
        let axes = [
            ((left.t + right.t) * 0.5) as f32,
            (((left.h + right.h) * 0.5) as f32) * spatial_scale,
            (((left.w + right.w) * 0.5) as f32) * spatial_scale,
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

    blocks: Vec<CoreBlock>,
    streamer: Option<Streamer>,
    final_layer: Final,
    patches: Patches,

    modulation: Vec<Tensor>,
    final_modulation: Tensor,
    row_maps: Vec<Tensor>,
    final_audio_maps: Vec<Tensor>,
    final_video_maps: Vec<Tensor>,
    /// `None` unless the caller asked for the reference's token pairing or
    /// the environment enabled it (`h3_dit.c:378-382`).
    reduction: Option<ReductionState>,
    /// Which of the 50 blocks run (`h3_dit.c:1181-1244`).
    active_blocks: Vec<bool>,
    /// `core_reuse_interval`, its forward counter and the cached delta
    /// (`h3_dit.c:103-105`).
    core_reuse: u32,
    core_forward_count: u32,
    core_residual_ready: bool,
    core_input: Option<Tensor>,
    core_residual: Option<Tensor>,
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
        Dit::load_with(
            dir,
            sequence,
            schedule,
            text,
            spatial_rope_scale,
            LoadOptions::resident(),
        )
    }

    /// `LoadMode::Stream` keeps two matrix slots on the device and reads each
    /// block from the checkpoint, as the reference's `--ssd-streaming` does
    /// (`h3_dit.c:1240-1300`, `:2178-2240`). The arithmetic is the same bytes
    /// through the same kernels; only residency changes.
    pub fn load_with(
        dir: &Path,
        sequence: Sequence,
        schedule: Schedule,
        text: &Embedding,
        spatial_rope_scale: f32,
        options: LoadOptions,
    ) -> Result<Dit, DitError> {
        if !(1..=6).contains(&options.core_reuse) {
            return Err(DitError::Geometry {
                detail: format!("core reuse {} is outside 1..=6", options.core_reuse),
            });
        }
        if options.layers < DIT_BLOCKS as u32 / 2 || options.layers > DIT_BLOCKS as u32 {
            return Err(DitError::Geometry {
                detail: format!(
                    "layers {} is outside {}..={}",
                    options.layers,
                    DIT_BLOCKS / 2,
                    DIT_BLOCKS
                ),
            });
        }
        let mode = options.mode;
        let reduction = options.reduction;
        let weights = Weights::open(dir).map_err(|error| DitError::Weights(error.to_string()))?;
        let config = weights.config().clone();
        let gpu = device(Gpu::create())?;
        gpu.set_attention(matches!(options.attention, Attention::Scalar));

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
        // `h3_dit.c:1028`: the presentation's tags reach the row map.
        let text_tags = text.tags();
        let (row_maps, final_audio_maps, final_video_maps) =
            modulation_maps(&gpu, &tree, &sequence, text_tags)?;

        // `h3_dit.c:1685-1688`: the core, then the activation arena.
        let blocks = load_blocks(&gpu, &weights, &config, mode)?;
        let mut streamer = match mode {
            LoadMode::Resident => None,
            LoadMode::Stream => Some(Streamer::new(&gpu, &weights, &config)?),
        };
        let final_layer = load_final(&gpu, &weights)?;
        let patches = load_patches(&gpu, &weights)?;
        let (video_projection_map, audio_projection_map) = projection_maps(&gpu, &sequence)?;

        let bf16 = |elements: usize| gpu_alloc(&gpu, Dtype::Bf16, elements);
        let f32t = |elements: usize| gpu_alloc(&gpu, Dtype::F32, elements);
        let rows = sequence.rows();

        // `h3_dit.c:440-470`, `:1131-1178` and `:896-948`: the pairing maps,
        // the reduced AdaLN rows, the averaged RoPE tables and the three
        // buffers one reduced pass snapshots into.
        let reduction = match reduction.window()? {
            Some(window) => {
                let pool = pool_geometry(&geometry, rows)?;
                let (pairs, baseline_indices, parents) = reduction_tables(&gpu, &sequence, pool)?;
                let (rope_cos, rope_sin) =
                    reduced_rope_tables(&gpu, &weights, &sequence, pool, spatial_rope_scale)?;
                let row_maps = reduced_maps(&gpu, &tree, &sequence, pool, text_tags)?;
                Some(ReductionState {
                    window,
                    rows: pool.rows,
                    baseline_rows: pool.baseline_rows(),
                    pairs,
                    baseline_indices,
                    parents,
                    rope_cos,
                    rope_sin,
                    row_maps,
                    hidden: gpu_alloc(
                        &gpu,
                        Dtype::Bf16,
                        pool.rows as usize * hidden_width as usize,
                    )?,
                    original: gpu_alloc(&gpu, Dtype::Bf16, rows * hidden_width as usize)?,
                    baseline: gpu_alloc(
                        &gpu,
                        Dtype::Bf16,
                        pool.baseline_rows() as usize * hidden_width as usize,
                    )?,
                })
            }
            None => None,
        };

        // `h3_dit.c:1181-1244`: the uniform mask first, then the gate-ranked
        // one the reference prefers unless `H3_DIT_LAYER_POLICY=uniform`.
        let active = active_blocks(
            options.layers,
            tree.times.len(),
            &gpu,
            &modulation,
            hidden_width,
        )?;

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
        // `h3_dit.c:1534-1543`: the cached core delta needs its own input copy.
        let (core_input, core_residual) = if options.core_reuse > 1 {
            (
                Some(bf16(rows * hidden_width as usize)?),
                Some(bf16(rows * hidden_width as usize)?),
            )
        } else {
            (None, None)
        };
        let final_norm_out =
            bf16(sequence.img_target_rows.max(sequence.audio_target_rows) * hidden_width as usize)?;
        let video_input = f32t(sequence.video_total_rows() * video_width as usize)?;
        let audio_input = f32t(sequence.audio_total_rows() * audio_width as usize)?;
        let video_output = bf16(sequence.img_target_rows * video_width as usize)?;
        let audio_output = bf16(sequence.audio_target_rows * audio_width as usize)?;

        device(gpu.sync())?;

        // `h3_dit.c:1272-1293`: the reference primes the first active block so
        // the first forward does not wait for it.
        if let Some(streamer) = streamer.as_mut() {
            streamer.ensure(&gpu, 0)?;
        }

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
            streamer,
            final_layer,
            patches,
            modulation,
            final_modulation,
            row_maps,
            final_audio_maps,
            final_video_maps,
            reduction,
            active_blocks: active,
            core_reuse: options.core_reuse,
            core_forward_count: 0,
            core_residual_ready: false,
            core_input,
            core_residual,
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

    /// One streaming slot's bytes, or `None` when every block is resident.
    pub fn stream_slot_bytes(&self) -> Option<u64> {
        self.streamer.as_ref().map(Streamer::slot_bytes)
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

    /// Write the anchor rows into the leading condition spans.
    ///
    /// This is `h3_dit_reset_run`'s write (`h3_dit.c:2408-2440`) after its
    /// count check: the rows land at offset 0 of `video_input` and
    /// `audio_input`, ahead of the sampled rows the forward appends at
    /// `img_cond_rows`/`audio_cond_rows` (`:2455-2462`). The reference also
    /// clears its core-reuse counters here, as this port does after upload.
    ///
    /// Empty slices are legal only when the packed layout has no condition
    /// rows for that modality; anything else is the reference's "prepared DiT
    /// condition rows do not match".
    pub fn set_conditions(&mut self, video: &[f32], audio: &[f32]) -> Result<(), DitError> {
        let wanted_video = self.video_cond_rows as usize * self.video_width as usize;
        let wanted_audio = self.audio_cond_rows as usize * self.audio_width as usize;

        if video.len() != wanted_video || audio.len() != wanted_audio {
            return Err(DitError::Geometry {
                detail: format!(
                    "condition rows of {}x{}, expected {}x{}",
                    video.len(),
                    audio.len(),
                    wanted_video,
                    wanted_audio
                ),
            });
        }

        if wanted_video != 0 {
            device(self.gpu.upload_at(&self.video_input, 0, &pack_f32(video)))?;
        }
        if wanted_audio != 0 {
            device(self.gpu.upload_at(&self.audio_input, 0, &pack_f32(audio)))?;
        }

        // `h3_dit.c:2437-2438`: a new run restarts the core-reuse counter.
        self.core_forward_count = 0;
        self.core_residual_ready = false;

        Ok(())
    }

    /// `h3_dit_forward` (`h3_dit.c:2442-2505`): pack the host latents, run the
    /// graph once at `step`, and return both velocities in the reference's
    /// `[24,T,H,W]` / `[32,2,T]` layouts.
    pub fn forward(
        &mut self,
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

    /// The whole Euler denoise on the host sampler
    /// (`h3_dit.c:2882-2985`).
    ///
    /// The latents are updated in place, one selected step at a time. A
    /// step that does not evaluate reuses the last velocity, extrapolated
    /// from the two most recent evaluations; with `reuse == 1` every step
    /// evaluates and no extrapolation runs (`h3_dit.c:2537-2554`).
    ///
    /// `gpu_sampler_requested` picks the device sampler only on M5 or when
    /// `H3_GPU_SAMPLER` says so (`h3_dit.c:2572-2578`); CUDA takes this
    /// path, so the port reproduces it.
    pub fn denoise(
        &mut self,
        video: &mut [f32],
        audio: &mut [f32],
        reuse: usize,
    ) -> Result<(), DitError> {
        self.denoise_with(video, audio, reuse, None)
    }

    /// `h3_dit_denoise_euler_preview` (`h3_dit.c:2861-2985`): the same sampler
    /// with a per-step hook. `preview` receives the step index and the video
    /// latent after that step's Euler update (`:2969-2974`), which is what the
    /// reference decodes into the preview frame.
    pub fn denoise_with(
        &mut self,
        video: &mut [f32],
        audio: &mut [f32],
        reuse: usize,
        mut preview: Option<PreviewHook>,
    ) -> Result<(), DitError> {
        let steps = self.schedule.steps;
        let selected = host::reuse_schedule(steps, reuse).ok_or_else(|| DitError::Geometry {
            detail: format!("reuse {reuse} outside 1..=32 or {steps} steps"),
        })?;
        if video.len() != self.video_elements() || audio.len() != self.audio_elements() {
            return Err(DitError::Geometry {
                detail: format!(
                    "denoise of {}x{} values, expected {}x{}",
                    video.len(),
                    audio.len(),
                    self.video_elements(),
                    self.audio_elements()
                ),
            });
        }

        // `last` is the newest evaluation's velocity, `previous` the one
        // before it; the ratio between their sigmas extrapolates a reused
        // step. Both are BF16-exact because `forward` expands BF16 readbacks.
        let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut previous: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut last_evaluated: Option<usize> = None;
        let mut previous_evaluated: Option<usize> = None;

        for (step, evaluate) in selected.iter().enumerate().take(steps) {
            let (video_velocity, audio_velocity) = if *evaluate {
                let (video_velocity, audio_velocity) = self.forward(step, video, audio)?;
                if reuse > 1 {
                    if let Some((video_last, audio_last)) = last.take() {
                        previous = Some((video_last, audio_last));
                        previous_evaluated = last_evaluated;
                    }
                    last = Some((video_velocity.clone(), audio_velocity.clone()));
                    last_evaluated = Some(step);
                }
                (video_velocity, audio_velocity)
            } else {
                let (video_last, audio_last) = last.as_ref().ok_or_else(|| DitError::Geometry {
                    detail: format!("step {step} reuses a velocity never evaluated"),
                })?;
                let mut video_velocity = vec![0.0f32; video_last.len()];
                let mut audio_velocity = vec![0.0f32; audio_last.len()];
                let last_step = last_evaluated.ok_or_else(|| DitError::Geometry {
                    detail: format!("step {step} reuses a velocity never evaluated"),
                })?;

                // `h3_dit.c:2525-2535`: without a second evaluation the last
                // velocity is copied unchanged.
                match previous.as_ref().zip(previous_evaluated) {
                    Some(((video_previous, audio_previous), previous_step)) => {
                        host::extrapolate_velocity(
                            &mut video_velocity,
                            video_last,
                            video_previous,
                            self.schedule.video[step],
                            self.schedule.video[last_step],
                            self.schedule.video[previous_step],
                            true,
                        );
                        host::extrapolate_velocity(
                            &mut audio_velocity,
                            audio_last,
                            audio_previous,
                            self.schedule.audio[step],
                            self.schedule.audio[last_step],
                            self.schedule.audio[previous_step],
                            true,
                        );
                    }
                    None => {
                        video_velocity.copy_from_slice(video_last);
                        audio_velocity.copy_from_slice(audio_last);
                    }
                }
                (video_velocity, audio_velocity)
            };

            let video_ok = host::euler_step(
                video,
                &video_velocity,
                self.schedule.video[step],
                self.schedule.video[step + 1],
            );
            let audio_ok = video_ok
                && host::euler_step(
                    audio,
                    &audio_velocity,
                    self.schedule.audio[step],
                    self.schedule.audio[step + 1],
                );
            if !audio_ok {
                return Err(DitError::Geometry {
                    detail: format!("Euler solver rejected step {step}"),
                });
            }

            // `h3_dit.c:2969-2974`: the hook runs after the update, on the
            // step's own latent, and a refusal ends the sampler.
            if let Some(hook) = preview.as_mut() {
                hook(step, video)?;
            }
        }

        Ok(())
    }

    /// `h3_dit.c:2316-2396`: the fused final head, AdaLN then projection,
    /// once per modality slice. Shared by the evaluated and the reused core
    /// step, which both end here.
    fn final_head(&self, step: usize) -> Result<(), DitError> {
        let hidden = self.hidden_width;

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

    /// `h3_dit.c:2050-2396` `encode_forward` with `core_reuse_interval = 1`,
    /// `ssd_streaming = 0` and the CUDA flag set of (`h3_dit.c:1601-1665`);
    /// token reduction runs when it was configured at load.
    fn encode_forward(&mut self, step: usize) -> Result<(), DitError> {
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

        // `h3_dit.c:2161-2165`: the core runs on the first forward, every
        // `core_reuse`-th one, and the last step; other steps add the cached
        // delta to the freshly packed rows.
        let evaluate_core = self.core_reuse == 1
            || !self.core_residual_ready
            || self.core_forward_count.is_multiple_of(self.core_reuse)
            || step == self.schedule.steps - 1;
        let hidden_elements = self.rows as usize * hidden as usize;

        if evaluate_core && self.core_reuse > 1 {
            let core_input = self
                .core_input
                .as_ref()
                .ok_or_else(|| DitError::Gpu("core reuse has no input buffer".into()))?;
            device(
                self.gpu
                    .copy(core_input, 0, &self.hidden, 0, hidden_elements),
            )?;
        }

        // `h3_dit.c:2166-2210`: the reduced window for this step, on the
        // steps that evaluate the core. A forward can skip it for diagnosis,
        // as `H3_DISABLE_TOKEN_REDUCTION` does.
        let window = self
            .reduction
            .as_ref()
            .filter(|_| evaluate_core && std::env::var_os("H3_DISABLE_TOKEN_REDUCTION").is_none())
            .map(|state| (state, state.window.end_for_step(step)));

        if !evaluate_core {
            let residual = self
                .core_residual
                .as_ref()
                .ok_or_else(|| DitError::Gpu("core reuse has no residual buffer".into()))?;
            device(self.gpu.add(&self.hidden, &self.hidden, residual))?;
            self.core_forward_count += 1;
            return self.final_head(step);
        }

        // `h3_dit.c:2181-2306`: every block, with the next block's attention
        // AdaLN fused into this block's MLP gate.
        let mut carried = false;
        let streamed = self.streamer.is_some();
        for block in 0..self.blocks.len() {
            let mut reduced = false;
            if let Some((state, end)) = window {
                let index = block as u32;
                if index == state.window.begin {
                    self.pool_tokens(state)?;
                }
                if index == end && (end as usize) < self.blocks.len() {
                    self.expand_tokens(state)?;
                }
                reduced = index >= state.window.begin && index < end;
            }
            // `h3_dit.c:2217`: a dropped block does no work, but the pool and
            // expand boundaries above still run at their own block indices.
            if !self.active_blocks[block] {
                continue;
            }
            let active = match (reduced, window) {
                (true, Some((state, _))) => Active {
                    rows: state.rows,
                    hidden: &state.hidden,
                    row_map: &state.row_maps[step],
                    rope_cos: &state.rope_cos,
                    rope_sin: &state.rope_sin,
                },
                _ => Active {
                    rows: self.rows,
                    hidden: &self.hidden,
                    row_map: &self.row_maps[step],
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                },
            };
            // `h3_dit.c:2210-2214`: the next block's attention AdaLN is not
            // fused across a row-count boundary, and only into a block that
            // runs (`h3_dit.c:2138-2140`).
            let boundary = window.is_some_and(|(state, end)| {
                let next = block as u32 + 1;
                next == state.window.begin || next == end
            });
            let fuse_next =
                block + 1 < self.blocks.len() && self.active_blocks[block + 1] && !boundary;
            if streamed {
                let streamer = self.streamer.as_mut().expect("streamed forward");
                streamer.ensure(&self.gpu, block)?;
                // `h3_dit.c:2223-2240`: the next block is read while this one
                // runs; after the last block the ring wraps to block 0, so the
                // next denoise step starts ready.
                let next = if block + 1 < self.blocks.len() {
                    block + 1
                } else {
                    0
                };
                streamer.start(next)?;
            }
            self.run_block(block, &active, fuse_next, &mut carried, streamed)?;
            if streamed {
                self.streamer
                    .as_mut()
                    .expect("streamed forward")
                    .finish(&self.gpu)?;
            }
        }

        // `h3_dit.c:2304-2306`: a window that ends at or past the last block
        // has no boundary inside the loop. (The reference always walks all 50
        // blocks; the port loads them all too, but the comparison keeps the
        // pooled stream from surviving a short load.)
        if let Some((state, end)) = window {
            if end as usize >= self.blocks.len() {
                self.expand_tokens(state)?;
            }
        }

        // `h3_dit.c:2306-2312`: keep the delta the next reused step adds.
        if self.core_reuse > 1 {
            let core_input = self
                .core_input
                .as_ref()
                .ok_or_else(|| DitError::Gpu("core reuse has no input buffer".into()))?;
            let core_residual = self
                .core_residual
                .as_ref()
                .ok_or_else(|| DitError::Gpu("core reuse has no residual buffer".into()))?;
            device(self.gpu.sub(core_residual, &self.hidden, core_input))?;
            self.core_residual_ready = true;
        }
        self.core_forward_count += 1;

        self.final_head(step)
    }

    /// `h3_dit.c:1789-1805` `enter_token_reduction`: snapshot every row the
    /// reduced pass replaces, average each video pair into the reduced stream,
    /// and keep the pooled values for the expansion to subtract.
    fn pool_tokens(&self, state: &ReductionState) -> Result<(), DitError> {
        device(self.gpu.token_pool(
            &state.hidden,
            &self.hidden,
            0,
            &state.original,
            0,
            &state.baseline,
            0,
            &state.baseline_indices,
            &state.pairs,
            self.rows,
            state.rows,
            state.baseline_rows,
            self.hidden_width,
        ))
    }

    /// `h3_dit.c:1831-1849` `leave_token_reduction`: every full row takes its
    /// parent's reduced value as a scaled delta over the pooled baseline, and
    /// the prefix rows take the reduced value directly.
    fn expand_tokens(&self, state: &ReductionState) -> Result<(), DitError> {
        device(self.gpu.token_expand(
            &self.hidden,
            &state.original,
            0,
            &state.hidden,
            &state.baseline,
            0,
            &state.baseline_indices,
            &state.parents,
            self.rows,
            state.rows,
            state.baseline_rows,
            self.hidden_width,
            self.video_target_start,
            state.window.scale,
        ))
    }

    /// `h3_dit.c:1876-2047` `run_block` with the int8 and NAX branches off.
    ///
    /// `carried` is the `attention_adaln_ready` flag the previous block left
    /// behind (`h3_dit.c:2184-2226`): when set, this block's attention AdaLN
    /// already ran and only the QKV projection is left to do.
    fn run_block(
        &self,
        block: usize,
        active: &Active,
        fuse_next_attention: bool,
        carried: &mut bool,
        streamed: bool,
    ) -> Result<(), DitError> {
        let rows = active.rows;
        let hidden = self.hidden_width;
        let inner = self.inner;
        let modulation = &self.modulation[block];
        let row_map = active.row_map;
        let norms = &self.blocks[block].norms;
        let matrices = match streamed {
            true => self
                .streamer
                .as_ref()
                .ok_or_else(|| DitError::Gpu("streamed block without a ring".into()))?
                .current(),
            false => self.blocks[block]
                .matrices
                .as_ref()
                .ok_or_else(|| DitError::Gpu("resident block without matrices".into()))?,
        };
        let attention_adaln_ready = *carried;
        *carried = false;

        if !attention_adaln_ready {
            device(self.gpu.adaln(
                &self.mod_attention,
                active.hidden,
                0,
                &norms.norm1,
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
            &matrices.qkv,
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
            &norms.q_norm,
            &norms.k_norm,
            active.rope_cos,
            active.rope_sin,
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
            &matrices.out,
            None,
            rows,
            inner,
            hidden,
            Dtype::Bf16,
        ))?;

        // `h3_dit.c:1958-1982`: the attention gate, then the MLP AdaLN, which
        // reuses the qkv buffer as `mod_mlp`.
        device(self.gpu.gate(
            active.hidden,
            active.hidden,
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
            active.hidden,
            0,
            &norms.norm2,
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
            &matrices.fc1,
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
            &matrices.fc2,
            None,
            rows,
            self.config.ffn as u32,
            hidden,
            Dtype::Bf16,
        ))?;

        // `h3_dit.c:2015-2046`: the MLP gate, and, when another block follows,
        // the next block's attention AdaLN in the same step.
        device(self.gpu.gate(
            active.hidden,
            active.hidden,
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
            active.hidden,
            0,
            &next.norms.norm1,
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
    text_tags: Option<&[u8]>,
) -> Result<ModulationMaps, DitError> {
    let mut row_maps = Vec::with_capacity(tree.steps);
    let mut final_audio = Vec::with_capacity(tree.steps);
    let mut final_video = Vec::with_capacity(tree.steps);

    for step in 0..tree.steps {
        let rows = row_map(tree, step, sequence, text_tags)?;
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

fn load_blocks(
    gpu: &Gpu,
    weights: &Weights,
    config: &Config,
    mode: LoadMode,
) -> Result<Vec<CoreBlock>, DitError> {
    let mut blocks = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        blocks.push(load_block(gpu, weights, layer, mode)?);
    }
    Ok(blocks)
}

/// `h3_dit.c:502-525` `load_block`, and `:1251-1262` `load_block_norms` when
/// the matrices stream.
fn load_block(
    gpu: &Gpu,
    weights: &Weights,
    layer: u32,
    mode: LoadMode,
) -> Result<CoreBlock, DitError> {
    let norms = BlockNorms {
        norm1: tensor_from_role(gpu, weights, Role::BlockNorm1(layer), Dtype::Bf16)?,
        norm2: tensor_from_role(gpu, weights, Role::BlockNorm2(layer), Dtype::Bf16)?,
        q_norm: tensor_from_role(gpu, weights, Role::BlockQNorm(layer), Dtype::Bf16)?,
        k_norm: tensor_from_role(gpu, weights, Role::BlockKNorm(layer), Dtype::Bf16)?,
    };
    let matrices = match mode {
        LoadMode::Resident => Some(load_block_matrices(gpu, weights, layer)?),
        LoadMode::Stream => None,
    };
    Ok(CoreBlock { norms, matrices })
}

fn load_block_matrices(
    gpu: &Gpu,
    weights: &Weights,
    layer: u32,
) -> Result<BlockMatrices, DitError> {
    Ok(BlockMatrices {
        qkv: tensor_from_role(gpu, weights, Role::BlockQkv(layer), Dtype::Bf16)?,
        out: tensor_from_role(gpu, weights, Role::BlockOut(layer), Dtype::Bf16)?,
        fc1: tensor_from_role(gpu, weights, Role::BlockFc1(layer), Dtype::Bf16)?,
        fc2: tensor_from_role(gpu, weights, Role::BlockFc2(layer), Dtype::Bf16)?,
    })
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
        let rows = row_map(&tree, 1, &sequence, None).expect("row map");

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

    /// `h3_multimodal.c:173`, `:182`: an anchored prompt's vision spans are
    /// tagged 0 and its prompt rows 1, and `row_map` must carry them through
    /// (`h3_dit.c:1028`). A tag set that does not cover the text rows, or one
    /// outside the modality count, is refused (`h3_dit_schedule.c:440`, `:453`).
    #[test]
    fn row_map_carries_the_presentation_tags() {
        let schedule = Schedule::serving(2).expect("two steps");
        let sequence = Sequence {
            positions: vec![
                Position {
                    t: 0.0,
                    h: 0.0,
                    w: 0.0
                };
                3
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
                    kind: SegmentKind::Video,
                },
            ],
            img_cond_rows: 0,
            img_target_rows: 1,
            audio_cond_rows: 0,
            audio_target_rows: 0,
            signature: [2, 2, 16, 16, 8],
        };
        let tree = time_rows(&schedule, false, false);

        // One vision span row, one prompt row: the picture row takes slot 0.
        let rows = row_map(&tree, 1, &sequence, Some(&[0, 1])).expect("row map");
        assert_eq!(
            rows,
            vec![
                tree.video[1] * MODALITIES,
                tree.video[1] * MODALITIES + 1,
                tree.video[1] * MODALITIES
            ]
        );

        // A short tag set and an out-of-range tag are both errors.
        assert!(row_map(&tree, 1, &sequence, Some(&[0])).is_err());
        assert!(row_map(&tree, 1, &sequence, Some(&[0, MODALITIES as u8])).is_err());
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

    /// `h3_dit.c:1181-1194`: the uniform mask keeps block 0 and block 49 and
    /// clamps every skipped index off both ends.
    #[test]
    fn uniform_mask_keeps_the_ends() {
        for layers in [45u32, 40, 25] {
            let mask = uniform_blocks(layers);
            assert_eq!(mask.len(), DIT_BLOCKS);
            assert_eq!(
                mask.iter().filter(|active| **active).count(),
                layers as usize
            );
            assert!(mask[0] && mask[DIT_BLOCKS - 1]);
        }
        assert!(uniform_blocks(DIT_BLOCKS as u32)
            .iter()
            .all(|active| *active));
    }

    /// `h3_dit.c:475-506`: every reduced row pools two adjacent columns of one
    /// video plane, and every full row expands from a parent that actually
    /// contains it. A pairing that dropped or duplicated a row would still
    /// produce plausible output, so the maps are checked as maps.
    #[test]
    fn pairing_covers_every_video_row_once() {
        // One 2x2 plane of 3x3 patches is impossible; use 3 planes of 2x5 so
        // the odd width exercises the unpaired last column.
        let pool = PoolGeometry {
            start: 6,
            spatial_width: 5,
            reduced_width: 3,
            rows: 6 + 2 * 3,
            video_rows: 10,
        };

        let mut seen = vec![0u8; 10];
        for row in pool.start..pool.rows {
            let (first, second) = pool_sources(pool, row);
            seen[(first - pool.start) as usize] += 1;
            if second != first {
                seen[(second - pool.start) as usize] += 1;
            }
        }
        assert_eq!(seen, vec![1u8; 10], "every video row pools exactly once");
        // Two paired reduced rows per 5-wide plane: 10 video rows - 6 reduced.
        assert_eq!(pool.baseline_rows(), 4);

        for row in 0..pool.rows + 4 {
            let parent = reduced_parent(pool, row);
            let (first, second) = pool_sources(pool, parent);
            assert!(
                row == first || row == second,
                "row {row} expands from {parent}, whose pair is {first}/{second}"
            );
        }
    }

    /// `h3_dit.c:475-493`: a prefix row pools from itself, so pooling leaves
    /// text, condition and audio rows untouched.
    #[test]
    fn prefix_rows_pool_from_themselves() {
        let pool = PoolGeometry {
            start: 3,
            spatial_width: 2,
            reduced_width: 1,
            rows: 4,
            video_rows: 2,
        };

        assert_eq!(pool_sources(pool, 0), (0, 0));
        assert_eq!(pool_sources(pool, 2), (2, 2));
        assert_eq!(pool_sources(pool, 3), (3, 4));
        assert_eq!(reduced_parent(pool, 1), 1);
        assert_eq!(reduced_parent(pool, 4), 3);
    }

    /// `h3_dit.c:2168-2170`: the early window is a step predicate, and with no
    /// early steps every step takes the base end.
    #[test]
    fn early_window_is_per_step() {
        let window = ReductionWindow {
            begin: 4,
            end: 30,
            early_steps: 2,
            early_end: 40,
            scale: 1.0,
        };
        assert_eq!(window.end_for_step(0), 40);
        assert_eq!(window.end_for_step(1), 40);
        assert_eq!(window.end_for_step(2), 30);

        let flat = ReductionWindow {
            early_steps: 0,
            ..window
        };
        assert_eq!(flat.end_for_step(0), 30);
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
