//! The AV connector and the feature extractor in front of it: unit 4, the half
//! of the offline conditioning that turns the Gemma4 tower's 49 hidden states
//! into the two contexts the refiner consumes.
//!
//! Source of truth: the released reference implementation, `ltx-core/src/ltx_core/text_encoders/gemma/`,
//! read first-hand at `/data/ltx2-pin/packages` (the pin the release runs,
//! `d1511477` = v1.2.0). Every step below cites it:
//!
//! - `feature_extractor.py:100-129` `FeatureExtractorV2`: `torch.stack(states,
//!   dim=-1)` over the 49 states, `norm_and_concat_per_token_rms`
//!   (`:48-64`), a `_rescale_norm` (`:68-69`) per modality, then the two
//!   biased aggregate embeds `video_aggregate_embed` `[4096, 188160]` and
//!   `audio_aggregate_embed` `[2048, 188160]`.
//! - `embeddings_processor.py:96-117` `process_hidden_states` and `:70-94`
//!   `create_embeddings`: the additive mask (`:15-19`), the right-pad
//!   permutation (`:23-43`), the connector call, and the binary mask the video
//!   context is multiplied by (`:87-88`).
//! - `embeddings_connector.py:153-191` `Embeddings1DConnector.forward`: the
//!   register replacement, the frequency grid, the eight blocks, the closing
//!   weightless RMS norm.
//! - `embeddings_connector.py:14-70` `_BasicTransformerBlock1D`: `rms_norm`,
//!   self-attention, residual, `rms_norm`, feed-forward, residual.
//! - `model/transformer/attention.py:478-579` `Attention` and
//!   `ops.py:94-106` `PytorchGatedAttention`; `model/transformer/
//!   feed_forward.py:4-15` and `gelu_approx.py:10-11`; `model/transformer/
//!   rope.py:42-85` (the split rotation) and `:113-184` (the grid and tables).
//!
//! Geometry comes from the checkpoint's own metadata, not from the
//! constructor defaults: `connector_num_attention_heads` 32 with
//! `connector_attention_head_dim` 128 for video and
//! `audio_connector_attention_head_dim` 64 for audio,
//! `connector_num_layers` 8, `connector_positional_embedding_max_pos` `[4096]`,
//! `connector_num_learnable_registers` 128, `connector_apply_gated_attention`
//! true, `rope_type` "split", `frequencies_precision` "float64" (read from the
//! transformer shard's `__metadata__.config`). `connector_ff_bias` is absent,
//! so `embeddings_connector.py:214`'s default `True` stands - which is what the
//! checkpoint's own `ff.net.*.bias` tensors show.
//!
//! Two operations are the reference's and are reproduced rather than
//! simplified: the extractor's output is written `d * 49 + l` because
//! `normed.reshape(B, T, D * L)` (`:62`) flattens the stacked `[B, T, D, L]`
//! in C order, so the aggregate embed reads a hidden dimension's 49 layers
//! contiguously; and the connector's attention takes no mask, because
//! `_replace_padded_with_learnable_registers` returns `zeros_like` and every
//! key is therefore valid (`embeddings_connector.py:151`).
//!
//! Deviation, named: the connector weights here are BF16, from the release's
//! own `ltx-2.5-22b-dev-transformer-bf16.safetensors` and
//! `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`. The recipe's INT8 files
//! (`gemma4-12b-...-comfy-int8-convrot.safetensors` and the never-delivered
//! `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors`) are the ones
//! the delivered cache must come from; a BF16 cache is not interchangeable
//! (`docs/setup.md:127-129`). The INT8 convrot route itself is already built
//! and gated (`h3d_linear_int8_convrot`), so swapping the operand source is a
//! loader change, not a graph change.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

use crate::minimax_h3::gemma::bf16_round;
use crate::minimax_h3::safetensors::{self, Dtype, Shard};

/// The hidden states the extractor stacks (`gemma.rs`'s `STATES`).
pub const STATES: usize = 49;
/// Gemma4's hidden width.
pub const HIDDEN: usize = 3840;
/// `hidden_size * num_hidden_layers + hidden_size`: the aggregate embeds'
/// input width (`encoder_configurator.py:195-196`, `flat_dim`).
pub const FLAT: usize = 188160;
/// The video connector's `inner_dim` (`connector_num_attention_heads` 32 x
/// `connector_attention_head_dim` 128).
pub const VIDEO_INNER: usize = 4096;
/// The audio connector's (`audio_connector_attention_head_dim` 64).
pub const AUDIO_INNER: usize = 2048;
/// Both connectors use 32 heads (`encoder_configurator.py:204`, `:232`).
pub const HEADS: usize = 32;
/// The video head width.
pub const VIDEO_HEAD_DIM: usize = 128;
/// The audio head width.
pub const AUDIO_HEAD_DIM: usize = 64;
/// `connector_num_layers`: eight blocks per connector.
pub const BLOCKS: usize = 8;
/// `num_learnable_registers` (`embeddings_connector.py:103`), the tiling the
/// length must divide.
pub const REGISTERS: usize = 128;
/// `FeedForward`'s `mult` (`feed_forward.py:8`).
pub const MULTIPLIER: usize = 4;
/// `rms_norm`'s default epsilon (`utils.py:7`); the connector never overrides
/// it.
pub const RMS_EPSILON: f32 = 1e-6;
/// `positional_embedding_theta`, the connector constructor's default
/// (`embeddings_connector.py:100`), which the configurator never overrides.
pub const ROPE_BASE: f64 = 10_000.0;
/// `connector_positional_embedding_max_pos[0]`: the connector's one position
/// dimension is normalized by 4096, not by the sequence length.
pub const ROPE_MAX_POSITION: f32 = 4096.0;
/// The tokenizer's fixed length (`gemma_assets.py:161`), the only length the
/// release runs.
pub const ROWS: usize = 1024;

/// Why the connector could not be loaded or run.
#[derive(Debug)]
pub enum ConnectorError {
    Reader(String),
    Missing(String),
    Shape { name: String, detail: String },
    Run(String),
}

impl ConnectorError {
    /// The one-line token a record quotes.
    pub fn token(&self) -> String {
        match self {
            ConnectorError::Reader(detail) => format!("reader: {detail}"),
            ConnectorError::Missing(name) => format!("missing: {name}"),
            ConnectorError::Shape { name, detail } => format!("shape: {name}: {detail}"),
            ConnectorError::Run(detail) => format!("run: {detail}"),
        }
    }
}

impl fmt::Display for ConnectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for ConnectorError {}

/// One biased BF16 linear, the reference's `torch.nn.Linear` in BF16. The
/// reference fuses its bias into the GEMM's accumulate; [`Gpu::linear`] adds
/// it after the matmul's rounding (a named deviation, recorded in the unit's
/// record).
struct Linear {
    weight: Tensor,
    bias: Tensor,
    input_dim: u32,
    output_dim: u32,
}

impl Linear {
    fn apply(
        &self,
        gpu: &Gpu,
        output: &Tensor,
        input: &Tensor,
        rows: u32,
    ) -> Result<(), ConnectorError> {
        gpu.linear(
            output,
            input,
            &self.weight,
            Some(&self.bias),
            rows,
            self.input_dim,
            self.output_dim,
            Device::Bf16,
        )
        .map_err(ConnectorError::Run)
    }
}

/// One `_BasicTransformerBlock1D` (`embeddings_connector.py:14-70`): the
/// attention's four projections and its per-head gate, and the feed-forward's
/// two, all biased (`connector_ff_bias` defaults to true).
struct Block {
    q_norm: Tensor,
    k_norm: Tensor,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// One `Embeddings1DConnector`: eight blocks, the register table and the
/// geometry its rope and gate need.
struct Connector {
    blocks: Vec<Block>,
    registers: Tensor,
    inner: usize,
    head_dim: usize,
}

/// The two biased aggregate embeds (`feature_extractor.py:100-129`).
struct Extractor {
    video: Linear,
    audio: Linear,
}

/// The loaded extractor and both connectors.
pub struct Processor {
    extractor: Extractor,
    video: Connector,
    audio: Connector,
    tensors: usize,
    bytes: u64,
}

/// Every tensor one forward produces, in order, so a gate can hash each stage
/// instead of only the two ends.
pub struct Stages {
    /// `[rows, 49]`: the per-(token, layer) rsqrt the extractor scales by.
    pub scales: Tensor,
    /// `[rows, 188160]`: the interleaved, scaled stack the aggregate embeds
    /// read.
    pub normed: Tensor,
    /// `[rows, 4096]`: `FeatureExtractorV2.forward`'s video output.
    pub video_features: Tensor,
    /// `[rows, 2048]`: its audio output.
    pub audio_features: Tensor,
    /// `[rows, 4096]`: the video connector's context.
    pub video: Tensor,
    /// `[rows, 2048]`: the audio connector's.
    pub audio: Tensor,
    pub rows: u32,
}

impl Processor {
    /// Read both shards and upload every tensor the graph needs. `gemma_path`
    /// is the text-encoder checkpoint (`text_embedding_projection.*`), and
    /// `connector_path` the transformer carrying
    /// `model.diffusion_model.{video,audio}_embeddings_connector.*`.
    pub fn open(
        gpu: &Gpu,
        gemma_path: &Path,
        connector_path: &Path,
    ) -> Result<Processor, ConnectorError> {
        let gemma = safetensors::read_shard(gemma_path)
            .map_err(|error| ConnectorError::Reader(error.token()))?;
        let transformer = safetensors::read_shard(connector_path)
            .map_err(|error| ConnectorError::Reader(error.token()))?;
        let mut reader = Reader {
            gpu,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };

        let extractor = Extractor {
            video: reader.linear(
                &gemma,
                "text_embedding_projection.video_aggregate_embed",
                FLAT,
                VIDEO_INNER,
            )?,
            audio: reader.linear(
                &gemma,
                "text_embedding_projection.audio_aggregate_embed",
                FLAT,
                AUDIO_INNER,
            )?,
        };

        let video = reader.connector(&transformer, "video", VIDEO_INNER, VIDEO_HEAD_DIM)?;
        let audio = reader.connector(&transformer, "audio", AUDIO_INNER, AUDIO_HEAD_DIM)?;

        Ok(Processor {
            extractor,
            video,
            audio,
            tensors: reader.tensors,
            bytes: reader.bytes,
        })
    }

    /// The tensors read, and their bytes, for the gate's record.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// `EmbeddingsProcessor.process_hidden_states` (`embeddings_processor.py:
    /// 96-117`) end to end: the 49 stacked states and the tokenizer's mask in,
    /// both contexts out.
    pub fn process(
        &self,
        gpu: &Gpu,
        states: &[Tensor],
        padding: &[u32],
    ) -> Result<Stages, ConnectorError> {
        let rows = padding.len();

        if rows == 0 || states.len() != STATES {
            return Err(ConnectorError::Run(format!(
                "the extractor needs {STATES} states and a non-empty mask, got {} and {rows}",
                states.len()
            )));
        }
        if rows % REGISTERS != 0 {
            return Err(ConnectorError::Run(format!(
                "rows must be a multiple of {REGISTERS} for the register tiling, got {rows}"
            )));
        }

        let rows = rows as u32;

        // `torch.stack(hidden_states, dim=-1)` (`feature_extractor.py:120`):
        // one `[layers, rows, height]` block, so the reduction and the
        // interleave are two passes over contiguous memory.
        let stacked = alloc(gpu, Device::Bf16, STATES * rows as usize * HIDDEN)?;
        for (layer, state) in states.iter().enumerate() {
            gpu.copy(
                &stacked,
                layer * rows as usize * HIDDEN,
                state,
                0,
                rows as usize * HIDDEN,
            )
            .map_err(ConnectorError::Run)?;
        }

        let scales = alloc(gpu, Device::Bf16, rows as usize * STATES)?;
        gpu.connector_stack_scale(&scales, &stacked, rows, HIDDEN as u32, STATES as u32)
            .map_err(ConnectorError::Run)?;

        let normed = alloc(gpu, Device::Bf16, rows as usize * FLAT)?;
        gpu.connector_interleave(
            &normed,
            &stacked,
            &scales,
            rows,
            HIDDEN as u32,
            STATES as u32,
        )
        .map_err(ConnectorError::Run)?;

        let video_features = self
            .extractor
            .feature(gpu, &normed, rows, &self.extractor.video)?;
        let audio_features = self
            .extractor
            .feature(gpu, &normed, rows, &self.extractor.audio)?;

        // The connectors expect right-padded input (`embeddings_processor.py:
        // 80-91`), so the mask is reordered with the features it masks.
        let (order, binary) = right_pad(padding);
        let order = upload_u32(gpu, &order)?;
        let binary = upload_u32(gpu, &binary)?;

        let video_ordered = alloc(gpu, Device::Bf16, rows as usize * VIDEO_INNER)?;
        gpu.gather_rows(
            &video_ordered,
            &video_features,
            &order,
            rows,
            VIDEO_INNER as u32,
        )
        .map_err(ConnectorError::Run)?;
        let audio_ordered = alloc(gpu, Device::Bf16, rows as usize * AUDIO_INNER)?;
        gpu.gather_rows(
            &audio_ordered,
            &audio_features,
            &order,
            rows,
            AUDIO_INNER as u32,
        )
        .map_err(ConnectorError::Run)?;

        let video = self.video.run(gpu, &video_ordered, &binary, rows)?;
        let audio = self.audio.run(gpu, &audio_ordered, &binary, rows)?;

        // `video_encoded = video_encoded * binary_mask` (`embeddings_processor
        // .py:87-88`): the connector returns the all-zero mask it was handed,
        // so `_to_binary_mask` (`:46-49`) is all ones and this is the identity.
        // Kept as a comment because a reader will look for the multiply.

        Ok(Stages {
            scales,
            normed,
            video_features,
            audio_features,
            video,
            audio,
            rows,
        })
    }
}

impl Extractor {
    /// `_rescale_norm` then the aggregate embed (`feature_extractor.py:
    /// 121-128`), the rescale a `sqrt(target / source)` on `normed` - the
    /// result is a BF16 tensor the embed reads, so the multiply happens before
    /// the linear rather than folded into its weights.
    fn feature(
        &self,
        gpu: &Gpu,
        normed: &Tensor,
        rows: u32,
        embed: &Linear,
    ) -> Result<Tensor, ConnectorError> {
        let factor = ((embed.output_dim as f32) / (HIDDEN as f32)).sqrt();
        let scaled = alloc(gpu, Device::Bf16, rows as usize * FLAT)?;
        gpu.scale_constant(&scaled, normed, factor)
            .map_err(ConnectorError::Run)?;

        let output = alloc(gpu, Device::Bf16, rows as usize * embed.output_dim as usize)?;
        embed.apply(gpu, &output, &scaled, rows)?;

        Ok(output)
    }
}

impl Connector {
    /// `Embeddings1DConnector.forward` (`embeddings_connector.py:153-191`).
    fn run(
        &self,
        gpu: &Gpu,
        features: &Tensor,
        binary: &Tensor,
        rows: u32,
    ) -> Result<Tensor, ConnectorError> {
        let filled = alloc(gpu, Device::Bf16, rows as usize * self.inner)?;
        gpu.register_fill(
            &filled,
            features,
            binary,
            &self.registers,
            rows,
            self.inner as u32,
            REGISTERS as u32,
        )
        .map_err(ConnectorError::Run)?;

        let heads = self.inner / self.head_dim;
        let (cos, sin) = self.tables(gpu, rows)?;
        let mut scratch = Scratch::new(gpu, rows, self.inner)?;

        let mut hidden = filled;
        for block in &self.blocks {
            hidden = block.apply(gpu, &hidden, &cos, &sin, rows, heads, &mut scratch)?;
        }

        let output = alloc(gpu, Device::Bf16, rows as usize * self.inner)?;
        gpu.rms_norm(&output, &hidden, None, rows, self.inner as u32, RMS_EPSILON)
            .map_err(ConnectorError::Run)?;

        Ok(output)
    }

    /// `precompute_freqs_cis` (`rope.py:196-223`) for this connector's
    /// geometry.
    fn tables(&self, gpu: &Gpu, rows: u32) -> Result<(Tensor, Tensor), ConnectorError> {
        rope_tables(gpu, rows, self.inner / self.head_dim, self.head_dim)
    }
}

/// `precompute_freqs_cis` (`rope.py:196-223`) with `generate_freq_grid_np`
/// (`:86-108`, the float64 grid `frequencies_precision: "float64"` selects),
/// built on the host as the reference builds it: the grid is
/// `theta ** linspace(0, 1, inner_dim / 2)` in float64 times `pi / 2`, cast to
/// float32, and each position's angle is `grid[g] * (2 * row / 4096 - 1)`
/// (`:159`, `:136-143`). The table is `[rows, heads, head_dim / 2]`,
/// head-major, so a block's `(row, head, dimension)` triple indexes it
/// directly.
pub fn rope_tables(
    gpu: &Gpu,
    rows: u32,
    heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor), ConnectorError> {
    let half = head_dim / 2;
    let angles = heads * half;
    let mut grid = Vec::with_capacity(angles);

    for index in 0..angles {
        let position = index as f64 / (angles as f64 - 1.0);
        grid.push((ROPE_BASE.powf(position) * std::f64::consts::FRAC_PI_2) as f32);
    }

    let mut cos = Vec::with_capacity(rows as usize * angles);
    let mut sin = Vec::with_capacity(rows as usize * angles);

    for row in 0..rows {
        let fraction = row as f32 / ROPE_MAX_POSITION;
        let scale = fraction * 2.0 - 1.0;

        for value in &grid {
            let angle = value * scale;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }

    Ok((upload_bf16(gpu, &cos)?, upload_bf16(gpu, &sin)?))
}

impl Block {
    /// `_BasicTransformerBlock1D.forward` (`embeddings_connector.py:41-70`).
    fn apply(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rows: u32,
        heads: usize,
        scratch: &mut Scratch,
    ) -> Result<Tensor, ConnectorError> {
        let inner = scratch.inner as u32;

        gpu.rms_norm(&scratch.normed, input, None, rows, inner, RMS_EPSILON)
            .map_err(ConnectorError::Run)?;

        self.query
            .apply(gpu, &scratch.query, &scratch.normed, rows)?;
        self.key.apply(gpu, &scratch.key, &scratch.normed, rows)?;
        self.value
            .apply(gpu, &scratch.value, &scratch.normed, rows)?;

        // `q_norm`/`k_norm` are `RMSNorm(inner_dim)`, over the whole inner
        // dimension and *before* the head split (`attention.py:505-506`).
        gpu.rms_norm(
            &scratch.query,
            &scratch.query,
            Some(&self.q_norm),
            rows,
            inner,
            RMS_EPSILON,
        )
        .map_err(ConnectorError::Run)?;
        gpu.rms_norm(
            &scratch.key,
            &scratch.key,
            Some(&self.k_norm),
            rows,
            inner,
            RMS_EPSILON,
        )
        .map_err(ConnectorError::Run)?;

        gpu.text_rope(
            &scratch.query,
            &scratch.key,
            cos,
            sin,
            rows,
            heads as u32,
            scratch.head_dim as u32,
        )
        .map_err(ConnectorError::Run)?;

        // `scaled_dot_product_attention` at its default scale,
        // `1 / sqrt(dim_head)` (`attention.py:99-100` with no `scale`), no
        // mask (the register replacement zeroed it), and `PytorchAttention`'s
        // own operand convention: `q.view(b, -1, heads, dim_head)` reads the
        // projections token-major, which is what the linears above produce.
        let scale = 1.0 / (scratch.head_dim as f32).sqrt();
        gpu.text_attention(
            &scratch.attention,
            &scratch.query,
            &scratch.key,
            &scratch.value,
            rows,
            rows,
            heads as u32,
            scratch.head_dim as u32,
            scale,
        )
        .map_err(ConnectorError::Run)?;

        self.gate
            .apply(gpu, &scratch.logits, &scratch.normed, rows)?;
        gpu.per_head_gate(
            &scratch.gated,
            &scratch.attention,
            &scratch.logits,
            rows,
            heads as u32,
            scratch.head_dim as u32,
        )
        .map_err(ConnectorError::Run)?;

        self.output
            .apply(gpu, &scratch.projected, &scratch.gated, rows)?;
        gpu.add(&scratch.residual, input, &scratch.projected)
            .map_err(ConnectorError::Run)?;

        gpu.rms_norm(
            &scratch.feedforward_normed,
            &scratch.residual,
            None,
            rows,
            inner,
            RMS_EPSILON,
        )
        .map_err(ConnectorError::Run)?;

        self.up
            .apply(gpu, &scratch.feedforward, &scratch.feedforward_normed, rows)?;
        gpu.gelu_tanh(&scratch.feedforward, &scratch.feedforward)
            .map_err(ConnectorError::Run)?;
        self.down
            .apply(gpu, &scratch.compressed, &scratch.feedforward, rows)?;

        let output = alloc(gpu, Device::Bf16, rows as usize * scratch.inner)?;
        gpu.add(&output, &scratch.residual, &scratch.compressed)
            .map_err(ConnectorError::Run)?;

        Ok(output)
    }
}

/// The block-sized buffers one connector reuses, so eight blocks allocate one
/// set rather than eight.
struct Scratch {
    normed: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention: Tensor,
    logits: Tensor,
    gated: Tensor,
    projected: Tensor,
    residual: Tensor,
    feedforward: Tensor,
    feedforward_normed: Tensor,
    /// The feed-forward's `4 * inner` hidden.
    compressed: Tensor,
    inner: usize,
    head_dim: usize,
}

impl Scratch {
    fn new(gpu: &Gpu, rows: u32, inner: usize) -> Result<Scratch, ConnectorError> {
        let row = rows as usize;
        let head_dim = inner / HEADS;

        Ok(Scratch {
            normed: alloc(gpu, Device::Bf16, row * inner)?,
            query: alloc(gpu, Device::Bf16, row * inner)?,
            key: alloc(gpu, Device::Bf16, row * inner)?,
            value: alloc(gpu, Device::Bf16, row * inner)?,
            attention: alloc(gpu, Device::Bf16, row * inner)?,
            logits: alloc(gpu, Device::Bf16, row * HEADS)?,
            gated: alloc(gpu, Device::Bf16, row * inner)?,
            projected: alloc(gpu, Device::Bf16, row * inner)?,
            residual: alloc(gpu, Device::Bf16, row * inner)?,
            feedforward: alloc(gpu, Device::Bf16, row * inner * MULTIPLIER)?,
            feedforward_normed: alloc(gpu, Device::Bf16, row * inner)?,
            compressed: alloc(gpu, Device::Bf16, row * inner)?,
            inner,
            head_dim,
        })
    }
}

/// The two shards' reader: one reusable host buffer, and the counters a gate
/// reports. Every read checks the checkpoint's own descriptor against what the
/// graph will do with the tensor before anything reaches the device.
struct Reader<'a> {
    gpu: &'a Gpu,
    buffer: Vec<u8>,
    tensors: usize,
    bytes: u64,
}

impl Reader<'_> {
    fn pull(
        &mut self,
        shard: &Shard,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
    ) -> Result<Tensor, ConnectorError> {
        let tensor = shard
            .find(name)
            .ok_or_else(|| ConnectorError::Missing(name.to_string()))?;
        let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

        if tensor.dtype != dtype || tensor.shape != want {
            return Err(ConnectorError::Shape {
                name: name.to_string(),
                detail: format!(
                    "checkpoint has {} {:?}, the graph needs {} {:?}",
                    tensor.dtype.name(),
                    tensor.shape,
                    dtype.name(),
                    want
                ),
            });
        }

        let elements: u64 = want.iter().product();

        if tensor.elements() != elements || tensor.bytes() != elements * dtype.size() {
            return Err(ConnectorError::Shape {
                name: name.to_string(),
                detail: format!(
                    "{} elements in {} bytes do not match {} x {}",
                    tensor.elements(),
                    tensor.bytes(),
                    elements,
                    dtype.size()
                ),
            });
        }

        // One host buffer for the whole load: the aggregate embed is 770 MB
        // and a per-tensor buffer would keep all 6.3 GB resident.
        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| ConnectorError::Reader(error.token()))?;

        let device = alloc(self.gpu, device_dtype(dtype), elements as usize)?;
        self.gpu
            .upload(&device, &self.buffer)
            .map_err(ConnectorError::Run)?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }

    /// One `torch.nn.Linear` in BF16: weight `[out, in]`, bias `[out]`.
    fn linear(
        &mut self,
        shard: &Shard,
        name: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Linear, ConnectorError> {
        Ok(Linear {
            weight: self.pull(
                shard,
                &format!("{name}.weight"),
                Dtype::Bf16,
                &[output_dim, input_dim],
            )?,
            bias: self.pull(shard, &format!("{name}.bias"), Dtype::Bf16, &[output_dim])?,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        })
    }

    /// One `Embeddings1DConnector`'s blocks and registers.
    fn connector(
        &mut self,
        shard: &Shard,
        flavour: &str,
        inner: usize,
        head_dim: usize,
    ) -> Result<Connector, ConnectorError> {
        let heads = inner / head_dim;
        let hidden = inner * MULTIPLIER;
        let prefix = format!("model.diffusion_model.{flavour}_embeddings_connector");
        let mut blocks = Vec::with_capacity(BLOCKS);

        for block in 0..BLOCKS {
            let at = format!("{prefix}.transformer_1d_blocks.{block}");
            blocks.push(Block {
                q_norm: self.pull(
                    shard,
                    &format!("{at}.attn1.q_norm.weight"),
                    Dtype::Bf16,
                    &[inner],
                )?,
                k_norm: self.pull(
                    shard,
                    &format!("{at}.attn1.k_norm.weight"),
                    Dtype::Bf16,
                    &[inner],
                )?,
                query: self.linear(shard, &format!("{at}.attn1.to_q"), inner, inner)?,
                key: self.linear(shard, &format!("{at}.attn1.to_k"), inner, inner)?,
                value: self.linear(shard, &format!("{at}.attn1.to_v"), inner, inner)?,
                output: self.linear(shard, &format!("{at}.attn1.to_out.0"), inner, inner)?,
                gate: self.linear(shard, &format!("{at}.attn1.to_gate_logits"), inner, heads)?,
                up: self.linear(shard, &format!("{at}.ff.net.0.proj"), inner, hidden)?,
                down: self.linear(shard, &format!("{at}.ff.net.2"), hidden, inner)?,
            });
        }

        let registers = self.pull(
            shard,
            &format!("{prefix}.learnable_registers"),
            Dtype::Bf16,
            &[REGISTERS, inner],
        )?;

        Ok(Connector {
            blocks,
            registers,
            inner,
            head_dim,
        })
    }
}

/// `_compute_right_pad_order` (`embeddings_processor.py:23-38`): the stable
/// descending argsort of the binary mask, which puts the valid tokens first in
/// their original order, and the mask the connector is handed (`[1; 0]`). Both
/// depend only on the mask, so they are computed once on the host and shared
/// by the two modalities, as the reference shares them.
fn right_pad(padding: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut order: Vec<u32> = (0..padding.len() as u32)
        .filter(|index| padding[*index as usize] != 0)
        .collect();
    order.extend((0..padding.len() as u32).filter(|index| padding[*index as usize] == 0));

    let valid = padding.iter().filter(|value| **value != 0).count();
    let binary: Vec<u32> = (0..padding.len() as u32)
        .map(|index| u32::from((index as usize) < valid))
        .collect();

    (order, binary)
}

fn device_dtype(dtype: Dtype) -> Device {
    match dtype {
        Dtype::Bf16 => Device::Bf16,
        Dtype::F32 => Device::F32,
        _ => Device::Bf16,
    }
}

fn alloc(gpu: &Gpu, dtype: Device, elements: usize) -> Result<Tensor, ConnectorError> {
    gpu.alloc(dtype, elements).map_err(ConnectorError::Run)
}

fn upload_u32(gpu: &Gpu, values: &[u32]) -> Result<Tensor, ConnectorError> {
    let tensor = alloc(gpu, Device::U32, values.len())?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload(&tensor, bytes).map_err(ConnectorError::Run)?;

    Ok(tensor)
}

/// Upload F32 values as BF16, rounding each the way `__float2bfloat16` does.
fn upload_bf16(gpu: &Gpu, values: &[f32]) -> Result<Tensor, ConnectorError> {
    let mut bytes = Vec::with_capacity(values.len() * 2);

    for value in values {
        bytes.extend_from_slice(&((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes());
    }

    let tensor = alloc(gpu, Device::Bf16, values.len())?;
    gpu.upload(&tensor, &bytes).map_err(ConnectorError::Run)?;

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flat_width_is_the_aggregate_embeds_input() {
        // `encoder_configurator.py:195-196`: `flat_dim = embedding_dim *
        // num_layers`, with `num_layers = num_hidden_layers + 1` for the
        // embedding state. 49 * 3840 = 188160, which is the checkpoint's own
        // `[4096, 188160]` and `[2048, 188160]`.
        assert_eq!(HIDDEN * STATES, FLAT);
        assert_eq!(FLAT, 188_160);
    }

    #[test]
    fn the_connectors_are_the_configs_own_geometry() {
        // `connector_num_attention_heads` 32 x 128 = 4096 and
        // `audio_connector_attention_head_dim` 64 -> 2048.
        assert_eq!(HEADS * VIDEO_HEAD_DIM, VIDEO_INNER);
        assert_eq!(HEADS * AUDIO_HEAD_DIM, AUDIO_INNER);
        assert_eq!(VIDEO_INNER / HEADS, VIDEO_HEAD_DIM);
    }

    #[test]
    fn the_right_pad_order_is_the_stable_descending_argsort() {
        // A left-padded mask: the valid tail moves to the front, in order.
        let padding = [0, 0, 1, 1];
        let (order, binary) = right_pad(&padding);

        assert_eq!(order, vec![2, 3, 0, 1]);
        assert_eq!(binary, vec![1, 1, 0, 0]);

        // Already right-padded input is idempotent, as the reference notes.
        let (order, binary) = right_pad(&[1, 1, 0, 0]);

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(binary, vec![1, 1, 0, 0]);

        // No padding at all hands the connector every row.
        let (order, binary) = right_pad(&[1, 1, 1]);
        assert_eq!(order, vec![0, 1, 2]);
        assert_eq!(binary, vec![1, 1, 1]);
    }
}
