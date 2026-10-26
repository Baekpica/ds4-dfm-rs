//! The MiniMax-H3 prompt-encoder forward.
//!
//! One host loop over the device primitives in `native/h3/h3_text_gpu.h`.
//! Source of truth `/data/h3.c` at `eae5d54`:
//!
//! - `h3_text_encoder.c:411-463` `encode_layer`: the sixteen ops of one layer,
//!   in this order: input RMSNorm, q/k/v projections, per-head RMSNorm on q
//!   and k, text RoPE, causal GQA, output projection, attention residual,
//!   post-attention RMSNorm, gate and up projections, fused SiLU multiply,
//!   down projection, MLP residual.
//! - `h3_text_encoder.c:543-567` the RoPE tables, built on the host. The
//!   prompt path passes `position_ids == NULL`, so the mRoPE axis interleave
//!   and the BF16 rounding do not apply (`minimax_h3::text_rope`).
//! - `h3_text_encoder.c:622-625` the embedding gather, and `:665-690` the
//!   50-layer loop.
//! - `h3_gpu_cuda.cu:733-779` `h3_linear_lt`: every projection is one
//!   cuBLASLt BF16 matmul with the descriptors the reference sets.
//!
//! The reference prefetch ring (`h3_text_encoder.c:124-143`, `:324-390`,
//! `:648-730`) uses the same
//! `H3_QWEN_PREFETCH` lanes and `H3_QWEN_PREFETCH_DEPTH` ring, reading each
//! layer into host buffers while the previous layer runs on the device. It
//! changes when bytes are read, never which bytes: the gate hash is the
//! acceptance test.
//!
//! The numerical contract is BF16 weights, BF16 activations, F32
//! accumulators, and the reference's operation order. The E4 gate compares an
//! FNV-1a hash over the BF16 output, so a change here is measured against
//! `fnv=1d9d71e8dd8544d6` for the recorded prompt.

use std::fmt;
use std::path::Path;
use std::thread::{Scope, ScopedJoinHandle};

use ds4_sys::h3_text::{Dtype, Gpu, Tensor};

use crate::minimax_h3::multimodal::VisionSpan;
use crate::minimax_h3::text::{
    Part, Role, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, QUERY_HEADS, RMS_EPSILON, VOCAB,
};
use crate::minimax_h3::text_rope::{Rope, ROPE_HALF};
use crate::minimax_h3::text_weights::{TextLoadError, TextWeights};

/// The probe's seed (`tools/minimax_h3/text_probe.c:37`). It is not the
/// canonical FNV-1a offset basis: the probe writes
/// `UINT64_C(1469598103934665603)`, and the recorded golden
/// `1d9d71e8dd8544d6` was produced with that value. Change it and the gate
/// stops comparing the same function.
pub const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;

/// FNV-1a 64 prime (`tools/minimax_h3/text_probe.c:42`).
pub const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a over raw bytes, the same loop the probe runs over BF16 values.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

/// The prompt encoder's output: one BF16 row per token, `HIDDEN` wide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    tokens: usize,
    width: usize,
    values: Vec<u8>,
    tags: Option<Vec<u8>>,
}

impl Embedding {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// The BF16 payload exactly as the device wrote it.
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    /// The reference gate's hash of the same payload.
    pub fn fnv(&self) -> u64 {
        fnv1a(&self.values)
    }

    /// The DiT modality tag of every row: 0 over a vision span and its two
    /// boundary ids, 1 elsewhere. Only the multimodal entry point returns them
    /// (`h3_text_encoder.c:737-745`).
    pub fn tags(&self) -> Option<&[u8]> {
        self.tags.as_deref()
    }

    /// FNV-1a over the tags, the probe's convention.
    pub fn tags_fnv(&self) -> Option<u64> {
        self.tags.as_deref().map(fnv1a)
    }

    /// The first `count` BF16 bit patterns, for an edge comparison.
    pub fn first_words(&self, count: usize) -> Vec<u16> {
        words(&self.values, 0, count)
    }

    /// The last `count` BF16 bit patterns, for an edge comparison.
    pub fn last_words(&self, count: usize) -> Vec<u16> {
        let end = self.values.len() / 2;
        words(&self.values, end.saturating_sub(count), count)
    }
}

fn words(values: &[u8], start: usize, count: usize) -> Vec<u16> {
    let (pairs, _) = values.as_chunks::<2>();

    pairs
        .iter()
        .skip(start)
        .take(count)
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// Why the forward could not run.
#[derive(Debug)]
pub enum ForwardError {
    /// The checkpoint is not the encoder the layout proved.
    Weights(TextLoadError),
    /// A device call failed; the native side's message.
    Gpu(String),
    /// No token to encode.
    Empty,
    /// A presentation the reference refuses (`h3_text_encoder.c:493-515`).
    Invalid(&'static str),
}

impl ForwardError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            ForwardError::Weights(error) => format!("weights: {}", error.token()),
            ForwardError::Gpu(detail) => format!("gpu: {detail}"),
            ForwardError::Empty => "the prompt has no tokens".to_string(),
            ForwardError::Invalid(detail) => {
                format!("invalid multimodal presentation: {detail}")
            }
        }
    }
}

impl fmt::Display for ForwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for ForwardError {}

impl From<TextLoadError> for ForwardError {
    fn from(error: TextLoadError) -> ForwardError {
        ForwardError::Weights(error)
    }
}

/// `1.0f / sqrtf(TEXT_HEAD_DIM)` (`h3_text_encoder.c:457-458`).
pub fn attention_scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

/// Encode `ids` with the checkpoint under `dir`.
pub fn encode(dir: &Path, ids: &[u32]) -> Result<Embedding, ForwardError> {
    encode_layers(dir, ids, LAYERS)
}

/// Encode `ids` through the first `layers` layers, for the E4 bisection: the
/// reference exposes the same prefix entry
/// (`h3_text_encode_multimodal_layers_bf16`, `h3_text_encoder.h:44-52`).
pub fn encode_layers(dir: &Path, ids: &[u32], layers: u32) -> Result<Embedding, ForwardError> {
    require_tokens(ids)?;

    let weights = TextWeights::open(dir)?;
    let gpu = Gpu::create().map_err(ForwardError::Gpu)?;

    run(&weights, &gpu, ids, layers.min(LAYERS), None)
}

/// Encode a prompt that carries Qwen vision spans
/// (`h3_text_encode_multimodal_bf16`, `h3_text_encoder.c:478-760`).
///
/// The three differences from [`encode_layers`] are the splice of the tower's
/// rows over the embedding lookup, the deepstack add after layers 0, 1 and 2,
/// and the per-token mRoPE tables. Everything else is the same layer walk, so
/// the E4 golden still covers the shared path.
pub fn encode_multimodal(
    dir: &Path,
    ids: &[u32],
    spans: &[VisionSpan<'_>],
    positions: &[u32],
    tags: &[u8],
) -> Result<Embedding, ForwardError> {
    require_tokens(ids)?;

    let weights = TextWeights::open(dir)?;
    let gpu = Gpu::create().map_err(ForwardError::Gpu)?;
    let presentation = Multimodal {
        spans,
        positions,
        tags,
    };

    run(&weights, &gpu, ids, LAYERS, Some(&presentation))
}

/// Refuse an empty prompt before the device is touched.
fn require_tokens(ids: &[u32]) -> Result<(), ForwardError> {
    if ids.is_empty() {
        return Err(ForwardError::Empty);
    }

    Ok(())
}

/// The eleven tensors of one layer in the reference's read order
/// (`h3_text_encoder.c:283-306`). The position is also the lane's item
/// counter, so this order decides which lane reads which tensor.
const PARTS: [Part; 11] = [
    Part::InputNorm,
    Part::Query,
    Part::Key,
    Part::Value,
    Part::QueryNorm,
    Part::KeyNorm,
    Part::Output,
    Part::PostNorm,
    Part::Gate,
    Part::Up,
    Part::Down,
];

/// `H3_QWEN_PREFETCH` (`h3_text_encoder.c:124-133`): reader threads per
/// layer, eight by default.
const PREFETCH_LANES: usize = 8;

/// `H3_QWEN_PREFETCH_DEPTH` (`h3_text_encoder.c:135-143`): layers in flight,
/// two by default.
const PREFETCH_DEPTH: usize = 2;

const PREFETCH_DEPTH_MAX: usize = 6;

/// `H3_QWEN_PREFETCH`: zero disables the ring, values above the cap clamp, and
/// anything unreadable is one lane (`text_prefetch_threads`, `:124-133`).
fn prefetch_lanes() -> usize {
    match std::env::var("H3_QWEN_PREFETCH") {
        Ok(value) if value.trim() == "0" => 0,
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|lanes| *lanes >= 1)
            .unwrap_or(1)
            .min(PREFETCH_LANES),
        Err(_) => PREFETCH_LANES,
    }
}

/// `H3_QWEN_PREFETCH_DEPTH`: one layer when the value is unreadable, capped at
/// six (`text_prefetch_depth`, `:135-143`).
fn prefetch_depth() -> usize {
    match std::env::var("H3_QWEN_PREFETCH_DEPTH") {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|depth| *depth >= 1)
            .unwrap_or(1)
            .min(PREFETCH_DEPTH_MAX),
        Err(_) => PREFETCH_DEPTH,
    }
}

/// What one lane returns: its tensors, each tagged with its [`PARTS`] index.
type LaneItems = Result<Vec<(usize, Vec<u8>)>, ForwardError>;

/// One layer's read in flight (`text_prefetch_slot`, `h3_text_encoder.c:75-81`).
struct Slot<'scope> {
    layer: u32,
    lanes: Vec<ScopedJoinHandle<'scope, LaneItems>>,
}

impl<'scope> Slot<'scope> {
    fn new(layer: u32) -> Self {
        Slot {
            layer,
            lanes: Vec::new(),
        }
    }
}

/// Start reading `layer` across `lanes` threads, each taking the tensors whose
/// position in [`PARTS`] it owns (`item++ % lanes == lane`,
/// `h3_text_encoder.c:267-277`, `:324-359`).
fn start_read<'scope>(
    scope: &'scope Scope<'scope, '_>,
    weights: &'scope TextWeights,
    layer: u32,
    lanes: usize,
) -> Slot<'scope> {
    let mut per_lane: Vec<Vec<usize>> = (0..lanes).map(|_| Vec::new()).collect();
    for part in 0..PARTS.len() {
        per_lane[part % lanes].push(part);
    }

    let mut slot = Slot::new(layer);
    for parts in per_lane {
        slot.lanes
            .push(scope.spawn(move || read_items(weights, layer, parts)));
    }

    slot
}

/// One lane's share: `pread` its tensors into buffers it owns.
fn read_items(
    weights: &TextWeights,
    layer: u32,
    parts: Vec<usize>,
) -> Result<Vec<(usize, Vec<u8>)>, ForwardError> {
    let mut items = Vec::with_capacity(parts.len());

    for part in parts {
        let mut buffer = Vec::new();
        weights.read_into(role(layer, PARTS[part]), &mut buffer)?;
        items.push((part, buffer));
    }

    Ok(items)
}

/// Read one layer on this thread, for layer 0 and for the ring turned off
/// (`layer_weights_load`, `h3_text_encoder.c:145-180`).
fn read_sync(weights: &TextWeights, layer: u32) -> Result<Vec<Vec<u8>>, ForwardError> {
    let mut buffers = Vec::with_capacity(PARTS.len());

    for part in PARTS {
        let mut buffer = Vec::new();
        weights.read_into(role(layer, part), &mut buffer)?;
        buffers.push(buffer);
    }

    Ok(buffers)
}

/// Join every lane and order the buffers as [`PARTS`]
/// (`prefetch_slot_take`, `h3_text_encoder.c:361-390`).
fn take(slot: &mut Slot<'_>) -> Result<Vec<Vec<u8>>, ForwardError> {
    let mut ordered: Vec<Option<Vec<u8>>> = (0..PARTS.len()).map(|_| None).collect();

    for lane in slot.lanes.drain(..) {
        let items = lane
            .join()
            .map_err(|_| ForwardError::Gpu("a Qwen prefetch lane panicked".to_string()))??;

        for (part, buffer) in items {
            ordered[part] = Some(buffer);
        }
    }

    ordered
        .into_iter()
        .map(|buffer| {
            buffer.ok_or_else(|| {
                ForwardError::Gpu("a Qwen prefetch lane returned no tensor".to_string())
            })
        })
        .collect()
}

/// The sixteen-op layer walk (`h3_text_encoder.c:411-463`).
fn run(
    weights: &TextWeights,
    gpu: &Gpu,
    ids: &[u32],
    layers: u32,
    presentation: Option<&Multimodal<'_>>,
) -> Result<Embedding, ForwardError> {
    let tokens = ids.len();
    let hidden = count(tokens, HIDDEN);

    if let Some(presentation) = presentation {
        check_presentation(ids, presentation)?;
    }

    let rope = match presentation {
        Some(presentation) => Rope::build_multimodal(tokens, presentation.positions)
            .ok_or(ForwardError::Invalid("the mRoPE block is not [3, tokens]"))?,
        None => Rope::build(tokens),
    };
    let cosines = allocate(gpu, Dtype::F32, count(tokens, ROPE_HALF as u64))?;
    let sines = allocate(gpu, Dtype::F32, count(tokens, ROPE_HALF as u64))?;
    gpu.upload(&cosines, &pack_f32(rope.cosines()))
        .map_err(ForwardError::Gpu)?;
    gpu.upload(&sines, &pack_f32(rope.sines()))
        .map_err(ForwardError::Gpu)?;

    let activations = Activations::allocate(gpu, tokens)?;
    let layer = LayerWeights::allocate(gpu)?;

    // `h3_text_encoder.c:585-605`: one zero-filled `[tokens, 5120]` buffer per
    // deepstack, each span's rows copied at that span's own start.
    let deepstack = match presentation {
        Some(presentation) => Some(pack_deepstacks(gpu, tokens, presentation.spans)?),
        None => None,
    };

    // The embedding is read once, gathered, and released before the layers
    // stream their own weights (`h3_text_encoder.c:622-625`, `:659`).
    {
        let embedding = allocate(gpu, Dtype::Bf16, VOCAB as usize * HIDDEN as usize)?;
        gpu.upload(
            &embedding,
            weights
                .bytes(Role::Embedding)
                .map_err(ForwardError::Weights)?,
        )
        .map_err(ForwardError::Gpu)?;
        gpu.embed(
            &activations.hidden,
            &embedding,
            ids,
            VOCAB as u32,
            HIDDEN as u32,
        )
        .map_err(ForwardError::Gpu)?;
    }

    // `h3_text_encoder.c:638-646`: each span's merged rows replace the gathered
    // embedding rows at its start.
    if let Some(presentation) = presentation {
        for span in presentation.spans {
            let width = HIDDEN as usize;
            let bytes = span.tokens * width * 2;

            gpu.write(
                &activations.hidden,
                span.start * width,
                &span.merged[..bytes],
            )
            .map_err(ForwardError::Gpu)?;
        }
    }

    // The reference reads layer 0 inline, keeps `depth` layers in flight, and
    // starts the next read once the current layer is submitted, so the host
    // reads while the device computes (`h3_text_encoder.c:648-730`).
    let lanes = prefetch_lanes();
    let depth = if lanes == 0 || layers < 2 {
        0
    } else {
        prefetch_depth()
    };

    std::thread::scope(|scope| -> Result<(), ForwardError> {
        let mut ring: Vec<Option<Slot>> = (0..depth).map(|_| None).collect();
        let mut next_layer = 1u32;

        // Fill the ring before the loop; each slot owns a distinct layer.
        let prefill = depth.min(layers.saturating_sub(1) as usize);
        for slot in ring.iter_mut().take(prefill) {
            *slot = Some(start_read(scope, weights, next_layer, lanes));
            next_layer += 1;
        }

        for index in 0..layers {
            let position = ring
                .iter()
                .position(|slot| slot.as_ref().is_some_and(|slot| slot.layer == index));
            let (mut slot, position, buffers) = match position {
                Some(position) => {
                    let mut slot = ring[position].take().expect("the slot was found");
                    let buffers = take(&mut slot)?;
                    (slot, Some(position), buffers)
                }
                None if index == 0 || depth == 0 => {
                    (Slot::new(index), None, read_sync(weights, index)?)
                }
                None => {
                    return Err(ForwardError::Gpu(format!(
                        "Qwen layer {index} is absent from the prefetch ring"
                    )))
                }
            };

            layer.upload_buffers(gpu, &buffers, index)?;
            encode_layer(gpu, &layer, &activations, &cosines, &sines, tokens, index)?;

            // `h3_text_encoder.c:675-679`: the deepstack row is added to the
            // layer output, before the next layer reads it.
            if let Some(rows) = deepstack.as_ref().filter(|_| index < 3) {
                gpu.add(
                    &activations.hidden,
                    &activations.hidden,
                    &rows[index as usize],
                    hidden,
                )
                .map_err(ForwardError::Gpu)?;
            }

            // The buffers have been staged for upload, so their slot can take
            // the next layer and the lanes read while the device works.
            if let Some(position) = position {
                if next_layer < layers {
                    slot = start_read(scope, weights, next_layer, lanes);
                    next_layer += 1;
                }
                ring[position] = Some(slot);
            }
        }

        Ok(())
    })?;

    let mut values = vec![0u8; hidden * 2];
    gpu.read(&activations.hidden, &mut values)
        .map_err(ForwardError::Gpu)?;

    Ok(Embedding {
        tokens,
        width: HIDDEN as usize,
        values,
        tags: presentation.map(|presentation| presentation.tags.to_vec()),
    })
}

/// The presentation the multimodal entry adds to the plain encoder
/// (`h3_text_encoder.h:40-52`).
#[derive(Clone, Copy)]
struct Multimodal<'a> {
    spans: &'a [VisionSpan<'a>],
    positions: &'a [u32],
    tags: &'a [u8],
}

/// `h3_text_encoder.c:493-515`: every id inside the vocabulary, every tag at
/// most 2, and the spans strictly increasing, non-overlapping, inside the
/// sequence and backed by the rows the splice copies.
fn check_presentation(ids: &[u32], presentation: &Multimodal<'_>) -> Result<(), ForwardError> {
    let tokens = ids.len();
    let width = HIDDEN as usize;

    if presentation.positions.len() != 3 * tokens {
        return Err(ForwardError::Invalid("the mRoPE block is not [3, tokens]"));
    }
    if presentation.tags.len() != tokens {
        return Err(ForwardError::Invalid("the tag row does not cover the ids"));
    }
    if ids.iter().any(|id| u64::from(*id) >= VOCAB) {
        return Err(ForwardError::Invalid("an id is outside the vocabulary"));
    }
    if presentation.tags.iter().any(|tag| *tag > 2) {
        return Err(ForwardError::Invalid("a presentation tag is above 2"));
    }

    let mut cursor = 0;
    for span in presentation.spans {
        if span.tokens == 0 {
            return Err(ForwardError::Invalid("a span has no rows"));
        }
        if span.start < cursor || span.start > tokens || span.tokens > tokens - span.start {
            return Err(ForwardError::Invalid(
                "a span is out of order or outside the ids",
            ));
        }
        let bytes = span.tokens * width * 2;
        if span.merged.len() < bytes || span.deepstack.iter().any(|payload| payload.len() < bytes) {
            return Err(ForwardError::Invalid(
                "a span payload is shorter than its rows",
            ));
        }
        cursor = span.start + span.tokens;
    }

    Ok(())
}

/// `h3_text_encoder.c:585-605`: pack one `[tokens, 5120]` buffer per deepstack
/// and upload it. The zero fill matters: every row outside a span contributes
/// nothing to the add.
fn pack_deepstacks(
    gpu: &Gpu,
    tokens: usize,
    spans: &[VisionSpan<'_>],
) -> Result<[Tensor; 3], ForwardError> {
    let width = HIDDEN as usize;
    let mut values = vec![0u8; tokens * width * 2];

    let mut pack = |layer: usize| -> Result<Tensor, ForwardError> {
        values.fill(0);
        for span in spans {
            let start = span.start * width * 2;
            let bytes = span.tokens * width * 2;

            values[start..start + bytes].copy_from_slice(&span.deepstack[layer][..bytes]);
        }
        let tensor = allocate(gpu, Dtype::Bf16, tokens * width)?;
        gpu.upload(&tensor, &values).map_err(ForwardError::Gpu)?;

        Ok(tensor)
    };

    Ok([pack(0)?, pack(1)?, pack(2)?])
}

/// One layer, op for op, with the reference's names in the error text.
fn encode_layer(
    gpu: &Gpu,
    weight: &LayerWeights,
    activation: &Activations,
    cosines: &Tensor,
    sines: &Tensor,
    tokens: usize,
    layer: u32,
) -> Result<(), ForwardError> {
    let rows = tokens as u32;
    let query = query_width() as u32;
    let kv = kv_width() as u32;
    let operation = |result: Result<(), String>, label: &str| {
        result.map_err(|detail| ForwardError::Gpu(format!("Qwen layer {layer} {label}: {detail}")))
    };
    let rows_hidden = count(tokens, HIDDEN);
    let rows_intermediate = count(tokens, INTERMEDIATE);

    operation(
        gpu.rms_norm(
            &activation.norm,
            &activation.hidden,
            &weight.input_norm,
            rows,
            HIDDEN as u32,
            RMS_EPSILON as f32,
        ),
        "input RMSNorm",
    )?;
    operation(
        gpu.linear(
            &activation.query,
            &activation.norm,
            &weight.query,
            rows,
            HIDDEN as u32,
            query,
        ),
        "query projection",
    )?;
    operation(
        gpu.linear(
            &activation.key,
            &activation.norm,
            &weight.key,
            rows,
            HIDDEN as u32,
            kv,
        ),
        "key projection",
    )?;
    operation(
        gpu.linear(
            &activation.value,
            &activation.norm,
            &weight.value,
            rows,
            HIDDEN as u32,
            kv,
        ),
        "value projection",
    )?;
    operation(
        gpu.head_rms_norm(
            &activation.query,
            &weight.query_norm,
            rows,
            QUERY_HEADS as u32,
            HEAD_DIM as u32,
            RMS_EPSILON as f32,
        ),
        "query RMSNorm",
    )?;
    operation(
        gpu.head_rms_norm(
            &activation.key,
            &weight.key_norm,
            rows,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            RMS_EPSILON as f32,
        ),
        "key RMSNorm",
    )?;
    operation(
        gpu.rope_text(
            &activation.query,
            &activation.key,
            cosines,
            sines,
            rows,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
        ),
        "RoPE",
    )?;
    operation(
        gpu.gqa_causal(
            &activation.attention_heads,
            &activation.query,
            &activation.key,
            &activation.value,
            rows,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            attention_scale(),
        ),
        "causal GQA",
    )?;
    operation(
        gpu.linear(
            &activation.attention_output,
            &activation.attention_heads,
            &weight.output,
            rows,
            query,
            HIDDEN as u32,
        ),
        "attention output projection",
    )?;
    operation(
        gpu.add(
            &activation.hidden,
            &activation.hidden,
            &activation.attention_output,
            rows_hidden,
        ),
        "attention residual",
    )?;
    operation(
        gpu.rms_norm(
            &activation.norm,
            &activation.hidden,
            &weight.post_norm,
            rows,
            HIDDEN as u32,
            RMS_EPSILON as f32,
        ),
        "post-attention RMSNorm",
    )?;
    operation(
        gpu.linear(
            &activation.gate,
            &activation.norm,
            &weight.gate,
            rows,
            HIDDEN as u32,
            INTERMEDIATE as u32,
        ),
        "MLP gate",
    )?;
    operation(
        gpu.linear(
            &activation.up,
            &activation.norm,
            &weight.up,
            rows,
            HIDDEN as u32,
            INTERMEDIATE as u32,
        ),
        "MLP up",
    )?;
    operation(
        gpu.silu_mul(
            &activation.gate,
            &activation.gate,
            &activation.up,
            rows_intermediate,
        ),
        "fused SwiGLU",
    )?;
    operation(
        gpu.linear(
            &activation.mlp_output,
            &activation.gate,
            &weight.down,
            rows,
            INTERMEDIATE as u32,
            HIDDEN as u32,
        ),
        "MLP down",
    )?;
    operation(
        gpu.add(
            &activation.hidden,
            &activation.hidden,
            &activation.mlp_output,
            rows_hidden,
        ),
        "MLP residual",
    )?;

    Ok(())
}

/// The eleven layer tensors, allocated once and reused by every layer: their
/// shapes do not vary, and an upload is stream-ordered before the next layer's
/// ops.
struct LayerWeights {
    input_norm: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    output: Tensor,
    post_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl LayerWeights {
    fn allocate(gpu: &Gpu) -> Result<LayerWeights, ForwardError> {
        let hidden = HIDDEN as usize;
        let query = query_width() as usize;
        let kv = kv_width() as usize;
        let intermediate = INTERMEDIATE as usize;

        Ok(LayerWeights {
            input_norm: allocate(gpu, Dtype::Bf16, hidden)?,
            query: allocate(gpu, Dtype::Bf16, query * hidden)?,
            key: allocate(gpu, Dtype::Bf16, kv * hidden)?,
            value: allocate(gpu, Dtype::Bf16, kv * hidden)?,
            query_norm: allocate(gpu, Dtype::Bf16, HEAD_DIM as usize)?,
            key_norm: allocate(gpu, Dtype::Bf16, HEAD_DIM as usize)?,
            output: allocate(gpu, Dtype::Bf16, hidden * query)?,
            post_norm: allocate(gpu, Dtype::Bf16, hidden)?,
            gate: allocate(gpu, Dtype::Bf16, intermediate * hidden)?,
            up: allocate(gpu, Dtype::Bf16, intermediate * hidden)?,
            down: allocate(gpu, Dtype::Bf16, hidden * intermediate)?,
        })
    }

    /// Upload the eleven host buffers of `layer`, in [`PARTS`] order.
    fn upload_buffers(
        &self,
        gpu: &Gpu,
        buffers: &[Vec<u8>],
        layer: u32,
    ) -> Result<(), ForwardError> {
        if buffers.len() != PARTS.len() {
            return Err(ForwardError::Gpu(format!(
                "layer {layer} has {} buffers, expected {}",
                buffers.len(),
                PARTS.len()
            )));
        }

        for (part, bytes) in PARTS.iter().zip(buffers) {
            gpu.upload(self.tensor(*part), bytes).map_err(|detail| {
                ForwardError::Gpu(format!("layer {layer} {part:?} upload: {detail}"))
            })?;
        }

        Ok(())
    }

    /// The device tensor for one part of the layer.
    fn tensor(&self, part: Part) -> &Tensor {
        match part {
            Part::InputNorm => &self.input_norm,
            Part::Query => &self.query,
            Part::Key => &self.key,
            Part::Value => &self.value,
            Part::QueryNorm => &self.query_norm,
            Part::KeyNorm => &self.key_norm,
            Part::Output => &self.output,
            Part::PostNorm => &self.post_norm,
            Part::Gate => &self.gate,
            Part::Up => &self.up,
            Part::Down => &self.down,
        }
    }
}

/// The shared activations, allocated once for the whole run.
struct Activations {
    hidden: Tensor,
    norm: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_heads: Tensor,
    attention_output: Tensor,
    gate: Tensor,
    up: Tensor,
    mlp_output: Tensor,
}

impl Activations {
    fn allocate(gpu: &Gpu, tokens: usize) -> Result<Activations, ForwardError> {
        let hidden = count(tokens, HIDDEN);
        let query = count(tokens, query_width());
        let kv = count(tokens, kv_width());
        let intermediate = count(tokens, INTERMEDIATE);

        Ok(Activations {
            hidden: allocate(gpu, Dtype::Bf16, hidden)?,
            norm: allocate(gpu, Dtype::Bf16, hidden)?,
            query: allocate(gpu, Dtype::Bf16, query)?,
            key: allocate(gpu, Dtype::Bf16, kv)?,
            value: allocate(gpu, Dtype::Bf16, kv)?,
            attention_heads: allocate(gpu, Dtype::Bf16, query)?,
            attention_output: allocate(gpu, Dtype::Bf16, hidden)?,
            gate: allocate(gpu, Dtype::Bf16, intermediate)?,
            up: allocate(gpu, Dtype::Bf16, intermediate)?,
            mlp_output: allocate(gpu, Dtype::Bf16, hidden)?,
        })
    }
}

/// `Role::Layer` for one part.
pub fn role(layer: u32, part: Part) -> Role {
    Role::Layer { layer, part }
}

fn allocate(gpu: &Gpu, dtype: Dtype, elements: usize) -> Result<Tensor, ForwardError> {
    gpu.alloc(dtype, elements).map_err(ForwardError::Gpu)
}

fn count(tokens: usize, width: u64) -> usize {
    tokens * width as usize
}

fn query_width() -> u64 {
    QUERY_HEADS * HEAD_DIM
}

fn kv_width() -> u64 {
    KV_HEADS * HEAD_DIM
}

/// Little-endian F32 bytes for an upload; the host and the target are both
/// little-endian, and this keeps the conversion checkable.
fn pack_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lane_split_covers_all_eleven_tensors_once() {
        // The reference assigns tensor `item` to lane `item % lanes`
        // (`h3_text_encoder.c:267-277`), so any lane count up to the cap must
        // partition the layer exactly once.
        for lanes in 1..=PREFETCH_LANES {
            let mut seen: Vec<usize> = Vec::new();

            for lane in 0..lanes {
                for part in 0..PARTS.len() {
                    if part % lanes == lane {
                        seen.push(part);
                    }
                }
            }

            seen.sort_unstable();
            assert_eq!(seen, (0..PARTS.len()).collect::<Vec<_>>(), "lanes {lanes}");
        }
    }

    #[test]
    fn the_prefetch_knobs_clamp_like_the_reference() {
        // `text_prefetch_threads` / `text_prefetch_depth`
        // (`h3_text_encoder.c:124-143`): zero lanes disables the ring, an
        // unreadable depth is one layer, and both clamp at their caps.
        assert_eq!(env_lanes("0"), 0);
        assert_eq!(env_lanes("3"), 3);
        assert_eq!(env_lanes("99"), PREFETCH_LANES);
        assert_eq!(env_lanes("nonsense"), 1);
        assert_eq!(env_lanes(""), 1);
        assert_eq!(env_depth("1"), 1);
        assert_eq!(env_depth("6"), 6);
        assert_eq!(env_depth("7"), PREFETCH_DEPTH_MAX);
        assert_eq!(env_depth("nonsense"), 1);
    }

    /// The knob parsers without touching the process environment.
    fn env_lanes(value: &str) -> usize {
        if value.trim() == "0" {
            return 0;
        }

        value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|lanes| *lanes >= 1)
            .unwrap_or(1)
            .min(PREFETCH_LANES)
    }

    fn env_depth(value: &str) -> usize {
        value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|depth| *depth >= 1)
            .unwrap_or(1)
            .min(PREFETCH_DEPTH_MAX)
    }

    #[test]
    fn fnv_matches_the_reference_probe_vectors() {
        assert_eq!(fnv1a(b""), 0x1465_0fb0_739d_0383);
        assert_eq!(fnv1a(b"a"), 0x44bd_8ad4_73cd_9906);
        assert_eq!(fnv1a(b"foobar"), 0x88fa_d7c0_a8ff_07f2);
    }

    #[test]
    fn an_empty_prompt_is_refused_before_the_device() {
        let error = require_tokens(&[]).expect_err("no tokens");

        assert!(matches!(error, ForwardError::Empty), "{error}");
        assert!(require_tokens(&[32]).is_ok());
    }

    #[test]
    fn the_attention_scale_is_the_reference_expression() {
        assert_eq!(
            attention_scale().to_bits(),
            (1.0f32 / 128f32.sqrt()).to_bits()
        );
    }

    #[test]
    fn f32_packing_is_little_endian() {
        assert_eq!(
            pack_f32(&[1.0, -2.0]),
            vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]
        );
    }

    #[test]
    fn the_embedding_edges_are_bf16_words() {
        let embedding = Embedding {
            tokens: 2,
            width: 2,
            values: vec![0x99, 0x3f, 0x71, 0xbf, 0xf4, 0xbe, 0x00, 0x40],
            tags: None,
        };

        assert_eq!(embedding.first_words(2), vec![0x3f99, 0xbf71]);
        assert_eq!(embedding.last_words(2), vec![0xbef4, 0x4000]);
        assert_eq!(
            embedding.first_words(99),
            vec![0x3f99, 0xbf71, 0xbef4, 0x4000]
        );
        assert_eq!(
            embedding.fnv(),
            fnv1a(&embedding.values),
            "the hash is over the raw payload"
        );
    }

    #[test]
    fn every_layer_needs_eleven_roles() {
        let parts = [
            Part::InputNorm,
            Part::Query,
            Part::Key,
            Part::Value,
            Part::QueryNorm,
            Part::KeyNorm,
            Part::Output,
            Part::PostNorm,
            Part::Gate,
            Part::Up,
            Part::Down,
        ];

        assert_eq!(parts.len(), 11);
        assert_eq!(
            role(49, Part::Down),
            Role::Layer {
                layer: 49,
                part: Part::Down
            }
        );
    }
}
