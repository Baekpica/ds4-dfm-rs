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
//! Deviation, named: the reference prefetches the next layers' weights on
//! extra threads (`h3_text_encoder.c:130-145`, `:649-707`) and submits each
//! layer with a stream sync. This port streams each layer's weights on one
//! thread right before its ops and never syncs between layers. Layer *i* only
//! reads weights *i*, and every upload is ordered on the same stream, so the
//! numbers are unchanged; the prefetch ring is a P9 performance item.
//!
//! The numerical contract is BF16 weights, BF16 activations, F32
//! accumulators, and the reference's operation order. The E4 gate compares an
//! FNV-1a hash over the BF16 output, so a change here is measured against
//! `fnv=1d9d71e8dd8544d6` for the recorded prompt.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_text::{Dtype, Gpu, Tensor};

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
}

impl ForwardError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            ForwardError::Weights(error) => format!("weights: {}", error.token()),
            ForwardError::Gpu(detail) => format!("gpu: {detail}"),
            ForwardError::Empty => "the prompt has no tokens".to_string(),
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

    run(&weights, &gpu, ids, layers.min(LAYERS))
}

/// Refuse an empty prompt before the device is touched.
fn require_tokens(ids: &[u32]) -> Result<(), ForwardError> {
    if ids.is_empty() {
        return Err(ForwardError::Empty);
    }

    Ok(())
}

/// The sixteen-op layer walk (`h3_text_encoder.c:411-463`).
fn run(
    weights: &TextWeights,
    gpu: &Gpu,
    ids: &[u32],
    layers: u32,
) -> Result<Embedding, ForwardError> {
    let tokens = ids.len();
    let hidden = count(tokens, HIDDEN);

    let rope = Rope::build(tokens);
    let cosines = allocate(gpu, Dtype::F32, count(tokens, ROPE_HALF as u64))?;
    let sines = allocate(gpu, Dtype::F32, count(tokens, ROPE_HALF as u64))?;
    gpu.upload(&cosines, &pack_f32(rope.cosines()))
        .map_err(ForwardError::Gpu)?;
    gpu.upload(&sines, &pack_f32(rope.sines()))
        .map_err(ForwardError::Gpu)?;

    let activations = Activations::allocate(gpu, tokens)?;
    let layer = LayerWeights::allocate(gpu)?;

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

    for index in 0..layers {
        layer.upload(gpu, weights, index)?;
        encode_layer(gpu, &layer, &activations, &cosines, &sines, tokens, index)?;
    }

    let mut values = vec![0u8; hidden * 2];
    gpu.read(&activations.hidden, &mut values)
        .map_err(ForwardError::Gpu)?;

    Ok(Embedding {
        tokens,
        width: HIDDEN as usize,
        values,
    })
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

    /// Read and upload the eleven tensors of `layer`.
    fn upload(&self, gpu: &Gpu, weights: &TextWeights, layer: u32) -> Result<(), ForwardError> {
        let parts = [
            (Part::InputNorm, &self.input_norm),
            (Part::Query, &self.query),
            (Part::Key, &self.key),
            (Part::Value, &self.value),
            (Part::QueryNorm, &self.query_norm),
            (Part::KeyNorm, &self.key_norm),
            (Part::Output, &self.output),
            (Part::PostNorm, &self.post_norm),
            (Part::Gate, &self.gate),
            (Part::Up, &self.up),
            (Part::Down, &self.down),
        ];

        for (part, tensor) in parts {
            let bytes = weights.bytes(role(layer, part))?;
            gpu.upload(tensor, bytes).map_err(|detail| {
                ForwardError::Gpu(format!("layer {layer} {part:?} upload: {detail}"))
            })?;
        }

        Ok(())
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
