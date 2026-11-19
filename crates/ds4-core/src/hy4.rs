//! Tencent Hy4-flash mixed-quant release. Four resident GGUF shards, a
//! row-read token embedding and four page-packed SSD Engram tables.
mod engram;
pub mod media;
pub(crate) use engram::{read_rows, Engram};
mod layout;
mod release;
pub(crate) mod sidecar;

pub(crate) use layout::layouts;
pub(crate) use release::root as release_root;
pub use release::snapshot_inputs;
pub(crate) use release::{
    chat_assets, identify, inventory, is_release, metadata, recipe, trunk_bytes,
};

use crate::{GgufFile, ValidateError};

pub(crate) const PREFILL_MAX: u32 = 8192;
/// 2,048 rows put ~64 routed rows on each expert; the planner halves to
/// `PREFILL_MIN` when a long context leaves no room for the workspace.
pub(crate) const PREFILL: u32 = 2048;
pub(crate) const PREFILL_MIN: u32 = 512;
pub(crate) const CONTEXT: u32 = 1_048_576;
/// DFly2 drafts against a Q8_0 copy of the BF16 output head: int8 quants
/// plus one F16 scale per 32 of the 120,832 x 3,072 weights.
pub(crate) const DRAFT_HEAD_BYTES: u64 = 120_832 * 3_072 + 120_832 * (3_072 / 32) * 2;
const ARCH: &[u8] = b"hy_v4_flash";
const SOURCE: &[u8] = b"d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959";

/// Published trunk recipes. MQ-Q5-Q8 is MQ-Q5 with its BF16 dense
/// projections at Q8_0 and the router at BF16 (`reproduction/derive_q8.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recipe {
    Mq5,
    Mq5Q8,
}

impl Recipe {
    const ALL: [Recipe; 2] = [Recipe::Mq5, Recipe::Mq5Q8];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Recipe::Mq5 => "MQ-Q5-SSD-ENGRAM-FP8",
            Recipe::Mq5Q8 => "MQ-Q5-Q8-SSD-ENGRAM-FP8",
        }
    }

    /// Drafting reads a runtime Q8_0 copy of a BF16 output head.
    pub(crate) fn bf16_head(self) -> bool {
        self == Recipe::Mq5
    }
}

pub(crate) fn validate(g: &GgufFile) -> Result<(), ValidateError> {
    if g.get_string("general.architecture") != Some(ARCH)
        || !Recipe::ALL
            .iter()
            .any(|r| g.get_string("general.quantization") == Some(r.name().as_bytes()))
        || g.get_string("general.source.revision") != Some(SOURCE)
        || g.get_string("hy4_mxq.part") != Some(b"trunk")
        || g.get_bool("hy4_mxq.engram.resident") != Some(false)
        || g.get_string("hy4_mxq.engram.storage") != Some(b"ssd_rows")
        || g.get_string("tokenizer.ggml.pre") != Some(b"hy_v4")
        || g.split_count() != 4
    {
        return Err(ValidateError::Token("Hy4 mixed-quant trunk required"));
    }
    for (suffix, value) in [
        ("block_count", 42),
        ("context_length", CONTEXT),
        ("embedding_length", 3072),
        ("vocab_size", 120832),
        ("leading_dense_block_count", 1),
        ("feed_forward_length", 13824),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 576),
        ("attention.value_length", 512),
        ("attention.key_length_mla", 256),
        ("attention.value_length_mla", 256),
        ("attention.q_lora_rank", 1024),
        ("attention.kv_lora_rank", 512),
        ("attention.sliding_window", 512),
        ("attention.indexer.head_count", 16),
        ("attention.indexer.key_length", 128),
        ("attention.indexer.top_k", 2048),
        ("attention.indexer.kpool", 4),
        ("rope.dimension_count", 64),
        ("expert_count", 512),
        ("expert_used_count", 16),
        ("expert_shared_count", 1),
        ("expert_latent_length", 1536),
        ("expert_feed_forward_length", 1536),
        ("hyper_connection.count", 4),
        ("nextn_predict_layers", 0),
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 8),
        ("embedding_length_per_layer_input", 96),
        ("engram.pad_token_id", 120002),
    ] {
        let key = format!("hy_v4_flash.{suffix}");
        if g.get_u32(&key) != Some(value) {
            return Err(ValidateError::TokenKey("Hy4 geometry", key));
        }
    }
    for (suffix, expected) in [
        ("attention.layer_norm_rms_epsilon", 1e-5),
        ("rope.freq_base", 1e7),
        ("rope.freq_base_swa", 1e4),
        ("expert_weights_scale", 3.998),
        ("hyper_connection.epsilon", 1e-6),
        ("hyper_connection.magnitude", 2.0),
    ] {
        let key = format!("hy_v4_flash.{suffix}");
        if g.get_f32_compat(&key) != Some(expected) {
            return Err(ValidateError::TokenKey("Hy4 arithmetic", key));
        }
    }
    for (suffix, expected) in [
        (
            "attention.head_count",
            (0..42)
                .map(|i| if layout::sparse(i) { 32 } else { 64 })
                .collect(),
        ),
        ("rope.dimension_sections", vec![12, 10, 10, 0]),
        ("ple.layers", vec![1, 11, 21, 31]),
    ] {
        let key = format!("hy_v4_flash.{suffix}");
        let actual = g
            .get_array(&key)
            .filter(|a| a.len == expected.len() as u64)
            .and_then(|a| g.array_le_u32s(&a).ok());
        if actual.as_ref() != Some(&expected) {
            return Err(ValidateError::TokenKey("Hy4 array", key));
        }
    }
    for (suffix, sparse_flag) in [
        ("attention.sliding_window_pattern", false),
        ("attention.indexer.types", true),
    ] {
        let key = format!("hy_v4_flash.{suffix}");
        let expected = (0..42)
            .map(|i| layout::sparse(i) == sparse_flag)
            .collect::<Vec<_>>();
        if g.get_array(&key)
            .filter(|a| a.len == 42)
            .and_then(|a| g.array_bools(&a).ok())
            != Some(expected)
        {
            return Err(ValidateError::TokenKey("Hy4 layer types", key));
        }
    }
    Ok(())
}

// Native graph allocations in ds4_hy4_bind.inc plus the row reader's host
// staging. Each additional bank borrows this workspace and owns its KV.
pub(crate) fn serving_bytes(ctx: u64, rows: u32) -> (u64, u64) {
    let rows = u64::from(rows);
    let sparse_row = if std::env::var_os("DS4_HY4_KV").as_deref() == Some("bf16".as_ref()) {
        1152
    } else {
        641
    };
    let window = ctx.min(rows + 511);
    let tail = ctx.min(rows + 515);
    let sparse = ctx * sparse_row + ctx.div_ceil(4) * 256 + tail * 259 * 4;
    let kv = 12 * sparse + 30 * window * 1152 + 7 * (3072 + 5 * 511) * 4;
    let hidden = 3072;
    let ff = 16 * 1536;
    let float_cols = 4 * hidden
        + 8
        + 4
        + 3 * hidden
        + 1024
        + 2 * 64 * 256
        + 576
        + 2 * 64 * 512
        + 3 * 128
        + 16 * 128
        + 16
        + 512
        + 1024
        + 512
        + 512
        + 16
        + 16
        + 1536
        + 3 * ff
        + hidden;
    // BF16 embedding rows and the BF16 copy of the norm beside the F32 buffers.
    let device = 2 * 2 * hidden + 4 * 16 * 98 + 4 * 4 + float_cols * 4 + 64 * 512 * 2;
    let host = 2 * hidden + 4 * 16 * 98 + 3 * 4 + 64 * 32;
    (kv, rows * (device + host) + 120832 * 4 + 512 * 4096)
}
