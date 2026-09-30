//! Pinned Naive GQA/DSA artifact contract. Execution remains native.
use std::collections::BTreeSet;

use crate::gguf::GgufFile;
use crate::layout::{LayoutSpec, TypeClass};
use crate::tensors::TensorInventory;
use crate::validate::ValidateError;

const ARCH: &str = "naive_n05_flash";
const SOURCE_REV: &[u8] = b"0235b3b5ff27422b1f57cdc2acddfaf643e08356";
const RECIPE: &[u8] = b"MQ87-IQ2XXS-IQ2XS-Q6Q8-BF16";
pub(crate) const LAYERS: u32 = 48;
pub(crate) const CONTEXT_MAX: u32 = 1_048_576;
pub(crate) const SWA_WINDOW: u32 = 128;
pub(crate) const DSA_LAYERS: [u32; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];
pub(crate) const TENSORS: usize = 613;
pub(crate) const SHARDS: u32 = 4;
const EMBED: u64 = 4096;
const VOCAB: u64 = 152576;
const HEADS: u64 = 64;
const KEY: u64 = 192;
const VALUE: u64 = 128;
const DENSE_FF: u64 = 16384;
const EXPERT_FF: u64 = 2048;
const EXPERTS: u64 = 256;
const INDEX_HEADS: u64 = 16;
const INDEX_DIM: u64 = 128;
const INDEX_TOP_K: u64 = 2048;
const SCORE_QUERY_TILE: u64 = 32;
const SCORE_HISTORY_TILE: u64 = 4096;
const F32: u32 = 0;
const Q8: u32 = 8;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const BF16: u32 = 30;
pub(crate) const PREFILL_CAP: u32 = 2048;
pub(crate) const PREFILL_MAX: u32 = 8192;

pub(crate) struct MemoryPlan {
    dsa: u64,
    swa: u64,
    index: u64,
    scratch: u64,
}

impl MemoryPlan {
    pub(crate) fn cache_bytes(&self) -> u64 {
        self.dsa + self.swa + self.index
    }

    pub(crate) fn scratch_bytes(&self) -> u64 {
        self.scratch
    }
}

/// Keep this in step with ds4_naive_plan.h and the native graph allocator.
pub(crate) fn memory_plan(ctx: u32, cap: u32) -> Option<MemoryPlan> {
    if ctx == 0 || ctx > CONTEXT_MAX || cap == 0 || cap > ctx || cap > PREFILL_MAX {
        return None;
    }
    let ctx = u64::from(ctx);
    let cap = u64::from(cap);
    let swa_rows = ctx.min(u64::from(SWA_WINDOW) - 1 + cap);
    let dsa_layers = DSA_LAYERS.len() as u64;
    let dsa = dsa_layers * ctx * 4 * (KEY + VALUE) * 2;
    let swa = (u64::from(LAYERS) - dsa_layers) * swa_rows * 8 * (KEY + VALUE) * 2;
    let index = dsa_layers * ctx * (INDEX_DIM + 4);

    // Main activation/route buffers, split projections and indexer inputs.
    let row = 4 * EMBED
        + HEADS * KEY
        + 8 * (KEY + VALUE)
        + HEADS * VALUE
        + 3 * DENSE_FF
        + EXPERTS
        + 2 * 8
        + 3 * 8 * EXPERT_FF
        + 8 * EMBED
        + 2
        + 2 * 64
        + INDEX_HEADS * INDEX_DIM
        + INDEX_DIM
        + INDEX_HEADS
        + INDEX_TOP_K;
    let main = (cap * row + 64 + VOCAB) * 4;

    // Only 32 queries use score scratch at once, independent of prefill cap.
    // Leaf top-k lists shrink at every merge; two buffers cover all levels.
    let queries = cap.min(SCORE_QUERY_TILE);
    let tiles = ctx.div_ceil(SCORE_HISTORY_TILE);
    let scores = queries * ctx * 4;
    let candidates = 2 * queries * tiles * ctx.min(INDEX_TOP_K) * 8;
    Some(MemoryPlan {
        dsa,
        swa,
        index,
        scratch: main + scores + candidates,
    })
}

fn mismatch(key: impl Into<String>) -> ValidateError {
    ValidateError::TokenKey("naive", key.into())
}

pub(crate) fn is_dsa(layer: u32) -> bool {
    DSA_LAYERS.contains(&layer)
}

pub(crate) fn kv_heads(layer: u32) -> u32 {
    if is_dsa(layer) {
        4
    } else {
        8
    }
}

pub(crate) fn validate_metadata(g: &GgufFile) -> Result<(), ValidateError> {
    check_metadata(g, 0)
}

fn check_metadata(g: &GgufFile, shard: u16) -> Result<(), ValidateError> {
    for (key, expected) in [
        ("general.architecture", ARCH.as_bytes()),
        (
            "general.source.huggingface.repository",
            b"NaiveAI/Naive-N0.5-Flash",
        ),
        ("general.source.huggingface.revision", SOURCE_REV),
        ("general.quantization_recipe", RECIPE),
        ("tokenizer.ggml.model", b"gpt2"),
        ("tokenizer.ggml.pre", b"qwen2"),
    ] {
        if g.get_string(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    if g.split_count() != SHARDS || g.get_u16("split.no") != Some(shard) {
        return Err(mismatch("split.count / split.no"));
    }
    if g.get_token_id("split.tensors.count") != Some(TENSORS as i32) {
        return Err(mismatch("split.tensors.count"));
    }
    for (suffix, expected) in [
        ("block_count", LAYERS),
        ("context_length", CONTEXT_MAX),
        ("embedding_length", EMBED as u32),
        ("attention.head_count", HEADS as u32),
        ("attention.key_length", KEY as u32),
        ("attention.value_length", VALUE as u32),
        ("attention.sliding_window", SWA_WINDOW),
        ("rope.dimension_count", 64),
        ("rope.dimension_count_swa", 64),
        ("expert_count", EXPERTS as u32),
        ("expert_used_count", 8),
        ("expert_group_count", 1),
        ("expert_group_used_count", 1),
        ("expert_gating_func", 2),
        ("expert_feed_forward_length", EXPERT_FF as u32),
        ("index_top_k", 2048),
        ("index_head_dim", INDEX_DIM as u32),
        ("index_n_heads", INDEX_HEADS as u32),
        ("index_n_kv_heads", 1),
        ("attention_chunk_size", SWA_WINDOW),
    ] {
        let key = format!("{ARCH}.{suffix}");
        if g.get_u32(&key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for (suffix, expected) in [
        ("rope.freq_base", 10_000_000.0),
        ("rope.freq_base_swa", 10_000.0),
        ("rope.partial_rotary_factor", 0.334),
        ("attention.value_scale", 0.707),
        ("attention.layer_norm_rms_epsilon", 1e-5),
    ] {
        let key = format!("{ARCH}.{suffix}");
        if g.get_f32_compat(&key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for (suffix, expected) in [
        ("attention.dsa_enabled", true),
        ("attention.add_swa_sink", true),
        ("attention.add_full_sink", false),
        ("expert_weights_norm", true),
    ] {
        let key = format!("{ARCH}.{suffix}");
        if g.get_bool(&key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for (suffix, expected) in [
        ("attention.projection_layout", b"split".as_slice()),
        ("rope.pairing", b"split-half / GPT-NeoX"),
        ("indexer_activation_dtype", b"fp8_e4m3"),
        ("moe.topk_method", b"noaux_tc"),
        ("moe.scoring_func", b"sigmoid"),
    ] {
        let key = format!("{ARCH}.{suffix}");
        if g.get_string(&key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for (suffix, expected) in [
        ("attention.dsa_layers", DSA_LAYERS.to_vec()),
        (
            "attention.head_count_kv",
            (0..LAYERS).map(kv_heads).collect(),
        ),
        (
            "attention.hybrid_layer_pattern",
            (0..LAYERS).map(|i| u32::from(!is_dsa(i))).collect(),
        ),
        (
            "expert.layer_pattern",
            (0..LAYERS).map(|i| u32::from(i != 0)).collect(),
        ),
        (
            "feed_forward_length",
            (0..LAYERS)
                .map(|i| {
                    if i == 0 {
                        DENSE_FF as u32
                    } else {
                        EXPERT_FF as u32
                    }
                })
                .collect(),
        ),
    ] {
        let key = format!("{ARCH}.{suffix}");
        let array = g.get_array(&key).ok_or_else(|| mismatch(&key))?;
        if g.array_le_u32s(&array)? != expected {
            return Err(mismatch(key));
        }
    }

    // No embedded predictor. The external DSpark owns a separate contract.
    let mtp = format!("{ARCH}.nextn_predict_layers");
    if g.get_u32(&mtp).unwrap_or(0) != 0 {
        return Err(mismatch(mtp));
    }
    let tokens = g
        .get_array("tokenizer.ggml.tokens")
        .ok_or_else(|| mismatch("tokenizer.ggml.tokens"))?;
    if tokens.typ != crate::gguf::GGUF_VALUE_STRING || tokens.len != VOCAB {
        return Err(mismatch("tokenizer.ggml.tokens"));
    }
    for (key, expected) in [
        ("tokenizer.ggml.eos_token_id", 151645),
        ("tokenizer.ggml.padding_token_id", 151643),
    ] {
        if g.get_token_id(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        if g.get_bool(key) != Some(false) {
            return Err(mismatch(key));
        }
    }
    if g.get_token_id("tokenizer.ggml.bos_token_id").is_some() {
        return Err(mismatch("tokenizer.ggml.bos_token_id"));
    }
    Ok(())
}

pub(crate) fn validate_inventory(inv: &TensorInventory) -> Result<(), ValidateError> {
    if inv.shards.len() != SHARDS as usize || inv.tensors.len() != TENSORS {
        return Err(mismatch("four shards / 613 tensors required"));
    }
    for (index, shard) in inv.shards.iter().enumerate() {
        check_metadata(&GgufFile::open(&shard.path)?, index as u16)?;
    }
    let mut names = BTreeSet::new();
    for tensor in &inv.tensors {
        if !names.insert(&tensor.name) {
            return Err(mismatch(format!("duplicate tensor {}", tensor.name)));
        }
    }
    // A valid directory must not alias independently bound weights.
    for shard in 0..inv.shards.len() {
        let mut spans = Vec::new();
        for tensor in inv.tensors.iter().filter(|t| t.shard as usize == shard) {
            let end = tensor
                .abs_offset
                .checked_add(tensor.bytes)
                .ok_or_else(|| mismatch("tensor span overflow"))?;
            spans.push((tensor.abs_offset, end));
        }
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(mismatch("overlapping tensor payloads"));
        }
    }
    Ok(())
}

pub(crate) fn layouts() -> Vec<LayoutSpec> {
    let mut specs = Vec::with_capacity(TENSORS);
    let mut add = |name: String, typ, dims: &[u64]| {
        let mut dim = [0; 8];
        dim[..dims.len()].copy_from_slice(dims);
        specs.push(LayoutSpec {
            name,
            class: TypeClass::Exact(typ),
            ndim: dims.len() as u32,
            dim,
        });
    };
    for name in ["token_embd.weight", "output.weight"] {
        add(name.into(), Q8, &[EMBED, VOCAB]);
    }
    add("output_norm.weight".into(), F32, &[EMBED]);
    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            add(format!("{prefix}.{name}"), F32, &[EMBED]);
        }
        add(
            format!("{prefix}.attn_q.weight"),
            Q6_K,
            &[EMBED, HEADS * KEY],
        );
        add(
            format!("{prefix}.attn_k.weight"),
            Q8,
            &[EMBED, u64::from(kv_heads(layer)) * KEY],
        );
        add(
            format!("{prefix}.attn_v.weight"),
            Q8,
            &[EMBED, u64::from(kv_heads(layer)) * VALUE],
        );
        add(
            format!("{prefix}.attn_output.weight"),
            Q6_K,
            &[HEADS * VALUE, EMBED],
        );
        if is_dsa(layer) {
            // Direct hidden projections; this indexer has no MLA q-lora input.
            for (name, rows) in [
                ("q_proj.weight", INDEX_HEADS * INDEX_DIM),
                ("k_proj.weight", INDEX_DIM),
                ("proj.weight", INDEX_HEADS),
            ] {
                add(format!("{prefix}.indexer.{name}"), BF16, &[EMBED, rows]);
            }
            for name in ["weight", "bias"] {
                add(format!("{prefix}.indexer.k_norm.{name}"), F32, &[INDEX_DIM]);
            }
        } else {
            add(format!("{prefix}.attn_sinks.weight"), F32, &[HEADS]);
        }
        if layer == 0 {
            for name in ["ffn_gate.weight", "ffn_up.weight"] {
                add(format!("{prefix}.{name}"), Q8, &[EMBED, DENSE_FF]);
            }
            add(format!("{prefix}.ffn_down.weight"), Q8, &[DENSE_FF, EMBED]);
            continue;
        }
        add(
            format!("{prefix}.ffn_gate_inp.weight"),
            F32,
            &[EMBED, EXPERTS],
        );
        add(format!("{prefix}.exp_probs_b.bias"), F32, &[EXPERTS]);
        for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            add(
                format!("{prefix}.{name}"),
                IQ2_XXS,
                &[EMBED, EXPERT_FF, EXPERTS],
            );
        }
        add(
            format!("{prefix}.ffn_down_exps.weight"),
            IQ2_XS,
            &[EXPERT_FF, EMBED, EXPERTS],
        );
    }
    specs
}
