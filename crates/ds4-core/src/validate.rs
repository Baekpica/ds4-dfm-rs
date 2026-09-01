//! Host-owned `config_validate_model`.
//!
//! Copied from `ds4.c`. Identify selects the pinned shape; this checks
//! every required key against that shape (plus DeepSeek compress / SwiGLU
//! arrays, Solar GQA schedule, EXAONE LLLG pattern, Motif/dots3 SHAs).
//! When `Model::open` succeeds, native applies the host shape and skips
//! C `config_validate_model`.

use crate::bind::{
    dots3_layer_is_full_attention, expected_compress_ratio, qwen4exp_layer_is_full_attention,
    solar_layer_is_gqa,
};
use crate::gguf::{
    GgufError, GgufFile, GGUF_VALUE_BOOL, GGUF_VALUE_FLOAT32, GGUF_VALUE_FLOAT64, GGUF_VALUE_INT32,
    GGUF_VALUE_STRING, GGUF_VALUE_UINT32,
};
use crate::identify::{identify_file, IdentifyError};
use crate::shape::{
    route_architecture, ArchRoute, ModelFamily, Shape, Variant, SHAPE_DOTS3_NOTE_PREV, SHAPE_FLASH,
    SHAPE_KEXAONE_236B, SHAPE_MOTIF3, SHAPE_PRO, SHAPE_QWEN38_FLASH_NEXT, SHAPE_SOLAR_OPEN2_250B,
};
use crate::tensors::TensorInventory;

const MOTIF_SHA: &[u8] = b"30f14b635d3258a18c3ff7e69829f8fbfa775e87477ffabb59a79115bba820a5";
const DOTS3_SHA: &[u8] = b"99b7de680dd456111c36efb8749f8ae7177328e97b65a3e39a6700cbc1173833";
const QWEN_REVISIONS: [&[u8]; 2] = [
    b"f5d08274bafd880402bd16f5e3e6c514136ec06c",
    b"8336e613ea508b13c2159bd0f68965d97a606b95",
];
const QWEN_CONFIG_SHA: &[u8] = b"889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b";
const QWEN_LICENSE_SHA: &[u8] = b"a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08";

#[derive(Debug)]
pub enum ValidateError {
    Identify(IdentifyError),
    Gguf(GgufError),
    Token(&'static str),
    TokenKey(&'static str, String),
    TokenLayer(&'static str, u32),
}

impl ValidateError {
    pub fn token(&self) -> String {
        match self {
            ValidateError::Identify(e) => e.token(),
            ValidateError::Gguf(e) => e.token(),
            ValidateError::Token(t) => (*t).into(),
            ValidateError::TokenKey(kind, k) => format!("{kind} {k}"),
            ValidateError::TokenLayer(kind, n) => format!("{kind} {n}"),
        }
    }
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for ValidateError {}

impl From<IdentifyError> for ValidateError {
    fn from(e: IdentifyError) -> Self {
        ValidateError::Identify(e)
    }
}

impl From<GgufError> for ValidateError {
    fn from(e: GgufError) -> Self {
        ValidateError::Gguf(e)
    }
}

fn missing(key: &'static str) -> ValidateError {
    ValidateError::TokenKey("missing-key", key.into())
}

fn mismatch(name: &'static str) -> ValidateError {
    ValidateError::TokenKey("mismatch", name.into())
}

fn mismatch_f32(name: &'static str) -> ValidateError {
    ValidateError::TokenKey("mismatch-f32", name.into())
}

fn mismatch_bool(name: &'static str) -> ValidateError {
    ValidateError::TokenKey("mismatch-bool", name.into())
}

fn mismatch_u64(name: &'static str) -> ValidateError {
    ValidateError::TokenKey("mismatch-u64", name.into())
}

fn f32_eq(got: f32, expected: f32) -> bool {
    let scale = if expected.abs() > 1.0 {
        expected.abs()
    } else {
        1.0
    };
    (got - expected).abs() <= scale * 1.0e-6
}

fn req_u32(g: &GgufFile, key: &'static str) -> Result<u32, ValidateError> {
    g.get_u32(key).ok_or_else(|| missing(key))
}

fn req_u64c(g: &GgufFile, key: &'static str) -> Result<u64, ValidateError> {
    g.get_u64_compat(key).ok_or_else(|| missing(key))
}

fn req_f32(g: &GgufFile, key: &'static str) -> Result<f32, ValidateError> {
    g.get_f32_compat(key).ok_or_else(|| missing(key))
}

fn req_bool(g: &GgufFile, key: &'static str) -> Result<bool, ValidateError> {
    g.get_bool(key).ok_or_else(|| missing(key))
}

fn expect_u32(name: &'static str, got: u32, want: u32) -> Result<(), ValidateError> {
    if got == want {
        Ok(())
    } else {
        Err(mismatch(name))
    }
}

fn expect_u64(name: &'static str, got: u64, want: u64) -> Result<(), ValidateError> {
    if got == want {
        Ok(())
    } else {
        Err(mismatch_u64(name))
    }
}

fn expect_f32(name: &'static str, got: f32, want: f32) -> Result<(), ValidateError> {
    if f32_eq(got, want) {
        Ok(())
    } else {
        Err(mismatch_f32(name))
    }
}

fn expect_bool(name: &'static str, got: bool, want: bool) -> Result<(), ValidateError> {
    if got == want {
        Ok(())
    } else {
        Err(mismatch_bool(name))
    }
}

fn expect_string(g: &GgufFile, key: &'static str, want: &[u8]) -> Result<(), ValidateError> {
    if g.get_string(key) == Some(want) {
        Ok(())
    } else {
        Err(ValidateError::TokenKey("mismatch-string", key.into()))
    }
}

fn has_key(g: &GgufFile, key: &str) -> bool {
    g.kv_entries()
        .iter()
        .any(|e| g.key_bytes(e) == key.as_bytes())
}

fn array_nonnegative_u32s(
    g: &GgufFile,
    key: &'static str,
    len: u64,
) -> Result<Vec<u32>, ValidateError> {
    let arr = g
        .get_array(key)
        .ok_or(ValidateError::TokenKey("missing-array", key.into()))?;
    if arr.len != len || (arr.typ != GGUF_VALUE_UINT32 && arr.typ != GGUF_VALUE_INT32) {
        return Err(ValidateError::TokenKey("array-shape", key.into()));
    }
    if arr.typ == GGUF_VALUE_UINT32 {
        return Ok(g.array_le_u32s(&arr)?);
    }
    let mut out = Vec::with_capacity(len as usize);
    let mut pos = arr.data_pos;
    for _ in 0..len {
        let bytes = g
            .as_bytes()
            .get(pos..pos + 4)
            .ok_or(ValidateError::Gguf(GgufError::Truncated))?;
        let value = i32::from_le_bytes(bytes.try_into().unwrap());
        if value < 0 {
            return Err(ValidateError::TokenKey("negative-array", key.into()));
        }
        out.push(value as u32);
        pos += 4;
    }
    Ok(out)
}

fn motif3_layer_is_full_attention(shape: &Shape, il: u32) -> bool {
    shape.family == ModelFamily::Motif3
        && il < shape.n_layer
        && shape.n_swa_period != 0
        && (il % shape.n_swa_period) == 0
}

fn validate_compress(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let key = "deepseek4.attention.compress_ratios";
    let arr = g
        .get_array(key)
        .ok_or(ValidateError::TokenKey("missing-array", key.into()))?;
    if arr.typ != GGUF_VALUE_UINT32 && arr.typ != GGUF_VALUE_INT32 {
        return Err(ValidateError::TokenKey("array-type", key.into()));
    }
    if arr.len < u64::from(shape.n_layer) {
        return Err(ValidateError::TokenKey("array-short", key.into()));
    }
    if arr.typ == GGUF_VALUE_INT32 {
        let mut c_pos = arr.data_pos;
        let data = g.as_bytes();
        for il in 0..shape.n_layer {
            let b = data
                .get(c_pos..c_pos + 4)
                .ok_or(ValidateError::Gguf(GgufError::Truncated))?;
            let v = i32::from_le_bytes(b.try_into().unwrap());
            if v < 0 {
                return Err(ValidateError::Token("negative-array"));
            }
            let got = v as u32;
            let want = expected_compress_ratio(shape.variant, shape.n_layer, il);
            if got != want {
                return Err(ValidateError::TokenLayer("compress-ratio", il));
            }
            c_pos += 4;
        }
        return Ok(());
    }
    let vals = g.array_le_u32s(&arr)?;
    for il in 0..shape.n_layer {
        let got = vals[il as usize];
        let want = expected_compress_ratio(shape.variant, shape.n_layer, il);
        if got != want {
            return Err(ValidateError::TokenLayer("compress-ratio", il));
        }
    }
    Ok(())
}

fn validate_swiglu(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let key = "deepseek4.swiglu_clamp_exp";
    let arr = g
        .get_array(key)
        .ok_or(ValidateError::TokenKey("missing-array", key.into()))?;
    if arr.typ != GGUF_VALUE_FLOAT32 && arr.typ != GGUF_VALUE_FLOAT64 {
        return Err(ValidateError::TokenKey("array-type", key.into()));
    }
    if arr.len < u64::from(shape.n_layer) {
        return Err(ValidateError::TokenKey("array-short", key.into()));
    }
    let vals = g.array_f32s(&arr)?;
    for il in 0..shape.n_layer {
        expect_f32(
            "swiglu_clamp_exp",
            vals[il as usize],
            shape.swiglu_clamp_exp,
        )?;
    }
    Ok(())
}

fn validate_deepseek(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let n_layer = req_u32(g, "deepseek4.block_count")?;
    let n_embd = req_u32(g, "deepseek4.embedding_length")?;
    let n_vocab = req_u32(g, "deepseek4.vocab_size")?;
    let n_head = req_u32(g, "deepseek4.attention.head_count")?;
    let n_head_kv = req_u32(g, "deepseek4.attention.head_count_kv")?;
    let n_head_dim = req_u32(g, "deepseek4.attention.key_length")?;
    let n_value_dim = req_u32(g, "deepseek4.attention.value_length")?;
    let n_rot = req_u32(g, "deepseek4.rope.dimension_count")?;
    let n_lora_q = req_u32(g, "deepseek4.attention.q_lora_rank")?;
    let n_lora_o = req_u32(g, "deepseek4.attention.output_lora_rank")?;
    let n_out_group = req_u32(g, "deepseek4.attention.output_group_count")?;
    let n_expert = req_u32(g, "deepseek4.expert_count")?;
    let n_expert_used = req_u32(g, "deepseek4.expert_used_count")?;
    let n_ff_exp = req_u32(g, "deepseek4.expert_feed_forward_length")?;
    let n_expert_shared = req_u32(g, "deepseek4.expert_shared_count")?;
    let n_hash_layer = req_u32(g, "deepseek4.hash_layer_count")?;
    let n_expert_groups = g.get_u32("deepseek4.expert_group_count").unwrap_or(0);
    let n_group_used = g.get_u32("deepseek4.expert_group_used_count").unwrap_or(0);
    let n_swa = req_u32(g, "deepseek4.attention.sliding_window")?;
    let n_indexer_head = req_u32(g, "deepseek4.attention.indexer.head_count")?;
    let n_indexer_head_dim = req_u32(g, "deepseek4.attention.indexer.key_length")?;
    let n_indexer_top_k = req_u32(g, "deepseek4.attention.indexer.top_k")?;
    let n_hc = req_u32(g, "deepseek4.hyper_connection.count")?;
    let n_hc_sinkhorn = req_u32(g, "deepseek4.hyper_connection.sinkhorn_iterations")?;

    expect_u32("embedding_length", n_embd, shape.n_embd)?;
    expect_u32("vocab_size", n_vocab, shape.n_vocab)?;
    expect_u32("attention.head_count", n_head, shape.n_head)?;
    expect_u32("attention.key_length", n_head_dim, shape.n_head_dim)?;
    expect_u32("attention.head_count_kv", n_head_kv, shape.n_head_kv)?;
    expect_u32("attention.value_length", n_value_dim, shape.n_value_dim)?;
    expect_u32("rope.dimension_count", n_rot, shape.n_rot)?;
    expect_u32(
        "attention.output_group_count",
        n_out_group,
        shape.n_out_group,
    )?;
    expect_u32("attention.q_lora_rank", n_lora_q, shape.n_lora_q)?;
    expect_u32("attention.output_lora_rank", n_lora_o, shape.n_lora_o)?;
    expect_u32("expert_count", n_expert, shape.n_expert)?;
    expect_u32("expert_used_count", n_expert_used, shape.n_expert_used)?;
    expect_u32("expert_feed_forward_length", n_ff_exp, shape.n_ff_exp)?;
    expect_u32(
        "expert_shared_count",
        n_expert_shared,
        shape.n_expert_shared,
    )?;
    expect_u32("hash_layer_count", n_hash_layer, shape.n_hash_layer)?;
    expect_u32("expert_group_count", n_expert_groups, 0)?;
    expect_u32("expert_group_used_count", n_group_used, 0)?;
    expect_u32("attention.sliding_window", n_swa, shape.n_swa)?;
    expect_u32(
        "attention.indexer.head_count",
        n_indexer_head,
        shape.n_indexer_head,
    )?;
    expect_u32(
        "attention.indexer.key_length",
        n_indexer_head_dim,
        shape.n_indexer_head_dim,
    )?;
    expect_u32(
        "attention.indexer.top_k",
        n_indexer_top_k,
        shape.n_indexer_top_k,
    )?;
    expect_u32("hyper_connection.count", n_hc, shape.n_hc)?;
    expect_u32(
        "hyper_connection.sinkhorn_iterations",
        n_hc_sinkhorn,
        shape.n_hc_sinkhorn_iter,
    )?;
    expect_u32("block_count", n_layer, shape.n_layer)?;
    validate_compress(g, shape)?;
    validate_swiglu(g, shape)?;

    let mut rope_orig = shape.rope_orig_ctx;
    if let Some(v) = g.get_u64_compat("deepseek4.rope.scaling.original_context_length") {
        rope_orig = v;
    }
    expect_u64(
        "rope.scaling.original_context_length",
        rope_orig,
        shape.rope_orig_ctx,
    )?;
    expect_f32(
        "rope.freq_base",
        req_f32(g, "deepseek4.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    let mut rope_scale = shape.rope_scale_factor;
    if let Some(v) = g.get_f32_compat("deepseek4.rope.scaling.factor") {
        rope_scale = v;
    }
    expect_f32("rope.scaling.factor", rope_scale, shape.rope_scale_factor)?;
    let mut yarn_fast = shape.rope_yarn_beta_fast;
    if let Some(v) = g.get_f32_compat("deepseek4.rope.scaling.yarn_beta_fast") {
        yarn_fast = v;
    }
    expect_f32(
        "rope.scaling.yarn_beta_fast",
        yarn_fast,
        shape.rope_yarn_beta_fast,
    )?;
    let mut yarn_slow = shape.rope_yarn_beta_slow;
    if let Some(v) = g.get_f32_compat("deepseek4.rope.scaling.yarn_beta_slow") {
        yarn_slow = v;
    }
    expect_f32(
        "rope.scaling.yarn_beta_slow",
        yarn_slow,
        shape.rope_yarn_beta_slow,
    )?;
    expect_f32(
        "attention.compress_rope_freq_base",
        req_f32(g, "deepseek4.attention.compress_rope_freq_base")?,
        shape.compress_rope_freq_base,
    )?;
    expect_f32(
        "expert_weights_scale",
        req_f32(g, "deepseek4.expert_weights_scale")?,
        shape.expert_weight_scale,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "deepseek4.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    expect_f32(
        "hyper_connection.epsilon",
        req_f32(g, "deepseek4.hyper_connection.epsilon")?,
        shape.hc_eps,
    )?;
    expect_bool(
        "expert_weights_norm",
        req_bool(g, "deepseek4.expert_weights_norm")?,
        true,
    )?;
    Ok(())
}

fn validate_motif3(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    expect_u32(
        "block_count",
        req_u32(g, "motif3.block_count")?,
        shape.n_layer,
    )?;
    expect_u64(
        "context_length",
        req_u64c(g, "motif3.context_length")?,
        262144,
    )?;
    expect_u32(
        "embedding_length",
        req_u32(g, "motif3.embedding_length")?,
        shape.n_embd,
    )?;
    expect_u32(
        "vocab_size",
        req_u32(g, "motif3.vocab_size")?,
        shape.n_vocab,
    )?;
    expect_u32(
        "feed_forward_length",
        req_u32(g, "motif3.feed_forward_length")?,
        shape.n_ff_dense,
    )?;
    expect_u32(
        "leading_dense_block_count",
        req_u32(g, "motif3.leading_dense_block_count")?,
        shape.n_leading_dense,
    )?;
    expect_u32(
        "expert_count",
        req_u32(g, "motif3.expert_count")?,
        shape.n_expert,
    )?;
    expect_u32(
        "expert_used_count",
        req_u32(g, "motif3.expert_used_count")?,
        shape.n_expert_used,
    )?;
    expect_u32(
        "expert_feed_forward_length",
        req_u32(g, "motif3.expert_feed_forward_length")?,
        shape.n_ff_exp,
    )?;
    expect_u32(
        "expert_shared_count",
        req_u32(g, "motif3.expert_shared_count")?,
        shape.n_expert_shared,
    )?;
    expect_u32(
        "expert_gating_func",
        req_u32(g, "motif3.expert_gating_func")?,
        1,
    )?;
    expect_u32(
        "attention.head_count",
        req_u32(g, "motif3.attention.head_count")?,
        shape.n_head,
    )?;
    expect_u32(
        "attention.head_count_kv",
        req_u32(g, "motif3.attention.head_count_kv")?,
        shape.n_head_kv,
    )?;
    expect_u32(
        "attention.noise_head_count",
        req_u32(g, "motif3.attention.noise_head_count")?,
        shape.n_noise_head,
    )?;
    expect_u32(
        "attention.key_length",
        req_u32(g, "motif3.attention.key_length")?,
        shape.n_head_dim,
    )?;
    expect_u32(
        "attention.value_length",
        req_u32(g, "motif3.attention.value_length")?,
        shape.n_value_dim,
    )?;
    expect_u32(
        "attention.q_lora_rank",
        req_u32(g, "motif3.attention.q_lora_rank")?,
        shape.n_lora_q,
    )?;
    expect_u32(
        "attention.kv_lora_rank",
        req_u32(g, "motif3.attention.kv_lora_rank")?,
        shape.n_kv_lora,
    )?;
    expect_u32(
        "attention.rope_dimension_count",
        req_u32(g, "motif3.attention.rope_dimension_count")?,
        shape.n_rot,
    )?;
    expect_u32(
        "attention.sliding_window",
        req_u32(g, "motif3.attention.sliding_window")?,
        shape.n_swa,
    )?;
    expect_u32(
        "attention.sliding_window_period",
        req_u32(g, "motif3.attention.sliding_window_period")?,
        shape.n_swa_period,
    )?;
    expect_u32(
        "mhc.expansion_rate",
        req_u32(g, "motif3.mhc.expansion_rate")?,
        shape.n_hc,
    )?;
    expect_u32(
        "mhc.sinkhorn_iterations",
        req_u32(g, "motif3.mhc.sinkhorn_iterations")?,
        shape.n_hc_sinkhorn_iter,
    )?;
    expect_u32(
        "mtp.block_count",
        req_u32(g, "motif3.mtp.block_count")?,
        shape.n_nextn_predict,
    )?;
    expect_bool(
        "expert_weights_norm",
        req_bool(g, "motif3.expert_weights_norm")?,
        true,
    )?;
    expect_bool(
        "attention.elementwise_output_gate",
        req_bool(g, "motif3.attention.elementwise_output_gate")?,
        true,
    )?;
    expect_bool("mhc.enabled", req_bool(g, "motif3.mhc.enabled")?, true)?;
    expect_bool(
        "polynorm.sigmoid_weight",
        req_bool(g, "motif3.polynorm.sigmoid_weight")?,
        true,
    )?;
    expect_bool(
        "rope.scaling.apply_mscale",
        req_bool(g, "motif3.rope.scaling.apply_mscale")?,
        false,
    )?;
    expect_f32(
        "expert_weights_scale",
        req_f32(g, "motif3.expert_weights_scale")?,
        shape.expert_weight_scale,
    )?;
    expect_f32(
        "expert_score_correction",
        req_f32(g, "motif3.expert_score_correction")?,
        1.0e-4,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "motif3.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    expect_f32(
        "rope.freq_base",
        req_f32(g, "motif3.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    expect_f32(
        "rope.freq_base_swa",
        req_f32(g, "motif3.rope.freq_base_swa")?,
        10000.0,
    )?;
    expect_f32(
        "rope.scaling.factor",
        req_f32(g, "motif3.rope.scaling.factor")?,
        shape.rope_scale_factor,
    )?;
    expect_f32(
        "rope.scaling.beta_fast",
        req_f32(g, "motif3.rope.scaling.beta_fast")?,
        shape.rope_yarn_beta_fast,
    )?;
    expect_f32(
        "rope.scaling.beta_slow",
        req_f32(g, "motif3.rope.scaling.beta_slow")?,
        shape.rope_yarn_beta_slow,
    )?;
    expect_f32(
        "rope.scaling.mscale",
        req_f32(g, "motif3.rope.scaling.mscale")?,
        1.0,
    )?;
    expect_f32(
        "mhc.h_post_coefficient",
        req_f32(g, "motif3.mhc.h_post_coefficient")?,
        1.0,
    )?;
    expect_f32(
        "polynorm.output_scale",
        req_f32(g, "motif3.polynorm.output_scale")?,
        0.5,
    )?;
    expect_f32(
        "polynorm.bias_clamp",
        req_f32(g, "motif3.polynorm.bias_clamp")?,
        0.5,
    )?;
    expect_f32(
        "hidden_clamp",
        req_f32(g, "motif3.hidden_clamp")?,
        1_000_000.0,
    )?;

    let pattern = g
        .get_string("motif3.attention.sliding_window_pattern")
        .ok_or(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.attention.sliding_window_pattern".into(),
        ))?;
    if pattern != b"interleave" {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.attention.sliding_window_pattern".into(),
        ));
    }
    let rope_type = g
        .get_string("motif3.rope.scaling.type")
        .ok_or(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.rope.scaling.type".into(),
        ))?;
    if rope_type != b"yarn" {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.rope.scaling.type".into(),
        ));
    }
    let activation = g
        .get_string("motif3.activation")
        .ok_or(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.activation".into(),
        ))?;
    if activation != b"poly_norm" {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.activation".into(),
        ));
    }
    let sha = g
        .get_string("motif3.source.config_sha256")
        .ok_or(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.source.config_sha256".into(),
        ))?;
    if sha != MOTIF_SHA {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "motif3.source.config_sha256".into(),
        ));
    }
    let mut full = 0u32;
    for il in 0..shape.n_layer {
        if motif3_layer_is_full_attention(shape, il) {
            full += 1;
        }
    }
    expect_u32("full_attention_layer_count", full, 14)
}

fn validate_dots3(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    expect_u32(
        "block_count",
        req_u32(g, "dots3-note.block_count")?,
        shape.n_layer,
    )?;
    expect_u64(
        "context_length",
        req_u64c(g, "dots3-note.context_length")?,
        524288,
    )?;
    expect_u32(
        "embedding_length",
        req_u32(g, "dots3-note.embedding_length")?,
        shape.n_embd,
    )?;
    expect_u32(
        "vocab_size",
        req_u32(g, "dots3-note.vocab_size")?,
        shape.n_vocab,
    )?;
    expect_u32(
        "feed_forward_length",
        req_u32(g, "dots3-note.feed_forward_length")?,
        shape.n_ff_dense,
    )?;
    expect_u32(
        "leading_dense_block_count",
        req_u32(g, "dots3-note.leading_dense_block_count")?,
        shape.n_leading_dense,
    )?;
    expect_u32(
        "expert_count",
        req_u32(g, "dots3-note.expert_count")?,
        shape.n_expert,
    )?;
    expect_u32(
        "expert_used_count",
        req_u32(g, "dots3-note.expert_used_count")?,
        shape.n_expert_used,
    )?;
    expect_u32(
        "expert_feed_forward_length",
        req_u32(g, "dots3-note.expert_feed_forward_length")?,
        shape.n_ff_exp,
    )?;
    expect_u32(
        "expert_shared_count",
        req_u32(g, "dots3-note.expert_shared_count")?,
        shape.n_expert_shared,
    )?;
    expect_u32(
        "attention.head_count",
        req_u32(g, "dots3-note.attention.head_count")?,
        shape.n_head,
    )?;
    expect_u32(
        "attention.head_count_kv",
        req_u32(g, "dots3-note.attention.head_count_kv")?,
        shape.n_head_kv,
    )?;
    expect_u32(
        "attention.key_length",
        req_u32(g, "dots3-note.attention.key_length")?,
        shape.n_key_mla,
    )?;
    expect_u32(
        "attention.value_length",
        req_u32(g, "dots3-note.attention.value_length")?,
        shape.n_value_mla,
    )?;
    expect_u32(
        "sliding_window",
        req_u32(g, "dots3-note.sliding_window")?,
        shape.n_swa,
    )?;
    expect_u32(
        "index_topk",
        req_u32(g, "dots3-note.index_topk")?,
        shape.n_indexer_top_k,
    )?;
    expect_u32(
        "q_lora_rank",
        req_u32(g, "dots3-note.q_lora_rank")?,
        shape.n_lora_q,
    )?;
    expect_u32(
        "kv_lora_rank",
        req_u32(g, "dots3-note.kv_lora_rank")?,
        shape.n_kv_lora,
    )?;
    expect_u32(
        "swa_kv_lora_rank",
        req_u32(g, "dots3-note.swa_kv_lora_rank")?,
        shape.n_swa_kv_lora,
    )?;
    expect_u32(
        "full_attention_count",
        req_u32(g, "dots3-note.full_attention_count")?,
        shape.n_full_attn_count,
    )?;
    expect_bool(
        "language_only",
        req_bool(g, "dots3-note.language_only")?,
        true,
    )?;
    expect_bool("mtp.present", req_bool(g, "dots3-note.mtp.present")?, true)?;
    expect_f32(
        "rope.freq_base",
        req_f32(g, "dots3-note.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    expect_f32(
        "rope.freq_base_swa",
        req_f32(g, "dots3-note.rope.freq_base_swa")?,
        shape.rope_freq_base_swa,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "dots3-note.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    let sha = g
        .get_string("dots3-note.source.config_sha256")
        .ok_or(ValidateError::TokenKey(
            "mismatch-string",
            "dots3-note.source.config_sha256".into(),
        ))?;
    if sha != DOTS3_SHA {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "dots3-note.source.config_sha256".into(),
        ));
    }
    let mut full = 0u32;
    for il in 0..shape.n_layer {
        if dots3_layer_is_full_attention(shape, il) {
            full += 1;
        }
    }
    expect_u32("full_attention_layer_count", full, shape.n_full_attn_count)
}

fn validate_solar(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let n_layer = req_u32(g, "solar-open2.block_count")?;
    expect_u32("block_count", n_layer, shape.n_layer)?;
    expect_u64(
        "context_length",
        req_u64c(g, "solar-open2.context_length")?,
        shape.rope_orig_ctx,
    )?;
    expect_u32(
        "embedding_length",
        req_u32(g, "solar-open2.embedding_length")?,
        shape.n_embd,
    )?;
    expect_u32(
        "vocab_size",
        req_u32(g, "solar-open2.vocab_size")?,
        shape.n_vocab,
    )?;
    expect_u32(
        "feed_forward_length",
        req_u32(g, "solar-open2.feed_forward_length")?,
        shape.n_ff_dense,
    )?;
    expect_u32(
        "attention.head_count",
        req_u32(g, "solar-open2.attention.head_count")?,
        shape.n_head,
    )?;
    expect_u32(
        "attention.key_length",
        req_u32(g, "solar-open2.attention.key_length")?,
        shape.n_head_dim,
    )?;
    expect_u32(
        "attention.value_length",
        req_u32(g, "solar-open2.attention.value_length")?,
        shape.n_value_dim,
    )?;
    expect_u32(
        "expert_count",
        req_u32(g, "solar-open2.expert_count")?,
        shape.n_expert,
    )?;
    expect_u32(
        "expert_used_count",
        req_u32(g, "solar-open2.expert_used_count")?,
        shape.n_expert_used,
    )?;
    expect_u32(
        "expert_feed_forward_length",
        req_u32(g, "solar-open2.expert_feed_forward_length")?,
        shape.n_ff_exp,
    )?;
    expect_u32(
        "expert_shared_count",
        req_u32(g, "solar-open2.expert_shared_count")?,
        shape.n_expert_shared,
    )?;
    expect_u32(
        "leading_dense_block_count",
        req_u32(g, "solar-open2.leading_dense_block_count")?,
        0,
    )?;
    expect_u32(
        "ssm.conv_kernel",
        req_u32(g, "solar-open2.ssm.conv_kernel")?,
        shape.n_ssm_conv,
    )?;
    expect_u32(
        "kda.head_dim",
        req_u32(g, "solar-open2.kda.head_dim")?,
        shape.n_kda_head_dim,
    )?;
    expect_u32(
        "expert_gating_func",
        req_u32(g, "solar-open2.expert_gating_func")?,
        2,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "solar-open2.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    expect_f32(
        "expert_weights_scale",
        req_f32(g, "solar-open2.expert_weights_scale")?,
        shape.expert_weight_scale,
    )?;
    expect_bool(
        "expert_weights_norm",
        req_bool(g, "solar-open2.expert_weights_norm")?,
        true,
    )?;
    expect_f32(
        "rope.freq_base (vestigial)",
        req_f32(g, "solar-open2.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    if let Some(rope_dim) = g.get_u32("solar-open2.rope.dimension_count") {
        expect_u32("rope.dimension_count (NoPE)", rope_dim, 0)?;
    }
    expect_bool("internal use_rope", shape.use_rope, false)?;

    let key = "solar-open2.attention.head_count_kv";
    let arr = g
        .get_array(key)
        .ok_or(ValidateError::TokenKey("missing-array", key.into()))?;
    if arr.typ != GGUF_VALUE_INT32 && arr.typ != GGUF_VALUE_UINT32 {
        return Err(ValidateError::TokenKey("array-type", key.into()));
    }
    if arr.len != u64::from(n_layer) {
        return Err(ValidateError::TokenKey("array-short", key.into()));
    }
    let data = g.as_bytes();
    let mut pos = arr.data_pos;
    for il in 0..n_layer {
        let b = data
            .get(pos..pos + 4)
            .ok_or(ValidateError::Gguf(GgufError::Truncated))?;
        let got = if arr.typ == GGUF_VALUE_UINT32 {
            u32::from_le_bytes(b.try_into().unwrap())
        } else {
            let v = i32::from_le_bytes(b.try_into().unwrap());
            if v < 0 {
                return Err(ValidateError::Token("negative-array"));
            }
            v as u32
        };
        let want = if solar_layer_is_gqa(shape.family, shape.n_layer, il) {
            shape.n_head_kv
        } else {
            0
        };
        if got != want {
            return Err(ValidateError::TokenLayer("schedule", il));
        }
        pos += 4;
    }
    expect_f32("internal KDA q/k l2 epsilon", shape.kda_l2_eps, 1.0e-6)?;
    expect_f32(
        "internal KDA gate clamp minimum",
        shape.kda_gate_clamp_min,
        -5.0,
    )?;
    Ok(())
}

fn validate_exaone(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let n_layer = req_u32(g, "exaone-moe.block_count")?;
    expect_u32("block_count", n_layer, shape.n_layer)?;
    expect_u64(
        "context_length",
        req_u64c(g, "exaone-moe.context_length")?,
        shape.rope_orig_ctx,
    )?;
    expect_u32(
        "embedding_length",
        req_u32(g, "exaone-moe.embedding_length")?,
        shape.n_embd,
    )?;
    expect_u32(
        "vocab_size",
        req_u32(g, "exaone-moe.vocab_size")?,
        shape.n_vocab,
    )?;
    expect_u32(
        "feed_forward_length",
        req_u32(g, "exaone-moe.feed_forward_length")?,
        shape.n_ff_dense,
    )?;
    expect_u32(
        "attention.head_count",
        req_u32(g, "exaone-moe.attention.head_count")?,
        shape.n_head,
    )?;
    expect_u32(
        "attention.head_count_kv",
        req_u32(g, "exaone-moe.attention.head_count_kv")?,
        shape.n_head_kv,
    )?;
    expect_u32(
        "attention.key_length",
        req_u32(g, "exaone-moe.attention.key_length")?,
        shape.n_head_dim,
    )?;
    expect_u32(
        "attention.value_length",
        req_u32(g, "exaone-moe.attention.value_length")?,
        shape.n_value_dim,
    )?;
    expect_u32(
        "expert_count",
        req_u32(g, "exaone-moe.expert_count")?,
        shape.n_expert,
    )?;
    expect_u32(
        "expert_used_count",
        req_u32(g, "exaone-moe.expert_used_count")?,
        shape.n_expert_used,
    )?;
    expect_u32(
        "expert_feed_forward_length",
        req_u32(g, "exaone-moe.expert_feed_forward_length")?,
        shape.n_ff_exp,
    )?;
    expect_u32(
        "expert_shared_feed_forward_length",
        req_u32(g, "exaone-moe.expert_shared_feed_forward_length")?,
        shape.n_ff_shexp,
    )?;
    expect_u32(
        "expert_shared_count",
        req_u32(g, "exaone-moe.expert_shared_count")?,
        shape.n_expert_shared,
    )?;
    expect_u32(
        "expert_group_count",
        req_u32(g, "exaone-moe.expert_group_count")?,
        1,
    )?;
    expect_u32(
        "expert_group_used_count",
        req_u32(g, "exaone-moe.expert_group_used_count")?,
        1,
    )?;
    expect_u32(
        "expert_gating_func",
        req_u32(g, "exaone-moe.expert_gating_func")?,
        2,
    )?;
    expect_u32(
        "leading_dense_block_count",
        req_u32(g, "exaone-moe.leading_dense_block_count")?,
        shape.n_leading_dense,
    )?;
    expect_u32(
        "nextn_predict_layers",
        req_u32(g, "exaone-moe.nextn_predict_layers")?,
        shape.n_nextn_predict,
    )?;
    expect_u32(
        "attention.sliding_window",
        req_u32(g, "exaone-moe.attention.sliding_window")?,
        shape.n_swa,
    )?;
    expect_f32(
        "rope.freq_base",
        req_f32(g, "exaone-moe.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "exaone-moe.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    expect_f32(
        "expert_weights_scale",
        req_f32(g, "exaone-moe.expert_weights_scale")?,
        shape.expert_weight_scale,
    )?;
    expect_bool(
        "expert_weights_norm",
        req_bool(g, "exaone-moe.expert_weights_norm")?,
        true,
    )?;

    let key = "exaone-moe.attention.sliding_window_pattern";
    let arr = g
        .get_array(key)
        .ok_or(ValidateError::TokenKey("missing-array", key.into()))?;
    if arr.typ != GGUF_VALUE_BOOL {
        return Err(ValidateError::TokenKey("array-type", key.into()));
    }
    if arr.len != u64::from(n_layer) {
        return Err(ValidateError::TokenKey("array-short", key.into()));
    }
    let vals = g.array_bools(&arr)?;
    let period = shape.n_swa_period;
    for il in 0..n_layer {
        let want = (il % period) != period - 1;
        if vals[il as usize] != want {
            return Err(ValidateError::TokenLayer("swa-pattern", il));
        }
    }
    Ok(())
}

fn qwen_ssd_precision(quant: Option<&[u8]>) -> Option<(u32, u32, u32, u32)> {
    match quant {
        Some(b"MQ-Q6-SSD-PLE-BF16") => Some((14, 13, 14, 6)),
        Some(b"MQ-Q5-SSD-PLE-BF16") => Some((13, 12, 13, 6)),
        _ => None,
    }
}

fn qwen_source_revision_supported(revision: Option<&[u8]>) -> bool {
    match revision {
        Some(got) => QWEN_REVISIONS.contains(&got),
        None => false,
    }
}

fn validate_qwen_external_ple(g: &GgufFile) -> Result<(), ValidateError> {
    let external = qwen_ssd_precision(g.get_string("general.quantization")).is_some();
    if external != has_key(g, "qwen4exp.ple.weight_storage") {
        return Err(ValidateError::Token("ple-contract"));
    }
    if !external {
        return Ok(());
    }
    expect_string(
        g,
        "qwen4exp.ple.weight_storage",
        b"ssd_backed_bounded_page_cache",
    )?;
    expect_string(g, "qwen4exp.ple.storage_dtype", b"BF16")?;
    expect_string(g, "qwen4exp.ple.sidecar_manifest", b"ple/ple-manifest.json")?;
    expect_bool(
        "ple.resident_weight",
        req_bool(g, "qwen4exp.ple.resident_weight")?,
        false,
    )?;
    expect_u64(
        "ple.sidecar_payload_bytes",
        req_u64c(g, "qwen4exp.ple.sidecar_payload_bytes")?,
        102_400_491_520,
    )?;
    expect_u32(
        "ple.sidecar_alignment",
        req_u32(g, "qwen4exp.ple.sidecar_alignment")?,
        4096,
    )
}

fn validate_qwen4exp(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    let u32s = [
        ("block_count", "qwen4exp.block_count", shape.n_layer),
        (
            "embedding_length",
            "qwen4exp.embedding_length",
            shape.n_embd,
        ),
        ("vocab_size", "qwen4exp.vocab_size", shape.n_vocab),
        (
            "feed_forward_length",
            "qwen4exp.feed_forward_length",
            shape.n_ff_exp,
        ),
        (
            "attention.head_count",
            "qwen4exp.attention.head_count",
            shape.n_head,
        ),
        (
            "attention.head_count_kv",
            "qwen4exp.attention.head_count_kv",
            shape.n_head_kv,
        ),
        (
            "attention.key_length",
            "qwen4exp.attention.key_length",
            shape.n_head_dim,
        ),
        (
            "attention.value_length",
            "qwen4exp.attention.value_length",
            shape.n_value_dim,
        ),
        (
            "rope.dimension_count",
            "qwen4exp.rope.dimension_count",
            shape.n_rot,
        ),
        ("expert_count", "qwen4exp.expert_count", shape.n_expert),
        (
            "expert_used_count",
            "qwen4exp.expert_used_count",
            shape.n_expert_used,
        ),
        (
            "expert_feed_forward_length",
            "qwen4exp.expert_feed_forward_length",
            shape.n_ff_exp,
        ),
        (
            "expert_shared_count",
            "qwen4exp.expert_shared_count",
            shape.n_expert_shared,
        ),
        (
            "expert_shared_feed_forward_length",
            "qwen4exp.expert_shared_feed_forward_length",
            shape.n_ff_shexp,
        ),
        ("nextn_predict_layers", "qwen4exp.nextn_predict_layers", 1),
        (
            "full_attention_interval",
            "qwen4exp.full_attention_interval",
            4,
        ),
        (
            "attention.indexer.head_count",
            "qwen4exp.attention.indexer.head_count",
            shape.n_indexer_head,
        ),
        (
            "attention.indexer.key_length",
            "qwen4exp.attention.indexer.key_length",
            shape.n_indexer_head_dim,
        ),
        (
            "attention.indexer.top_k",
            "qwen4exp.attention.indexer.top_k",
            shape.n_indexer_top_k,
        ),
        (
            "attention.linear_key_head_count",
            "qwen4exp.attention.linear_key_head_count",
            16,
        ),
        (
            "attention.linear_value_head_count",
            "qwen4exp.attention.linear_value_head_count",
            48,
        ),
        (
            "attention.linear_key_length",
            "qwen4exp.attention.linear_key_length",
            128,
        ),
        (
            "attention.linear_value_length",
            "qwen4exp.attention.linear_value_length",
            128,
        ),
        (
            "attention.linear_conv_kernel",
            "qwen4exp.attention.linear_conv_kernel",
            4,
        ),
        (
            "attention.indexer.compress_ratio",
            "qwen4exp.attention.indexer.compress_ratio",
            4,
        ),
        (
            "attention.indexer.kv_head_count",
            "qwen4exp.attention.indexer.kv_head_count",
            1,
        ),
        (
            "hyper_connection.count",
            "qwen4exp.hyper_connection.count",
            shape.n_hc,
        ),
        (
            "hyper_connection.lowrank",
            "qwen4exp.hyper_connection.lowrank",
            320,
        ),
        ("ple.ngram_size", "qwen4exp.ple.ngram_size", 3),
        ("ple.heads_per_ngram", "qwen4exp.ple.heads_per_ngram", 8),
        (
            "ple.ngram_vocab_size_base",
            "qwen4exp.ple.ngram_vocab_size_base",
            20_000_000,
        ),
        ("ple.ngram_parts", "qwen4exp.ple.ngram_parts", 128),
        (
            "ple.embedding_length",
            "qwen4exp.ple.embedding_length",
            shape.n_embd,
        ),
        ("ple.conv_kernel", "qwen4exp.ple.conv_kernel", 4),
        ("mtp.block_count", "qwen4exp.mtp.block_count", 1),
        ("vision.block_count", "qwen4exp.vision.block_count", 27),
        (
            "vision.embedding_length",
            "qwen4exp.vision.embedding_length",
            1152,
        ),
        (
            "vision.feed_forward_length",
            "qwen4exp.vision.feed_forward_length",
            4304,
        ),
        ("vision.head_count", "qwen4exp.vision.head_count", 16),
        (
            "vision.output_embedding_length",
            "qwen4exp.vision.output_embedding_length",
            shape.n_embd,
        ),
        ("vision.patch_size", "qwen4exp.vision.patch_size", 16),
        (
            "vision.temporal_patch_size",
            "qwen4exp.vision.temporal_patch_size",
            2,
        ),
        (
            "vision.spatial_merge_size",
            "qwen4exp.vision.spatial_merge_size",
            2,
        ),
        (
            "vision.position_count",
            "qwen4exp.vision.position_count",
            2304,
        ),
    ];
    for (name, key, want) in u32s {
        expect_u32(name, req_u32(g, key)?, want)?;
    }
    expect_u64(
        "context_length",
        req_u64c(g, "qwen4exp.context_length")?,
        shape.rope_orig_ctx,
    )?;
    expect_bool(
        "rms_norm.zero_centered",
        req_bool(g, "qwen4exp.rms_norm.zero_centered")?,
        true,
    )?;
    validate_qwen_external_ple(g)?;
    expect_bool("mtp.present", req_bool(g, "qwen4exp.mtp.present")?, true)?;
    expect_bool(
        "vision.present",
        req_bool(g, "qwen4exp.vision.present")?,
        true,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        req_f32(g, "qwen4exp.attention.layer_norm_rms_epsilon")?,
        shape.rms_eps,
    )?;
    expect_f32(
        "rope.freq_base",
        req_f32(g, "qwen4exp.rope.freq_base")?,
        shape.rope_freq_base,
    )?;
    expect_string(
        g,
        "qwen4exp.attention.output_gate_type",
        b"sigmoid".as_slice(),
    )?;
    if !qwen_source_revision_supported(g.get_string("general.source.revision")) {
        return Err(ValidateError::TokenKey(
            "mismatch-string",
            "general.source.revision".into(),
        ));
    }
    for (key, want) in [
        ("qwen4exp.source.config_sha256", QWEN_CONFIG_SHA),
        ("qwen4exp.source.license_sha256", QWEN_LICENSE_SHA),
        ("general.license", b"other".as_slice()),
        ("general.license.name", b"qwen-community-1.0".as_slice()),
    ] {
        expect_string(g, key, want)?;
    }

    let types = g
        .get_array("qwen4exp.attention.layer_types")
        .ok_or(ValidateError::TokenKey(
            "missing-array",
            "qwen4exp.attention.layer_types".into(),
        ))?;
    if types.typ != GGUF_VALUE_STRING || types.len != u64::from(shape.n_layer) {
        return Err(ValidateError::TokenKey(
            "array-shape",
            "qwen4exp.attention.layer_types".into(),
        ));
    }
    for (il, got) in g.array_strings(&types)?.into_iter().enumerate() {
        let full = qwen4exp_layer_is_full_attention(shape, il as u32);
        let want: &[u8] = if full {
            b"full_attention"
        } else {
            b"linear_attention"
        };
        if got != want {
            return Err(ValidateError::TokenLayer("layer-type", il as u32));
        }
    }
    for (i, got) in array_nonnegative_u32s(
        g,
        "qwen4exp.attention.full_layers",
        u64::from(shape.n_full_attn_count),
    )?
    .into_iter()
    .enumerate()
    {
        if got != 4 * i as u32 + 3 {
            return Err(ValidateError::TokenLayer("full-layer", i as u32));
        }
    }
    for (key, want) in [
        ("qwen4exp.ple.layer_ids_one_based", 2),
        ("qwen4exp.ple.checkpoint_layer_ids_zero_based", 1),
    ] {
        if array_nonnegative_u32s(g, key, 1)? != [want] {
            return Err(mismatch(key));
        }
    }
    Ok(())
}

/// Qwen tensor-side storage and Q5/Q6 SSD precision contract.
pub fn validate_qwen_inventory(
    g: &GgufFile,
    inventory: &TensorInventory,
) -> Result<(), ValidateError> {
    let Some((edge, interior, down, tail)) =
        qwen_ssd_precision(g.get_string("general.quantization"))
    else {
        return Ok(());
    };
    for part in 0..128 {
        let name = format!("blk.1.ple.ngram_embd.part_{part:03}.weight");
        if inventory.find(&name).is_some() {
            return Err(ValidateError::TokenKey("resident-ple", name));
        }
    }
    for il in 0..48 {
        let gate = if il < 2 || il >= 46 { edge } else { interior };
        for (suffix, want) in [
            ("ffn_gate_exps.weight", gate),
            ("ffn_up_exps.weight", gate),
            ("ffn_down_exps.main.weight", down),
            ("ffn_down_exps.tail.weight", tail),
        ] {
            let name = format!("blk.{il}.{suffix}");
            let got = inventory
                .find(&name)
                .ok_or_else(|| ValidateError::TokenKey("missing-tensor", name.clone()))?;
            if got.typ != want {
                return Err(ValidateError::TokenLayer("precision-map", il));
            }
        }
    }
    Ok(())
}

/// Full C `config_validate_model` against an already-identified shape.
pub fn validate_file(g: &GgufFile, shape: &Shape) -> Result<(), ValidateError> {
    match shape.family {
        ModelFamily::Qwen4Exp => validate_qwen4exp(g, shape),
        ModelFamily::DeepSeek4 => validate_deepseek(g, shape),
        ModelFamily::Motif3 => validate_motif3(g, shape),
        ModelFamily::Dots3Note => validate_dots3(g, shape),
        ModelFamily::SolarOpen2 => validate_solar(g, shape),
        ModelFamily::ExaoneMoe => validate_exaone(g, shape),
    }
}

pub fn validate_gguf(path: &std::path::Path) -> Result<Shape, ValidateError> {
    let g = GgufFile::open(path)?;
    let id = identify_file(&g)?;
    validate_file(&g, &id.shape)?;
    Ok(id.shape)
}

pub fn host_compress_ratios(shape: &Shape) -> Vec<u32> {
    if shape.family != ModelFamily::DeepSeek4 {
        return Vec::new();
    }
    (0..shape.n_layer)
        .map(|il| expected_compress_ratio(shape.variant, shape.n_layer, il))
        .collect()
}

/// C `validate_c_oracle PATH` stdout (one token line).
pub fn dump_validate(path: &std::path::Path) -> String {
    match GgufFile::open(path) {
        Ok(g) => {
            let arch = g.get_string("general.architecture");
            match route_architecture(arch) {
                ArchRoute::Unsupported => {
                    let a = arch.unwrap_or(b"");
                    return format!("unsupported-arch {}\n", String::from_utf8_lossy(a));
                }
                ArchRoute::DeepSeekSelect => match identify_file(&g) {
                    Ok(id) => match validate_file(&g, &id.shape) {
                        Ok(()) => "ok\n".into(),
                        Err(e) => format!("{}\n", e.token()),
                    },
                    Err(IdentifyError::UnsupportedShape) => "unsupported\n".into(),
                    Err(e) => format!("{}\n", e.token()),
                },
                ArchRoute::Fixed(v) => {
                    let shape = match v {
                        Variant::Motif3 => SHAPE_MOTIF3,
                        Variant::SolarOpen2_250B => SHAPE_SOLAR_OPEN2_250B,
                        Variant::Kexaone236B => SHAPE_KEXAONE_236B,
                        Variant::Dots3NotePrev => SHAPE_DOTS3_NOTE_PREV,
                        Variant::Qwen38FlashNext => SHAPE_QWEN38_FLASH_NEXT,
                        Variant::Flash => SHAPE_FLASH,
                        Variant::Pro => SHAPE_PRO,
                    };
                    match validate_file(&g, &shape) {
                        Ok(()) => "ok\n".into(),
                        Err(e) => format!("{}\n", e.token()),
                    }
                }
            }
        }
        Err(e) => format!("{}\n", e.token()),
    }
}

#[cfg(test)]
mod tests {
    use super::qwen_source_revision_supported;

    #[test]
    fn qwen_source_revision_allowlist_is_exact() {
        assert!(qwen_source_revision_supported(Some(
            b"f5d08274bafd880402bd16f5e3e6c514136ec06c"
        )));
        assert!(qwen_source_revision_supported(Some(
            b"8336e613ea508b13c2159bd0f68965d97a606b95"
        )));
        assert!(!qwen_source_revision_supported(Some(b"unknown")));
        assert!(!qwen_source_revision_supported(None));
    }
}
