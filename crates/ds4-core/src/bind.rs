//! Host-owned `weights_bind` name catalog + inventory resolve.
//!
//! Copied from `ds4.c` (`weights_bind` and the family layer binders).
//! CUDA upload still happens inside `ds4_engine_open`; this crate owns
//! the name table native bind consumes and refuses open when a required
//! tensor is missing from the host inventory.

use crate::shape::{shape_for_variant, ModelFamily, Shape, Variant};
use crate::tensors::{tensor_type_name, TensorInfo, TensorInventory, MAX_DIMS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindNeed {
    Required,
    Optional,
}

impl BindNeed {
    pub fn token(self) -> &'static str {
        match self {
            BindNeed::Required => "REQ",
            BindNeed::Optional => "OPT",
        }
    }

    pub fn required(self) -> bool {
        matches!(self, BindNeed::Required)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindName {
    pub name: String,
    pub need: BindNeed,
}

#[derive(Clone, Debug)]
pub struct BindSlot {
    pub name: String,
    pub need: BindNeed,
    pub tensor: Option<TensorInfo>,
    pub index: Option<u32>,
}

#[derive(Debug)]
pub enum BindError {
    Missing(String),
    NameEmpty,
    BadNdim,
    CountMismatch,
    NameMismatch,
    NeedMismatch,
    FoundMismatch,
    TypeMismatch,
    DimMismatch,
    OffsetMismatch,
    BytesMismatch,
    ShardMismatch,
    DataMismatch,
    PlanNull,
    SlotsNull,
    ShardsNull,
}

impl BindError {
    pub fn token(&self) -> String {
        match self {
            BindError::Missing(n) => format!("missing {n}"),
            BindError::NameEmpty => "name-empty".into(),
            BindError::BadNdim => "bad-ndim".into(),
            BindError::CountMismatch => "count-mismatch".into(),
            BindError::NameMismatch => "name-mismatch".into(),
            BindError::NeedMismatch => "need-mismatch".into(),
            BindError::FoundMismatch => "found-mismatch".into(),
            BindError::TypeMismatch => "type-mismatch".into(),
            BindError::DimMismatch => "dim-mismatch".into(),
            BindError::OffsetMismatch => "offset-mismatch".into(),
            BindError::BytesMismatch => "bytes-mismatch".into(),
            BindError::ShardMismatch => "shard-mismatch".into(),
            BindError::DataMismatch => "data-mismatch".into(),
            BindError::PlanNull => "plan-null".into(),
            BindError::SlotsNull => "slots-null".into(),
            BindError::ShardsNull => "shards-null".into(),
        }
    }
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for BindError {}

#[derive(Clone, Debug)]
pub struct BindPlan {
    pub shape: Shape,
    pub slots: Vec<BindSlot>,
    pub n_shards: u32,
    pub data_pos: u64,
    pub alignment: u64,
    pub page: u64,
}

fn req(out: &mut Vec<BindName>, name: impl Into<String>) {
    out.push(BindName {
        name: name.into(),
        need: BindNeed::Required,
    });
}

fn opt(out: &mut Vec<BindName>, name: impl Into<String>) {
    out.push(BindName {
        name: name.into(),
        need: BindNeed::Optional,
    });
}

fn reqf(out: &mut Vec<BindName>, fmt: &str, il: u32) {
    req(out, format_name(fmt, il));
}

fn optf(out: &mut Vec<BindName>, fmt: &str, il: u32) {
    opt(out, format_name(fmt, il));
}

/// C `snprintf(name, 128, fmt, layer)`.
fn format_name(fmt: &str, il: u32) -> String {
    // All C formats are a single `%u` substitution.
    fmt.replacen("%u", &il.to_string(), 1)
}

/// C `ds4_expected_layer_compress_ratio`.
pub fn expected_compress_ratio(variant: Variant, n_layer: u32, il: u32) -> u32 {
    if il >= n_layer {
        return 0;
    }
    match variant {
        Variant::Flash => {
            if il < 2 {
                0
            } else if (il & 1) == 0 {
                4
            } else {
                128
            }
        }
        Variant::Pro => {
            if il < 2 {
                128
            } else if (il & 1) == 0 {
                4
            } else {
                128
            }
        }
        _ => 0,
    }
}

/// C `ds4_solar_layer_is_gqa`.
pub fn solar_layer_is_gqa(family: ModelFamily, n_layer: u32, il: u32) -> bool {
    family == ModelFamily::SolarOpen2 && il < n_layer && (il % 4) == 0
}

/// C `ds4_dots3_layer_is_full_attention`.
pub fn dots3_layer_is_full_attention(shape: &Shape, il: u32) -> bool {
    if shape.family != ModelFamily::Dots3Note || il >= shape.n_layer {
        return false;
    }
    if shape.n_nextn_predict != 0 && il + shape.n_nextn_predict >= shape.n_layer {
        return false;
    }
    il == 0 || (shape.n_swa_period != 0 && (il % shape.n_swa_period) == 1)
}

/// C `ds4_qwen4exp_layer_is_full_attention`.
pub fn qwen4exp_layer_is_full_attention(shape: &Shape, il: u32) -> bool {
    shape.family == ModelFamily::Qwen4Exp
        && il < shape.n_layer
        && shape.n_swa_period != 0
        && (il % shape.n_swa_period) == 3
}

/// GLM 5.3 uses KDA,KDA,KDA,DSA in each trunk group; the final block is MTP/DSA.
pub fn glm53_layer_is_kda(shape: &Shape, il: u32) -> bool {
    shape.family == ModelFamily::Glm53
        && il < shape.n_layer
        && !is_nextn(shape, il)
        && (il % 4) != 3
}

fn is_nextn(shape: &Shape, il: u32) -> bool {
    shape.n_nextn_predict != 0 && il + shape.n_nextn_predict >= shape.n_layer
}

fn bind_glm53_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.ffn_norm.weight", il);
    if !is_nextn(shape, il) {
        for suffix in [
            "hc_attn_fn.weight",
            "hc_attn_scale.weight",
            "hc_attn_base.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_scale.weight",
            "hc_ffn_base.weight",
        ] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    }
    if glm53_layer_is_kda(shape, il) {
        for suffix in [
            "kda_q.weight",
            "kda_k.weight",
            "kda_v.weight",
            "kda_q_conv.weight",
            "kda_k_conv.weight",
            "kda_v_conv.weight",
            "kda_f_a.weight",
            "kda_f_b.weight",
            "kda_dt_bias.weight",
            "kda_a_log.weight",
            "kda_beta.weight",
            "kda_g_a.weight",
            "kda_g_b.weight",
            "kda_o_norm.weight",
            "kda_output.weight",
        ] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    } else {
        for suffix in [
            "attn_q_a.weight",
            "attn_q_a_norm.weight",
            "attn_q_b.weight",
            "attn_kv_a_mqa.weight",
            "attn_kv_a_norm.weight",
            "attn_k_b.weight",
            "attn_v_b.weight",
            "attn_output.weight",
            "indexer.attn_q_b.weight",
            "indexer.attn_k.weight",
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
            "indexer.proj.weight",
            "indexer.pool_ape.weight",
            "indexer.pool_gate.weight",
        ] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    }
    if il < shape.n_leading_dense {
        for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    } else {
        for suffix in [
            "ffn_gate_inp.weight",
            "exp_probs_b.bias",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    }
    if is_nextn(shape, il) {
        for suffix in [
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
        ] {
            reqf(out, &format!("blk.%u.{suffix}"), il);
        }
    }
}

fn bind_motif3_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.mhc_attn.rms_norm.weight", il);
    reqf(out, "blk.%u.mhc_attn.proj_pre.weight", il);
    reqf(out, "blk.%u.mhc_attn.proj_post.weight", il);
    reqf(out, "blk.%u.mhc_attn.proj_res.weight", il);
    reqf(out, "blk.%u.mhc_attn.alpha_pre", il);
    reqf(out, "blk.%u.mhc_attn.alpha_post", il);
    reqf(out, "blk.%u.mhc_attn.alpha_res", il);
    reqf(out, "blk.%u.mhc_attn.bias_pre", il);
    reqf(out, "blk.%u.mhc_attn.bias_post", il);
    reqf(out, "blk.%u.mhc_attn.bias_res", il);
    reqf(out, "blk.%u.attn_q_a.weight", il);
    reqf(out, "blk.%u.attn_q_a_norm.weight", il);
    reqf(out, "blk.%u.attn_q_b.weight", il);
    reqf(out, "blk.%u.attn_q_gate.weight", il);
    reqf(out, "blk.%u.attn_kv_a.weight", il);
    reqf(out, "blk.%u.attn_kv_a_norm.weight", il);
    reqf(out, "blk.%u.attn_kv_b.weight", il);
    reqf(out, "blk.%u.attn_lambda.weight", il);
    reqf(out, "blk.%u.attn_output.weight", il);
    reqf(out, "blk.%u.mhc_ffn.rms_norm.weight", il);
    reqf(out, "blk.%u.mhc_ffn.proj_pre.weight", il);
    reqf(out, "blk.%u.mhc_ffn.proj_post.weight", il);
    reqf(out, "blk.%u.mhc_ffn.proj_res.weight", il);
    reqf(out, "blk.%u.mhc_ffn.alpha_pre", il);
    reqf(out, "blk.%u.mhc_ffn.alpha_post", il);
    reqf(out, "blk.%u.mhc_ffn.alpha_res", il);
    reqf(out, "blk.%u.mhc_ffn.bias_pre", il);
    reqf(out, "blk.%u.mhc_ffn.bias_post", il);
    reqf(out, "blk.%u.mhc_ffn.bias_res", il);
    reqf(out, "blk.%u.ffn_norm.weight", il);
    if il < shape.n_leading_dense {
        reqf(out, "blk.%u.ffn_gate.weight", il);
        reqf(out, "blk.%u.ffn_up.weight", il);
        reqf(out, "blk.%u.ffn_down.weight", il);
        reqf(out, "blk.%u.ffn_polynorm.weight", il);
        reqf(out, "blk.%u.ffn_polynorm.bias", il);
        return;
    }
    reqf(out, "blk.%u.ffn_gate_inp.weight", il);
    reqf(out, "blk.%u.exp_probs_b.bias", il);
    reqf(out, "blk.%u.ffn_gate_exps.weight", il);
    reqf(out, "blk.%u.ffn_up_exps.weight", il);
    reqf(out, "blk.%u.ffn_down_exps.weight", il);
    reqf(out, "blk.%u.ffn_polynorm_exps.weight", il);
    reqf(out, "blk.%u.ffn_polynorm_exps.bias", il);
    reqf(out, "blk.%u.ffn_gate_shexp.weight", il);
    reqf(out, "blk.%u.ffn_up_shexp.weight", il);
    reqf(out, "blk.%u.ffn_down_shexp.weight", il);
    reqf(out, "blk.%u.ffn_polynorm_shexp.weight", il);
    reqf(out, "blk.%u.ffn_polynorm_shexp.bias", il);
}

fn bind_motif3_mtp(out: &mut Vec<BindName>) {
    req(out, "mtp.0.embed_norm.weight");
    req(out, "mtp.0.input_layernorm.weight");
    req(out, "mtp.0.input_proj.weight");
    req(out, "mtp.0.final_layernorm.weight");
    req(out, "mtp.0.attn_q_a.weight");
    req(out, "mtp.0.attn_q_a_norm.weight");
    req(out, "mtp.0.attn_q_b.weight");
    req(out, "mtp.0.attn_q_gate.weight");
    req(out, "mtp.0.attn_kv_a.weight");
    req(out, "mtp.0.attn_kv_a_norm.weight");
    req(out, "mtp.0.attn_kv_b.weight");
    req(out, "mtp.0.attn_lambda.weight");
    req(out, "mtp.0.attn_output.weight");
    req(out, "mtp.0.post_attention_layernorm.weight");
    req(out, "mtp.0.ffn_gate.weight");
    req(out, "mtp.0.ffn_up.weight");
    req(out, "mtp.0.ffn_down.weight");
    req(out, "mtp.0.ffn_polynorm.weight");
    req(out, "mtp.0.ffn_polynorm.bias");
}

fn bind_exaone_moe_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.attn_q.weight", il);
    reqf(out, "blk.%u.attn_k.weight", il);
    reqf(out, "blk.%u.attn_v.weight", il);
    reqf(out, "blk.%u.attn_output.weight", il);
    if shape.use_qk_norm {
        reqf(out, "blk.%u.attn_q_norm.weight", il);
        reqf(out, "blk.%u.attn_k_norm.weight", il);
    }
    reqf(out, "blk.%u.ffn_norm.weight", il);
    if il < shape.n_leading_dense || is_nextn(shape, il) {
        reqf(out, "blk.%u.ffn_gate.weight", il);
        reqf(out, "blk.%u.ffn_up.weight", il);
        reqf(out, "blk.%u.ffn_down.weight", il);
    } else {
        reqf(out, "blk.%u.ffn_gate_inp.weight", il);
        reqf(out, "blk.%u.exp_probs_b.bias", il);
        reqf(out, "blk.%u.ffn_gate_exps.weight", il);
        reqf(out, "blk.%u.ffn_up_exps.weight", il);
        reqf(out, "blk.%u.ffn_down_exps.weight", il);
        reqf(out, "blk.%u.ffn_gate_shexp.weight", il);
        reqf(out, "blk.%u.ffn_up_shexp.weight", il);
        reqf(out, "blk.%u.ffn_down_shexp.weight", il);
    }
    if is_nextn(shape, il) {
        reqf(out, "blk.%u.nextn.eh_proj.weight", il);
        reqf(out, "blk.%u.nextn.enorm.weight", il);
        reqf(out, "blk.%u.nextn.hnorm.weight", il);
        reqf(out, "blk.%u.nextn.shared_head_norm.weight", il);
    }
}

fn bind_dots3_note_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.attn_q_a.weight", il);
    reqf(out, "blk.%u.attn_q_a_norm.weight", il);
    reqf(out, "blk.%u.attn_q_b.weight", il);
    reqf(out, "blk.%u.attn_kv_a_mqa.weight", il);
    reqf(out, "blk.%u.attn_kv_a_norm.weight", il);
    reqf(out, "blk.%u.attn_kv_b.weight", il);
    reqf(out, "blk.%u.attn_k_rope_norm.weight", il);
    reqf(out, "blk.%u.attn_gate.weight", il);
    reqf(out, "blk.%u.attn_output.weight", il);
    if dots3_layer_is_full_attention(shape, il) {
        reqf(out, "blk.%u.attn_idx_q_b.weight", il);
        reqf(out, "blk.%u.attn_idx_k.weight", il);
        reqf(out, "blk.%u.attn_idx_w.weight", il);
        reqf(out, "blk.%u.attn_idx_k_norm.weight", il);
        reqf(out, "blk.%u.attn_idx_k_norm.bias", il);
    }
    reqf(out, "blk.%u.ffn_norm.weight", il);
    if il < shape.n_leading_dense || is_nextn(shape, il) {
        reqf(out, "blk.%u.ffn_gate.weight", il);
        reqf(out, "blk.%u.ffn_up.weight", il);
        reqf(out, "blk.%u.ffn_down.weight", il);
    } else {
        reqf(out, "blk.%u.ffn_gate_inp.weight", il);
        reqf(out, "blk.%u.exp_probs_b.bias", il);
        reqf(out, "blk.%u.ffn_gate_exps.weight", il);
        reqf(out, "blk.%u.ffn_up_exps.weight", il);
        reqf(out, "blk.%u.ffn_down_exps.weight", il);
        reqf(out, "blk.%u.ffn_gate_shexp.weight", il);
        reqf(out, "blk.%u.ffn_up_shexp.weight", il);
        reqf(out, "blk.%u.ffn_down_shexp.weight", il);
    }
    if is_nextn(shape, il) {
        reqf(out, "blk.%u.eh_proj.weight", il);
        reqf(out, "blk.%u.enorm.weight", il);
        reqf(out, "blk.%u.hnorm.weight", il);
        reqf(out, "blk.%u.shared_head_norm.weight", il);
    }
}

fn bind_solar_open2_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.attn_q.weight", il);
    reqf(out, "blk.%u.attn_k.weight", il);
    reqf(out, "blk.%u.attn_v.weight", il);
    reqf(out, "blk.%u.attn_output.weight", il);
    if solar_layer_is_gqa(shape.family, shape.n_layer, il) {
        reqf(out, "blk.%u.attn_gate.weight", il);
    } else {
        reqf(out, "blk.%u.ssm_conv1d_q.weight", il);
        reqf(out, "blk.%u.ssm_conv1d_k.weight", il);
        reqf(out, "blk.%u.ssm_conv1d_v.weight", il);
        reqf(out, "blk.%u.ssm_f_a.weight", il);
        reqf(out, "blk.%u.ssm_f_b.weight", il);
        reqf(out, "blk.%u.ssm_beta.weight", il);
        reqf(out, "blk.%u.ssm_a", il);
        reqf(out, "blk.%u.ssm_dt.bias", il);
        reqf(out, "blk.%u.ssm_g_a.weight", il);
        reqf(out, "blk.%u.ssm_g_b.weight", il);
        reqf(out, "blk.%u.ssm_norm.weight", il);
    }
    reqf(out, "blk.%u.ffn_norm.weight", il);
    reqf(out, "blk.%u.ffn_gate_inp.weight", il);
    reqf(out, "blk.%u.exp_probs_b.bias", il);
    reqf(out, "blk.%u.ffn_gate_exps.weight", il);
    reqf(out, "blk.%u.ffn_up_exps.weight", il);
    reqf(out, "blk.%u.ffn_down_exps.weight", il);
    reqf(out, "blk.%u.ffn_gate_shexp.weight", il);
    reqf(out, "blk.%u.ffn_up_shexp.weight", il);
    reqf(out, "blk.%u.ffn_down_shexp.weight", il);
}

fn bind_qwen4exp_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    for name in [
        "blk.%u.hc_attn.norm.weight",
        "blk.%u.hc_attn.mix_down.weight",
        "blk.%u.hc_attn.mix_up.weight",
        "blk.%u.hc_attn.inject.weight",
    ] {
        reqf(out, name, il);
    }
    if qwen4exp_layer_is_full_attention(shape, il) {
        for name in [
            "blk.%u.attn_index_qk.weight",
            "blk.%u.attn_index_q_norm.weight",
            "blk.%u.attn_index_k_norm.weight",
            "blk.%u.attn_q.weight",
            "blk.%u.attn_q_norm.weight",
            "blk.%u.attn_k.weight",
            "blk.%u.attn_k_norm.weight",
            "blk.%u.attn_v.weight",
            "blk.%u.attn_output.weight",
        ] {
            reqf(out, name, il);
        }
    } else {
        for name in [
            "blk.%u.linear_attn.a_log",
            "blk.%u.linear_attn.conv.weight",
            "blk.%u.linear_attn.dt_bias",
            "blk.%u.linear_attn.in_a.weight",
            "blk.%u.linear_attn.in_b.weight",
            "blk.%u.linear_attn.qkv.weight",
            "blk.%u.linear_attn.z.weight",
            "blk.%u.linear_attn.norm.weight",
            "blk.%u.linear_attn.out.weight",
        ] {
            reqf(out, name, il);
        }
    }
    for name in [
        "blk.%u.ffn_gate_inp.weight",
        "blk.%u.ffn_gate_exps.weight",
        "blk.%u.ffn_up_exps.weight",
        "blk.%u.ffn_down_exps.main.weight",
        "blk.%u.ffn_down_exps.tail.weight",
        "blk.%u.ffn_gate_shexp.weight",
        "blk.%u.ffn_up_shexp.weight",
        "blk.%u.ffn_down_shexp.weight",
        "blk.%u.ffn_shexp_gate_inp.weight",
        "blk.%u.hc_ffn.norm.weight",
        "blk.%u.hc_ffn.mix_down.weight",
        "blk.%u.hc_ffn.mix_up.weight",
        "blk.%u.hc_ffn.inject.weight",
    ] {
        reqf(out, name, il);
    }
    if il == 1 {
        for name in [
            "blk.1.ple.conv.weight",
            "blk.1.ple.key.weight",
            "blk.1.ple.value.weight",
            "blk.1.ple.conv_norm.weight",
            "blk.1.ple.key_norm.weight",
            "blk.1.ple.query_norm.weight",
            "blk.1.ple.layer_multipliers",
            "blk.1.ple.head_offsets",
            "blk.1.ple.head_vocab_sizes",
        ] {
            req(out, name);
        }
    }
}

fn bind_qwen4exp_mtp(out: &mut Vec<BindName>) {
    for name in [
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.fc_embedding_norm.weight",
        "mtp.fc_hidden_norm.weight",
        "mtp.hc_input.norm.weight",
        "mtp.hc_input.mix_down.weight",
        "mtp.hc_input.mix_up.weight",
        "mtp.blk.0.hc_attn.norm.weight",
        "mtp.blk.0.hc_attn.mix_down.weight",
        "mtp.blk.0.hc_attn.mix_up.weight",
        "mtp.blk.0.hc_attn.inject.weight",
        "mtp.blk.0.attn_index_qk.weight",
        "mtp.blk.0.attn_index_q_norm.weight",
        "mtp.blk.0.attn_index_k_norm.weight",
        "mtp.blk.0.attn_q.weight",
        "mtp.blk.0.attn_q_norm.weight",
        "mtp.blk.0.attn_k.weight",
        "mtp.blk.0.attn_k_norm.weight",
        "mtp.blk.0.attn_v.weight",
        "mtp.blk.0.attn_output.weight",
        "mtp.blk.0.ffn_gate_inp.weight",
        "mtp.blk.0.ffn_gate_exps.weight",
        "mtp.blk.0.ffn_up_exps.weight",
        "mtp.blk.0.ffn_down_exps.main.weight",
        "mtp.blk.0.ffn_down_exps.tail.weight",
        "mtp.blk.0.ffn_gate_shexp.weight",
        "mtp.blk.0.ffn_up_shexp.weight",
        "mtp.blk.0.ffn_down_shexp.weight",
        "mtp.blk.0.ffn_shexp_gate_inp.weight",
        "mtp.blk.0.hc_ffn.norm.weight",
        "mtp.blk.0.hc_ffn.mix_down.weight",
        "mtp.blk.0.hc_ffn.mix_up.weight",
        "mtp.blk.0.hc_ffn.inject.weight",
    ] {
        req(out, name);
    }
}

fn bind_qwen4exp_vision(out: &mut Vec<BindName>) {
    req(out, "vision.patch_embed.weight");
    req(out, "vision.patch_embed.bias");
    req(out, "vision.position_embd.weight");
    for il in 0..27 {
        for name in [
            "vblk.%u.norm1.weight",
            "vblk.%u.norm1.bias",
            "vblk.%u.attn_qkv.weight",
            "vblk.%u.attn_qkv.bias",
            "vblk.%u.attn_output.weight",
            "vblk.%u.attn_output.bias",
            "vblk.%u.norm2.weight",
            "vblk.%u.norm2.bias",
            "vblk.%u.ffn_up.weight",
            "vblk.%u.ffn_up.bias",
            "vblk.%u.ffn_down.weight",
            "vblk.%u.ffn_down.bias",
        ] {
            reqf(out, name, il);
        }
    }
    for name in [
        "vision.merger.norm.weight",
        "vision.merger.norm.bias",
        "vision.merger.ffn_up.weight",
        "vision.merger.ffn_up.bias",
        "vision.merger.ffn_down.weight",
        "vision.merger.ffn_down.bias",
    ] {
        req(out, name);
    }
}

fn bind_deepseek_layer(out: &mut Vec<BindName>, shape: &Shape, il: u32) {
    let compress_ratio = expected_compress_ratio(shape.variant, shape.n_layer, il);
    reqf(out, "blk.%u.hc_attn_fn.weight", il);
    reqf(out, "blk.%u.hc_attn_scale.weight", il);
    reqf(out, "blk.%u.hc_attn_base.weight", il);
    reqf(out, "blk.%u.attn_norm.weight", il);
    reqf(out, "blk.%u.attn_q_a.weight", il);
    reqf(out, "blk.%u.attn_q_a_norm.weight", il);
    reqf(out, "blk.%u.attn_q_b.weight", il);
    reqf(out, "blk.%u.attn_kv.weight", il);
    reqf(out, "blk.%u.attn_kv_a_norm.weight", il);
    reqf(out, "blk.%u.attn_sinks.weight", il);
    reqf(out, "blk.%u.attn_output_a.weight", il);
    reqf(out, "blk.%u.attn_output_b.weight", il);
    if compress_ratio != 0 {
        reqf(out, "blk.%u.attn_compressor_ape.weight", il);
        reqf(out, "blk.%u.attn_compressor_kv.weight", il);
        reqf(out, "blk.%u.attn_compressor_gate.weight", il);
        reqf(out, "blk.%u.attn_compressor_norm.weight", il);
    }
    if compress_ratio == 4 {
        reqf(out, "blk.%u.indexer.attn_q_b.weight", il);
        reqf(out, "blk.%u.indexer.proj.weight", il);
        reqf(out, "blk.%u.indexer_compressor_ape.weight", il);
        reqf(out, "blk.%u.indexer_compressor_kv.weight", il);
        reqf(out, "blk.%u.indexer_compressor_gate.weight", il);
        reqf(out, "blk.%u.indexer_compressor_norm.weight", il);
    }
    reqf(out, "blk.%u.hc_ffn_fn.weight", il);
    reqf(out, "blk.%u.hc_ffn_scale.weight", il);
    reqf(out, "blk.%u.hc_ffn_base.weight", il);
    reqf(out, "blk.%u.ffn_norm.weight", il);
    reqf(out, "blk.%u.ffn_gate_inp.weight", il);
    optf(out, "blk.%u.exp_probs_b.bias", il);
    reqf(out, "blk.%u.ffn_gate_exps.weight", il);
    reqf(out, "blk.%u.ffn_up_exps.weight", il);
    reqf(out, "blk.%u.ffn_down_exps.weight", il);
    reqf(out, "blk.%u.ffn_gate_shexp.weight", il);
    reqf(out, "blk.%u.ffn_up_shexp.weight", il);
    reqf(out, "blk.%u.ffn_down_shexp.weight", il);
    if il < shape.n_hash_layer {
        reqf(out, "blk.%u.ffn_gate_tid2eid.weight", il);
    }
}

pub const DSPARK_N_LAYER: u32 = 3;
pub const DSPARK_MARKOV_RANK: u32 = 256;

/// Names `mtp_weights_bind` looks up on a DeepSeek MTP sibling GGUF.
pub fn bind_mtp_names() -> Vec<BindName> {
    let mut out = Vec::new();
    req(&mut out, "mtp.0.hc_head_base.weight");
    req(&mut out, "mtp.0.hc_head_fn.weight");
    req(&mut out, "mtp.0.hc_head_scale.weight");
    req(&mut out, "mtp.0.e_proj.weight");
    req(&mut out, "mtp.0.h_proj.weight");
    req(&mut out, "mtp.0.enorm.weight");
    req(&mut out, "mtp.0.hnorm.weight");
    req(&mut out, "mtp.0.norm.weight");
    req(&mut out, "mtp.0.hc_attn_fn.weight");
    req(&mut out, "mtp.0.hc_attn_scale.weight");
    req(&mut out, "mtp.0.hc_attn_base.weight");
    req(&mut out, "mtp.0.attn_norm.weight");
    req(&mut out, "mtp.0.attn_q_a.weight");
    req(&mut out, "mtp.0.attn_q_a_norm.weight");
    req(&mut out, "mtp.0.attn_q_b.weight");
    req(&mut out, "mtp.0.attn_kv.weight");
    req(&mut out, "mtp.0.attn_kv_a_norm.weight");
    req(&mut out, "mtp.0.attn_sinks.weight");
    req(&mut out, "mtp.0.attn_output_a.weight");
    req(&mut out, "mtp.0.attn_output_b.weight");
    req(&mut out, "mtp.0.hc_ffn_fn.weight");
    req(&mut out, "mtp.0.hc_ffn_scale.weight");
    req(&mut out, "mtp.0.hc_ffn_base.weight");
    req(&mut out, "mtp.0.ffn_norm.weight");
    req(&mut out, "mtp.0.ffn_gate_inp.weight");
    req(&mut out, "mtp.0.exp_probs_b.bias");
    req(&mut out, "mtp.0.ffn_gate_exps.weight");
    req(&mut out, "mtp.0.ffn_up_exps.weight");
    req(&mut out, "mtp.0.ffn_down_exps.weight");
    req(&mut out, "mtp.0.ffn_gate_shexp.weight");
    req(&mut out, "mtp.0.ffn_up_shexp.weight");
    req(&mut out, "mtp.0.ffn_down_shexp.weight");
    out
}

/// Names `dspark_weights_bind` looks up on a DeepSeek DSpark sibling GGUF.
pub fn bind_dspark_names() -> Vec<BindName> {
    let mut out = Vec::new();
    req(&mut out, "dspark.main_proj.weight");
    req(&mut out, "dspark.main_norm.weight");
    req(&mut out, "dspark.markov_w1.weight");
    req(&mut out, "dspark.markov_w2.weight");
    req(&mut out, "dspark.conf_proj.weight");
    req(&mut out, "dspark.hc_head_fn.weight");
    req(&mut out, "dspark.hc_head_base.weight");
    req(&mut out, "dspark.hc_head_scale.weight");
    req(&mut out, "dspark.norm.weight");
    for il in 0..DSPARK_N_LAYER {
        reqf(&mut out, "dspark.%u.hc_attn_fn.weight", il);
        reqf(&mut out, "dspark.%u.hc_attn_scale.weight", il);
        reqf(&mut out, "dspark.%u.hc_attn_base.weight", il);
        reqf(&mut out, "dspark.%u.attn_norm.weight", il);
        reqf(&mut out, "dspark.%u.attn_q_a.weight", il);
        reqf(&mut out, "dspark.%u.attn_q_a_norm.weight", il);
        reqf(&mut out, "dspark.%u.attn_q_b.weight", il);
        reqf(&mut out, "dspark.%u.attn_kv.weight", il);
        reqf(&mut out, "dspark.%u.attn_kv_a_norm.weight", il);
        reqf(&mut out, "dspark.%u.attn_sinks.weight", il);
        reqf(&mut out, "dspark.%u.attn_output_a.weight", il);
        reqf(&mut out, "dspark.%u.attn_output_b.weight", il);
        reqf(&mut out, "dspark.%u.hc_ffn_fn.weight", il);
        reqf(&mut out, "dspark.%u.hc_ffn_scale.weight", il);
        reqf(&mut out, "dspark.%u.hc_ffn_base.weight", il);
        reqf(&mut out, "dspark.%u.ffn_norm.weight", il);
        reqf(&mut out, "dspark.%u.ffn_gate_inp.weight", il);
        reqf(&mut out, "dspark.%u.exp_probs_b.bias", il);
        reqf(&mut out, "dspark.%u.ffn_gate_exps.weight", il);
        reqf(&mut out, "dspark.%u.ffn_up_exps.weight", il);
        reqf(&mut out, "dspark.%u.ffn_down_exps.weight", il);
        reqf(&mut out, "dspark.%u.ffn_gate_shexp.weight", il);
        reqf(&mut out, "dspark.%u.ffn_up_shexp.weight", il);
        reqf(&mut out, "dspark.%u.ffn_down_shexp.weight", il);
    }
    out
}

fn dump_name_table(header: String, names: &[BindName]) -> String {
    let mut req_n = 0u32;
    let mut opt_n = 0u32;
    let mut out = header;
    for n in names {
        if n.need.required() {
            req_n += 1;
        } else {
            opt_n += 1;
        }
        out.push_str(&format!("NAME {} {}\n", n.name, n.need.token()));
    }
    out.push_str(&format!(
        "COUNT n={} req={} opt={}\n",
        names.len(),
        req_n,
        opt_n
    ));
    out
}

/// Names `weights_bind` looks up for the main GGUF (not MTP/DSpark siblings).
pub fn bind_names(shape: &Shape) -> Vec<BindName> {
    let mut out = Vec::new();
    match shape.family {
        ModelFamily::Glm53 => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_glm53_layer(&mut out, shape, il);
            }
        }
        ModelFamily::Qwen4Exp => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output.weight");
            req(&mut out, "hc_input.norm.weight");
            req(&mut out, "hc_input.mix_down.weight");
            req(&mut out, "hc_input.mix_up.weight");
            for il in 0..shape.n_layer {
                bind_qwen4exp_layer(&mut out, shape, il);
            }
            bind_qwen4exp_mtp(&mut out);
            bind_qwen4exp_vision(&mut out);
        }
        ModelFamily::Motif3 => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_motif3_layer(&mut out, shape, il);
            }
            bind_motif3_mtp(&mut out);
        }
        ModelFamily::Dots3Note => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_dots3_note_layer(&mut out, shape, il);
            }
            req(&mut out, "token_embd_mtp.weight");
        }
        ModelFamily::SolarOpen2 => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_solar_open2_layer(&mut out, shape, il);
            }
        }
        ModelFamily::ExaoneMoe => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_exaone_moe_layer(&mut out, shape, il);
            }
        }
        ModelFamily::DeepSeek4 => {
            req(&mut out, "token_embd.weight");
            req(&mut out, "output_hc_base.weight");
            req(&mut out, "output_hc_fn.weight");
            req(&mut out, "output_hc_scale.weight");
            req(&mut out, "output_norm.weight");
            req(&mut out, "output.weight");
            for il in 0..shape.n_layer {
                bind_deepseek_layer(&mut out, shape, il);
            }
        }
    }
    out
}

pub fn dump_bind_names_shape(shape: &Shape) -> String {
    dump_name_table(
        format!(
            "BIND name={} family={} variant={} n_layer={}\n",
            shape.name, shape.family as u32, shape.variant as u32, shape.n_layer
        ),
        &bind_names(shape),
    )
}

pub fn dump_bind_mtp_shape(shape: &Shape) -> String {
    dump_name_table(
        format!(
            "BIND kind=mtp name={} family={} variant={}\n",
            shape.name, shape.family as u32, shape.variant as u32
        ),
        &bind_mtp_names(),
    )
}

pub fn dump_bind_dspark_shape(shape: &Shape) -> String {
    dump_name_table(
        format!(
            "BIND kind=dspark name={} family={} variant={} n_layer={}\n",
            shape.name, shape.family as u32, shape.variant as u32, DSPARK_N_LAYER
        ),
        &bind_dspark_names(),
    )
}

pub fn dump_bind_support() -> String {
    let mut out = String::new();
    for v in [Variant::Flash, Variant::Pro] {
        let shape = shape_for_variant(v);
        out.push_str(&dump_bind_mtp_shape(&shape));
        out.push_str(&dump_bind_dspark_shape(&shape));
    }
    out
}

pub fn dump_bind_names() -> String {
    let mut out = String::new();
    for v in [
        Variant::Flash,
        Variant::Pro,
        Variant::SolarOpen2_250B,
        Variant::Motif3,
        Variant::Kexaone236B,
        Variant::Dots3NotePrev,
        Variant::Qwen38FlashNext,
        Variant::Glm53Flash,
    ] {
        out.push_str(&dump_bind_names_shape(&shape_for_variant(v)));
    }
    out
}

impl BindPlan {
    pub fn resolve_names(shape: Shape, names: Vec<BindName>, inventory: &TensorInventory) -> Self {
        let slots = names
            .into_iter()
            .map(|n| {
                let index = inventory.find_index(&n.name).map(|i| i as u32);
                BindSlot {
                    tensor: index.and_then(|i| inventory.tensors.get(i as usize).cloned()),
                    name: n.name,
                    need: n.need,
                    index,
                }
            })
            .collect();
        Self {
            shape,
            slots,
            n_shards: inventory.shards.len() as u32,
            data_pos: inventory.data_pos,
            alignment: inventory.alignment,
            page: inventory.page,
        }
    }

    pub fn resolve(shape: Shape, inventory: &TensorInventory) -> Self {
        Self::resolve_names(shape, bind_names(&shape), inventory)
    }

    pub fn resolve_mtp(shape: Shape, inventory: &TensorInventory) -> Self {
        Self::resolve_names(shape, bind_mtp_names(), inventory)
    }

    pub fn resolve_dspark(shape: Shape, inventory: &TensorInventory) -> Self {
        Self::resolve_names(shape, bind_dspark_names(), inventory)
    }

    pub fn resolve_catalog(
        support: Option<SupportCatalog>,
        shape: Shape,
        inventory: &TensorInventory,
    ) -> Self {
        match support {
            None => Self::resolve(shape, inventory),
            Some(SupportCatalog::Mtp) => Self::resolve_mtp(shape, inventory),
            Some(SupportCatalog::Dspark) => Self::resolve_dspark(shape, inventory),
        }
    }

    /// C `ds4_engine_routed_quant_bits`: first base `ffn_gate_exps` wins.
    /// Q4_K → 4, any other present type → 2, none → 0.
    pub fn routed_quant_bits(&self) -> i32 {
        const T_Q4_K: u32 = 12;
        for slot in &self.slots {
            if slot.name.starts_with("mtp.") || slot.name.starts_with("dspark.") {
                continue;
            }
            if !slot.name.contains("ffn_gate_exps") {
                continue;
            }
            if let Some(tensor) = &slot.tensor {
                return if tensor.typ == T_Q4_K { 4 } else { 2 };
            }
        }
        0
    }

    pub fn missing_required(&self) -> Vec<&str> {
        self.slots
            .iter()
            .filter(|s| s.need.required() && s.tensor.is_none())
            .map(|s| s.name.as_str())
            .collect()
    }

    pub fn check(&self) -> Result<(), BindError> {
        for s in &self.slots {
            if s.name.is_empty() {
                return Err(BindError::NameEmpty);
            }
            if s.need.required() && s.tensor.is_none() {
                return Err(BindError::Missing(s.name.clone()));
            }
            if let Some(t) = &s.tensor {
                if t.ndim == 0 || t.ndim > MAX_DIMS {
                    return Err(BindError::BadNdim);
                }
            }
        }
        Ok(())
    }

    pub fn dump(&self) -> String {
        let mut found = 0u32;
        let mut missing_req = 0u32;
        let mut out = format!(
            "BIND name={} family={} variant={} n_layer={}\n",
            self.shape.name,
            self.shape.family as u32,
            self.shape.variant as u32,
            self.shape.n_layer
        );
        for s in &self.slots {
            match &s.tensor {
                Some(t) => {
                    found += 1;
                    let mut dims = String::new();
                    for i in 0..t.ndim as usize {
                        if i > 0 {
                            dims.push(',');
                        }
                        dims.push_str(&t.dim[i].to_string());
                    }
                    out.push_str(&format!(
                        "SLOT {} {} FOUND type={}({}) ndim={} dims={} bytes={} abs={} shard={}\n",
                        s.name,
                        s.need.token(),
                        t.typ,
                        tensor_type_name(t.typ),
                        t.ndim,
                        dims,
                        t.bytes,
                        t.abs_offset,
                        t.shard
                    ));
                }
                None => {
                    if s.need.required() {
                        missing_req += 1;
                    }
                    out.push_str(&format!("SLOT {} {} MISS\n", s.name, s.need.token()));
                }
            }
        }
        out.push_str(&format!(
            "COUNT n={} found={} missing_req={}\n",
            self.slots.len(),
            found,
            missing_req
        ));
        out
    }
}

pub fn match_plans(host: &BindPlan, native: &BindPlan) -> Result<(), BindError> {
    if host.slots.len() != native.slots.len() {
        return Err(BindError::CountMismatch);
    }
    for (h, n) in host.slots.iter().zip(native.slots.iter()) {
        if h.name != n.name {
            return Err(BindError::NameMismatch);
        }
        if h.need != n.need {
            return Err(BindError::NeedMismatch);
        }
        match (&h.tensor, &n.tensor) {
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => return Err(BindError::FoundMismatch),
            (Some(a), Some(b)) => {
                if a.typ != b.typ {
                    return Err(BindError::TypeMismatch);
                }
                if a.ndim != b.ndim || a.dim != b.dim {
                    return Err(BindError::DimMismatch);
                }
                if a.rel_offset != b.rel_offset || a.abs_offset != b.abs_offset {
                    return Err(BindError::OffsetMismatch);
                }
                if a.bytes != b.bytes {
                    return Err(BindError::BytesMismatch);
                }
                if a.shard != b.shard {
                    return Err(BindError::ShardMismatch);
                }
            }
        }
    }
    if host.n_shards != native.n_shards
        || host.data_pos != native.data_pos
        || host.alignment != native.alignment
        || host.page != native.page
    {
        return Err(BindError::DataMismatch);
    }
    Ok(())
}

fn slot(
    name: &'static str,
    required: bool,
    found: bool,
    typ: u32,
    ndim: u32,
    dim: [u64; 8],
    rel: u64,
    abs: u64,
    bytes: u64,
    shard: u32,
) -> BindSlot {
    BindSlot {
        name: name.into(),
        need: if required {
            BindNeed::Required
        } else {
            BindNeed::Optional
        },
        tensor: if found {
            Some(TensorInfo {
                name: name.into(),
                ndim,
                dim,
                typ,
                rel_offset: rel,
                abs_offset: abs,
                elements: 0,
                bytes,
                shard,
            })
        } else {
            None
        },
        index: None,
    }
}

fn check_line(plan: &BindPlan) -> String {
    match plan.check() {
        Ok(()) => "CHECK OK\n".into(),
        Err(e) => format!("CHECK {}\n", e.token()),
    }
}

/// C `bind_c_oracle check` tape.
pub fn dump_bind_check_oracle() -> String {
    let base = slot(
        "token_embd.weight",
        true,
        true,
        8,
        2,
        [4, 8, 0, 0, 0, 0, 0, 0],
        0,
        32,
        272,
        0,
    );
    let mut out = String::new();
    out.push_str(&format!("CHECK {}\n", BindError::PlanNull.token()));
    out.push_str(&format!("CHECK {}\n", BindError::SlotsNull.token()));
    out.push_str(&format!("CHECK {}\n", BindError::ShardsNull.token()));
    let mut ok = BindPlan {
        shape: shape_for_variant(Variant::Flash),
        slots: vec![base.clone()],
        n_shards: 1,
        data_pos: 32,
        alignment: 32,
        page: 4096,
    };
    out.push_str(&check_line(&ok));
    ok.slots[0].tensor = None;
    out.push_str(&check_line(&ok));
    ok.slots[0].name.clear();
    ok.slots[0].need = BindNeed::Required;
    out.push_str(&check_line(&ok));
    ok.slots[0] = slot("token_embd.weight", false, false, 8, 0, [0; 8], 0, 0, 0, 0);
    out.push_str(&check_line(&ok));
    ok.slots[0] = slot("token_embd.weight", true, true, 8, 0, [0; 8], 0, 32, 272, 0);
    out.push_str(&check_line(&ok));
    out
}

fn match_line(host: &BindPlan, native: &BindPlan) -> String {
    match match_plans(host, native) {
        Ok(()) => "MATCH OK\n".into(),
        Err(e) => format!("MATCH {}\n", e.token()),
    }
}

/// C `bind_c_oracle match` tape.
pub fn dump_bind_match_oracle() -> String {
    let slot0 = slot(
        "token_embd.weight",
        true,
        true,
        8,
        2,
        [4, 8, 0, 0, 0, 0, 0, 0],
        0,
        32,
        272,
        0,
    );
    let host = BindPlan {
        shape: shape_for_variant(Variant::Flash),
        slots: vec![slot0.clone()],
        n_shards: 1,
        data_pos: 32,
        alignment: 32,
        page: 4096,
    };
    let mut native = host.clone();
    let mut out = String::new();
    out.push_str(&match_line(&host, &native));
    native.slots.push(slot0.clone());
    out.push_str(&match_line(&host, &native));
    native = host.clone();
    native.slots[0].name = "other.weight".into();
    out.push_str(&match_line(&host, &native));
    native = host.clone();
    if let Some(t) = native.slots[0].tensor.as_mut() {
        t.typ = 0;
    }
    out.push_str(&match_line(&host, &native));
    native = host.clone();
    if let Some(t) = native.slots[0].tensor.as_mut() {
        t.bytes = 1;
    }
    out.push_str(&match_line(&host, &native));
    native = host.clone();
    native.data_pos = 99;
    out.push_str(&match_line(&host, &native));
    out
}

pub fn variant_from_bind_name(s: &str) -> Option<Variant> {
    match s {
        "flash" => Some(Variant::Flash),
        "pro" => Some(Variant::Pro),
        "solar-open2" => Some(Variant::SolarOpen2_250B),
        "motif3" => Some(Variant::Motif3),
        "exaone-moe" => Some(Variant::Kexaone236B),
        "dots3-note" => Some(Variant::Dots3NotePrev),
        "qwen4exp" => Some(Variant::Qwen38FlashNext),
        "glm5-next" => Some(Variant::Glm53Flash),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportCatalog {
    Mtp,
    Dspark,
}

/// `flash` / `mtp-flash` / `dspark-pro`. Support catalogs are DeepSeek-only.
pub fn dump_bind_names_variant(name: &str) -> Option<String> {
    let (support, v) = catalog_from_bind_name(name)?;
    let shape = shape_for_variant(v);
    Some(match support {
        None => dump_bind_names_shape(&shape),
        Some(SupportCatalog::Mtp) => dump_bind_mtp_shape(&shape),
        Some(SupportCatalog::Dspark) => dump_bind_dspark_shape(&shape),
    })
}

/// `flash` / `mtp-flash` / `dspark-pro`. Support catalogs are DeepSeek-only.
pub fn catalog_from_bind_name(s: &str) -> Option<(Option<SupportCatalog>, Variant)> {
    if let Some(rest) = s.strip_prefix("mtp-") {
        return variant_from_bind_name(rest)
            .filter(|v| matches!(v, Variant::Flash | Variant::Pro))
            .map(|v| (Some(SupportCatalog::Mtp), v));
    }
    if let Some(rest) = s.strip_prefix("dspark-") {
        return variant_from_bind_name(rest)
            .filter(|v| matches!(v, Variant::Flash | Variant::Pro))
            .map(|v| (Some(SupportCatalog::Dspark), v));
    }
    variant_from_bind_name(s).map(|v| (None, v))
}

pub const HOST_BIND_MISS: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct HostBindLook {
    pub name: Option<String>,
    pub required: bool,
    pub found: bool,
    pub index: u32,
}

/// C `ds4_host_bind_lookup`. `map` is `None` for a NULL map; `Some((n, None))`
/// is `n > 0 && v == NULL`.
pub fn host_bind_lookup(
    map: Option<(u32, Option<&[HostBindLook]>)>,
    name: Option<&str>,
    n_tensors: u32,
) -> (i32, String, u32) {
    let index_out = HOST_BIND_MISS;
    let Some((n, looks)) = map else {
        return (1, "map-null".into(), index_out);
    };
    if name.map(|s| s.is_empty()).unwrap_or(true) {
        return (1, "name-empty".into(), index_out);
    }
    let name = name.unwrap();
    if n > 0 && looks.is_none() {
        return (1, "looks-null".into(), index_out);
    }
    let looks = looks.unwrap_or(&[]);
    for e in looks.iter().take(n as usize) {
        match e.name.as_deref() {
            None | Some("") => return (1, "name-empty".into(), index_out),
            Some(en) if en != name => continue,
            Some(_) => {
                if !e.found {
                    if e.required {
                        return (1, format!("missing {name}"), index_out);
                    }
                    return (0, String::new(), index_out);
                }
                if e.index == HOST_BIND_MISS || e.index >= n_tensors {
                    return (1, "index-range".into(), index_out);
                }
                return (0, String::new(), e.index);
            }
        }
    }
    (2, "unknown".into(), index_out)
}

fn lookup_line(label: &str, rc: i32, err: &str, idx: u32) -> String {
    if rc == 0 {
        if idx == HOST_BIND_MISS {
            format!("{label} miss\n")
        } else {
            format!("{label} {idx}\n")
        }
    } else if rc == 2 {
        format!("{label} unknown\n")
    } else {
        format!("{label} {err}\n")
    }
}

/// Fixed C↔Rust lookup tapes (same cases as `bind_lookup_c_oracle`).
pub fn dump_bind_lookup_tapes() -> String {
    let ok = [
        HostBindLook {
            name: Some("token_embd.weight".into()),
            required: true,
            found: true,
            index: 3,
        },
        HostBindLook {
            name: Some("exp_probs_b.bias".into()),
            required: false,
            found: false,
            index: HOST_BIND_MISS,
        },
        HostBindLook {
            name: Some("output.weight".into()),
            required: true,
            found: false,
            index: HOST_BIND_MISS,
        },
    ];
    let mut out = String::new();
    let (rc, err, idx) = host_bind_lookup(None, Some("token_embd.weight"), 8);
    out.push_str(&lookup_line("map-null", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&ok))), None, 8);
    out.push_str(&lookup_line("name-empty", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, None)), Some("token_embd.weight"), 8);
    out.push_str(&lookup_line("looks-null", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&ok))), Some("not.in.plan"), 8);
    out.push_str(&lookup_line("unknown", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&ok))), Some("output.weight"), 8);
    out.push_str(&lookup_line("missing", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&ok))), Some("exp_probs_b.bias"), 8);
    out.push_str(&lookup_line("miss", rc, &err, idx));
    let mut range = ok.clone();
    range[0].index = 8;
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&range))), Some("token_embd.weight"), 8);
    out.push_str(&lookup_line("index-range", rc, &err, idx));
    let mut miss = ok.clone();
    miss[0].index = HOST_BIND_MISS;
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&miss))), Some("token_embd.weight"), 8);
    out.push_str(&lookup_line("index-miss", rc, &err, idx));
    let mut empty = ok.clone();
    empty[0].name = None;
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&empty))), Some("output.weight"), 8);
    out.push_str(&lookup_line("slot-empty", rc, &err, idx));
    let (rc, err, idx) = host_bind_lookup(Some((3, Some(&ok))), Some("token_embd.weight"), 8);
    out.push_str(&lookup_line("ok", rc, &err, idx));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::SHAPE_FLASH;
    use crate::tensors::TensorInfo;

    fn gate(name: &str, typ: u32) -> BindSlot {
        BindSlot {
            name: name.into(),
            need: BindNeed::Required,
            tensor: Some(TensorInfo {
                name: name.into(),
                ndim: 2,
                dim: [1, 1, 0, 0, 0, 0, 0, 0],
                typ,
                rel_offset: 0,
                abs_offset: 0,
                elements: 1,
                bytes: 1,
                shard: 0,
            }),
            index: Some(0),
        }
    }

    fn plan(slots: Vec<BindSlot>) -> BindPlan {
        BindPlan {
            shape: SHAPE_FLASH,
            slots,
            n_shards: 1,
            data_pos: 0,
            alignment: 32,
            page: 4096,
        }
    }

    #[test]
    fn routed_quant_bits_matches_c_gate_type() {
        assert_eq!(plan(vec![]).routed_quant_bits(), 0);
        assert_eq!(
            plan(vec![gate("blk.0.ffn_gate_exps.weight", 12)]).routed_quant_bits(),
            4
        );
        assert_eq!(
            plan(vec![gate("blk.0.ffn_gate_exps.weight", 10)]).routed_quant_bits(),
            2
        );
        assert_eq!(
            plan(vec![
                gate("mtp.0.ffn_gate_exps.weight", 12),
                gate("blk.2.ffn_gate_exps.weight", 16),
            ])
            .routed_quant_bits(),
            2
        );
    }
}
