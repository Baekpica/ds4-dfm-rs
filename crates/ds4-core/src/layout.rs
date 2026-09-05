//! Host-owned `weights_validate_layout` expected table.
//!
//! Copied from `ds4.c` (`tensor_expect_*` + family layout validators).
//! When the host bind map is installed, native skips the main-model
//! `weights_validate_layout` because `Model::open` already ran this
//! check. Host also owns the DeepSeek MTP/DSpark sibling tables
//! (`--layout mtp-flash` / `dspark-pro`). Sibling pointer assignment
//! and sibling validate stay C until a separate sibling bind map exists.

use crate::bind::{
    dots3_layer_is_full_attention, expected_compress_ratio, glm53_layer_is_kda,
    qwen4exp_layer_is_full_attention, solar_layer_is_gqa, BindPlan, DSPARK_MARKOV_RANK,
    DSPARK_N_LAYER,
};
use crate::shape::{shape_for_variant, ModelFamily, Shape, Variant};
use crate::tensors::{tensor_type_name, TensorInfo};

const T_F32: u32 = 0;
const T_F16: u32 = 1;
const T_Q4_0: u32 = 2;
const T_Q5_0: u32 = 6;
const T_Q8_0: u32 = 8;
const T_Q2_K: u32 = 10;
const T_Q3_K: u32 = 11;
const T_Q4_K: u32 = 12;
const T_Q5_K: u32 = 13;
const T_Q6_K: u32 = 14;
const T_IQ2_XXS: u32 = 16;
const T_I32: u32 = 26;
const T_I64: u32 = 27;
const T_BF16: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeClass {
    Exact(u32),
    OptionalExact(u32),
    Plain,
    MotifProj,
    Routed,
    ExaoneQuant,
    SolarGateUp,
    SolarDown,
    SolarConv,
    SolarDecay,
    QwenMatrix,
    QwenPlain,
    QwenMtpRouted,
    GlmDense,
}

impl TypeClass {
    pub fn token(self) -> String {
        match self {
            TypeClass::Exact(t) => format!("exact:{}", tensor_type_name(t)),
            TypeClass::OptionalExact(t) => format!("optional:{}", tensor_type_name(t)),
            TypeClass::Plain => "plain".into(),
            TypeClass::MotifProj => "motif-proj".into(),
            TypeClass::Routed => "routed".into(),
            TypeClass::ExaoneQuant => "exaone-quant".into(),
            TypeClass::SolarGateUp => "solar-gateup".into(),
            TypeClass::SolarDown => "solar-down".into(),
            TypeClass::SolarConv => "solar-conv".into(),
            TypeClass::SolarDecay => "solar-decay".into(),
            TypeClass::QwenMatrix => "qwen-matrix".into(),
            TypeClass::QwenPlain => "qwen-plain".into(),
            TypeClass::QwenMtpRouted => "qwen-mtp-routed".into(),
            TypeClass::GlmDense => "glm-dense".into(),
        }
    }

    fn optional(self) -> bool {
        matches!(self, TypeClass::OptionalExact(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutSpec {
    pub name: String,
    pub class: TypeClass,
    pub ndim: u32,
    pub dim: [u64; 8],
}

#[derive(Debug)]
pub enum LayoutError {
    Missing(String),
    Type(String),
    Ndim(String),
    Dim(String),
    GateUp(u32),
}

impl LayoutError {
    pub fn token(&self) -> String {
        match self {
            LayoutError::Missing(n) => format!("missing {n}"),
            LayoutError::Type(n) => format!("type {n}"),
            LayoutError::Ndim(n) => format!("ndim {n}"),
            LayoutError::Dim(n) => format!("dim {n}"),
            LayoutError::GateUp(il) => format!("gate-up {il}"),
        }
    }
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for LayoutError {}

fn spec(
    out: &mut Vec<LayoutSpec>,
    name: impl Into<String>,
    class: TypeClass,
    ndim: u32,
    dim: [u64; 4],
) {
    let mut dims = [0; 8];
    dims[..4].copy_from_slice(&dim);
    out.push(LayoutSpec {
        name: name.into(),
        class,
        ndim,
        dim: dims,
    });
}

fn spec5(out: &mut Vec<LayoutSpec>, name: impl Into<String>, class: TypeClass, dim: [u64; 5]) {
    let mut dims = [0; 8];
    dims[..5].copy_from_slice(&dim);
    out.push(LayoutSpec {
        name: name.into(),
        class,
        ndim: 5,
        dim: dims,
    });
}

fn specf(
    out: &mut Vec<LayoutSpec>,
    fmt: &str,
    il: u32,
    class: TypeClass,
    ndim: u32,
    dim: [u64; 4],
) {
    spec(out, fmt.replace("%u", &il.to_string()), class, ndim, dim);
}

fn is_nextn(shape: &Shape, il: u32) -> bool {
    shape.n_nextn_predict != 0 && il + shape.n_nextn_predict >= shape.n_layer
}

fn type_ok(class: TypeClass, typ: u32) -> bool {
    match class {
        TypeClass::Exact(t) | TypeClass::OptionalExact(t) => typ == t,
        TypeClass::Plain => typ == T_F16 || typ == T_F32,
        TypeClass::MotifProj => typ == T_F16 || typ == T_BF16,
        TypeClass::Routed => typ == T_IQ2_XXS || typ == T_Q2_K || typ == T_Q4_K,
        TypeClass::ExaoneQuant => {
            typ == T_Q8_0
                || typ == T_Q6_K
                || typ == T_Q5_K
                || typ == T_Q4_K
                || typ == T_Q3_K
                || typ == T_Q2_K
                || typ == T_IQ2_XXS
                || typ == T_F16
                || typ == T_F32
        }
        TypeClass::SolarGateUp => typ == T_Q4_K || typ == T_Q2_K || typ == T_IQ2_XXS,
        TypeClass::SolarDown => typ == T_Q4_K || typ == T_Q3_K || typ == T_Q2_K,
        TypeClass::SolarConv | TypeClass::SolarDecay => typ == T_F32,
        TypeClass::QwenMatrix => matches!(
            typ,
            T_BF16
                | T_Q8_0
                | T_Q6_K
                | T_Q5_K
                | T_Q5_0
                | T_Q4_K
                | T_Q4_0
                | T_Q3_K
                | T_Q2_K
                | T_IQ2_XXS
        ),
        TypeClass::QwenPlain => typ == T_F32 || typ == T_BF16,
        TypeClass::QwenMtpRouted => typ == T_Q8_0 || typ == T_BF16,
        TypeClass::GlmDense => typ == T_BF16 || typ == T_Q8_0 || typ == T_Q4_K || typ == T_Q4_0,
    }
}

fn dims_match(spec: &LayoutSpec, t: &TensorInfo) -> Result<(), LayoutError> {
    match spec.class {
        TypeClass::SolarConv => {
            let d_inner = spec.dim[2];
            if t.ndim == 4
                && t.dim[0] == spec.dim[0]
                && t.dim[1] == 1
                && t.dim[2] == d_inner
                && t.dim[3] == 1
            {
                return Ok(());
            }
            if t.ndim == 3 && t.dim[0] == spec.dim[0] && t.dim[1] == 1 && t.dim[2] == d_inner {
                return Ok(());
            }
            Err(LayoutError::Dim(spec.name.clone()))
        }
        TypeClass::SolarDecay => {
            if t.ndim == 4
                && t.dim[0] == 1
                && t.dim[1] == spec.dim[1]
                && t.dim[2] == 1
                && t.dim[3] == 1
            {
                return Ok(());
            }
            if t.ndim == 2 && t.dim[0] == 1 && t.dim[1] == spec.dim[1] {
                return Ok(());
            }
            Err(LayoutError::Dim(spec.name.clone()))
        }
        _ => {
            if t.ndim != spec.ndim {
                return Err(LayoutError::Ndim(spec.name.clone()));
            }
            for i in 0..spec.ndim as usize {
                if t.dim[i] != spec.dim[i] {
                    return Err(LayoutError::Dim(spec.name.clone()));
                }
            }
            Ok(())
        }
    }
}

pub fn expect_tensor(spec: &LayoutSpec, t: Option<&TensorInfo>) -> Result<(), LayoutError> {
    let Some(t) = t else {
        if spec.class.optional() {
            return Ok(());
        }
        return Err(LayoutError::Missing(spec.name.clone()));
    };
    if !type_ok(spec.class, t.typ) {
        return Err(LayoutError::Type(spec.name.clone()));
    }
    dims_match(spec, t)
}

fn motif_mhc(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape) {
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    spec(
        out,
        format!("{prefix}.rms_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [hc_dim, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.proj_pre.weight"),
        TypeClass::MotifProj,
        2,
        [hc_dim, hc, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.proj_post.weight"),
        TypeClass::MotifProj,
        2,
        [hc_dim, hc, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.proj_res.weight"),
        TypeClass::MotifProj,
        2,
        [hc_dim, hc * hc, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.alpha_pre"),
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.alpha_post"),
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.alpha_res"),
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.bias_pre"),
        TypeClass::Exact(T_F32),
        1,
        [hc, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.bias_post"),
        TypeClass::Exact(T_F32),
        1,
        [hc, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.bias_res"),
        TypeClass::Exact(T_F32),
        2,
        [hc, hc, 0, 0],
    );
}

fn motif_attention(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape, include_norm: bool) {
    let e = shape.n_embd as u64;
    let q_dim = shape.n_head as u64 * shape.n_head_dim as u64;
    let signal_heads = (shape.n_head - shape.n_noise_head) as u64;
    let signal_value_dim = signal_heads * shape.n_value_dim as u64;
    let kv_b_dim = shape.n_head_kv as u64
        * ((shape.n_head_dim - shape.n_rot) as u64 + shape.n_value_dim as u64);
    if include_norm {
        spec(
            out,
            format!("{prefix}attn_norm.weight"),
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
    }
    spec(
        out,
        format!("{prefix}attn_q_a.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_lora_q as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_a_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_lora_q as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_b.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [shape.n_lora_q as u64, q_dim, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_gate.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [shape.n_lora_q as u64, signal_value_dim, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_kv_a.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_kv_lora as u64 + shape.n_rot as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_kv_a_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_kv_lora as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_kv_b.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [shape.n_kv_lora as u64, kv_b_dim, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_lambda.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, signal_heads, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_output.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [signal_value_dim, e, 0, 0],
    );
}

fn motif_dense_ffn(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape) {
    let e = shape.n_embd as u64;
    let ff = shape.n_ff_dense as u64;
    spec(
        out,
        format!("{prefix}ffn_gate.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, ff, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_up.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, ff, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_down.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [ff, e, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_polynorm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [3, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_polynorm.bias"),
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
}

fn expected_motif3(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    for il in 0..shape.n_layer {
        let p = format!("blk.{il}.");
        motif_mhc(&mut out, &format!("blk.{il}.mhc_attn"), shape);
        motif_attention(&mut out, &p, shape, true);
        motif_mhc(&mut out, &format!("blk.{il}.mhc_ffn"), shape);
        spec(
            &mut out,
            format!("blk.{il}.ffn_norm.weight"),
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        if il < shape.n_leading_dense {
            motif_dense_ffn(&mut out, &p, shape);
        } else {
            spec(
                &mut out,
                format!("blk.{il}.ffn_gate_inp.weight"),
                TypeClass::Exact(T_F32),
                2,
                [e, shape.n_expert as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.exp_probs_b.bias"),
                TypeClass::Exact(T_F32),
                1,
                [shape.n_expert as u64, 0, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_gate_exps.weight"),
                TypeClass::Exact(T_IQ2_XXS),
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_up_exps.weight"),
                TypeClass::Exact(T_IQ2_XXS),
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_down_exps.weight"),
                TypeClass::Exact(T_Q2_K),
                3,
                [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_polynorm_exps.weight"),
                TypeClass::Exact(T_F32),
                2,
                [3, shape.n_expert as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_polynorm_exps.bias"),
                TypeClass::Exact(T_F32),
                2,
                [1, shape.n_expert as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_gate_shexp.weight"),
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_exp as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_up_shexp.weight"),
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_exp as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_down_shexp.weight"),
                TypeClass::Exact(T_Q8_0),
                2,
                [shape.n_ff_exp as u64, e, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_polynorm_shexp.weight"),
                TypeClass::Exact(T_F32),
                1,
                [3, 0, 0, 0],
            );
            spec(
                &mut out,
                format!("blk.{il}.ffn_polynorm_shexp.bias"),
                TypeClass::Exact(T_F32),
                1,
                [1, 0, 0, 0],
            );
        }
    }
    spec(
        &mut out,
        "mtp.0.embed_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.input_layernorm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.input_proj.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [2 * e, e, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.final_layernorm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    motif_attention(&mut out, "mtp.0.", shape, false);
    spec(
        &mut out,
        "mtp.0.post_attention_layernorm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    motif_dense_ffn(&mut out, "mtp.0.", shape);
    out
}

fn expected_dots3(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    for il in 0..shape.n_layer {
        let full = dots3_layer_is_full_attention(shape, il);
        let heads = if full { shape.n_head } else { shape.n_swa_head } as u64;
        let kv_lora = if full {
            shape.n_kv_lora
        } else {
            shape.n_swa_kv_lora
        } as u64;
        let qk_dim = if full {
            shape.n_key_mla
        } else {
            shape.n_swa_key_mla
        } as u64;
        let nope = qk_dim - shape.n_rot as u64;
        let v_dim = shape.n_value_mla as u64;
        specf(
            &mut out,
            "blk.%u.attn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_a.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_lora_q as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_a_norm.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            1,
            [shape.n_lora_q as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_b.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [shape.n_lora_q as u64, heads * qk_dim, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_kv_a_mqa.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, kv_lora + shape.n_rot as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_kv_a_norm.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            1,
            [kv_lora, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_kv_b.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [kv_lora, heads * (nope + v_dim), 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_k_rope_norm.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            1,
            [shape.n_rot as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_gate.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, heads, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_output.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [heads * v_dim, e, 0, 0],
        );
        if full {
            specf(
                &mut out,
                "blk.%u.attn_idx_q_b.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [
                    shape.n_lora_q as u64,
                    shape.n_indexer_head as u64 * shape.n_indexer_head_dim as u64,
                    0,
                    0,
                ],
            );
            specf(
                &mut out,
                "blk.%u.attn_idx_k.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_indexer_head_dim as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_idx_w.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_indexer_head as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_idx_k_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_indexer_head_dim as u64, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_idx_k_norm.bias",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_indexer_head_dim as u64, 0, 0, 0],
            );
        }
        specf(
            &mut out,
            "blk.%u.ffn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        if il < shape.n_leading_dense || is_nextn(shape, il) {
            specf(
                &mut out,
                "blk.%u.ffn_gate.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_dense as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_dense as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [shape.n_ff_dense as u64, e, 0, 0],
            );
        } else {
            specf(
                &mut out,
                "blk.%u.ffn_gate_inp.weight",
                il,
                TypeClass::Exact(T_F32),
                2,
                [e, shape.n_expert as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.exp_probs_b.bias",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_expert as u64, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_gate_exps.weight",
                il,
                TypeClass::Exact(T_IQ2_XXS),
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up_exps.weight",
                il,
                TypeClass::Exact(T_IQ2_XXS),
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down_exps.weight",
                il,
                TypeClass::Exact(T_Q2_K),
                3,
                [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_gate_shexp.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_exp as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up_shexp.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_ff_exp as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down_shexp.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [shape.n_ff_exp as u64, e, 0, 0],
            );
        }
        if is_nextn(shape, il) {
            specf(
                &mut out,
                "blk.%u.eh_proj.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [2 * e, e, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.enorm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.hnorm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.shared_head_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
        }
    }
    spec(
        &mut out,
        "token_embd_mtp.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    out
}

fn expected_solar(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let q_dim = shape.n_head as u64 * shape.n_head_dim as u64;
    let kv_dim = shape.n_head_kv as u64 * shape.n_head_dim as u64;
    let kda_dim = shape.n_head as u64 * shape.n_kda_head_dim as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    for il in 0..shape.n_layer {
        specf(
            &mut out,
            "blk.%u.attn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        if solar_layer_is_gqa(shape.family, shape.n_layer, il) {
            specf(
                &mut out,
                "blk.%u.attn_q.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, q_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_k.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, kv_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_v.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, kv_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_gate.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, q_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_output.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [q_dim, e, 0, 0],
            );
        } else {
            specf(
                &mut out,
                "blk.%u.attn_q.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, kda_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_k.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, kda_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_v.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, kda_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_conv1d_q.weight",
                il,
                TypeClass::SolarConv,
                3,
                [shape.n_ssm_conv as u64, 1, kda_dim, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_conv1d_k.weight",
                il,
                TypeClass::SolarConv,
                3,
                [shape.n_ssm_conv as u64, 1, kda_dim, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_conv1d_v.weight",
                il,
                TypeClass::SolarConv,
                3,
                [shape.n_ssm_conv as u64, 1, kda_dim, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_f_a.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_kda_head_dim as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_f_b.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [shape.n_kda_head_dim as u64, kda_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_beta.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_head as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_a",
                il,
                TypeClass::SolarDecay,
                2,
                [1, shape.n_head as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_dt.bias",
                il,
                TypeClass::Exact(T_F32),
                1,
                [kda_dim, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_g_a.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [e, shape.n_kda_head_dim as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_g_b.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [shape.n_kda_head_dim as u64, kda_dim, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ssm_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_kda_head_dim as u64, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_output.weight",
                il,
                TypeClass::Exact(T_Q8_0),
                2,
                [kda_dim, e, 0, 0],
            );
        }
        specf(
            &mut out,
            "blk.%u.ffn_gate_inp.weight",
            il,
            TypeClass::Exact(T_F32),
            2,
            [e, shape.n_expert as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.exp_probs_b.bias",
            il,
            TypeClass::Exact(T_F32),
            1,
            [shape.n_expert as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_gate_exps.weight",
            il,
            TypeClass::SolarGateUp,
            3,
            [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_up_exps.weight",
            il,
            TypeClass::SolarGateUp,
            3,
            [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_down_exps.weight",
            il,
            TypeClass::SolarDown,
            3,
            [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_gate_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_ff_shexp as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_up_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_ff_shexp as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_down_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [shape.n_ff_shexp as u64, e, 0, 0],
        );
    }
    out
}

fn expected_exaone(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let q_dim = shape.n_head as u64 * shape.n_head_dim as u64;
    let kv_dim = shape.n_head_kv as u64 * shape.n_head_dim as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::ExaoneQuant,
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::ExaoneQuant,
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    for il in 0..shape.n_layer {
        specf(
            &mut out,
            "blk.%u.attn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q.weight",
            il,
            TypeClass::ExaoneQuant,
            2,
            [e, q_dim, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_k.weight",
            il,
            TypeClass::ExaoneQuant,
            2,
            [e, kv_dim, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_v.weight",
            il,
            TypeClass::ExaoneQuant,
            2,
            [e, kv_dim, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_output.weight",
            il,
            TypeClass::ExaoneQuant,
            2,
            [q_dim, e, 0, 0],
        );
        if shape.use_qk_norm {
            specf(
                &mut out,
                "blk.%u.attn_q_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_head_dim as u64, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_k_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_head_dim as u64, 0, 0, 0],
            );
        }
        specf(
            &mut out,
            "blk.%u.ffn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        if il < shape.n_leading_dense || is_nextn(shape, il) {
            specf(
                &mut out,
                "blk.%u.ffn_gate.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [e, shape.n_ff_dense as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [e, shape.n_ff_dense as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [shape.n_ff_dense as u64, e, 0, 0],
            );
        } else {
            specf(
                &mut out,
                "blk.%u.ffn_gate_inp.weight",
                il,
                TypeClass::Exact(T_F32),
                2,
                [e, shape.n_expert as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.exp_probs_b.bias",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_expert as u64, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_gate_exps.weight",
                il,
                TypeClass::ExaoneQuant,
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up_exps.weight",
                il,
                TypeClass::ExaoneQuant,
                3,
                [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down_exps.weight",
                il,
                TypeClass::ExaoneQuant,
                3,
                [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_gate_shexp.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [e, shape.n_ff_shexp as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_up_shexp.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [e, shape.n_ff_shexp as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.ffn_down_shexp.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [shape.n_ff_shexp as u64, e, 0, 0],
            );
        }
        if is_nextn(shape, il) {
            specf(
                &mut out,
                "blk.%u.nextn.eh_proj.weight",
                il,
                TypeClass::ExaoneQuant,
                2,
                [2 * e, e, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.nextn.enorm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.nextn.hnorm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.nextn.shared_head_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [e, 0, 0, 0],
            );
        }
    }
    out
}

fn qwen_hc(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape, inject: bool) {
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    spec(
        out,
        format!("{prefix}.norm.weight"),
        TypeClass::QwenPlain,
        1,
        [hc_dim, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.mix_down.weight"),
        TypeClass::Exact(T_BF16),
        2,
        [hc_dim, 320, 0, 0],
    );
    spec(
        out,
        format!("{prefix}.mix_up.weight"),
        TypeClass::Exact(T_BF16),
        2,
        [320, hc_dim, 0, 0],
    );
    if inject {
        spec(
            out,
            format!("{prefix}.inject.weight"),
            TypeClass::Exact(T_BF16),
            2,
            [hc_dim, hc, 0, 0],
        );
    }
}

fn qwen_linear_attention(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape) {
    let e = shape.n_embd as u64;
    let key_heads = 16;
    let value_heads = 48;
    let head_dim = shape.n_kda_head_dim as u64;
    let key_dim = key_heads * head_dim;
    let value_dim = value_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    for (suffix, class, ndim, dims) in [
        ("a_log", TypeClass::QwenPlain, 1, [value_heads, 0, 0, 0]),
        (
            "conv.weight",
            TypeClass::Exact(T_BF16),
            3,
            [shape.n_ssm_conv as u64, 1, conv_dim, 0],
        ),
        ("dt_bias", TypeClass::QwenPlain, 1, [value_heads, 0, 0, 0]),
        (
            "in_a.weight",
            TypeClass::QwenMatrix,
            2,
            [e, value_heads, 0, 0],
        ),
        (
            "in_b.weight",
            TypeClass::QwenMatrix,
            2,
            [e, value_heads, 0, 0],
        ),
        ("qkv.weight", TypeClass::QwenMatrix, 2, [e, conv_dim, 0, 0]),
        ("z.weight", TypeClass::QwenMatrix, 2, [e, value_dim, 0, 0]),
        ("norm.weight", TypeClass::QwenPlain, 1, [head_dim, 0, 0, 0]),
        ("out.weight", TypeClass::QwenMatrix, 2, [value_dim, e, 0, 0]),
    ] {
        spec(out, format!("{prefix}.{suffix}"), class, ndim, dims);
    }
}

fn qwen_qsa(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape) {
    let e = shape.n_embd as u64;
    let head_dim = shape.n_head_dim as u64;
    let index_head_dim = shape.n_indexer_head_dim as u64;
    let index_dim = (shape.n_indexer_head as u64 + 1) * index_head_dim;
    let q_dim = 2 * shape.n_head as u64 * head_dim;
    let kv_dim = shape.n_head_kv as u64 * head_dim;
    let value_dim = shape.n_head as u64 * head_dim;
    for (suffix, class, ndim, dims) in [
        (
            "attn_index_qk.weight",
            TypeClass::QwenMatrix,
            2,
            [e, index_dim, 0, 0],
        ),
        (
            "attn_index_q_norm.weight",
            TypeClass::QwenPlain,
            1,
            [index_head_dim, 0, 0, 0],
        ),
        (
            "attn_index_k_norm.weight",
            TypeClass::QwenPlain,
            1,
            [index_head_dim, 0, 0, 0],
        ),
        ("attn_q.weight", TypeClass::QwenMatrix, 2, [e, q_dim, 0, 0]),
        (
            "attn_q_norm.weight",
            TypeClass::QwenPlain,
            1,
            [head_dim, 0, 0, 0],
        ),
        ("attn_k.weight", TypeClass::QwenMatrix, 2, [e, kv_dim, 0, 0]),
        (
            "attn_k_norm.weight",
            TypeClass::QwenPlain,
            1,
            [head_dim, 0, 0, 0],
        ),
        ("attn_v.weight", TypeClass::QwenMatrix, 2, [e, kv_dim, 0, 0]),
        (
            "attn_output.weight",
            TypeClass::QwenMatrix,
            2,
            [value_dim, e, 0, 0],
        ),
    ] {
        spec(out, format!("{prefix}.{suffix}"), class, ndim, dims);
    }
}

fn qwen_moe(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape, mtp: bool) {
    let e = shape.n_embd as u64;
    let experts = shape.n_expert as u64;
    let ff = shape.n_ff_exp as u64;
    let sh = shape.n_ff_shexp as u64;
    let routed = if mtp {
        TypeClass::QwenMtpRouted
    } else {
        TypeClass::QwenMatrix
    };
    for (suffix, class, ndim, dims) in [
        (
            "ffn_gate_inp.weight",
            TypeClass::QwenPlain,
            2,
            [e, experts, 0, 0],
        ),
        ("ffn_gate_exps.weight", routed, 3, [e, ff, experts, 0]),
        ("ffn_up_exps.weight", routed, 3, [e, ff, experts, 0]),
        (
            "ffn_down_exps.main.weight",
            TypeClass::QwenMatrix,
            3,
            [512, e, experts, 0],
        ),
        (
            "ffn_down_exps.tail.weight",
            TypeClass::QwenMatrix,
            3,
            [128, e, experts, 0],
        ),
        (
            "ffn_gate_shexp.weight",
            TypeClass::QwenMatrix,
            2,
            [e, sh, 0, 0],
        ),
        (
            "ffn_up_shexp.weight",
            TypeClass::QwenMatrix,
            2,
            [e, sh, 0, 0],
        ),
        (
            "ffn_down_shexp.weight",
            TypeClass::QwenMatrix,
            2,
            [sh, e, 0, 0],
        ),
        (
            "ffn_shexp_gate_inp.weight",
            TypeClass::QwenPlain,
            2,
            [e, 1, 0, 0],
        ),
    ] {
        spec(out, format!("{prefix}.{suffix}"), class, ndim, dims);
    }
}

fn qwen_ple(out: &mut Vec<LayoutSpec>, shape: &Shape) {
    let e = shape.n_embd as u64;
    let hc_dim = e * shape.n_hc as u64;
    for (name, class, ndim, dims) in [
        (
            "blk.1.ple.conv.weight",
            TypeClass::Exact(T_BF16),
            3,
            [shape.n_ssm_conv as u64, 1, hc_dim, 0],
        ),
        (
            "blk.1.ple.key.weight",
            TypeClass::QwenMatrix,
            2,
            [e, hc_dim, 0, 0],
        ),
        (
            "blk.1.ple.value.weight",
            TypeClass::QwenMatrix,
            2,
            [e, e, 0, 0],
        ),
        (
            "blk.1.ple.conv_norm.weight",
            TypeClass::QwenPlain,
            1,
            [hc_dim, 0, 0, 0],
        ),
        (
            "blk.1.ple.key_norm.weight",
            TypeClass::QwenPlain,
            1,
            [hc_dim, 0, 0, 0],
        ),
        (
            "blk.1.ple.query_norm.weight",
            TypeClass::QwenPlain,
            1,
            [hc_dim, 0, 0, 0],
        ),
        (
            "blk.1.ple.layer_multipliers",
            TypeClass::Exact(T_I64),
            1,
            [3, 0, 0, 0],
        ),
        (
            "blk.1.ple.head_offsets",
            TypeClass::Exact(T_I64),
            1,
            [16, 0, 0, 0],
        ),
        (
            "blk.1.ple.head_vocab_sizes",
            TypeClass::Exact(T_I64),
            1,
            [16, 0, 0, 0],
        ),
    ] {
        spec(out, name, class, ndim, dims);
    }
}

fn qwen_vision(out: &mut Vec<LayoutSpec>, shape: &Shape) {
    const HIDDEN: u64 = 1152;
    const FF: u64 = 4304;
    const MERGED: u64 = 4 * HIDDEN;
    spec5(
        out,
        "vision.patch_embed.weight",
        TypeClass::Exact(T_BF16),
        [16, 16, 2, 3, HIDDEN],
    );
    spec(
        out,
        "vision.patch_embed.bias",
        TypeClass::Exact(T_F32),
        1,
        [HIDDEN, 0, 0, 0],
    );
    spec(
        out,
        "vision.position_embd.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [HIDDEN, 2304, 0, 0],
    );
    for il in 0..27 {
        for (fmt, class, ndim, dims) in [
            (
                "vblk.%u.norm1.weight",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.norm1.bias",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.attn_qkv.weight",
                TypeClass::Exact(T_Q8_0),
                2,
                [HIDDEN, 3 * HIDDEN, 0, 0],
            ),
            (
                "vblk.%u.attn_qkv.bias",
                TypeClass::Exact(T_F32),
                1,
                [3 * HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.attn_output.weight",
                TypeClass::Exact(T_Q8_0),
                2,
                [HIDDEN, HIDDEN, 0, 0],
            ),
            (
                "vblk.%u.attn_output.bias",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.norm2.weight",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.norm2.bias",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
            (
                "vblk.%u.ffn_up.weight",
                TypeClass::Exact(T_Q8_0),
                2,
                [HIDDEN, FF, 0, 0],
            ),
            (
                "vblk.%u.ffn_up.bias",
                TypeClass::Exact(T_F32),
                1,
                [FF, 0, 0, 0],
            ),
            (
                "vblk.%u.ffn_down.weight",
                TypeClass::Exact(T_BF16),
                2,
                [FF, HIDDEN, 0, 0],
            ),
            (
                "vblk.%u.ffn_down.bias",
                TypeClass::Exact(T_F32),
                1,
                [HIDDEN, 0, 0, 0],
            ),
        ] {
            specf(out, fmt, il, class, ndim, dims);
        }
    }
    for (name, class, ndim, dims) in [
        (
            "vision.merger.norm.weight",
            TypeClass::Exact(T_F32),
            1,
            [HIDDEN, 0, 0, 0],
        ),
        (
            "vision.merger.norm.bias",
            TypeClass::Exact(T_F32),
            1,
            [HIDDEN, 0, 0, 0],
        ),
        (
            "vision.merger.ffn_up.weight",
            TypeClass::Exact(T_Q8_0),
            2,
            [MERGED, MERGED, 0, 0],
        ),
        (
            "vision.merger.ffn_up.bias",
            TypeClass::Exact(T_F32),
            1,
            [MERGED, 0, 0, 0],
        ),
        (
            "vision.merger.ffn_down.weight",
            TypeClass::Exact(T_Q8_0),
            2,
            [MERGED, shape.n_embd as u64, 0, 0],
        ),
        (
            "vision.merger.ffn_down.bias",
            TypeClass::Exact(T_F32),
            1,
            [shape.n_embd as u64, 0, 0, 0],
        ),
    ] {
        spec(out, name, class, ndim, dims);
    }
}

fn expected_qwen4exp(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::QwenMatrix,
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::QwenMatrix,
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    qwen_hc(&mut out, "hc_input", shape, false);
    qwen_vision(&mut out, shape);
    for il in 0..shape.n_layer {
        let prefix = format!("blk.{il}");
        qwen_hc(&mut out, &format!("{prefix}.hc_attn"), shape, true);
        if qwen4exp_layer_is_full_attention(shape, il) {
            qwen_qsa(&mut out, &prefix, shape);
        } else {
            qwen_linear_attention(&mut out, &format!("{prefix}.linear_attn"), shape);
        }
        qwen_moe(&mut out, &prefix, shape, false);
        qwen_hc(&mut out, &format!("{prefix}.hc_ffn"), shape, true);
        if il == 1 {
            qwen_ple(&mut out, shape);
        }
    }
    for (name, class, ndim, dims) in [
        (
            "mtp.fc_embedding.weight",
            TypeClass::QwenMatrix,
            2,
            [e, e, 0, 0],
        ),
        (
            "mtp.fc_hidden.weight",
            TypeClass::QwenMatrix,
            2,
            [e, e, 0, 0],
        ),
        (
            "mtp.fc_embedding_norm.weight",
            TypeClass::QwenPlain,
            1,
            [e, 0, 0, 0],
        ),
        (
            "mtp.fc_hidden_norm.weight",
            TypeClass::QwenPlain,
            1,
            [e * shape.n_hc as u64, 0, 0, 0],
        ),
    ] {
        spec(&mut out, name, class, ndim, dims);
    }
    qwen_hc(&mut out, "mtp.hc_input", shape, false);
    qwen_hc(&mut out, "mtp.blk.0.hc_attn", shape, true);
    qwen_qsa(&mut out, "mtp.blk.0", shape);
    qwen_moe(&mut out, "mtp.blk.0", shape, true);
    qwen_hc(&mut out, "mtp.blk.0.hc_ffn", shape, true);
    out
}

fn expected_glm53(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    let hc_mix = 2 * hc + hc * hc;
    let kda = shape.n_head as u64 * shape.n_kda_head_dim as u64;
    let q_dim = shape.n_head as u64 * shape.n_key_mla as u64;
    let index_q = shape.n_indexer_head as u64 * shape.n_indexer_head_dim as u64;

    for (name, class, ndim, dims) in [
        (
            "token_embd.weight",
            TypeClass::GlmDense,
            2,
            [e, shape.n_vocab as u64, 0, 0],
        ),
        (
            "output_norm.weight",
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        ),
        (
            "output.weight",
            TypeClass::GlmDense,
            2,
            [e, shape.n_vocab as u64, 0, 0],
        ),
    ] {
        spec(&mut out, name, class, ndim, dims);
    }

    for il in 0..shape.n_layer {
        let p = format!("blk.{il}.");
        spec(
            &mut out,
            format!("{p}attn_norm.weight"),
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        spec(
            &mut out,
            format!("{p}ffn_norm.weight"),
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        if !is_nextn(shape, il) {
            for stem in ["hc_attn", "hc_ffn"] {
                spec(
                    &mut out,
                    format!("{p}{stem}_fn.weight"),
                    TypeClass::Exact(T_BF16),
                    2,
                    [hc_dim, hc_mix, 0, 0],
                );
                spec(
                    &mut out,
                    format!("{p}{stem}_scale.weight"),
                    TypeClass::Exact(T_F32),
                    1,
                    [3, 0, 0, 0],
                );
                spec(
                    &mut out,
                    format!("{p}{stem}_base.weight"),
                    TypeClass::Exact(T_F32),
                    1,
                    [hc_mix, 0, 0, 0],
                );
            }
        }

        if glm53_layer_is_kda(shape, il) {
            for (suffix, dims) in [
                ("kda_q.weight", [e, kda, 0, 0]),
                ("kda_k.weight", [e, kda, 0, 0]),
                ("kda_v.weight", [e, kda, 0, 0]),
                ("kda_f_a.weight", [e, shape.n_kda_head_dim as u64, 0, 0]),
                ("kda_f_b.weight", [shape.n_kda_head_dim as u64, kda, 0, 0]),
                ("kda_beta.weight", [e, shape.n_head as u64, 0, 0]),
                ("kda_g_a.weight", [e, shape.n_kda_head_dim as u64, 0, 0]),
                ("kda_g_b.weight", [shape.n_kda_head_dim as u64, kda, 0, 0]),
                ("kda_output.weight", [kda, e, 0, 0]),
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::GlmDense,
                    2,
                    dims,
                );
            }
            for suffix in [
                "kda_q_conv.weight",
                "kda_k_conv.weight",
                "kda_v_conv.weight",
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::Exact(T_F32),
                    3,
                    [shape.n_ssm_conv as u64, 1, kda, 0],
                );
            }
            for (suffix, dim) in [
                ("kda_dt_bias.weight", kda),
                ("kda_a_log.weight", shape.n_head as u64),
                ("kda_o_norm.weight", shape.n_kda_head_dim as u64),
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::Exact(T_F32),
                    1,
                    [dim, 0, 0, 0],
                );
            }
        } else {
            for (suffix, ndim, dims) in [
                ("attn_q_a.weight", 2, [e, shape.n_lora_q as u64, 0, 0]),
                ("attn_q_b.weight", 2, [shape.n_lora_q as u64, q_dim, 0, 0]),
                (
                    "attn_kv_a_mqa.weight",
                    2,
                    [e, (shape.n_kv_lora + shape.n_rot) as u64, 0, 0],
                ),
                (
                    "attn_k_b.weight",
                    3,
                    [
                        (shape.n_key_mla - shape.n_rot) as u64,
                        shape.n_kv_lora as u64,
                        shape.n_head as u64,
                        0,
                    ],
                ),
                (
                    "attn_v_b.weight",
                    3,
                    [
                        shape.n_kv_lora as u64,
                        shape.n_value_mla as u64,
                        shape.n_head as u64,
                        0,
                    ],
                ),
                (
                    "attn_output.weight",
                    2,
                    [shape.n_head as u64 * shape.n_value_mla as u64, e, 0, 0],
                ),
                (
                    "indexer.attn_q_b.weight",
                    2,
                    [shape.n_lora_q as u64, index_q, 0, 0],
                ),
                (
                    "indexer.attn_k.weight",
                    2,
                    [e, shape.n_indexer_head_dim as u64, 0, 0],
                ),
                (
                    "indexer.proj.weight",
                    2,
                    [e, shape.n_indexer_head as u64, 0, 0],
                ),
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::GlmDense,
                    ndim,
                    dims,
                );
            }
            for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias"] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::Exact(T_F32),
                    1,
                    [shape.n_indexer_head_dim as u64, 0, 0, 0],
                );
            }
            spec(
                &mut out,
                format!("{p}indexer.pool_ape.weight"),
                TypeClass::Exact(T_BF16),
                2,
                [shape.n_indexer_head_dim as u64, 4, 0, 0],
            );
            spec(
                &mut out,
                format!("{p}indexer.pool_gate.weight"),
                TypeClass::Exact(T_BF16),
                2,
                [e, shape.n_indexer_head_dim as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("{p}attn_q_a_norm.weight"),
                TypeClass::Exact(T_F32),
                1,
                [shape.n_lora_q as u64, 0, 0, 0],
            );
            spec(
                &mut out,
                format!("{p}attn_kv_a_norm.weight"),
                TypeClass::Exact(T_F32),
                1,
                [shape.n_kv_lora as u64, 0, 0, 0],
            );
        }

        if il < shape.n_leading_dense {
            for (suffix, dims) in [
                ("ffn_gate.weight", [e, shape.n_ff_dense as u64, 0, 0]),
                ("ffn_up.weight", [e, shape.n_ff_dense as u64, 0, 0]),
                ("ffn_down.weight", [shape.n_ff_dense as u64, e, 0, 0]),
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::GlmDense,
                    2,
                    dims,
                );
            }
        } else {
            spec(
                &mut out,
                format!("{p}ffn_gate_inp.weight"),
                TypeClass::Exact(T_F32),
                2,
                [e, shape.n_expert as u64, 0, 0],
            );
            spec(
                &mut out,
                format!("{p}exp_probs_b.bias"),
                TypeClass::Exact(T_F32),
                1,
                [shape.n_expert as u64, 0, 0, 0],
            );
            for (suffix, dims) in [
                (
                    "ffn_gate_exps.weight",
                    [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
                ),
                (
                    "ffn_up_exps.weight",
                    [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
                ),
                (
                    "ffn_down_exps.weight",
                    [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
                ),
            ] {
                spec(&mut out, format!("{p}{suffix}"), TypeClass::Routed, 3, dims);
            }
            for (suffix, dims) in [
                ("ffn_gate_shexp.weight", [e, shape.n_ff_exp as u64, 0, 0]),
                ("ffn_up_shexp.weight", [e, shape.n_ff_exp as u64, 0, 0]),
                ("ffn_down_shexp.weight", [shape.n_ff_exp as u64, e, 0, 0]),
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::GlmDense,
                    2,
                    dims,
                );
            }
        }

        if is_nextn(shape, il) {
            spec(
                &mut out,
                format!("{p}nextn.eh_proj.weight"),
                TypeClass::GlmDense,
                2,
                [2 * e, e, 0, 0],
            );
            for suffix in [
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
            ] {
                spec(
                    &mut out,
                    format!("{p}{suffix}"),
                    TypeClass::Exact(T_F32),
                    1,
                    [e, 0, 0, 0],
                );
            }
        }
    }
    out
}

fn expected_deepseek(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    let hc_mix = 2 * hc + hc * hc;
    let q_dim = shape.n_head as u64 * shape.n_head_dim as u64;
    let out_low = shape.n_out_group as u64 * shape.n_lora_o as u64;
    spec(
        &mut out,
        "token_embd.weight",
        TypeClass::Exact(T_F16),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "output_hc_base.weight",
        TypeClass::Exact(T_F32),
        1,
        [hc, 0, 0, 0],
    );
    spec(
        &mut out,
        "output_hc_fn.weight",
        TypeClass::Exact(T_F16),
        2,
        [hc_dim, hc, 0, 0],
    );
    spec(
        &mut out,
        "output_hc_scale.weight",
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        &mut out,
        "output_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "output.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_vocab as u64, 0, 0],
    );
    for il in 0..shape.n_layer {
        let ratio = expected_compress_ratio(shape.variant, shape.n_layer, il);
        specf(
            &mut out,
            "blk.%u.hc_attn_fn.weight",
            il,
            TypeClass::Exact(T_F16),
            2,
            [hc_dim, hc_mix, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.hc_attn_scale.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [3, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.hc_attn_base.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [hc_mix, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_a.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_lora_q as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_a_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [shape.n_lora_q as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_q_b.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [shape.n_lora_q as u64, q_dim, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_kv.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_head_dim as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_kv_a_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [shape.n_head_dim as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_sinks.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [shape.n_head as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.attn_output_a.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [
                shape.n_head_dim as u64 * (shape.n_head / shape.n_out_group) as u64,
                out_low,
                0,
                0,
            ],
        );
        specf(
            &mut out,
            "blk.%u.attn_output_b.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [out_low, e, 0, 0],
        );
        if ratio != 0 {
            let coff = if ratio == 4 { 2u64 } else { 1 };
            let comp_width = coff * shape.n_head_dim as u64;
            specf(
                &mut out,
                "blk.%u.attn_compressor_ape.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [comp_width, ratio as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_compressor_kv.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [e, comp_width, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_compressor_gate.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [e, comp_width, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.attn_compressor_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_head_dim as u64, 0, 0, 0],
            );
        }
        if ratio == 4 {
            let index_q = shape.n_indexer_head as u64 * shape.n_indexer_head_dim as u64;
            let index_width = 2 * shape.n_indexer_head_dim as u64;
            specf(
                &mut out,
                "blk.%u.indexer.attn_q_b.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [shape.n_lora_q as u64, index_q, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.indexer.proj.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [e, shape.n_indexer_head as u64, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.indexer_compressor_ape.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [index_width, 4, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.indexer_compressor_kv.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [e, index_width, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.indexer_compressor_gate.weight",
                il,
                TypeClass::Exact(T_F16),
                2,
                [e, index_width, 0, 0],
            );
            specf(
                &mut out,
                "blk.%u.indexer_compressor_norm.weight",
                il,
                TypeClass::Exact(T_F32),
                1,
                [shape.n_indexer_head_dim as u64, 0, 0, 0],
            );
        }
        specf(
            &mut out,
            "blk.%u.hc_ffn_fn.weight",
            il,
            TypeClass::Exact(T_F16),
            2,
            [hc_dim, hc_mix, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.hc_ffn_scale.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [3, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.hc_ffn_base.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [hc_mix, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_norm.weight",
            il,
            TypeClass::Exact(T_F32),
            1,
            [e, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_gate_inp.weight",
            il,
            TypeClass::Exact(T_F16),
            2,
            [e, shape.n_expert as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.exp_probs_b.bias",
            il,
            TypeClass::OptionalExact(T_F32),
            1,
            [shape.n_expert as u64, 0, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_gate_exps.weight",
            il,
            TypeClass::Routed,
            3,
            [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_up_exps.weight",
            il,
            TypeClass::Routed,
            3,
            [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_down_exps.weight",
            il,
            TypeClass::Routed,
            3,
            [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_gate_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_ff_exp as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_up_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [e, shape.n_ff_exp as u64, 0, 0],
        );
        specf(
            &mut out,
            "blk.%u.ffn_down_shexp.weight",
            il,
            TypeClass::Exact(T_Q8_0),
            2,
            [shape.n_ff_exp as u64, e, 0, 0],
        );
        if il < shape.n_hash_layer {
            specf(
                &mut out,
                "blk.%u.ffn_gate_tid2eid.weight",
                il,
                TypeClass::Exact(T_I32),
                2,
                [shape.n_expert_used as u64, shape.n_vocab as u64, 0, 0],
            );
        }
    }
    out
}

fn deepseek_block(out: &mut Vec<LayoutSpec>, prefix: &str, shape: &Shape) {
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    let hc_mix = 2 * hc + hc * hc;
    let q_dim = shape.n_head as u64 * shape.n_head_dim as u64;
    let out_low = shape.n_out_group as u64 * shape.n_lora_o as u64;
    spec(
        out,
        format!("{prefix}hc_attn_fn.weight"),
        TypeClass::Plain,
        2,
        [hc_dim, hc_mix, 0, 0],
    );
    spec(
        out,
        format!("{prefix}hc_attn_scale.weight"),
        TypeClass::Exact(T_F32),
        1,
        [3, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}hc_attn_base.weight"),
        TypeClass::Exact(T_F32),
        1,
        [hc_mix, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_a.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_lora_q as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_a_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_lora_q as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_q_b.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [shape.n_lora_q as u64, q_dim, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_kv.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_head_dim as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_kv_a_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_head_dim as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_sinks.weight"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_head as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}attn_output_a.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [
            shape.n_head_dim as u64 * (shape.n_head / shape.n_out_group) as u64,
            out_low,
            0,
            0,
        ],
    );
    spec(
        out,
        format!("{prefix}attn_output_b.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [out_low, e, 0, 0],
    );
    spec(
        out,
        format!("{prefix}hc_ffn_fn.weight"),
        TypeClass::Plain,
        2,
        [hc_dim, hc_mix, 0, 0],
    );
    spec(
        out,
        format!("{prefix}hc_ffn_scale.weight"),
        TypeClass::Exact(T_F32),
        1,
        [3, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}hc_ffn_base.weight"),
        TypeClass::Exact(T_F32),
        1,
        [hc_mix, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_norm.weight"),
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_gate_inp.weight"),
        TypeClass::Plain,
        2,
        [e, shape.n_expert as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}exp_probs_b.bias"),
        TypeClass::Exact(T_F32),
        1,
        [shape.n_expert as u64, 0, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_gate_exps.weight"),
        TypeClass::Routed,
        3,
        [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_up_exps.weight"),
        TypeClass::Routed,
        3,
        [e, shape.n_ff_exp as u64, shape.n_expert as u64, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_down_exps.weight"),
        TypeClass::Routed,
        3,
        [shape.n_ff_exp as u64, e, shape.n_expert as u64, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_gate_shexp.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_ff_exp as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_up_shexp.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [e, shape.n_ff_exp as u64, 0, 0],
    );
    spec(
        out,
        format!("{prefix}ffn_down_shexp.weight"),
        TypeClass::Exact(T_Q8_0),
        2,
        [shape.n_ff_exp as u64, e, 0, 0],
    );
}

pub fn expected_mtp_layouts(shape: &Shape) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    spec(
        &mut out,
        "mtp.0.hc_head_base.weight",
        TypeClass::Exact(T_F32),
        1,
        [hc, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.hc_head_fn.weight",
        TypeClass::Plain,
        2,
        [hc_dim, hc, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.hc_head_scale.weight",
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.e_proj.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, e, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.h_proj.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [e, e, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.enorm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.hnorm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "mtp.0.norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    deepseek_block(&mut out, "mtp.0.", shape);
    out
}

pub fn expected_dspark_layouts(shape: &Shape, markov_rank: u32) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    let e = shape.n_embd as u64;
    let hc = shape.n_hc as u64;
    let hc_dim = e * hc;
    let rank = markov_rank as u64;
    spec(
        &mut out,
        "dspark.main_proj.weight",
        TypeClass::Exact(T_Q8_0),
        2,
        [3 * e, e, 0, 0],
    );
    spec(
        &mut out,
        "dspark.main_norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    spec(
        &mut out,
        "dspark.markov_w1.weight",
        TypeClass::Exact(T_F16),
        2,
        [rank, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "dspark.markov_w2.weight",
        TypeClass::Exact(T_F16),
        2,
        [rank, shape.n_vocab as u64, 0, 0],
    );
    spec(
        &mut out,
        "dspark.conf_proj.weight",
        TypeClass::Exact(T_F32),
        2,
        [e + rank, 1, 0, 0],
    );
    spec(
        &mut out,
        "dspark.hc_head_fn.weight",
        TypeClass::Plain,
        2,
        [hc_dim, hc, 0, 0],
    );
    spec(
        &mut out,
        "dspark.hc_head_base.weight",
        TypeClass::Exact(T_F32),
        1,
        [hc, 0, 0, 0],
    );
    spec(
        &mut out,
        "dspark.hc_head_scale.weight",
        TypeClass::Exact(T_F32),
        1,
        [1, 0, 0, 0],
    );
    spec(
        &mut out,
        "dspark.norm.weight",
        TypeClass::Exact(T_F32),
        1,
        [e, 0, 0, 0],
    );
    for il in 0..DSPARK_N_LAYER {
        deepseek_block(&mut out, &format!("dspark.{il}."), shape);
    }
    out
}

pub fn expected_layouts(shape: &Shape) -> Vec<LayoutSpec> {
    match shape.family {
        ModelFamily::Glm53 => expected_glm53(shape),
        ModelFamily::Qwen4Exp => expected_qwen4exp(shape),
        ModelFamily::Motif3 => expected_motif3(shape),
        ModelFamily::Dots3Note => expected_dots3(shape),
        ModelFamily::SolarOpen2 => expected_solar(shape),
        ModelFamily::ExaoneMoe => expected_exaone(shape),
        ModelFamily::DeepSeek4 => expected_deepseek(shape),
    }
}

fn spec_line(s: &LayoutSpec) -> String {
    let mut dims = String::new();
    let n = if matches!(s.class, TypeClass::SolarConv | TypeClass::SolarDecay) {
        3.min(s.ndim.max(2))
    } else {
        s.ndim
    };
    for i in 0..n as usize {
        if i > 0 {
            dims.push(',');
        }
        dims.push_str(&s.dim[i].to_string());
    }
    format!("SPEC {} {} {} {}\n", s.name, s.class.token(), s.ndim, dims)
}

pub fn dump_expected_layouts_shape(shape: &Shape) -> String {
    let specs = expected_layouts(shape);
    let mut out = format!(
        "LAYOUT name={} family={} variant={} n_layer={}\n",
        shape.name, shape.family as u32, shape.variant as u32, shape.n_layer
    );
    for s in &specs {
        out.push_str(&spec_line(s));
    }
    out.push_str(&format!("COUNT n={}\n", specs.len()));
    out
}

fn dump_layout_table(header: String, specs: &[LayoutSpec]) -> String {
    let mut out = header;
    for s in specs {
        out.push_str(&spec_line(s));
    }
    out.push_str(&format!("COUNT n={}\n", specs.len()));
    out
}

pub fn dump_expected_mtp_shape(shape: &Shape) -> String {
    dump_layout_table(
        format!(
            "LAYOUT kind=mtp name={} family={} variant={}\n",
            shape.name, shape.family as u32, shape.variant as u32
        ),
        &expected_mtp_layouts(shape),
    )
}

pub fn dump_expected_dspark_shape(shape: &Shape) -> String {
    dump_layout_table(
        format!(
            "LAYOUT kind=dspark name={} family={} variant={} markov_rank={} n_layer={}\n",
            shape.name,
            shape.family as u32,
            shape.variant as u32,
            DSPARK_MARKOV_RANK,
            DSPARK_N_LAYER
        ),
        &expected_dspark_layouts(shape, DSPARK_MARKOV_RANK),
    )
}

pub fn dump_expected_support() -> String {
    let mut out = String::new();
    for v in [Variant::Flash, Variant::Pro] {
        let shape = shape_for_variant(v);
        out.push_str(&dump_expected_mtp_shape(&shape));
        out.push_str(&dump_expected_dspark_shape(&shape));
    }
    out
}

pub fn dump_expected_layouts_variant(name: &str) -> Option<String> {
    let (support, v) = crate::bind::catalog_from_bind_name(name)?;
    let shape = shape_for_variant(v);
    Some(match support {
        None => dump_expected_layouts_shape(&shape),
        Some(crate::bind::SupportCatalog::Mtp) => dump_expected_mtp_shape(&shape),
        Some(crate::bind::SupportCatalog::Dspark) => dump_expected_dspark_shape(&shape),
    })
}

pub fn dump_expected_layouts() -> String {
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
        out.push_str(&dump_expected_layouts_shape(&shape_for_variant(v)));
    }
    out
}

fn plan_by_name(plan: &BindPlan) -> std::collections::HashMap<&str, &TensorInfo> {
    plan.slots
        .iter()
        .filter_map(|s| s.tensor.as_ref().map(|t| (s.name.as_str(), t)))
        .collect()
}

fn expect_specs(
    specs: &[LayoutSpec],
    by_name: &std::collections::HashMap<&str, &TensorInfo>,
) -> Result<(), LayoutError> {
    for spec in specs {
        expect_tensor(spec, by_name.get(spec.name.as_str()).copied())?;
    }
    Ok(())
}

fn expect_gate_up(
    by_name: &std::collections::HashMap<&str, &TensorInfo>,
    gate: &str,
    up: &str,
    il: u32,
) -> Result<(), LayoutError> {
    if let (Some(g), Some(u)) = (by_name.get(gate), by_name.get(up)) {
        if g.typ != u.typ {
            return Err(LayoutError::GateUp(il));
        }
    }
    Ok(())
}

pub fn validate_layouts(plan: &BindPlan) -> Result<(), LayoutError> {
    let by_name = plan_by_name(plan);
    expect_specs(&expected_layouts(&plan.shape), &by_name)?;
    if matches!(
        plan.shape.family,
        ModelFamily::DeepSeek4
            | ModelFamily::SolarOpen2
            | ModelFamily::Qwen4Exp
            | ModelFamily::Glm53
    ) {
        for il in 0..plan.shape.n_layer {
            expect_gate_up(
                &by_name,
                &format!("blk.{il}.ffn_gate_exps.weight"),
                &format!("blk.{il}.ffn_up_exps.weight"),
                il,
            )?;
        }
    }
    if plan.shape.family == ModelFamily::Qwen4Exp {
        expect_gate_up(
            &by_name,
            "mtp.blk.0.ffn_gate_exps.weight",
            "mtp.blk.0.ffn_up_exps.weight",
            0,
        )?;
    }
    Ok(())
}

pub fn validate_mtp_layouts(plan: &BindPlan) -> Result<(), LayoutError> {
    let by_name = plan_by_name(plan);
    expect_specs(&expected_mtp_layouts(&plan.shape), &by_name)?;
    expect_gate_up(
        &by_name,
        "mtp.0.ffn_gate_exps.weight",
        "mtp.0.ffn_up_exps.weight",
        0,
    )
}

pub fn validate_dspark_layouts(plan: &BindPlan, markov_rank: u32) -> Result<(), LayoutError> {
    let by_name = plan_by_name(plan);
    expect_specs(&expected_dspark_layouts(&plan.shape, markov_rank), &by_name)?;
    for il in 0..DSPARK_N_LAYER {
        expect_gate_up(
            &by_name,
            &format!("dspark.{il}.ffn_gate_exps.weight"),
            &format!("dspark.{il}.ffn_up_exps.weight"),
            il,
        )?;
    }
    Ok(())
}

pub fn validate_support_layouts(
    plan: &BindPlan,
    support: Option<crate::bind::SupportCatalog>,
) -> Result<(), LayoutError> {
    match support {
        None => validate_layouts(plan),
        Some(crate::bind::SupportCatalog::Mtp) => validate_mtp_layouts(plan),
        Some(crate::bind::SupportCatalog::Dspark) => {
            validate_dspark_layouts(plan, DSPARK_MARKOV_RANK)
        }
    }
}

fn fake_tensor(name: &str, typ: u32, ndim: u32, dim: [u64; 8]) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        ndim,
        dim,
        typ,
        rel_offset: 0,
        abs_offset: 0,
        elements: 0,
        bytes: 0,
        shard: 0,
    }
}

/// Fixed C↔Rust expect tapes (same cases as `layout_c_oracle check`).
pub fn dump_layout_check_tapes() -> String {
    let embd = LayoutSpec {
        name: "token_embd.weight".into(),
        class: TypeClass::Exact(T_Q8_0),
        ndim: 2,
        dim: [4, 8, 0, 0, 0, 0, 0, 0],
    };
    let opt = LayoutSpec {
        name: "exp_probs_b.bias".into(),
        class: TypeClass::OptionalExact(T_F32),
        ndim: 1,
        dim: [4, 0, 0, 0, 0, 0, 0, 0],
    };
    let mut out = String::new();
    out.push_str(&format!(
        "missing {}\n",
        expect_tensor(&embd, None).unwrap_err().token()
    ));
    out.push_str(&format!(
        "optional {}\n",
        match expect_tensor(&opt, None) {
            Ok(()) => "ok".to_string(),
            Err(e) => e.token(),
        }
    ));
    let bad_ty = fake_tensor("token_embd.weight", T_F32, 2, [4, 8, 0, 0, 0, 0, 0, 0]);
    out.push_str(&format!(
        "type {}\n",
        expect_tensor(&embd, Some(&bad_ty)).unwrap_err().token()
    ));
    let bad_nd = fake_tensor("token_embd.weight", T_Q8_0, 1, [4, 0, 0, 0, 0, 0, 0, 0]);
    out.push_str(&format!(
        "ndim {}\n",
        expect_tensor(&embd, Some(&bad_nd)).unwrap_err().token()
    ));
    let bad_dim = fake_tensor("token_embd.weight", T_Q8_0, 2, [4, 7, 0, 0, 0, 0, 0, 0]);
    out.push_str(&format!(
        "dim {}\n",
        expect_tensor(&embd, Some(&bad_dim)).unwrap_err().token()
    ));
    let ok = fake_tensor("token_embd.weight", T_Q8_0, 2, [4, 8, 0, 0, 0, 0, 0, 0]);
    out.push_str(&format!(
        "ok {}\n",
        match expect_tensor(&embd, Some(&ok)) {
            Ok(()) => "ok".to_string(),
            Err(e) => e.token(),
        }
    ));
    let conv = LayoutSpec {
        name: "ssm_conv.weight".into(),
        class: TypeClass::SolarConv,
        ndim: 3,
        dim: [4, 1, 16, 0, 0, 0, 0, 0],
    };
    let conv3 = fake_tensor("ssm_conv.weight", T_F32, 3, [4, 1, 16, 0, 0, 0, 0, 0]);
    out.push_str(&format!(
        "conv3 {}\n",
        match expect_tensor(&conv, Some(&conv3)) {
            Ok(()) => "ok".to_string(),
            Err(e) => e.token(),
        }
    ));
    let conv4 = fake_tensor("ssm_conv.weight", T_F32, 4, [4, 1, 16, 1, 0, 0, 0, 0]);
    out.push_str(&format!(
        "conv4 {}\n",
        match expect_tensor(&conv, Some(&conv4)) {
            Ok(()) => "ok".to_string(),
            Err(e) => e.token(),
        }
    ));
    out
}
