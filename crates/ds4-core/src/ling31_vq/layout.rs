//! Original HF names and the selected recipe's stored types: 505 Q4_K
//! projections (504 nonexpert + head), native FP8 KDA f/g and MLA kv_b,
//! BF16/F32 controls, and per-layer GVQ1 experts. Layer 72 is the optional
//! MTP block; its tensors are present together or not at all.
use super::{BF16, CODEBOOK, F32, FP8, Q4_K, SCALE_BLOCK, VECTOR, VQ};
use crate::{LayoutSpec, TypeClass};

pub(super) const EMBED: u64 = 4096;
pub(super) const VOCAB: u64 = 157_184;
pub(super) const TRUNK: u32 = 72;
pub(super) const MTP: u32 = TRUNK;
const EXPERTS: u32 = 512;
const FFN: u64 = 1280;
const DENSE_FFN: u64 = 10240;
const LEADING_DENSE: u32 = 4;
const LAYER_GROUP: u32 = 8;
const HEADS: u64 = 64;
const KDA_HEAD: u64 = 128;
const KDA_WIDTH: u64 = HEADS * KDA_HEAD;
const KDA_CONV: u64 = 4;
const Q_RANK: u64 = 1536;
const KV_RANK: u64 = 512;
const ROPE: u64 = 64;
const NOPE: u64 = 128;
const VALUE: u64 = 128;
/// RECIPE.md: these trunk layers and MTP use K8192; the rest K4096.
const K8192_LAYERS: [u32; 9] = [4, 5, 6, 9, 12, 43, 57, 58, MTP];

pub(super) fn is_mla(il: u32) -> bool {
    il == MTP || (il + 1).is_multiple_of(LAYER_GROUP)
}

pub(super) fn codewords(il: u32) -> u64 {
    if K8192_LAYERS.contains(&il) {
        8192
    } else {
        4096
    }
}

fn tensor(out: &mut Vec<LayoutSpec>, il: u32, name: String, typ: u32, shape: &[u64]) {
    let mut dim = [0; 8];
    for (i, &n) in shape.iter().rev().enumerate() {
        dim[i] = n;
    }
    let class = if il == MTP {
        TypeClass::OptionalExact(typ)
    } else {
        TypeClass::Exact(typ)
    };
    out.push(LayoutSpec {
        name,
        class,
        ndim: shape.len() as u32,
        dim,
    });
}

fn matrix(out: &mut Vec<LayoutSpec>, il: u32, prefix: &str, typ: u32, rows: u64, cols: u64) {
    tensor(out, il, format!("{prefix}.weight"), typ, &[rows, cols]);
    if typ == FP8 {
        tensor(
            out,
            il,
            format!("{prefix}.weight_scale_inv"),
            F32,
            &[rows.div_ceil(SCALE_BLOCK), cols.div_ceil(SCALE_BLOCK)],
        );
    }
}

fn mlp(out: &mut Vec<LayoutSpec>, il: u32, prefix: &str, typ: u32, hidden: u64) {
    for name in ["gate_proj", "up_proj"] {
        matrix(out, il, &format!("{prefix}.{name}"), typ, hidden, EMBED);
    }
    matrix(out, il, &format!("{prefix}.down_proj"), typ, EMBED, hidden);
}

fn attention(out: &mut Vec<LayoutSpec>, il: u32, a: &str) {
    if !is_mla(il) {
        for role in ["q", "k", "v"] {
            matrix(out, il, &format!("{a}.{role}_proj"), Q4_K, KDA_WIDTH, EMBED);
            tensor(
                out,
                il,
                format!("{a}.{role}_conv1d.weight"),
                BF16,
                &[KDA_WIDTH, 1, KDA_CONV],
            );
        }
        for role in ["f", "g"] {
            matrix(out, il, &format!("{a}.{role}_proj"), FP8, KDA_WIDTH, EMBED);
        }
        matrix(out, il, &format!("{a}.b_proj"), BF16, HEADS, EMBED);
        matrix(out, il, &format!("{a}.o_proj"), Q4_K, EMBED, KDA_WIDTH);
        tensor(out, il, format!("{a}.o_norm.weight"), BF16, &[KDA_HEAD]);
        tensor(out, il, format!("{a}.A_log"), F32, &[HEADS]);
        tensor(out, il, format!("{a}.dt_bias"), F32, &[KDA_WIDTH]);
        return;
    }
    // MTP nonexperts keep their original FP8 (and the BF16 trunk gate).
    let (proj, gate) = if il == MTP { (FP8, FP8) } else { (Q4_K, BF16) };
    matrix(out, il, &format!("{a}.q_a_proj"), proj, Q_RANK, EMBED);
    matrix(
        out,
        il,
        &format!("{a}.q_b_proj"),
        proj,
        HEADS * (NOPE + ROPE),
        Q_RANK,
    );
    matrix(
        out,
        il,
        &format!("{a}.kv_a_proj_with_mqa"),
        proj,
        KV_RANK + ROPE,
        EMBED,
    );
    matrix(
        out,
        il,
        &format!("{a}.kv_b_proj"),
        FP8,
        HEADS * (NOPE + VALUE),
        KV_RANK,
    );
    matrix(out, il, &format!("{a}.dense"), proj, EMBED, HEADS * VALUE);
    matrix(out, il, &format!("{a}.g_proj"), gate, HEADS, EMBED);
    tensor(
        out,
        il,
        format!("{a}.q_a_layernorm.weight"),
        BF16,
        &[Q_RANK],
    );
    tensor(
        out,
        il,
        format!("{a}.kv_a_layernorm.weight"),
        BF16,
        &[KV_RANK],
    );
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.layers.{il}");
    for name in ["input_layernorm", "post_attention_layernorm"] {
        tensor(out, il, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    attention(out, il, &format!("{p}.attention"));
    if il < LEADING_DENSE {
        mlp(out, il, &format!("{p}.mlp"), Q4_K, DENSE_FFN);
        return;
    }
    let experts = u64::from(EXPERTS);
    matrix(out, il, &format!("{p}.mlp.gate"), BF16, experts, EMBED);
    tensor(
        out,
        il,
        format!("{p}.mlp.gate.expert_bias"),
        F32,
        &[experts],
    );
    let shared = if il == MTP { FP8 } else { Q4_K };
    mlp(out, il, &format!("{p}.mlp.shared_experts"), shared, FFN);
    for expert in 0..EXPERTS {
        mlp(out, il, &format!("{p}.mlp.experts.{expert}"), VQ, FFN);
    }
    tensor(
        out,
        il,
        format!("{p}.mlp.vq_codebook"),
        CODEBOOK,
        &[codewords(il), u64::from(VECTOR)],
    );
    if il == MTP {
        matrix(out, il, &format!("{p}.eh_proj"), BF16, EMBED, 2 * EMBED);
        for role in ["enorm", "hnorm", "final_layernorm"] {
            tensor(out, il, format!("{p}.{role}.weight"), BF16, &[EMBED]);
        }
    }
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    matrix(&mut out, 0, "model.word_embeddings", BF16, VOCAB, EMBED);
    matrix(&mut out, 0, "lm_head", Q4_K, VOCAB, EMBED);
    tensor(&mut out, 0, "model.norm.weight".into(), BF16, &[EMBED]);
    for il in 0..=MTP {
        layer(&mut out, il);
    }
    out
}
