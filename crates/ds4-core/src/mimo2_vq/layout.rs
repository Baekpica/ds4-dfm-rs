//! Original HF names and the recipe's stored types: Q8_0 fused qkv (stored
//! de-interleaved `[q | k | v]`), attention output, dense FFN (layer 0) and
//! head; BF16 router, norms and SWA sink biases; F32 correction biases;
//! per-layer 13-bit GVQ1 experts (plus the GVR8 residual stage on
//! `RESIDUAL_LAYERS`); the BF16 embedding of its own part. The embedded MTP
//! blocks are optional: they bind only when their part is selected.
use super::{BF16, CODEBOOK, F32, Q8_0, RESIDUAL, RESIDUAL_WORDS, VECTOR, VQ, WORDS};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows (`embedding` part).
pub const EMBEDDING: &str = "model.embed_tokens.weight";

const EMBED: u64 = 4096;
const VOCAB: u64 = 152_576;
const LAYERS: u32 = 48;
const HEADS: u64 = 64;
const KEY: u64 = 192;
const VALUE: u64 = 128;
const DENSE_FFN: u64 = 16_384;
const EXPERTS: u32 = 256;
const FFN: u64 = 2048;
const MTP_BLOCKS: u32 = 3;
/// config.json hybrid_layer_pattern: full attention (4 KV heads, no sink)
/// on these layers, SWA-128 (8 KV heads, sink bias) elsewhere.
const FULL: [u32; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];
/// release.json `r13p8_layers`: the 32 layers with the 8-bit residual
/// stage (the 8 GA MoE layers, then the 24 best measured SWA layers).
pub(super) const RESIDUAL_LAYERS: [u32; 32] = [
    5, 8, 11, 13, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
];

fn tensor(out: &mut Vec<LayoutSpec>, name: String, class: TypeClass, shape: &[u64]) {
    let mut dim = [0; 8];
    for (i, &n) in shape.iter().rev().enumerate() {
        dim[i] = n;
    }
    out.push(LayoutSpec {
        name,
        class,
        ndim: shape.len() as u32,
        dim,
    });
}

fn exact(out: &mut Vec<LayoutSpec>, name: String, typ: u32, shape: &[u64]) {
    tensor(out, name, TypeClass::Exact(typ), shape);
}

fn mlp(out: &mut Vec<LayoutSpec>, prefix: &str, class: TypeClass) {
    for name in ["gate_proj", "up_proj"] {
        tensor(
            out,
            format!("{prefix}.{name}.weight"),
            class,
            &[DENSE_FFN, EMBED],
        );
    }
    tensor(
        out,
        format!("{prefix}.down_proj.weight"),
        class,
        &[EMBED, DENSE_FFN],
    );
}

/// Fused qkv, output projection and (SWA) sink bias of one attention block.
fn attention(out: &mut Vec<LayoutSpec>, prefix: &str, class: TypeClass, full: bool) {
    let kv = if full { 4 } else { 8 };
    tensor(
        out,
        format!("{prefix}.qkv_proj.weight"),
        class,
        &[HEADS * KEY + kv * (KEY + VALUE), EMBED],
    );
    tensor(
        out,
        format!("{prefix}.o_proj.weight"),
        class,
        &[EMBED, HEADS * VALUE],
    );
    if !full {
        let sink = match class {
            TypeClass::OptionalExact(_) => TypeClass::OptionalExact(BF16),
            _ => TypeClass::Exact(BF16),
        };
        tensor(out, format!("{prefix}.attention_sink_bias"), sink, &[HEADS]);
    }
}

fn experts(out: &mut Vec<LayoutSpec>, prefix: &str, residual: bool) {
    exact(
        out,
        format!("{prefix}.vq_codebook"),
        CODEBOOK,
        &[u64::from(WORDS), u64::from(VECTOR)],
    );
    if residual {
        exact(
            out,
            format!("{prefix}.vq_codebook_residual"),
            CODEBOOK,
            &[u64::from(RESIDUAL_WORDS), u64::from(VECTOR)],
        );
    }
    for expert in 0..EXPERTS {
        for (name, shape) in [
            ("gate_proj", [FFN, EMBED]),
            ("up_proj", [FFN, EMBED]),
            ("down_proj", [EMBED, FFN]),
        ] {
            let matrix = format!("{prefix}.experts.{expert}.{name}.weight");
            if residual {
                exact(out, format!("{matrix}_residual"), RESIDUAL, &shape);
            }
            exact(out, matrix, VQ, &shape);
        }
    }
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.layers.{il}");
    for name in ["input_layernorm", "post_attention_layernorm"] {
        exact(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    attention(
        out,
        &format!("{p}.self_attn"),
        TypeClass::Exact(Q8_0),
        FULL.contains(&il),
    );
    if il == 0 {
        mlp(out, &format!("{p}.mlp"), TypeClass::Exact(Q8_0));
        return;
    }
    exact(out, format!("{p}.mlp.gate.weight"), BF16, &[256, EMBED]);
    exact(
        out,
        format!("{p}.mlp.gate.e_score_correction_bias"),
        F32,
        &[256],
    );
    experts(out, &format!("{p}.mlp"), RESIDUAL_LAYERS.contains(&il));
}

/// One embedded MTP block (`mtp` part): eh_proj over [enorm(embed) |
/// hnorm(hidden)], an SWA attention block and a dense SiLU MLP.
fn mtp_block(out: &mut Vec<LayoutSpec>, block: u32) {
    let p = format!("model.mtp.layers.{block}");
    let norm = TypeClass::OptionalExact(BF16);
    let q8 = TypeClass::OptionalExact(Q8_0);
    for name in [
        "enorm",
        "hnorm",
        "input_layernorm",
        "pre_mlp_layernorm",
        "final_layernorm",
    ] {
        tensor(out, format!("{p}.{name}.weight"), norm, &[EMBED]);
    }
    tensor(out, format!("{p}.eh_proj.weight"), q8, &[EMBED, 2 * EMBED]);
    attention(out, &format!("{p}.self_attn"), q8, false);
    mlp(out, &format!("{p}.mlp"), q8);
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    exact(&mut out, EMBEDDING.into(), BF16, &[VOCAB, EMBED]);
    exact(&mut out, "model.norm.weight".into(), BF16, &[EMBED]);
    exact(&mut out, "lm_head.weight".into(), Q8_0, &[VOCAB, EMBED]);
    for il in 0..LAYERS {
        layer(&mut out, il);
    }
    for block in 0..MTP_BLOCKS {
        mtp_block(&mut out, block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_cover_the_release_tensor_counts() {
        let all = layouts();
        let required = all
            .iter()
            .filter(|s| !matches!(s.class, TypeClass::OptionalExact(_)))
            .count();
        // Body 330, embedding 1, 15 R13 parts x 769 and 32 R13+8 parts x 1538
        // (release.json `tensors`), and the MTP part's 36 optional tensors.
        assert_eq!(required, 330 + 1 + 15 * 769 + 32 * 1538);
        assert_eq!(all.len() - required, 36);
        let mut names: Vec<_> = all.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
    }
}
