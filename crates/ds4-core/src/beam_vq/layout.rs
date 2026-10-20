//! Original HF names and the recipe's stored types: 270 Q4_K matrices
//! (attention q/k/v/o/gate of 52 layers, dense FFN of layers 0-2, the head),
//! source-precision BF16 norms and gate biases, F32 routers, per-layer GVQ1
//! experts and the BF16 embedding of its own part.
use super::{BF16, CODEBOOK, F32, Q4_K, VECTOR, VQ};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows (`embedding` part).
pub const EMBEDDING: &str = "model.embed_tokens.weight";

const EMBED: u64 = 5120;
const VOCAB: u64 = 151_936;
const LAYERS: u32 = 52;
const DENSE: u32 = 3;
const HEADS: u64 = 48;
const KV_HEADS: u64 = 8;
const HEAD: u64 = 128;
const DENSE_FFN: u64 = 20_480;
const EXPERTS: u32 = 256;
const FFN: u64 = 2560;
/// recipe.json `r13_layers`: the bottom eight MoE layers keep K8192 books.
const K8192_LAYERS: std::ops::RangeInclusive<u32> = 3..=10;

pub(super) fn codewords(il: u32) -> u64 {
    if K8192_LAYERS.contains(&il) {
        8192
    } else {
        4096
    }
}

fn tensor(out: &mut Vec<LayoutSpec>, name: String, typ: u32, shape: &[u64]) {
    let mut dim = [0; 8];
    for (i, &n) in shape.iter().rev().enumerate() {
        dim[i] = n;
    }
    out.push(LayoutSpec {
        name,
        class: TypeClass::Exact(typ),
        ndim: shape.len() as u32,
        dim,
    });
}

fn mlp(out: &mut Vec<LayoutSpec>, prefix: &str, typ: u32, hidden: u64) {
    for name in ["gate_proj", "up_proj"] {
        tensor(
            out,
            format!("{prefix}.{name}.weight"),
            typ,
            &[hidden, EMBED],
        );
    }
    tensor(
        out,
        format!("{prefix}.down_proj.weight"),
        typ,
        &[EMBED, hidden],
    );
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.layers.{il}");
    for name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        tensor(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    let a = format!("{p}.self_attn");
    for (name, rows, cols) in [
        ("q_proj", HEADS * HEAD, EMBED),
        ("k_proj", KV_HEADS * HEAD, EMBED),
        ("v_proj", KV_HEADS * HEAD, EMBED),
        ("o_proj", EMBED, HEADS * HEAD),
        ("gate_proj", HEADS * HEAD, EMBED),
    ] {
        tensor(out, format!("{a}.{name}.weight"), Q4_K, &[rows, cols]);
    }
    tensor(out, format!("{a}.gate_bias"), BF16, &[HEADS]);
    tensor(out, format!("{a}.q_norm.weight"), BF16, &[HEAD]);
    tensor(out, format!("{a}.k_norm.weight"), BF16, &[HEAD]);
    if il < DENSE {
        mlp(out, &format!("{p}.mlp"), Q4_K, DENSE_FFN);
        return;
    }
    let experts = u64::from(EXPERTS);
    tensor(out, format!("{p}.mlp.gate.weight"), F32, &[experts, EMBED]);
    tensor(
        out,
        format!("{p}.mlp.gate.e_score_correction_bias"),
        F32,
        &[experts],
    );
    for expert in 0..EXPERTS {
        mlp(out, &format!("{p}.mlp.experts.{expert}"), VQ, FFN);
    }
    tensor(
        out,
        format!("{p}.mlp.vq_codebook"),
        CODEBOOK,
        &[codewords(il), u64::from(VECTOR)],
    );
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    tensor(&mut out, EMBEDDING.into(), BF16, &[VOCAB, EMBED]);
    tensor(&mut out, "lm_head.weight".into(), Q4_K, &[VOCAB, EMBED]);
    tensor(&mut out, "model.norm.weight".into(), BF16, &[EMBED]);
    for il in 0..LAYERS {
        layer(&mut out, il);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // FORMAT.md: 38,415 stored tensors, 49 books, 37,632 GVQ1 matrices.
    #[test]
    fn layout_matches_the_published_inventory() {
        let specs = layouts();
        assert_eq!(specs.len(), 38_415);
        let count = |typ| {
            specs
                .iter()
                .filter(|s| matches!(s.class, TypeClass::Exact(t) if t == typ))
                .count()
        };
        assert_eq!(count(CODEBOOK), 49);
        assert_eq!(count(VQ), 37_632);
        assert_eq!(count(Q4_K), 270);
        assert_eq!(codewords(3), 8192);
        assert_eq!(codewords(10), 8192);
        assert_eq!(codewords(11), 4096);
    }
}
