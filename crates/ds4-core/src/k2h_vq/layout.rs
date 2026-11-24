//! Original HF names and the recipe's stored types (FORMAT.md "Body part",
//! "Expert parts", "Embedding part"): Q8_0 attention, dense FFN, shared
//! experts, head and token table; F32 norms and routers (BF16 values widened
//! exactly); per-layer 13-bit GVQ1 experts on one K8192 book.
use super::{CODEBOOK, F32, Q8_0, VECTOR, VQ, WORDS};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows or native uploads (`embedding`
/// part, the inventory's last tensor).
pub(super) const EMBEDDING: &str = "model.embed_tokens.weight";

const EMBED: u64 = 6144;
const VOCAB: u64 = 250_624;
const LAYERS: u32 = 61;
const DENSE: u32 = 3;
const HEADS: u64 = 48;
const KV_HEADS: u64 = 8;
const HEAD: u64 = 128;
const DENSE_FFN: u64 = 16_384;
const EXPERTS: u32 = 192;
const FFN: u64 = 1792;

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
    for norm in ["input_layernorm", "post_attention_layernorm"] {
        tensor(out, format!("{p}.{norm}.weight"), F32, &[EMBED]);
    }
    for (name, rows, cols) in [
        ("q_proj", HEADS * HEAD, EMBED),
        ("k_proj", KV_HEADS * HEAD, EMBED),
        ("v_proj", KV_HEADS * HEAD, EMBED),
        ("o_proj", EMBED, HEADS * HEAD),
    ] {
        tensor(
            out,
            format!("{p}.self_attn.{name}.weight"),
            Q8_0,
            &[rows, cols],
        );
    }
    if il < DENSE {
        mlp(out, &format!("{p}.mlp"), Q8_0, DENSE_FFN);
        return;
    }
    let experts = u64::from(EXPERTS);
    tensor(out, format!("{p}.mlp.gate.weight"), F32, &[experts, EMBED]);
    tensor(out, format!("{p}.mlp.gate.bias"), F32, &[experts]);
    mlp(out, &format!("{p}.mlp.shared_experts"), Q8_0, FFN);
    tensor(
        out,
        format!("{p}.mlp.vq_codebook"),
        CODEBOOK,
        &[u64::from(WORDS), u64::from(VECTOR)],
    );
    for expert in 0..EXPERTS {
        mlp(out, &format!("{p}.mlp.experts.{expert}"), VQ, FFN);
    }
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    for il in 0..LAYERS {
        layer(&mut out, il);
    }
    tensor(&mut out, "model.norm.weight".into(), F32, &[EMBED]);
    tensor(&mut out, "lm_head.weight".into(), Q8_0, &[VOCAB, EMBED]);
    tensor(&mut out, EMBEDDING.into(), Q8_0, &[VOCAB, EMBED]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // tensor-manifest.json: 34,134 stored tensors, 58 books, 33,408 GVQ1
    // matrices, 429 Q8_0 matrices and 239 F32 vectors and routers.
    #[test]
    fn layout_matches_the_manifest() {
        let specs = layouts();
        assert_eq!(specs.len(), 34_134);
        let count = |typ| {
            specs
                .iter()
                .filter(|s| matches!(s.class, TypeClass::Exact(t) if t == typ))
                .count()
        };
        assert_eq!(count(CODEBOOK), 58);
        assert_eq!(count(VQ), 33_408);
        assert_eq!(count(Q8_0), 429);
        assert_eq!(count(F32), 239);
        assert_eq!(specs.last().unwrap().name, EMBEDDING);
    }
}
