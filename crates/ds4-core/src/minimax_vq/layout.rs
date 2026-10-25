//! Original HF names and the recipe's stored types: Q4_K attention, shared
//! experts, dense FFN (layers 0-2) and head; Q8_0 indexer projections; F32
//! routers and correction biases; source BF16 norms; per-layer 13-bit GVQ1
//! experts (plus the GVR8 residual stage on `RESIDUAL_LAYERS`); the BF16
//! embedding of its own part. The DSpark drafter's tensors are optional:
//! they bind only when the drafter part is selected.
use super::{BF16, CODEBOOK, F32, Q4_K, Q8_0, RESIDUAL, RESIDUAL_WORDS, VECTOR, VQ, WORDS};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows (`embedding` part).
pub const EMBEDDING: &str = "language_model.model.embed_tokens.weight";

const EMBED: u64 = 6144;
const VOCAB: u64 = 200_064;
const LAYERS: u32 = 60;
const DENSE: u32 = 3;
const HEADS: u64 = 64;
const KV_HEADS: u64 = 4;
const HEAD: u64 = 128;
const INDEX: u64 = 128;
const DENSE_FFN: u64 = 12_288;
const EXPERTS: u32 = 128;
const FFN: u64 = 3072;
const DRAFT_LAYERS: u32 = 5;
const DRAFT_TAPS: u64 = 5;
const MARKOV_RANK: u64 = 256;
/// release.json `r13p8_layers`: the twenty layers with the 8-bit residual
/// stage, chosen by the converter's end-to-end KL sweep.
pub(super) const RESIDUAL_LAYERS: [u32; 20] = [
    16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 40, 43, 53,
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

fn mlp(out: &mut Vec<LayoutSpec>, prefix: &str, class: TypeClass, hidden: u64) {
    for name in ["gate_proj", "up_proj"] {
        tensor(
            out,
            format!("{prefix}.{name}.weight"),
            class,
            &[hidden, EMBED],
        );
    }
    tensor(
        out,
        format!("{prefix}.down_proj.weight"),
        class,
        &[EMBED, hidden],
    );
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
            ("w1", [FFN, EMBED]),
            ("w3", [FFN, EMBED]),
            ("w2", [EMBED, FFN]),
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
    let p = format!("language_model.model.layers.{il}");
    for name in ["input_layernorm", "post_attention_layernorm"] {
        exact(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    let a = format!("{p}.self_attn");
    for (name, typ, rows, cols) in [
        ("q_proj", Q4_K, HEADS * HEAD, EMBED),
        ("k_proj", Q4_K, KV_HEADS * HEAD, EMBED),
        ("v_proj", Q4_K, KV_HEADS * HEAD, EMBED),
        ("o_proj", Q4_K, EMBED, HEADS * HEAD),
        ("index_q_proj", Q8_0, KV_HEADS * INDEX, EMBED),
        ("index_k_proj", Q8_0, INDEX, EMBED),
    ] {
        exact(out, format!("{a}.{name}.weight"), typ, &[rows, cols]);
    }
    for name in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
        exact(out, format!("{a}.{name}.weight"), BF16, &[HEAD]);
    }
    if il < DENSE {
        mlp(out, &format!("{p}.mlp"), TypeClass::Exact(Q4_K), DENSE_FFN);
        return;
    }
    let m = format!("{p}.block_sparse_moe");
    let experts_n = u64::from(EXPERTS);
    exact(out, format!("{m}.gate.weight"), F32, &[experts_n, EMBED]);
    exact(
        out,
        format!("{m}.e_score_correction_bias"),
        F32,
        &[experts_n],
    );
    mlp(
        out,
        &format!("{m}.shared_experts"),
        TypeClass::Exact(Q4_K),
        FFN,
    );
    experts(out, &m, RESIDUAL_LAYERS.contains(&il));
}

/// DSpark (`dspark-q8_0.gguf`): five SWA decoder layers, the tap fusion
/// `fc` over five target rows, the Markov head and the confidence head.
fn drafter(out: &mut Vec<LayoutSpec>) {
    let d = "language_model.model.dspark";
    let q8 = TypeClass::OptionalExact(Q8_0);
    let bf16 = TypeClass::OptionalExact(BF16);
    tensor(
        out,
        format!("{d}.fc.weight"),
        q8,
        &[EMBED, DRAFT_TAPS * EMBED],
    );
    for name in ["hidden_norm", "final_norm"] {
        tensor(out, format!("{d}.{name}.weight"), bf16, &[EMBED]);
    }
    for name in ["markov_w1", "markov_w2"] {
        tensor(
            out,
            format!("{d}.markov_head.{name}.weight"),
            bf16,
            &[VOCAB, MARKOV_RANK],
        );
    }
    tensor(
        out,
        format!("{d}.confidence_head.proj.weight"),
        bf16,
        &[1, EMBED + MARKOV_RANK],
    );
    tensor(out, format!("{d}.confidence_head.proj.bias"), bf16, &[1]);
    for il in 0..DRAFT_LAYERS {
        let p = format!("{d}.layers.{il}.decoder_layer");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            tensor(out, format!("{p}.{name}.weight"), bf16, &[EMBED]);
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
                q8,
                &[rows, cols],
            );
        }
        for name in ["q_norm", "k_norm"] {
            tensor(out, format!("{p}.self_attn.{name}.weight"), bf16, &[HEAD]);
        }
        mlp(out, &format!("{p}.mlp"), q8, DENSE_FFN);
    }
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    exact(&mut out, EMBEDDING.into(), BF16, &[VOCAB, EMBED]);
    exact(
        &mut out,
        "language_model.lm_head.weight".into(),
        Q4_K,
        &[VOCAB, EMBED],
    );
    exact(
        &mut out,
        "language_model.model.norm.weight".into(),
        BF16,
        &[EMBED],
    );
    for il in 0..LAYERS {
        layer(&mut out, il);
    }
    drafter(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // release tensor-manifest.json type_counts minus the vision part:
    // 30,662 trunk and embedding tensors, 62 DSpark tensors.
    #[test]
    fn layout_matches_the_published_inventory() {
        let specs = layouts();
        let required = |typ| {
            specs
                .iter()
                .filter(|s| matches!(s.class, TypeClass::Exact(t) if t == typ))
                .count()
        };
        let optional = specs
            .iter()
            .filter(|s| matches!(s.class, TypeClass::OptionalExact(_)))
            .count();
        assert_eq!(specs.len() - optional, 30_662);
        assert_eq!(optional, 62);
        assert_eq!(required(CODEBOOK), 77);
        assert_eq!(required(VQ), 21_888);
        assert_eq!(required(RESIDUAL), 7680);
        assert_eq!(required(Q4_K), 421);
        assert_eq!(required(Q8_0), 120);
        assert_eq!(required(F32), 114);
        assert_eq!(required(BF16), 362);
    }
}
