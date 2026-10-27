//! Original HF names and the recipe's stored types (FORMAT.md "Body part",
//! "Expert parts"): Q4_K attention, shared experts, dense FFN and head;
//! Q8_0 indexer projections; source-precision BF16 norms, gated-norm gates,
//! indexer weights and routers with F32 correction biases; per-layer GVQ1
//! v2 experts with gate/up/down books and the BF16 embedding of its own part.
use super::{BF16, CODEBOOK, F32, Q4_K, Q8_0, VECTOR, VQ};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows (`embedding` part).
pub const EMBEDDING: &str = "model.embed_tokens.weight";

const EMBED: u64 = 7168;
const VOCAB: u64 = 163_840;
const LAYERS: u32 = 61;
const DENSE: u32 = 1;
const HEADS: u64 = 64;
const Q_RANK: u64 = 1536;
const KV_RANK: u64 = 512;
const ROPE: u64 = 64;
const KEY: u64 = 192;
const VALUE: u64 = 128;
const NOPE: u64 = KEY - ROPE;
const GATE_RANK: u64 = 16;
const INDEX_HEADS: u64 = 64;
const INDEX_DIM: u64 = 128;
const DENSE_FFN: u64 = 18_432;
const EXPERTS: u32 = 256;
const FFN: u64 = 2048;
/// recipe.json: seven mid-stack MoE layers keep K1024 (10-bit) books.
const K1024_LAYERS: [u32; 7] = [16, 20, 24, 28, 32, 36, 40];

pub(super) fn codewords(il: u32) -> u64 {
    if K1024_LAYERS.contains(&il) {
        1024
    } else {
        512
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

// Gated RMSNorm: the norm weight and the rank-16 W_down/W_up gate.
fn gated_norm(out: &mut Vec<LayoutSpec>, prefix: &str) {
    tensor(out, format!("{prefix}.norm.weight"), BF16, &[EMBED]);
    tensor(
        out,
        format!("{prefix}.W_down.weight"),
        BF16,
        &[GATE_RANK, EMBED],
    );
    tensor(
        out,
        format!("{prefix}.W_up.weight"),
        BF16,
        &[EMBED, GATE_RANK],
    );
}

fn attention(out: &mut Vec<LayoutSpec>, a: &str) {
    // q_b takes [q_a_layernorm(q_a), q_a]; each head's 320 outputs are its
    // 192 query values and 128 output-gate logits.
    for (name, rows, cols) in [
        ("q_a_proj", Q_RANK, EMBED),
        ("q_b_proj", HEADS * (KEY + VALUE), 2 * Q_RANK),
        ("kv_a_proj_with_mqa", KV_RANK + ROPE, EMBED),
        ("kv_b_proj", HEADS * (NOPE + VALUE), KV_RANK),
        ("o_proj", EMBED, HEADS * VALUE),
    ] {
        tensor(out, format!("{a}.{name}.weight"), Q4_K, &[rows, cols]);
    }
    tensor(out, format!("{a}.q_a_layernorm.weight"), BF16, &[Q_RANK]);
    tensor(out, format!("{a}.kv_a_layernorm.weight"), BF16, &[KV_RANK]);
    let i = format!("{a}.indexer");
    tensor(
        out,
        format!("{i}.wq_b.weight"),
        Q8_0,
        &[INDEX_HEADS * INDEX_DIM, Q_RANK],
    );
    tensor(out, format!("{i}.wk.weight"), Q8_0, &[INDEX_DIM, EMBED]);
    tensor(out, format!("{i}.k_norm.weight"), BF16, &[INDEX_DIM]);
    tensor(out, format!("{i}.k_norm.bias"), BF16, &[INDEX_DIM]);
    tensor(
        out,
        format!("{i}.weights_proj.weight"),
        BF16,
        &[INDEX_HEADS, EMBED],
    );
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.layers.{il}");
    gated_norm(out, &format!("{p}.input_layernorm"));
    attention(out, &format!("{p}.self_attn"));
    if il < DENSE {
        tensor(
            out,
            format!("{p}.post_attention_layernorm.weight"),
            BF16,
            &[EMBED],
        );
        mlp(out, &format!("{p}.mlp"), Q4_K, DENSE_FFN);
        return;
    }
    gated_norm(out, &format!("{p}.post_attention_layernorm"));
    let experts = u64::from(EXPERTS);
    tensor(out, format!("{p}.mlp.gate.weight"), BF16, &[experts, EMBED]);
    tensor(
        out,
        format!("{p}.mlp.gate.e_score_correction_bias"),
        F32,
        &[experts],
    );
    mlp(out, &format!("{p}.mlp.shared_experts"), Q4_K, FFN);
    for expert in 0..EXPERTS {
        mlp(out, &format!("{p}.mlp.experts.{expert}"), VQ, FFN);
    }
    for book in ["vq_codebook", "vq_codebook_down", "vq_codebook_up"] {
        tensor(
            out,
            format!("{p}.mlp.{book}"),
            CODEBOOK,
            &[codewords(il), u64::from(VECTOR)],
        );
    }
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

    // FORMAT.md / release verification: 47,662 stored tensors, 180 books,
    // 46,080 GVQ1 matrices, 489 Q4_K and 122 Q8_0 matrices.
    #[test]
    fn layout_matches_the_published_inventory() {
        let specs = layouts();
        assert_eq!(specs.len(), 47_662);
        let count = |typ| {
            specs
                .iter()
                .filter(|s| matches!(s.class, TypeClass::Exact(t) if t == typ))
                .count()
        };
        assert_eq!(count(CODEBOOK), 180);
        assert_eq!(count(VQ), 46_080);
        assert_eq!(count(Q4_K), 489);
        assert_eq!(count(Q8_0), 122);
        assert_eq!(count(F32), 60);
        assert_eq!(codewords(15), 512);
        assert_eq!(codewords(16), 1024);
        assert_eq!(codewords(41), 512);
    }
}
