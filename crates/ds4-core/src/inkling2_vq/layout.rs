//! Original HF names and the recipe's stored types: Q4_K attention
//! projections and head; source BF16 norms, router, relative-position
//! profiles, clip scales, sink projection and convolution taps; F32 router
//! bias and block attention-residual queries; per-layer 12-bit GVQ1 experts;
//! the BF16 embedding of its own part. The MTP drafter's tensors are
//! optional: they bind only when the drafter part is selected.
use super::{BF16, CODEBOOK, F32, Q4_K, Q8_0, VECTOR, VQ, WORDS};
use crate::{LayoutSpec, TypeClass};

/// The token table the host reads in rows (`embedding` part).
pub const EMBEDDING: &str = "model.llm.embed.weight";

const EMBED: u64 = 4608;
const VOCAB: u64 = 201_024;
const LAYERS: u32 = 48;
const HEADS: u64 = 48;
const KV_HEADS: u64 = 8;
const HEAD: u64 = 128;
const REL: u64 = 16;
const EXTENT: u64 = 1024;
const TAPS: u64 = 4;
const EXPERTS: u32 = 384;
const FFN: u64 = 2304;
const DRAFT_LAYERS: u32 = 5;
const DRAFT_FFN: u64 = 18_432;
const MARKOV_RANK: u64 = 256;
/// The Markov head's `rnn_proj` maps 9,728 inputs to a 768-wide state
/// (mtp-q8_0.gguf).
const MARKOV_STATE: u64 = 768;
const MARKOV_IN: u64 = 9728;
const FUSE: [u64; 2] = [6, 8];

/// Full-attention layers: (l + 1) % 6 == 0 (config.json local_layer_ids).
fn global(il: u32) -> bool {
    (il + 1).is_multiple_of(6)
}

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

fn experts(out: &mut Vec<LayoutSpec>, prefix: &str) {
    exact(
        out,
        format!("{prefix}.vq_codebook"),
        CODEBOOK,
        &[u64::from(WORDS), u64::from(VECTOR)],
    );
    for expert in 0..EXPERTS {
        for (name, shape) in [
            ("gate_proj", [FFN, EMBED]),
            ("up_proj", [FFN, EMBED]),
            ("down_proj", [EMBED, FFN]),
        ] {
            exact(
                out,
                format!("{prefix}.experts.{expert}.{name}.weight"),
                VQ,
                &shape,
            );
        }
    }
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.llm.layers.{il}");
    let a = format!("{p}.attn");
    for (name, rows, cols) in [
        ("wq_du", HEADS * HEAD, EMBED),
        ("wk_dv", KV_HEADS * HEAD, EMBED),
        ("wv_dv", KV_HEADS * HEAD, EMBED),
        ("wr_du", HEADS * REL, EMBED),
        ("gate_attn", HEADS * HEAD, EMBED),
        ("wo_ud", EMBED, HEADS * HEAD),
    ] {
        exact(out, format!("{a}.{name}.weight"), Q4_K, &[rows, cols]);
    }
    exact(
        out,
        format!("{a}.attn_sink_proj.weight"),
        BF16,
        &[HEADS, EMBED],
    );
    exact(out, format!("{a}.q_l2_clip.scale"), BF16, &[HEADS, HEAD]);
    exact(out, format!("{a}.rel_l2_clip.scale"), BF16, &[HEADS, REL]);
    exact(
        out,
        format!("{a}.rel_logits_proj.proj"),
        BF16,
        &[REL, EXTENT],
    );
    if global(il) {
        for kv in ["k", "v"] {
            exact(
                out,
                format!("{a}.{kv}_sconv.static_w"),
                BF16,
                &[TAPS, KV_HEADS * HEAD],
            );
            exact(
                out,
                format!("{a}.{kv}_sconv.weight_proj"),
                BF16,
                &[KV_HEADS * TAPS, EMBED],
            );
        }
        for name in ["q_input_norm", "k_input_norm", "v_input_norm"] {
            exact(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
        }
    } else {
        for kv in ["k", "v"] {
            exact(
                out,
                format!("{a}.{kv}_sconv.weight"),
                BF16,
                &[KV_HEADS * HEAD, 1, TAPS],
            );
        }
        exact(out, format!("{p}.attn_norm.weight"), BF16, &[EMBED]);
    }
    for name in ["attn_sconv", "mlp_sconv"] {
        exact(out, format!("{p}.{name}.weight"), BF16, &[EMBED, 1, TAPS]);
    }
    exact(out, format!("{p}.mlp_norm.weight"), BF16, &[EMBED]);
    exact(
        out,
        format!("{p}.mlp.gate.weight"),
        BF16,
        &[u64::from(EXPERTS), EMBED],
    );
    exact(
        out,
        format!("{p}.mlp.gate.bias"),
        F32,
        &[u64::from(EXPERTS)],
    );
    experts(out, &format!("{p}.mlp"));
}

/// Block attention residuals: a query per layer input (three for a
/// full-attention layer: its q, k and v inputs), one per MLP input and the
/// output query, all F32.
fn attn_res(out: &mut Vec<LayoutSpec>) {
    let r = "model.llm.attn_res";
    for il in 1..LAYERS {
        let name = format!("{r}.attn_queries.{}", il - 1);
        if global(il) {
            exact(out, name, F32, &[3, EMBED]);
        } else {
            exact(out, name, F32, &[EMBED]);
        }
    }
    for il in 0..LAYERS {
        exact(out, format!("{r}.mlp_queries.{il}"), F32, &[EMBED]);
    }
    exact(out, format!("{r}.output_query"), F32, &[EMBED]);
}

/// MTP drafter (`mtp-q8_0.gguf`): five decoder layers (Q8_0 projections,
/// dense 18,432 FFN, context K/V projections), the tap fusion weights, the
/// mask embedding and the recurrent Markov head.
fn drafter(out: &mut Vec<LayoutSpec>) {
    let q8 = TypeClass::OptionalExact(Q8_0);
    let bf16 = TypeClass::OptionalExact(BF16);
    tensor(out, "mtp.fuse_weight".into(), bf16, &FUSE);
    tensor(out, "mtp.mask_embed".into(), bf16, &[EMBED]);
    tensor(out, "mtp.norm.weight".into(), bf16, &[EMBED]);
    for il in 0..DRAFT_LAYERS {
        let p = format!("mtp.layers.{il}");
        for name in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            tensor(out, format!("{p}.{name}.weight"), bf16, &[EMBED]);
        }
        for name in ["gate_proj", "up_proj"] {
            tensor(
                out,
                format!("{p}.mlp.{name}.weight"),
                q8,
                &[DRAFT_FFN, EMBED],
            );
        }
        tensor(
            out,
            format!("{p}.mlp.down_proj.weight"),
            q8,
            &[EMBED, DRAFT_FFN],
        );
        let a = format!("{p}.self_attn");
        for name in ["q_norm", "k_norm", "k_norm_ctx"] {
            tensor(out, format!("{a}.{name}.weight"), bf16, &[HEAD]);
        }
        tensor(
            out,
            format!("{a}.q_proj.weight"),
            q8,
            &[HEADS * HEAD, EMBED],
        );
        for name in ["k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"] {
            tensor(
                out,
                format!("{a}.{name}.weight"),
                q8,
                &[KV_HEADS * HEAD, EMBED],
            );
        }
        tensor(
            out,
            format!("{a}.o_proj.weight"),
            q8,
            &[EMBED, HEADS * HEAD],
        );
    }
    let m = "mtp.markov_head";
    tensor(out, format!("{m}.dflare_norm.weight"), bf16, &[EMBED]);
    tensor(
        out,
        format!("{m}.hidden_out.weight"),
        bf16,
        &[EMBED, MARKOV_RANK],
    );
    tensor(out, format!("{m}.hidden_out.bias"), bf16, &[EMBED]);
    tensor(
        out,
        format!("{m}.markov_w1.weight"),
        bf16,
        &[VOCAB, MARKOV_RANK],
    );
    tensor(
        out,
        format!("{m}.rnn_proj.weight"),
        bf16,
        &[MARKOV_STATE, MARKOV_IN],
    );
    tensor(out, format!("{m}.rnn_proj.bias"), bf16, &[MARKOV_STATE]);
}

pub fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    for il in 0..LAYERS {
        layer(&mut out, il);
    }
    attn_res(&mut out);
    for name in ["model.llm.embed_norm.weight", "model.llm.norm.weight"] {
        exact(&mut out, name.into(), BF16, &[EMBED]);
    }
    exact(
        &mut out,
        "model.llm.unembed.weight".into(),
        Q4_K,
        &[VOCAB, EMBED],
    );
    drafter(&mut out);
    exact(&mut out, EMBEDDING.into(), BF16, &[VOCAB, EMBED]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // tensor-manifest.json: 995 body tensors, 48 x 1,153 expert-part
    // tensors, 84 drafter tensors and the embedding.
    #[test]
    fn layout_counts_match_the_manifest() {
        let all = layouts();
        let drafter = all
            .iter()
            .filter(|s| matches!(s.class, TypeClass::OptionalExact(_)))
            .count();
        let experts = all
            .iter()
            .filter(|s| s.name.contains(".mlp.experts.") || s.name.ends_with(".vq_codebook"))
            .count();
        assert_eq!(drafter, 84);
        assert_eq!(experts, 48 * 1153);
        assert_eq!(all.len() - drafter - experts - 1, 995);
    }
}
