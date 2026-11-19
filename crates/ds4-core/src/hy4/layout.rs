//! GGUF dimensions (input width first) of the MQ-Q5-SSD-ENGRAM-FP8 recipe.
use crate::{LayoutSpec, TypeClass};

const F32: u32 = 0;
const BF16: u32 = 30;
const Q8_0: u32 = 8;
const Q4_K: u32 = 12;
const Q5_K: u32 = 13;

pub(super) fn sparse(layer: u32) -> bool {
    layer == 0 || layer % 4 == 1
}

pub(super) fn engram(layer: u32) -> bool {
    matches!(layer, 1 | 11 | 21 | 31)
}

fn tensor(name: String, ty: u32, dims: &[u64]) -> LayoutSpec {
    let mut dim = [0; 8];
    dim[..dims.len()].copy_from_slice(dims);
    LayoutSpec {
        name,
        class: TypeClass::Exact(ty),
        ndim: dims.len() as u32,
        dim,
    }
}

pub(crate) fn layouts() -> Vec<LayoutSpec> {
    let mut out = vec![
        tensor("token_embd.weight".into(), BF16, &[3072, 120832]),
        tensor("output.weight".into(), BF16, &[3072, 120832]),
        tensor("output_norm.weight".into(), F32, &[3072]),
        tensor("output_hc_fn.weight".into(), F32, &[12288, 4]),
        tensor("output_hc_scale.weight".into(), F32, &[1]),
        tensor("output_hc_base.weight".into(), F32, &[4]),
    ];
    for i in 0..42 {
        let heads = if sparse(i) { 32 } else { 64 };
        let projection = if i == 0 { BF16 } else { Q8_0 };
        let mut add = |name: &str, ty, dims: &[u64]| {
            out.push(tensor(format!("blk.{i}.{name}"), ty, dims));
        };
        for name in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            add(&format!("{name}.weight"), F32, &[3072]);
        }
        add("attn_q_a.weight", BF16, &[3072, 1024]);
        add("attn_q_a_norm.weight", F32, &[1024]);
        add("attn_q_b.weight", projection, &[1024, heads * 256]);
        add("attn_kv_a_mqa.weight", BF16, &[3072, 576]);
        add("attn_kv_a_norm.weight", F32, &[512]);
        add("attn_k_b.weight", BF16, &[192, 512, heads]);
        add("attn_v_b.weight", BF16, &[512, 256, heads]);
        add("attn_output.weight", projection, &[heads * 256, 3072]);
        add("attn_gate.weight", projection, &[3072, heads * 256]);
        add("attn_sinks.weight", F32, &[heads]);
        for sub in ["attn", "ffn"] {
            add(&format!("hc_{sub}_fn.weight"), F32, &[12288, 8]);
            add(&format!("hc_{sub}_base.weight"), F32, &[8]);
            add(&format!("hc_{sub}_scale.weight"), F32, &[2]);
        }
        if sparse(i) {
            add("indexer.attn_k.weight", BF16, &[3072, 128]);
            add("indexer.k_norm.weight", F32, &[128]);
            add("indexer.attn_q_b.weight", BF16, &[1024, 2048]);
            add("indexer.proj.weight", F32, &[3072, 16]);
            for name in ["value", "gate"] {
                add(
                    &format!("indexer_compressor_{name}.weight"),
                    F32,
                    &[128, 128],
                );
            }
            add("indexer_compressor_ape.weight", F32, &[128, 4]);
            add("indexer_compressor_norm.weight", F32, &[128]);
        }
        if engram(i) {
            add("engram_kv.weight", BF16, &[1536, 15360]);
            add("engram_kv_b.weight", F32, &[3072, 5]);
            for name in ["gate", "norm_key", "norm_query", "hc_scale"] {
                add(&format!("engram_{name}.weight"), F32, &[3072, 4]);
            }
            add("engram_lookup_norm.weight", F32, &[96, 16]);
            add("engram_pre_gamma.weight", F32, &[3072]);
        }
        if i == 0 {
            add("ffn_gate.weight", BF16, &[3072, 13824]);
            add("ffn_up.weight", BF16, &[3072, 13824]);
            add("ffn_down.weight", BF16, &[13824, 3072]);
        } else {
            let gate = if (7..=29).contains(&i) { Q5_K } else { Q4_K };
            add("ffn_gate_inp.weight", F32, &[3072, 512]);
            add("exp_probs_b.bias", F32, &[512]);
            add("ffn_routed_down.weight", BF16, &[3072, 1536]);
            add("ffn_routed_up.weight", BF16, &[1536, 3072]);
            add("ffn_gate_shexp.weight", BF16, &[3072, 1536]);
            add("ffn_up_shexp.weight", BF16, &[3072, 1536]);
            add("ffn_down_shexp.weight", BF16, &[1536, 3072]);
            add("ffn_gate_exps.weight", gate, &[1536, 1536, 512]);
            add("ffn_up_exps.weight", gate, &[1536, 1536, 512]);
            add("ffn_down_exps.weight", Q5_K, &[1536, 1536, 512]);
        }
    }
    out
}
