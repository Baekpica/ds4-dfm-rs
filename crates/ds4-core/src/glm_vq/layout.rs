//! Original names and types are retained, including the FP8 MTP experts.
//! Shapes follow the pinned source configuration and GVQ1 release manifest.
use super::{BF16, CODEBOOK, F32, FP8, VQ};
use crate::{LayoutSpec, TypeClass};

const EMBED: u64 = 4096;
const FFN: u64 = 2048;
const DENSE_FFN: u64 = 12288;
const VOCAB: u64 = 154880;
const EXPERTS: u32 = 288;
const TRUNK: u32 = 45;
const VISION: u32 = 24;
const LEADING_DENSE: u32 = 3;
const DSA_PERIOD: u32 = 4;
const HC_STREAMS: u64 = 4;
const HC_MIX: u64 = HC_STREAMS * (HC_STREAMS + 2);
const HEADS: u64 = 64;
const KDA_HEAD: u64 = 128;
const KDA_WIDTH: u64 = HEADS * KDA_HEAD;
const KDA_CONV: u64 = 4;
const Q_RANK: u64 = 1536;
const KV_RANK: u64 = 512;
const DSA_HEAD: u64 = 256;
const DSA_WIDTH: u64 = HEADS * DSA_HEAD;
const INDEX_HEAD: u64 = 128;
const INDEX_POOL: u64 = 4;
const CODEWORDS: u64 = 8192;
const VISION_WIDTH: u64 = 1024;
const VISION_FFN: u64 = 4096;
const MERGER_FFN: u64 = 10240;

fn tensor(out: &mut Vec<LayoutSpec>, name: String, typ: u32, shape: &[u64]) {
    let mut dim = [0; 8];
    for (i, &n) in shape.iter().rev().enumerate() {
        dim[i] = n;
    }
    let class = if q4k_target(&name) {
        TypeClass::GlmVqNative(typ)
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

// Quantize the bandwidth-heavy trunk matrices; state/routing controls and
// the original draft remain independent quality contracts.
pub(super) fn q4k_target(name: &str) -> bool {
    if matches!(
        name,
        "lm_head.weight" | "model.language_model.embed_tokens.weight"
    ) {
        return true;
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((layer, role)) = rest.split_once('.') else {
        return false;
    };
    if !layer.parse::<u32>().is_ok_and(|il| il < TRUNK) {
        return false;
    }
    matches!(
        role,
        "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "self_attn.q_a_proj.weight"
            | "self_attn.q_b_proj.weight"
            | "self_attn.kv_a_proj_with_mqa.weight"
            | "self_attn.kv_b_proj.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
            | "mlp.down_proj.weight"
    ) || q4k_shared(name)
}

/// Trunk shared experts: Q4_K only under the v3 nonexpert recipe.
pub(super) fn q4k_shared(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((layer, role)) = rest.split_once('.') else {
        return false;
    };
    layer.parse::<u32>().is_ok_and(|il| il < TRUNK)
        && matches!(
            role,
            "mlp.shared_experts.gate_proj.weight"
                | "mlp.shared_experts.up_proj.weight"
                | "mlp.shared_experts.down_proj.weight"
        )
}

fn matrix(out: &mut Vec<LayoutSpec>, prefix: &str, typ: u32, rows: u64, cols: u64) {
    tensor(out, format!("{prefix}.weight"), typ, &[rows, cols]);
    if typ == FP8 {
        tensor(
            out,
            format!("{prefix}.weight_scale_inv"),
            F32,
            &[rows / super::SCALE_BLOCK, cols / super::SCALE_BLOCK],
        );
    }
}

fn mlp(out: &mut Vec<LayoutSpec>, prefix: &str, typ: u32, hidden: u64) {
    for name in ["gate_proj", "up_proj"] {
        matrix(out, &format!("{prefix}.{name}"), typ, hidden, EMBED);
    }
    matrix(out, &format!("{prefix}.down_proj"), typ, EMBED, hidden);
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("model.language_model.layers.{il}");
    let mtp = il == TRUNK;
    for name in ["input_layernorm", "post_attention_layernorm"] {
        tensor(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    if !mtp {
        for role in ["attn", "ffn"] {
            tensor(
                out,
                format!("{p}.hc_{role}_fn"),
                BF16,
                &[HC_MIX, HC_STREAMS * EMBED],
            );
            tensor(out, format!("{p}.hc_{role}_base"), F32, &[HC_MIX]);
            tensor(out, format!("{p}.hc_{role}_scale"), F32, &[3]);
        }
    }
    if il < LEADING_DENSE {
        mlp(out, &format!("{p}.mlp"), FP8, DENSE_FFN);
    } else {
        matrix(
            out,
            &format!("{p}.mlp.gate"),
            BF16,
            u64::from(EXPERTS),
            EMBED,
        );
        tensor(
            out,
            format!("{p}.mlp.gate.e_score_correction_bias"),
            F32,
            &[u64::from(EXPERTS)],
        );
        mlp(out, &format!("{p}.mlp.shared_experts"), FP8, FFN);
        for expert in 0..EXPERTS {
            mlp(
                out,
                &format!("{p}.mlp.experts.{expert}"),
                if mtp { FP8 } else { VQ },
                FFN,
            );
        }
        if !mtp {
            // The published model contract is uniform8192 E4M3. Primitive
            // 4096/F16 support does not imply a different model is qualified.
            tensor(
                out,
                format!("glm_vq.layer.{il}.codebook.0"),
                CODEBOOK,
                &[CODEWORDS, super::VECTOR as u64],
            );
        }
    }
    let a = format!("{p}.self_attn");
    if !mtp && il % DSA_PERIOD != DSA_PERIOD - 1 {
        for role in ["q", "k", "v"] {
            matrix(out, &format!("{a}.{role}_proj"), BF16, KDA_WIDTH, EMBED);
            tensor(
                out,
                format!("{a}.{role}_conv1d.weight"),
                BF16,
                &[KDA_WIDTH, 1, KDA_CONV],
            );
        }
        for role in ["f", "g"] {
            matrix(out, &format!("{a}.{role}_a_proj"), BF16, KDA_HEAD, EMBED);
            matrix(
                out,
                &format!("{a}.{role}_b_proj"),
                BF16,
                KDA_WIDTH,
                KDA_HEAD,
            );
        }
        matrix(out, &format!("{a}.b_proj"), BF16, HEADS, EMBED);
        matrix(out, &format!("{a}.o_proj"), BF16, EMBED, KDA_WIDTH);
        tensor(out, format!("{a}.o_norm.weight"), BF16, &[KDA_HEAD]);
        tensor(out, format!("{a}.A_log"), F32, &[HEADS]);
        tensor(out, format!("{a}.dt_bias"), F32, &[KDA_WIDTH]);
    } else {
        matrix(out, &format!("{a}.q_a_proj"), FP8, Q_RANK, EMBED);
        matrix(out, &format!("{a}.q_b_proj"), FP8, DSA_WIDTH, Q_RANK);
        matrix(out, &format!("{a}.kv_a_proj_with_mqa"), FP8, KV_RANK, EMBED);
        matrix(out, &format!("{a}.kv_b_proj"), BF16, 2 * DSA_WIDTH, KV_RANK);
        matrix(out, &format!("{a}.o_proj"), FP8, EMBED, DSA_WIDTH);
        tensor(out, format!("{a}.q_a_layernorm.weight"), BF16, &[Q_RANK]);
        tensor(out, format!("{a}.kv_a_layernorm.weight"), BF16, &[KV_RANK]);
        for (role, rows, cols) in [
            ("wk", INDEX_HEAD, EMBED),
            ("wq_b", 32 * INDEX_HEAD, Q_RANK),
            ("weights_proj", 32, EMBED),
        ] {
            matrix(out, &format!("{a}.indexer.{role}"), BF16, rows, cols);
        }
        for suffix in ["weight", "bias"] {
            tensor(
                out,
                format!("{a}.indexer.k_norm.{suffix}"),
                BF16,
                &[KDA_HEAD],
            );
        }
        tensor(
            out,
            format!("{a}.indexer.index_kpool_compress_ape"),
            BF16,
            &[INDEX_POOL, INDEX_HEAD],
        );
        tensor(
            out,
            format!("{a}.indexer.index_kpool_compress_gate"),
            BF16,
            &[INDEX_HEAD, EMBED],
        );
    }
    if mtp {
        matrix(out, &format!("{p}.eh_proj"), BF16, EMBED, 2 * EMBED);
        for role in ["enorm", "hnorm", "shared_head.norm"] {
            tensor(out, format!("{p}.{role}.weight"), BF16, &[EMBED]);
        }
    }
}

fn vision(out: &mut Vec<LayoutSpec>) {
    let p = "model.visual";
    // GGUF v3 has at most four dimensions; this one source rank-5 tensor
    // stores a flat row. The loader checks its original logical shape too.
    matrix(
        out,
        &format!("{p}.patch_embed.proj"),
        BF16,
        VISION_WIDTH,
        3 * 2 * 14 * 14,
    );
    tensor(
        out,
        format!("{p}.patch_embed.proj.bias"),
        BF16,
        &[VISION_WIDTH],
    );
    tensor(
        out,
        format!("{p}.post_layernorm.weight"),
        BF16,
        &[VISION_WIDTH],
    );
    tensor(
        out,
        format!("{p}.downsample.weight"),
        BF16,
        &[EMBED, VISION_WIDTH, 2, 2],
    );
    tensor(out, format!("{p}.downsample.bias"), BF16, &[EMBED]);
    matrix(out, &format!("{p}.merger.proj"), BF16, EMBED, EMBED);
    mlp(out, &format!("{p}.merger"), BF16, MERGER_FFN);
    for suffix in ["weight", "bias"] {
        tensor(
            out,
            format!("{p}.merger.post_projection_norm.{suffix}"),
            BF16,
            &[EMBED],
        );
    }
    for il in 0..VISION {
        let b = format!("{p}.blocks.{il}");
        for role in ["norm1", "norm2"] {
            tensor(out, format!("{b}.{role}.weight"), BF16, &[VISION_WIDTH]);
        }
        for role in ["q", "k"] {
            tensor(out, format!("{b}.attn.{role}_norm.weight"), BF16, &[64]);
        }
        for (role, rows, cols) in [
            ("attn.qkv", 3 * VISION_WIDTH, VISION_WIDTH),
            ("attn.proj", VISION_WIDTH, VISION_WIDTH),
            ("mlp.gate_proj", VISION_FFN, VISION_WIDTH),
            ("mlp.up_proj", VISION_FFN, VISION_WIDTH),
            ("mlp.down_proj", VISION_WIDTH, VISION_FFN),
        ] {
            matrix(out, &format!("{b}.{role}"), BF16, rows, cols);
            tensor(out, format!("{b}.{role}.bias"), BF16, &[rows]);
        }
    }
}

pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    matrix(
        &mut out,
        "model.language_model.embed_tokens",
        BF16,
        VOCAB,
        EMBED,
    );
    matrix(&mut out, "lm_head", BF16, VOCAB, EMBED);
    tensor(
        &mut out,
        "model.language_model.norm.weight".into(),
        BF16,
        &[EMBED],
    );
    for il in 0..=TRUNK {
        layer(&mut out, il);
    }
    vision(&mut out);
    out
}
