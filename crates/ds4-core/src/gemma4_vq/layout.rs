//! Original HF names and the artifact's stored types, transcribed from the
//! emitted file (`vq-gguf list`), not from memory: Q4_K attention and dense
//! FFN projections, Q8_0 for the dense down-projection whose 2112 columns are
//! not a multiple of the Q4_K block of 256, BF16 for every norm, the layer
//! scalar, the router and its two scales, the tied embedding and the final
//! norm; per-layer 13-bit GVQ1 experts (128 x gate, up, down) sharing one
//! K8192 book per stack.
//!
//! Geometry: the checkpoint's `config.json` (hidden 2816, 30 layers, 16
//! heads, head_dim 256 sliding / 512 global, 8 KV heads sliding / 2 global,
//! dense FFN 2112, expert FFN 704, 128 experts, vocab 262144, tied
//! embeddings, `attention_k_eq_v`) and its safetensors headers for the
//! per-layer shapes.
use super::{BF16, CODEBOOK, Q4_K, Q8_0, RESIDUAL, RESIDUAL_WORDS, VECTOR, VQ, WORDS};
use crate::{LayoutSpec, TypeClass};

/// The token table the head ties to: there is no `lm_head` in the artifact.
pub const EMBEDDING: &str = "model.embed_tokens.weight";

const EMBED: u64 = 2816;
const VOCAB: u64 = 262_144;
const LAYERS: u32 = 30;
const HEADS: u64 = 16;
const HEAD_DIM: u64 = 256;
const GLOBAL_HEAD_DIM: u64 = 512;
const KV_HEADS: u64 = 8;
const GLOBAL_KV_HEADS: u64 = 2;
const DENSE_FFN: u64 = 2112;
const EXPERTS: u32 = 128;
const FFN: u64 = 704;
/// config.json `layer_types`: full attention on these five layers, sliding
/// window 1024 on the other 25.
const FULL: [u32; 5] = [5, 11, 17, 23, 29];

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

/// q/k/v/o and the two per-head norms of one attention block. A full layer
/// has no `v_proj`: config.json `attention_k_eq_v` makes value equal key.
fn attention(out: &mut Vec<LayoutSpec>, prefix: &str, full: bool) {
    let dim = if full { GLOBAL_HEAD_DIM } else { HEAD_DIM };
    let kv = if full { GLOBAL_KV_HEADS } else { KV_HEADS };
    exact(
        out,
        format!("{prefix}.q_proj.weight"),
        Q4_K,
        &[HEADS * dim, EMBED],
    );
    exact(
        out,
        format!("{prefix}.k_proj.weight"),
        Q4_K,
        &[kv * dim, EMBED],
    );
    if !full {
        exact(
            out,
            format!("{prefix}.v_proj.weight"),
            Q4_K,
            &[kv * dim, EMBED],
        );
    }
    exact(
        out,
        format!("{prefix}.o_proj.weight"),
        Q4_K,
        &[EMBED, HEADS * dim],
    );
    for name in ["q_norm", "k_norm"] {
        exact(out, format!("{prefix}.{name}.weight"), BF16, &[dim]);
    }
}

/// The dense FFN every layer keeps beside its experts.
fn dense_mlp(out: &mut Vec<LayoutSpec>, prefix: &str) {
    for name in ["gate_proj", "up_proj"] {
        exact(
            out,
            format!("{prefix}.{name}.weight"),
            Q4_K,
            &[DENSE_FFN, EMBED],
        );
    }
    exact(
        out,
        format!("{prefix}.down_proj.weight"),
        Q8_0,
        &[EMBED, DENSE_FFN],
    );
}

/// One layer's routed experts: a book per stack and one GVQ1 matrix per
/// expert. The checkpoint stores gate and up fused in one stack, and the
/// split gives that stack one book for both halves.
///
/// `residual` is the R13+8 build: a second 256-entry book per stack and one
/// GVR8 index plane per matrix, named `<matrix>.res` (crates/vq-gguf/src/gvr1.rs).
fn experts(out: &mut Vec<LayoutSpec>, prefix: &str, residual: bool) {
    for stack in ["down_proj", "gate_up_proj"] {
        exact(
            out,
            format!("{prefix}.{stack}.vq_codebook"),
            CODEBOOK,
            &[u64::from(WORDS), u64::from(VECTOR)],
        );
        if residual {
            exact(
                out,
                format!("{prefix}.{stack}.vq_residual_book"),
                CODEBOOK,
                &[u64::from(RESIDUAL_WORDS), u64::from(VECTOR)],
            );
        }
    }
    for expert in 0..EXPERTS {
        for (name, shape) in [
            ("gate_proj", [FFN, EMBED]),
            ("up_proj", [FFN, EMBED]),
            ("down_proj", [EMBED, FFN]),
        ] {
            let matrix = format!("{prefix}.{expert}.{name}");
            exact(out, matrix.clone(), VQ, &shape);
            if residual {
                exact(out, format!("{matrix}.res"), RESIDUAL, &shape);
            }
        }
    }
}

fn layer(out: &mut Vec<LayoutSpec>, il: u32, residual: bool) {
    let p = format!("model.layers.{il}");
    for name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ] {
        exact(out, format!("{p}.{name}.weight"), BF16, &[EMBED]);
    }
    exact(out, format!("{p}.layer_scalar"), BF16, &[1]);
    attention(out, &format!("{p}.self_attn"), FULL.contains(&il));
    dense_mlp(out, &format!("{p}.mlp"));
    // The checkpoint's router: a BF16 projection with an input scale and a
    // per-expert scale. The family vocabulary names the projection
    // `mlp.gate`, the scales beside it.
    exact(
        out,
        format!("{p}.mlp.gate.weight"),
        BF16,
        &[u64::from(EXPERTS), EMBED],
    );
    exact(out, format!("{p}.mlp.gate.scale"), BF16, &[EMBED]);
    exact(
        out,
        format!("{p}.mlp.gate.per_expert"),
        BF16,
        &[u64::from(EXPERTS)],
    );
    experts(out, &format!("{p}.experts"), residual);
}

/// The plain 13-bit family artifact (format version 1).
pub(super) fn layouts() -> Vec<LayoutSpec> {
    layouts_with(false)
}

/// The R13+8 artifact (format version 2): the same set plus the residual
/// books and index planes.
pub(super) fn layouts_with(residual: bool) -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    exact(&mut out, EMBEDDING.into(), BF16, &[VOCAB, EMBED]);
    exact(&mut out, "model.norm.weight".into(), BF16, &[EMBED]);
    for il in 0..LAYERS {
        layer(&mut out, il, residual);
    }
    out
}

/// The MTP drafter (`gemma4_assistant`), which the family ships as its own
/// part: four dense layers at hidden 1024 with no experts and no k/v
/// projection (it borrows the trunk's), plus the projections into and out of
/// the backbone width. Transcribed from the checkpoint's own tensor table.
const MTP_LAYERS: u32 = 4;
const MTP_EMBED: u64 = 1024;
const MTP_FFN: u64 = 8192;
const MTP_FULL: u32 = 3;

pub(super) fn mtp_layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    exact(
        &mut out,
        "mtp.model.embed_tokens.weight".into(),
        BF16,
        &[VOCAB, MTP_EMBED],
    );
    for il in 0..MTP_LAYERS {
        let p = format!("mtp.model.layers.{il}");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            exact(&mut out, format!("{p}.{name}.weight"), BF16, &[MTP_EMBED]);
        }
        exact(&mut out, format!("{p}.layer_scalar"), BF16, &[1]);
        for name in ["gate_proj", "up_proj"] {
            exact(
                &mut out,
                format!("{p}.mlp.{name}.weight"),
                Q4_K,
                &[MTP_FFN, MTP_EMBED],
            );
        }
        exact(
            &mut out,
            format!("{p}.mlp.down_proj.weight"),
            Q4_K,
            &[MTP_EMBED, MTP_FFN],
        );
        // The full-attention layer rotates and projects at 512; the other
        // three are sliding at 256 (config.json `layer_types`).
        let dim = if il == MTP_FULL {
            GLOBAL_HEAD_DIM
        } else {
            HEAD_DIM
        };
        exact(
            &mut out,
            format!("{p}.self_attn.q_proj.weight"),
            Q4_K,
            &[HEADS * dim, MTP_EMBED],
        );
        exact(
            &mut out,
            format!("{p}.self_attn.o_proj.weight"),
            Q4_K,
            &[MTP_EMBED, HEADS * dim],
        );
        exact(
            &mut out,
            format!("{p}.self_attn.q_norm.weight"),
            BF16,
            &[dim],
        );
    }
    exact(&mut out, "mtp.model.norm.weight".into(), BF16, &[MTP_EMBED]);
    // Into the drafter from the trunk's hidden state and token row, and back
    // out to the backbone width.
    exact(
        &mut out,
        "mtp.pre_projection.weight".into(),
        Q4_K,
        &[MTP_EMBED, 2 * EMBED],
    );
    exact(
        &mut out,
        "mtp.post_projection.weight".into(),
        Q4_K,
        &[EMBED, MTP_EMBED],
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(all: &[LayoutSpec], typ: u32) -> usize {
        all.iter()
            .filter(|s| s.class == TypeClass::Exact(typ))
            .count()
    }

    /// The counts are the emitted artifact's own (`vq-gguf list` over
    /// gemma-4-26b-a4b-vq8.gguf): 12177 tensors, 11520 of them GVQ1.
    #[test]
    fn layouts_cover_the_artifact_tensor_counts() {
        let all = layouts();
        assert_eq!(all.len(), 12_177);
        assert_eq!(of(&all, VQ), 11_520);
        assert_eq!(of(&all, CODEBOOK), 60);
        assert_eq!(of(&all, Q4_K), 175);
        assert_eq!(of(&all, Q8_0), 30);
        assert_eq!(of(&all, BF16), 392);
        let mut names: Vec<_> = all.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
    }

    /// The R13+8 artifact adds one 256-entry residual book per stack and one
    /// GVR8 plane per expert matrix, and nothing else.
    #[test]
    fn residual_layouts_cover_the_r8_artifact_counts() {
        let all = layouts_with(true);
        assert_eq!(all.len(), 23_757);
        assert_eq!(of(&all, VQ), 11_520);
        assert_eq!(of(&all, RESIDUAL), 11_520);
        assert_eq!(of(&all, CODEBOOK), 120);
        let res_book = all
            .iter()
            .find(|s| s.name == "model.layers.0.experts.down_proj.vq_residual_book")
            .expect("residual book");
        assert_eq!(res_book.dim[..2], [8, 256]);
        let plane = all
            .iter()
            .find(|s| s.name == "model.layers.0.experts.0.down_proj.res")
            .expect("residual plane");
        assert_eq!(plane.dim[..2], [704, 2816]);
    }

    /// Every expert matrix is 704 x 2816 in the artifact's column-major
    /// order, and the down projection is the transposed pair.
    #[test]
    fn expert_shapes_match_the_stack_split() {
        let all = layouts();
        let find = |name: &str| {
            all.iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("{name} missing"))
        };
        assert_eq!(
            find("model.layers.29.experts.127.gate_proj").dim[..2],
            [2816, 704]
        );
        assert_eq!(
            find("model.layers.29.experts.127.down_proj").dim[..2],
            [704, 2816]
        );
        assert_eq!(
            find("model.layers.5.self_attn.o_proj.weight").dim[..2],
            [8192, 2816]
        );
        assert_eq!(
            find("model.layers.4.self_attn.o_proj.weight").dim[..2],
            [4096, 2816]
        );
    }

    /// The drafter's own tensor set, transcribed from the assistant
    /// checkpoint's safetensors header: 48 tensors, 22 Q4_K and 26 BF16, no
    /// experts and no k/v projection.
    #[test]
    fn mtp_layouts_cover_the_drafter() {
        let all = mtp_layouts();
        assert_eq!(all.len(), 48);
        assert_eq!(of(&all, Q4_K), 22);
        assert_eq!(of(&all, BF16), 26);
        assert!(all.iter().all(|s| s.name.starts_with("mtp.")));
        let mut names: Vec<_> = all.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
    }

    /// Binds the layout to a real artifact when the operator offers one:
    /// `GEMMA4_VQ_INVENTORY=<vq-gguf list output> cargo test -p ds4-core
    /// --lib gemma4`. Every name, its stored type and its dimensions must
    /// agree, so a change to either side fails here instead of at load.
    #[test]
    fn layouts_match_an_emitted_artifact() {
        let Ok(path) = std::env::var("GEMMA4_VQ_INVENTORY") else {
            return;
        };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut want: Vec<(String, u32, Vec<u64>)> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut f = l.split_whitespace();
                let name = f.next().unwrap().to_string();
                let typ = f.next().unwrap().parse().unwrap();
                let dims = f.map(|d| d.parse().unwrap()).collect();
                (name, typ, dims)
            })
            .collect();
        let residual = want.iter().any(|(_, typ, _)| *typ == RESIDUAL);
        let mut have: Vec<(String, u32, Vec<u64>)> = layouts_with(residual)
            .into_iter()
            .map(|s| {
                let TypeClass::Exact(typ) = s.class else {
                    panic!("{} is {:?}", s.name, s.class)
                };
                (s.name, typ, s.dim[..s.ndim as usize].to_vec())
            })
            .collect();
        want.sort();
        have.sort();
        assert_eq!(have.len(), want.len(), "tensor count");
        for (h, w) in have.iter().zip(&want) {
            assert_eq!((&h.0, h.1, &h.2), (&w.0, w.1, &w.2));
        }
    }
}
