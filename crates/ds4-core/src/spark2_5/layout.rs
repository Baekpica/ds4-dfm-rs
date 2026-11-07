//! Generated from the release's runtime pack (`vq-gguf runtime`): every
//! stored tensor with the artifact's own name, type word and dimensions.
//! Do not edit by hand; regenerate with
//!   cargo run -q -p ds4-core --example layout_from_pack -- <runtime/layout.json>
//! The dimensions are GGUF order (innermost first), verbatim from the pack.
use crate::{LayoutSpec, TypeClass};

/// The stored type words this table uses.
mod ty {
    pub(super) const F32: u32 = 0;
    pub(super) const BF16: u32 = 30;
    pub(super) const GVBC: u32 = 0x4756_4342;
    pub(super) const GVQ8: u32 = 0x4756_5138;
    pub(super) const GVR8: u32 = 0x4756_5238;
}

/// The token table the head ties to: the artifact carries no `lm_head`.
pub const EMBEDDING: &str = "model.embedding.weight";

fn spec(out: &mut Vec<LayoutSpec>, name: &str, typ: u32, dims: &[u64]) {
    let mut dim = [0u64; 8];
    dim[..dims.len()].copy_from_slice(dims);
    out.push(LayoutSpec {
        name: name.to_string(),
        class: TypeClass::Exact(typ),
        ndim: dims.len() as u32,
        dim,
    });
}

/// One entry per stored tensor, in pack order: the embedding part
/// first, the resident trunk second.
pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::with_capacity(830);
    // embedding part: spark-x2.6-4b-vq8.embedding.gguf
    spec(
        &mut out,
        "model.embedding.weight",
        ty::BF16,
        &[2560, 131072],
    );
    // trunk part: spark-x2.6-4b-vq8.gguf
    spec(
        &mut out,
        "model.layers.0.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.0.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.0.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.0.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.1.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.1.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.1.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.10.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.10.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.10.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.11.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.11.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.11.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.12.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.12.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.12.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.13.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.13.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.13.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.14.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.14.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.14.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.15.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.15.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.15.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.16.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.16.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.16.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.17.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.17.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.17.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.18.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.18.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.18.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.19.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.19.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.19.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.2.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.2.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.2.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.20.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.20.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.20.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.21.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.21.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.21.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.22.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.22.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.22.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.23.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.23.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.23.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.24.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.24.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.24.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.25.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.25.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.25.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.26.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.26.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.26.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.27.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.27.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.27.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.28.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.28.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.28.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.29.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.29.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.29.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.3.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.3.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.3.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.30.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.30.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.30.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.31.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.31.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.31.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.32.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.32.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.32.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.33.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.33.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.33.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.34.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.34.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.34.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.35.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.35.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.35.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.4.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.4.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.4.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.5.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.5.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.5.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.6.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.6.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.6.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.7.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.7.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.7.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.8.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.8.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.8.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.9.input_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.down_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.down_proj.weight",
        ty::GVQ8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.down_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.down_proj.weight.res",
        ty::GVR8,
        &[10240, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.gate_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.gate_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.gate_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.gate_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.up_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.up_proj.weight",
        ty::GVQ8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.up_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.9.mlp.up_proj.weight.res",
        ty::GVR8,
        &[2560, 10240],
    );
    spec(
        &mut out,
        "model.layers.9.post_attention_layernorm.weight",
        ty::F32,
        &[1, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.out_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.out_proj.weight",
        ty::GVQ8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.out_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.out_proj.weight.res",
        ty::GVR8,
        &[4096, 2560],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.q_k_v_proj.weight.codebook",
        ty::GVBC,
        &[8, 8192],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.q_k_v_proj.weight",
        ty::GVQ8,
        &[2560, 6144],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.q_k_v_proj.weight.vq_residual_book",
        ty::GVBC,
        &[8, 256],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.q_k_v_proj.weight.res",
        ty::GVR8,
        &[2560, 6144],
    );
    spec(&mut out, "model.norm.weight", ty::F32, &[1, 2560]);
    spec(
        &mut out,
        "model.layers.0.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.1.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.10.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.11.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.12.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.13.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.14.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.15.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.16.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.17.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.18.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.19.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.2.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.20.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.21.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.22.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.23.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.24.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.25.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.26.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.27.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.28.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.29.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.3.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.30.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.31.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.32.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.33.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.34.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.35.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.4.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.5.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.6.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.7.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.8.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    spec(
        &mut out,
        "model.layers.9.self_attn.g_proj.weight",
        ty::BF16,
        &[2560, 16],
    );
    out
}
