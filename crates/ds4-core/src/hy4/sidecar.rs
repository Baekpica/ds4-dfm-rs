//! A sidecar is opened only after its source, layout and dimensions match the shipped model.
use crate::{Error, GgufFile, Result, TensorInventory};
use std::path::Path;

pub const VISION_FILE: &str = "mmproj-Hy4-flash-BF16.gguf";

fn invalid(message: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: format!("Hy4 sidecar: {message}"),
    }
}

pub fn vision(path: &Path) -> Result<()> {
    let g = GgufFile::open(path).map_err(invalid)?;
    if g.get_string("general.architecture") != Some(b"clip")
        || g.get_string("general.source.revision") != Some(super::SOURCE)
        || g.get_string("clip.projector_type") != Some(b"hy_v4_flash")
        || g.get_string("hy4_mxq.part") != Some(b"vision")
        || g.split_count() > 1
        || g.get_f32_compat("clip.vision.attention.layer_norm_epsilon") != Some(1e-5)
    {
        return Err(invalid("expected Hy4 CLIP projector"));
    }
    for (key, v) in [
        ("projection_dim", 3072),
        ("image_size", 2048),
        ("patch_size", 16),
        ("embedding_length", 1152),
        ("feed_forward_length", 4304),
        ("block_count", 27),
        ("attention.head_count", 16),
        ("spatial_merge_size", 2),
    ] {
        if g.get_u32(&format!("clip.vision.{key}")) != Some(v) {
            return Err(invalid(key));
        }
    }
    let inv = TensorInventory::from_file(path, &g).map_err(invalid)?;
    if inv.tensors.len() != 442 {
        return Err(invalid("vision tensor count"));
    }
    let check = |name: &str, ty, dims: &[u64]| -> Result<()> {
        let t = inv.find(name).ok_or_else(|| invalid(name))?;
        if t.typ != ty || t.ndim as usize != dims.len() || &t.dim[..dims.len()] != dims {
            return Err(invalid(name));
        }
        Ok(())
    };
    for (name, ty, dims) in [
        ("v.patch_embd.weight", 30, &[16, 16, 3, 1152][..]),
        ("v.patch_embd.bias", 0, &[1152]),
        ("v.position_embd.weight", 30, &[1152, 16384]),
        ("v.position_embd.cls.weight", 30, &[1152, 1]),
        ("mm.pre_norm.weight", 0, &[1152]),
        ("mm.0.weight", 30, &[2, 2, 1152, 4608]),
        ("mm.0.bias", 0, &[4608]),
        ("mm.2.weight", 30, &[1, 1, 4608, 3072]),
        ("mm.2.bias", 0, &[3072]),
        ("v.image_newline", 30, &[3072]),
    ] {
        check(name, ty, dims)?;
    }
    for i in 0..27 {
        for (name, input, output) in [
            ("attn_q", 1152, 1152),
            ("attn_k", 1152, 1152),
            ("attn_v", 1152, 1152),
            ("attn_out", 1152, 1152),
            ("ffn_up", 1152, 4304),
            ("ffn_down", 4304, 1152),
        ] {
            check(&format!("v.blk.{i}.{name}.weight"), 30, &[input, output])?;
            check(&format!("v.blk.{i}.{name}.bias"), 0, &[output])?;
        }
        for name in ["ln1", "ln2"] {
            for suffix in ["weight", "bias"] {
                check(&format!("v.blk.{i}.{name}.{suffix}"), 0, &[1152])?;
            }
        }
    }
    Ok(())
}

pub fn draft(path: &Path) -> Result<()> {
    let g = GgufFile::open(path).map_err(invalid)?;
    if g.get_string("general.architecture") != Some(super::ARCH)
        || g.get_string("general.source.revision") != Some(super::SOURCE)
        || g.get_string("hy4_mxq.part") != Some(b"drafter")
        || g.split_count() > 1
        || g.n_tensors != 92
        || g.get_string("hy_v4_flash.dfly.fusion") != Some(b"slot0")
        || g.get_bool("hy_v4_flash.dfly.hidden_correction") != Some(true)
    {
        return Err(invalid("expected Hy4 DFly2 Q8_0 or BF16 checkpoint"));
    }
    for (k, n) in [
        ("block_count", 5),
        ("target_index_topk", 2048),
        ("mask_token_id", 120009),
        ("num_speculative_tokens", 5),
        ("attention.head_count", 32),
    ] {
        if g.get_u32(&format!("hy_v4_flash.dfly.{k}")) != Some(n) {
            return Err(invalid(k));
        }
    }
    for k in ["target_kv_layers", "target_index_layers"] {
        if g.get_array(&format!("hy_v4_flash.dfly.{k}"))
            .and_then(|a| g.array_le_u32s(&a).ok())
            .as_deref()
            != Some(&[25, 29, 33, 37, 41])
        {
            return Err(invalid(k));
        }
    }
    let inv = TensorInventory::from_file(path, &g).map_err(invalid)?;
    let quant = inv
        .find("dfly.blk.0.attn_q_b.weight")
        .ok_or_else(|| invalid("query projection"))?
        .typ;
    if !matches!(quant, 8 | 30) {
        return Err(invalid("DFly projection dtype"));
    }
    let check = |name: &str, ty, dims: &[u64]| -> Result<()> {
        if !inv.find(name).is_some_and(|t| {
            t.typ == ty && t.ndim as usize == dims.len() && &t.dim[..dims.len()] == dims
        }) {
            return Err(invalid(name));
        }
        Ok(())
    };
    for i in 0..5 {
        for (n, t, d) in [
            ("attn_norm", 0, &[3072][..]),
            ("attn_q_a", 30, &[3072, 1024]),
            ("attn_q_a_norm", 0, &[1024]),
            ("attn_q_b", quant, &[1024, 8192]),
            ("attn_kv_a_mqa", 30, &[3072, 576]),
            ("attn_kv_a_norm", 0, &[512]),
            ("attn_k_b", 30, &[192, 512, 32]),
            ("attn_v_b", 30, &[512, 256, 32]),
            ("attn_output", quant, &[8192, 3072]),
            ("attn_gate", quant, &[3072, 8192]),
            ("attn_sinks", 0, &[32]),
            ("post_attention_norm", 0, &[3072]),
            ("ffn_norm", 0, &[3072]),
            ("ffn_gate", quant, &[3072, 9216]),
            ("ffn_up", quant, &[3072, 9216]),
            ("ffn_down", quant, &[9216, 3072]),
        ] {
            check(&format!("dfly.blk.{i}.{n}.weight"), t, d)?;
        }
    }
    for (n, t, d) in [
        ("fusion.eh_proj.weight", 30, &[6144, 3072][..]),
        ("fusion.enorm.weight", 0, &[3072]),
        ("fusion.hnorm.weight", 0, &[3072]),
        ("fusion.gate.weight", 0, &[3072]),
        ("correction.gate_proj.weight", 30, &[6144, 3072]),
        ("correction.up_proj.weight", 30, &[6144, 3072]),
        ("correction.down_proj.weight", 30, &[3072, 3072]),
        ("correction.hidden_norm.weight", 0, &[3072]),
        ("correction.embed_norm.weight", 0, &[3072]),
        ("output_norm.weight", 0, &[3072]),
        ("conf_proj.weight", 30, &[3072, 1]),
        ("conf_proj.bias", 0, &[1]),
    ] {
        check(&format!("dfly.{n}"), t, d)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "set HY4_MODEL_PATH to a release with both DFly2 files"]
    fn released_dfly_variants() {
        let root = std::path::PathBuf::from(std::env::var_os("HY4_MODEL_PATH").unwrap());
        for f in ["dfly2-bf16.gguf", "dfly2-q8_0.gguf"] {
            super::draft(&root.join(f)).unwrap();
        }
    }
}
