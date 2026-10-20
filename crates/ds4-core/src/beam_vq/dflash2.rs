//! The DFlash2 drafter file (`--mtp`): tools/beam_dflash2.py's conversion of
//! reflection/Beam-1.0-501B-A23B-DFlash2. The native bind only looks tensors
//! up, so this checks the whole contract first: the trained source, every
//! config value the graph hard-codes, and each tensor's type and dims.

use std::path::Path;

use crate::gguf::GgufFile;
use crate::tensors::TensorInventory;

const SOURCE: &[u8] = b"reflection/Beam-1.0-501B-A23B-DFlash2";
/// model.safetensors of revision 163b276be744c8ce2b8385ea85de54edb3088211.
const SOURCE_SHA256: &[u8] = b"34b99ca539b1d595d0db0edf072158173d89411eca0e6b1dd2c2601437352bf4";
const F32: u32 = 0;
const Q8_0: u32 = 8;
const BF16: u32 = 30;
const HIDDEN: u64 = 5120;
const KV: u64 = 1024;
const FF: u64 = 17408;
const DYN: u64 = 1280;
const RANK: u64 = 256;
const VOCAB: u64 = 151936;

const U32S: [(&str, u32); 14] = [
    ("block_count", 5),
    ("embedding_length", 5120),
    ("feed_forward_length", 17408),
    ("attention.head_count", 40),
    ("attention.head_count_kv", 8),
    ("attention.key_length", 128),
    ("attention.sliding_window", 4096),
    ("vocab_size", 151936),
    ("block_size", 16),
    ("mask_token_id", 151669),
    ("conv_kernel_size", 2),
    ("conv_group_size", 16),
    ("selector_rank", 256),
    ("selector_top_k", 16),
];

fn tensors() -> Vec<(String, u32, Vec<u64>)> {
    let mut out = vec![
        ("hidden_norm.weight".into(), F32, vec![HIDDEN]),
        ("norm.weight".into(), F32, vec![HIDDEN]),
        (
            "candidate_selector.hidden_projection.weight".into(),
            Q8_0,
            vec![HIDDEN, RANK],
        ),
        (
            "candidate_selector.predecessor_codebook".into(),
            BF16,
            vec![RANK, VOCAB],
        ),
        (
            "candidate_selector.successor_codebook".into(),
            BF16,
            vec![RANK, VOCAB],
        ),
    ];
    out.extend((0..5).map(|k| (format!("fc.{k}.weight"), Q8_0, vec![HIDDEN, HIDDEN])));
    for il in 0..5 {
        let layer: [(&str, u32, Vec<u64>); 15] = [
            ("input_layernorm.weight", F32, vec![HIDDEN]),
            ("post_attention_layernorm.weight", F32, vec![HIDDEN]),
            ("self_attn.q_proj.weight", Q8_0, vec![HIDDEN, HIDDEN]),
            ("self_attn.k_proj.weight", Q8_0, vec![HIDDEN, KV]),
            ("self_attn.v_proj.weight", Q8_0, vec![HIDDEN, KV]),
            ("self_attn.o_proj.weight", Q8_0, vec![HIDDEN, HIDDEN]),
            ("self_attn.q_norm.weight", F32, vec![128]),
            ("self_attn.k_norm.weight", F32, vec![128]),
            ("mlp.gate_proj.weight", Q8_0, vec![HIDDEN, FF]),
            ("mlp.up_proj.weight", Q8_0, vec![HIDDEN, FF]),
            ("mlp.down_proj.weight", Q8_0, vec![FF, HIDDEN]),
            (
                "attention_conv.kernel_projection.weight",
                Q8_0,
                vec![HIDDEN, DYN],
            ),
            ("attention_conv.base_kernel", F32, vec![HIDDEN, 2, 2]),
            ("mlp_conv.kernel_projection.weight", Q8_0, vec![HIDDEN, DYN]),
            ("mlp_conv.base_kernel", F32, vec![HIDDEN, 2, 2]),
        ];
        out.extend(
            layer
                .into_iter()
                .map(|(name, typ, dims)| (format!("layers.{il}.{name}"), typ, dims)),
        );
    }
    out
}

pub(crate) fn inspect(path: &Path) -> Result<(), String> {
    let g = GgufFile::open(path).map_err(|_| "DFlash2 file is not a GGUF".to_string())?;
    let string = |key: &str| g.get_string(key);
    if string("general.architecture") != Some(b"beam_dflash2".as_slice())
        || string("general.source.repo") != Some(SOURCE)
        || string("general.source.sha256") != Some(SOURCE_SHA256)
        || string("beam_dflash2.target_layers") != Some(b"4,16,28,40,46".as_slice())
    {
        return Err("DFlash2 file is not the Beam-1.0 drafter conversion".into());
    }
    for (key, want) in U32S {
        if g.get_u32(&format!("beam_dflash2.{key}")) != Some(want) {
            return Err(format!("DFlash2 config beam_dflash2.{key}"));
        }
    }
    let f32s = [
        ("attention.layer_norm_rms_epsilon", 1e-6),
        ("rope.freq_base", 10000.0),
        ("final_logit_softcapping", 30.0),
    ];
    for (key, want) in f32s {
        if g.get_f32_compat(&format!("beam_dflash2.{key}")) != Some(want) {
            return Err(format!("DFlash2 config beam_dflash2.{key}"));
        }
    }
    let inventory = TensorInventory::open(path).map_err(|_| "DFlash2 tensors".to_string())?;
    let want = tensors();
    if inventory.tensors.len() != want.len() {
        return Err("DFlash2 tensor count".into());
    }
    for (name, typ, dims) in &want {
        let Some(t) = inventory.tensors.iter().find(|t| &t.name == name) else {
            return Err(format!("DFlash2 missing {name}"));
        };
        if t.typ != *typ || t.ndim as usize != dims.len() || t.dim[..dims.len()] != dims[..] {
            return Err(format!("DFlash2 tensor {name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn contract_covers_the_converter_output() {
        let want = super::tensors();
        // tools/beam_dflash2.py writes 85 tensors: 5 heads + 5 fc + 5 x 15.
        assert_eq!(want.len(), 85);
        let names: std::collections::HashSet<_> = want.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(names.len(), want.len());
    }
}
