//! Inkling source-interleaved-v1, pinned to MQ85GB and the BF16 MTP sidecar.
//! Config assets retain the upstream source contract; no HF default is inferred.
use serde_json::Value;

use crate::gguf::GgufFile;
use crate::layout::{LayoutSpec, TypeClass};
use crate::shape::SHAPE_INKLING_SMALL as SHAPE;
use crate::validate::ValidateError;

const SOURCE_REV: &[u8] = b"8cc5877b44d343f88b92086aa1fb72897950f06a";
const CONFIG: &str = include_str!("inkling/config.json");
const PROCESSOR: &str = include_str!("inkling/processor_config.json");
const F32: u32 = 0;
const BF16: u32 = 30;
const Q8_0: u32 = 8;
const Q3_K: u32 = 11;
const Q4_K: u32 = 12;
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const EMBED: u64 = SHAPE.n_embd as u64;
const HEAD_DIM: u64 = SHAPE.n_head_dim as u64;
const KV_WIDTH: u64 = SHAPE.n_head_kv as u64 * HEAD_DIM;
const FF_DENSE: u64 = SHAPE.n_ff_dense as u64;
const FF_EXPERT: u64 = SHAPE.n_ff_exp as u64;
const EXPERTS: u64 = SHAPE.n_expert as u64;
const SHARED: u64 = SHAPE.n_expert_shared as u64;
const REL_DIM: u64 = 16;
const REL_FULL: u64 = 1024;
const REL_LOCAL: u64 = SHAPE.n_swa as u64;
const MEL_BINS: u64 = 80;
const MEL_LEVELS: u64 = 16;
const VISION_LAYERS: [(u64, u64); 4] = [(128, 75), (320, 512), (4800, 5120), (EMBED, 9600)];

fn mismatch(key: &str) -> ValidateError {
    ValidateError::TokenKey("mismatch", key.into())
}

// Compare known fields structurally, accepting JSON whitespace and key ordering.
// Missing null-valued fields must still fail: they control softcapping semantics.
fn check_config(got: &Value, want: &Value, path: &str) -> Result<(), ValidateError> {
    if let Some(fields) = want.as_object() {
        for (key, expected) in fields {
            let path = format!("{path}.{key}");
            let actual = got.get(key).ok_or_else(|| mismatch(&path))?;
            check_config(actual, expected, &path)?;
        }
        return Ok(());
    }
    if got != want {
        return Err(mismatch(path));
    }
    Ok(())
}

enum Artifact {
    Main,
    Mtp,
}

pub(crate) fn validate_main(g: &GgufFile) -> Result<(), ValidateError> {
    validate_metadata(g, Artifact::Main)
}

pub(crate) fn validate_mtp(g: &GgufFile) -> Result<(), ValidateError> {
    validate_metadata(g, Artifact::Mtp)
}

fn validate_metadata(g: &GgufFile, artifact: Artifact) -> Result<(), ValidateError> {
    for (key, expected) in [
        ("general.architecture", b"inkling".as_slice()),
        ("inkling.tensor_layout", b"source-interleaved-v1".as_slice()),
        (
            "general.source.huggingface.repository",
            b"thinkingmachines/Inkling-Small".as_slice(),
        ),
        ("general.source.huggingface.revision", SOURCE_REV),
    ] {
        if g.get_string(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    let (sidecar, recipe) = match artifact {
        Artifact::Main => (false, b"MQ85GB".as_slice()),
        Artifact::Mtp => (true, b"MTP-BF16".as_slice()),
    };
    if g.get_bool("inkling.mtp.sidecar") != Some(sidecar) {
        return Err(mismatch("inkling.mtp.sidecar"));
    }
    if g.get_string("inkling.quantization.recipe") != Some(recipe) {
        return Err(mismatch("inkling.quantization.recipe"));
    }
    if !sidecar && g.get_u32("general.quantization_version") != Some(2) {
        return Err(mismatch("general.quantization_version"));
    }
    for (key, expected) in [
        ("inkling.config.json", CONFIG),
        ("inkling.processor_config.json", PROCESSOR),
    ] {
        let bytes = g.get_string(key).ok_or_else(|| mismatch(key))?;
        let got: Value = serde_json::from_slice(bytes).map_err(|_| mismatch(key))?;
        let want: Value = serde_json::from_str(expected).expect("pinned Inkling config");
        check_config(&got, &want, key)?;
    }
    Ok(())
}

// Specs take original safetensors dimensions. GGUF reverses dimensions only;
// it does not change interleaved gate/up rows or merge the shared experts.
fn tensor(out: &mut Vec<LayoutSpec>, name: String, typ: u32, source: &[u64]) {
    let mut dim = [0; 8];
    for (dst, src) in dim.iter_mut().zip(source.iter().rev()) {
        *dst = *src;
    }
    out.push(LayoutSpec {
        name,
        class: TypeClass::Exact(typ),
        ndim: source.len() as u32,
        dim,
    });
}

fn attention(out: &mut Vec<LayoutSpec>, prefix: &str, extent: u64) {
    for name in ["attn_norm.weight", "mlp_norm.weight"] {
        tensor(out, format!("{prefix}.{name}"), BF16, &[EMBED]);
    }
    for (name, channels) in [
        ("attn.k_sconv.weight", KV_WIDTH),
        ("attn.v_sconv.weight", KV_WIDTH),
        ("attn_sconv.weight", EMBED),
        ("mlp_sconv.weight", EMBED),
    ] {
        tensor(
            out,
            format!("{prefix}.{name}"),
            BF16,
            &[channels, 1, SHAPE.n_ssm_conv as u64],
        );
    }
    for name in ["q_norm.weight", "k_norm.weight"] {
        tensor(out, format!("{prefix}.attn.{name}"), BF16, &[HEAD_DIM]);
    }
    for (name, rows) in [
        ("wq_du.weight", SHAPE.n_head as u64 * HEAD_DIM),
        ("wk_dv.weight", KV_WIDTH),
        ("wv_dv.weight", KV_WIDTH),
        ("wo_ud.weight", EMBED),
        ("wr_du.weight", SHAPE.n_head as u64 * REL_DIM),
    ] {
        tensor(out, format!("{prefix}.attn.{name}"), BF16, &[rows, EMBED]);
    }
    tensor(
        out,
        format!("{prefix}.attn.rel_logits_proj.proj"),
        BF16,
        &[REL_DIM, extent],
    );
}

fn dense(out: &mut Vec<LayoutSpec>, prefix: &str, typ: u32) {
    tensor(
        out,
        format!("{prefix}.mlp.w13_dn.weight"),
        typ,
        &[2 * FF_DENSE, EMBED],
    );
    tensor(
        out,
        format!("{prefix}.mlp.w2_md.weight"),
        typ,
        &[EMBED, FF_DENSE],
    );
    tensor(out, format!("{prefix}.mlp.global_scale"), BF16, &[1]);
}

pub(crate) fn main_layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    for name in ["embed.weight", "unembed.weight"] {
        tensor(
            &mut out,
            format!("model.llm.{name}"),
            Q8_0,
            &[SHAPE.n_vocab as u64, EMBED],
        );
    }
    for name in ["embed_norm.weight", "norm.weight"] {
        tensor(&mut out, format!("model.llm.{name}"), BF16, &[EMBED]);
    }
    for il in 0..SHAPE.n_layer {
        let prefix = format!("model.llm.layers.{il}");
        attention(
            &mut out,
            &prefix,
            if (il + 1) % SHAPE.n_swa_period == 0 {
                REL_FULL
            } else {
                REL_LOCAL
            },
        );
        if il < SHAPE.n_leading_dense {
            dense(&mut out, &prefix, Q8_0);
            continue;
        }
        let (gate, down) = match il {
            2 => (Q8_0, Q8_0),
            40 => (Q3_K, Q4_K),
            41 => (Q4_K, Q4_K),
            _ => (IQ2_XXS, IQ2_XS),
        };
        tensor(
            &mut out,
            format!("{prefix}.mlp.experts.w13_weight"),
            gate,
            &[EXPERTS, 2 * FF_EXPERT, EMBED],
        );
        tensor(
            &mut out,
            format!("{prefix}.mlp.experts.w2_weight"),
            down,
            &[EXPERTS, EMBED, FF_EXPERT],
        );
        tensor(
            &mut out,
            format!("{prefix}.mlp.gate.weight"),
            BF16,
            &[EXPERTS + SHARED, EMBED],
        );
        tensor(&mut out, format!("{prefix}.mlp.gate.bias"), F32, &[EXPERTS]);
        tensor(
            &mut out,
            format!("{prefix}.mlp.gate.global_scale"),
            F32,
            &[1],
        );
        tensor(
            &mut out,
            format!("{prefix}.mlp.shared_experts.shared_w13_weight"),
            Q8_0,
            &[SHARED, 2 * FF_EXPERT, EMBED],
        );
        tensor(
            &mut out,
            format!("{prefix}.mlp.shared_experts.shared_w2_weight"),
            Q8_0,
            &[SHARED, EMBED, FF_EXPERT],
        );
    }
    tensor(
        &mut out,
        "model.audio.encoder.weight".into(),
        BF16,
        &[MEL_BINS * MEL_LEVELS, EMBED],
    );
    tensor(
        &mut out,
        "model.audio.final_norm.weight".into(),
        BF16,
        &[EMBED],
    );
    tensor(
        &mut out,
        "model.visual.final_norm.weight".into(),
        BF16,
        &[EMBED],
    );
    for (il, (rows, cols)) in VISION_LAYERS.into_iter().enumerate() {
        tensor(
            &mut out,
            format!("model.visual.layers.linear_{il}.weight"),
            BF16,
            &[rows, cols],
        );
        if il + 1 < VISION_LAYERS.len() {
            tensor(
                &mut out,
                format!("model.visual.layers.norm_{il}.weight"),
                BF16,
                &[rows],
            );
        }
    }
    out
}

pub(crate) fn mtp_layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    for il in 0..SHAPE.n_nextn_predict {
        let prefix = format!("model.mtp.layers.{il}");
        for name in ["embed_norm.weight", "hidden_norm.weight"] {
            tensor(&mut out, format!("{prefix}.{name}"), BF16, &[EMBED]);
        }
        tensor(
            &mut out,
            format!("{prefix}.input_proj.weight"),
            BF16,
            &[EMBED, 2 * EMBED],
        );
        let block = format!("{prefix}.transformer_block");
        // Draft global layers are 1 and 3, unlike the main six-layer period.
        attention(
            &mut out,
            &block,
            if il == 1 || il == 3 {
                REL_FULL
            } else {
                REL_LOCAL
            },
        );
        dense(&mut out, &block, BF16);
    }
    out
}
