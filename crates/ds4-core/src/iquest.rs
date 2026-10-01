//! IQuest-Q1 artifact contract, native cache layout and execution budgeting.
//! Keep the learned key sink and normalized residuals separate from scalar
//! sink families; assigning an existing architecture would change logits.
use crate::gguf::GgufFile;
use crate::layout::{LayoutSpec, TypeClass};
use crate::validate::ValidateError;

pub(crate) const LAYERS: u32 = 88;
pub(crate) const CONTEXT: u32 = 524_288;
pub(crate) const WINDOW: u32 = 4096;
pub(crate) const DRAFT_WINDOW: u32 = 512;
pub(crate) const DRAFT_SLOTS: u32 = 7;
pub(crate) const PREFILL: u32 = 128;
pub(crate) const PREFILL_MAX: u32 = 8192;
const CHECKPOINTS: u64 = 8;
const FULL_LAYERS: u64 = 25;
const EMBED: u64 = 3072;
const VOCAB: u64 = 160_000;
const HEADS: u64 = 48;
const KV_HEADS: u64 = 8;
const HEAD: u64 = 128;
const EXPERTS: u64 = 256;
const EXPERT_FF: u64 = 1536;
const DENSE_FF: u64 = 12288;
const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q4_K: u32 = 12;
const Q5_K: u32 = 13;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const Q8_ROW_BYTES: u64 = 2 * KV_HEADS * HEAD / 32 * 34;
const CARRY_BYTES: u64 = EMBED * 4;

enum Block {
    Dense,
    Routed(u32),
    Draft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IQuestCache {
    Target,
    WithMtp,
}

/// Artifact inspection and cache budgeting, independent of native serving.
pub struct IQuestPlan;

impl IQuestPlan {
    /// Maximum recursive draft tokens supported by the embedded native MTP.
    pub const MAX_MTP_DRAFT: u32 = DRAFT_SLOTS;

    /// Exact device session tensor quote. Weights remain in the engine ledger;
    /// IQuest disables persistent derived and expanded weight copies.
    pub fn session_bytes(ctx: u32, chunk: u32) -> Option<u64> {
        session_bytes(ctx, chunk)
    }

    pub fn layouts() -> Vec<LayoutSpec> {
        layouts()
    }

    pub fn full_layers() -> Vec<u32> {
        (0..LAYERS).filter(|&layer| is_full(layer)).collect()
    }

    pub fn kv_bytes(ctx: u32, chunk: u32, draft: IQuestCache) -> Option<u64> {
        kv_bytes(ctx, chunk, draft)
    }
}

// Layer 0, 21 four-block groups starting at 1, then three full layers.
pub(crate) fn is_full(layer: u32) -> bool {
    layer < LAYERS && (layer == 0 || layer >= 85 || (layer - 1) % 4 == 0)
}

pub(crate) fn kv_rows(layer: u32, ctx: u32, chunk: u32) -> Option<u32> {
    if layer >= LAYERS || ctx == 0 || ctx > CONTEXT || chunk == 0 || chunk > ctx {
        return None;
    }
    if is_full(layer) {
        return Some(ctx);
    }
    // All query rows need their left window after the whole chunk is stored.
    Some(ctx.min(WINDOW.checked_add(chunk - 1)?))
}

/// Q8_0 is 34 bytes per 32 values, including the FP16 scale. This is the
/// cache storage quote, not a promise that the session allocator exists.
pub(crate) fn kv_bytes(ctx: u32, chunk: u32, draft: IQuestCache) -> Option<u64> {
    let mut rows = 0u64;
    for layer in 0..LAYERS {
        rows += u64::from(kv_rows(layer, ctx, chunk)?);
    }
    if draft == IQuestCache::WithMtp {
        rows += u64::from(DRAFT_WINDOW + DRAFT_SLOTS);
    }
    Some(rows * Q8_ROW_BYTES)
}

pub(crate) fn bank_bytes(ctx: u32, chunk: u32) -> Option<u64> {
    kv_bytes(ctx, chunk, IQuestCache::WithMtp).map(|bytes| bytes + CARRY_BYTES)
}

pub(crate) fn checkpoint_bytes() -> u64 {
    let rows = (u64::from(LAYERS) - FULL_LAYERS) * u64::from(WINDOW) + u64::from(DRAFT_WINDOW);
    CHECKPOINTS * (rows * Q8_ROW_BYTES + CARRY_BYTES)
}

pub(crate) fn session_bytes(ctx: u32, chunk: u32) -> Option<u64> {
    if chunk == 0 || chunk > PREFILL_MAX {
        return None;
    }
    let cache = kv_bytes(ctx, chunk, IQuestCache::Target)?;
    let scratch = u64::from(chunk)
        * (5 * EMBED
            + 2 * HEADS * HEAD
            + 2 * KV_HEADS * HEAD
            + 3 * DENSE_FF
            + EXPERTS
            + 2 * 8
            + 3 * 8 * EXPERT_FF
            + 8 * EMBED
            + 2
            + 4 * EMBED)
        * 4;
    Some(
        cache
            + scratch
            + 2 * VOCAB * 4
            + 3 * EMBED * 4
            + 2 * u64::from(DRAFT_WINDOW + DRAFT_SLOTS) * Q8_ROW_BYTES
            + u64::from(DRAFT_SLOTS + 1) * (u64::from(LAYERS) + 1) * Q8_ROW_BYTES
            + u64::from(DRAFT_SLOTS + 1) * EMBED * 4,
    )
}

fn mismatch(key: impl Into<String>) -> ValidateError {
    ValidateError::TokenKey("iquest-q1", key.into())
}

pub(crate) fn validate(g: &GgufFile) -> Result<(), ValidateError> {
    for (key, value) in [
        ("general.architecture", b"iquest_q1".as_slice()),
        (
            "general.source.huggingface.repository",
            b"IQuestLab/IQuest-Q1",
        ),
        (
            "general.source.huggingface.revision",
            b"5c21b0630586ef77d38cff1094b8cf37a417fcd8",
        ),
        (
            "iquest_q1.tensor_layout",
            b"canonical-v1-source-fc-gate-up-split",
        ),
        ("iquest_q1.quantization.minimum", b"IQ2_XXS"),
        ("iquest_q1.rope.pairing", b"split-half / GPT-NeoX"),
        ("iquest_q1.attention.sink_type", b"learned-key-zero-value"),
        ("iquest_q1.mtp.shared_embedding", b"token_embd.weight"),
        ("iquest_q1.mtp.shared_output", b"output.weight"),
        ("iquest_q1.mtp.shared_target_norm", b"output_norm.weight"),
    ] {
        if g.get_string(key) != Some(value) {
            return Err(mismatch(key));
        }
    }
    for (suffix, value) in [
        ("block_count", LAYERS),
        ("context_length", CONTEXT),
        ("embedding_length", EMBED as u32),
        ("attention.head_count", HEADS as u32),
        ("attention.head_count_kv", KV_HEADS as u32),
        ("attention.key_length", HEAD as u32),
        ("attention.value_length", HEAD as u32),
        ("attention.sliding_window", WINDOW),
        ("rope.dimension_count", 32),
        ("expert_count", EXPERTS as u32),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", EXPERT_FF as u32),
        ("feed_forward_length", DENSE_FF as u32),
        ("expert_gating_func", 1),
        ("nextn_predict_layers", 1),
        ("mtp.sliding_window", DRAFT_WINDOW),
        ("mtp.draft_slots", DRAFT_SLOTS),
    ] {
        let key = format!("iquest_q1.{suffix}");
        if g.get_u32(&key) != Some(value) {
            return Err(mismatch(key));
        }
    }
    for (suffix, value) in [
        ("rope.freq_base", 1_000_000.0),
        ("rope.freq_base_swa", 10_000.0),
        ("attention.layer_norm_rms_epsilon", 1e-6),
        ("attention.output_scale", 1.0),
        ("feed_forward.output_scale", 0.538_815_9),
    ] {
        let key = format!("iquest_q1.{suffix}");
        if g.get_f32_compat(&key) != Some(value) {
            return Err(mismatch(key));
        }
    }
    for suffix in ["expert_weights_norm", "mtp.fp32_residual"] {
        let key = format!("iquest_q1.{suffix}");
        if g.get_bool(&key) != Some(true) {
            return Err(mismatch(key));
        }
    }
    let key = "iquest_q1.attention.full_attention_layers";
    let array = g.get_array(key).ok_or_else(|| mismatch(key))?;
    if g.array_le_u32s(&array)? != (0..LAYERS).filter(|&i| is_full(i)).collect::<Vec<_>>() {
        return Err(mismatch(key));
    }
    let key = "tokenizer.ggml.tokens";
    let tokens = g.get_array(key).ok_or_else(|| mismatch(key))?;
    if tokens.typ != crate::gguf::GGUF_VALUE_STRING || tokens.len != VOCAB {
        return Err(mismatch(key));
    }
    if g.get_token_id("tokenizer.ggml.eos_token_id") != Some(0) {
        return Err(mismatch("tokenizer.ggml.eos_token_id"));
    }
    Ok(())
}

fn push(out: &mut Vec<LayoutSpec>, prefix: &str, name: &str, class: TypeClass, dims: &[u64]) {
    let mut dim = [0; 8];
    dim[..dims.len()].copy_from_slice(dims);
    out.push(LayoutSpec {
        name: format!("{prefix}{name}"),
        class,
        ndim: dims.len() as u32,
        dim,
    });
}

fn block(out: &mut Vec<LayoutSpec>, prefix: &str, kind: Block) {
    for name in [
        "attn_norm.weight",
        "attn_output_norm.weight",
        "ffn_norm.weight",
    ] {
        push(out, prefix, name, TypeClass::Exact(F32), &[EMBED]);
    }
    if matches!(kind, Block::Dense | Block::Draft) {
        push(
            out,
            prefix,
            "ffn_output_norm.weight",
            TypeClass::Exact(F32),
            &[EMBED],
        );
    }
    for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
        push(out, prefix, name, TypeClass::Exact(F32), &[HEAD]);
    }
    push(
        out,
        prefix,
        "attn_sink_k.weight",
        TypeClass::Exact(F32),
        &[HEAD, KV_HEADS],
    );
    for (name, dims) in [
        ("attn_q.weight", [EMBED, HEADS * HEAD]),
        ("attn_k.weight", [EMBED, KV_HEADS * HEAD]),
        ("attn_v.weight", [EMBED, KV_HEADS * HEAD]),
        ("attn_output.weight", [HEADS * HEAD, EMBED]),
    ] {
        let typ = match kind {
            Block::Draft => Q8_0,
            _ if matches!(name, "attn_k.weight" | "attn_v.weight") => Q6_K,
            _ => Q5_K,
        };
        push(out, prefix, name, TypeClass::Exact(typ), &dims);
    }
    if matches!(kind, Block::Dense) {
        for (name, dims) in [
            ("ffn_gate.weight", [EMBED, DENSE_FF]),
            ("ffn_up.weight", [EMBED, DENSE_FF]),
            ("ffn_down.weight", [DENSE_FF, EMBED]),
        ] {
            push(out, prefix, name, TypeClass::Exact(Q5_K), &dims);
        }
        return;
    }
    // The FP32 router has no correction bias. Selected softmax weights sum to 1.
    push(
        out,
        prefix,
        "ffn_gate_inp.weight",
        TypeClass::Exact(F32),
        &[EMBED, EXPERTS],
    );
    for (name, dims) in [
        ("ffn_gate_exps.weight", [EMBED, EXPERT_FF, EXPERTS]),
        ("ffn_up_exps.weight", [EMBED, EXPERT_FF, EXPERTS]),
        ("ffn_down_exps.weight", [EXPERT_FF, EMBED, EXPERTS]),
    ] {
        // Only these measured projections were promoted in the published artifact.
        let typ = match kind {
            Block::Draft => Q4_K,
            Block::Routed(layer)
                if name != "ffn_down_exps.weight"
                    && (matches!(layer, 62..=66 | 68)
                        || (layer == 67 && name == "ffn_gate_exps.weight")
                        || (matches!(layer, 78 | 80) && name == "ffn_up_exps.weight")) =>
            {
                IQ2_XS
            }
            _ => IQ2_XXS,
        };
        push(out, prefix, name, TypeClass::Exact(typ), &dims);
    }
}

pub(crate) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::new();
    for (name, typ) in [("token_embd.weight", Q6_K), ("output.weight", Q8_0)] {
        push(&mut out, "", name, TypeClass::Exact(typ), &[EMBED, VOCAB]);
    }
    push(
        &mut out,
        "",
        "output_norm.weight",
        TypeClass::Exact(F32),
        &[EMBED],
    );
    for layer in 0..LAYERS {
        let kind = if layer == 0 {
            Block::Dense
        } else {
            Block::Routed(layer)
        };
        block(&mut out, &format!("blk.{layer}."), kind);
    }
    block(&mut out, "mtp.0.", Block::Draft);
    for name in ["enorm.weight", "hnorm.weight", "output_norm.weight"] {
        push(&mut out, "mtp.0.", name, TypeClass::Exact(F32), &[EMBED]);
    }
    push(
        &mut out,
        "mtp.0.",
        "eh_proj.weight",
        TypeClass::Exact(Q8_0),
        &[2 * EMBED, EMBED],
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_schedule_edges() {
        let full: Vec<_> = (0..LAYERS).filter(|&i| is_full(i)).collect();
        assert_eq!(full.len(), 25);
        assert_eq!(&full[..3], &[0, 1, 5]);
        assert_eq!(&full[21..], &[81, 85, 86, 87]);
        assert!(!is_full(88));
    }

    #[test]
    fn kv_512k_chunk_rows() {
        let row = 2 * 8 * 128 / 32 * 34;
        assert_eq!(
            kv_bytes(CONTEXT, 1, IQuestCache::Target),
            Some((25 * 524_288 + 63 * 4096) * row)
        );
        assert_eq!(
            kv_bytes(CONTEXT, 1024, IQuestCache::WithMtp),
            Some((25 * 524_288 + 63 * 5119 + 519) * row)
        );
        assert_eq!(kv_rows(2, 512, 128), Some(512));
        assert_eq!(kv_bytes(0, 1, IQuestCache::Target), None);
        assert_eq!(kv_bytes(CONTEXT + 1, 1, IQuestCache::Target), None);
        assert_eq!(kv_bytes(128, 129, IQuestCache::Target), None);
    }

    #[test]
    fn sink_and_mtp_layout() {
        let all = layouts();
        assert_eq!(all.len(), 1254);
        let sink = all
            .iter()
            .find(|t| t.name == "blk.1.attn_sink_k.weight")
            .unwrap();
        assert_eq!(&sink.dim[..2], &[128, 8]);
        assert!(all.iter().any(|t| t.name == "mtp.0.ffn_gate_exps.weight"));
        assert!(all.iter().any(|t| t.name == "mtp.0.ffn_output_norm.weight"));
        assert!(!all.iter().any(|t| t.name == "blk.1.ffn_output_norm.weight"));
        assert!(!all.iter().any(|t| t.name.contains("exp_probs_b")));
        assert!(!all
            .iter()
            .any(|t| t.name.starts_with("mtp.") && t.name.contains("token_embd")));
    }
}

const TRIAL_CAP: usize = (DRAFT_SLOTS + 1) as usize;

pub(super) unsafe extern "C" fn accept_banked(
    tokens: *const i32,
    target: *const i32,
    n: i32,
    eos: i32,
) -> i32 {
    if tokens.is_null() || target.is_null() || n < 1 || n as usize > TRIAL_CAP {
        return 0;
    }
    let tokens = unsafe { std::slice::from_raw_parts(tokens, n as usize) };
    let target = unsafe { std::slice::from_raw_parts(target, n as usize) };
    if tokens
        .iter()
        .chain(target)
        .any(|&t| t < 0 || t >= VOCAB as i32)
    {
        return 0;
    }
    let mut keep = 1;
    while keep < tokens.len() && tokens[keep - 1] != eos && tokens[keep] == target[keep - 1] {
        keep += 1;
    }
    keep as i32
}

impl crate::Session<'_> {
    pub(super) fn eval_iquest_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let mut tokens = [0; TRIAL_CAP];
        let mut target = [0; TRIAL_CAP];
        let mut err = [0u8; 512];
        let n = unsafe {
            ds4_sys::ds4_bridge_iquest_trial(
                self.raw.as_ptr(),
                first,
                max_tokens,
                tokens.as_mut_ptr(),
                target.as_mut_ptr(),
                TRIAL_CAP as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if n == 0 {
            self.eval(first)?;
            return Ok(vec![first]);
        }
        if n < 0 {
            self.step_failed();
            return Err(crate::fail(n, &err));
        }
        let n = n as usize;
        if n > TRIAL_CAP || tokens[0] != first {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid IQuest-Q1 trial result".into(),
            });
        }
        let prompt = self.pos();
        if prompt < 0 {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid IQuest-Q1 frontier".into(),
            });
        }
        let mut keep = 1i32;
        while (keep as usize) < n
            && tokens[(keep - 1) as usize] != eos
            && tokens[keep as usize] == target[(keep - 1) as usize]
        {
            keep += 1;
        }
        let rc = unsafe {
            ds4_sys::ds4_bridge_iquest_commit(
                self.raw.as_ptr(),
                keep,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            self.step_failed();
            return Err(crate::fail(rc, &err));
        }
        for &token in &tokens[..keep as usize] {
            self.host.commit_eval(token);
        }
        Ok(tokens[..keep as usize].to_vec())
    }
}
