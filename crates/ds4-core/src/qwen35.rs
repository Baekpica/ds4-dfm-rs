//! qwen35 (Prism Bonsai 2 27B) family constants.
//!
//! The host catalog (`crate::shape::SHAPE_QWEN35`) carries the dimensions the
//! Rust side reasons about, but the gated-delta-net head geometry only exists
//! as GGUF metadata (`qwen35.ssm.*`). The validator pins those keys against the
//! constants here, so the dimensions can be used without widening `Shape`.

/// `qwen35.ssm.group_count`: gated delta-net query/key heads.
pub const LIN_K_HEAD: u32 = 16;
/// `qwen35.ssm.time_step_rank`: gated delta-net value heads.
pub const LIN_V_HEAD: u32 = 48;
/// `qwen35.ssm.state_size`: gated delta-net head dimension.
pub const LIN_HEAD_DIM: u32 = 128;
/// `qwen35.ssm.conv_kernel`: gated delta-net convolution width.
pub const LIN_CONV: u32 = 4;
/// `qwen35.full_attention_interval`: one gated-attention layer per interval.
pub const FULL_ATTN_INTERVAL: u32 = 4;
/// `qwen35.context_length`, which the validator pins to the shape's
/// `rope_orig_ctx`: the trunk is trained at 262144 positions with no scaling.
pub const CTX_MAX: u32 = 262144;

/// `ssm.group_count * ssm.state_size`: the projected query/key width.
pub const LIN_K_DIM: u64 = LIN_K_HEAD as u64 * LIN_HEAD_DIM as u64;
/// `ssm.time_step_rank * ssm.state_size`: the projected value width.
pub const LIN_V_DIM: u64 = LIN_V_HEAD as u64 * LIN_HEAD_DIM as u64;

/// The gated delta-net layers run on `lin_*` tensors, the gated-attention
/// layers on `attn_*`; the attention layer is the last of every interval.
pub fn layer_is_full_attention(il: u32) -> bool {
    (il + 1) % FULL_ATTN_INTERVAL == 0
}

/// Matmul weights the Prism exporter rotates by the Hadamard fold, by name
/// suffix. `token_embd.weight` is folded in the inverse direction only and is
/// deliberately absent; the head `output.weight` is accepted whole.
pub const FOLDABLE_SUFFIXES: [&[u8]; 10] = [
    b"attn_q.weight",
    b"attn_k.weight",
    b"attn_v.weight",
    b"attn_qkv.weight",
    b"attn_gate.weight",
    b"attn_output.weight",
    b"ffn_gate.weight",
    b"ffn_up.weight",
    b"ffn_down.weight",
    b"ssm_out.weight",
];

/// C `qwen35_is_foldable_weight_name`.
pub fn is_foldable_weight_name(name: &[u8]) -> bool {
    if name == b"output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix(b"blk.") else {
        return false;
    };
    // blk.<layer>.<suffix>
    let Some(dot) = rest.iter().position(|b| *b == b'.') else {
        return false;
    };
    if dot == 0 {
        return false;
    }
    FOLDABLE_SUFFIXES.contains(&&rest[dot + 1..])
}
