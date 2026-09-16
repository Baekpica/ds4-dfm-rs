//! Ling-3.0-flash-VL keeps the official Bailing V3 BPE and role controls.
//!
//! The splitter is llama.cpp's `bailingmoe` pre-tokenizer: the GPT-2 regex
//! with single-digit number runs. That is byte-for-byte what the Solar
//! splitter already implements, so only the control-token contract is new.
use super::{GgufFile, TokError, Vocab};

use crate::ling3vl::{IMAGE_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN};

const VOCAB: usize = 157_184;
const BOS: i32 = 156_891;
const EOS: i32 = 156_895;
/// The official `generation_config` stops on `<|role_end|>` and
/// `<|endoftext|>`; the GGUF can only carry the first as `eos_token_id`.
const END_OF_TEXT: i32 = 156_892;

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(b"bailingmoe2") {
        return Err(TokError::InvalidTokenizer("Ling-3.0-flash-VL pretokenizer"));
    }
    v.bos_id = v.lookup("<|startoftext|>")?;
    v.eos_id = v.lookup("<|role_end|>")?;
    v.eot_id = v.lookup("<|endoftext|>")?;
    v.im_start_id = v.lookup("<role>")?;
    v.im_content_id = v.lookup("</role>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.tool_response_start_id = v.lookup("<tool_response>")?;
    v.tool_response_end_id = v.lookup("</tool_response>")?;
    v.arg_key_start_id = v.lookup("<arg_key>")?;
    v.arg_key_end_id = v.lookup("</arg_key>")?;
    v.arg_value_start_id = v.lookup("<arg_value>")?;
    v.arg_value_end_id = v.lookup("</arg_value>")?;
    v.dsml_id = -1;
    if v.bos_id != BOS || v.eos_id != EOS || v.eot_id != END_OF_TEXT || v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("Ling-3.0-flash-VL vocabulary"));
    }
    // The image span the vision bridge fills is a fixed id triple. Reject a
    // vocabulary that renumbered it rather than placing features blindly.
    for (id, text) in [
        (VISION_START_TOKEN, "<|vision_start|>"),
        (IMAGE_TOKEN, "<|image_pad|>"),
        (VISION_END_TOKEN, "<|vision_end|>"),
    ] {
        if v.lookup(text)? != id as i32 {
            return Err(TokError::InvalidTokenizer(
                "Ling-3.0-flash-VL vision tokens",
            ));
        }
    }
    Ok(())
}
