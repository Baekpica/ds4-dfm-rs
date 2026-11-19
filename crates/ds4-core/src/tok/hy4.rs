//! hy_v4 uses the DeepSeek-v3 three-stage split and a byte-level BPE.
//! The template emits BOS itself. Every added token is matched literally.
use super::{GgufFile, TokError, Vocab};

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    v.bos_id = v.lookup("<｜hy_start:opensource｜>")?;
    v.eos_id = v.lookup("<｜hy_end:opensource｜>")?;
    v.im_start_id = v.bos_id;
    v.im_content_id = v.lookup("<｜hy_middle:opensource｜>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think:opensource>")?;
    v.think_end_id = v.lookup("</think:opensource>")?;
    v.tool_call_start_id = v.lookup("<tool_call:opensource>")?;
    v.tool_call_end_id = v.lookup("</tool_call:opensource>")?;
    v.tool_response_start_id = v.lookup("<tool_response:opensource>")?;
    v.tool_response_end_id = v.lookup("</tool_response:opensource>")?;
    v.arg_key_start_id = v.lookup("<arg_key:opensource>")?;
    v.arg_key_end_id = v.lookup("</arg_key:opensource>")?;
    v.arg_value_start_id = v.lookup("<arg_value:opensource>")?;
    v.arg_value_end_id = v.lookup("</arg_value:opensource>")?;
    v.image_id = v.lookup("<｜hy_image▁pad:opensource｜>")?;
    if g.get_string("tokenizer.ggml.pre") != Some(b"hy_v4")
        || g.get_bool("tokenizer.ggml.add_bos_token") != Some(false)
        || g.get_token_id("tokenizer.ggml.eos_token_id") != Some(v.eos_id)
        || v.bos_id != 120000
        || v.eos_id != 120025
        || v.n_vocab() != 120832
        || v.merges.len() != 119744
        || v.user_defined.len() != 832
    {
        return Err(TokError::InvalidTokenizer("Hy4 source vocabulary"));
    }
    Ok(())
}
