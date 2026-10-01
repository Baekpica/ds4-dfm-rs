//! Naive uses NFC/Qwen2 BPE and the official XML-tool Jinja template.
use super::{GgufFile, TokError, Vocab};

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(b"qwen2") || v.tokens.len() != 152576 {
        return Err(TokError::InvalidTokenizer("Naive vocabulary/pretokenizer"));
    }
    v.bos_id = -1;
    v.eos_id = v.lookup("<|im_end|>")?;
    v.eot_id = v.lookup("<|endoftext|>")?;
    v.end_of_turn_id = -1;
    v.im_start_id = v.lookup("<|im_start|>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.tool_response_start_id = v.lookup("<tool_response>")?;
    v.tool_response_end_id = v.lookup("</tool_response>")?;
    v.dsml_id = -1;
    if (v.eot_id, v.im_start_id, v.eos_id) != (151643, 151644, 151645) {
        return Err(TokError::InvalidTokenizer("Naive control token IDs"));
    }
    Ok(())
}
