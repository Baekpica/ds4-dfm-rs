//! Original Qwen2-style vocabulary; input grammar stays in official Jinja.
//! The mixed GGUF carries it as metadata; the VQ release ships the source
//! tokenizer.json (Beam's Qwen2 pipeline, 32 added tokens).
use super::{GgufFile, TokError, Vocab};
use std::path::Path;

const VOCAB: usize = 152_576;
const ADDED: usize = 32;

pub(super) fn tables(root: &Path) -> Result<super::glm_vq::Tables, TokError> {
    let invalid = |_| TokError::InvalidTokenizer("MiMo source tokenizer");
    let bytes = crate::mimo2_vq::asset(root, "tokenizer.json").map_err(invalid)?;
    super::beam::qwen2(&bytes, ADDED, VOCAB)
        .map_err(|_| TokError::InvalidTokenizer("MiMo source tokenizer"))
}

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    // A VQ part has no tokenizer metadata: tables() checked the pipeline.
    let source = g.get_string("general.architecture") == Some(b"mimo2_vq")
        || g.get_string("tokenizer.ggml.pre") == Some(b"qwen2");
    if !source || v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("MiMo vocabulary/pretokenizer"));
    }
    v.bos_id = -1; // Tokenizer adds no BOS; generation_config's fallback is not input grammar.
    v.eos_id = v.lookup("<|im_end|>")?;
    v.eot_id = v.lookup("<|endoftext|>")?;
    v.end_of_turn_id = v.lookup("<|mimo_audio_eod|>")?;
    v.im_start_id = v.lookup("<|im_start|>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.tool_response_start_id = v.lookup("<tool_response>")?;
    v.tool_response_end_id = v.lookup("</tool_response>")?;
    v.dsml_id = -1;
    for (id, text) in [
        (151643, "<|endoftext|>"),
        (151644, "<|im_start|>"),
        (151645, "<|im_end|>"),
        (151652, "<|vision_start|>"),
        (151653, "<|vision_end|>"),
        (151655, "<|image_pad|>"),
        (151656, "<|video_pad|>"),
        (151669, "<|audio_pad|>"),
        (151670, "<|mimo_video_start|>"),
        (151671, "<|mimo_video_end|>"),
        (151672, "<|mimo_audio_eod|>"),
        (151673, "<|mimo_audio_start|>"),
        (151674, "<|mimo_audio_end|>"),
    ] {
        if v.lookup(text)? != id {
            return Err(TokError::InvalidTokenizer("MiMo control token IDs"));
        }
    }
    Ok(())
}

/// The source tokenizer applies NFC before Qwen2's single-digit splitter.
pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    if let Ok(text) = std::str::from_utf8(text) {
        let normalized = crate::mimo2::nfc(text);
        super::bpe_tokenize_text_solar(v, normalized.as_bytes(), out);
    } else {
        super::bpe_tokenize_text_solar(v, text, out);
    }
}
