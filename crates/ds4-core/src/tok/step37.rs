//! Step 3.7 keeps the official DeepSeek-v3 BPE and Step message controls.
use super::{bpe_emit_piece, GgufFile, TokError, Vocab};
use regex::Regex;
use std::sync::OnceLock;

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(b"deepseek-v3") {
        return Err(TokError::InvalidTokenizer("Step 3.7 pretokenizer"));
    }
    v.bos_id = v.lookup("<｜begin▁of▁sentence｜>")?;
    v.eos_id = v.lookup("<|im_end|>")?;
    v.im_start_id = v.lookup("<|im_start|>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    if v.bos_id != 0 || v.eos_id != 128007 || v.tokens.len() != 128896 {
        return Err(TokError::InvalidTokenizer("Step 3.7 vocabulary"));
    }
    Ok(())
}

struct Splitter {
    numbers: Regex,
    cjk: Regex,
    word: Regex,
}

fn isolated(text: &str, re: &Regex, mut emit: impl FnMut(&str)) {
    let mut start = 0;
    for m in re.find_iter(text) {
        if start < m.start() {
            emit(&text[start..m.start()]);
        }
        emit(m.as_str());
        start = m.end();
    }
    if start < text.len() {
        emit(&text[start..]);
    }
}

impl Splitter {
    fn new() -> Self {
        // Apply the official Split stages in order. A combined alternation
        // lets letter runs absorb CJK or digits across a source boundary.
        Self {
            numbers: Regex::new(r"\p{N}{1,3}").unwrap(),
            cjk: Regex::new(r"[一-龥\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+").unwrap(),
            word: Regex::new(concat!(
                r##"^[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"##,
                r"|^[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+",
                r"|^ ?[\p{P}\p{S}]+[\r\n]*|^\s*[\r\n]+",
            ))
            .unwrap(),
        }
    }

    fn piece(&self, text: &str) -> Option<usize> {
        if let Some(m) = self.word.find(text) {
            return Some(m.end());
        }
        let n = text.chars().take_while(|c| c.is_whitespace()).count();
        if n == 0 {
            return None;
        }
        let end = text
            .char_indices()
            .nth(n)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        // Implement the source's greedy whitespace lookahead without a
        // backtracking regex: keep the last space for the following word.
        let keep = if end < text.len() && n > 1 { n - 1 } else { n };
        Some(
            text.char_indices()
                .nth(keep)
                .map(|(i, _)| i)
                .unwrap_or(text.len()),
        )
    }

    fn words(&self, v: &Vocab, mut text: &str, out: &mut Vec<i32>) {
        while !text.is_empty() {
            let end = self.piece(text).unwrap_or_else(|| {
                text.char_indices()
                    .skip(1)
                    .find(|&(i, _)| self.piece(&text[i..]).is_some())
                    .map(|(i, _)| i)
                    .unwrap_or(text.len())
            });
            bpe_emit_piece(v, text[..end].as_bytes(), out);
            text = &text[end..];
        }
    }
}

pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    static SPLITTER: OnceLock<Splitter> = OnceLock::new();
    let splitter = SPLITTER.get_or_init(Splitter::new);
    let Ok(text) = std::str::from_utf8(text) else {
        bpe_emit_piece(v, text, out);
        return;
    };
    isolated(text, &splitter.numbers, |part| {
        isolated(part, &splitter.cjk, |piece| splitter.words(v, piece, out));
    });
}
