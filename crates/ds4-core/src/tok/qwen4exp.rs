//! Declared Qwen4Exp source input: NFC and the Unicode-mark-aware splitter.
use super::{bpe_emit_piece, user_defined_at, GgufFile, TokError, Vocab};
use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

pub(super) fn enabled(g: &GgufFile) -> Result<bool, TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(b"qwen4exp") {
        return Ok(false);
    }
    if g.get_string("tokenizer.ggml.normalizer") != Some(b"nfc") {
        return Err(TokError::InvalidTokenizer("Qwen4Exp source normalizer"));
    }
    Ok(true)
}

fn piece(text: &str) -> usize {
    static SPLITTER: OnceLock<Regex> = OnceLock::new();
    let re = SPLITTER.get_or_init(|| {
        Regex::new(concat!(
            r"^(?:(?i:'s|'t|'re|'ve|'m|'ll|'d)",
            r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}",
            r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+)",
        ))
        .unwrap()
    });
    if let Some(found) = re.find(text) {
        return found.end();
    }

    // The source's greedy whitespace lookahead retains the last space
    // before a word. Implement it without a backtracking regex.
    let count = text.chars().take_while(|c| c.is_whitespace()).count();
    let end = text
        .char_indices()
        .nth(count)
        .map_or(text.len(), |(i, _)| i);
    let keep = if end < text.len() && count > 1 {
        count - 1
    } else {
        count.max(1)
    };
    text.char_indices().nth(keep).map_or(text.len(), |(i, _)| i)
}

fn words(v: &Vocab, text: &str, out: &mut Vec<i32>) {
    let normalized = if text.is_ascii() {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.nfc().collect::<String>())
    };
    let mut text = normalized.as_ref();
    while !text.is_empty() {
        let end = piece(text);
        bpe_emit_piece(v, text[..end].as_bytes(), out);
        text = &text[end..];
    }
}

pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    let Ok(text) = std::str::from_utf8(text) else {
        super::bpe_tokenize_text_dots3(v, text, out);
        return;
    };
    // Match added tokens before normalization, as the source tokenizer does.
    let mut span = 0;
    let mut pos = 0;
    while pos < text.len() {
        if let Some((id, size)) = user_defined_at(v, text.as_bytes(), pos) {
            if span < pos {
                words(v, &text[span..pos], out);
            }
            out.push(id);
            pos += size;
            span = pos;
            continue;
        }
        pos += text[pos..].chars().next().unwrap().len_utf8();
    }
    if span < text.len() {
        words(v, &text[span..], out);
    }
}

#[cfg(test)]
mod tests {
    use super::piece;

    #[test]
    fn source_unicode_splits() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/darwin/tokenizer-splits.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut text = case["normalized"].as_str().unwrap();
            let mut parts = Vec::new();
            while !text.is_empty() {
                let end = piece(text);
                parts.push(&text[..end]);
                text = &text[end..];
            }
            assert_eq!(serde_json::json!(parts), case["pieces"], "{}", case["text"]);
        }
    }
}
