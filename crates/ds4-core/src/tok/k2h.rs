//! K2-Horizon VQ ships the source tokenizer.json, not GGUF tokenizer
//! metadata: NFC, then the source split regex, then byte-level BPE with list
//! merges, ranked as tokenizers' `merge_all`. Its 626 added tokens (special
//! or not, never normalized) are split out of the raw text first:
//! `<|ifm|begin_of_text|>` and `<|ifm|endoftext|>` inside the 250,000-entry
//! base vocabulary, 624 above it. The GGUF path's `bpe_tokenize_text_k2`
//! approximates the split by character class and differs where marks, ZWJ
//! or ZWNJ continue a punctuation run and on `\p{No}` numbers (fractions,
//! circled digits).
use super::{bpe_emit_piece, TokError, Vocab};
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::OnceLock;

const BASE: usize = 250_000;
const ADDED: usize = 626;
const VOCAB: usize = 250_624;
const PIECES: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+",
);
const WHITESPACE: &str = r"|\s+(?!\S)|\s+";

fn invalid() -> TokError {
    TokError::InvalidTokenizer("K2-Horizon source tokenizer")
}

/// The pipeline `encode` implements; the post-processor's BOS is the
/// template's own text, never added here.
fn pipeline(v: &Value) -> Result<(), TokError> {
    let pattern = format!("{PIECES}{WHITESPACE}");
    let pre = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]});
    if v["version"] != "1.0"
        || v["pre_tokenizer"] != pre
        || v["normalizer"] != json!({"type":"NFC"})
    {
        return Err(invalid());
    }
    for (key, expected) in [
        ("type", json!("BPE")),
        ("ignore_merges", json!(false)),
        ("byte_fallback", json!(false)),
        ("fuse_unk", json!(false)),
        ("dropout", Value::Null),
        ("unk_token", Value::Null),
        ("continuing_subword_prefix", json!("")),
        ("end_of_word_suffix", json!("")),
    ] {
        if v["model"].get(key) != Some(&expected) {
            return Err(invalid());
        }
    }
    Ok(())
}

pub(super) fn tables(root: &Path) -> Result<super::glm_vq::Tables, TokError> {
    let bytes = crate::k2h_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    pipeline(&v)?;
    let vocab = v["model"]["vocab"].as_object().ok_or_else(invalid)?;
    if vocab.len() != BASE {
        return Err(invalid());
    }
    let mut tokens = vec![Vec::new(); VOCAB];
    for (word, id) in vocab {
        let id = id
            .as_u64()
            .filter(|&i| i < BASE as u64)
            .ok_or_else(invalid)? as usize;
        if word.is_empty() || !tokens[id].is_empty() {
            return Err(invalid());
        }
        tokens[id] = word.as_bytes().to_vec();
    }
    let entries = v["added_tokens"].as_array().ok_or_else(invalid)?;
    if entries.len() != ADDED {
        return Err(invalid());
    }
    let mut added = Vec::with_capacity(ADDED);
    for entry in entries {
        let id = entry["id"]
            .as_u64()
            .filter(|&i| i < VOCAB as u64)
            .ok_or_else(invalid)? as usize;
        let word = entry["content"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        // A base-vocabulary entry keeps its own bytes; the rest fill the top.
        let fresh = id >= BASE && tokens[id].is_empty();
        if !(fresh || tokens[id] == word.as_bytes()) || added.contains(&id) {
            return Err(invalid());
        }
        for key in ["single_word", "lstrip", "rstrip", "normalized"] {
            if entry[key] != false {
                return Err(invalid());
            }
        }
        tokens[id] = word.as_bytes().to_vec();
        added.push(id);
    }
    let merges = v["model"]["merges"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|entry| {
            let pair = entry
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(invalid)?;
            let a = pair[0]
                .as_str()
                .filter(|s| !s.contains(' '))
                .ok_or_else(invalid)?;
            let b = pair[1]
                .as_str()
                .filter(|s| !s.contains(' '))
                .ok_or_else(invalid)?;
            if !vocab.contains_key(a)
                || !vocab.contains_key(b)
                || !vocab.contains_key(&format!("{a}{b}"))
            {
                return Err(invalid());
            }
            Ok(format!("{a} {b}").into_bytes())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(super::glm_vq::Tables {
        tokens,
        merges,
        added,
    })
}

/// generation_config.json ends on `<|ifm|endoftext|>` (1) and
/// `<|ifm|im_end|>` (250,019): the turn closer is the EOS, the document end
/// the EOT; both stop. The template writes its own BOS text.
pub(super) fn specials(v: &mut Vocab) -> Result<(), TokError> {
    v.bos_id = v.lookup("<|ifm|begin_of_text|>")?;
    v.eos_id = v.lookup("<|ifm|im_end|>")?;
    v.eot_id = v.lookup("<|ifm|endoftext|>")?;
    v.im_start_id = v.lookup("<|ifm|im_start|>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<ifm|think>")?;
    v.think_end_id = v.lookup("</ifm|think>")?;
    v.tool_call_start_id = v.lookup("<ifm|tool_calls>")?;
    v.tool_call_end_id = v.lookup("</ifm|tool_calls>")?;
    v.tool_response_start_id = v.lookup("<ifm|tool_call>")?;
    v.tool_response_end_id = v.lookup("</ifm|tool_call>")?;
    v.system_id = -1;
    v.user_id = -1;
    v.assistant_id = -1;
    v.observation_id = -1;
    v.sop_id = -1;
    v.dsml_id = -1;
    if v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("K2-Horizon vocabulary"));
    }
    Ok(())
}

/// The source split's first piece of `text`. `regex` has no look-ahead:
/// the last alternatives, \s+(?!\S)|\s+, leave a run's last whitespace for
/// the next word's optional prefix when non-whitespace follows.
fn piece_len(text: &str) -> usize {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let re =
        SPLIT.get_or_init(|| Regex::new(&format!(r"\A(?:{PIECES})")).expect("K2-Horizon regex"));
    if let Some(m) = re.find(text) {
        return m.end();
    }
    let mut end = 0;
    let mut last = 0;
    for (at, ch) in text.char_indices() {
        if !ch.is_whitespace() {
            break;
        }
        last = at;
        end = at + ch.len_utf8();
    }
    if end < text.len() && last > 0 {
        return last;
    }
    end.max(text.chars().next().map_or(0, char::len_utf8))
}

/// The longest run of a split piece BPE merges at once. A request can hold
/// one piece of megabytes (a run of `!`), and the merge allocates per byte
/// before any context check; no word or whitespace run comes near this.
const PIECE_CAP: usize = 16 * 1024;

/// `piece` in runs of at most PIECE_CAP bytes, cut on character boundaries.
fn capped(mut piece: &str) -> impl Iterator<Item = &str> {
    std::iter::from_fn(move || {
        if piece.is_empty() {
            return None;
        }
        let mut cut = piece.len().min(PIECE_CAP);
        while !piece.is_char_boundary(cut) {
            cut -= 1;
        }
        let (head, tail) = piece.split_at(cut);
        piece = tail;
        Some(head)
    })
}

/// NFC on ordinary spans (added tokens were split out of the raw text), then
/// the source split. Invalid UTF-8, which the source never sees, takes the
/// GGUF path's split.
pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    let Ok(text) = std::str::from_utf8(text) else {
        super::bpe_tokenize_text_llama3(v, text, out, 3, false, true);
        return;
    };
    let normalized = crate::mimo2::nfc(text);
    let mut rest = normalized.as_str();
    while !rest.is_empty() {
        let n = piece_len(rest);
        for part in capped(&rest[..n]) {
            bpe_emit_piece(v, part.as_bytes(), out);
        }
        rest = &rest[n..];
    }
}

#[cfg(test)]
mod tests {
    use super::{capped, piece_len, PIECE_CAP};

    // Long pieces go to BPE in bounded runs, never inside a character.
    #[test]
    fn long_pieces_cap_on_character_boundaries() {
        let bangs = "!".repeat(2 * PIECE_CAP + 7);
        let runs: Vec<usize> = capped(&bangs).map(str::len).collect();
        assert_eq!(runs, [PIECE_CAP, PIECE_CAP, 7]);
        let hangul = "\u{ac00}".repeat(PIECE_CAP / 3 + 5);
        let parts: Vec<&str> = capped(&hangul).collect();
        assert!(parts
            .iter()
            .all(|p| p.len() <= PIECE_CAP && p.len() % 3 == 0));
        assert_eq!(parts.concat(), hangul);
        assert_eq!(capped("word").collect::<Vec<_>>(), ["word"]);
    }

    // tokenizers 0.22.2's Split pieces for the release's regex: whitespace
    // tails, digit triples, marks and ZWJ inside punctuation runs, \p{No}.
    #[test]
    fn split_matches_the_source_regex() {
        let cases: &[(&str, &[&str])] = &[
            ("  hello", &[" ", " hello"]),
            ("a   b", &["a", "  ", " b"]),
            ("x\n\n  y", &["x", "\n\n", " ", " y"]),
            ("abc   ", &["abc", "   "]),
            ("it's I'LL", &["it", "'s", " I", "'LL"]),
            ("1234567", &["123", "456", "7"]),
            ("<{~-<\u{200d}|a~", &["<{~-<\u{200d}|", "a", "~"]),
            ("!\u{301}x", &["!\u{301}x"]),
            ("\u{bd}\u{2153}\u{2460}", &["\u{bd}\u{2153}\u{2460}"]),
            ("a\u{200c}b c", &["a\u{200c}b", " c"]),
            ("tab\t\tx", &["tab", "\t", "\tx"]),
            (" \u{3000}x", &[" ", "\u{3000}x"]),
        ];
        for (text, pieces) in cases {
            let mut rest = *text;
            let mut got = Vec::new();
            while !rest.is_empty() {
                let n = piece_len(rest);
                got.push(&rest[..n]);
                rest = &rest[n..];
            }
            assert_eq!(got, *pieces, "{text:?}");
        }
    }
}
