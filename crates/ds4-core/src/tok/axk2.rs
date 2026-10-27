//! A.X-K2 ships the source tokenizer.json, not GGUF tokenizer metadata:
//! NFC, then CJK runs split out (Step 3.7's character classes), then Qwen2's
//! single-digit split over each piece, then byte-level BPE with string
//! merges. The 185 added tokens sit inside the 163,840-entry vocabulary
//! (whitespace runs 0-35 and the controls at the top) and are split out
//! before the BPE, special or not (`<think>` and the tool tokens are
//! `special: false`).
use super::{TokError, Vocab};
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::OnceLock;

const VOCAB: usize = 163_840;
const ADDED: usize = 185;
const CJK: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Chat protocol tokens (chat_template.jinja, generation_config.json).
const CONTROLS: [(i32, &str); 9] = [
    (163_691, "<|endoftext|>"),
    (163_702, "<|im_start|>"),
    (163_704, "<|im_end|>"),
    (163_705, "<think>"),
    (163_706, "</think>"),
    (163_707, "<tool_response>"),
    (163_708, "</tool_response>"),
    (163_709, "<tool_call>"),
    (163_710, "</tool_call>"),
];

fn invalid() -> TokError {
    TokError::InvalidTokenizer("A.X-K2 source tokenizer")
}

fn pipeline(v: &Value) -> Result<(), TokError> {
    let split = |pattern: &str| json!({"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false});
    let pre = json!({"type":"Sequence","pretokenizers":[
        split(CJK),
        split(PATTERN),
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]});
    let decoder =
        json!({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true});
    if v["version"] != "1.0"
        || v["pre_tokenizer"] != pre
        || v["normalizer"] != json!({"type":"NFC"})
        || v["decoder"] != decoder
    {
        return Err(invalid());
    }
    for key in ["post_processor", "truncation", "padding"] {
        if v.get(key) != Some(&Value::Null) {
            return Err(invalid());
        }
    }
    for (key, expected) in [
        ("type", json!("BPE")),
        ("ignore_merges", json!(false)),
        ("byte_fallback", json!(false)),
        ("fuse_unk", json!(false)),
        ("dropout", Value::Null),
        ("unk_token", Value::Null),
        ("continuing_subword_prefix", Value::Null),
        ("end_of_word_suffix", Value::Null),
    ] {
        if v["model"].get(key) != Some(&expected) {
            return Err(invalid());
        }
    }
    Ok(())
}

pub(super) fn tables(root: &Path) -> Result<super::glm_vq::Tables, TokError> {
    let bytes = crate::axk2_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    pipeline(&v)?;
    let vocab = v["model"]["vocab"].as_object().ok_or_else(invalid)?;
    if vocab.len() != VOCAB {
        return Err(invalid());
    }
    let mut tokens = vec![Vec::new(); VOCAB];
    for (word, id) in vocab {
        let id = id
            .as_u64()
            .filter(|&i| i < VOCAB as u64)
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
    let mut added = Vec::new();
    for entry in entries {
        let id = entry["id"]
            .as_u64()
            .filter(|&i| i < VOCAB as u64)
            .ok_or_else(invalid)? as usize;
        // Every added token keeps its own vocabulary entry, written raw.
        let word = entry["content"].as_str().ok_or_else(invalid)?;
        if tokens[id] != word.as_bytes() || added.contains(&id) {
            return Err(invalid());
        }
        for key in ["single_word", "lstrip", "rstrip", "normalized"] {
            if entry[key] != false {
                return Err(invalid());
            }
        }
        if !entry["special"].is_boolean() {
            return Err(invalid());
        }
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

/// Generation ends on `<|im_end|>` (generation_config's eos). The template
/// never adds a BOS.
pub(super) fn specials(v: &mut Vocab) -> Result<(), TokError> {
    for (id, text) in CONTROLS {
        if v.lookup(text)? != id {
            return Err(TokError::InvalidTokenizer("A.X-K2 control token IDs"));
        }
    }
    v.bos_id = -1;
    v.eos_id = v.lookup("<|im_end|>")?;
    v.eot_id = v.lookup("<|endoftext|>")?;
    v.im_start_id = v.lookup("<|im_start|>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.tool_response_start_id = v.lookup("<tool_response>")?;
    v.tool_response_end_id = v.lookup("</tool_response>")?;
    v.dsml_id = -1;
    if v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("A.X-K2 vocabulary"));
    }
    Ok(())
}

/// The source normalizes to NFC before both splits; ordinary spans only
/// (added tokens were split out on the raw text).
pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    static CJK_RUNS: OnceLock<Regex> = OnceLock::new();
    let cjk = CJK_RUNS.get_or_init(|| Regex::new(CJK).unwrap());
    let Ok(text) = std::str::from_utf8(text) else {
        super::bpe_tokenize_text_solar(v, text, out);
        return;
    };
    let normalized = crate::mimo2::nfc(text);
    super::step37::isolated(&normalized, cjk, |piece| {
        super::bpe_tokenize_text_solar(v, piece.as_bytes(), out)
    });
}
