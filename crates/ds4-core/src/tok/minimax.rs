//! MiniMax-M3.1 ships the source tokenizer.json, not GGUF tokenizer
//! metadata: NFC, then the o200k pre-split (Inkling's pieces; HF writes it as
//! an inverted `Removed` split, which keeps exactly the matches), then
//! byte-level BPE with string merges. 61 added tokens are split out before
//! the BPE, special or not (`<mm:think>`, the tool-call tokens and the
//! `]<]minimax[>[` namespace are `special: false`).
use super::{TokError, Vocab};
use serde_json::{json, Value};
use std::path::Path;

const BASE: usize = 200_000;
const ADDED: usize = 61;
/// The embedding and head have three padded rows past the last added token.
/// Empty entries decode to nothing and never become invented input tokens.
const VOCAB: usize = 200_064;
const PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
);
/// Chat protocol tokens (chat_template.jinja, tokenizer_config.json).
const CONTROLS: [(i32, &str); 9] = [
    (200_019, "]~b]"),
    (200_020, "[e~["),
    (200_025, "]<]image[>["),
    (200_026, "]<]video[>["),
    (200_034, "]~!b["),
    (200_052, "<tool_call>"),
    (200_053, "</tool_call>"),
    (200_059, "<mm:think>"),
    (200_060, "</mm:think>"),
];

fn invalid() -> TokError {
    TokError::InvalidTokenizer("MiniMax source tokenizer")
}

fn pipeline(v: &Value) -> Result<(), TokError> {
    let pre = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":PATTERN},"behavior":"Removed","invert":true},
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
    // No `ignore_merges` key: whole-vocabulary pieces get no shortcut.
    if v["model"].get("ignore_merges").is_some() {
        return Err(invalid());
    }
    for (key, expected) in [
        ("type", json!("BPE")),
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
    let bytes = crate::minimax_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
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
    let mut added = Vec::new();
    for entry in entries {
        let id = entry["id"]
            .as_u64()
            .filter(|&i| (BASE as u64..(BASE + ADDED) as u64).contains(&i))
            .ok_or_else(invalid)? as usize;
        let word = entry["content"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        if !tokens[id].is_empty() || vocab.contains_key(word) || !entry["special"].is_boolean() {
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
    // Merges are "left right" strings; byte-level symbols never hold a space.
    let merges = v["model"]["merges"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|entry| {
            let pair = entry.as_str().ok_or_else(invalid)?;
            let (a, b) = pair.split_once(' ').ok_or_else(invalid)?;
            if a.is_empty()
                || b.is_empty()
                || b.contains(' ')
                || !vocab.contains_key(a)
                || !vocab.contains_key(b)
                || !vocab.contains_key(&format!("{a}{b}"))
            {
                return Err(invalid());
            }
            Ok(pair.as_bytes().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(super::glm_vq::Tables {
        tokens,
        merges,
        added,
    })
}

/// Generation ends on `[e~[`, which closes every turn; thinking is the
/// `<mm:think>` span and tool calls the `<tool_call>` block.
pub(super) fn specials(v: &mut Vocab) -> Result<(), TokError> {
    for (id, text) in CONTROLS {
        if v.lookup(text)? != id {
            return Err(TokError::InvalidTokenizer("MiniMax control token IDs"));
        }
    }
    v.bos_id = -1;
    v.eos_id = v.lookup("[e~[")?;
    v.think_start_id = v.lookup("<mm:think>")?;
    v.think_end_id = v.lookup("</mm:think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.dsml_id = -1;
    if v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("MiniMax vocabulary"));
    }
    Ok(())
}

/// NFC on valid UTF-8 spans (tokenizer.json `normalizer`), then the o200k
/// pieces and byte-level BPE (Inkling's splitter, no merge shortcut).
pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    use unicode_normalization::UnicodeNormalization;

    let Ok(text) = std::str::from_utf8(text) else {
        super::inkling::encode(v, text, out);
        return;
    };
    let normalized: String = text.nfc().collect();
    super::inkling::encode(v, normalized.as_bytes(), out);
}
