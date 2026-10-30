//! Beam-1.0 ships the source tokenizer.json, not GGUF tokenizer metadata.
//! It is Qwen2's byte-level BPE: NFC, then the single-digit Qwen2 split
//! (`mimo2::encode`), with the Harmony controls as added tokens.
use super::{TokError, Vocab};
use serde_json::{json, Value};
use std::path::Path;

const BASE: usize = 151_643;
const ADDED: usize = 38;
/// The embedding and head have 255 padded rows past the last added token.
/// Empty entries decode to nothing and never become invented input tokens.
const VOCAB: usize = 151_936;
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Harmony controls (tokenizer_config.json `response_template`).
const CONTROLS: [(i32, &str); 8] = [
    (151_643, "<|endoftext|>"),
    (151_669, "<|start|>"),
    (151_670, "<|end|>"),
    (151_671, "<|message|>"),
    (151_672, "<|channel|>"),
    (151_673, "<|constrain|>"),
    (151_674, "<|return|>"),
    (151_675, "<|call|>"),
];

fn invalid() -> TokError {
    TokError::InvalidTokenizer("Beam source tokenizer")
}

fn pipeline(v: &Value) -> Result<(), TokError> {
    let byte_level =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false});
    let pre = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":PATTERN},"behavior":"Isolated","invert":false},
        byte_level
    ]});
    if v["version"] != "1.0"
        || v["pre_tokenizer"] != pre
        || v["normalizer"] != json!({"type":"NFC"})
        || v["decoder"] != byte_level
        || v["post_processor"] != byte_level
    {
        return Err(invalid());
    }
    for key in ["truncation", "padding"] {
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
    let bytes = crate::beam_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
    qwen2(&bytes, ADDED, VOCAB)
}

/// Qwen2's byte-level BPE tokenizer.json: the 151,643-word base, then
/// `added` added tokens, padded to `rows` embedding rows (Beam, MiMo).
pub(super) fn qwen2(
    bytes: &[u8],
    added: usize,
    rows: usize,
) -> Result<super::glm_vq::Tables, TokError> {
    let v: Value = serde_json::from_slice(bytes).map_err(|_| invalid())?;
    pipeline(&v)?;
    let vocab = v["model"]["vocab"].as_object().ok_or_else(invalid)?;
    if vocab.len() != BASE {
        return Err(invalid());
    }
    let mut tokens = vec![Vec::new(); rows];
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
    if entries.len() != added {
        return Err(invalid());
    }
    let ids = BASE as u64..(BASE + added) as u64;
    let mut added = Vec::new();
    for entry in entries {
        let id = entry["id"]
            .as_u64()
            .filter(|i| ids.contains(i))
            .ok_or_else(invalid)? as usize;
        let word = entry["content"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        if !tokens[id].is_empty() {
            return Err(invalid());
        }
        for key in ["single_word", "lstrip", "rstrip", "normalized"] {
            if entry[key] != false {
                return Err(invalid());
            }
        }
        // Harmony controls are `special: false`; every added token is still
        // split out before the BPE, special or not.
        if !entry["special"].is_boolean() {
            return Err(invalid());
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

/// Generation ends on `<|return|>` (final answer) or `<|call|>` (tool call);
/// `<|end|>` closes a message inside the turn and is not a stop.
pub(super) fn specials(v: &mut Vocab) -> Result<(), TokError> {
    for (id, text) in CONTROLS {
        if v.lookup(text)? != id {
            return Err(TokError::InvalidTokenizer("Beam control token IDs"));
        }
    }
    v.bos_id = -1;
    v.eos_id = v.lookup("<|return|>")?;
    v.eot_id = v.lookup("<|call|>")?;
    v.dsml_id = -1;
    if v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("Beam vocabulary"));
    }
    Ok(())
}
