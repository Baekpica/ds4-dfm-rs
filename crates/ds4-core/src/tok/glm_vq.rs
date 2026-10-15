//! Source tokenizer.json remains the authority; no GGUF tokenizer rewrite.
use super::TokError;
use serde_json::{json, Value};
use std::path::Path;

const BASE: usize = 154820;
const ADDED: usize = 36;
const VOCAB: usize = 154880;
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub(super) struct Tables {
    pub(super) tokens: Vec<Vec<u8>>,
    pub(super) merges: Vec<Vec<u8>>,
    pub(super) added: Vec<usize>,
}

fn invalid() -> TokError {
    TokError::InvalidTokenizer("GLM VQ source tokenizer")
}

fn pipeline(v: &Value) -> Result<(), TokError> {
    let pre = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":PATTERN},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]});
    if v["version"] != "1.0" || v["pre_tokenizer"] != pre {
        return Err(invalid());
    }
    for key in ["normalizer", "truncation", "padding"] {
        if v.get(key) != Some(&Value::Null) {
            return Err(invalid());
        }
    }
    for key in ["decoder", "post_processor"] {
        let expected = json!({"type":"ByteLevel","add_prefix_space":true,
            "trim_offsets":key == "decoder","use_regex":true});
        if v[key] != expected {
            return Err(invalid());
        }
    }
    for (key, expected) in [
        ("type", json!("BPE")),
        ("ignore_merges", json!(true)),
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

pub(super) fn tables(root: &Path) -> Result<Tables, TokError> {
    let bytes = crate::glm_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    pipeline(&v)?;
    let vocab = v["model"]["vocab"].as_object().ok_or_else(invalid)?;
    if vocab.len() != BASE {
        return Err(invalid());
    }
    // The output matrix has 24 padded rows without tokenizer entries. Empty
    // entries decode to nothing and never become invented input tokens.
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
        if !tokens[id].is_empty() {
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
    Ok(Tables {
        tokens,
        merges,
        added,
    })
}
