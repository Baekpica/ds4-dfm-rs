//! Ling-3.1 ships the source tokenizer.json, not GGUF tokenizer metadata.
//! It is Ling-3.0's Bailing V3 BPE (identical merges): NFC, then the
//! `bailingmoe2` split — the Solar splitter with single-digit runs.
use super::{GgufFile, TokError, Vocab};
use serde_json::{json, Value};
use std::path::Path;

const BASE: usize = 156_891;
const ADDED: usize = 262;
/// The embedding has 31 padded rows past the last tokenizer entry. Empty
/// entries decode to nothing and never become invented input tokens.
const VOCAB: usize = 157_184;
const BOS: i32 = 156_891;
const EOS: i32 = 156_895;
const END_OF_TEXT: i32 = 156_892;
const PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

fn invalid() -> TokError {
    TokError::InvalidTokenizer("Ling-3.1 source tokenizer")
}

fn pipeline(v: &Value) -> Result<(), TokError> {
    let pre = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":PATTERN},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]});
    if v["version"] != "1.0"
        || v["pre_tokenizer"] != pre
        || v["normalizer"] != json!({"type":"NFC"})
    {
        return Err(invalid());
    }
    for key in ["truncation", "padding"] {
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
    let bytes = crate::ling31_vq::asset(root, "tokenizer.json").map_err(|_| invalid())?;
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
    Ok(super::glm_vq::Tables {
        tokens,
        merges,
        added,
    })
}

/// Role and tool controls of the Bailing V3 template (as Ling-3.0).
pub(super) fn specials(v: &mut Vocab, _g: &GgufFile) -> Result<(), TokError> {
    v.bos_id = v.lookup("<|startoftext|>")?;
    v.eos_id = v.lookup("<|role_end|>")?;
    v.eot_id = v.lookup("<|endoftext|>")?;
    v.im_start_id = v.lookup("<role>")?;
    v.im_content_id = v.lookup("</role>")?;
    v.im_end_id = v.eos_id;
    v.think_start_id = v.lookup("<think>")?;
    v.think_end_id = v.lookup("</think>")?;
    v.tool_call_start_id = v.lookup("<tool_call>")?;
    v.tool_call_end_id = v.lookup("</tool_call>")?;
    v.tool_response_start_id = v.lookup("<tool_response>")?;
    v.tool_response_end_id = v.lookup("</tool_response>")?;
    v.arg_key_start_id = v.lookup("<arg_key>")?;
    v.arg_key_end_id = v.lookup("</arg_key>")?;
    v.arg_value_start_id = v.lookup("<arg_value>")?;
    v.arg_value_end_id = v.lookup("</arg_value>")?;
    v.dsml_id = -1;
    if v.bos_id != BOS || v.eos_id != EOS || v.eot_id != END_OF_TEXT || v.tokens.len() != VOCAB {
        return Err(TokError::InvalidTokenizer("Ling-3.1 vocabulary"));
    }
    Ok(())
}
