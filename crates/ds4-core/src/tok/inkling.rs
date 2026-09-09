//! Source JSON BPE. Only the pinned Inkling tokenizer pipeline is accepted.
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Value};

use super::{bpe_emit_piece, tokenize_rendered_chat, ChatThinkMode, GgufFile, TokError, Vocab};
use crate::TokenBuffer;

const BASE_VOCAB: usize = 199998;
const SPECIAL_COUNT: usize = 60;
const VOCAB: usize = BASE_VOCAB + SPECIAL_COUNT;
const PIECES: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+",
);
const WHITESPACE: &str = r"|\s+(?!\S)|\s+";
type Tables = (Vec<Vec<u8>>, Vec<Vec<u8>>);

const SPECIAL_WORDS: &[(usize, &str)] = &[
    (199998, "<|unused|>"),
    (199999, "<|endoftext|>"),
    (200000, "<|message_user|>"),
    (200001, "<|message_model|>"),
    (200002, "<|message_system|>"),
    (200003, "<|message_tool|>"),
    (200004, "<|content_text|>"),
    (200005, "<|content_image|>"),
    (200006, "<|content_model_end_sampling|>"),
    (200008, "<|content_thinking|>"),
    (200010, "<|end_message|>"),
    (200020, "<|content_audio_input|>"),
    (200022, "<|content_tool_error|>"),
    (200023, "<|audio|>"),
    (200024, "<|content_xml|>"),
    (200028, "<|begin_of_text|>"),
    (200043, "<|audio_end|>"),
    (200049, "<|content_invoke_tool_json|>"),
    (200057, "<|content_invoke_tool_text|>"),
];

fn special_word(id: usize) -> String {
    SPECIAL_WORDS
        .iter()
        .find(|(i, _)| *i == id)
        .map(|(_, word)| (*word).into())
        .unwrap_or_else(|| format!("<|unused_{id}|>"))
}

fn invalid(key: &'static str) -> TokError {
    TokError::InvalidTokenizer(key)
}

fn pipeline(config: &Value) -> Result<(), TokError> {
    let pattern = format!("{PIECES}{WHITESPACE}");
    let expected = json!({
        "type": "Sequence", "pretokenizers": [
            {"type":"Split", "pattern":{"Regex":pattern}, "behavior":"Isolated", "invert":false},
            {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false}
        ]
    });
    if config.get("pre_tokenizer") != Some(&expected) {
        return Err(invalid("pre_tokenizer"));
    }
    for key in ["decoder", "post_processor"] {
        let expected = json!({"type":"ByteLevel", "add_prefix_space":true,
            "trim_offsets":key == "decoder", "use_regex":true});
        if config.get(key) != Some(&expected) {
            return Err(invalid(key));
        }
    }
    if config.get("normalizer") != Some(&Value::Null) {
        return Err(invalid("normalizer"));
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
        if config["model"].get(key) != Some(&expected) {
            return Err(invalid(key));
        }
    }
    Ok(())
}

pub(super) fn tables(g: &GgufFile) -> Result<Tables, TokError> {
    let data = g
        .get_string("inkling.tokenizer.json")
        .ok_or(TokError::MissingTable("inkling.tokenizer.json"))?;
    let config: Value =
        serde_json::from_slice(data).map_err(|_| invalid("inkling.tokenizer.json"))?;
    pipeline(&config)?;
    let vocab = config["model"]["vocab"]
        .as_object()
        .ok_or_else(|| invalid("vocab"))?;
    if vocab.len() != BASE_VOCAB {
        return Err(invalid("vocab count"));
    }
    let mut tokens = vec![Vec::new(); VOCAB];
    for (word, value) in vocab {
        let id = value
            .as_u64()
            .filter(|&id| id < BASE_VOCAB as u64)
            .ok_or_else(|| invalid("vocab id"))? as usize;
        if word.is_empty() || !tokens[id].is_empty() {
            return Err(invalid("vocab duplicate/empty"));
        }
        tokens[id] = word.as_bytes().to_vec();
    }
    let added = config["added_tokens"]
        .as_array()
        .ok_or_else(|| invalid("added_tokens"))?;
    if added.len() != SPECIAL_COUNT {
        return Err(invalid("added_tokens count"));
    }
    for value in added {
        let id = value["id"]
            .as_u64()
            .filter(|&id| (BASE_VOCAB as u64..VOCAB as u64).contains(&id))
            .ok_or_else(|| invalid("added_tokens id"))? as usize;
        let word = value["content"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("added_tokens content"))?;
        for (key, expected) in [
            ("single_word", false),
            ("lstrip", false),
            ("rstrip", false),
            ("normalized", false),
            ("special", true),
        ] {
            if value[key].as_bool() != Some(expected) {
                return Err(invalid("added_tokens flags"));
            }
        }
        // IDs are part of the model contract, including the unused image/audio
        // embedding placeholders. Swapped or duplicate strings must fail.
        if word != special_word(id) || !tokens[id].is_empty() || vocab.contains_key(word) {
            return Err(invalid("added_tokens contract"));
        }
        tokens[id] = word.as_bytes().to_vec();
    }
    let raw = config["model"]["merges"]
        .as_array()
        .ok_or_else(|| invalid("merges"))?;
    let mut merges = Vec::with_capacity(raw.len());
    for pair in raw {
        let pair = pair
            .as_array()
            .filter(|p| p.len() == 2)
            .ok_or_else(|| invalid("merge pair"))?;
        let left = pair[0].as_str().ok_or_else(|| invalid("merge left"))?;
        let right = pair[1].as_str().ok_or_else(|| invalid("merge right"))?;
        if !vocab.contains_key(left)
            || !vocab.contains_key(right)
            || !vocab.contains_key(&format!("{left}{right}"))
        {
            return Err(invalid("merge vocabulary"));
        }
        merges.push(format!("{left} {right}").into_bytes());
    }
    Ok((tokens, merges))
}

pub(super) fn specials(v: &mut Vocab) -> Result<(), TokError> {
    v.bos_id = v.lookup("<|begin_of_text|>")?;
    v.eos_id = v.lookup("<|content_model_end_sampling|>")?;
    v.system_id = v.lookup("<|message_system|>")?;
    v.user_id = v.lookup("<|message_user|>")?;
    v.assistant_id = v.lookup("<|message_model|>")?;
    v.tool_id = v.lookup("<|message_tool|>")?;
    v.im_content_id = v.lookup("<|content_text|>")?;
    v.im_end_id = v.lookup("<|end_message|>")?;
    v.think_start_id = v.lookup("<|content_thinking|>")?;
    v.think_end_id = v.im_end_id;
    v.tool_call_start_id = v.lookup("<|content_invoke_tool_json|>")?;
    v.tool_call_end_id = v.im_end_id;
    v.dsml_id = -1;
    for (id, token) in v.tokens.iter().enumerate().skip(BASE_VOCAB) {
        v.user_defined.insert(token.clone(), id as i32);
        v.user_defined_max_len = v.user_defined_max_len.max(token.len() as u32);
        v.user_defined_first[token[0] as usize] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_pipeline() -> Value {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/inkling/tokenizer-vectors.json"
        ))
        .unwrap();
        fixture["pipeline"].clone()
    }

    #[test]
    fn accepts_source_pipeline() {
        pipeline(&source_pipeline()).unwrap();
    }

    #[test]
    fn rejects_pipeline_drift() {
        for (pointer, value) in [
            ("/normalizer", json!({"type": "NFC"})),
            ("/model/ignore_merges", json!(false)),
            ("/model/byte_fallback", json!(true)),
            ("/model/dropout", json!(0.1)),
            (
                "/pre_tokenizer/pretokenizers/1/add_prefix_space",
                json!(true),
            ),
            ("/pre_tokenizer/pretokenizers/1/use_regex", json!(true)),
            ("/decoder/trim_offsets", json!(false)),
            ("/post_processor/trim_offsets", json!(true)),
        ] {
            let mut config = source_pipeline();
            *config.pointer_mut(pointer).unwrap() = value;
            assert!(pipeline(&config).is_err(), "{pointer}");
        }
    }
}

fn piece_len(text: &str) -> usize {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let re = SPLIT.get_or_init(|| Regex::new(&format!(r"\A(?:{PIECES})")).expect("Inkling regex"));
    if let Some(m) = re.find(text) {
        return m.end();
    }
    // The remaining alternatives are \s+(?!\S)|\s+. If non-whitespace
    // follows, leave one trailing space for the next word's optional prefix.
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
    if end > 0 {
        return end;
    }
    text.chars().next().expect("nonempty input").len_utf8()
}

pub(super) fn encode(v: &Vocab, mut text: &[u8], out: &mut Vec<i32>) {
    while !text.is_empty() {
        let (valid, invalid) = match std::str::from_utf8(text) {
            Ok(s) => (s, 0),
            Err(e) => (
                std::str::from_utf8(&text[..e.valid_up_to()]).unwrap(),
                e.error_len().unwrap_or(text.len() - e.valid_up_to()),
            ),
        };
        let mut part = valid;
        while !part.is_empty() {
            let n = piece_len(part);
            bpe_emit_piece(v, &part.as_bytes()[..n], out);
            part = &part[n..];
        }
        let n = valid.len();
        if invalid > 0 {
            bpe_emit_piece(v, &text[n..n + invalid], out);
        }
        text = &text[n + invalid..];
    }
}

pub(super) fn effort(v: &Vocab, out: &mut TokenBuffer, mode: ChatThinkMode) {
    let level = match mode {
        ChatThinkMode::None => "0",
        ChatThinkMode::Low => "0.2",
        ChatThinkMode::High => "0.9",
        ChatThinkMode::Max => "0.99",
    };
    out.push(v.system_id);
    out.push(v.im_content_id);
    encode(
        v,
        format!("Thinking effort level: {level}").as_bytes(),
        &mut out.tokens,
    );
    out.push(v.im_end_id);
}

pub(super) fn message(
    v: &Vocab,
    out: &mut TokenBuffer,
    role: &str,
    content: &[u8],
) -> Result<(), TokError> {
    let token = match role {
        "system" | "developer" => v.system_id,
        "user" => v.user_id,
        "assistant" => v.assistant_id,
        "tool" | "function" => v.tool_id,
        _ => return Err(invalid("message role")),
    };
    out.push(token);
    out.push(v.im_content_id);
    tokenize_rendered_chat(v, content, &mut out.tokens);
    out.push(v.im_end_id);
    if role == "assistant" {
        out.push(v.eos_id);
    }
    Ok(())
}
