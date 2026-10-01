//! /v1/models id from the GGUF path, and the OpenAI list JSON.

use crate::json::json_escape;

pub fn model_alias_disables_thinking(model: &str) -> bool {
    model == "deepseek-chat" || model == "k-exaone-236b-a23b-chat"
}

pub fn model_alias_enables_thinking(model: &str) -> bool {
    model == "deepseek-reasoner"
}

pub fn model_alias_known(id: &str) -> bool {
    matches!(
        id,
        "deepseek-v4-flash"
            | "deepseek-v4-pro"
            | "solar-open2-250b"
            | "motif-3"
            | "Motif-Technologies/Motif-3"
            | "Motif-3-Mixed-Quant-GGUF"
            | "k-exaone-236b-a23b"
            | "k-exaone-236b-a23b-chat"
            | "K-EXAONE-236B-A23B"
            | "LGAI-EXAONE/K-EXAONE-236B-A23B"
            | "dots3-note-prev"
            | "dots-studio/dots3-note-prev"
            | "dots3-note-prev-Mixed-Quant-GGUF"
            // Prism Bonsai 2 27B (qwen35); the names the sibling C server
            // answers to. The advertised id stays the GGUF stem, as it does
            // for every family here.
            | "prism-bonsai-2-27b"
            | "prism-bonsai-2-27b-chat"
            | "prism-bonsai-2-27b-no-think"
            | "prism-bonsai-2-27b-nothink"
            | "prism-bonsai-2-27b-reasoner"
            | "prism/bonsai-2-27b"
    )
}

pub fn model_id_known(advertised: &str, id: &str) -> bool {
    model_alias_known(id) || id == advertised
}

pub fn parent_is_generic_dir(parent: &str) -> bool {
    parent.is_empty()
        || parent == "."
        || parent == ".."
        || parent == "gguf"
        || parent == "GGUF"
        || parent == "models"
        || parent == "model"
        || parent == "weights"
        || parent == "artifacts"
        || parent == "tmp"
        || parent == "temp"
        || parent == "scratch"
        || parent == "data"
}

pub fn parent_is_gguf_artifact_dir(parent: &str) -> bool {
    if parent_is_generic_dir(parent) {
        return false;
    }
    if parent.contains("Mixed-Quant") {
        return true;
    }
    let n = parent.len();
    n > 4
        && parent.as_bytes()[n - 4].eq_ignore_ascii_case(&b'G')
        && parent.as_bytes()[n - 3].eq_ignore_ascii_case(&b'G')
        && parent.as_bytes()[n - 2].eq_ignore_ascii_case(&b'U')
        && parent.as_bytes()[n - 1].eq_ignore_ascii_case(&b'F')
}

fn strip_gguf_ext(name: &mut String) {
    let n = name.len();
    if n >= 5 && name.as_bytes()[n - 5] == b'.' {
        let ext = &name[n - 4..];
        if ext.eq_ignore_ascii_case("gguf") {
            name.truncate(n - 5);
        }
    }
}

fn strip_gguf_shard_suffix(name: &mut String) {
    let bytes = name.as_bytes();
    let n = bytes.len();
    let mut i = n;
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i == n || i < 4 || &bytes[i - 4..i] != b"-of-" {
        return;
    }
    let of = i - 4;
    let mut d = of;
    while d > 0 && bytes[d - 1].is_ascii_digit() {
        d -= 1;
    }
    if d == of || d == 0 || bytes[d - 1] != b'-' {
        return;
    }
    name.truncate(d - 1);
}

pub fn model_id_from_gguf_path(path: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let mut n = path.len();
    while n > 1 && path.as_bytes()[n - 1] == b'/' {
        n -= 1;
    }
    let path = &path[..n];
    let base = path.rfind('/').map(|i| i + 1).unwrap_or(0);
    let mut stem = path[base..].to_string();
    if stem.is_empty() || stem.len() >= 256 {
        return None;
    }
    strip_gguf_ext(&mut stem);
    strip_gguf_shard_suffix(&mut stem);
    if stem.is_empty() {
        return None;
    }
    if base > 0 {
        let parent_end = base - 1;
        let parent_start = path[..parent_end].rfind('/').map(|i| i + 1).unwrap_or(0);
        let parent = &path[parent_start..parent_end];
        if !parent.is_empty() && parent.len() < 256 && parent_is_gguf_artifact_dir(parent) {
            return Some(parent.to_string());
        }
    }
    if stem.len() >= 256 {
        return None;
    }
    Some(stem)
}

pub fn append_model_json_values(id: &str, name: &str, ctx: i32, default_tokens: i32) -> String {
    let max_completion = if default_tokens < ctx {
        default_tokens
    } else {
        ctx
    };
    format!(
        "{{\"id\":{},\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"ds4.c\",\"name\":{},\
         \"context_length\":{ctx},\"top_provider\":{{\"context_length\":{ctx},\
         \"max_completion_tokens\":{max_completion},\"is_moderated\":false}},\
         \"supported_parameters\":[\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\",\
         \"top_p\",\"top_k\",\"min_p\",\"stop\",\"seed\",\"stream\",\"reasoning_effort\"]}}",
        json_escape(id),
        json_escape(name)
    )
}

pub fn models_list_json(
    id: &str,
    name: &str,
    ctx: i32,
    default_tokens: i32,
    codex: Option<&str>,
) -> String {
    let mut b = String::from("{\"object\":\"list\",\"data\":[");
    b.push_str(&append_model_json_values(id, name, ctx, default_tokens));
    b.push(']');
    if let Some(arr) = codex {
        b.push_str(",\"models\":");
        b.push_str(arr);
    }
    b.push_str("}\n");
    b
}

pub fn model_one_json(id: &str, name: &str, ctx: i32, default_tokens: i32) -> String {
    let mut b = append_model_json_values(id, name, ctx, default_tokens);
    b.push('\n');
    b
}

pub fn json_models_array_dup(text: &str) -> Option<String> {
    let mut search = text;
    while let Some(at) = search.find("\"models\"") {
        let after = &search[at + 8..];
        let q = after.trim_start_matches(|c: char| c.is_ascii_whitespace());
        if !q.starts_with(':') {
            search = &search[at + 8..];
            continue;
        }
        let q = q[1..].trim_start_matches(|c: char| c.is_ascii_whitespace());
        if !q.starts_with('[') {
            search = &search[at + 8..];
            continue;
        }
        let start_off = text.len() - q.len();
        let bytes = q.as_bytes();
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        for (i, &c) in bytes.iter().enumerate() {
            if in_str {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    in_str = false;
                }
                continue;
            }
            if c == b'"' {
                in_str = true;
            } else if c == b'[' {
                depth += 1;
            } else if c == b']' {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start_off..start_off + i + 1].to_string());
                }
            }
        }
        return None;
    }
    None
}
