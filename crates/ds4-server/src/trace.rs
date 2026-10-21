//! C `--trace FILE` (ds4_help.c:323, "Write prompt, token, and DSML debug
//! trace"; the server opens the file `w` at ds4_server.c:16010-16019 and logs
//! "tracing session to %s").  One section per request, no KV state read.
//!
//! The reference writes it from four points: the identity and the rendered
//! prompt at `trace_begin` (:11967-12017), the prompt token ids with the
//! engine's own text for each (`trace_write_token`, :11936-11941), the cache
//! diagnostics (:12000-12015), and the parsed message with its tool calls at
//! `trace_finish` (:12044-12107).
//!
//! Named deviations, so the file is read for what it is:
//! - one section, written at the finish, instead of writes at each point: the
//!   lines are grouped rather than interleaved with generation;
//! - the raw request body block (:12008-12016) is absent -- this host's parse
//!   does not retain the body;
//! - the live-vs-prompt token window (:11920-11950) is absent: the host keeps
//!   no per-position live id array.  The prompt's own ids are written, taken by
//!   tokenizing the rendered prompt (`GenerateEngine::tokenize_rendered_chat`),
//!   which is the reference's "prompt" column;
//! - the replay account is the host's cached read/write pair, not the
//!   reference's mem/disk/canonical/missing split.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::parse::ToolCall;
use crate::route::{Api, ReqKind, ThinkMode};

static PATH: OnceLock<PathBuf> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Set once at boot from `--trace FILE`; a later call is ignored (one server,
/// one trace file, the reference's `s->trace`).
pub fn set_path(path: PathBuf) {
    let _ = PATH.set(path);
}

pub fn enabled() -> bool {
    PATH.get().is_some()
}

/// One request's section.  Borrowed only: the caller writes it after the turn,
/// when the prompt, the account and the generated text all exist.
pub struct TraceRequest<'a> {
    pub api: Api,
    pub kind: ReqKind,
    pub stream: bool,
    pub model_id: i32,
    pub model: &'a str,
    pub tools: usize,
    pub think: ThinkMode,
    pub prompt: &'a [u8],
    /// The prompt's own token ids; empty when the caller did not tokenize.
    pub tokens: &'a [i32],
    pub cached: i32,
    pub prompt_tokens: i32,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub finish: &'a str,
    pub completion: i32,
    pub saw_tool_start: bool,
    pub saw_tool_end: bool,
    pub elapsed_sec: f64,
    pub content: &'a [u8],
    pub reasoning: &'a [u8],
    pub calls: &'a [ToolCall],
}

fn api_name(api: Api) -> &'static str {
    match api {
        Api::Openai => "openai",
        Api::Anthropic => "anthropic",
        Api::Responses => "responses",
    }
}

fn kind_name(kind: ReqKind) -> &'static str {
    match kind {
        ReqKind::Chat => "chat",
        ReqKind::Completion => "completion",
    }
}

fn think_name(think: ThinkMode) -> &'static str {
    match think {
        ThinkMode::None => "none",
        ThinkMode::Low => "low",
        ThinkMode::High => "high",
        ThinkMode::Max => "max",
    }
}

/// `trace_write_token` (:11936-11941): the engine's own text for one id,
/// escaped so a control token cannot break the line layout.
fn token_text_escaped(text: &[u8]) -> String {
    let mut out = String::with_capacity(text.len());
    for &byte in text {
        match byte {
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b'\\' => out.push_str("\\\\"),
            0x20..=0x7e => out.push(char::from(byte)),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out
}

/// Write one section.  `text_of` is the engine's token decoder; the caller
/// passes it because the trace module holds no engine.
pub fn write_request(req: &TraceRequest<'_>, text_of: &mut dyn FnMut(i32) -> Vec<u8>) {
    let Some(path) = PATH.get() else {
        return;
    };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let id = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let mut out = String::new();
    out.push_str(&format!("\n===== request {id} {now:.3} =====\n"));
    out.push_str(&format!(
        "kind: {}\napi: {}\nmodel: {} (id {})\nstream: {}\ntools: {}\nthink_mode: {}\n\
         prompt_tokens: {}\ncached_tokens: {}\nmax_tokens: {}\ntemperature: {:.3}\n\
         top_k: {}\ntop_p: {:.3}\nmin_p: {:.3}\nseed: {}\n",
        kind_name(req.kind),
        api_name(req.api),
        req.model,
        req.model_id,
        u8::from(req.stream),
        req.tools,
        think_name(req.think),
        req.prompt_tokens,
        req.cached,
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
        req.min_p,
        req.seed,
    ));
    out.push_str("--- rendered prompt ---\n");
    out.push_str(&String::from_utf8_lossy(req.prompt));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("--- prompt tokens ---\n");
    for (pos, &token) in req.tokens.iter().enumerate() {
        out.push_str(&format!(
            "{pos:>7} {token:>8} {}\n",
            token_text_escaped(&text_of(token))
        ));
    }
    out.push_str("--- generated text ---\n");
    out.push_str(&String::from_utf8_lossy(req.reasoning));
    out.push_str(&String::from_utf8_lossy(req.content));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!(
        "--- parsed message ---\nfinish: {}\ngenerated_tokens: {}\ndsml_start: {}\n\
         dsml_end: {}\nelapsed_sec: {:.3}\n",
        req.finish,
        req.completion,
        u8::from(req.saw_tool_start),
        u8::from(req.saw_tool_end),
        req.elapsed_sec,
    ));
    if !req.reasoning.is_empty() {
        out.push_str(&format!(
            "\nreasoning:\n{}\n",
            String::from_utf8_lossy(req.reasoning)
        ));
    }
    if !req.content.is_empty() {
        out.push_str(&format!(
            "\ncontent:\n{}\n",
            String::from_utf8_lossy(req.content)
        ));
    }
    for (i, call) in req.calls.iter().enumerate() {
        out.push_str(&format!(
            "\ntool_call[{i}]:\nid: {}\nname: {}\narguments:\n{}\n",
            call.id, call.name, call.arguments
        ));
    }
    let _ = file.write_all(out.as_bytes());
    let _ = file.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One section carries the identity, the rendered prompt, the prompt token
    /// ids with the engine's text for each, and the parsed message with its
    /// calls (ds4_server.c:11976-12017 and :12044-12107).
    #[test]
    fn a_section_carries_the_prompt_tokens_and_parsed_message() {
        let path = std::env::temp_dir().join(format!("ds4-trace-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&path);
        set_path(path.clone());

        let prompt = b"<|user|>hi<|assistant|>";
        let tokens = [7i32, 8];
        let calls = [ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"pwd"}"#.into(),
        }];
        let mut text_of = |token: i32| match token {
            7 => b"<|user|>".to_vec(),
            _ => b"hi".to_vec(),
        };
        write_request(
            &TraceRequest {
                api: Api::Openai,
                kind: ReqKind::Chat,
                stream: false,
                model_id: 5,
                model: "deepseek-v4.1-flash",
                tools: 1,
                think: ThinkMode::Low,
                prompt,
                tokens: &tokens,
                cached: 0,
                prompt_tokens: tokens.len() as i32,
                max_tokens: 64,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
                finish: "tool_calls",
                completion: 12,
                saw_tool_start: true,
                saw_tool_end: true,
                elapsed_sec: 1.5,
                content: b"",
                reasoning: b"thinking",
                calls: &calls,
            },
            &mut text_of,
        );

        let got = std::fs::read_to_string(&path).expect("the trace file is written");
        assert!(got.contains("kind: chat"), "{got}");
        assert!(
            got.contains("--- rendered prompt ---\n<|user|>hi<|assistant|>\n"),
            "{got}"
        );
        assert!(got.contains("      0        7 <|user|>"), "{got}");
        assert!(got.contains("finish: tool_calls"), "{got}");
        assert!(got.contains("dsml_start: 1\ndsml_end: 1"), "{got}");
        assert!(
            got.contains("tool_call[0]:\nid: call_1\nname: bash"),
            "{got}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
