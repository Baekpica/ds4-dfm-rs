//! C↔Rust four-surface stream projectors (tape oracle, no model).

use ds4_server::{
    anthropic_final_response, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, project_anthropic_thinking, project_openai_chat_thinking,
    project_openai_chat_utf8, project_openai_completion, project_responses_thinking,
    responses_final_response, utf8_stream_safe_len, ChatFormat, ModelSyntax, ReqKind, StreamReq,
    ThinkBlock, ThinkMode, ToolCall, Writer, CREATED_TEST, TEST_MSG_ID, TEST_RESP_ID, TEST_RS_ID,
};

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_STREAM_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/stream_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/stream_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_bytes(args: &[&str]) -> Vec<u8> {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run stream_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn c_str(args: &[&str]) -> String {
    String::from_utf8(c_bytes(args)).expect("oracle utf8")
}

fn assert_bytes_eq(label: &str, rust: &[u8], c: &[u8]) {
    if rust != c {
        let r = String::from_utf8_lossy(rust);
        let k = String::from_utf8_lossy(c);
        panic!("{label} mismatch\n--- rust ---\n{r}\n--- c ---\n{k}");
    }
}

fn count_substr(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

fn sse_assert_every_data_object(out: &str, must_contain: &str, must_not: Option<&str>) {
    let mut saw_done = false;
    let mut records = 0;
    let mut rest = out;
    while let Some(i) = rest.find("data: ") {
        rest = &rest[i + 6..];
        let end = rest.find("\n\n").unwrap_or(rest.len());
        let rec = &rest[..end];
        if rec == "[DONE]" {
            saw_done = true;
            assert!(!rest[end..].contains("data: "));
        } else {
            records += 1;
            assert!(
                rec.contains(must_contain),
                "missing {must_contain} in {rec}"
            );
            if let Some(bad) = must_not {
                assert!(!rec.contains(bad), "found {bad} in {rec}");
            }
        }
        rest = if end < rest.len() {
            &rest[end + 2..]
        } else {
            ""
        };
    }
    assert!(records > 0);
    assert!(saw_done);
}

fn sse_validate_anthropic(out: &str) {
    let start = out.find("event: message_start").expect("message_start");
    let stop = out.find("event: message_stop").expect("message_stop");
    assert!(start < stop);
    assert_eq!(
        count_substr(out, "event: content_block_start"),
        count_substr(out, "event: content_block_stop")
    );
    let mdelta = out.find("event: message_delta").expect("message_delta");
    assert!(mdelta < stop);
    let mut rest = out;
    while let Some(i) = rest.find("event: ") {
        rest = &rest[i + 7..];
        let n = rest.find(['\r', '\n']).unwrap_or(rest.len());
        let name = &rest[..n];
        let data = rest.find("data: ").expect("data after event");
        let rec_end = rest[data..]
            .find("\n\n")
            .map(|e| data + e)
            .unwrap_or(rest.len());
        let rec = &rest[data..rec_end];
        let typekey = format!("\"type\":\"{name}\"");
        assert!(rec.contains(&typekey), "missing {typekey} in {rec}");
        rest = if rec_end < rest.len() {
            &rest[rec_end + 2..]
        } else {
            ""
        };
    }
}

fn sse_validate_responses(out: &str) {
    let created = out.find("\"type\":\"response.created\"").expect("created");
    let completed = out
        .find("\"type\":\"response.completed\"")
        .expect("completed");
    assert!(created < completed);
    assert_eq!(
        count_substr(out, "\"type\":\"response.output_item.added\""),
        count_substr(out, "\"type\":\"response.output_item.done\"")
    );
    let mut expect = 0;
    let mut rest = out;
    while let Some(i) = rest.find("\"sequence_number\":") {
        rest = &rest[i + "\"sequence_number\":".len()..];
        let n: i32 = rest
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap();
        assert_eq!(n, expect);
        expect += 1;
    }
    assert!(expect > 0);
}

#[test]
fn utf8_safe_len_matches_c() {
    let cases = [
        (0usize, 2usize, 0, "c3a9"),
        (0, 1, 0, "c3"),
        (0, 3, 0, "616263"),
        (0, 4, 0, "636166c3"),
        (0, 5, 0, "636166c3a9"),
    ];
    for (start, limit, final_, hex) in cases {
        let rust = utf8_stream_safe_len(&decode_hex(hex), start, limit, final_ != 0);
        let c: usize = c_str(&[
            "utf8-safe",
            &start.to_string(),
            &limit.to_string(),
            &final_.to_string(),
            hex,
        ])
        .trim()
        .parse()
        .unwrap();
        assert_eq!(rust, c, "utf8-safe {hex} start={start} limit={limit}");
    }
}

fn decode_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn openai_chat_thinking_tape_matches_c() {
    let rust = project_openai_chat_thinking(CREATED_TEST);
    let c = c_bytes(&["openai-chat-tape"]);
    assert_bytes_eq("openai-chat-tape", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    sse_assert_every_data_object(&out, "\"object\":\"chat.completion.chunk\"", None);
    let reason = out.find("\"reasoning_content\":").expect("reason");
    let content = out.find("\"content\":\"Answer").expect("content");
    assert!(out.contains("\"delta\":{\"role\":\"assistant\"}"));
    assert!(reason < content);
    assert!(out.contains("\"finish_reason\":\"stop\""));
    assert!(!out.contains("</think>"));
}

#[test]
fn openai_tool_before_think_streams_as_call() {
    let raw = b"<tool_call><function=get_weather><parameter=city>Seoul</parameter></function></tool_call>";
    let r = StreamReq {
        think_mode: ThinkMode::High,
        has_tools: true,
        chat_format: ChatFormat::Qwen4Exp,
        syntax: ModelSyntax::Mimo2,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    for n in 1..=raw.len() {
        assert!(openai_sse_stream_update(
            &mut w,
            &r,
            "chatcmpl_mimo",
            &mut st,
            &raw[..n],
            false,
        ));
    }
    let calls = [ToolCall {
        name: "get_weather".into(),
        arguments: r#"{"city":"Seoul"}"#.into(),
        ..Default::default()
    }];
    assert!(openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_mimo",
        &mut st,
        raw,
        "tool_calls",
        100,
        18,
        &calls,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(out.contains("\"tool_calls\""), "{out}");
    assert!(!out.contains("reasoning_content"), "{out}");
}

#[test]
fn qwen_and_step_tool_inside_open_think_stays_reasoning() {
    let raw = b"<tool_call><function=get_weather><parameter=city>Seoul</parameter></function></tool_call>";
    for syntax in [ModelSyntax::Qwen4Exp, ModelSyntax::Step37] {
        let r = StreamReq {
            model: format!("{syntax:?}"),
            think_mode: ThinkMode::High,
            has_tools: true,
            chat_format: ChatFormat::Qwen4Exp,
            syntax,
            ..Default::default()
        };
        let mut w = Writer::new(CREATED_TEST);
        let mut st = openai_stream_start(&r);
        for n in 1..=raw.len() {
            assert!(openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_qwen",
                &mut st,
                &raw[..n],
                false,
            ));
        }
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "chatcmpl_qwen",
            &mut st,
            raw,
            "stop",
            100,
            18,
            &[],
        ));
        let out = String::from_utf8(w.out).unwrap();
        assert!(out.contains("reasoning_content"), "{out}");
        assert!(!out.contains("tool_calls"), "{out}");
    }
}

#[test]
fn openai_chat_utf8_tape_matches_c() {
    let rust = project_openai_chat_utf8(CREATED_TEST);
    let c = c_bytes(&["openai-chat-utf8"]);
    assert_bytes_eq("openai-chat-utf8", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\u{e9}"));
    assert!(!out.contains("\u{fffd}"));
    assert!(!rust.windows(2).any(|w| w == [0xc3, b'"']));
}

#[test]
fn openai_completion_tape_matches_c() {
    let rust = project_openai_completion(CREATED_TEST);
    let c = c_bytes(&["openai-completion-tape"]);
    assert_bytes_eq("openai-completion-tape", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    sse_assert_every_data_object(
        &out,
        "\"object\":\"text_completion\"",
        Some("chat.completion.chunk"),
    );
    assert!(out.contains("\"choices\":[{\"text\":\"Hel\""));
    assert!(out.contains("\"finish_reason\":\"stop\""));
    assert!(!out.contains("\"delta\""));
}

#[test]
fn anthropic_thinking_tape_matches_c() {
    let rust = project_anthropic_thinking(CREATED_TEST);
    let c = c_bytes(&["anthropic-tape"]);
    assert_bytes_eq("anthropic-tape", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    sse_validate_anthropic(&out);
    let thinking = out.find("\"thinking\":\"plan\"").expect("thinking");
    let signature = out.find("\"type\":\"signature_delta\"").expect("signature");
    let text = out.find("\"text\":\"Answer").expect("text");
    assert!(thinking < signature && signature < text);
    assert!(out.contains("\"stop_reason\":\"end_turn\""));
    assert!(!out.contains("</think>"));
}

#[test]
fn responses_thinking_tape_matches_c() {
    let rust = project_responses_thinking(CREATED_TEST);
    let c = c_bytes(&["responses-tape"]);
    assert_bytes_eq("responses-tape", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    sse_validate_responses(&out);
    let summary = out
        .find("\"type\":\"response.reasoning_summary_text.delta\"")
        .expect("summary");
    let text = out
        .find("\"type\":\"response.output_text.delta\"")
        .expect("text");
    assert!(summary < text);
    assert!(out.contains("\"status\":\"completed\""));
    assert!(!out.contains("</think>"));
}

#[test]
fn buffered_finals_match_c() {
    let mut chat = StreamReq::default();
    chat.stream = false;
    let rust = final_response(
        &chat,
        "chatcmpl_buf",
        b"Hello world.",
        None,
        "stop",
        4,
        4,
        CREATED_TEST,
        false,
        &[],
        &[],
    );
    let c = c_bytes(&["final", "openai-chat", "Hello world.", "stop"]);
    assert_bytes_eq("final-openai-chat", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\"object\":\"chat.completion\""));
    assert!(out.contains("\"role\":\"assistant\",\"content\":\"Hello world.\""));
    assert!(out.contains("\"finish_reason\":\"stop\""));

    let mut comp = StreamReq::default();
    comp.kind = ReqKind::Completion;
    comp.stream = false;
    let rust = final_response(
        &comp,
        "cmpl_buf",
        b"Hello world.",
        None,
        "length",
        4,
        4,
        CREATED_TEST,
        false,
        &[],
        &[],
    );
    let c = c_bytes(&["final", "openai-completion", "Hello world.", "length"]);
    assert_bytes_eq("final-completion", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\"object\":\"text_completion\""));
    assert!(out.contains("\"choices\":[{\"text\":\"Hello world.\""));
    assert!(out.contains("\"finish_reason\":\"length\""));

    let rust = anthropic_final_response(
        &chat,
        "msg_buf",
        b"Hello world.",
        None,
        "length",
        None,
        4,
        4,
        false,
        &[],
    );
    let c = c_bytes(&["final", "anthropic", "Hello world.", "length"]);
    assert_bytes_eq("final-anthropic", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\"type\":\"message\""));
    assert!(out.contains("\"stop_reason\":\"max_tokens\""));

    let rust = responses_final_response(
        &chat,
        b"Hello world.",
        None,
        "stop",
        ThinkBlock::Closed,
        4,
        4,
        0,
        CREATED_TEST,
        false,
        TEST_RESP_ID,
        TEST_RS_ID,
        TEST_MSG_ID,
        &[],
    );
    let c = c_bytes(&["final", "responses", "Hello world.", "stop"]);
    assert_bytes_eq("final-responses", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\"object\":\"response\""));
    assert!(out.contains("\"status\":\"completed\""));
    assert!(out.contains("Hello world."));
}
