//! C↔Rust four-surface stream projectors (tape oracle, no model).

use ds4_server::{
    anthropic_final_response, anthropic_sse_start_live, anthropic_sse_stream_update,
    final_response, openai_sse_finish_live, openai_sse_stream_update, openai_stream_start,
    project_anthropic_thinking, project_openai_chat_thinking, project_openai_chat_utf8,
    project_openai_completion, project_responses_thinking, responses_final_response, think_end,
    utf8_stream_safe_len, ChatFormat, ModelSyntax, ReqKind, StreamReq, ThinkBlock, ThinkMode,
    ToolCall, Writer, CREATED_TEST, TEST_MSG_ID, TEST_RESP_ID, TEST_RS_ID,
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

/// A model can emit a byte that starts no valid UTF-8 sequence mid-text (GLM-5.3
/// VQ wrote "서", a lone 0xEB, then "ving"). The buffered bodies are built as
/// strings, so such a byte must become U+FFFD instead of panicking the
/// request into a 500; the C reference passes the byte through raw.
#[test]
fn buffered_finals_replace_invalid_utf8() {
    let text: &[u8] = b"\xec\x84\x9c\xebving";
    let want = "서\u{fffd}ving";
    let mut chat = StreamReq::default();
    chat.stream = false;
    let bodies = [
        final_response(
            &chat,
            "chatcmpl_bad",
            text,
            Some(text),
            "stop",
            4,
            4,
            CREATED_TEST,
            false,
            &[],
            &[],
        ),
        anthropic_final_response(
            &chat,
            "msg_bad",
            text,
            Some(text),
            "stop",
            None,
            4,
            4,
            false,
            &[],
        ),
        responses_final_response(
            &chat,
            text,
            Some(text),
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
        ),
    ];
    for body in bodies {
        let body = String::from_utf8(body).expect("the body is valid UTF-8");
        assert!(body.contains(want), "{body}");
    }
}

/// Beam served through Hermes wrote two of a Hangul syllable's three bytes,
/// then English; openai-python decodes each SSE line as strict UTF-8 and
/// dropped the stream. That delta carries U+FFFD now, while a syllable cut
/// between two tokens still streams whole.
#[test]
fn stream_deltas_replace_invalid_utf8() {
    let r = StreamReq {
        chat_format: ChatFormat::Beam,
        syntax: ModelSyntax::BeamVq,
        ..Default::default()
    };
    let raw: &[u8] = b"<|channel|>analysis<|message|>\xec\x84\x9c\xeb\xb2ving<|end|>";
    let head = raw.len() - b"\x9c\xeb\xb2ving<|end|>".len();
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    for (cut, last) in [(head, false), (head + 3, false), (raw.len(), true)] {
        assert!(openai_sse_stream_update(
            &mut w,
            &r,
            "chatcmpl_bad",
            &mut st,
            &raw[..cut],
            last,
        ));
    }
    let out = String::from_utf8(w.out).expect("every SSE line is valid UTF-8");
    assert!(out.contains("\"reasoning_content\":\"서\""), "{out}");
    assert!(
        out.contains("\"reasoning_content\":\"\u{fffd}ving\""),
        "{out}"
    );
}

/// A cut character that nothing can complete becomes U+FFFD as well: one
/// before a closer, and one left when the generation ends, streamed or
/// buffered (the C reference drops both).
#[test]
fn stranded_utf8_tails_become_replacement() {
    let r = StreamReq {
        chat_format: ChatFormat::Beam,
        syntax: ModelSyntax::BeamVq,
        ..Default::default()
    };
    let raw: &[u8] = b"<|channel|>analysis<|message|>a\xeb<|end|>\
        <|start|>assistant<|channel|>final<|message|>b\xeb";
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    let n = raw.len();
    for (cut, last) in [(n - 1, false), (n, false), (n, true)] {
        assert!(openai_sse_stream_update(
            &mut w,
            &r,
            "chatcmpl_cut",
            &mut st,
            &raw[..cut],
            last,
        ));
    }
    let out = String::from_utf8(w.out).expect("every SSE line is valid UTF-8");
    assert!(out.contains("\"reasoning_content\":\"a\u{fffd}\""), "{out}");
    assert!(out.contains("\"content\":\"b\""), "{out}");
    assert!(out.contains("\"content\":\"\u{fffd}\""), "{out}");
    let chat = StreamReq {
        stream: false,
        ..Default::default()
    };
    let body = final_response(
        &chat,
        "chatcmpl_cut",
        b"b\xeb",
        Some(b"a\xeb"),
        "length",
        4,
        4,
        CREATED_TEST,
        false,
        &[],
        &[],
    );
    let body = String::from_utf8(body).expect("the body is valid UTF-8");
    assert!(
        body.contains("b\u{fffd}") && body.contains("a\u{fffd}"),
        "{body}"
    );
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

#[test]
fn bare_invoke_block_holds_back_and_finishes_with_the_call() {
    // The V4.1 artifact dropped the wrapper in a live open-grok session
    // (01a127cf, 2026-10-11): the block opened straight with <invoke>, the
    // live start detector missed it, and the markup streamed as visible
    // content while the finish still served the call.  The detector now
    // knows the invoke spellings; the block must be held back and the call
    // emitted at the finish.
    let raw = b"Here you go.\n\n<invoke name=\"web_fetch\">\n\
        <parameter name=\"url\">https://example.com</parameter>\n\
        </invoke>\n</tool_calls>";
    let r = StreamReq {
        think_mode: ThinkMode::None,
        has_tools: true,
        chat_format: ChatFormat::DeepSeek,
        syntax: ModelSyntax::DeepSeek,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    // "Here you go." streams as one delta; a chunk ending mid-marker is
    // held back; the full text holds the block in suppress mode.
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_bare",
        &mut st,
        &raw[..12],
        false,
    ));
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_bare",
        &mut st,
        &raw[..17],
        false,
    ));
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_bare",
        &mut st,
        raw,
        false,
    ));
    let calls = [ToolCall {
        name: "web_fetch".into(),
        arguments: r#"{"url": "https://example.com"}"#.into(),
        ..Default::default()
    }];
    assert!(openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_bare",
        &mut st,
        raw,
        "tool_calls",
        12,
        20,
        &calls,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(
        !out.contains("<invoke"),
        "the invoke markup leaked as content: {out}"
    );
    assert!(
        out.contains("Here you go."),
        "the pre-block text must stream: {out}"
    );
    assert!(
        out.contains("\"tool_calls\""),
        "the finish must emit the call: {out}"
    );
    assert!(
        out.contains("web_fetch"),
        "the call name must be present: {out}"
    );
}

const SPACED_CALLS_START: &str = "<\u{ff5c}DSML\u{ff5c} calls>";
const SPACED_CALLS_END: &str = "</\u{ff5c}DSML\u{ff5c} calls>";
const SPACED_INVOKE_START: &str = "<\u{ff5c}DSML\u{ff5c} invoke";
const SPACED_INVOKE_END: &str = "</\u{ff5c}DSML\u{ff5c} invoke>";

#[test]
fn bare_parameter_close_finishes_the_value_live() {
    // The artifact's deep-night shape (fable5.md, 10-11 late night; engine
    // f302be7): the parameter close goes bare while the wrapper and invoke
    // opens stay DSML.  Before the alias the live stream found no close and
    // pushed the rest of the block -- the closing tag and the next invoke
    // included -- as the first argument's value; the engine's assistant
    // looped five steps on a tag-laden query (fable5.md:11081).
    let raw = format!(
        "Checking.\n\n{SPACED_CALLS_START}\n\
         {SPACED_INVOKE_START} name=\"web_search\">\n\
         <parameter name=\"query\" string=\"true\">DGX Spark lowest price</parameter>\n\
         {SPACED_INVOKE_END}\n\
         {SPACED_INVOKE_START} name=\"web_search\">\n\
         <parameter name=\"query\" string=\"true\">DGX Spark second</parameter>\n\
         {SPACED_INVOKE_END}\n\
         {SPACED_CALLS_END}"
    );
    let r = StreamReq {
        think_mode: ThinkMode::None,
        has_tools: true,
        chat_format: ChatFormat::DeepSeek,
        syntax: ModelSyntax::DeepSeek,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_bareclose",
        &mut st,
        raw.as_bytes(),
        false,
    ));
    let calls = [
        ToolCall {
            name: "web_search".into(),
            arguments: r#"{"query": "DGX Spark lowest price"}"#.into(),
            ..Default::default()
        },
        ToolCall {
            name: "web_search".into(),
            arguments: r#"{"query": "DGX Spark second"}"#.into(),
            ..Default::default()
        },
    ];
    assert!(openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_bareclose",
        &mut st,
        raw.as_bytes(),
        "tool_calls",
        10,
        40,
        &calls,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(
        !out.contains("</parameter>"),
        "the bare close leaked into a value: {out}"
    );
    assert!(
        out.contains("DGX Spark lowest price"),
        "the first value must stream: {out}"
    );
    assert!(
        out.contains("DGX Spark second"),
        "the second value must stream: {out}"
    );
}

#[test]
fn parameter_without_string_attr_streams_as_string() {
    // Engine f302be7: a bare parameter carries no string attribute and the
    // engine defaults it to string (matching the final parser).  Before the
    // alias the live stream failed the call at the parameter open.
    let raw = format!(
        "go\n\n{SPACED_CALLS_START}\n\
         {SPACED_INVOKE_START} name=\"web_search\">\n\
         <parameter name=\"query\">DGX Spark</parameter>\n\
         {SPACED_INVOKE_END}\n\
         {SPACED_CALLS_END}"
    );
    let r = StreamReq {
        think_mode: ThinkMode::None,
        has_tools: true,
        chat_format: ChatFormat::DeepSeek,
        syntax: ModelSyntax::DeepSeek,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_nostring",
        &mut st,
        raw.as_bytes(),
        false,
    ));
    let calls = [ToolCall {
        name: "web_search".into(),
        arguments: r#"{"query": "DGX Spark"}"#.into(),
        ..Default::default()
    }];
    assert!(openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_nostring",
        &mut st,
        raw.as_bytes(),
        "tool_calls",
        10,
        20,
        &calls,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(out.contains("query"), "the parameter must stream: {out}");
    assert!(out.contains("DGX Spark"), "the value must stream: {out}");
}

/// f302be7 (2026-10-11): in non-thinking mode the artifact sometimes samples
/// one more close marker ahead of its narration.  The stream pushes the
/// narration but never the marker itself; a marker tail cut between two tokens
/// is held back until the next chunk proves it.
#[test]
fn stray_close_marker_never_streams_as_content() {
    let marker = think_end(ChatFormat::DeepSeek);
    let prefix = "Let me check the price.";
    let suffix = "Actually, first the specs.";
    let raw = format!("{prefix}{marker}{suffix}");
    let r = StreamReq {
        think_mode: ThinkMode::None,
        has_tools: true,
        chat_format: ChatFormat::DeepSeek,
        syntax: ModelSyntax::DeepSeek,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    // A chunk ending mid-marker holds the partial back.
    let head = &raw[..prefix.len() + 5];
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_stray",
        &mut st,
        head.as_bytes(),
        false,
    ));
    let mid = String::from_utf8(w.out.clone()).unwrap();
    assert!(mid.contains(prefix), "the narration must stream: {mid}");
    assert!(
        !mid.contains(&marker[..5]),
        "a partial marker leaked: {mid}"
    );
    // The completed marker is skipped; the narration after it streams.
    assert!(openai_sse_stream_update(
        &mut w,
        &r,
        "chatcmpl_stray",
        &mut st,
        raw.as_bytes(),
        true,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(out.contains(prefix), "{out}");
    assert!(out.contains(suffix), "{out}");
    assert!(!out.contains(marker), "the marker leaked: {out}");
}

/// The same stray-marker suppression on the anthropic surface (f302be7,
/// server_anthropic_stream.c:200-206).
#[test]
fn anthropic_stray_close_marker_never_streams_as_content() {
    let marker = think_end(ChatFormat::DeepSeek);
    let prefix = "Let me check the price.";
    let suffix = "Actually, first the specs.";
    let raw = format!("{prefix}{marker}{suffix}");
    let r = StreamReq {
        think_mode: ThinkMode::None,
        has_tools: true,
        chat_format: ChatFormat::DeepSeek,
        syntax: ModelSyntax::DeepSeek,
        ..Default::default()
    };
    let mut w = Writer::new(CREATED_TEST);
    let mut st = anthropic_sse_start_live(&mut w, &r, "job", 2);
    assert!(anthropic_sse_stream_update(
        &mut w,
        &r,
        "job",
        &mut st,
        raw.as_bytes(),
        true,
    ));
    let out = String::from_utf8(w.out).unwrap();
    assert!(out.contains(prefix), "{out}");
    assert!(out.contains(suffix), "{out}");
    assert!(!out.contains(marker), "the marker leaked: {out}");
}
