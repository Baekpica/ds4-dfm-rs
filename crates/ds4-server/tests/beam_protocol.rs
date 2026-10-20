//! Beam-1.0's Harmony output protocol; all tests are model-free.
use ds4_server::generate::chat_format_for_syntax;
use ds4_server::parse::ChatMsg;
use ds4_server::render::{render_chat, tool_start_marker};
use ds4_server::tools::{parse_generated_for_model_id, parse_generated_for_response};
use ds4_server::{
    openai_sse_finish_live, openai_sse_stream_update, openai_stream_start, syntax_for_model_id,
    SemAccum, StreamReq, ThinkMode, ToolSchemaOrder, Writer, CREATED_TEST,
};
use serde_json::{json, Value};

const BEAM: i32 = 19;
const PROMPT: &[u8] = b"<|start|>user<|message|>hi<|end|><|start|>assistant";
const ANSWER: &str =
    "<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>final<|message|>Hello";
const CALL: &str = concat!(
    " to=lookup<|channel|>analysis <|constrain|>xml<|message|>",
    "<rfl:parameter name=\"text\">\n 123 \n</rfl:parameter>\n",
    "<rfl:parameter name=\"count\">\n2\n</rfl:parameter>"
);

fn orders() -> Vec<ToolSchemaOrder> {
    vec![ToolSchemaOrder {
        name: "lookup".into(),
        prop: vec!["text".into(), "count".into()],
        prop_type: vec!["string".into(), "integer".into()],
        ..Default::default()
    }]
}

// Two calls closed as the decode lanes leave them: `<|call|>` is fed at the stop.
fn calls() -> String {
    format!(
        "<|channel|>analysis<|message|>plan<|end|><|start|>assistant{CALL}<|end|><|start|>assistant \
         to=ping<|channel|>analysis <|constrain|>xml<|message|><|call|>"
    )
}

#[test]
fn beam_model_mapping() {
    let syntax = syntax_for_model_id(BEAM);
    assert_eq!(syntax as i32, BEAM);
    assert_eq!(tool_start_marker(syntax), "<|start|>assistant to");
    let msgs = [ChatMsg {
        role: "user".into(),
        content: "Hello".into(),
        ..Default::default()
    }];
    // Beam input is rendered by its official Jinja template only.
    assert!(render_chat(syntax, &msgs, "", ThinkMode::Low).is_err());
}

#[test]
fn beam_reasoning_and_answer() {
    let parsed = parse_generated_for_model_id(BEAM, ANSWER.as_bytes(), true, &[]);
    assert!(parsed.ok && parsed.calls.is_empty());
    assert_eq!(parsed.reasoning, b"plan");
    assert_eq!(parsed.content, b"Hello");
    // Unframed output is surfaced as raw text, not dropped.
    let plain = parse_generated_for_model_id(BEAM, b"Hello", true, &[]);
    assert!(!plain.ok && plain.content == b"Hello");
}

#[test]
fn beam_typed_parallel_calls() {
    let raw = calls();
    let parsed = parse_generated_for_model_id(BEAM, raw.as_bytes(), true, &orders());
    assert!(parsed.ok);
    assert_eq!(parsed.reasoning, b"plan");
    assert_eq!(parsed.calls.len(), 2);
    assert_eq!(parsed.calls[0].name, "lookup");
    let args: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
    // A declared string keeps its text unstripped; other values are JSON.
    assert_eq!(args, json!({"text": " 123 ", "count": 2}));
    assert_eq!(parsed.calls[1].name, "ping");
    assert_eq!(parsed.calls[1].arguments, "{}");
    let syntax = syntax_for_model_id(BEAM);
    let format = chat_format_for_syntax(syntax);
    let (no_tools, finish) = parse_generated_for_response(
        syntax,
        raw.as_bytes(),
        false,
        true,
        true,
        format,
        &orders(),
        "stop",
    );
    assert!(no_tools.calls.is_empty() && no_tools.reasoning == b"plan");
    assert_eq!(finish, "stop");
}

#[test]
fn beam_budget_cut_call_is_not_published() {
    let raw = format!("<|channel|>analysis<|message|>plan<|end|><|start|>assistant{CALL}");
    let syntax = syntax_for_model_id(BEAM);
    let (parsed, finish) = parse_generated_for_response(
        syntax,
        raw.as_bytes(),
        true,
        true,
        true,
        chat_format_for_syntax(syntax),
        &orders(),
        "length",
    );
    assert!(parsed.calls.is_empty());
    assert_eq!(finish, "length");
}

#[test]
fn beam_bytewise_semantic_markers() {
    let format = chat_format_for_syntax(syntax_for_model_id(BEAM));
    let raw = calls();
    for width in 1..=raw.len() {
        // Every Harmony reply opens with analysis.
        let mut acc = SemAccum::init(true, true, true, format, PROMPT);
        assert!(acc.thinking_inside(), "width {width}");
        for chunk in raw.as_bytes().chunks(width) {
            acc.feed(chunk, &[]);
        }
        assert!(!acc.thinking_inside(), "width {width}");
        assert!(acc.saw_tool_start && acc.saw_tool_end, "width {width}");
    }
}

#[test]
fn beam_call_terminator_is_not_counted() {
    // `<|call|>` is a stop token: the parser sees it, usage counts it as
    // little as `<|return|>`.
    let format = chat_format_for_syntax(syntax_for_model_id(BEAM));
    let mut acc = SemAccum::init(true, true, true, format, PROMPT);
    let raw = calls();
    acc.feed(raw.strip_suffix("<|call|>").unwrap().as_bytes(), &[]);
    acc.feed_stop(b"<|call|>", &[]);
    assert_eq!((acc.completion, acc.reasoning_tokens), (1, 1));
    assert!(acc.saw_tool_end);
}

fn events(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn beam_openai_stream_every_split() {
    let syntax = syntax_for_model_id(BEAM);
    let r = StreamReq {
        syntax,
        chat_format: chat_format_for_syntax(syntax),
        think_mode: ThinkMode::Low,
        ..Default::default()
    };
    let raw = ANSWER.as_bytes();
    for width in 1..=raw.len() {
        let mut w = Writer::new(CREATED_TEST);
        let mut state = openai_stream_start(&r);
        let mut current = Vec::new();
        for chunk in raw.chunks(width) {
            current.extend(chunk);
            assert!(openai_sse_stream_update(
                &mut w, &r, "job", &mut state, &current, false
            ));
        }
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut state,
            &current,
            "stop",
            2,
            5,
            &[]
        ));
        let (mut text, mut thinking) = (String::new(), String::new());
        for e in events(&w.out) {
            let delta = &e["choices"][0]["delta"];
            text.push_str(delta["content"].as_str().unwrap_or(""));
            thinking.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
        }
        assert_eq!(
            (thinking.as_str(), text.as_str()),
            ("plan", "Hello"),
            "width {width}"
        );
    }
}
