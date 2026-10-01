use ds4_server::generate::chat_format_for_syntax;
use ds4_server::parse::ChatMsg;
use ds4_server::render::{render_chat, syntax_for_model_id, tool_start_marker};
use ds4_server::route::ThinkMode;
use ds4_server::tools::parse_generated_for_model_id;
use ds4_server::{SemAccum, ToolSchemaOrder};

const MODEL_ID: i32 = 13;

#[test]
fn naive_requires_official_input() {
    let msgs = [ChatMsg {
        role: "user".into(),
        content: "Hello".into(),
        ..Default::default()
    }];
    assert!(render_chat(syntax_for_model_id(MODEL_ID), &msgs, "", ThinkMode::None).is_err());
    assert_eq!(
        tool_start_marker(syntax_for_model_id(MODEL_ID)),
        "<tool_call>"
    );
}

#[test]
fn naive_splits_thought_and_xml() {
    let text = concat!(
        "<think>검토</think>\n<tool_call>\n<function=read>",
        "\n<parameter=path>notes.txt</parameter>\n</function>\n</tool_call>"
    );
    let parsed = parse_generated_for_model_id(MODEL_ID, text.as_bytes(), true, &[]);
    assert!(parsed.ok);
    assert_eq!(parsed.reasoning, "검토".as_bytes());
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "read");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parsed.calls[0].arguments).unwrap(),
        serde_json::json!({"path": "notes.txt"})
    );
}

#[test]
fn naive_preserves_param_lines() {
    let text = "<tool_call>\n<function=read>\n<parameter=path>\n notes.txt \n</parameter>\n</function>\n</tool_call>";
    let orders = [ToolSchemaOrder {
        name: "read".into(),
        prop: vec!["path".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let parsed = parse_generated_for_model_id(MODEL_ID, text.as_bytes(), true, &orders);
    assert!(parsed.ok && parsed.calls.len() == 1);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parsed.calls[0].arguments).unwrap(),
        serde_json::json!({"path": "\n notes.txt \n"})
    );
}

#[test]
fn naive_stream_handles_splits() {
    let text = "<think>검토</think>\n<tool_call>\n<function=read>\n<parameter=path>notes.txt</parameter>\n</function>\n</tool_call>".as_bytes();
    let format = chat_format_for_syntax(syntax_for_model_id(MODEL_ID));
    for width in 1..=text.len() {
        let mut acc = SemAccum::init(true, true, true, format, b"<|im_start|>assistant\n");
        assert!(!acc.thinking_inside());
        for chunk in text.chunks(width) {
            acc.feed(chunk, &[]);
        }
        assert!(!acc.thinking_inside(), "width {width}");
        assert!(acc.saw_tool_start && acc.saw_tool_end, "width {width}");
        assert_eq!(acc.verdict, None);
    }
}

#[test]
fn naive_plain_and_open_thought() {
    let plain = parse_generated_for_model_id(MODEL_ID, b"Hello", true, &[]);
    assert!(plain.ok && plain.content == b"Hello" && plain.reasoning.is_empty());
    let thought = parse_generated_for_model_id(MODEL_ID, b"<think>unfinished", true, &[]);
    assert!(thought.ok && thought.content.is_empty() && thought.reasoning == b"unfinished");
}
