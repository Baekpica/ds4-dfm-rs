//! Step's official XML output envelope is independent of its Jinja input.
use ds4_server::generate::chat_format_for_syntax;
use ds4_server::tools::parse_generated_for_model_id;
use ds4_server::{syntax_for_model_id, SemAccum, ToolSchemaOrder};

const STEP: i32 = ds4_core::Variant::Step37Flash as i32;
const PROMPT: &[u8] = b"<|im_start|>assistant\n<think>\n";
const OUTPUT: &[u8] = b"Need weather.\n</think>\n<tool_call>\n<function=weather>\n<parameter=city>\nSeoul\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";

#[test]
fn step_tool_output() {
    let orders = [ToolSchemaOrder {
        name: "weather".into(),
        prop: vec!["city".into(), "days".into()],
        prop_type: vec!["string".into(), "integer".into()],
        ..Default::default()
    }];
    let parsed = parse_generated_for_model_id(STEP, OUTPUT, true, &orders);
    assert!(parsed.ok);
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "weather");
    assert_eq!(parsed.calls[0].arguments, r#"{"city": "Seoul", "days": 2}"#);
    assert!(parsed.content.is_empty());
    assert_eq!(parsed.reasoning.trim_ascii(), b"Need weather.");
}

#[test]
fn step_tool_stream_splits() {
    let format = chat_format_for_syntax(syntax_for_model_id(STEP));
    for width in 1..=OUTPUT.len() {
        let mut acc = SemAccum::init(true, true, true, format, PROMPT);
        assert!(acc.thinking_inside());
        for chunk in OUTPUT.chunks(width) {
            acc.feed(chunk, &[]);
        }
        assert!(!acc.thinking_inside(), "width {width}");
        assert!(acc.saw_tool_start && acc.saw_tool_end, "width {width}");
        assert_eq!(acc.verdict, None);
    }
}
