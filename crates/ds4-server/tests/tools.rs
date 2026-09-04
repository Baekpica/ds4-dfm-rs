//! Generated-message parse + SemAccum goldens from `ds4_server.c` unit tests.

use ds4_server::{
    append_tool_calls_json, assign_tool_ids, parse_generated_for_response, parse_generated_message,
    ChatFormat, ModelSyntax, SemAccum, ToolCall, ToolSchemaOrder,
};

const DSML_START: &str = "<｜DSML｜tool_calls>";
const DSML_END: &str = "</｜DSML｜tool_calls>";
const DSML_INVOKE: &str = "<｜DSML｜invoke";
const DSML_INVOKE_END: &str = "</｜DSML｜invoke>";
const DSML_PARAM: &str = "<｜DSML｜parameter";
const DSML_PARAM_END: &str = "</｜DSML｜parameter>";

#[test]
fn parse_dots3_tool_call_message() {
    let text = b"<think>\nplan\n</think>\n\nChecking now.\n\
        <dots_function_call>\n\
        <invoke name=\"get_weather\">\n\
        <parameter name=\"city\">\nSeoul\n</parameter>\n\
        <parameter name=\"days\">\n3\n</parameter>\n\
        </invoke>\n\
        </dots_function_call>";
    let p = parse_generated_message(ModelSyntax::Dots3, text, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, "{\"city\": \"Seoul\", \"days\": 3}");
    assert_eq!(p.content, b"\n\nChecking now.");
    assert_eq!(p.reasoning, b"\nplan\n");

    let multi = b"</think>\n\n<dots_function_call>\n\
        <invoke name=\"a\">\n<parameter name=\"x\">\n1\n</parameter>\n</invoke>\n\
        <invoke name=\"b\">\n<parameter name=\"y\">\ntwo\n</parameter>\n</invoke>\n\
        </dots_function_call>";
    let p = parse_generated_message(ModelSyntax::Dots3, multi, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].arguments, "{\"x\": 1}");
    assert_eq!(p.calls[1].arguments, "{\"y\": \"two\"}");
}

#[test]
fn parse_motif3_tool_call_message() {
    let text = b"<think>need weather</think>\n\
        <tool_call>{\"name\": \"get_weather\", \"arguments\": \
        {\"city\": \"Seoul\"}}</tool_call>";
    let p = parse_generated_message(ModelSyntax::Motif3, text, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.reasoning, b"need weather");
    assert!(p.content.is_empty());
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert!(p.calls[0].arguments.contains("\"city\""));
    assert!(p.raw_tool_text.starts_with("\n<tool_call>"));
}

#[test]
fn parse_exaone_two_hermes_calls() {
    let text = b"<think>need weather</think>\n\n\
        <tool_call>{\"name\":\"get_weather\",\"arguments\":\
        {\"city\":\"Seoul\"}}</tool_call>\n\
        <tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>";
    let p = parse_generated_message(ModelSyntax::Exaone, text, true, ChatFormat::Exaone, &[]);
    assert!(p.ok);
    assert_eq!(p.reasoning, b"need weather");
    assert!(p.content.is_empty());
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, "{\"city\":\"Seoul\"}");
    assert_eq!(p.calls[1].name, "get_time");
}

#[test]
fn parse_k2_horizon_ifm_think_not_exaone() {
    let text = b"<ifm|think>\nplan\n</ifm|think>hello\
        <ifm|tool_calls>\n\
        <ifm|tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Seoul\"}}</ifm|tool_call>\n\
        <ifm|tool_call>{\"name\":\"get_time\",\"arguments\":{}}</ifm|tool_call>\n\
        </ifm|tool_calls>";
    let p = parse_generated_message(
        ModelSyntax::K2Horizon,
        text,
        true,
        ChatFormat::K2Horizon,
        &[],
    );
    assert!(p.ok);
    assert_eq!(p.reasoning, b"\nplan\n");
    assert_eq!(p.content, b"hello");
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, r#"{"city":"Seoul"}"#);
    assert_eq!(p.calls[1].name, "get_time");
    assert!(p.raw_dsml.starts_with("<ifm|tool_calls>"));
    assert!(!String::from_utf8_lossy(&p.content).contains("<|user|>"));
}

#[test]
fn parse_dsml_nested_parameters() {
    let generated = format!(
        "review done\n\n{DSML_START}\n{DSML_INVOKE} name=\"edit\">\n\
         {DSML_PARAM} name=\"path\">/private/tmp/tetris.c{DSML_PARAM_END}\n\
         {DSML_PARAM} name=\"edits\">\n\
         {DSML_PARAM} name=\"oldText\" string=\"true\">old &lt;text&gt;{DSML_PARAM_END}\n\
         {DSML_PARAM} name=\"newText\" string=\"true\">new text{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let p = parse_generated_message(
        ModelSyntax::DeepSeek,
        generated.as_bytes(),
        false,
        ChatFormat::DeepSeek,
        &[],
    );
    assert!(p.ok);
    assert_eq!(p.content, b"review done");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "edit");
    assert!(p.calls[0]
        .arguments
        .contains("\"path\": \"/private/tmp/tetris.c\""));
    assert!(p.calls[0].arguments.contains("\"edits\": {"));
    assert!(
        p.calls[0].arguments.contains("\"oldText\":\"old <text>\""),
        "{}",
        p.calls[0].arguments
    );
    assert!(p.calls[0].arguments.contains("\"newText\":\"new text\""));
}

#[test]
fn parse_solar_native_tool_call() {
    let generated = format!(
        "I should inspect the directory.<|think:end|><|tool_call:start|>list_files\n\
         <|tool_arg:start|>path<|tool_arg:value|>/tmp<|tool_arg:end|>\n\
         <|tool_arg:start|>recursive<|tool_arg:value|>false<|tool_arg:end|>\n\
         <|tool_arg:start|>literal<|tool_arg:value|>false<|tool_arg:end|>\n\
         <|tool_call:end|>"
    );
    let orders = [ToolSchemaOrder {
        name: "list_files".into(),
        prop: vec!["path".into(), "recursive".into(), "literal".into()],
        prop_type: vec!["string".into(), "boolean".into(), "string".into()],
        ..Default::default()
    }];
    let p = parse_generated_message(
        ModelSyntax::SolarOpen2,
        generated.as_bytes(),
        true,
        ChatFormat::SolarOpen2,
        &orders,
    );
    assert!(p.ok);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "list_files");
    assert!(p.calls[0].arguments.contains("\"path\": \"/tmp\""));
    assert!(p.calls[0].arguments.contains("\"recursive\": false"));
    assert!(p.calls[0].arguments.contains("\"literal\": \"false\""));
    assert!(p.content.is_empty());
    assert_eq!(p.reasoning, b"I should inspect the directory.");
    assert!(p.raw_dsml.starts_with("<|tool_call:start|>"));
}

#[test]
fn parse_qwen_native_tool_call() {
    let generated = b"<think>\nNeed weather.\n</think>\n\n\
        <tool_call>\n<function=weather>\n\
        <parameter=city>\nSeoul\n</parameter>\n\
        <parameter=days>\n2\n</parameter>\n\
        </function>\n</tool_call>";
    let orders = [ToolSchemaOrder {
        name: "weather".into(),
        prop: vec!["city".into(), "days".into()],
        prop_type: vec!["string".into(), "integer".into()],
        ..Default::default()
    }];
    let p = parse_generated_message(
        ModelSyntax::Qwen4Exp,
        generated,
        true,
        ChatFormat::Qwen4Exp,
        &orders,
    );
    assert!(p.ok);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "weather");
    assert_eq!(p.calls[0].arguments, "{\"city\": \"Seoul\", \"days\": 2}");
    assert!(p.content.is_empty());
    assert!(p.reasoning.windows(13).any(|s| s == b"Need weather."));
    assert!(p.raw_dsml.contains("<function=weather>"));
}

#[test]
fn parse_glm53_native_tool_call() {
    let generated = b"<think>need bash</think>OK\n\n\
        <tool_call>bash\
        <arg_key>command</arg_key><arg_value>echo hi</arg_value>\
        <arg_key>timeout</arg_key><arg_value>10</arg_value>\
        </tool_call>";
    let p = parse_generated_message(
        ModelSyntax::Glm53,
        generated,
        true,
        ChatFormat::DeepSeek,
        &[],
    );
    assert!(p.ok);
    assert_eq!(p.reasoning, b"need bash");
    assert_eq!(p.content, b"OK");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "bash");
    assert_eq!(
        p.calls[0].arguments,
        r#"{"command": "echo hi", "timeout": "10"}"#
    );
    assert!(p.raw_dsml.starts_with("\n\n<tool_call>bash"));
}

#[test]
fn qwen_stream_observes_native_tool_markers_without_dsml_verdict() {
    let mut acc = SemAccum::init(true, true, false, ChatFormat::Qwen4Exp, b"");
    let first = acc.feed(b"<tool_call>\n<function=weather>\n", &[]);
    assert!(first.entered_tool_block && acc.saw_tool_start);
    let last = acc.feed(b"</function>\n</tool_call>", &[]);
    assert!(last.tool_block_closed && acc.saw_tool_end);
    assert_eq!(acc.verdict, None);
}

#[test]
fn k2_stream_leaves_thinking_before_observing_tool_block() {
    let prompt = b"<|ifm|im_start|>assistant\n<ifm|think>\n";
    let mut acc = SemAccum::init(true, true, true, ChatFormat::K2Horizon, prompt);
    assert!(acc.thinking_inside());
    acc.feed(b"plan</ifm|thi", &[]);
    let closed = acc.feed(b"nk>", &[]);
    assert!(!closed.entered_tool_block);
    assert!(!acc.thinking_inside());
    let first = acc.feed(b"<ifm|tool_calls>\n<ifm|tool_call>{}</ifm|tool_call>", &[]);
    assert!(first.entered_tool_block && acc.saw_tool_start);
    let last = acc.feed(b"</ifm|tool_calls>", &[]);
    assert!(last.tool_block_closed && acc.saw_tool_end);
}

#[test]
fn no_tools_does_not_extract_calls() {
    let text = format!(
        "hi\n\n{DSML_START}\n{DSML_INVOKE} name=\"bash\">\n\
         {DSML_PARAM} name=\"command\" string=\"true\">ls{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let (p, finish) = parse_generated_for_response(
        ModelSyntax::DeepSeek,
        text.as_bytes(),
        false,
        true,
        false,
        ChatFormat::DeepSeek,
        &[],
        "stop",
    );
    assert!(p.ok);
    assert!(p.calls.is_empty());
    assert_eq!(finish, "stop");
}

#[test]
fn append_tool_calls_json_uses_job_fallback_id() {
    let calls = [ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"/tmp\"}".into(),
        ..Default::default()
    }];
    let mut out = Vec::new();
    append_tool_calls_json(&mut out, &calls, "chatcmpl-1");
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"id\":\"chatcmpl-1_tool_0\""));
    assert!(s.contains("\"type\":\"function\""));
    assert!(s.contains("\"name\":\"edit\""));
    assert!(s.contains("\"arguments\":\"{\\\"path\\\":\\\"/tmp\\\"}\""));
}

#[test]
fn sem_accum_dsml_closes_and_no_tools_cuts() {
    let block = format!(
        "{DSML_START}\n{DSML_INVOKE} name=\"bash\">\n\
         {DSML_PARAM} name=\"command\" string=\"true\">ls{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    let f = acc.feed(block.as_bytes(), &[]);
    assert!(acc.saw_tool_start);
    assert!(acc.saw_tool_end);
    assert_eq!(acc.verdict, Some("tool_calls"));
    assert!(f.tool_block_closed);

    let mut cut = SemAccum::init(true, false, false, ChatFormat::DeepSeek, b"");
    let f = cut.feed(b"hello <|", &[]);
    assert!(!f.hit_stop);
    let f = cut.feed(format!("{DSML_START} tail").as_bytes(), &[]);
    assert!(f.hit_stop);
    assert!(f.tool_syntax_cut);
    assert_eq!(cut.verdict, Some("stop"));
    assert!(!cut
        .text
        .windows(DSML_START.len())
        .any(|w| w == DSML_START.as_bytes()));
}

#[test]
fn assign_tool_ids_fills_empty() {
    let mut calls = vec![
        ToolCall {
            name: "a".into(),
            arguments: "{}".into(),
            ..Default::default()
        },
        ToolCall {
            id: "supplied".into(),
            ..Default::default()
        },
        ToolCall::default(),
    ];
    assign_tool_ids(&mut calls, "call_");
    for id in [&calls[0].id, &calls[2].id] {
        assert_eq!(id.len(), 37);
        assert!(id.starts_with("call_"));
        assert!(id[5..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
    assert_ne!(calls[0].id, calls[2].id);
    assert_eq!(calls[1].id, "supplied");
}
