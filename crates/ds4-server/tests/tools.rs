//! Generated-message parse + SemAccum goldens from `ds4_server.c` unit tests.

use ds4_server::{
    append_tool_calls_json, assign_tool_ids, parse_dsml_tolerant, parse_generated_for_response,
    parse_generated_message, think_start, ChatFormat, ModelSyntax, SemAccum, ToolCall,
    ToolSchemaOrder,
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
        <ifm|tool_call>get_weather\n\
        <ifm|arg_key>city</ifm|arg_key>\n\
        <ifm|arg_value>Seoul</ifm|arg_value>\n\
        <ifm|arg_key>days</ifm|arg_key>\n\
        <ifm|arg_value>2</ifm|arg_value>\n\
        </ifm|tool_call>\n\
        <ifm|tool_call>get_time\n\
        </ifm|tool_call>\n\
        </ifm|tool_calls>";
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into(), "days".into()],
        prop_type: vec!["string".into(), "integer".into()],
        ..Default::default()
    }];
    let p = parse_generated_message(
        ModelSyntax::K2Horizon,
        text,
        true,
        ChatFormat::K2Horizon,
        &orders,
    );
    assert!(p.ok);
    assert_eq!(p.reasoning, b"\nplan\n");
    assert_eq!(p.content, b"hello");
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, r#"{"city": "Seoul", "days": 2}"#);
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
fn parse_mimo_tool_without_think() {
    let generated = b"<tool_call><function=get_weather><parameter=city>Seoul</parameter></function></tool_call>";
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let p = parse_generated_message(
        ModelSyntax::Mimo2,
        generated,
        true,
        ChatFormat::Qwen4Exp,
        &orders,
    );
    assert!(p.ok);
    assert!(p.content.is_empty());
    assert!(p.reasoning.is_empty());
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, r#"{"city": "Seoul"}"#);
}

#[test]
fn parse_mimo_unclosed_think_stays_reasoning() {
    let generated = b"<think>Need weather. <tool_call><function=get_weather><parameter=city>Seoul</parameter></function></tool_call>";
    let p = parse_generated_message(
        ModelSyntax::Mimo2,
        generated,
        true,
        ChatFormat::Qwen4Exp,
        &[],
    );
    assert!(p.ok);
    assert!(p.calls.is_empty());
    assert!(p.content.is_empty());
    assert_eq!(p.reasoning, &generated[b"<think>".len()..]);
}

#[test]
fn qwen_and_step_tool_inside_open_think_stays_reasoning() {
    let generated = b"<tool_call><function=get_weather><parameter=city>Seoul</parameter></function></tool_call>";
    for syntax in [ModelSyntax::Qwen4Exp, ModelSyntax::Step37] {
        let p = parse_generated_message(syntax, generated, true, ChatFormat::Qwen4Exp, &[]);
        assert!(p.ok);
        assert!(p.calls.is_empty(), "{syntax:?}");
        assert!(p.content.is_empty(), "{syntax:?}");
        assert_eq!(p.reasoning, generated, "{syntax:?}");
    }
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

#[test]
fn parse_dsml_tolerant_parses_the_v41_mixed_block() {
    // The V4.1 artifact's real output, captured 2026-10-10 (docs 6.23):
    // decorated wrapper and invoke opens, plain parameter tags and a plain
    // invoke close inside the same block.
    let text = format!(
        "I'll check the current weather in Stockholm for you.\n\n\
         {DSML_START}\n{DSML_INVOKE} name=\"get_weather\">\n\
         <parameter name=\"city\" string=\"true\">Stockholm</parameter>\n\
         </invoke>\n{DSML_END}"
    );
    // The strict one-style parse rejects the mix; that is why the fallback
    // exists.
    let strict = parse_generated_message(
        ModelSyntax::DeepSeek,
        text.as_bytes(),
        false,
        ChatFormat::DeepSeek,
        &[],
    );
    assert!(!strict.ok);
    assert!(strict.calls.is_empty());

    let p = parse_dsml_tolerant(text.as_bytes(), &[], false).expect("tolerant parse");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, r#"{"city": "Stockholm"}"#);
    assert_eq!(
        p.content,
        b"I'll check the current weather in Stockholm for you."
    );
}

#[test]
fn parse_dsml_tolerant_tolerates_the_shims_variants() {
    // shim.py:44-48: a short wrapper, an omitted closer, and a bare
    // <parameter> without the string attribute, all seen in live sessions.
    let short = format!(
        "working\n\n<\u{ff5c}DSML\u{ff5c}calls>\n{DSML_INVOKE} name=\"get_time\">\n\
         <parameter name=\"zone\">UTC</parameter>\n\
         </invoke>\n</\u{ff5c}DSML\u{ff5c}calls>"
    );
    let p = parse_dsml_tolerant(short.as_bytes(), &[], false).expect("short wrapper");
    assert_eq!(p.calls[0].name, "get_time");
    assert_eq!(p.calls[0].arguments, r#"{"zone": "UTC"}"#);
    assert_eq!(p.content, b"working");

    // Wrapper close omitted, invoke complete: still served.
    let unclosed = format!(
        "done\n\n{DSML_START}\n{DSML_INVOKE} name=\"ping\">\n\
         <parameter name=\"host\" string=\"true\">10.0.0.1</parameter>\n\
         </invoke>"
    );
    let p = parse_dsml_tolerant(unclosed.as_bytes(), &[], false).expect("unclosed wrapper");
    assert_eq!(p.calls[0].name, "ping");
    assert_eq!(p.calls[0].arguments, r#"{"host": "10.0.0.1"}"#);

    // A truncated stream (a value running to EOF with no boundary) is
    // refused rather than served half an argument.
    let truncated = format!(
        "{DSML_START}\n{DSML_INVOKE} name=\"bash\">\n\
         <parameter name=\"cmd\" string=\"true\">rm -rf /tmp/fo"
    );
    assert!(parse_dsml_tolerant(truncated.as_bytes(), &[], false).is_none());
}

#[test]
fn parse_dsml_tolerant_coerces_to_the_declared_schema_type() {
    let orders = [ToolSchemaOrder {
        name: "search".into(),
        prop: vec!["limit".into(), "query".into()],
        prop_type: vec!["integer".into(), "string".into()],
        ..Default::default()
    }];
    let text = format!(
        "{DSML_START}\n{DSML_INVOKE} name=\"search\">\n\
         <parameter name=\"limit\" string=\"true\">5</parameter>\n\
         <parameter name=\"query\" string=\"true\">rtx 5090</parameter>\n\
         </invoke>\n{DSML_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("tolerant parse");
    assert_eq!(p.calls[0].arguments, r#"{"limit": 5, "query": "rtx 5090"}"#);
}

#[test]
fn parse_dsml_tolerant_skips_undeclared_names() {
    let orders = [ToolSchemaOrder {
        name: "declared".into(),
        ..Default::default()
    }];
    let text = format!(
        "{DSML_START}\n{DSML_INVOKE} name=\"ghost\">\n\
         <parameter name=\"x\">1</parameter>\n</invoke>\n\
         {DSML_INVOKE} name=\"declared\">\n\
         <parameter name=\"y\">2</parameter>\n</invoke>\n{DSML_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("tolerant parse");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "declared");
}

#[test]
fn parse_dsml_tolerant_nests_parameters_like_the_shim() {
    let text = format!(
        "go\n\n{DSML_START}\n{DSML_INVOKE} name=\"use_tool\">\n\
         <parameter name=\"tool_input\">\n\
         <parameter name=\"url\">https://www.amazon.se</parameter>\n\
         </parameter>\n\
         </invoke>\n{DSML_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &[], false).expect("tolerant parse");
    assert_eq!(p.calls[0].name, "use_tool");
    assert!(
        p.calls[0].arguments.contains("\"tool_input\": {"),
        "{}",
        p.calls[0].arguments
    );
    assert!(p.calls[0]
        .arguments
        .contains("\"url\":\"https://www.amazon.se\""));
}

#[test]
fn parse_dsml_tolerant_binds_a_bare_child_of_a_declared_property() {
    // The artifact's bare-child shape (Spark tool-call gate requests 8/23,
    // 2026-10-11): spaced opens, the property written as a plain child
    // element, no wrapper close.  The engine rejects the block
    // (server_dsml_parse.c:365-369) and the shim yields {} (shim.py:59-61);
    // the schema declares `city`, so the recovery binds it (plan 6.23).
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let text = format!(
        "I'll check Tokyo's weather for you.\n\n{DSML_SPACED_START}\n\
         {DSML_SPACED_INVOKE} name=\"get_weather\">\n\
         <city>Tokyo</city>\n{DSML_SPACED_INVOKE_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("tolerant parse");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, r#"{"city": "Tokyo"}"#);
    assert_eq!(p.content, b"I'll check Tokyo's weather for you.");
}

#[test]
fn parse_dsml_tolerant_keeps_an_undeclared_child_unbound() {
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["town".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let text = format!(
        "{DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"get_weather\">\n\
         <city>Tokyo</city>\n{DSML_SPACED_INVOKE_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("tolerant parse");
    assert_eq!(p.calls[0].arguments, "{}");
}

#[test]
fn parse_dsml_tolerant_coerces_a_child_like_a_parameter() {
    let orders = [ToolSchemaOrder {
        name: "search".into(),
        prop: vec!["limit".into()],
        prop_type: vec!["integer".into()],
        ..Default::default()
    }];
    let text = format!(
        "{DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"search\">\n\
         <limit>5</limit>\n{DSML_SPACED_INVOKE_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("tolerant parse");
    assert_eq!(p.calls[0].arguments, r#"{"limit": 5}"#);
}

#[test]
fn parse_dsml_tolerant_ignores_a_block_inside_closed_thinking() {
    // The engine refuses DSML inside a closed reasoning section: with
    // thinking on, the tool search starts after the LAST </think>
    // (server_dsml_parse.c:291-316; "DSML inside reasoning is just model
    // text ... 思考里的工具块不可执行").  Before the scope fix the fallback
    // scanned the whole text and served the call with a mangled split.
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let think_open = think_start(ChatFormat::DeepSeek);
    let text = format!(
        "{think_open}the user wants the weather; emit the tool now\n\
         {DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"get_weather\">\n\
         {DSML_SPACED_PARAM} name=\"city\" string=\"true\">Paris{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_INVOKE_END}\n{DSML_SPACED_END}\n\n</think>\n\nHello."
    );
    assert!(parse_dsml_tolerant(text.as_bytes(), &orders, true).is_none());
    // Thinking off: the engine searches the whole text and the block counts.
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, false).expect("thinking off");
    assert_eq!(p.calls[0].arguments, r#"{"city": "Paris"}"#);
}

#[test]
fn parse_dsml_tolerant_recovers_a_block_after_closed_thinking() {
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let think_open = think_start(ChatFormat::DeepSeek);
    let text = format!(
        "{think_open}check the weather tool</think>\n\nI'll check.\n\n\
         {DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"get_weather\">\n\
         {DSML_SPACED_PARAM} name=\"city\" string=\"true\">Paris{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_INVOKE_END}"
    );
    let p = parse_dsml_tolerant(text.as_bytes(), &orders, true).expect("tolerant parse");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].arguments, r#"{"city": "Paris"}"#);
    assert_eq!(p.reasoning, b"check the weather tool");
    assert!(
        p.content.ends_with(b"I'll check."),
        "{:?}",
        String::from_utf8_lossy(&p.content)
    );
}

#[test]
fn parse_dsml_tolerant_refuses_a_block_in_unclosed_thinking() {
    // No think close: the engine does no tool search at all -- the text is
    // returned as reasoning with empty content (server_dsml_parse.c:299-313).
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into()],
        prop_type: vec!["string".into()],
        ..Default::default()
    }];
    let think_open = think_start(ChatFormat::DeepSeek);
    let text = format!(
        "{think_open}I'll emit the tool inside the open thinking\n\
         {DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"get_weather\">\n\
         {DSML_SPACED_PARAM} name=\"city\" string=\"true\">Paris{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_INVOKE_END}\n{DSML_SPACED_END}"
    );
    assert!(parse_dsml_tolerant(text.as_bytes(), &orders, true).is_none());
}

const DSML_SPACED_START: &str = "<｜DSML｜ calls>";
const DSML_SPACED_END: &str = "</｜DSML｜ calls>";
const DSML_SPACED_INVOKE: &str = "<｜DSML｜ invoke";
const DSML_SPACED_INVOKE_END: &str = "</｜DSML｜ invoke>";
const DSML_SPACED_PARAM: &str = "<｜DSML｜ parameter";
const DSML_SPACED_PARAM_END: &str = "</｜DSML｜ parameter>";

/// The engine's `test_parse_spaced_dsml_tool_calls` vector (4294136): the
/// official generation's block parses; values are raw (no entity unescape --
/// "&lt;b&gt;" in the value stays exactly that).
#[test]
fn parse_dsml_spaced_official_block() {
    let text = format!(
        "Sure.\n\n{DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"bash\">\n\
         {DSML_SPACED_PARAM} name=\"command\" string=\"true\">echo '&lt;b&gt;' && ls{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_PARAM} name=\"n\" string=\"false\">3{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_INVOKE_END}\n{DSML_SPACED_END}"
    );
    let p = parse_generated_message(
        ModelSyntax::DeepSeek,
        text.as_bytes(),
        false,
        ChatFormat::DeepSeek,
        &[],
    );
    assert!(p.ok);
    assert_eq!(p.content, b"Sure.");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "bash");
    assert_eq!(
        p.calls[0].arguments,
        "{\"command\": \"echo '&lt;b&gt;' && ls\", \"n\": 3}"
    );
    assert!(
        p.raw_dsml.starts_with(&format!("\n\n{DSML_SPACED_START}")),
        "raw_dsml must carry the separator: {:?}",
        p.raw_dsml
    );
}

/// The render-side `&lt;` guard before a closing tag inside a value is
/// restored; a value that genuinely contains "&lt;" keeps it (the engine's
/// dsml_value_unescape, server_dsml_stream.c:37-53).  The renderer writes
/// "&lt;" + the closing tag minus its leading '<' (append_dsml_parameter_text).
#[test]
fn parse_dsml_spaced_value_guard_round_trip() {
    let guard = format!("&lt;{}", &DSML_SPACED_PARAM_END[1..]);
    let text = format!(
        "ok\n\n{DSML_SPACED_START}\n{DSML_SPACED_INVOKE} name=\"write\">\n\
         {DSML_SPACED_PARAM} name=\"body\" string=\"true\">x{guard}y &amp; z{DSML_SPACED_PARAM_END}\n\
         {DSML_SPACED_INVOKE_END}\n{DSML_SPACED_END}"
    );
    let p = parse_generated_message(
        ModelSyntax::DeepSeek,
        text.as_bytes(),
        false,
        ChatFormat::DeepSeek,
        &[],
    );
    assert_eq!(p.calls.len(), 1);
    assert_eq!(
        p.calls[0].arguments,
        format!("{{\"body\": \"x{DSML_SPACED_PARAM_END}y &amp; z\"}}")
    );
}
