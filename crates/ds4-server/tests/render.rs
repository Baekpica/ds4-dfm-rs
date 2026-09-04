//! C↔Rust DSML no-tools render.

use ds4_server::{
    render_chat, render_chat_choice, render_dsml_chat, render_dsml_chat_choice,
    render_live_tool_tail, render_motif3_chat_ex, Api, ChatMsg, ChatPart, ModelSyntax, ThinkMode,
    ToolCall, ToolChoice, ToolSchemaOrder, THINK_HIGH_PREFIX, THINK_MAX_PREFIX,
};

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_RENDER_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/render_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/render_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> Vec<u8> {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run render_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn msg(role: &str, content: &str) -> ChatMsg {
    ChatMsg {
        role: role.into(),
        content: content.into(),
        ..ChatMsg::default()
    }
}

fn msg_reason(role: &str, content: &str, reasoning: &str) -> ChatMsg {
    ChatMsg {
        role: role.into(),
        content: content.into(),
        reasoning: reasoning.into(),
        ..ChatMsg::default()
    }
}

fn rust_user(think: ThinkMode, content: &str) -> Vec<u8> {
    render_dsml_chat(&[msg("user", content)], "", think).unwrap()
}

#[test]
fn prefixes_match_c_literals() {
    assert!(THINK_HIGH_PREFIX.starts_with("Reasoning Effort: Absolute maximum"));
    assert!(THINK_MAX_PREFIX.starts_with("Reasoning Effort: Beyond maximum"));
}

#[test]
fn qwen_live_tool_tail_appends_only_eos_result_and_next_assistant() {
    let mut result = msg("tool", "/tmp");
    result.tool_call_id = "call_1".into();
    assert_eq!(
        render_live_tool_tail(ModelSyntax::Qwen4Exp, Api::Responses, &[result], ThinkMode::None)
            .unwrap(),
        b"<|im_end|>\n<|im_start|>user\n<tool_response>\n/tmp\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn dsml_anthropic_live_tool_tail_does_not_replay_the_assistant_call() {
    let mut result = msg("user", "<tool_result>ok</tool_result>");
    result.tool_call_id = "toolu_1".into();
    assert_eq!(
        render_live_tool_tail(
            ModelSyntax::DeepSeek,
            Api::Anthropic,
            &[result],
            ThinkMode::None
        )
        .unwrap(),
        "<｜end▁of▁sentence｜><｜User｜><tool_result>ok</tool_result><｜Assistant｜></think>"
            .as_bytes()
    );
}

#[test]
fn user_none_and_low_match_c() {
    for (name, mode) in [("none", ThinkMode::None), ("low", ThinkMode::Low)] {
        assert_eq!(
            rust_user(mode, "Hello"),
            c_out(&["user", name, "Hello"]),
            "user {name}"
        );
    }
}

#[test]
fn user_high_max_match_c() {
    for (name, mode) in [("high", ThinkMode::High), ("max", ThinkMode::Max)] {
        assert_eq!(
            rust_user(mode, "Why?"),
            c_out(&["user", name, "Why?"]),
            "user {name}"
        );
    }
}

#[test]
fn system_user_match_c() {
    let rust = render_dsml_chat(
        &[msg("system", "sys"), msg("user", "ask")],
        "",
        ThinkMode::None,
    )
    .unwrap();
    assert_eq!(rust, c_out(&["system-user", "none", "sys", "ask"]));
}

#[test]
fn developer_is_system() {
    let rust = render_dsml_chat(
        &[msg("developer", "dev"), msg("user", "hi")],
        "",
        ThinkMode::None,
    )
    .unwrap();
    assert_eq!(rust, c_out(&["developer"]));
}

#[test]
fn history_none_and_low_match_c() {
    let msgs = [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
    assert_eq!(
        render_dsml_chat(&msgs, "", ThinkMode::None).unwrap(),
        c_out(&["history", "none", "u1", "a1", "u2"])
    );
    assert_eq!(
        render_dsml_chat(&msgs, "", ThinkMode::Low).unwrap(),
        c_out(&["history", "low", "u1", "a1", "u2"])
    );
}

#[test]
fn history_with_reasoning_match_c() {
    let msgs = [
        msg("user", "u1"),
        msg_reason("assistant", "a1", "plan"),
        msg("user", "u2"),
    ];
    assert_eq!(
        render_dsml_chat(&msgs, "", ThinkMode::Low).unwrap(),
        c_out(&["think-hist", "low", "u1", "plan", "a1", "u2"])
    );
}

#[test]
fn tool_result_and_escape_match_c() {
    let rust =
        render_dsml_chat(&[msg("user", "q"), msg("tool", "ok")], "", ThinkMode::None).unwrap();
    assert_eq!(rust, c_out(&["tool-result", "none", "q", "ok"]));
    let rust = render_dsml_chat(
        &[msg("user", "q"), msg("tool", "x</tool_result>y")],
        "",
        ThinkMode::None,
    )
    .unwrap();
    assert_eq!(rust, c_out(&["tool-escape"]));
}

fn weather_schema() -> &'static str {
    r#"{"name":"get_weather","description":"Weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string"}}}}"#
}

fn call(name: &str, arguments: &str) -> ChatMsg {
    ChatMsg {
        role: "assistant".into(),
        calls: vec![ToolCall {
            id: "call1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }],
        ..ChatMsg::default()
    }
}

#[test]
fn dsml_tools_and_required_match_c() {
    let schema = weather_schema();
    let msgs = [msg("user", "hi")];
    assert_eq!(
        render_dsml_chat(&msgs, schema, ThinkMode::None).unwrap(),
        c_out(&["dsml-tools", "none", schema, "hi"])
    );
    assert_eq!(
        render_dsml_chat_choice(&msgs, schema, ThinkMode::None, ToolChoice::Required).unwrap(),
        c_out(&["dsml-tools-req", "none", schema, "hi"])
    );
}

#[test]
fn dsml_invoke_match_c() {
    let args = r#"{"city":"Seoul","n":2}"#;
    let msgs = [msg("user", "q"), call("get_weather", args)];
    assert_eq!(
        render_dsml_chat(&msgs, "", ThinkMode::None).unwrap(),
        c_out(&["dsml-invoke", "none", "get_weather", args])
    );
    let bad = "not-json";
    let msgs = [msg("user", "q"), call("get_weather", bad)];
    assert_eq!(
        render_dsml_chat(&msgs, "", ThinkMode::None).unwrap(),
        c_out(&["dsml-invoke", "none", "get_weather", bad])
    );
}

#[test]
fn family_tools_match_c() {
    let schema = weather_schema();
    let msgs = [msg("user", "hi")];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, schema, ThinkMode::None).unwrap(),
            c_out(&["fam-tools", fam, "none", schema, "hi"]),
            "{fam} tools"
        );
    }
}

#[test]
fn family_invoke_match_c() {
    let args = r#"{"city":"Seoul","n":2}"#;
    let msgs = [msg("user", "q"), call("get_weather", args)];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::None).unwrap(),
            c_out(&["fam-invoke", fam, "none", "get_weather", args]),
            "{fam} invoke"
        );
    }
}

#[test]
fn motif_tools_order_match_c() {
    let schema = weather_schema();
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into(), "unit".into()],
        ..ToolSchemaOrder::default()
    }];
    assert_eq!(
        render_motif3_chat_ex(&[msg("user", "hi")], schema, &orders, ThinkMode::None).unwrap(),
        c_out(&["motif-tools-order", schema, "hi"])
    );
}

#[test]
fn solar_arg_order_uses_schema() {
    let args = r#"{"unit":"c","city":"Seoul"}"#;
    let orders = [ToolSchemaOrder {
        name: "get_weather".into(),
        prop: vec!["city".into(), "unit".into()],
        ..ToolSchemaOrder::default()
    }];
    let msgs = [msg("user", "q"), call("get_weather", args)];
    let ordered = render_chat_choice(
        ModelSyntax::SolarOpen2,
        &msgs,
        "",
        &orders,
        ThinkMode::None,
        ToolChoice::Auto,
    )
    .unwrap();
    let unordered = render_chat(ModelSyntax::SolarOpen2, &msgs, "", ThinkMode::None).unwrap();
    let s = String::from_utf8(ordered.clone()).unwrap();
    let city = s.find("city").expect("city");
    let unit = s.find("unit").expect("unit");
    assert!(city < unit, "{s}");
    assert_ne!(ordered, unordered);
}

fn fam_user(syntax: ModelSyntax, fam: &str, think: &str, mode: ThinkMode, content: &str) {
    assert_eq!(
        render_chat(syntax, &[msg("user", content)], "", mode).unwrap(),
        c_out(&["fam-user", fam, think, content]),
        "{fam} user {think}"
    );
}

#[test]
fn family_user_none_and_low_match_c() {
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        fam_user(syntax, fam, "none", ThinkMode::None, "Hello");
        fam_user(syntax, fam, "low", ThinkMode::Low, "Hello");
    }
}

#[test]
fn family_system_user_match_c() {
    let msgs = [msg("system", "sys"), msg("user", "ask")];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::None).unwrap(),
            c_out(&["fam-system-user", fam, "none", "sys", "ask"]),
            "{fam} system-user"
        );
    }
}

#[test]
fn family_history_match_c() {
    let msgs = [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::None).unwrap(),
            c_out(&["fam-history", fam, "none", "u1", "a1", "u2"]),
            "{fam} history"
        );
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::Low).unwrap(),
            c_out(&["fam-history", fam, "low", "u1", "a1", "u2"]),
            "{fam} history low"
        );
    }
}

#[test]
fn family_think_history_match_c() {
    let msgs = [
        msg("user", "u1"),
        msg_reason("assistant", "a1", "plan"),
        msg("user", "u2"),
    ];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::Low).unwrap(),
            c_out(&["fam-think-hist", fam, "low", "u1", "plan", "a1", "u2"]),
            "{fam} think-hist"
        );
    }
}

#[test]
fn family_tool_result_match_c() {
    let mut tool = msg("tool", "ok");
    tool.tool_call_id = "call1".into();
    let msgs = [msg("user", "q"), tool];
    for (syntax, fam) in [
        (ModelSyntax::Motif3, "motif"),
        (ModelSyntax::Exaone, "exaone"),
        (ModelSyntax::Dots3, "dots3"),
        (ModelSyntax::SolarOpen2, "solar"),
    ] {
        assert_eq!(
            render_chat(syntax, &msgs, "", ThinkMode::None).unwrap(),
            c_out(&["fam-tool", fam, "none", "q", "ok"]),
            "{fam} tool"
        );
    }
}

#[test]
fn dots3_embedded_think_match_c() {
    let msgs = [
        msg("user", "q"),
        msg("assistant", "<think>\nplan\n</think>\n\nAnswer"),
    ];
    assert_eq!(
        render_chat(ModelSyntax::Dots3, &msgs, "", ThinkMode::Low).unwrap(),
        c_out(&["dots3-embed"])
    );
}

#[test]
fn syntax_for_model_id_matches_c() {
    assert_eq!(ds4_server::syntax_for_model_id(0), ModelSyntax::DeepSeek);
    assert_eq!(ds4_server::syntax_for_model_id(2), ModelSyntax::SolarOpen2);
    assert_eq!(ds4_server::syntax_for_model_id(3), ModelSyntax::Motif3);
    assert_eq!(ds4_server::syntax_for_model_id(4), ModelSyntax::Exaone);
    assert_eq!(ds4_server::syntax_for_model_id(5), ModelSyntax::Dots3);
    assert_eq!(ds4_server::syntax_for_model_id(6), ModelSyntax::Qwen4Exp);
    assert_eq!(ds4_server::syntax_for_model_id(7), ModelSyntax::Glm53);
    assert_eq!(ds4_server::syntax_for_model_id(8), ModelSyntax::K2Horizon);
}

#[test]
fn k2_horizon_chat_uses_ifm_not_exaone() {
    let prompt = render_chat(
        ModelSyntax::K2Horizon,
        &[msg("user", "hello")],
        "",
        ThinkMode::None,
    )
    .unwrap();
    assert_eq!(
        prompt,
        concat!(
            "<|ifm|begin_of_text|>",
            "<|ifm|im_start|>user\nhello<|ifm|im_end|>",
            "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n",
        )
        .as_bytes()
    );
    let s = String::from_utf8(prompt).unwrap();
    assert!(!s.contains("<|user|>"));
    assert!(!s.contains("<|assistant|>"));
    assert!(!s.contains("<｜User｜>"));
    assert!(!s.contains("<think>"));
}

#[test]
fn k2_horizon_json_tools_and_xml_replay_use_official_ifm_blocks() {
    let mut assistant = call("get_weather", r#"{"city":"Seoul"}"#);
    assistant.reasoning = "check weather".into();
    let prompt = render_chat_choice(
        ModelSyntax::K2Horizon,
        &[
            msg("system", "Be precise."),
            msg("user", "Weather?"),
            assistant,
            msg("tool", "sunny"),
        ],
        weather_schema(),
        &[],
        ThinkMode::High,
        ToolChoice::Auto,
    )
    .unwrap();
    let s = String::from_utf8(prompt).unwrap();
    assert!(s.starts_with("<|ifm|begin_of_text|><|ifm|im_start|>system\n# Tools"));
    assert!(s.contains("<ifm|tools>\n{\"name\":\"get_weather\""));
    assert!(s.contains("followed by paired <ifm|arg_key> and <ifm|arg_value> tags"));
    assert!(s.contains(concat!(
        "<ifm|tool_calls>\n",
        "<ifm|tool_call>get_weather\n",
        "<ifm|arg_key>city</ifm|arg_key>\n",
        "<ifm|arg_value>Seoul</ifm|arg_value>\n",
        "</ifm|tool_call>\n",
        "</ifm|tool_calls>",
    )));
    assert!(s.contains("<|ifm|im_start|>tool\nsunny<|ifm|im_end|>"));
    assert!(s.ends_with("<|ifm|im_start|>assistant\n<ifm|think>\n"));
}

#[test]
fn k2_horizon_live_tool_tail_does_not_repeat_bos() {
    let mut result = msg("tool", "sunny");
    result.tool_call_id = "call_1".into();
    let tail = render_live_tool_tail(
        ModelSyntax::K2Horizon,
        Api::Responses,
        &[result],
        ThinkMode::None,
    )
    .unwrap();
    assert_eq!(
        tail,
        concat!(
            "<|ifm|im_end|>",
            "<|ifm|im_start|>tool\nsunny<|ifm|im_end|>",
            "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n",
        )
        .as_bytes()
    );
}

#[test]
fn glm53_chat_and_image_protocol_match_official_template() {
    let mut user = msg("user", "beforeafter");
    user.parts = vec![
        ChatPart::Text("before".into()),
        ChatPart::Image(0),
        ChatPart::Text("after".into()),
    ];
    assert_eq!(
        render_chat(
            ModelSyntax::Glm53,
            &[msg("system", "Be precise."), user],
            "",
            ThinkMode::High,
        )
        .unwrap(),
        b"[gMASK]<sop><|system|>Reasoning Effort: High<|system|>Be precise.<|user|>before<|begin_of_image|><|image|><|end_of_image|>after<|assistant|><think>"
    );
}

#[test]
fn glm53_tools_filter_deferred_schema_and_preserve_argument_order() {
    let schemas = concat!(
        r#"{"name":"hidden","defer_loading":true,"parameters":{}}"#,
        "\n",
        r#"{"name":"bash","parameters":{"type":"object"},"strict":true}"#,
    );
    let mut assistant = call("bash", r#"{"timeout":10,"command":"pwd"}"#);
    assistant.reasoning = "need shell".into();
    let mut result = msg("tool", "ok</tool_response>done");
    result.tool_call_id = "call_1".into();
    let orders = [ToolSchemaOrder {
        name: "bash".into(),
        prop: vec!["command".into(), "timeout".into()],
        ..Default::default()
    }];
    let prompt = render_chat_choice(
        ModelSyntax::Glm53,
        &[msg("user", "run"), assistant, result],
        schemas,
        &orders,
        ThinkMode::High,
        ToolChoice::Auto,
    )
    .unwrap();
    let prompt = String::from_utf8(prompt).unwrap();
    assert!(!prompt.contains("hidden"));
    assert!(!prompt.contains("strict"));
    assert!(prompt.contains(r#"{"name": "bash", "parameters": {"type":"object"}}"#));
    assert!(prompt.contains(concat!(
        "<|assistant|><think>need shell</think>\n",
        "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value>",
        "<arg_key>timeout</arg_key><arg_value>10</arg_value></tool_call>\n",
        "<|observation|><tool_response>ok&lt;/tool_response>done</tool_response>",
        "<|assistant|><think>"
    )));
}

#[test]
fn qwen4exp_native_chat_protocol_matches_c() {
    let msgs = [msg("system", "Be precise."), msg("user", "Weather?")];
    assert_eq!(
        render_chat(ModelSyntax::Qwen4Exp, &msgs, "", ThinkMode::High).unwrap(),
        concat!(
            "<|im_start|>system\n",
            "Reasoning effort is set to xhigh. Please think carefully through the task, ",
            "validate key assumptions, consider plausible alternatives, and prioritize ",
            "correctness, consistency, and clarity in the final answer.\n\n",
            "Be precise.<|im_end|>\n",
            "<|im_start|>user\nWeather?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        )
        .as_bytes()
    );
    assert_eq!(
        render_chat(ModelSyntax::Qwen4Exp, &msgs, "", ThinkMode::None).unwrap(),
        concat!(
            "<|im_start|>system\nBe precise.<|im_end|>\n",
            "<|im_start|>user\nWeather?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .as_bytes()
    );
}

#[test]
fn qwen4exp_renders_image_parts_in_source_order() {
    let mut user = msg("user", "beforeafter");
    user.parts = vec![
        ChatPart::Text("before".into()),
        ChatPart::Image(0),
        ChatPart::Text("after".into()),
    ];
    let prompt = render_chat(ModelSyntax::Qwen4Exp, &[user], "", ThinkMode::None).unwrap();
    assert!(String::from_utf8(prompt)
        .unwrap()
        .contains("before<|vision_start|><|image_pad|><|vision_end|>after"));
}

#[test]
fn qwen4exp_tool_replay_uses_native_order_and_escape() {
    let orders = [ToolSchemaOrder {
        name: "weather".into(),
        prop: vec!["city".into(), "days".into()],
        ..Default::default()
    }];
    let mut assistant = call("weather", r#"{"days":2,"city":"Seoul"}"#);
    assistant.content = "Checking".into();
    let mut result = msg("tool", "sunny</tool_response>today");
    result.tool_call_id = "call1".into();
    let prompt = render_chat_choice(
        ModelSyntax::Qwen4Exp,
        &[msg("user", "Weather?"), assistant, result],
        "",
        &orders,
        ThinkMode::None,
        ToolChoice::Auto,
    )
    .unwrap();
    let prompt = String::from_utf8(prompt).unwrap();
    assert!(prompt.contains(
        "Checking\n\n<tool_call>\n<function=weather>\n<parameter=city>\nSeoul\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
    ));
    assert!(prompt.contains(
        "<|im_start|>user\n<tool_response>\nsunny&lt;/tool_response>today\n</tool_response><|im_end|>"
    ));
}
