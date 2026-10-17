//! C↔Rust DSML no-tools render.

use ds4_server::render::DSML_IMAGE;
use ds4_server::tools::{
    DSML_INVOKE_END_SPACED, DSML_INVOKE_START_SPACED, DSML_PARAM_END_SPACED,
    DSML_PARAM_START_SPACED, DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START_SPACED,
};
use ds4_server::{
    render_chat, render_chat_choice, render_dsml_chat, render_dsml_chat_choice,
    render_dsml_chat_choice_ex, render_live_tool_tail, render_motif3_chat_ex, Api, ChatMsg,
    ChatPart, DsmlHead, ModelSyntax, ThinkMode, ToolCall, ToolChoice, ToolSchemaOrder,
    DSML_ASSISTANT, DSML_BOS, DSML_EOS, DSML_SYSTEM_TOKEN, DSML_THINK_END, DSML_THINK_START,
    DSML_USER, THINK_HIGH_PREFIX, THINK_MAX_PREFIX, V41_THINK_HIGH_PREFIX,
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
fn dsml_v41_head_matches_official_encoding() {
    // The engine's golden byte test `test_render_matches_official_v41_encoding`
    // (tests/server_tests_render_v41.c): the official encoding.py render of
    // four conversations, plus the V4-tokenizer fallback.  A V4.1 tokenizer
    // owns the system token; the head writes it once before the effort prefix.
    let v41 = DsmlHead::V41 { system_token: true };
    let with_sys = [msg("system", "S"), msg("user", "U")];
    let only_user = [msg("user", "U")];
    let render = |msgs: &[ChatMsg], mode| {
        render_dsml_chat_choice_ex(msgs, "", mode, ToolChoice::Auto, v41).unwrap()
    };
    // The message list ends in a user turn, so the render closes with the
    // pending assistant head (the think-end tag when plain, the think-start tag
    // when thinking).
    let sys_tail = format!("S{DSML_USER}U{DSML_ASSISTANT}");
    let user_tail = format!("{DSML_USER}U{DSML_ASSISTANT}");
    assert_eq!(
        render(&with_sys, ThinkMode::None),
        format!("{DSML_BOS}{DSML_SYSTEM_TOKEN}{sys_tail}{DSML_THINK_END}").as_bytes()
    );
    // A user-only chat with no effort prefix writes no head token at all.
    assert_eq!(
        render(&only_user, ThinkMode::None),
        format!("{DSML_BOS}{user_tail}{DSML_THINK_END}").as_bytes()
    );
    assert_eq!(
        render(&with_sys, ThinkMode::High),
        format!("{DSML_BOS}{DSML_SYSTEM_TOKEN}{V41_THINK_HIGH_PREFIX}{sys_tail}{DSML_THINK_START}")
            .as_bytes()
    );
    assert_eq!(
        render(&only_user, ThinkMode::High),
        format!(
            "{DSML_BOS}{DSML_SYSTEM_TOKEN}{V41_THINK_HIGH_PREFIX}{user_tail}{DSML_THINK_START}"
        )
        .as_bytes()
    );
    // The V4 tokenizer has no system token: BOS then the system body (V4 head).
    assert_eq!(
        render_dsml_chat_choice_ex(
            &with_sys,
            "",
            ThinkMode::None,
            ToolChoice::Auto,
            DsmlHead::V4
        )
        .unwrap(),
        format!("{DSML_BOS}{sys_tail}{DSML_THINK_END}").as_bytes()
    );
}

#[test]
fn dsml_live_tool_tail_has_no_head() {
    // The engine's tail writes EOS + messages only (server_live_prep.c:21-69;
    // the port's own C: ds4_server.c:4700-4748): no effort prefix and no
    // system token; the head is already in the live KV.
    let mut result = msg("tool", "ok");
    result.tool_call_id = "call_1".into();
    let tail = render_live_tool_tail(
        ModelSyntax::DeepSeek,
        Api::Responses,
        &[result],
        ThinkMode::High,
    )
    .unwrap();
    assert!(!tail.starts_with(V41_THINK_HIGH_PREFIX.as_bytes()));
    let expected = format!(
        "{DSML_EOS}{DSML_USER}<tool_result>ok</tool_result>{DSML_ASSISTANT}{DSML_THINK_START}"
    );
    assert_eq!(tail, expected.as_bytes());
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
fn glm_vq_uses_glm_syntax() {
    // GLM VQ keeps GLM-5.3's chat template, tool markers and output parser.
    assert_eq!(
        ds4_server::syntax_for_model_id(ds4_core::Variant::Glm53FlashVq as i32),
        ModelSyntax::Glm53
    );
}

#[test]
fn ling31_uses_bailing_syntax() {
    // Ling-3.1 keeps Bailing V3's chat template, tool markers and parser.
    let id = ds4_core::Variant::Ling31FlashVq as i32;
    assert_eq!(ds4_server::syntax_for_model_id(id), ModelSyntax::Ling3Vl);
    let raw = b"read it</think><tool_call>read\n<arg_key>path</arg_key>\n<arg_value>a.rs</arg_value>\n</tool_call>";
    let parsed = ds4_server::parse_generated_for_model_id(id, raw, true, &[]);
    assert_eq!(parsed.reasoning, b"read it");
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "read");
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

/// The engine's SPACED_TOOLS_HEAD (tests/server_tests_render_v41.c, 4294136):
/// the official encoding.py TOOLS_TEMPLATE verbatim.  Built from the crate's
/// tag constants, whose bytes are the engine's (od-verified); the think tags
/// use the crate's DSML_THINK_START/DSML_THINK_END spellings.
fn spaced_tools_head() -> String {
    let mut s = String::new();
    s.push_str("## Tools\n\n");
    s.push_str("You have access to a set of tools to help answer the user's question. ");
    s.push_str("You can invoke tools by writing a \"");
    s.push_str(DSML_TOOL_CALLS_START_SPACED);
    s.push_str("\" block like the following:\n\n");
    s.push_str(DSML_TOOL_CALLS_START_SPACED);
    s.push_str("\n");
    s.push_str(DSML_INVOKE_START_SPACED);
    s.push_str(" name=\"$TOOL_NAME\">\n");
    s.push_str(DSML_PARAM_START_SPACED);
    s.push_str(" name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE");
    s.push_str(DSML_PARAM_END_SPACED);
    s.push_str("\n...\n");
    s.push_str(DSML_INVOKE_END_SPACED);
    s.push_str("\n");
    s.push_str(DSML_INVOKE_START_SPACED);
    s.push_str(" name=\"$TOOL_NAME2\">\n...\n");
    s.push_str(DSML_INVOKE_END_SPACED);
    s.push_str("\n");
    s.push_str(DSML_TOOL_CALLS_END_SPACED);
    s.push_str("\n\n");
    s.push_str("String parameters should be specified as is and set `string=\"true\"`. ");
    s.push_str("For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n");
    s.push_str("If thinking_mode is enabled (triggered by ");
    s.push_str(DSML_THINK_START);
    s.push_str("), you MUST output your complete reasoning inside ");
    s.push_str(DSML_THINK_START);
    s.push_str("...");
    s.push_str(DSML_THINK_END);
    s.push_str(" BEFORE any tool calls or final response.\n\n");
    s.push_str("Otherwise, output directly after ");
    s.push_str(DSML_THINK_END);
    s.push_str(" with tool calls or final response.\n\n");
    s.push_str("### Available Tool Schemas\n\n");
    s
}

const SPACED_TOOLS_TAIL: &str =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

fn assert_spaced_eq(got: Vec<u8>, want: &str) {
    if got == want.as_bytes() {
        return;
    }
    let at = got
        .iter()
        .zip(want.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    panic!(
        "spaced render mismatch at byte {at}\n  want: {:?}\n  got:  {:?}",
        &want.as_bytes()
            [at.min(want.len())..at.min(want.len()) + 120.min(want.len() - at.min(want.len()))],
        &got[at.min(got.len())..at.min(got.len()) + 120.min(got.len() - at.min(got.len()))]
    );
}

#[test]
fn dsml_v41_spaced_tools_match_official_encoding() {
    // The engine's golden byte test (tests/server_tests_render_v41.c, 4294136):
    // case1 is the official repo's encoding/tests/test_input_1.json, byte-identical
    // to its own test_output_1.txt; case2 adds parallel calls, a zero-parameter
    // call, json.dumps-shaped values, "<" and "&" inside a value, two adjacent
    // tool results merged into one <｜User｜>, and a schema carrying a \u escape.  A
    // byte mismatch here is a bug in the spaced template.
    let v41 = DsmlHead::V41 { system_token: true };

    // case1: the official get_weather flow, THINK_HIGH.
    let c1 = [
        msg("system", "You are a helpful assistant."),
        msg("user", "What's the weather like in Beijing?"),
        ChatMsg {
            role: "assistant".into(),
            reasoning: "The user wants the weather in Beijing. I should call get_weather.".into(),
            calls: vec![ToolCall {
                id: "call1".into(),
                name: "get_weather".into(),
                arguments: r#"{"location": "Beijing", "unit": "celsius"}"#.into(),
            }],
            ..ChatMsg::default()
        },
        msg(
            "tool",
            r#"{"temperature": 22, "condition": "sunny", "humidity": 45}"#,
        ),
        msg_reason(
            "assistant",
            "The weather in Beijing is currently sunny with a temperature of 22\u{b0}C and 45% humidity.",
            "Got the weather data. Let me format a nice response.",
        ),
    ];
    let schemas1 = concat!(
        r#"{"name":"get_weather","description":"Get the weather for a specific location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city name"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}"#,
        "\n",
        r#"{"name":"search","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results to return"}},"required":["query"]}}"#,
    );
    let mut want1 = String::new();
    want1.push_str(DSML_BOS);
    want1.push_str(DSML_SYSTEM_TOKEN);
    want1.push_str(V41_THINK_HIGH_PREFIX);
    want1.push_str("You are a helpful assistant.\n\n");
    want1.push_str(&spaced_tools_head());
    want1.push_str("{\"name\": \"get_weather\", \"description\": \"Get the weather for a specific location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The city name\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"], \"description\": \"Temperature unit\"}}, \"required\": [\"location\"]}}\n");
    want1.push_str("{\"name\": \"search\", \"description\": \"Search the web for information\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"Search query\"}, \"num_results\": {\"type\": \"integer\", \"description\": \"Number of results to return\"}}, \"required\": [\"query\"]}}");
    want1.push_str(SPACED_TOOLS_TAIL);
    want1.push_str(DSML_USER);
    want1.push_str("What's the weather like in Beijing?");
    want1.push_str(DSML_ASSISTANT);
    want1.push_str(DSML_THINK_START);
    want1.push_str("The user wants the weather in Beijing. I should call get_weather.");
    want1.push_str(DSML_THINK_END);
    want1.push_str("\n\n");
    want1.push_str(DSML_TOOL_CALLS_START_SPACED);
    want1.push_str("\n");
    want1.push_str(DSML_INVOKE_START_SPACED);
    want1.push_str(" name=\"get_weather\">\n");
    want1.push_str(DSML_PARAM_START_SPACED);
    want1.push_str(" name=\"location\" string=\"true\">Beijing");
    want1.push_str(DSML_PARAM_END_SPACED);
    want1.push_str("\n");
    want1.push_str(DSML_PARAM_START_SPACED);
    want1.push_str(" name=\"unit\" string=\"true\">celsius");
    want1.push_str(DSML_PARAM_END_SPACED);
    want1.push_str("\n");
    want1.push_str(DSML_INVOKE_END_SPACED);
    want1.push_str("\n");
    want1.push_str(DSML_TOOL_CALLS_END_SPACED);
    want1.push_str(DSML_EOS);
    want1.push_str(DSML_USER);
    want1.push_str(
        "<tool_result>{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}</tool_result>",
    );
    want1.push_str(DSML_ASSISTANT);
    want1.push_str(DSML_THINK_START);
    want1.push_str("Got the weather data. Let me format a nice response.");
    want1.push_str(DSML_THINK_END);
    want1.push_str(
        "The weather in Beijing is currently sunny with a temperature of 22\u{b0}C and 45% humidity.",
    );
    want1.push_str(DSML_EOS);
    assert_spaced_eq(
        render_dsml_chat_choice_ex(&c1, schemas1, ThinkMode::High, ToolChoice::Auto, v41).unwrap(),
        &want1,
    );

    // case2: parallel calls, a zero-parameter call, merged tool results, THINK_NONE.
    let c2 = [
        msg("system", "S"),
        msg("user", "U"),
        ChatMsg {
            role: "assistant".into(),
            calls: vec![
                ToolCall {
                    id: "call1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"location":"a<b & c","days":3,"opts":{"a":[1,2],"b":true}}"#
                        .into(),
                },
                ToolCall {
                    id: "call2".into(),
                    name: "now".into(),
                    arguments: "{}".into(),
                },
            ],
            ..ChatMsg::default()
        },
        msg("tool", "R1"),
        msg("tool", "R2"),
        msg("user", "U2"),
    ];
    let schemas2 = "{\n  \"name\": \"get_weather\",\n  \"description\": \"Weather \\u00b0C\",\n  \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"location\"]}\n}\n{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}";
    let mut want2 = String::new();
    want2.push_str(DSML_BOS);
    want2.push_str(DSML_SYSTEM_TOKEN);
    want2.push_str("S\n\n");
    want2.push_str(&spaced_tools_head());
    want2.push_str("{\"name\": \"get_weather\", \"description\": \"Weather \u{b0}C\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"location\"]}}\n");
    want2.push_str(
        "{\"name\": \"now\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}",
    );
    want2.push_str(SPACED_TOOLS_TAIL);
    want2.push_str(DSML_USER);
    want2.push_str("U");
    want2.push_str(DSML_ASSISTANT);
    want2.push_str(DSML_THINK_END);
    want2.push_str("\n\n");
    want2.push_str(DSML_TOOL_CALLS_START_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_INVOKE_START_SPACED);
    want2.push_str(" name=\"get_weather\">\n");
    want2.push_str(DSML_PARAM_START_SPACED);
    want2.push_str(" name=\"location\" string=\"true\">a<b & c");
    want2.push_str(DSML_PARAM_END_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_PARAM_START_SPACED);
    want2.push_str(" name=\"days\" string=\"false\">3");
    want2.push_str(DSML_PARAM_END_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_PARAM_START_SPACED);
    want2.push_str(" name=\"opts\" string=\"false\">{\"a\": [1, 2], \"b\": true}");
    want2.push_str(DSML_PARAM_END_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_INVOKE_END_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_INVOKE_START_SPACED);
    want2.push_str(" name=\"now\">\n");
    want2.push_str("\n");
    want2.push_str(DSML_INVOKE_END_SPACED);
    want2.push_str("\n");
    want2.push_str(DSML_TOOL_CALLS_END_SPACED);
    want2.push_str(DSML_EOS);
    want2.push_str(DSML_USER);
    want2.push_str("<tool_result>R1</tool_result>");
    want2.push_str("\n\n");
    want2.push_str("<tool_result>R2</tool_result>");
    want2.push_str("\n\n");
    want2.push_str("U2");
    want2.push_str(DSML_ASSISTANT);
    want2.push_str(DSML_THINK_END);
    assert_spaced_eq(
        render_dsml_chat_choice_ex(&c2, schemas2, ThinkMode::None, ToolChoice::Auto, v41).unwrap(),
        &want2,
    );
}

#[test]
fn dsml_v41_places_tool_result_images_inside_the_envelope() {
    // A tool_result whose content carried an image reaches the renderer with
    // the envelope and the placeholder already written into the content in the
    // client's block order (parse.rs json_tool_result_content; the engine's
    // text-level render, server_parse_tools.c:330).  The user body must write
    // that content -- running the parts loop instead would drop the envelope.
    let v41 = DsmlHead::V41 { system_token: true };
    let tool_result = ChatMsg {
        role: "user".into(),
        content: format!("<tool_result>shot:{DSML_IMAGE}done</tool_result>"),
        parts: vec![
            ChatPart::Image(0),
            ChatPart::ToolResult {
                id: "toolu_02".into(),
                content: "shot:done".into(),
            },
        ],
        tool_call_ids: vec!["toolu_02".into()],
        ..ChatMsg::default()
    };
    let out =
        render_dsml_chat_choice_ex(&[tool_result], "", ThinkMode::None, ToolChoice::Auto, v41)
            .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(&format!("<tool_result>shot:{DSML_IMAGE}done</tool_result>")));
    assert_eq!(text.matches(DSML_IMAGE).count(), 1);

    // A plain user image still renders from the parts, placeholder where the
    // part sits (the content carries no placeholder for this shape).
    let plain = ChatMsg {
        role: "user".into(),
        content: "beforeafter".into(),
        parts: vec![
            ChatPart::Text("before".into()),
            ChatPart::Image(0),
            ChatPart::Text("after".into()),
        ],
        ..ChatMsg::default()
    };
    let out =
        render_dsml_chat_choice_ex(&[plain], "", ThinkMode::None, ToolChoice::Auto, v41).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(&format!("before{DSML_IMAGE}after")));
}

#[test]
fn dsml_v41_writes_a_tool_message_image_inside_its_envelope() {
    // The Chat Completions MCP loop: the image arrives in a tool-role message
    // whose content is a block array, so parse.rs writes the placeholder into
    // that content in client block order and this render wraps the content in
    // its <tool_result> envelope.  The envelope text is the only place the
    // V4.1 head can put the placeholder for a tool result (the engine's
    // render, server_dsml_render.c:409-414), so the image must reach the
    // prompt here or it is silently dropped.
    let v41 = DsmlHead::V41 { system_token: true };
    let tool = ChatMsg {
        role: "tool".into(),
        content: format!("shot:{DSML_IMAGE}done"),
        parts: vec![
            ChatPart::Text("shot:".into()),
            ChatPart::Image(0),
            ChatPart::Text("done".into()),
        ],
        tool_call_id: "c1".into(),
        ..ChatMsg::default()
    };
    let out =
        render_dsml_chat_choice_ex(&[tool], "", ThinkMode::None, ToolChoice::Auto, v41).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(&format!("<tool_result>shot:{DSML_IMAGE}done</tool_result>")));
    assert_eq!(text.matches(DSML_IMAGE).count(), 1);
}
