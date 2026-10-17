//! GLM-5.3 chat-render and tool-parse vectors, mirrored from the C reference
//! at /data/ds4 (commit fc80bd6, READ-ONLY).  Every expected byte is the C
//! test's expectation, never the port's current output: a mismatch here is a
//! port divergence.  Each test names the C test and line it encodes.
//!
//! The reasoning tags are written as byte escapes so the exact engine bytes
//! survive any editor: open `\x3cthink\x3e`, close `\x3c/think\x3e`.

use ds4_server::{
    parse_generated_message, render_chat, render_chat_choice, render_live_tool_tail, Api,
    ChatFormat, ChatMsg, ModelSyntax, ThinkMode, ToolCall, ToolChoice, ToolSchemaOrder,
};

fn msg(role: &str, content: &str) -> ChatMsg {
    ChatMsg {
        role: role.into(),
        content: content.into(),
        ..ChatMsg::default()
    }
}

/// C `make_bash_order()` (ds4_server.c:17070): the `bash` order the GLM vectors
/// use to fix argument order.
fn bash_order() -> Vec<ToolSchemaOrder> {
    vec![ToolSchemaOrder {
        name: "bash".into(),
        prop: vec!["command".into()],
        ..Default::default()
    }]
}

/// One assistant turn carrying a single tool call, empty content.
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
fn render_glm53_chat_prompt_text_matches_c() {
    // C `test_render_glm_chat_prompt_text` (ds4_server.c:18960).  The `hidden`
    // schema is dropped (defer_loading) and `strict` stripped, so only `bash`
    // survives; the expected string is the C `expected` literal verbatim.
    let schemas = concat!(
        r#"{"name":"hidden","defer_loading":true,"parameters":{}}"#,
        "\n",
        r#"{"name":"bash","parameters":{"type":"object","properties":{"command":{}}},"strict":true}"#,
    );
    let msgs = [msg("system", "You are terse."), msg("user", "Hello")];
    let prompt = render_chat_choice(
        ModelSyntax::Glm53,
        &msgs,
        schemas,
        &bash_order(),
        ThinkMode::High,
        ToolChoice::Auto,
    )
    .unwrap();
    assert_eq!(
        prompt,
        concat!(
            "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>\n",
            "# Tools\n\n",
            "You may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n",
            "<tools>\n\n",
            "{\"name\": \"bash\", \"parameters\": {\"type\":\"object\",\"properties\":{\"command\":{}}}}\n\n\n",
            "</tools>\n\n",
            "For each function call, output the function name and arguments within the following XML format:\n",
            "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>",
            "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>",
            "<arg_value>{arg-value-2}</arg_value>...</tool_call>",
            "<|system|>You are terse.<|user|>Hello<|assistant|>\x3cthink\x3e",
        )
        .as_bytes()
    );
}

#[test]
fn render_glm53_drops_old_reasoning_without_tools() {
    // C `test_render_glm_drops_old_reasoning_without_tools` (ds4_server.c:19000):
    // an assistant turn before the last user turn loses its reasoning when the
    // history carries no tool context.
    let mut assistant = msg("assistant", "first answer");
    assistant.reasoning = "old hidden reasoning".into();
    let msgs = [msg("user", "first"), assistant, msg("user", "second")];

    let prompt =
        String::from_utf8(render_chat(ModelSyntax::Glm53, &msgs, "", ThinkMode::High).unwrap())
            .unwrap();

    assert!(!prompt.contains("old hidden reasoning"));
    assert!(prompt.contains("<|assistant|>\x3cthink\x3e\x3c/think\x3efirst answer"));
    assert!(prompt.contains("<|user|>second<|assistant|>\x3cthink\x3e"));
}

#[test]
fn render_glm53_preserves_reasoning_with_tools() {
    // C `test_render_glm_preserves_reasoning_with_tools` (ds4_server.c:19027).
    // A tool message in the history keeps the assistant reasoning, and the two
    // tool calls render inline with no newline between them.
    let mut assistant = call("bash", r#"{"command":"pwd"}"#);
    assistant.reasoning = "tool reasoning".into();
    assistant.calls.push(ToolCall {
        id: "call2".into(),
        name: "bash".into(),
        arguments: r#"{"command":"ls"}"#.into(),
    });
    let msgs = [msg("user", "first"), assistant, msg("tool", "/tmp")];

    let prompt = render_chat_choice(
        ModelSyntax::Glm53,
        &msgs,
        "",
        &bash_order(),
        ThinkMode::High,
        ToolChoice::Auto,
    )
    .unwrap();

    assert_eq!(
        prompt,
        concat!(
            "[gMASK]<sop><|system|>Reasoning Effort: High",
            "<|user|>first<|assistant|>\x3cthink\x3etool reasoning\x3c/think\x3e",
            "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value>",
            "</tool_call><tool_call>bash<arg_key>command</arg_key>",
            "<arg_value>ls</arg_value></tool_call>",
            "<|observation|><tool_response>/tmp</tool_response>",
            "<|assistant|>\x3cthink\x3e",
        )
        .as_bytes()
    );
}

#[test]
fn glm53_tool_calls_render_matches_template() {
    // C `test_glm_tool_calls_render_matches_template` (ds4_server.c:19070): the
    // reconstructed block must byte-match the sampled live block.  GLM emits it
    // inline (C append_glm_tool_calls_text, ds4_server.c:2983-2985: "no newline
    // before or after"), and a re-render from the raw splice must equal the
    // reconstructed render.
    let tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\
                 \"parameters\":{\"type\":\"object\",\"properties\":\
                 {\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";
    let raw_block =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    let msgs = [
        msg("user", "What is the weather in Paris?"),
        call("get_weather", r#"{"city":"Paris"}"#),
        msg("tool", "18C, light rain."),
    ];

    let canonical =
        String::from_utf8(render_chat(ModelSyntax::Glm53, &msgs, tools, ThinkMode::High).unwrap())
            .unwrap();
    assert!(canonical.contains(raw_block));
    assert!(canonical.contains("\x3c/think\x3e<tool_call>"));
    assert!(canonical.contains("</tool_call><|observation|>"));
    assert!(!canonical.contains("\x3c/think\x3e\n<tool_call>"));
    assert!(!canonical.contains("</tool_call>\n"));

    // Raw leg. C splices the sampled block from `calls.raw_tool_text`
    // (ds4_server.c:19104).
    let mut spliced = msgs.clone();
    let sampled = raw_block.replace("Paris", "&#80;aris");
    spliced[1].raw_tool_text = sampled.clone();
    let raw = render_chat(ModelSyntax::Glm53, &spliced, tools, ThinkMode::High).unwrap();
    assert_eq!(raw, canonical.replace(raw_block, &sampled).as_bytes());
}

#[test]
fn render_glm53_groups_tool_results() {
    // C `test_render_glm_groups_tool_results` (ds4_server.c:19116).
    // Part A: consecutive tool-role messages collapse under one observation,
    // each wrapped in its own <tool_response>.
    let msgs = [
        msg("user", "inspect both"),
        msg("tool", "first result"),
        msg("tool", "second result"),
    ];
    let prompt =
        String::from_utf8(render_chat(ModelSyntax::Glm53, &msgs, "", ThinkMode::High).unwrap())
            .unwrap();
    assert_eq!(prompt.matches("<|observation|>").count(), 1);
    assert!(prompt.contains(concat!(
        "<|observation|><tool_response>first result</tool_response>",
        "<tool_response>second result</tool_response>",
        "<|assistant|>\x3cthink\x3e",
    )));

    // Part B: a user message carrying multiple tool-call ids holds
    // <tool_result> blocks that normalize into one observation, never
    // rendering as user text.
    let wrapped = ChatMsg {
        role: "user".into(),
        content: "<tool_result>one</tool_result><tool_result>two</tool_result>".into(),
        tool_call_ids: vec!["call_1".into(), "call_2".into()],
        ..ChatMsg::default()
    };
    let prompt = String::from_utf8(
        render_chat(ModelSyntax::Glm53, &[wrapped], "", ThinkMode::High).unwrap(),
    )
    .unwrap();
    assert!(prompt.contains(concat!(
        "<|observation|><tool_response>one</tool_response>",
        "<tool_response>two</tool_response>",
        "<|assistant|>\x3cthink\x3e",
    )));
    assert!(!prompt.contains("<|user|><tool_result>"));
}

#[test]
fn render_glm53_live_tool_tail() {
    // C `render_glm_live_tool_tail` (ds4_server.c:3792), through the port's
    // render_live_tool_tail: the trailing tool results form one observation
    // with one <tool_response> per result, ending on the assistant think head.
    let msgs = [
        msg("user", "inspect both"),
        msg("tool", "first result"),
        msg("tool", "second result"),
    ];
    let tail =
        render_live_tool_tail(ModelSyntax::Glm53, Api::Responses, &msgs, ThinkMode::High).unwrap();
    assert_eq!(
        tail,
        concat!(
            "<|observation|><tool_response>first result</tool_response>",
            "<tool_response>second result</tool_response>",
            "<|assistant|>\x3cthink\x3e",
        )
        .as_bytes()
    );
}

#[test]
fn glm53_parse_tool_call_message_matches_c() {
    // C `test_parse_glm_tool_call_message` (ds4_server.c:19262).  With no
    // declared schema the reference's GLM rule stands: every argument is text
    // (ds4_server.c:6435), so `timeout` is the string "10".
    let generated = concat!(
        "\x3cthink\x3eneed bash\x3c/think\x3eOK\n\n",
        "<tool_call>bash",
        "<arg_key>command</arg_key><arg_value>echo hi</arg_value>",
        "<arg_key>timeout</arg_key><arg_value>10</arg_value>",
        "</tool_call>",
    );

    let parsed = parse_generated_message(
        ModelSyntax::Glm53,
        generated.as_bytes(),
        true,
        ChatFormat::DeepSeek,
        &[],
    );

    assert!(parsed.ok);
    assert_eq!(parsed.reasoning, b"need bash");
    assert_eq!(parsed.content, b"OK");
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "bash");
    assert!(parsed.calls[0]
        .arguments
        .contains("\"command\": \"echo hi\""));
    assert!(parsed.calls[0].arguments.contains("\"timeout\": \"10\""));
    // C keeps the sampled block on `calls.raw_tool_text` (ds4_server.c:19262).
    assert!(parsed.raw_tool_text.starts_with("\n\n<tool_call>bash"));
}
