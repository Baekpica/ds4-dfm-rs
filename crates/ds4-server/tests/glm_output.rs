use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::ChatThinkMode;
use ds4_server::{
    openai_sse_finish_live, openai_sse_stream_update, openai_stream_start, parse_chat_request,
    parse_generated_for_response, parse_generated_message, ChatFormat, ModelSyntax, ParseEnv,
    StreamReq, ThinkMode, ToolSchemaOrder, Writer,
};
use serde_json::{json, Value};

const TEMPLATE: &str =
    include_str!("../../../tests/fixtures/chat-template/models/glm-uncensored/chat_template.jinja");

fn source_call(arguments: Value) -> Vec<u8> {
    let template = Template::compile(TEMPLATE, RenderClock::Fixed(0)).unwrap();
    let messages = [
        json!({"role":"user","content":"Use inspect."}),
        json!({"role":"assistant","content":"", "tool_calls":[{
            "type":"function", "function":{"name":"inspect","arguments":arguments}
        }]}),
    ];
    let rendered = template
        .render_chat(&messages, &[], ChatOptions::new(7, ChatThinkMode::None))
        .unwrap();
    let start = rendered.find("<tool_call>").unwrap();
    let end = start + rendered[start..].find("</tool_call>").unwrap() + "</tool_call>".len();
    rendered.as_bytes()[start..end].to_vec()
}

fn orders(properties: Value) -> Vec<ToolSchemaOrder> {
    let env = ParseEnv {
        default_model: "glm-5.3-flash".into(),
        default_tokens: 2048,
        default_effort: ds4_server::ThinkMode::None,
        default_temp: ds4_server::default_temperature(),
        live_ids: Vec::new(),
        engine_defaults: false,
    };
    let request = json!({"messages":[{"role":"user","content":"Use inspect."}],
        "tools":[{"type":"function","function":{"name":"inspect",
        "parameters":{"type":"object","properties":properties}}}]});
    parse_chat_request(&env, &request.to_string())
        .unwrap()
        .tool_orders
}

fn parse_call(text: &[u8], orders: &[ToolSchemaOrder]) -> Value {
    let (parsed, finish) = parse_generated_for_response(
        ModelSyntax::Glm53,
        text,
        true,
        true,
        false,
        ChatFormat::DeepSeek,
        orders,
        "tool_calls",
    );
    assert!(parsed.ok);
    assert_eq!(finish, "tool_calls");
    assert_eq!(parsed.calls.len(), 1);
    serde_json::from_str(&parsed.calls[0].arguments).unwrap()
}

#[test]
fn glm_source_types_round_trip() {
    let arguments = json!({
        "text":"  &amp; &lt; true [1] {\"x\":2}  ",
        "number":1.25, "integer":7, "flag":true,
        "list":[1,"&amp;"], "map":{"x":2}, "empty":null,
    });
    let schemas = orders(json!({
        "text":{"type":"string"}, "number":{"type":"number"},
        "integer":{"type":"integer"}, "flag":{"type":"boolean"},
        "list":{"type":"array"}, "map":{"type":"object"}, "empty":{"type":"null"},
    }));
    assert_eq!(
        parse_call(&source_call(arguments.clone()), &schemas),
        arguments
    );
}

#[test]
fn glm_ambiguous_values_stay_raw() {
    let arguments = json!({
        "unknown":"[1]", "union":"true", "mismatch":"[1]", "escaped":"&amp;",
    });
    let schemas = orders(json!({
        "union":{"type":["string","boolean"]}, "mismatch":{"type":"integer"},
        "escaped":{"type":"string"},
    }));
    assert_eq!(
        parse_call(&source_call(arguments.clone()), &schemas),
        arguments
    );
    assert_eq!(parse_call(&source_call(arguments.clone()), &[]), arguments);
}

#[test]
fn glm_stream_keeps_source_types() {
    let arguments = json!({"text":"&amp;", "number":1.25});
    let raw = source_call(arguments.clone());
    let req = StreamReq {
        syntax: ModelSyntax::Glm53,
        chat_format: ChatFormat::DeepSeek,
        has_tools: true,
        think_mode: ThinkMode::None,
        tool_orders: orders(json!({"text":{"type":"string"}, "number":{"type":"number"}})),
        ..StreamReq::default()
    };
    let mut writer = Writer::new(0);
    let mut stream = openai_stream_start(&req);
    for end in 1..=raw.len() {
        assert!(openai_sse_stream_update(
            &mut writer,
            &req,
            "glm",
            &mut stream,
            &raw[..end],
            false,
        ));
    }
    let parsed =
        parse_generated_message(req.syntax, &raw, false, req.chat_format, &req.tool_orders);
    assert!(openai_sse_finish_live(
        &mut writer,
        &req,
        "glm",
        &mut stream,
        &raw,
        "tool_calls",
        1,
        1,
        &parsed.calls,
    ));
    let tape = String::from_utf8(writer.out).unwrap();
    let mut emitted = String::new();
    for data in tape.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            continue;
        }
        let frame: Value = serde_json::from_str(data).unwrap();
        let delta = &frame["choices"][0]["delta"];
        assert!(
            delta.get("content").is_none(),
            "raw tool XML leaked: {frame}"
        );
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                if let Some(part) = call["function"]["arguments"].as_str() {
                    emitted.push_str(part);
                }
            }
        }
    }
    assert_eq!(serde_json::from_str::<Value>(&emitted).unwrap(), arguments);
}
