use ds4_server::{render_chat, syntax_for_model_id, ChatMsg, ChatPart, ThinkMode, ToolCall};
use serde_json::Value;

fn stream_events(api: ds4_server::Api, think: ThinkMode, raw: &[u8], width: usize) -> Vec<Value> {
    use ds4_server::stream::*;
    let req = StreamReq {
        api,
        think_mode: think,
        chat_format: ChatFormat::Inkling,
        reasoning_summary_emit: true,
        has_tools: true,
        ..StreamReq::default()
    };
    let mut writer = Writer::new(1);
    let calls = ds4_server::parse_generated_for_model_id(9, raw, true, &[]).calls;
    let finish = if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    match api {
        ds4_server::Api::Openai => {
            let mut state = openai_stream_start(&req);
            for end in (width..raw.len())
                .step_by(width)
                .chain(std::iter::once(raw.len()))
            {
                assert!(openai_sse_stream_update(
                    &mut writer,
                    &req,
                    "test",
                    &mut state,
                    &raw[..end],
                    false
                ));
            }
            assert!(openai_sse_finish_live(
                &mut writer,
                &req,
                "test",
                &mut state,
                raw,
                finish,
                1,
                10,
                &calls
            ));
        }
        ds4_server::Api::Anthropic => {
            let mut state = anthropic_sse_start_live(&mut writer, &req, "test", 1);
            for end in (width..raw.len())
                .step_by(width)
                .chain(std::iter::once(raw.len()))
            {
                assert!(anthropic_sse_stream_update(
                    &mut writer,
                    &req,
                    "test",
                    &mut state,
                    &raw[..end],
                    false
                ));
            }
            assert!(anthropic_sse_finish_live(
                &mut writer,
                &req,
                "test",
                &mut state,
                raw,
                finish,
                None,
                10,
                &calls
            ));
        }
        ds4_server::Api::Responses => {
            let mut state = responses_stream_init(&req, "resp_test", "rs_test", "msg_test");
            responses_sse_created(&mut writer, &req, &mut state, 1);
            for end in (width..raw.len())
                .step_by(width)
                .chain(std::iter::once(raw.len()))
            {
                assert!(responses_sse_stream_update(
                    &mut writer,
                    &req,
                    &mut state,
                    &raw[..end],
                    false
                ));
            }
            assert!(responses_sse_finish_live(
                &mut writer,
                &req,
                &mut state,
                raw,
                finish,
                1,
                10,
                2,
                1,
                &calls
            ));
        }
    }
    String::from_utf8(writer.out)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn streamed_channels() {
    use ds4_server::Api;
    let raw = "<|content_thinking|>Plan🙂<|end_message|><|message_model|><|content_text|>답<|end_message|><|message_model|><|content_text|>변<|end_message|><|content_model_end_sampling|>".as_bytes();
    for api in [Api::Openai, Api::Anthropic, Api::Responses] {
        for think in [ThinkMode::None, ThinkMode::High] {
            for width in [1, 7, raw.len()] {
                let events = stream_events(api, think, raw, width);
                let (mut text, mut reasoning) = (String::new(), String::new());
                for event in &events {
                    match api {
                        Api::Openai => {
                            text.push_str(
                                event["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .unwrap_or(""),
                            );
                            reasoning.push_str(
                                event["choices"][0]["delta"]["reasoning_content"]
                                    .as_str()
                                    .unwrap_or(""),
                            );
                        }
                        Api::Anthropic if event["type"] == "content_block_delta" => {
                            text.push_str(event["delta"]["text"].as_str().unwrap_or(""));
                            reasoning.push_str(event["delta"]["thinking"].as_str().unwrap_or(""));
                        }
                        Api::Responses if event["type"] == "response.output_text.delta" => {
                            text.push_str(event["delta"].as_str().unwrap())
                        }
                        Api::Responses
                            if event["type"] == "response.reasoning_summary_text.delta" =>
                        {
                            reasoning.push_str(event["delta"].as_str().unwrap())
                        }
                        _ => {}
                    }
                }
                assert_eq!(text, "답변", "{api:?} {think:?} width={width}");
                assert_eq!(reasoning, "Plan🙂", "{api:?} {think:?} width={width}");
                assert!(!serde_json::to_string(&events).unwrap().contains("<|"));
                if api == Api::Responses {
                    let output = events.last().unwrap()["response"]["output"]
                        .as_array()
                        .unwrap();
                    assert_eq!(output[0]["summary"][0]["text"], "Plan🙂");
                    assert_eq!(output[1]["content"][0]["text"], "답변");
                }
            }
        }
    }
}

#[test]
fn inkling_thinking_tool_gate() {
    let mut state = ds4_server::SemAccum::init(
        true,
        true,
        false,
        ds4_server::ChatFormat::Inkling,
        b"<|message_model|>",
    );
    state.feed(b"<|content_thinking|>", &[]);
    assert!(state.thinking_inside());
    state.feed(b"Mention <|content_invoke_tool_json|> literally.", &[]);
    assert!(!state.saw_tool_start);
    state.feed(b"<|end_message|>", &[]);
    assert!(!state.thinking_inside());
    state.feed(b"<|message_model|>weather<|content_invoke_tool_json|>", &[]);
    assert!(state.saw_tool_start);
}

#[test]
fn generated_channels() {
    let cases: &[(&[u8], &[u8], &[u8])] = &[
        (b"<|content_text|>4<|end_message|>", b"4", b""),
        (b"<|content_thinking|>Plan<|end_message|><|message_model|><|content_text|>Answer<|end_message|>", b"Answer", b"Plan"),
        (b"<|content_thinking|><|end_message|><|message_model|><|content_text|>4", b"4", b""),
        (b"<|content_thinking|>Not finished", b"", b"Not finished"),
    ];
    for &(raw, content, reasoning) in cases {
        let parsed = ds4_server::parse_generated_for_model_id(9, raw, true, &[]);
        assert!(parsed.ok);
        assert_eq!(parsed.content, content);
        assert_eq!(parsed.reasoning, reasoning);
        assert!(parsed.calls.is_empty());
    }
}

#[test]
fn generated_tool_envelope() {
    let raw = br#"weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Seoul"}}<|end_message|>"#;
    let parsed = ds4_server::parse_generated_for_model_id(9, raw, false, &[]);
    assert!(parsed.ok);
    assert!(parsed.content.is_empty());
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "weather");
    assert_eq!(parsed.calls[0].arguments, r#"{"city":"Seoul"}"#);
    for bad in [
        br#"other<|content_invoke_tool_json|>{"name":"weather","args":{}}<|end_message|>"#.as_slice(),
        br#"weather<|content_invoke_tool_json|>{"name":"weather","args":[]}<|end_message|>"#,
        br#"weather<|content_invoke_tool_json|>{"name":"weather","args":{}} trailing<|end_message|>"#,
    ] {
        let parsed = ds4_server::parse_generated_for_model_id(9, bad, false, &[]);
        assert!(!parsed.ok);
        assert!(parsed.calls.is_empty());
    }
}

#[test]
fn source_chat_vectors() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/inkling/server-vectors.json"
    ))
    .unwrap();
    for row in fixture["vectors"].as_array().unwrap() {
        let messages: Vec<_> = row["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut msg = ChatMsg {
                    role: m["role"].as_str().unwrap().into(),
                    name: m["name"].as_str().unwrap_or_default().into(),
                    content: m["content"].as_str().unwrap_or_default().into(),
                    reasoning: m["reasoning_content"].as_str().unwrap_or_default().into(),
                    tool_call_id: m["tool_call_id"].as_str().unwrap_or_default().into(),
                    ..ChatMsg::default()
                };
                if let Some(calls) = m["tool_calls"].as_array() {
                    msg.calls = calls
                        .iter()
                        .map(|call| ToolCall {
                            id: call["id"].as_str().unwrap().into(),
                            name: call["function"]["name"].as_str().unwrap().into(),
                            arguments: call["function"]["arguments"].to_string(),
                        })
                        .collect();
                }
                if let Some(parts) = m["content"].as_array() {
                    msg.parts = parts
                        .iter()
                        .map(|part| match part["type"].as_str().unwrap() {
                            "text" => ChatPart::Text(part["text"].as_str().unwrap().into()),
                            "image" => ChatPart::Image(0),
                            _ => unreachable!(),
                        })
                        .collect();
                }
                msg
            })
            .collect();
        let think = match row["effort"].as_str().unwrap() {
            "none" => ThinkMode::None,
            "low" => ThinkMode::Low,
            "high" => ThinkMode::High,
            "max" => ThinkMode::Max,
            _ => unreachable!(),
        };
        let tools = if row["tools"].as_array().unwrap().is_empty() {
            String::new()
        } else {
            row["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        };
        let actual = render_chat(syntax_for_model_id(9), &messages, &tools, think).unwrap();
        assert_eq!(
            String::from_utf8(actual).unwrap(),
            row["rendered"].as_str().unwrap(),
            "{} / {}",
            row["name"],
            row["effort"]
        );
    }
}

#[test]
fn streamed_unclosed_text() {
    let events = stream_events(
        ds4_server::Api::Anthropic,
        ThinkMode::None,
        b"<|content_text|>unfinished",
        1,
    );
    let starts = events
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .count();
    let stops = events
        .iter()
        .filter(|e| e["type"] == "content_block_stop")
        .count();
    assert_eq!(starts, 1);
    assert_eq!(starts, stops);
}

#[test]
fn wire_tool_declarations() {
    use ds4_server::route::WireSurface;
    let cases = [
        (
            WireSurface::OpenaiChat,
            serde_json::json!({"messages": [{"role": "user", "content": "Hi"}], "tools": [
                {"type": "function", "function": {"name": "one", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "two", "description": "Second", "parameters": {"type": "object"}}}
            ]}),
        ),
        (
            WireSurface::Anthropic,
            serde_json::json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "tools": [
                {"name": "one", "input_schema": {"type": "object"}},
                {"name": "two", "description": "Second", "input_schema": {"type": "object"}}
            ]}),
        ),
        (
            WireSurface::Responses,
            serde_json::json!({"input": "Hi", "tools": [
                {"type": "function", "name": "one", "parameters": {"type": "object"}},
                {"type": "function", "name": "two", "description": "Second", "parameters": {"type": "object"}}
            ]}),
        ),
    ];
    let expected = r#"<|message_system|>tool_declare<|content_xml|>[{"description":"","name":"one","parameters":{"type":"object"},"type":"function"},{"description":"Second","name":"two","parameters":{"type":"object"},"type":"function"}]<|end_message|>"#;
    for (surface, body) in cases {
        let request =
            ds4_server::parse_request(surface, &ds4_server::ParseEnv::default(), &body.to_string())
                .unwrap();
        let actual = ds4_server::render_prompt(&request, 9).unwrap();
        assert!(String::from_utf8(actual).unwrap().starts_with(expected));
    }
}

#[test]
fn responses_tool_deltas() {
    let raw = br#"weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Seoul"}}<|end_message|>"#;
    let events = stream_events(ds4_server::Api::Responses, ThinkMode::None, raw, 1);
    let mut arguments = String::new();
    for event in events {
        match event["type"].as_str().unwrap() {
            "response.output_item.added" if event["item"]["type"] == "function_call" => {
                arguments = event["item"]["arguments"].as_str().unwrap().into();
            }
            "response.function_call_arguments.delta" => {
                arguments.push_str(event["delta"].as_str().unwrap());
            }
            "response.function_call_arguments.done" => {
                assert_eq!(arguments, event["arguments"].as_str().unwrap());
                assert_eq!(
                    serde_json::from_str::<Value>(&arguments).unwrap(),
                    serde_json::json!({"city": "Seoul"})
                );
            }
            _ => {}
        }
    }
}
