use ds4_server::generate::chat_format_for_syntax;
use ds4_server::tools::parse_generated_for_model_id;
use ds4_server::{
    openai_sse_stream_update, openai_stream_start, syntax_for_model_id, StreamReq, ThinkMode,
    ToolSchemaOrder, Writer, CREATED_TEST,
};
use serde_json::{json, Value};

const GROUP: &str = concat!(
    "<tool_calls:opensource><tool_call:opensource>lookup",
    "<arg_key:opensource>place</arg_key:opensource><arg_value:opensource> 서울 </arg_value:opensource>",
    "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>2</arg_value:opensource>",
    "</tool_call:opensource><tool_call:opensource>ping</tool_call:opensource></tool_calls:opensource>");
fn orders() -> Vec<ToolSchemaOrder> {
    vec![ToolSchemaOrder {
        name: "lookup".into(),
        prop: vec!["place".into(), "days".into()],
        prop_type: vec!["string".into(), "integer".into()],
        ..Default::default()
    }]
}

#[test]
fn reasoning_and_parallel_calls_at_byte_boundaries() {
    let raw = format!("생각</think:opensource>앞{GROUP}뒤");
    let syntax = syntax_for_model_id(25);
    assert_eq!(syntax as i32, 25);
    let r = StreamReq {
        syntax,
        chat_format: chat_format_for_syntax(syntax),
        has_tools: true,
        think_mode: ThinkMode::High,
        tool_orders: orders(),
        ..Default::default()
    };
    let parsed = parse_generated_for_model_id(25, raw.as_bytes(), true, &r.tool_orders);
    assert!(parsed.ok);
    assert_eq!(parsed.reasoning, "생각".as_bytes());
    assert_eq!(parsed.content, "앞뒤".as_bytes());
    assert_eq!(parsed.raw_tool_text, GROUP);
    assert_eq!(parsed.calls.len(), 2);
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.calls[0].arguments).unwrap(),
        json!({"place":" 서울 ","days":2})
    );
    for width in [1, 2, 7, GROUP.len(), raw.len()] {
        let mut writer = Writer::new(CREATED_TEST);
        let mut state = openai_stream_start(&r);
        let mut current = Vec::new();
        for chunk in raw.as_bytes().chunks(width) {
            current.extend_from_slice(chunk);
            assert!(openai_sse_stream_update(
                &mut writer,
                &r,
                "hy4",
                &mut state,
                &current,
                false
            ));
        }
        assert!(openai_sse_stream_update(
            &mut writer,
            &r,
            "hy4",
            &mut state,
            &current,
            true
        ));
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut names = Vec::new();
        for event in std::str::from_utf8(&writer.out)
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
        {
            let event: Value = serde_json::from_str(event).unwrap();
            let delta = &event["choices"][0]["delta"];
            text.push_str(delta["content"].as_str().unwrap_or(""));
            reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    if let Some(name) = call["function"]["name"].as_str() {
                        names.push(name.to_owned());
                    }
                }
            }
        }
        assert_eq!((reasoning.as_str(), text.as_str()), ("생각", "앞뒤"));
        assert_eq!(names, ["lookup", "ping"]);
    }
}
#[test]
fn incomplete_group_stays_in_answer_channel() {
    let raw = "비밀</think:opensource>답<tool_calls:opensource><tool_call:opensource>lookup";
    let parsed = parse_generated_for_model_id(25, raw.as_bytes(), true, &orders());
    assert!(!parsed.ok);
    assert!(parsed.calls.is_empty());
    assert_eq!(parsed.reasoning, "비밀".as_bytes());
}
