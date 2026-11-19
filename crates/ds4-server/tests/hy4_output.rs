use ds4_server::generate::chat_format_for_syntax;
use ds4_server::tools::parse_generated_for_model_id;
use ds4_server::{
    anthropic_sse_finish_live, anthropic_sse_start_live, anthropic_sse_stream_update,
    openai_sse_stream_update, openai_stream_start, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init,
    syntax_for_model_id, Api, SemAccum, StreamReq, ThinkMode, ToolSchemaOrder, Writer,
    CREATED_TEST,
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

fn events(writer: &Writer) -> impl Iterator<Item = Value> + '_ {
    std::str::from_utf8(&writer.out)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
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
        let mut scan = SemAccum::init(true, true, true, r.chat_format, b"<think:opensource>");
        let mut current = Vec::new();
        for chunk in raw.as_bytes().chunks(width) {
            current.extend_from_slice(chunk);
            scan.feed(chunk, &[]);
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
        assert!(!scan.thinking_inside() && scan.saw_tool_start && scan.saw_tool_end);
        for event in events(&writer) {
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
fn messages_and_responses_keep_grouped_calls_out_of_text() {
    let raw = format!("생각</think:opensource>앞{GROUP}뒤");
    let syntax = syntax_for_model_id(25);
    let messages_req = StreamReq {
        api: Api::Anthropic,
        syntax,
        chat_format: chat_format_for_syntax(syntax),
        has_tools: true,
        think_mode: ThinkMode::High,
        tool_orders: orders(),
        ..Default::default()
    };
    let responses_req = StreamReq {
        api: Api::Responses,
        reasoning_summary_emit: true,
        ..messages_req.clone()
    };
    let parsed = parse_generated_for_model_id(25, raw.as_bytes(), true, &messages_req.tool_orders);
    for width in [1, raw.len()] {
        let mut messages = Writer::new(CREATED_TEST);
        let mut anth = anthropic_sse_start_live(&mut messages, &messages_req, "hy4", 2);
        let mut responses = Writer::new(CREATED_TEST);
        let mut resp = responses_stream_init(&responses_req, "hy4", "reason", "message");
        responses_sse_created(&mut responses, &responses_req, &mut resp, CREATED_TEST);
        let mut buf = Vec::new();
        for piece in raw.as_bytes().chunks(width) {
            buf.extend_from_slice(piece);
            assert!(anthropic_sse_stream_update(
                &mut messages,
                &messages_req,
                "hy4",
                &mut anth,
                &buf,
                false
            ));
            assert!(responses_sse_stream_update(
                &mut responses,
                &responses_req,
                &mut resp,
                &buf,
                false
            ));
        }
        assert!(anthropic_sse_finish_live(
            &mut messages,
            &messages_req,
            "hy4",
            &mut anth,
            &buf,
            "tool_calls",
            None,
            5,
            &parsed.calls
        ));
        assert!(responses_sse_finish_live(
            &mut responses,
            &responses_req,
            &mut resp,
            &buf,
            "tool_calls",
            2,
            5,
            0,
            CREATED_TEST,
            &parsed.calls
        ));
        let mes: Vec<_> = events(&messages).collect();
        let res: Vec<_> = events(&responses).collect();
        let text: String = mes
            .iter()
            .filter_map(|e| e["delta"]["text"].as_str())
            .collect();
        let thinking: String = mes
            .iter()
            .filter_map(|e| e["delta"]["thinking"].as_str())
            .collect();
        assert_eq!((thinking.as_str(), text.as_str()), ("생각", "앞뒤"));
        let text: String = res
            .iter()
            .filter(|e| e["type"] == "response.output_text.delta")
            .filter_map(|e| e["delta"].as_str())
            .collect();
        let thinking: String = res
            .iter()
            .filter(|e| e["type"] == "response.reasoning_summary_text.delta")
            .filter_map(|e| e["delta"].as_str())
            .collect();
        assert_eq!((thinking.as_str(), text.as_str()), ("생각", "앞뒤"));
        let names: Vec<_> = mes
            .iter()
            .filter_map(|e| e["content_block"]["name"].as_str())
            .collect();
        assert_eq!(names, ["lookup", "ping"]);
        let names: Vec<_> = res
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .filter_map(|e| e["item"]["name"].as_str())
            .collect();
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
