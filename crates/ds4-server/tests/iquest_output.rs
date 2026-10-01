//! IQuest's source XML output protocol; all tests are model-free.
use ds4_server::generate::chat_format_for_syntax;
use ds4_server::tools::{parse_generated_for_model_id, parse_generated_for_response};
use ds4_server::{
    anthropic_sse_finish_live, anthropic_sse_start_live, anthropic_sse_stream_update,
    openai_sse_finish_live, openai_sse_stream_update, openai_stream_start, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init,
    syntax_for_model_id, Api, SemAccum, StreamReq, ThinkMode, ToolSchemaOrder, Writer,
    CREATED_TEST,
};
use serde_json::{json, Value};

const IQUEST: i32 = 15;
const PROMPT: &[u8] = b"<|iquest_assistant|><think>";
const CALL: &str = concat!(
    "<iquest_tool_call>lookup",
    "<arg_key>text</arg_key><arg_value> 123 </arg_value>",
    "<arg_key>count</arg_key><arg_value>2</arg_value>",
    "<arg_key>active</arg_key><arg_value>true</arg_value>",
    "<arg_key>data</arg_key><arg_value>{\"city\":\"서울\",\"items\":[1,null]}</arg_value>",
    "</iquest_tool_call>"
);

fn orders() -> Vec<ToolSchemaOrder> {
    vec![ToolSchemaOrder {
        name: "lookup".into(),
        prop: vec![
            "text".into(),
            "count".into(),
            "active".into(),
            "data".into(),
        ],
        prop_type: vec![
            "string".into(),
            "integer".into(),
            "boolean".into(),
            "object".into(),
        ],
        ..Default::default()
    }]
}

#[test]
fn iquest_model_mapping() {
    assert_eq!(syntax_for_model_id(IQUEST) as i32, IQUEST);
}

#[test]
fn iquest_source_types_and_surrounding_text() {
    let raw =
        format!("plan</think>Before.{CALL}Between.<iquest_tool_call>ping</iquest_tool_call>After.");
    let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), true, &orders());
    assert!(parsed.ok);
    assert_eq!(parsed.reasoning, b"plan");
    assert_eq!(parsed.content, b"Before.Between.After.");
    assert_eq!(parsed.calls.len(), 2);
    assert_eq!(parsed.calls[0].name, "lookup");
    let args: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
    assert_eq!(
        args,
        json!({"text":" 123 ","count":2,"active":true,"data":{"city":"서울","items":[1,null]}})
    );
    assert_eq!(parsed.calls[1].name, "ping");
    assert_eq!(parsed.calls[1].arguments, "{}");
}

#[test]
fn iquest_malformed_and_disabled_tools_stay_text() {
    let broken = b"<iquest_tool_call>run<arg_key>x</arg_key><arg_value>broken</iquest_tool_call>";
    let parsed = parse_generated_for_model_id(IQUEST, broken, false, &[]);
    assert!(parsed.ok && parsed.calls.is_empty());
    assert_eq!(parsed.content, broken);
    let syntax = syntax_for_model_id(IQUEST);
    let format = chat_format_for_syntax(syntax);
    let (parsed, _) = parse_generated_for_response(
        syntax,
        CALL.as_bytes(),
        false,
        false,
        false,
        format,
        &orders(),
        "stop",
    );
    assert!(parsed.calls.is_empty());
    assert_eq!(parsed.content, CALL.as_bytes());
}

#[test]
fn iquest_bytewise_semantic_markers() {
    let raw = format!("plan</think>{CALL}");
    let format = chat_format_for_syntax(syntax_for_model_id(IQUEST));
    for width in 1..=raw.len() {
        let mut acc = SemAccum::init(true, true, true, format, PROMPT);
        for chunk in raw.as_bytes().chunks(width) {
            acc.feed(chunk, &[]);
        }
        assert!(!acc.thinking_inside(), "width {width}");
        assert!(acc.saw_tool_start && acc.saw_tool_end, "width {width}");
        // Closing one call must leave generation open for subsequent calls/text.
        assert_eq!(acc.verdict, None, "width {width}");
    }
}

fn events(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn request() -> StreamReq {
    let syntax = syntax_for_model_id(IQUEST);
    StreamReq {
        syntax,
        chat_format: chat_format_for_syntax(syntax),
        has_tools: true,
        think_mode: ThinkMode::None,
        tool_orders: orders(),
        ..Default::default()
    }
}

#[test]
fn iquest_openai_stream_every_split() {
    let raw = format!("before{CALL}between<iquest_tool_call>ping</iquest_tool_call>after");
    let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), false, &orders());
    let r = request();
    for width in 1..=raw.len() {
        let mut w = Writer::new(CREATED_TEST);
        let mut state = openai_stream_start(&r);
        let mut current = Vec::new();
        for chunk in raw.as_bytes().chunks(width) {
            current.extend(chunk);
            assert!(openai_sse_stream_update(
                &mut w, &r, "job", &mut state, &current, false
            ));
        }
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut state,
            &current,
            "tool_calls",
            2,
            5,
            &parsed.calls
        ));
        let mut text = String::new();
        let mut calls: std::collections::BTreeMap<i64, (String, String, String)> =
            Default::default();
        for e in events(&w.out) {
            let delta = &e["choices"][0]["delta"];
            text.push_str(delta["content"].as_str().unwrap_or(""));
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let entry = calls.entry(call["index"].as_i64().unwrap()).or_default();
                if let Some(id) = call["id"].as_str() {
                    entry.0.push_str(id);
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    entry.1.push_str(name);
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    entry.2.push_str(args);
                }
            }
        }
        assert_eq!(text, "beforebetweenafter", "width {width}");
        assert_eq!(calls.len(), 2, "width {width}");
        assert_eq!(calls[&0].1, "lookup");
        assert_eq!(calls[&1].1, "ping");
        assert_ne!(calls[&0].0, calls[&1].0);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[&0].2).unwrap(),
            serde_json::from_str::<Value>(&parsed.calls[0].arguments).unwrap()
        );
        assert_eq!(calls[&1].2, "{}");
    }
}

#[test]
fn iquest_anthropic_and_responses_stream_splits() {
    let raw = format!("before{CALL}between<iquest_tool_call>ping</iquest_tool_call>after");
    let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), false, &orders());
    for width in [1, 3, 23, raw.len()] {
        let mut r = request();
        r.api = Api::Anthropic;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = anthropic_sse_start_live(&mut w, &r, "job", 2);
        let mut current = Vec::new();
        for chunk in raw.as_bytes().chunks(width) {
            current.extend(chunk);
            assert!(anthropic_sse_stream_update(
                &mut w, &r, "job", &mut st, &current, false
            ));
        }
        assert!(anthropic_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            &current,
            "tool_calls",
            None,
            5,
            &parsed.calls
        ));
        let records = events(&w.out);
        let text: String = records
            .iter()
            .filter_map(|e| e["delta"]["text"].as_str())
            .collect();
        let names: Vec<_> = records
            .iter()
            .filter_map(|e| e["content_block"]["name"].as_str())
            .collect();
        assert_eq!(text, "beforebetweenafter");
        assert_eq!(names, ["lookup", "ping"]);
        assert_eq!(
            records
                .iter()
                .filter(|e| e["type"] == "content_block_start")
                .count(),
            records
                .iter()
                .filter(|e| e["type"] == "content_block_stop")
                .count()
        );
        r.api = Api::Responses;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = responses_stream_init(&r, "resp", "reason", "message");
        responses_sse_created(&mut w, &r, &mut st, CREATED_TEST);
        let mut current = Vec::new();
        for chunk in raw.as_bytes().chunks(width) {
            current.extend(chunk);
            assert!(responses_sse_stream_update(
                &mut w, &r, &mut st, &current, false
            ));
        }
        assert!(responses_sse_finish_live(
            &mut w,
            &r,
            &mut st,
            &current,
            "tool_calls",
            2,
            5,
            0,
            CREATED_TEST,
            &parsed.calls
        ));
        let records = events(&w.out);
        let text: String = records
            .iter()
            .filter(|e| e["type"] == "response.output_text.delta")
            .filter_map(|e| e["delta"].as_str())
            .collect();
        assert_eq!(text, "beforebetweenafter");
        let output = records.last().unwrap()["response"]["output"]
            .as_array()
            .unwrap();
        assert_eq!(output[0]["content"][0]["text"], "beforebetweenafter");
        assert_eq!(output[1]["name"], "lookup");
        assert_eq!(output[2]["name"], "ping");
        assert!(!std::str::from_utf8(&w.out)
            .unwrap()
            .contains("iquest_tool_call"));
    }
}

#[test]
fn iquest_stream_malformed_and_disabled() {
    let raw = "<iquest_tool_call>run<arg_key>x</arg_key><arg_value>broken</iquest_tool_call>";
    for enabled in [true, false] {
        let mut r = request();
        r.has_tools = enabled;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = openai_stream_start(&r);
        for end in 1..=raw.len() {
            assert!(openai_sse_stream_update(
                &mut w,
                &r,
                "job",
                &mut st,
                &raw.as_bytes()[..end],
                false
            ));
        }
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            raw.as_bytes(),
            "stop",
            2,
            5,
            &[]
        ));
        let text: String = events(&w.out)
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, raw);
        let mut acc = SemAccum::init(true, enabled, false, r.chat_format, b"");
        for byte in raw.bytes() {
            acc.feed(&[byte], &[]);
        }
        assert_eq!(acc.verdict, None);
    }
}

#[test]
fn iquest_official_parser_vectors() {
    let vectors: Value =
        serde_json::from_str(include_str!("fixtures/iquest-tool-parser.json")).unwrap();
    for case in vectors["cases"].as_array().unwrap() {
        let raw = case["text"].as_str().unwrap().as_bytes();
        let enabled = case["tools_enabled"].as_bool().unwrap();
        let r = request();
        let (parsed, _) = parse_generated_for_response(
            r.syntax,
            raw,
            enabled,
            false,
            false,
            r.chat_format,
            &orders(),
            "stop",
        );
        assert!(parsed.ok, "{}", case["name"]);
        assert_eq!(
            std::str::from_utf8(&parsed.content).unwrap(),
            case["content"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        let calls: Vec<Value> = parsed.calls.iter().map(|call| json!({
            "name":call.name,"arguments":serde_json::from_str::<Value>(&call.arguments).unwrap()
        })).collect();
        assert_eq!(Value::Array(calls), case["calls"], "{}", case["name"]);
        let mut r = r;
        r.has_tools = enabled;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = openai_stream_start(&r);
        for end in 1..=raw.len() {
            assert!(openai_sse_stream_update(
                &mut w,
                &r,
                "job",
                &mut st,
                &raw[..end],
                false
            ));
        }
        let records = events(&w.out);
        let text: String = records
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(
            text, case["stream_content_before_final"],
            "{} streamed before final",
            case["name"]
        );
        let mut calls: std::collections::BTreeMap<i64, (String, String)> = Default::default();
        for call in records
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
        {
            let entry = calls.entry(call["index"].as_i64().unwrap()).or_default();
            entry
                .0
                .push_str(call["function"]["name"].as_str().unwrap_or(""));
            entry
                .1
                .push_str(call["function"]["arguments"].as_str().unwrap_or(""));
        }
        let calls: Vec<Value> = calls.into_iter().map(|(index,(name,args))|
            json!({"index":index,"name":name,"arguments":serde_json::from_str::<Value>(&args).unwrap()})).collect();
        assert_eq!(
            Value::Array(calls),
            case["stream_calls_before_final"],
            "{} streamed calls",
            case["name"]
        );
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            raw,
            "stop",
            1,
            1,
            &parsed.calls
        ));
        let text: String = events(&w.out)
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, case["content"], "{} streamed", case["name"]);
    }
}

#[test]
fn iquest_reasoning_boundaries() {
    let hidden = format!("private{CALL}");
    let visible = format!("visible{CALL}");
    for prefix in ["", "<think>"] {
        let raw = format!("{prefix}{hidden}</think>{visible}");
        let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), true, &orders());
        assert_eq!(parsed.reasoning, hidden.as_bytes());
        assert_eq!(parsed.content, b"visible");
        assert_eq!(parsed.calls.len(), 1);
        let mut r = request();
        r.think_mode = ThinkMode::High;
        for width in [1, 3, 17, raw.len()] {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let mut current = Vec::new();
            for chunk in raw.as_bytes().chunks(width) {
                current.extend(chunk);
                assert!(openai_sse_stream_update(
                    &mut w, &r, "job", &mut st, &current, false
                ));
            }
            assert!(openai_sse_finish_live(
                &mut w,
                &r,
                "job",
                &mut st,
                &current,
                "tool_calls",
                1,
                1,
                &parsed.calls
            ));
            let records = events(&w.out);
            let reasoning: String = records
                .iter()
                .filter_map(|e| e["choices"][0]["delta"]["reasoning_content"].as_str())
                .collect();
            let text: String = records
                .iter()
                .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
                .collect();
            assert_eq!(reasoning, hidden);
            assert_eq!(text, "visible");
            let calls = records
                .iter()
                .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
                .flatten()
                .filter(|call| call["function"]["name"].is_string())
                .count();
            assert_eq!(calls, 1);
        }
        let unfinished = format!("{prefix}{hidden}");
        let parsed = parse_generated_for_model_id(IQUEST, unfinished.as_bytes(), true, &orders());
        assert_eq!(parsed.reasoning, hidden.as_bytes());
        assert!(parsed.content.is_empty() && parsed.calls.is_empty());
    }
    let raw = format!("discarded<think>private</think>{visible}");
    let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), true, &orders());
    assert_eq!(parsed.reasoning, b"private");
    let (parsed, _) = parse_generated_for_response(
        syntax_for_model_id(IQUEST),
        raw.as_bytes(),
        false,
        false,
        true,
        chat_format_for_syntax(syntax_for_model_id(IQUEST)),
        &orders(),
        "stop",
    );
    assert_eq!(parsed.reasoning, b"private");
    assert_eq!(parsed.content, visible.as_bytes());
    assert!(parsed.calls.is_empty());
}

#[test]
fn iquest_arbitrary_integer_arguments() {
    for number in ["18446744073709551617", "-18446744073709551617"] {
        let raw = format!("<iquest_tool_call>lookup<arg_key>count</arg_key><arg_value>{number}</arg_value></iquest_tool_call>");
        let parsed = parse_generated_for_model_id(IQUEST, raw.as_bytes(), false, &orders());
        let expected = format!("{{\"count\":{number}}}");
        assert_eq!(parsed.calls[0].arguments, expected);
        let r = request();
        let mut w = Writer::new(CREATED_TEST);
        let mut st = openai_stream_start(&r);
        assert!(openai_sse_stream_update(
            &mut w,
            &r,
            "job",
            &mut st,
            raw.as_bytes(),
            false
        ));
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            raw.as_bytes(),
            "tool_calls",
            1,
            1,
            &parsed.calls
        ));
        let args: String = events(&w.out)
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
            .filter_map(|call| call["function"]["arguments"].as_str())
            .collect();
        assert_eq!(args, expected);
    }
}

#[test]
fn iquest_malformed_think_prelude_contract() {
    let raw = b"discarded<think>private</think>visible";
    let parsed = parse_generated_for_model_id(IQUEST, raw, true, &orders());
    assert_eq!(parsed.reasoning, b"private");
    assert_eq!(parsed.content, b"visible");
    for width in [1, 2, raw.len()] {
        let mut r = request();
        r.think_mode = ThinkMode::High;
        r.has_tools = false;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = openai_stream_start(&r);
        let mut current = Vec::new();
        for chunk in raw.chunks(width) {
            current.extend(chunk);
            assert!(openai_sse_stream_update(
                &mut w, &r, "job", &mut st, &current, false
            ));
        }
        assert!(openai_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            raw,
            "stop",
            1,
            1,
            &[]
        ));
        let records = events(&w.out);
        let reasoning: String = records
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["reasoning_content"].as_str())
            .collect();
        let text: String = records
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        // Already emitted malformed prelude bytes cannot be retracted.
        assert_eq!(reasoning, "discarded<think>private");
        assert_eq!(text, "visible");
        r.api = Api::Anthropic;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = anthropic_sse_start_live(&mut w, &r, "job", 1);
        let mut current = Vec::new();
        for chunk in raw.chunks(width) {
            current.extend(chunk);
            assert!(anthropic_sse_stream_update(
                &mut w, &r, "job", &mut st, &current, false
            ));
        }
        assert!(anthropic_sse_finish_live(
            &mut w,
            &r,
            "job",
            &mut st,
            raw,
            "stop",
            None,
            1,
            &[]
        ));
        let records = events(&w.out);
        let reasoning: String = records
            .iter()
            .filter_map(|e| e["delta"]["thinking"].as_str())
            .collect();
        let text: String = records
            .iter()
            .filter_map(|e| e["delta"]["text"].as_str())
            .collect();
        assert_eq!(reasoning, "discarded<think>private");
        assert_eq!(text, "visible");
        r.api = Api::Responses;
        r.reasoning_summary_emit = true;
        let mut w = Writer::new(CREATED_TEST);
        let mut st = responses_stream_init(&r, "resp", "reason", "message");
        responses_sse_created(&mut w, &r, &mut st, CREATED_TEST);
        let mut current = Vec::new();
        for chunk in raw.chunks(width) {
            current.extend(chunk);
            assert!(responses_sse_stream_update(
                &mut w, &r, &mut st, &current, false
            ));
        }
        assert!(responses_sse_finish_live(
            &mut w,
            &r,
            &mut st,
            raw,
            "stop",
            1,
            1,
            1,
            CREATED_TEST,
            &[]
        ));
        let records = events(&w.out);
        let output = records.last().unwrap()["response"]["output"]
            .as_array()
            .unwrap();
        assert_eq!(output[0]["summary"][0]["text"], "discarded<think>private");
        assert_eq!(output[1]["content"][0]["text"], "visible");
    }
}

#[test]
fn iquest_implicit_reasoning_is_incremental() {
    let mut r = request();
    r.think_mode = ThinkMode::High;
    r.has_tools = false;
    let raw = b"ordinary native reasoning continues";
    let mut w = Writer::new(CREATED_TEST);
    let mut st = openai_stream_start(&r);
    assert!(openai_sse_stream_update(
        &mut w, &r, "job", &mut st, raw, false
    ));
    let reasoning: String = events(&w.out)
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    assert!(!reasoning.is_empty());
    assert!(std::str::from_utf8(raw).unwrap().starts_with(&reasoning));
    assert!(!st.checked_think_prefix || st.mode == ds4_server::stream::OpenaiMode::Thinking);
}
