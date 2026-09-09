use super::*;
use crate::parse::{ImageMime, ParseEnv, RequestAudio, RequestImage, ToolCall, ToolChoice};
use crate::tools::ParsedGenerated;
use crate::{parse_anthropic_request, parse_responses_request};
use std::sync::Arc;

const CALL_ID: &str = "call_saved";
const FOREIGN_ID: &str = "call_foreign";

fn env() -> ParseEnv {
    ParseEnv {
        default_model: "inkling-small-mq85gb".into(),
        default_effort: ThinkMode::High,
        live_ids: vec![CALL_ID.into(), FOREIGN_ID.into()],
        ..ParseEnv::default()
    }
}

fn original(api: Api) -> ParsedRequest {
    match api {
        Api::Responses => parse_responses_request(
            &env(),
            r#"{
                "instructions":"원래 시스템",
                "input":[
                    {"role":"user","content":"이전 질문"},
                    {"role":"assistant","content":"이전 답"},
                    {"role":"user","content":"서울을 찾아 줘."}
                ],
                "tools":[{"type":"function","name":"lookup","description":"장소 검색",
                    "parameters":{"type":"object","properties":{"city":{"type":"string"},"nested":{"type":"object"}},"required":["city"]}
                }]
            }"#,
        )
        .unwrap(),
        Api::Anthropic => parse_anthropic_request(
            &env(),
            r#"{
                "system":"원래 시스템",
                "messages":[
                    {"role":"user","content":"이전 질문"},
                    {"role":"assistant","content":"이전 답"},
                    {"role":"user","content":"서울을 찾아 줘."}
                ],
                "tools":[{"name":"lookup","description":"장소 검색",
                    "input_schema":{"type":"object","properties":{"city":{"type":"string"},"nested":{"type":"object"}},"required":["city"]}
                }]
            }"#,
        )
        .unwrap(),
        _ => panic!("unsupported fixture API"),
    }
}

fn generated() -> ParsedGenerated {
    ParsedGenerated {
        content: "확인하겠습니다.".as_bytes().to_vec(),
        reasoning: "검증 후 호출한다.".as_bytes().to_vec(),
        calls: vec![ToolCall {
            id: CALL_ID.into(),
            name: "lookup".into(),
            arguments: r#"{"city":"서울","nested":{"z":2,"a":true}}"#.into(),
        }],
        raw_dsml: String::new(),
        raw_tool_text: String::new(),
        ok: true,
        recovered: false,
    }
}

fn saved_messages() -> Vec<Value> {
    vec![
        json!({"role":"system","content":"원래 시스템"}),
        json!({"role":"user","content":"이전 질문"}),
        json!({"role":"assistant","content":"이전 답"}),
        json!({"role":"user","content":"서울을 찾아 줘."}),
        json!({"role":"assistant","content":"확인하겠습니다.","reasoning_content":"검증 후 호출한다.",
            "tool_calls":[{"id":"call_saved","type":"function","function":{"name":"lookup",
                "arguments":{"city":"서울","nested":{"z":2,"a":true}}
            }}]
        }),
    ]
}

fn response_tail() -> ParsedRequest {
    parse_responses_request(
        &env(),
        r#"{"input":[{"type":"function_call_output","call_id":"call_saved","output":"서울: 맑음"}]}"#,
    )
    .unwrap()
}

fn anthropic_tail() -> ParsedRequest {
    parse_anthropic_request(
        &env(),
        r#"{"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"call_saved","content":"서울: 맑음"}
        ]}]}"#,
    )
    .unwrap()
}

#[test]
fn response_tail_rebuild() {
    let captured = original(Api::Responses);
    let schemas = captured.tool_schemas.clone();
    let orders = format!("{:?}", captured.tool_orders);
    let history = History::capture(captured, &generated());
    let mut next = response_tail();

    assert!(history.matches(&next));
    assert!(history.restore(&mut next).unwrap());

    let mut expected = saved_messages();
    expected.push(json!({"role":"tool","tool_call_id":"call_saved","content":"서울: 맑음"}));
    assert_eq!(messages(&next).unwrap(), expected);
    assert_eq!(next.tool_schemas, schemas);
    assert_eq!(format!("{:?}", next.tool_orders), orders);
    assert!(next.has_tools);
}

#[test]
fn anthropic_tail_rebuild() {
    let captured = original(Api::Anthropic);
    let schemas = captured.tool_schemas.clone();
    let history = History::capture(captured, &generated());
    let mut next = anthropic_tail();

    assert!(history.matches(&next));
    assert!(history.restore(&mut next).unwrap());

    let mut expected = saved_messages();
    expected.push(json!({"role":"tool","tool_call_id":"call_saved","content":"서울: 맑음"}));
    assert_eq!(messages(&next).unwrap(), expected);
    assert_eq!(next.tool_schemas, schemas);
}

#[test]
fn reject_foreign_live_ids() {
    let history = History::capture(original(Api::Responses), &generated());
    let mut next = parse_responses_request(
        &env(),
        r#"{"input":[
            {"type":"function_call_output","call_id":"call_saved","output":"ours"},
            {"type":"function_call_output","call_id":"call_foreign","output":"unrelated"}
        ]}"#,
    )
    .unwrap();
    let before = format!("{next:?}");

    assert!(!history.matches(&next));
    assert!(history.restore(&mut next).is_err());
    assert_eq!(format!("{next:?}"), before);
}

#[test]
fn retain_caller_history() {
    let history = History::capture(original(Api::Responses), &generated());
    let mut next = parse_responses_request(
        &env(),
        r#"{
            "instructions":"수정한 시스템",
            "input":[
                {"role":"user","content":"수정한 질문"},
                {"role":"assistant","content":"수정한 답"},
                {"type":"function_call","call_id":"call_saved","name":"lookup",
                 "arguments":{"city":"서울","nested":{"z":2,"a":true}}},
                {"type":"function_call_output","call_id":"call_saved","output":"새 결과"}
            ],
            "tools":[{"type":"function","name":"new_tool","description":"새 도구",
                "parameters":{"type":"object","properties":{}}}]
        }"#,
    )
    .unwrap();
    let schemas = next.tool_schemas.clone();
    let orders = format!("{:?}", next.tool_orders);

    assert!(history.matches(&next));
    history.restore(&mut next).unwrap();

    assert_eq!(
        messages(&next).unwrap(),
        vec![
            json!({"role":"system","content":"수정한 시스템"}),
            json!({"role":"user","content":"수정한 질문"}),
            json!({"role":"assistant","content":"수정한 답","reasoning_content":"검증 후 호출한다.",
                "tool_calls":[{"id":"call_saved","type":"function","function":{"name":"lookup",
                    "arguments":{"city":"서울","nested":{"z":2,"a":true}}
                }}]
            }),
            json!({"role":"tool","tool_call_id":"call_saved","content":"새 결과"}),
        ]
    );
    assert_eq!(next.tool_schemas, schemas);
    assert_eq!(format!("{:?}", next.tool_orders), orders);
}

#[test]
fn override_system_and_tools() {
    let history = History::capture(original(Api::Anthropic), &generated());
    let mut next = parse_anthropic_request(
        &env(),
        r#"{
            "system":"새 시스템",
            "tool_choice":{"type":"none"},
            "messages":[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"call_saved","content":"서울: 맑음"}
            ]}]
        }"#,
    )
    .unwrap();

    assert!(history.restore(&mut next).unwrap());

    let mut expected = saved_messages();
    expected[0] = json!({"role":"system","content":"새 시스템"});
    expected.push(json!({"role":"tool","tool_call_id":"call_saved","content":"서울: 맑음"}));
    assert_eq!(messages(&next).unwrap(), expected);
    assert_eq!(next.tool_choice, ToolChoice::None);
    assert!(next.tool_schemas.is_empty());
    assert!(next.tool_orders.is_empty());
    assert!(!next.has_tools);
}

fn image(data: &[u8]) -> RequestImage {
    RequestImage {
        mime: ImageMime::Png,
        data: Arc::from(data),
    }
}

fn audio(data: &[u8]) -> RequestAudio {
    RequestAudio {
        data: Arc::from(data),
    }
}

#[test]
fn prepend_media_indices() {
    let mut captured = original(Api::Anthropic);
    captured.images = vec![image(b"old-image-a"), image(b"old-image-b")];
    captured.audios = vec![audio(b"old-audio")];
    let old_parts = vec![ChatPart::Image(1), ChatPart::Audio(0), ChatPart::Image(0)];
    captured
        .messages
        .iter_mut()
        .find(|m| m.content == "서울을 찾아 줘.")
        .unwrap()
        .parts = old_parts.clone();
    let history = History::capture(captured, &generated());
    let mut next = anthropic_tail();
    next.images = vec![image(b"new-image-a"), image(b"new-image-b")];
    next.audios = vec![audio(b"new-audio")];
    next.messages[0].parts.extend([
        ChatPart::Text("새 미디어".into()),
        ChatPart::Image(0),
        ChatPart::Audio(0),
        ChatPart::Image(1),
    ]);

    assert!(history.restore(&mut next).unwrap());

    assert_eq!(
        next.images,
        vec![
            image(b"old-image-a"),
            image(b"old-image-b"),
            image(b"new-image-a"),
            image(b"new-image-b")
        ]
    );
    assert_eq!(next.audios, vec![audio(b"old-audio"), audio(b"new-audio")]);
    let old = next
        .messages
        .iter()
        .find(|m| m.content == "서울을 찾아 줘.")
        .unwrap();
    assert_eq!(old.parts, old_parts);
    let new = next
        .messages
        .iter()
        .find(|m| m.tool_call_id == CALL_ID)
        .unwrap();
    assert_eq!(
        new.parts,
        vec![
            ChatPart::ToolResult {
                id: CALL_ID.into(),
                content: "서울: 맑음".into()
            },
            ChatPart::Text("새 미디어".into()),
            ChatPart::Image(2),
            ChatPart::Audio(1),
            ChatPart::Image(3),
        ]
    );
}

#[test]
fn leave_ordinary_request() {
    let history = History::capture(original(Api::Responses), &generated());
    let mut next = parse_responses_request(&env(), r#"{"input":"완전히 새로운 질문"}"#).unwrap();
    let before = format!("{next:?}");

    assert!(next.live_call_ids.is_empty());
    assert!(!history.matches(&next));
    assert!(!history.restore(&mut next).unwrap());
    assert_eq!(format!("{next:?}"), before);
}
