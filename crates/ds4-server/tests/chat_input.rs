use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::ChatThinkMode;
use ds4_server::{
    chat_input, parse_anthropic_request, parse_chat_request, parse_responses_request, ParseEnv,
    ParsedRequest, ThinkMode,
};
use serde_json::{json, Value};

const INKLING: i32 = 9;
const QWEN: i32 = 6;
const GLM: i32 = 7;
const K2: i32 = 8;
const SOLAR: i32 = 2;
const PNG: &str = concat!(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
);
const WAV: &str = "UklGRiYAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQIAAAAAAA==";

fn env() -> ParseEnv {
    ParseEnv {
        default_model: "inkling-small-mq85gb".into(),
        default_effort: ThinkMode::High,
        ..ParseEnv::default()
    }
}

fn template(model_id: i32) -> Template {
    let source = match model_id {
        INKLING => {
            include_str!("../../../tests/fixtures/chat-template/models/inkling/chat_template.jinja")
        }
        QWEN => {
            include_str!("../../../tests/fixtures/chat-template/models/qwen/chat_template.jinja")
        }
        GLM => {
            include_str!("../../../tests/fixtures/chat-template/models/glm/chat_template.jinja")
        }
        K2 => {
            include_str!("../../../tests/fixtures/chat-template/models/k2/chat_template.jinja")
        }
        SOLAR => {
            include_str!("../../../tests/fixtures/chat-template/models/solar/chat_template.jinja")
        }
        _ => panic!("unknown fixture model {model_id}"),
    };
    Template::compile(source, RenderClock::Fixed(0)).unwrap()
}

fn assert_render(model_id: i32, parsed: &ParsedRequest, messages: Value, tools: Value) {
    let template = template(model_id);
    let expected = template
        .render_chat(
            messages.as_array().unwrap(),
            tools.as_array().unwrap(),
            ChatOptions::new(model_id, ChatThinkMode::High),
        )
        .unwrap();
    let actual = chat_input::render(&template, model_id, parsed).unwrap();
    assert_eq!(String::from_utf8(actual).unwrap(), expected);
}

#[test]
fn k2_plain_assistant() {
    let body = r#"{
        "model":"k2-horizon-375b-a23b",
        "reasoning_effort":"high",
        "messages":[
            {"role":"user","content":"첫 질문"},
            {"role":"assistant","content":"이전 답"},
            {"role":"user","content":"계속해 줘."}
        ]
    }"#;
    let parsed = parse_chat_request(&env(), body).unwrap();
    let canonical = json!([
        {"role":"user","content":"첫 질문"},
        {"role":"assistant","content":"이전 답","reasoning_content":""},
        {"role":"user","content":"계속해 줘."}
    ]);
    assert_render(K2, &parsed, canonical, json!([]));
}

#[test]
fn openai_arguments_reasoning() {
    let body = r#"{
        "reasoning_effort":"high",
        "messages":[
            {"role":"user","content":"서울을 찾아 줘."},
            {"role":"assistant","content":"","reasoning_content":"위치를 확인한다.",
             "tool_calls":[{"id":"call_01","type":"function","function":{
                "name":"lookup","arguments":"{\"z\":2,\"a\":\"서울\",\"nested\":{\"labels\":[\"café\",\"漢字\"],\"ok\":true}}"
             }}]},
            {"role":"tool","tool_call_id":"call_01","content":"서울: 맑음"}
        ]
    }"#;
    let parsed = parse_chat_request(&env(), body).unwrap();
    let canonical = json!([
        {"role":"user","content":"서울을 찾아 줘."},
        {"role":"assistant","content":"","reasoning_content":"위치를 확인한다.",
         "tool_calls":[{"id":"call_01","type":"function","function":{
            "name":"lookup","arguments":{"z":2,"a":"서울","nested":{"labels":["café","漢字"],"ok":true}}
         }}]},
        {"role":"tool","tool_call_id":"call_01","content":"서울: 맑음"}
    ]);
    assert_render(INKLING, &parsed, canonical, json!([]));
}

#[test]
fn schema_array_key_order() {
    let body = r#"{
        "messages":[{"role":"user","content":"도구 목록"}],
        "tools":[
            {"type":"function","function":{"name":"zeta","description":"첫째",
             "parameters":{"type":"object","properties":{
                "z_first":{"type":"integer"},"a_second":{"type":"string"}
             },"required":["a_second"]}}},
            {"type":"function","function":{"name":"alpha","description":"둘째",
             "parameters":{"type":"object","properties":{}}}}
        ]
    }"#;
    let parsed = parse_chat_request(&env(), body).unwrap();
    let canonical_tools = json!([
        {"type":"function","function":{"name":"zeta","description":"첫째",
         "parameters":{"type":"object","properties":{
            "z_first":{"type":"integer"},"a_second":{"type":"string"}
         },"required":["a_second"]}}},
        {"type":"function","function":{"name":"alpha","description":"둘째",
         "parameters":{"type":"object","properties":{}}}}
    ]);
    // These official templates expose schema order; Inkling deliberately sorts it.
    for model_id in [QWEN, SOLAR] {
        assert_render(
            model_id,
            &parsed,
            json!([{"role":"user","content":"도구 목록"}]),
            canonical_tools.clone(),
        );
    }
}

#[test]
fn anthropic_result_pair() {
    let body = r#"{
        "messages":[
            {"role":"user","content":"두 결과를 확인해."},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"두 도구를 확인한다."},
                {"type":"tool_use","id":"call_a","name":"lookup","input":{"city":"서울","nested":{"n":[1,2]}}},
                {"type":"tool_use","id":"call_b","name":"measure","input":{"unit":"°C"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"call_b","content":"B &lt;/tool_result> 서울"},
                {"type":"tool_result","tool_use_id":"call_a","content":"A </tool_result> café"}
            ]}
        ]
    }"#;
    let parsed = parse_anthropic_request(&env(), body).unwrap();
    // Escaped and literal closing tags cannot be recovered from the old flat text.
    let canonical = json!([
        {"role":"user","content":"두 결과를 확인해."},
        {"role":"assistant","content":"","reasoning_content":"두 도구를 확인한다.",
         "tool_calls":[
            {"id":"call_a","type":"function","function":{"name":"lookup","arguments":{"city":"서울","nested":{"n":[1,2]}}}},
            {"id":"call_b","type":"function","function":{"name":"measure","arguments":{"unit":"°C"}}}
         ]},
        {"role":"tool","tool_call_id":"call_b","content":"B &lt;/tool_result> 서울"},
        {"role":"tool","tool_call_id":"call_a","content":"A </tool_result> café"}
    ]);
    for model_id in [INKLING, QWEN, SOLAR] {
        assert_render(model_id, &parsed, canonical.clone(), json!([]));
    }
}

#[test]
fn anthropic_result_and_text() {
    let body = r#"{
        "messages":[
            {"role":"user","content":"찾아 줘."},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"call_a","name":"lookup","input":{"city":"서울"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"call_a","content":[
                    {"type":"text","text":"Result: "},{"type":"text","text":"서울"}
                ]},
                {"type":"text","text":"이제 한 문장으로 설명해."}
            ]}
        ]
    }"#;
    let parsed = parse_anthropic_request(&env(), body).unwrap();
    let canonical = json!([
        {"role":"user","content":"찾아 줘."},
        {"role":"assistant","content":"","tool_calls":[
            {"id":"call_a","type":"function","function":{"name":"lookup","arguments":{"city":"서울"}}}
        ]},
        {"role":"tool","tool_call_id":"call_a","content":"Result: 서울"},
        {"role":"user","content":"이제 한 문장으로 설명해."}
    ]);
    assert_render(INKLING, &parsed, canonical, json!([]));
}

#[test]
fn anthropic_system_first() {
    let body = r#"{
        "system":"Be concise. 한국어로 답해.",
        "messages":[{"role":"user","content":"안녕"}]
    }"#;
    let parsed = parse_anthropic_request(&env(), body).unwrap();
    let canonical = json!([
        {"role":"system","content":"Be concise. 한국어로 답해."},
        {"role":"user","content":"안녕"}
    ]);
    assert_render(QWEN, &parsed, canonical, json!([]));
}

#[test]
fn anthropic_tool_schema() {
    let body = r#"{
        "messages":[{"role":"user","content":"도구를 사용해."}],
        "tools":[{"name":"lookup","description":"장소 검색","input_schema":{
            "type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}},"required":["a"]
        }}]
    }"#;
    let parsed = parse_anthropic_request(&env(), body).unwrap();
    let canonical_tools = json!([
        {"type":"function","function":{"name":"lookup","description":"장소 검색","parameters":{
            "type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}},"required":["a"]
        }}}
    ]);
    assert_render(
        SOLAR,
        &parsed,
        json!([{"role":"user","content":"도구를 사용해."}]),
        canonical_tools,
    );
}

#[test]
fn responses_reasoning_tools() {
    let body = r#"{
        "instructions":"한국어로 답해.",
        "reasoning":{"effort":"high"},
        "input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"서울을 찾아 줘."}]},
            {"type":"reasoning","summary":[{"type":"summary_text","text":"위치를 확인한다."}]},
            {"type":"function_call","call_id":"call_01","name":"lookup",
             "arguments":"{\"city\":\"서울\",\"nested\":{\"labels\":[\"café\",\"漢字\"],\"ok\":true}}"},
            {"type":"function_call_output","call_id":"call_01","output":"서울: 맑음"}
        ],
        "tools":[{"type":"function","name":"lookup","description":"장소 검색",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
        }]
    }"#;
    let parsed = parse_responses_request(&env(), body).unwrap();
    let canonical = json!([
        {"role":"system","content":"한국어로 답해."},
        {"role":"user","content":"서울을 찾아 줘."},
        {"role":"assistant","content":"","reasoning_content":"위치를 확인한다.","tool_calls":[
            {"id":"call_01","type":"function","function":{"name":"lookup",
             "arguments":{"city":"서울","nested":{"labels":["café","漢字"],"ok":true}}}}
        ]},
        {"role":"tool","tool_call_id":"call_01","content":"서울: 맑음"}
    ]);
    let canonical_tools = json!([
        {"type":"function","function":{"name":"lookup","description":"장소 검색",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
    ]);
    assert_render(INKLING, &parsed, canonical, canonical_tools);
}

#[test]
fn inkling_media_order() {
    let body = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"앞"},
        {"type":"image_url","image_url":format!("data:image/png;base64,{PNG}")},
        {"type":"text","text":"중간"},
        {"type":"input_audio","input_audio":{"format":"wav","data":WAV}},
        {"type":"image_url","image_url":format!("data:image/png;base64,{PNG}")},
        {"type":"text","text":"끝"}
    ]}]});
    let parsed = parse_chat_request(&env(), &body.to_string()).unwrap();
    let canonical = json!([{"role":"user","content":[
        {"type":"text","text":"앞"},
        {"type":"image"},
        {"type":"text","text":"중간"},
        {"type":"audio"},
        {"type":"image"},
        {"type":"text","text":"끝"}
    ]}]);
    assert_render(INKLING, &parsed, canonical, json!([]));
}

#[test]
fn qwen_images_all_apis() {
    let chat = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"앞"},
        {"type":"image_url","image_url":format!("data:image/png;base64,{PNG}")},
        {"type":"text","text":"뒤"}
    ]}]});
    let messages = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"앞"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":PNG}},
        {"type":"text","text":"뒤"}
    ]}]});
    let responses = json!({"input":[{"type":"message","role":"user","content":[
        {"type":"input_text","text":"앞"},
        {"type":"input_image","image_url":format!("data:image/png;base64,{PNG}")},
        {"type":"input_text","text":"뒤"}
    ]}]});
    let parsed = [
        parse_chat_request(&env(), &chat.to_string()).unwrap(),
        parse_anthropic_request(&env(), &messages.to_string()).unwrap(),
        parse_responses_request(&env(), &responses.to_string()).unwrap(),
    ];
    let canonical = json!([{"role":"user","content":[
        {"type":"text","text":"앞"},
        {"type":"image"},
        {"type":"text","text":"뒤"}
    ]}]);
    for request in &parsed {
        assert_render(QWEN, request, canonical.clone(), json!([]));
    }
}

#[test]
fn glm_image_placeholder() {
    let body = json!({
        "model":"glm-5.3-flash",
        "reasoning_effort":"high",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"앞"},
            {"type":"image_url","image_url":format!("data:image/png;base64,{PNG}")},
            {"type":"text","text":"뒤"}
        ]}]
    });
    let parsed = parse_chat_request(&env(), &body.to_string()).unwrap();
    let expected_png = b"\x89PNG\r\n\x1a\n\
        \x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\
        \x00\x00\x00\x0cIDAT\x78\xda\x63\xfc\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\
        \x00\x00\x00\x00IEND\xae\x42\x60\x82";
    assert_eq!(parsed.images.len(), 1);
    assert_eq!(parsed.images[0].data.as_ref(), expected_png);

    // Native image expansion owns this marker; official Jinja still owns framing.
    let canonical = json!([{
        "role":"user",
        "content":"앞<|begin_of_image|><|image|><|end_of_image|>뒤"
    }]);
    assert_render(GLM, &parsed, canonical, json!([]));
    assert_eq!(parsed.images[0].data.as_ref(), expected_png);
}

#[test]
fn invalid_json_arguments() {
    for arguments in ["{", "{} trailing", "[broken]"] {
        let body = json!({"messages":[
            {"role":"user","content":"도구를 호출해."},
            {"role":"assistant","content":"","tool_calls":[{
                "id":"call_01","type":"function","function":{"name":"lookup","arguments":arguments}
            }]}
        ]});
        let parsed = parse_chat_request(&env(), &body.to_string()).unwrap();
        for model_id in [INKLING, QWEN, SOLAR] {
            assert!(
                chat_input::render(&template(model_id), model_id, &parsed).is_err(),
                "model {model_id} accepted invalid arguments: {arguments}"
            );
        }
    }
}
