use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::ChatThinkMode;
use serde_json::json;

const K2: i32 = 8;

#[test]
fn glm_embedded_history() {
    let messages = vec![
        json!({"role":"user","content":"Compute 2 + 2."}),
        json!({"role":"assistant","content":"<think>Two pairs make four.</think>4"}),
        json!({"role":"user","content":"Now add 1."}),
    ];
    for source in [
        include_str!("../../../tests/fixtures/chat-template/models/glm/chat_template.jinja"),
        include_str!(
            "../../../tests/fixtures/chat-template/models/glm-uncensored/chat_template.jinja"
        ),
    ] {
        let template = Template::compile(source, RenderClock::Fixed(0)).unwrap();
        let expected = template
            .render(&json!({
                "messages":messages,"tools":[],"add_generation_prompt":true,
                "enable_thinking":true,"reasoning_effort":"low",
            }))
            .unwrap();
        let actual = template
            .render_chat(
                &messages,
                &[],
                ChatOptions::new(ds4_core::Variant::Glm53Flash as i32, ChatThinkMode::Low),
            )
            .unwrap();
        assert_eq!(actual, expected);
        assert!(!actual.contains("<think></think><think>"));
    }
}

#[test]
fn glm_low_effort() {
    let template = Template::compile(
        include_str!("../../../tests/fixtures/chat-template/models/glm/chat_template.jinja"),
        RenderClock::Fixed(0),
    )
    .unwrap();
    let messages = vec![json!({"role":"user","content":"Hello"})];
    let actual = template
        .render_chat(
            &messages,
            &[],
            ChatOptions::new(ds4_core::Variant::Glm53Flash as i32, ChatThinkMode::Low),
        )
        .unwrap();
    assert!(actual.starts_with("[gMASK]<sop><|system|>Reasoning Effort: Low"));
}

#[test]
fn k2_plain_assistant() {
    let template = Template::compile(
        include_str!("../../../tests/fixtures/chat-template/models/k2/chat_template.jinja"),
        RenderClock::Fixed(0),
    )
    .unwrap();
    let messages = vec![
        json!({"role":"user","content":"첫 질문"}),
        json!({"role":"assistant","content":"이전 답"}),
        json!({"role":"user","content":"계속해 줘."}),
    ];
    let canonical = vec![
        json!({"role":"user","content":"첫 질문"}),
        json!({"role":"assistant","content":"이전 답","reasoning_content":""}),
        json!({"role":"user","content":"계속해 줘."}),
    ];
    let options = ChatOptions::new(K2, ChatThinkMode::High);
    let expected = template.render_chat(&canonical, &[], options).unwrap();
    let actual = template.render_chat(&messages, &[], options).unwrap();
    assert_eq!(actual, expected);
}
