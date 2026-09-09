use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::ChatThinkMode;
use serde_json::json;

const K2: i32 = 8;

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
