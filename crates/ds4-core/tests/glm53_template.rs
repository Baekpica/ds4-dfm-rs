use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::{ChatThinkMode, Variant};
use serde_json::Value;

#[test]
fn artifact_grammar() {
    let template = Template::compile(
        include_str!(
            "../../../tests/fixtures/chat-template/models/glm-uncensored/chat_template.jinja"
        ),
        RenderClock::Fixed(0),
    )
    .unwrap();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/chat-template/glm53-vectors.json"
    ))
    .unwrap();
    for row in fixture["vectors"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let context = &row["context"];
        assert_eq!(
            template.render(context).unwrap(),
            row["expected"].as_str().unwrap(),
            "{name}"
        );
        let Some(mode) = row["adapter"].as_str() else {
            continue;
        };
        let mode = match mode {
            "low" => ChatThinkMode::Low,
            "high" => ChatThinkMode::High,
            "max" => ChatThinkMode::Max,
            "none" => ChatThinkMode::None,
            _ => panic!("invalid fixture mode"),
        };
        assert_eq!(
            template
                .render_chat(
                    context["messages"].as_array().unwrap(),
                    context["tools"].as_array().unwrap(),
                    ChatOptions::new(Variant::Glm53Flash as i32, mode),
                )
                .unwrap(),
            row["adapter_expected"].as_str().unwrap(),
            "adapter {name}"
        );
    }
}
