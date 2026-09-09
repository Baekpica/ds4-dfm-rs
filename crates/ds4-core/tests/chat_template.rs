use ds4_core::chat_template::{RenderClock, Template};
use serde_json::{json, Value};

#[test]
fn render_official_inkling() {
    let template = Template::compile(
        include_str!("../../../tests/fixtures/inkling/chat_template.jinja"),
        RenderClock::Fixed(0),
    )
    .unwrap();
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/inkling/server-vectors.json"
    ))
    .unwrap();
    for case in fixtures["vectors"].as_array().unwrap() {
        let context = json!({
            "messages": case["messages"],
            "tools": case["tools"],
            "reasoning_effort": case["effort"],
            "add_generation_prompt": true,
        });
        assert_eq!(
            template.render(&context).unwrap(),
            case["rendered"].as_str().unwrap(),
            "{} / {}",
            case["name"],
            case["effort"]
        );
    }
}

#[test]
fn preserve_template_errors() {
    assert!(Template::compile("{% if %}", RenderClock::Fixed(0)).is_err());
    let template = Template::compile(
        "{{ raise_exception('unsupported role') }}",
        RenderClock::Fixed(0),
    )
    .unwrap();
    let error = template.render(&json!({})).unwrap_err().to_string();
    assert!(error.contains("unsupported role"), "{error}");
}

#[test]
fn preserve_bos_and_clock() {
    let template = Template::compile(
        "{{ bos_token }}{{ strftime_now('%Y-%m-%d') }}",
        RenderClock::Fixed(0),
    )
    .unwrap();
    assert_eq!(
        template.render(&json!({"bos_token": "<BOS>"})).unwrap(),
        "<BOS>1970-01-01"
    );
}
