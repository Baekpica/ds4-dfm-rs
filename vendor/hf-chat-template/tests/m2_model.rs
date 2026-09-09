//! M2 — typed input model + `tokenizer_config.json` loading.

use hf_chat_template::{ChatTemplate, Message, RenderInput, TokenizerConfig};

const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

#[test]
fn render_typed_input_chatml() {
    let tmpl = ChatTemplate::from_str(CHATML).unwrap();
    let input = RenderInput {
        messages: vec![
            Message::new("system", "be terse"),
            Message::new("user", "hi"),
        ],
        add_generation_prompt: true,
        ..Default::default()
    };
    let out = tmpl.render(&input).unwrap();
    assert_eq!(
        out,
        "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn render_input_deserializes_from_request_json() {
    // The exact JSON a caller would already have on hand.
    let input: RenderInput = serde_json::from_str(
        r#"{
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": true
        }"#,
    )
    .unwrap();
    let tmpl = ChatTemplate::from_str(CHATML).unwrap();
    let out = tmpl.render(&input).unwrap();
    assert_eq!(
        out,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn role_constructors_match_new() {
    assert_eq!(
        serde_json::to_value(Message::system("s")).unwrap(),
        serde_json::to_value(Message::new("system", "s")).unwrap()
    );
    assert_eq!(
        serde_json::to_value(Message::user("u")).unwrap(),
        serde_json::to_value(Message::new("user", "u")).unwrap()
    );
    assert_eq!(
        serde_json::to_value(Message::assistant("a")).unwrap(),
        serde_json::to_value(Message::new("assistant", "a")).unwrap()
    );

    let tmpl = ChatTemplate::from_str(CHATML).unwrap();
    let out = tmpl
        .render_messages(&[Message::system("be terse"), Message::user("hi")], true)
        .unwrap();
    assert_eq!(
        out,
        "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn render_messages_convenience() {
    let tmpl = ChatTemplate::from_str(CHATML).unwrap();
    let msgs = [Message::new("user", "yo")];
    let out = tmpl.render_messages(&msgs, false).unwrap();
    assert_eq!(out, "<|im_start|>user\nyo<|im_end|>\n");
}

#[test]
fn tool_arguments_preserve_insertion_order_through_typed_path() {
    // The whole reason serde_json `preserve_order` is enabled: a tool serialized via
    // `tools | tojson` must keep author-written key order, not sort alphabetically.
    let tmpl = ChatTemplate::from_str("{{ tools | tojson }}").unwrap();
    let input: RenderInput = serde_json::from_str(
        r#"{
            "messages": [],
            "tools": [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]
        }"#,
    )
    .unwrap();
    let out = tmpl.render(&input).unwrap();
    assert_eq!(
        out,
        r#"[{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]"#
    );
}

#[test]
fn content_string_vs_parts_distinguished() {
    let tmpl =
        ChatTemplate::from_str("{% for m in messages %}{{ m.content is string }}{% endfor %}")
            .unwrap();

    let text: RenderInput =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert_eq!(tmpl.render(&text).unwrap(), "True");

    let parts: RenderInput = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
    )
    .unwrap();
    assert_eq!(tmpl.render(&parts).unwrap(), "False");
}

#[test]
fn from_tokenizer_config_string_form_injects_bos() {
    // bos_token is referenced as a context variable; from_tokenizer_config must inject it.
    let cfg: TokenizerConfig = serde_json::from_str(
        r#"{
            "bos_token": "<s>",
            "eos_token": {"content": "</s>", "lstrip": false},
            "chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{{ eos_token }}{% endfor %}"
        }"#,
    )
    .unwrap();
    let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap();
    let out = tmpl
        .render_messages(&[Message::new("user", "hi")], false)
        .unwrap();
    assert_eq!(out, "<s>hi</s>");
}

#[test]
fn extra_overrides_injected_special_token() {
    let cfg: TokenizerConfig =
        serde_json::from_str(r#"{"bos_token": "<s>", "chat_template": "{{ bos_token }}"}"#)
            .unwrap();
    let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap();
    let mut input = RenderInput::default();
    input.extra.insert(
        "bos_token".into(),
        serde_json::Value::String("[BOS]".into()),
    );
    assert_eq!(tmpl.render(&input).unwrap(), "[BOS]");
}

#[test]
fn named_template_resolution() {
    let cfg: TokenizerConfig = serde_json::from_str(
        r#"{
            "chat_template": [
                {"name": "default", "template": "D:{{ messages[0].content }}"},
                {"name": "tool_use", "template": "T:{{ messages[0].content }}"}
            ]
        }"#,
    )
    .unwrap();
    let msgs = [Message::new("user", "x")];

    // Default resolution picks the "default" entry.
    let def = ChatTemplate::from_tokenizer_config(&cfg).unwrap();
    assert_eq!(def.render_messages(&msgs, false).unwrap(), "D:x");

    // Explicit name selects the other.
    let tool = ChatTemplate::builder_from_config(&cfg)
        .unwrap()
        .template_name("tool_use")
        .build()
        .unwrap();
    assert_eq!(tool.render_messages(&msgs, false).unwrap(), "T:x");
}

#[test]
fn unknown_named_template_is_config_error() {
    let cfg: TokenizerConfig =
        serde_json::from_str(r#"{"chat_template":[{"name":"default","template":"x"}]}"#).unwrap();
    let err = ChatTemplate::builder_from_config(&cfg)
        .unwrap()
        .template_name("nope")
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("nope"), "got: {msg}");
    assert!(
        msg.contains("default"),
        "should list available names: {msg}"
    );
}

#[test]
fn ambiguous_named_templates_without_default_errors() {
    let cfg: TokenizerConfig = serde_json::from_str(
        r#"{"chat_template":[{"name":"a","template":"x"},{"name":"b","template":"y"}]}"#,
    )
    .unwrap();
    let err = ChatTemplate::from_tokenizer_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("ambiguous"), "got: {err}");
}

#[test]
fn missing_chat_template_is_config_error() {
    let cfg: TokenizerConfig = serde_json::from_str(r#"{"bos_token":"<s>"}"#).unwrap();
    let err = ChatTemplate::from_tokenizer_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("no chat_template"), "got: {err}");
}
