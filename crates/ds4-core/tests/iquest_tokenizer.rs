use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::{ChatThinkMode, GgufFile, ModelFamily, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

fn source_cases() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/iquest-q1/tokenizer-vectors.json"
    ))
    .unwrap()
}

fn source_template() -> Template {
    Template::compile(
        include_str!("../../../tests/fixtures/chat-template/models/iquest/chat_template.jinja"),
        RenderClock::Fixed(0),
    )
    .unwrap()
}

#[test]
fn source_jinja_vectors() {
    let cases = source_cases();
    assert_eq!(
        cases["source_template"].as_str().unwrap(),
        include_str!("../../../tests/fixtures/chat-template/models/iquest/chat_template.jinja")
    );
    let template = source_template();
    for case in cases["chats"].as_array().unwrap() {
        assert_eq!(
            template.render(&case["context"]).unwrap(),
            case["rendered"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn source_thinking_controls() {
    let template = source_template();
    let messages = [serde_json::json!({"role":"user", "content":"Hello"})];
    for (mode, suffix) in [
        (ChatThinkMode::None, "<think></think>"),
        (ChatThinkMode::Low, "<think>"),
        (ChatThinkMode::High, "<think>"),
        (ChatThinkMode::Max, "<think>"),
    ] {
        let rendered = template
            .render_chat(
                &messages,
                &[],
                ChatOptions::new(Variant::IQuestQ1 as i32, mode),
            )
            .unwrap();
        assert_eq!(
            rendered,
            format!("<|iquest_user|>Hello<|iquest_end|><|iquest_assistant|>{suffix}")
        );
    }
}

#[test]
#[ignore = "requires DS4_IQUEST_TOKENIZER_FIXTURE: metadata-only GGUF or first model shard"]
fn source_tokenizer_vectors() {
    let path =
        std::env::var("DS4_IQUEST_TOKENIZER_FIXTURE").expect("set DS4_IQUEST_TOKENIZER_FIXTURE");
    let file = GgufFile::open(&PathBuf::from(path)).unwrap();
    let vocab = Vocab::load(&file, ModelFamily::IQuestQ1).unwrap();
    let cases = source_cases();
    for case in cases["vectors"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let actual = match case["mode"].as_str().unwrap() {
            "text" => vocab.encode_text(text),
            "rendered" => vocab.encode_rendered_chat(text),
            mode => panic!("unknown oracle mode {mode}"),
        };
        let expected: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(actual, expected, "{}", case["name"]);
    }
    let template = source_template();
    for case in cases["chats"].as_array().unwrap() {
        let rendered = template.render(&case["context"]).unwrap();
        assert_eq!(
            rendered,
            case["rendered"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        let expected: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            vocab.encode_rendered_chat(&rendered),
            expected,
            "{}",
            case["name"]
        );
    }
    assert!(vocab.is_stop(0));
    for token in [1, 2, 24, 25, 26, 27, 28, 29, 30, 37, 38, 39, 511] {
        assert!(!vocab.is_stop(token), "unexpected stop token {token}");
    }
    assert_eq!(vocab.encode_text("café"), vocab.encode_text("cafe\u{301}"));
}
