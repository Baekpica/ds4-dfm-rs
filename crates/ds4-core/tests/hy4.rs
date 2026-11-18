//! Python Jinja2 and Hugging Face tokenizers vectors from the pinned release.
use ds4_core::chat_template::{ChatOptions, Template};
use ds4_core::{ChatThinkMode, ModelFamily, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

#[test]
#[ignore = "set HY4_MODEL_PATH to the MQ release directory"]
fn official_chat_bytes_and_ids() {
    let path = PathBuf::from(std::env::var_os("HY4_MODEL_PATH").expect("HY4_MODEL_PATH"));
    let template = Template::from_model(&path).unwrap().unwrap();
    let vocab = Vocab::load_path(&path, ModelFamily::Hy4).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/hy4/chat-input.json")).unwrap();
    for row in fixture["chat"].as_array().unwrap() {
        let mode = match row["effort"].as_str().unwrap() {
            "no_think" => ChatThinkMode::None,
            "high" => ChatThinkMode::High,
            "max" => ChatThinkMode::Max,
            _ => unreachable!(),
        };
        let options = ChatOptions::new(Variant::Hy4FlashMxq as i32, mode);
        let text = template
            .render_chat(
                row["messages"].as_array().unwrap(),
                row["tools"].as_array().unwrap(),
                options,
            )
            .unwrap();
        assert_eq!(text, row["text"].as_str().unwrap());
    }
    for row in fixture["chat"]
        .as_array()
        .unwrap()
        .iter()
        .chain(fixture["text"].as_array().unwrap())
    {
        let text = row["text"].as_str().unwrap();
        let ids: Vec<i32> = serde_json::from_value(row["ids"].clone()).unwrap();
        assert_eq!(vocab.encode_rendered_chat(text), ids);
        let bytes: Vec<_> = ids.iter().flat_map(|&id| vocab.token_text(id)).collect();
        assert_eq!(bytes, text.as_bytes());
    }
    assert_eq!(vocab.engine_eos(), 120025);
}

#[test]
#[ignore = "set HY4_MODEL_PATH to the complete MQ release"]
fn shipping_artifact_passes_serving_preflight() {
    let path = std::env::var("HY4_MODEL_PATH").expect("HY4_MODEL_PATH");
    ds4_core::probe_model_artifact(&path, ds4_core::ProbeParts::Trunk).unwrap();
}
