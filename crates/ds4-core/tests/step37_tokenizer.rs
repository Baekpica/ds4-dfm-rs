//! Exact source-tokenizer vectors with the downloaded MQ83 tokenizer tables.
use ds4_core::{GgufFile, ModelFamily, Vocab};
use std::path::Path;

#[test]
#[ignore = "requires STEP37_MODEL_PATH pointing to the first MQ83 shard"]
fn source_tokenizer_vectors() {
    let path = std::env::var("STEP37_MODEL_PATH").expect("set STEP37_MODEL_PATH");
    let g = GgufFile::open(Path::new(&path)).unwrap();
    let v = Vocab::load(&g, ModelFamily::Step37).unwrap();
    let vectors: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/step37/tokenizer-vectors.json"
    ))
    .unwrap();
    for row in vectors["vectors"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let expected: Vec<i32> = serde_json::from_value(row["ids"].clone()).unwrap();
        let got = v.encode_rendered_chat(text);
        if got != expected {
            let at = got
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(expected.len()));
            panic!(
                "{} mismatch at {at}: got {:?}, expected {:?}",
                row["id"],
                &got[at..got.len().min(at + 8)],
                &expected[at..expected.len().min(at + 8)]
            );
        }
        let decoded: Vec<_> = got.iter().flat_map(|&id| v.token_text(id)).collect();
        assert_eq!(decoded, text.as_bytes());
    }
    assert_eq!(v.engine_eos(), 128007);
    assert!(v.is_stop(128007));
    for id in [
        0, 1, 128000, 128001, 128002, 128005, 128006, 128008, 128009, 128798, 128799,
    ] {
        assert!(!v.is_stop(id), "unexpected stop {id}");
    }
}

#[test]
#[ignore = "requires STEP37_MODEL_PATH pointing to the first MQ83 shard"]
fn history_token_frontier() {
    use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
    use serde_json::json;

    let path = std::env::var("STEP37_MODEL_PATH").expect("set STEP37_MODEL_PATH");
    let gguf = GgufFile::open(Path::new(&path)).unwrap();
    let vocab = Vocab::load(&gguf, ModelFamily::Step37).unwrap();
    let template = Template::compile(
        include_str!("../../../tests/fixtures/step37/chat_template.jinja"),
        RenderClock::Fixed(0),
    )
    .unwrap();
    let options = ChatOptions::new(10, ds4_core::ChatThinkMode::None);
    for content in [
        "4",
        "Hello",
        "안녕하세요",
        "\nA code example:",
        "{\"ok\":true}",
    ] {
        let first = template
            .render_chat(
                &[json!({"role":"user", "content":"Reply briefly."})],
                &[],
                options,
            )
            .unwrap();
        let history = first
            .strip_suffix("assistant\n<think>\n</think>\n")
            .unwrap();
        let follow = template
            .render_chat(
                &[
                    json!({"role":"user", "content":"Reply briefly."}),
                    json!({"role":"assistant", "content":content}),
                    json!({"role":"user", "content":"Continue."}),
                ],
                &[],
                options,
            )
            .unwrap();
        let prefix = vocab.encode_rendered_chat(history);
        assert!(vocab.encode_rendered_chat(&first).starts_with(&prefix));
        assert!(
            vocab.encode_rendered_chat(&follow).starts_with(&prefix),
            "{content:?}"
        );
        assert_eq!(
            prefix
                .iter()
                .flat_map(|&id| vocab.token_text(id))
                .collect::<Vec<_>>(),
            history.as_bytes()
        );
    }
}
