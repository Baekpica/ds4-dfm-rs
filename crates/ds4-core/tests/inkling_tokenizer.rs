//! Source tokenizers vectors; the real tokenizer is embedded in the MTP GGUF too.
use ds4_core::{ChatThinkMode, GgufFile, ModelFamily, TokenBuffer, Vocab};
use std::path::PathBuf;

#[test]
#[ignore = "requires Inkling tokenizer metadata in the MTP-BF16 artifact"]
fn source_tokenizer_vectors() {
    let root =
        PathBuf::from(std::env::var("INKLING_ARTIFACT_DIR").expect("set INKLING_ARTIFACT_DIR"));
    let g = GgufFile::open(&root.join("MTP-BF16/Inkling-Small-MTP-BF16.gguf")).unwrap();
    let vocab = Vocab::load(&g, ModelFamily::Inkling).unwrap();
    assert_eq!(vocab.n_vocab(), 200058);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/inkling/tokenizer-vectors.json"
    ))
    .unwrap();
    for row in reference["vectors"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        for (key, actual) in [
            ("text_ids", vocab.encode_text(text)),
            ("chat_ids", vocab.encode_rendered_chat(text)),
        ] {
            let expected: Vec<i32> = row[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(actual, expected, "{key}: {text:?}");
            let bytes: Vec<_> = actual.iter().flat_map(|&id| vocab.token_text(id)).collect();
            assert_eq!(bytes, text.as_bytes(), "decode: {text:?}");
        }
    }
    assert_eq!(vocab.engine_eos(), 200006);
    assert!(vocab.is_stop(200006));
    // Message and thinking boundaries are not generation ends.
    for id in [
        199999, 200008, 200010, 200020, 200043, 200049, 200053, 200054,
    ] {
        assert!(!vocab.is_stop(id), "unexpected stop {id}");
    }
    let raw = [b'a', 0xff, 0xf0, 0x9f, b'b'];
    let ids = vocab.encode_bytes(&raw);
    assert_eq!(
        ids.iter()
            .flat_map(|&id| vocab.token_text(id))
            .collect::<Vec<_>>(),
        raw
    );
}

#[test]
#[ignore = "requires Inkling tokenizer metadata in the MTP-BF16 artifact"]
fn source_chat_controls() {
    let root =
        PathBuf::from(std::env::var("INKLING_ARTIFACT_DIR").expect("set INKLING_ARTIFACT_DIR"));
    let g = GgufFile::open(&root.join("MTP-BF16/Inkling-Small-MTP-BF16.gguf")).unwrap();
    let vocab = Vocab::load(&g, ModelFamily::Inkling).unwrap();
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/inkling/chat-vectors.json"
    ))
    .unwrap();
    for (mode, effort) in [
        (ChatThinkMode::None, "none"),
        (ChatThinkMode::Low, "low"),
        (ChatThinkMode::High, "high"),
        (ChatThinkMode::Max, "max"),
    ] {
        let mut tokens = TokenBuffer::new();
        vocab.chat_begin(&mut tokens).unwrap();
        vocab
            .chat_append_message(&mut tokens, "system", b"Be concise.")
            .unwrap();
        vocab.chat_append_effort_prefix(&mut tokens, mode);
        vocab
            .chat_append_message(&mut tokens, "user", "안녕하세요".as_bytes())
            .unwrap();
        vocab
            .chat_append_message(&mut tokens, "assistant", b"Hello.")
            .unwrap();
        vocab
            .chat_append_message(&mut tokens, "tool", b"result")
            .unwrap();
        vocab
            .chat_append_message(&mut tokens, "user", b"Continue.")
            .unwrap();
        vocab
            .chat_append_assistant_prefix(&mut tokens, mode)
            .unwrap();
        let expected = reference["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["effort"] == effort)
            .unwrap()["rendered"]
            .as_str()
            .unwrap();
        assert_eq!(
            tokens.as_slice(),
            vocab.encode_rendered_chat(expected),
            "{mode:?}"
        );
    }
}
