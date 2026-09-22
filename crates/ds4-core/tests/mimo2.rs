//! Original-tokenizer goldens against the mixed MiMo GGUF.
use ds4_core::{chat_template::Template, identify_gguf, Mimo2Plan, ModelFamily, Vocab};
use std::path::Path;

const DEFAULT_GGUF: &str = "/home/sunghoon/workspace/ds4-exaone/models/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF/MQ-IQ2-XXS-XS-Q8-MM-BF16/MiMo-V2.6-Flash-RL-MQ-IQ2-XXS-XS-Q8-00001-of-00004.gguf";

#[test]
fn original_tokenizer_goldens() {
    // CI has no weights. A set MIMO2_GGUF is required to exist; the local
    // shard is used only when that file is actually on this machine.
    let path = match std::env::var("MIMO2_GGUF") {
        Ok(path) => path,
        Err(_) if Path::new(DEFAULT_GGUF).is_file() => DEFAULT_GGUF.to_string(),
        Err(_) => return,
    };
    let path = Path::new(&path);
    let id = identify_gguf(path).unwrap();
    assert_eq!(id.shape.family, ModelFamily::Mimo2);
    Mimo2Plan::check_metadata(&ds4_core::GgufFile::open(path).unwrap()).unwrap();
    let vocab = Vocab::load_path(path, id.shape.family).unwrap();
    let template = Template::from_model(path).unwrap().unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/mimo2-tokenizer.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let prompt = template.render(&case["context"]).unwrap();
        assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{}", case["id"]);
        let ids = vocab.encode_rendered_chat(&prompt);
        assert_eq!(serde_json::json!(ids), case["token_ids"], "{}", case["id"]);
    }
    for stop in [151643, 151645, 151672] {
        assert!(vocab.is_stop(stop));
    }
    for control in [151644, 151655, 151669, 151674] {
        assert!(!vocab.is_stop(control));
    }
    assert!(vocab
        .chat_begin(&mut ds4_core::TokenBuffer::default())
        .is_err());
}
