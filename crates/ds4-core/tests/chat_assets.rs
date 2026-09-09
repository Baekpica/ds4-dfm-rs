use ds4_core::chat_template::{ChatOptions, Template};
use ds4_core::ChatThinkMode;
use serde_json::json;
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new(name: &str, source: Option<&str>) -> Self {
        let path = std::env::temp_dir().join(format!("ds4-jinja-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(u64::from(source.is_some()).to_le_bytes());
        if let Some(source) = source {
            let key = "tokenizer.chat_template";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(8u32.to_le_bytes());
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source.as_bytes());
        }
        bytes.resize((bytes.len() + 31) / 32 * 32, 0);
        std::fs::write(path.join("model.gguf"), bytes).unwrap();
        Self(path)
    }
    fn load(&self) -> ds4_core::Result<Option<Template>> {
        Template::from_model(&self.0.join("model.gguf"))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn sidecar_overrides_embedded() {
    let fixture = Fixture::new("sidecar", Some("embedded"));
    std::fs::write(
        fixture.0.join("chat_template.jinja"),
        "{{ bos_token }}sidecar",
    )
    .unwrap();
    std::fs::write(
        fixture.0.join("tokenizer_config.json"),
        r#"{"bos_token":{"content":"<BOS>"},"chat_template":"config"}"#,
    )
    .unwrap();
    let template = fixture.load().unwrap().unwrap();
    assert_eq!(template.render(&json!({})).unwrap(), "<BOS>sidecar");
    assert!(template.source().ends_with("chat_template.jinja"));
}

#[test]
fn config_then_embedded() {
    let fixture = Fixture::new("config", Some("embedded"));
    assert_eq!(
        fixture.load().unwrap().unwrap().render(&json!({})).unwrap(),
        "embedded"
    );
    std::fs::write(
        fixture.0.join("tokenizer_config.json"),
        r#"{"chat_template":"config"}"#,
    )
    .unwrap();
    assert_eq!(
        fixture.load().unwrap().unwrap().render(&json!({})).unwrap(),
        "config"
    );
}

#[test]
fn broken_sidecar_is_an_error() {
    let fixture = Fixture::new("invalid", Some("embedded"));
    std::fs::write(fixture.0.join("chat_template.jinja"), "{% if %}").unwrap();
    assert!(fixture.load().is_err());
}

#[test]
fn absent_template_is_explicit() {
    let fixture = Fixture::new("missing", None);
    assert!(fixture.load().unwrap().is_none());
}

#[test]
fn shared_option_mapping() {
    let template = Template::compile(
        "{{ reasoning_effort }}|{{ enable_thinking }}",
        ds4_core::chat_template::RenderClock::Fixed(0),
    )
    .unwrap();
    for (id, mode, expected) in [
        (6, ChatThinkMode::High, "xhigh|True"),
        (2, ChatThinkMode::Low, "medium|True"),
        (9, ChatThinkMode::None, "none|False"),
    ] {
        assert_eq!(
            template
                .render_chat(&[], &[], ChatOptions::new(id, mode))
                .unwrap(),
            expected
        );
    }
}
