//! Spark-X2.6 input adapter: tokenizer and chat template parity against the
//! reference pipeline that produced tests/fixtures/spark2_5/eval-cases.json.
//!
//! The template gate is model-free: it compiles the committed copy of the
//! checkpoint's chat_template.jinja. The tokenizer gate needs the release's
//! trunk metadata (no GPU, no weights): point SPARK2_5_RELEASE at the release
//! directory, the same way the other families gate on their artifact.
use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::{ChatThinkMode, ModelFamily, Vocab};
use serde_json::Value;
use std::path::PathBuf;

const TEMPLATE: &str = include_str!("../../../tests/fixtures/spark2_5/chat_template.jinja");

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("SPARK2_5_RELEASE").expect("SPARK2_5_RELEASE"))
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/spark2_5/eval-cases.json"
    ))
    .unwrap()
}

fn ids(case: &Value) -> Vec<i32> {
    case["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn chat_render_parity() {
    let template = Template::compile(TEMPLATE, RenderClock::System).unwrap();
    for case in fixture()["cases"].as_array().unwrap() {
        if case["context"].is_null() {
            continue;
        }
        let text = template.render(&case["context"]).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{}", case["id"]);
    }
}

#[test]
fn template_refuses_the_reference_errors() {
    let template = Template::compile(TEMPLATE, RenderClock::System).unwrap();
    for case in fixture()["errors"].as_array().unwrap() {
        let error = template.render(&case["context"]).unwrap_err();
        assert!(
            error.message.contains(case["error"].as_str().unwrap()),
            "{}: {}",
            case["id"],
            error.message
        );
    }
}

/// Spark accepts only `high` and `low`; the request mode must never reach the
/// template as `max`/`none`, which raise or mean "no thinking" respectively.
#[test]
fn request_modes_render_without_a_template_error() {
    let template = Template::compile(TEMPLATE, RenderClock::System).unwrap();
    let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
    for mode in [
        ChatThinkMode::None,
        ChatThinkMode::Low,
        ChatThinkMode::High,
        ChatThinkMode::Max,
    ] {
        let options = ChatOptions::new(ds4_core::Variant::SparkX26_4bVq as i32, mode);
        let text = template
            .render_chat(&messages, &[], options)
            .unwrap_or_else(|error| panic!("{mode:?}: {}", error.message));
        assert!(
            text.ends_with("<|Bot|><think>") || text.ends_with("<|Bot|></think>"),
            "{text}"
        );
        if mode == ChatThinkMode::None {
            assert!(text.ends_with("</think>"), "{text}");
        } else {
            assert!(text.ends_with("<think>"), "{text}");
        }
    }
}

#[test]
#[ignore = "requires the release in SPARK2_5_RELEASE; reads tokenizer metadata only"]
fn tokenizer_parity() {
    let vocab = Vocab::load_path(&root(), ModelFamily::Spark2_5Vq).unwrap();
    assert_eq!((vocab.bos_id, vocab.eos_id), (0, 1));
    assert_eq!(vocab.n_vocab(), 131_072);
    assert!(vocab.is_stop(1));
    assert!(!vocab.is_stop(7));
    assert_eq!(
        (
            vocab.system_id,
            vocab.user_id,
            vocab.assistant_id,
            vocab.tool_id
        ),
        (130_972, 130_973, 130_976, 130_974)
    );
    assert_eq!((vocab.think_start_id, vocab.think_end_id), (3, 4));
    for case in fixture()["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(vocab.encode_text(text), ids(case), "encode {}", case["id"]);
        assert_eq!(
            vocab.encode_rendered_chat(text),
            ids(case),
            "render {}",
            case["id"]
        );
    }
}

/// The artifact's trunk carries the official template verbatim, and the
/// release path renders the same bytes as the committed fixture copy.
#[test]
#[ignore = "requires the release in SPARK2_5_RELEASE; reads metadata only"]
fn artifact_carries_the_pinned_template() {
    let template = Template::from_model(&root())
        .unwrap()
        .expect("the artifact must carry a chat template");
    assert!(
        template.source().ends_with("tokenizer.chat_template"),
        "template came from {}",
        template.source()
    );
    for case in fixture()["cases"].as_array().unwrap() {
        if case["context"].is_null() {
            continue;
        }
        assert_eq!(
            template.render(&case["context"]).unwrap(),
            case["text"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
}
