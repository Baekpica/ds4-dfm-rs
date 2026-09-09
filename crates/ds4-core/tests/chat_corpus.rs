use ds4_core::chat_template::{RenderClock, Template};
use serde_json::Value;

fn mismatch(actual: &str, expected: &str) -> String {
    let offset = actual
        .chars()
        .zip(expected.chars())
        .take_while(|(left, right)| left == right)
        .count();
    let start = offset.saturating_sub(24);
    let actual: String = actual.chars().skip(start).take(100).collect();
    let expected: String = expected.chars().skip(start).take(100).collect();
    format!("character {offset}: got {actual:?}, expected {expected:?}")
}

fn check_model(name: &str, source: &str) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/chat-template/model-vectors.json"
    ))
    .unwrap();
    let model = fixture["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["name"] == name)
        .unwrap_or_else(|| panic!("missing model fixture: {name}"));
    let template = Template::compile(source, RenderClock::Fixed(0))
        .unwrap_or_else(|error| panic!("{name}: official template did not compile: {error}"));
    let vectors = model["vectors"].as_array().unwrap();
    assert!(!vectors.is_empty(), "{name}: empty reference corpus");
    let mut failures = Vec::new();

    // Check every case so one unsupported source construct cannot hide others.
    for case in vectors {
        let label = case["name"].as_str().unwrap();
        let result = template.render(&case["context"]);
        match (case["expected"].as_str(), result) {
            (Some(expected), Ok(actual)) if actual == expected => {}
            (Some(expected), Ok(actual)) => {
                failures.push(format!("{label}: {}", mismatch(&actual, expected)));
            }
            (Some(_), Err(error)) => failures.push(format!("{label}: {error}")),
            (None, Err(_)) => assert!(case["error_type"].is_string()),
            (None, Ok(actual)) => failures.push(format!(
                "{label}: expected Python {} ({}), got {actual:?}",
                case["error_type"], case["error_message"]
            )),
        }
    }

    assert!(failures.is_empty(), "{name}:\n{}", failures.join("\n"));
}

macro_rules! model_case {
    ($test:ident, $model:literal) => {
        #[test]
        fn $test() {
            check_model(
                $model,
                include_str!(concat!(
                    "../../../tests/fixtures/chat-template/models/",
                    $model,
                    "/chat_template.jinja"
                )),
            );
        }
    };
}

model_case!(solar, "solar");
model_case!(exaone, "exaone");
model_case!(motif, "motif");
model_case!(dots, "dots");
model_case!(qwen, "qwen");
model_case!(qwen_uncensored, "qwen-uncensored");
model_case!(glm, "glm");
model_case!(k2, "k2");
model_case!(inkling, "inkling");
