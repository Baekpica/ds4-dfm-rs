#[path = "support/serving_docs.rs"]
mod serving_docs;

#[test]
fn serving_docs_match_runtime() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/serving-capabilities.md");
    let actual = std::fs::read_to_string(path).expect("generated serving capabilities document");
    assert_eq!(
        actual,
        serving_docs::render(),
        "run cargo run -q -p ds4-core --example serving_docs > docs/serving-capabilities.md"
    );
}

#[test]
fn v41_checkpoint_is_unqualified() {
    let rendered = serving_docs::render();
    assert!(rendered.contains(
        "| DeepSeek V4.1 Flash | serial / none | exact / unverified | unverified | none |"
    ));
    assert!(rendered.contains("--prefix-reuse off|exact|auto"));
    assert!(!rendered.contains("serving layer keeps no reuse plan"));
}
