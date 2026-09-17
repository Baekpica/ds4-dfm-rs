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
