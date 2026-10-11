//! C↔Rust DSML decode-state + sampling-override tapes (no model).

use ds4_server::dsml_dump_script;
use ds4_server::tools::{
    DSML_INVOKE_START_SPACED, DSML_PARAM_END_SPACED, DSML_PARAM_START_SPACED,
    DSML_TOOL_CALLS_START_SPACED,
};
use ds4_server::{ChatFormat, SemAccum};
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_DSML_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dsml_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/dsml_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(name: &str) -> String {
    let out = Command::new(require_oracle())
        .arg(name)
        .output()
        .expect("run dsml_c_oracle");
    assert!(
        out.status.success(),
        "oracle {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_script(name: &str) {
    let rust = dsml_dump_script(name);
    let c = c_str(name);
    if rust != c {
        panic!("{name} mismatch\n--- rust ---\n{rust}\n--- c ---\n{c}");
    }
}

#[test]
fn dsml_states_match_c() {
    for name in [
        "state-prefix",
        "state-path-param",
        "state-path-closing",
        "state-json-struct",
        "state-json-string",
        "state-done",
    ] {
        assert_script(name);
    }
}

#[test]
fn sampling_override_matches_c() {
    for name in [
        "override-required",
        "override-think-cap",
        "override-tool-result",
        "cap-low-128",
        "cap-high-128",
        "cap-max-600",
    ] {
        assert_script(name);
    }
    let req = dsml_dump_script("override-required");
    assert_eq!(req.trim(), "103 204");
    let think = dsml_dump_script("override-think-cap");
    assert_eq!(think.trim(), "0 305");
    let tr = dsml_dump_script("override-tool-result");
    assert_eq!(tr.trim(), "0 305");
}

/// The tool-syntax greedy question (the engine's v41_syntax_greedy +
/// test_syntax_greedy_spaced_dsml, server_generate_v41.c): the tag, the
/// parameter head and the JSON punctuation take the raw argmax; the string
/// value body and the thinking section keep the request's draw.
fn greedy_after(text: &str, thinking_inside: bool) -> bool {
    let mut acc = SemAccum::init(true, true, thinking_inside, ChatFormat::DeepSeek, b"");
    acc.feed(text.as_bytes(), &[]);
    acc.syntax_greedy(&[])
}

#[test]
fn syntax_greedy_spaced_dsml() {
    let head = format!(
        "Listing.\n\n{}{} name=\"bash\">\n",
        DSML_TOOL_CALLS_START_SPACED, DSML_INVOKE_START_SPACED
    );
    assert!(!greedy_after("Listing.", false));
    assert!(greedy_after(&head, false));
    assert!(!greedy_after(&head, true));
    // The parameter value keeps the request's draw.
    assert!(!greedy_after(
        &format!("{head}{DSML_PARAM_START_SPACED} name=\"command\" string=\"true\">ls -la"),
        false
    ));
    // The JSON value's structural position (the number / bracket head) is greedy.
    assert!(greedy_after(
        &format!("{head}{DSML_PARAM_START_SPACED} name=\"n\" string=\"false\">"),
        false
    ));
    // The value closed: back at a tag position, greedy again.
    assert!(greedy_after(
        &format!(
            "{head}{DSML_PARAM_START_SPACED} name=\"command\" string=\"true\">ls{DSML_PARAM_END_SPACED}\n"
        ),
        false
    ));
}
