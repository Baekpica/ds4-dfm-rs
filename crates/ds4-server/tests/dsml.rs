//! C↔Rust DSML decode-state + sampling-override tapes (no model).

use ds4_server::dsml_dump_script;
use ds4_server::tools::{
    DSML_INVOKE_END_SPACED, DSML_INVOKE_START_SPACED, DSML_PARAM_END_SPACED,
    DSML_PARAM_START_SPACED, DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START_SPACED,
};
use ds4_server::{dsml_decode_state_for_text, ChatFormat, DsmlDecodeState, SemAccum};
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

/// The stream marker scan must see the official spaced close: find_tool_end's
/// DeepSeek candidate list carried the LONG / SHORT / PLAIN wrapper ends but
/// not the SPACED one (found 2026-10-11 on the chat gate against the 221b9e2
/// oracle: the live stream never closed the block, so the loop drew one extra
/// token -- EOS -- after the wrapper; the engine stops at the tag, its
/// find_any_tool_end walks all four syntaxes, server_dsml_parse.c:88-93).
#[test]
fn stream_closes_on_spaced_wrapper() {
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    let block = format!(
        "Checking.\n\n{tcs} name=\"get_weather\">\n{ps} name=\"city\" string=\"true\">Paris{pe}\n{ie}\n{tce}",
        tcs = DSML_TOOL_CALLS_START_SPACED,
        ps = DSML_PARAM_START_SPACED,
        pe = DSML_PARAM_END_SPACED,
        ie = DSML_INVOKE_END_SPACED,
        tce = DSML_TOOL_CALLS_END_SPACED,
    );
    acc.feed(block.as_bytes(), &[]);
    assert!(
        acc.saw_tool_start,
        "the spaced wrapper open must enter tool mode"
    );
    assert!(
        acc.saw_tool_end,
        "the spaced wrapper close must close the block (the engine's find_any_tool_end walks all four syntaxes)"
    );
}

/// The artifact mixes spellings inside one block (f302be7, 2026-10-11): the
/// spaced block carrying bare parameter tags.  The tracker must take the bare
/// open and close as parameter boundaries and reach Outside at the block end;
/// before the aliases it stopped at Structural, holding the block open.
#[test]
fn tracker_accepts_bare_parameter_tags() {
    let block = format!(
        "{tcs}\n{is} name=\"a\">\n<parameter name=\"x\" string=\"true\">v</parameter>\n{ie}\n{tce}",
        tcs = DSML_TOOL_CALLS_START_SPACED,
        is = DSML_INVOKE_START_SPACED,
        ie = DSML_INVOKE_END_SPACED,
        tce = DSML_TOOL_CALLS_END_SPACED,
    );
    assert_eq!(
        dsml_decode_state_for_text(block.as_bytes()),
        DsmlDecodeState::Outside
    );
}
