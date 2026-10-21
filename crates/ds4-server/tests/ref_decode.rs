//! The reference leg, decode side: our DSML parse against the official
//! DeepSeek-V4.1 encoding implementation (the oracle).
//!
//! The oracle is vendored at `tests/ds41_ref/oracle/encoding.py` (sha256
//! 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2), pinned
//! by that digest in `tests/ds41_ref/ref_drive.py`.  Both
//! halves of every case under `tests/fixtures/ds41-ref/` are the oracle's own
//! output: `<case>.txt` is the completion the oracle's templates rendered, and
//! `<case>.calls.json` is the oracle parsing that same text with
//! `parse_message_from_completion_text` (encoding.py:946-990).  Regenerate with
//! `bash tests/ds41_ref/regen_fixtures.sh`; the driver refuses any other
//! oracle revision.
//!
//! What is compared: the content and reasoning split byte for byte, and each
//! call's name and arguments as JSON values.  The arguments *string* is not
//! compared byte for byte on purpose -- our shape minifies non-string children
//! exactly as the C implementation this port descends from does
//! (`tool_call_json_args_add`, quoted there at ds4_server.c:5971-5982, over
//! json_minify_raw_value :411), while the oracle
//! keeps the parameter body verbatim; both decode to the same value, and the
//! client echoes whichever string it received (requirement 2.3.5).  The
//! minified shape is pinned by our own test below, so the difference is
//! recorded rather than silent.

use ds4_server::{parse_generated_message, ChatFormat, ModelSyntax, ToolSchemaOrder};
use serde_json::Value;

/// The oracle's parsed message, as its fixture records it.
struct Oracle {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ds41-ref")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn oracle(name: &str) -> Oracle {
    let raw = fixture(&format!("{name}.calls.json"));
    let v: Value = serde_json::from_slice(&raw).expect("oracle fixture is JSON");
    let calls = v["tool_calls"]
        .as_array()
        .expect("tool_calls array")
        .iter()
        .map(|c| {
            let f = &c["function"];
            (
                f["name"].as_str().expect("name").to_string(),
                f["arguments"].as_str().expect("arguments").to_string(),
            )
        })
        .collect();
    Oracle {
        content: v["content"].as_str().unwrap_or_default().to_string(),
        reasoning: v["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        calls,
    }
}

/// Parse one completion as the ds41 route parses it and check both sides agree.
fn check(name: &str, thinking: bool) {
    let text = fixture(&format!("{name}.txt"));
    let want = oracle(name);

    let parsed = parse_generated_message(
        ModelSyntax::DeepSeek,
        &text,
        thinking,
        ChatFormat::DeepSeek,
        &[] as &[ToolSchemaOrder],
    );

    assert!(
        parsed.ok,
        "{name}: our parse rejected a completion the oracle accepts"
    );
    assert_eq!(
        String::from_utf8_lossy(&parsed.content),
        want.content,
        "{name}: content differs"
    );
    assert_eq!(
        String::from_utf8_lossy(&parsed.reasoning),
        want.reasoning,
        "{name}: reasoning differs"
    );

    let got: Vec<(String, String)> = parsed
        .calls
        .iter()
        .map(|c| (c.name.clone(), c.arguments.clone()))
        .collect();

    assert_eq!(got.len(), want.calls.len(), "{name}: call count differs");
    for (i, ((gn, ga), (wn, wa))) in got.iter().zip(&want.calls).enumerate() {
        assert_eq!(gn, wn, "{name}: call {i} name differs");
        let gv: Value = serde_json::from_str(ga)
            .unwrap_or_else(|e| panic!("{name}: our call {i} arguments are not JSON: {e} ({ga})"));
        let wv: Value = serde_json::from_str(wa).unwrap_or_else(|e| {
            panic!("{name}: oracle call {i} arguments are not JSON: {e} ({wa})")
        });
        assert_eq!(gv, wv, "{name}: call {i} arguments differ as values");
    }
}

/// One call, one string and one integer child: the canonical shape.
#[test]
fn canonical_single_call_matches_official_parser() {
    check("dec-one-call", false);
}

/// Two invokes in one block (the shape TP-03 exercises on the wire).
#[test]
fn two_invokes_match_official_parser() {
    check("dec-two-calls", false);
}

/// Every JSON child kind the type attribute distinguishes, including a nested
/// array and object.
#[test]
fn complex_argument_kinds_match_official_parser() {
    check("dec-complex-args", false);
}

/// A non-ASCII string child: the oracle's json.dumps is ensure_ascii False, so
/// the value round-trips raw and both sides must agree on the character.
#[test]
fn unicode_argument_matches_official_parser() {
    check("dec-unicode", false);
}

/// A block marker inside a parameter value is data, never structure
/// (inventory G9; the reference rejects markers inside values at
/// ds4_server.c:5890 find_tool_structural_text): one call, whose body carries
/// the wrapper's own text.
#[test]
fn marker_inside_a_value_is_not_structure() {
    check("dec-marker-in-value", false);
}

/// Thinking mode: reasoning before the close, prose after it, then the block.
#[test]
fn thinking_split_matches_official_parser() {
    check("dec-thinking", true);
}

/// Our arguments string for a composite child is the minified form, as the
/// reference server builds it (ds4_server.c:5971-5982).  The oracle keeps the
/// parameter body verbatim; the values are equal, the bytes are not, and this
/// test pins our side of that difference so a future edit cannot change it
/// unnoticed.
#[test]
fn composite_arguments_are_minified_as_the_reference_does() {
    let text = fixture("dec-complex-args.txt");
    let parsed = parse_generated_message(
        ModelSyntax::DeepSeek,
        &text,
        false,
        ChatFormat::DeepSeek,
        &[] as &[ToolSchemaOrder],
    );
    assert_eq!(parsed.calls.len(), 1);

    let ours = &parsed.calls[0].arguments;
    let oracle_args = oracle("dec-complex-args").calls[0].1.clone();

    // Members are joined with ", " (the reference's tool_call_json_args_add),
    // while a composite *value* is minified inside.
    assert!(
        ours.contains(r#""deps": ["a","b"]"#),
        "our composite child is not minified: {ours}"
    );
    assert!(
        oracle_args.contains(r#""deps": ["a", "b"]"#),
        "the oracle's child is no longer verbatim: {oracle_args}"
    );

    let gv: Value = serde_json::from_str(ours).expect("our arguments are JSON");
    let wv: Value = serde_json::from_str(&oracle_args).expect("oracle arguments are JSON");
    assert_eq!(
        gv, wv,
        "the minified and verbatim forms must carry one value"
    );
}
