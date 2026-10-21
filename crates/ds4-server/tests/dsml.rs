//! C↔Rust DSML decode-state + sampling-override tapes (no model).

use ds4_server::dsml_dump_script;
use ds4_server::parse::ToolSchemaOrder;
use ds4_server::tools::find_tool_start;
use ds4_server::tools::parse_dsml_tolerant;
use ds4_server::tools::{
    DSML_INVOKE_END_SPACED, DSML_INVOKE_START_SPACED, DSML_PARAM_END_SPACED,
    DSML_PARAM_START_SPACED, DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START_SPACED,
};
use ds4_server::{
    dsml_decode_state_for_text, ChatFormat, DsmlDecodeState, DsmlDecodeTracker, ModelSyntax,
    SemAccum, ToolSyntax,
};
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
        // The GLM vectors are the reference's own (ds4_server.c:20571, :20608).
        "glm-boundaries-token",
        "glm-boundaries-byte",
        "glm-control-text",
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

// ---------------------------------------------------------------------------
// The live served turn (Spark ds41 deployment, open-grok session 01a14f5d,
// 2026-10-18).  Fixtures are the assistant content exactly as the client
// recorded it.
// ---------------------------------------------------------------------------

fn live_orders() -> Vec<ToolSchemaOrder> {
    let mut out = Vec::new();
    for (name, props, types) in [
        ("memory_search", vec!["query"], vec!["string"]),
        (
            "read_file",
            vec!["target_file", "limit"],
            vec!["string", "integer"],
        ),
        (
            "list_dir",
            vec!["target_directory", "description"],
            vec!["string", "string"],
        ),
    ] {
        out.push(ToolSchemaOrder {
            name: name.to_string(),
            wire_name: name.to_string(),
            namespace: name.to_string(),
            responses_tool_search: false,
            prop: props.into_iter().map(str::to_string).collect(),
            prop_type: types.iter().copied().map(str::to_string).collect(),
            // These fixtures declare a scalar type, so the head of the `type`
            // list is the type itself (parse.rs schema_property_type_first).
            prop_type_first: types.iter().copied().map(str::to_string).collect(),
        });
    }
    out
}

fn live_text(file: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/ds41-dsml/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn live_turn(file: &str) -> Vec<ds4_server::parse::ToolCall> {
    let text = live_text(file);
    let parsed = parse_dsml_tolerant(&text, &live_orders(), false)
        .unwrap_or_else(|| panic!("{file}: the finished turn recovered no call at all"));
    parsed.calls
}

/// The artifact roots each call on the TOOL NAME and never writes the
/// wrapper or invoke opens; its closes are the decorated DS41 forms, and the
/// root tag carries the residue of the `name="` attribute the reference's own
/// prompt teaches (`<read_file">`).  Measured before this case: the tolerant
/// parse recovered nothing from either turn, so the route served the block as
/// assistant text and no tool ran (the client's turn recorded zero tool
/// calls).
#[test]
fn live_tool_root_block_recovers_its_calls() {
    let t1 = live_turn("live-01a14f5d-turn1.txt");
    let names: Vec<&str> = t1.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        ["memory_search", "memory_search", "read_file", "read_file"]
    );
    assert!(t1[0]
        .arguments
        .contains("SolidCraft GUI worker app visible display headless session"));

    let t2 = live_turn("live-01a14f5d-turn2.txt");
    let names: Vec<&str> = t2.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        ["read_file", "read_file", "read_file", "list_dir"],
        "the malformed list_dir (its child close is the wrong tag) must be dropped, not invented"
    );
    assert!(
        t2[2].arguments.contains("\"limit\": 200"),
        "a declared integer child keeps its type: {}",
        t2[2].arguments
    );
    assert!(t2[3].arguments.contains("/data/3D"));
}

/// The tracker half of the same defect.  The parser now serves the calls, but
/// the live stream still drew the block as assistant content, because the
/// start scan listed the wrapper and invoke opens only and the artifact's root
/// is the bare tool name.  A block the scan does not recognise is a block the
/// tracker cannot hold back.
#[test]
fn tracker_enters_tool_mode_on_the_quoted_tool_root() {
    let text = live_text("live-01a14f5d-turn1.txt");

    let at = find_tool_start(&text, ChatFormat::DeepSeek, &live_orders())
        .expect("the artifact's root tag must start a tool block");
    assert_eq!(
        &text[at..at + b"<memory_search\">".len()],
        b"<memory_search\">",
        "the scan must stop at the root tag, not at some later marker"
    );

    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    acc.set_tool_orders(&live_orders());
    acc.feed(&text, &[]);
    assert!(
        acc.saw_tool_start,
        "the quoted tool root must take the live tracker into tool mode"
    );
    assert!(
        acc.saw_tool_end,
        "the decorated wrapper close must close the block"
    );
}

/// The reference's GLM tracker vectors, ds4_server.c:20571
/// (`test_glm_decode_tracker_boundaries` in /data/ds4): the block's fragments,
/// token-sized and byte-by-byte, must land on the states the reference asserts.
/// GLM's block is not in the DSML syntax table, so before the dialect the
/// tracker never left Outside and every structural position kept the request's
/// draw instead of the raw argmax.
#[test]
fn glm_tracker_boundaries_match_the_reference() {
    let parts = [
        "prose <tool_call>bash<arg_key>command</arg_key>",
        "<arg_value>printf '<html>'",
        "<",
        "/arg_",
        "value>",
        "<arg_key>description</arg_key><arg_value>test",
        "</arg_value></tool_call>",
    ];
    let expected = [
        DsmlDecodeState::Structural,
        DsmlDecodeState::StringBody,
        // A lone '<' stays payload; two bytes of the close marker are structure.
        DsmlDecodeState::StringBody,
        DsmlDecodeState::Structural,
        DsmlDecodeState::Structural,
        DsmlDecodeState::StringBody,
        DsmlDecodeState::Outside,
    ];
    for bytewise in [false, true] {
        let mut tracker = DsmlDecodeTracker::default();
        tracker.set_tool_syntax(ToolSyntax::Glm);
        let mut raw = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if bytewise {
                for byte in part.bytes() {
                    raw.push(byte);
                    tracker.update(&raw);
                }
            } else {
                raw.extend_from_slice(part.as_bytes());
                tracker.update(&raw);
            }
            assert_eq!(
                tracker.decode,
                expected[i],
                "fragment {i} bytewise={bytewise} raw={}",
                String::from_utf8_lossy(&raw)
            );
        }
    }
}

/// Before the dialect the tracker recognized no GLM block at all: `<tool_call>`
/// is not one of the DSML spellings (dsml.rs `SYNTAXES`), so the whole block read
/// Outside and every structural position kept the request's draw.  This test
/// pins the defect the GLM dialect fixes.
#[test]
fn glm_text_under_the_dsml_dialect_is_untracked() {
    let raw = b"prose <tool_call>bash<arg_key>command</arg_key><arg_value>ls -la";
    let mut tracker = DsmlDecodeTracker::default();
    tracker.set_tool_syntax(ToolSyntax::Dsml);
    tracker.update(raw);
    assert_eq!(tracker.decode, DsmlDecodeState::Outside);
}

/// ds4_server.c (test_tool_control_text_inside_arguments): protocol-looking
/// bytes inside an `<arg_value>` body are payload.  The block stays open, so the
/// stream does not cut the argument at a `</tool_call>` the model wrote as text.
#[test]
fn glm_control_text_inside_an_argument_is_payload() {
    // The reference body (ds4_server.c:20609): control-looking text written
    // INTO the argument value.
    let body = "literal </tool_call> </\u{ff5c}DSML\u{ff5c}tool_calls> </think> <think>";
    let raw =
        format!("<think>reason</think><tool_call>write<arg_key>content</arg_key><arg_value>{body}");
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    acc.set_model_syntax(ModelSyntax::Glm53);
    acc.feed(raw.as_bytes(), &[]);
    assert_eq!(acc.dsml_state(), DsmlDecodeState::StringBody);
    assert!(acc.saw_tool_start, "the block opened");
    assert!(
        !acc.saw_tool_end,
        "an argument body must not close the block (the reference reads the tracker, ds4_server.c:7755)"
    );
}

/// GLM's block opens only on its own `<tool_call>`.  A DSML spelling such as
/// `<tool_calls>` in GLM prose is text: if it opened the block, the GLM tracker
/// would never close it, and the turn's EOS would read as an unterminated call
/// (non-streaming retries it, streaming reports an error).
#[test]
fn glm_dsml_spelling_in_prose_does_not_open_a_block() {
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    acc.set_model_syntax(ModelSyntax::Glm53);
    acc.feed(
        b"DeepSeek wraps calls in <tool_calls> blocks; GLM does not.",
        &[],
    );
    assert!(
        !acc.saw_tool_start,
        "a DSML spelling is not GLM's opening marker"
    );
    assert!(!acc.saw_tool_end);

    acc.feed(
        b" <tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>",
        &[],
    );
    assert!(acc.saw_tool_start, "GLM's own marker opens the block");
    assert!(acc.saw_tool_end, "and its tracker closes it");
}

/// The sampling consequence of the dialect: the tag head and the argument tags
/// take the raw argmax, the value body keeps the request's draw
/// (ds4_server.c:7575 `dsml_decode_state_is_syntax`).
#[test]
fn glm_structural_positions_take_the_argmax() {
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    acc.set_model_syntax(ModelSyntax::Glm53);
    acc.feed(b"prose <tool_call>bash<arg_key>command</arg_key>", &[]);
    assert!(acc.dsml_state().is_tool(), "the GLM block opens tool mode");
    assert!(acc.syntax_greedy(&[]), "the tag head is structural");
    acc.feed(b"<arg_value>ls -la", &[]);
    assert!(!acc.syntax_greedy(&[]), "an argument body keeps the draw");
    acc.feed(b"</arg_value></tool_call>", &[]);
    assert!(!acc.dsml_state().is_tool(), "the block closed");
}
