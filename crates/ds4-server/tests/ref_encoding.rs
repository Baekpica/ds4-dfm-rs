//! The reference leg, encode side: our DS41 prompt bytes against the official
//! DeepSeek-V4.1 encoding implementation (the oracle).
//!
//! The oracle is vendored at `tests/ds41_ref/oracle/encoding.py` (sha256
//! 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2), an
//! operator-supplied file that `tests/ds41_ref/ref_drive.py` pins by that
//! digest; the external tree it was first read from is no longer consulted.
//! Each
//! fixture beside this test is the oracle's own output for the case of the
//! same name under `tests/ds41_ref/cases/`, regenerated with
//! `bash tests/ds41_ref/regen_fixtures.sh`; driving the oracle is what
//! `tests/ds41_ref/ref_drive.py` does, and it refuses any other revision.
//! The fixtures therefore pin bytes we did not author.
//!
//! Coverage is traced in `docs/ds41-dsml-reference-matrix.md`: every CTR
//! requirement and every inventory gap names the case that exercises it, and
//! a case that does not yet hold is marked OPEN there rather than dropped.

use ds4_server::{
    render_dsml_chat_choice_ex, ChatMsg, ChatPart, DsmlHead, ThinkMode, ToolCall, ToolChoice,
};

/// The head our ds41 route passes: the engine's V4.1 encoding.py head with the
/// tokenizer's system token (render.rs `dsml_head`).
const V41: DsmlHead = DsmlHead::V41 {
    system_token: true,
    effort: None,
};

fn msg(role: &str, content: &str) -> ChatMsg {
    ChatMsg {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}

fn assistant(content: &str, calls: Vec<ToolCall>) -> ChatMsg {
    ChatMsg {
        role: "assistant".into(),
        content: content.into(),
        calls,
        ..Default::default()
    }
}

fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
    }
}

/// A `tool` role message as the chat parse builds it: the body plus the id of
/// the call it answers (`add_tool_call_id`, parse.rs:885).  The id is what the
/// renderer orders a run of results by, so a case that omits it cannot
/// exercise the order rule.
fn tool_result(id: &str, body: &str) -> ChatMsg {
    ChatMsg {
        role: "tool".into(),
        content: body.into(),
        tool_call_id: id.into(),
        tool_call_ids: vec![id.into()],
        ..Default::default()
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ds41-ref")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Report the first differing byte, with both sides escaped, so a mismatch is
/// readable without dumping two long prompts.
fn diff_report(got: &[u8], want: &[u8]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.len().min(want.len()));
    let lo = at.saturating_sub(60);
    format!(
        "first difference at byte {at} (got {} bytes, want {} bytes)\n  got  ...{}\n  want ...{}",
        got.len(),
        want.len(),
        String::from_utf8_lossy(&got[lo..(at + 60).min(got.len())]),
        String::from_utf8_lossy(&want[lo..(at + 60).min(want.len())]),
    )
}

/// Render one case and compare with the oracle's fixture of the same name.
fn check(name: &str, msgs: &[ChatMsg], schemas: &str, mode: ThinkMode) {
    check_head(name, msgs, schemas, mode, V41);
}

/// The same, with the effort budget the request carried (O-01's numeric half).
fn check_with_budget(
    name: &str,
    msgs: &[ChatMsg],
    schemas: &str,
    mode: ThinkMode,
    budget: Option<u8>,
) {
    check_head(name, msgs, schemas, mode, V41.with_effort(budget));
}

fn check_head(name: &str, msgs: &[ChatMsg], schemas: &str, mode: ThinkMode, head: DsmlHead) {
    let got = render_dsml_chat_choice_ex(msgs, schemas, mode, ToolChoice::Auto, head)
        .unwrap_or_else(|e| panic!("{name}: render refused: {}", e.0));
    let want = fixture(&format!("{name}.prompt.txt"));

    assert_eq!(
        String::from_utf8_lossy(&got),
        String::from_utf8_lossy(&want),
        "{name}: {}",
        diff_report(&got, &want)
    );
}

/// A case's tool, as the route receives it: one client JSON value per line
/// (render.rs:475-520 reformats each to json.dumps shape, which is what the
/// oracle emits).  Each constant mirrors the `tools` array of the case that
/// uses it -- a schema that declares a property its case does not is a test
/// bug, not a carriage difference.
const READ_FILE_SCHEMA: &str = concat!(
    r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"},"limit":{"type":"integer"}},"required":["target_file"]}}"#
);
const READ_FILE_SCHEMA_MIN: &str = concat!(
    r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}"#
);

#[test]
fn tools_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-tools-chat", &msgs, READ_FILE_SCHEMA, ThinkMode::None);
}

#[test]
fn plain_chat_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hello."),
        assistant("Hello.", Vec::new()),
        msg("user", "Say it again."),
    ];
    check("enc-plain-chat", &msgs, "", ThinkMode::None);
}

/// The type attribute: `string="true"` for a string child, `string="false"`
/// with the value in json.dumps shape for every other kind
/// (encoding.py encode_arguments_to_dsml:524-556).  The echoed arguments are
/// the client's raw JSON string, as requirement 2.3.5 replays them.
#[test]
fn typed_parameters_match_official_encoding() {
    let args = r#"{"target_file": "/tmp/x.txt", "limit": 40, "force": true, "ratio": 1.5, "note": null, "deps": ["a", "b"], "opts": {"k": 1}}"#;
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file and label it."),
        assistant("Reading it now.", vec![call("call_1", "read_file", args)]),
        msg("tool", "file body"),
        msg("user", "Now summarise it."),
    ];
    let schemas = concat!(
        r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"},"limit":{"type":"integer"},"force":{"type":"boolean"},"ratio":{"type":"number"},"note":{"type":"string"},"deps":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"}},"required":["target_file"]}}"#
    );
    check("enc-argtypes-chat", &msgs, schemas, ThinkMode::None);
}

/// A replayed assistant's `arguments` is a string on the wire, and the oracle
/// tolerates two shapes beside the object JSON: it decodes a string twice
/// before giving up, and a value that is still not a dict becomes one
/// parameter named `arguments` carrying the value as it stood
/// (encoding.py encode_arguments_to_dsml:524-556).
///
/// The three arms measured 2026-10-20.  A string that decodes to an object on
/// the second pass renders that object's parameters; a JSON array renders one
/// `arguments` parameter with `string="false"` and the json.dumps spelling of
/// the value; a string that is not JSON at all renders one `arguments`
/// parameter with `string="true"` and the text verbatim.
#[test]
fn double_encoded_arguments_match_official_encoding() {
    // The wire string is a JSON object wrapped in one more JSON string, so the
    // first decode yields a string and the second the object.
    let args = r#""{\"target_file\": \"/tmp/x.txt\"}""#;
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
        assistant("Calling it now.", vec![call("call_1", "read_file", args)]),
    ];
    check(
        "enc-args-double",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

#[test]
fn non_dict_arguments_match_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
        assistant(
            "Calling it now.",
            vec![call("call_1", "read_file", "[1, 2]")],
        ),
    ];
    check(
        "enc-args-nondict",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

#[test]
fn unparsable_arguments_match_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
        assistant(
            "Calling it now.",
            vec![call("call_1", "read_file", "not json")],
        ),
    ];
    check(
        "enc-args-plain",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

/// A user message whose content is an array of blocks: a text block beside an
/// image block, and a text-only pair.
///
/// The oracle joins the pieces the client sent as separate blocks with a blank
/// line -- `_process_image_blocks` turns an image into a placeholder text block
/// and the user branch writes `"\n\n".join(parts)` (encoding.py:323-353,
/// :633-658) -- while this port concatenates them, at parse time for the text
/// pieces (`append_text_part`, parse.rs:587) and at render time for the
/// placeholder (`append_dsml_user_content`, render.rs:1376).
///
/// Measured 2026-10-20 on both cases: the two prompts carry the same pieces in
/// the same order and differ by that one separator, which is what these cases
/// pin.  Recorded as an open row in docs/ds41-dsml-function-map.md row
/// `process_image_messages`; the fix is a render/parse join gated to the V4.1
/// head, and it is the operator's call because it moves every image-bearing
/// prompt.
fn check_block_join(name: &str, msgs: &[ChatMsg]) {
    let got = render_dsml_chat_choice_ex(msgs, "", ThinkMode::None, ToolChoice::Auto, V41)
        .unwrap_or_else(|e| panic!("{name}: render refused: {}", e.0));
    let got = String::from_utf8(got).expect("our prompt is UTF-8");
    let want = String::from_utf8(fixture(&format!("{name}.prompt.txt"))).expect("fixture is UTF-8");

    assert_eq!(
        want.matches("\n\n").count(),
        1,
        "{name}: the fixture must hold exactly the one separator this test removes"
    );
    assert_eq!(
        got,
        want.replace("\n\n", ""),
        "{name}: the two prompts differ beyond the block separator"
    );
}

/// A text block and an image block in one user message.
#[test]
fn image_block_beside_text_is_pinned() {
    let mut user = msg("user", "What is in this image?");
    user.parts = vec![
        ChatPart::Text("What is in this image?".into()),
        ChatPart::Image(0),
    ];
    let msgs = [msg("system", "You are a helpful assistant."), user];
    check_block_join("enc-image-chat", &msgs);
}

/// Two text blocks in one user message, with no image anywhere.
#[test]
fn text_blocks_are_pinned() {
    let mut user = msg("user", "First part.Second part.");
    user.parts = vec![
        ChatPart::Text("First part.".into()),
        ChatPart::Text("Second part.".into()),
    ];
    let msgs = [msg("system", "You are a helpful assistant."), user];
    check_block_join("enc-blocks-chat", &msgs);
}

#[test]
fn thinking_tools_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check(
        "enc-think-tools",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::High,
    );
}

/// The numeric reasoning-effort header: 75 for the default, 100 for max
/// (encoding.py:444-472).  Only thinking mode renders it, and only at index 0.
#[test]
fn thinking_max_effort_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-think-max", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::Max);
}

/// CLOSED (matrix row O-01): the oracle renders "Reasoning Effort: 50" for the
/// low effort (encoding.py:449-451).  Measured before the fix: low rendered no
/// line at all (the requested effort never reached the renderer on the ds41
/// route, where the engine collapses low into high) and the fixture was 1372
/// bytes against our 1311.  The route now carries the requested effort beside
/// the collapsed mode (`ParsedRequest::effort`, route.rs `RequestedEffort`) and
/// the V4.1 head writes the low mapping's 50 (render.rs `v41_effort_prefix`).
#[test]
fn thinking_low_effort_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-think-low", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::Low);
}

/// CLOSED (matrix row O-01, the numeric half): the oracle renders an integer
/// budget in [1,100] as itself (encoding.py:456-473).  Measured before the fix:
/// a numeric `reasoning_effort` was refused at the parse, which read a string
/// name and nothing else.  The request now carries the number as its own budget
/// and the head writes it verbatim; the engine keeps its collapsed mode.
#[test]
fn thinking_numeric_effort_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check_with_budget(
        "enc-think-budget",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::High,
        Some(40),
    );
}

/// A mid-conversation system message: the oracle writes the system token
/// again and lets the message introduce its own turn (encoding.py:621-627,
/// :733-739), and counts that index as user-like for the assistant header
/// that follows (:562-582).  Measured before the fix: our renderer hoisted
/// every system message into the head, so "Answer in one line." landed before
/// the tools preamble instead of after the user turn.
#[test]
fn mid_conversation_system_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
        msg("system", "Answer in one line."),
    ];
    check(
        "enc-midsystem-chat",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

/// Tool results are ordered by the preceding assistant's call order, not by
/// arrival (encoding.py:233-266 sort_tool_results_by_call_order).  Measured
/// before the fix: the oracle's block carried body a then body b, ours body b
/// then body a, every other byte of the 1767 matching.
#[test]
fn tool_results_in_call_order_match_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Check both files."),
        assistant(
            "Checking.",
            vec![
                call("call_a", "read_file", r#"{"target_file": "/tmp/a.txt"}"#),
                call("call_b", "read_file", r#"{"target_file": "/tmp/b.txt"}"#),
            ],
        ),
        tool_result("call_b", "body b"),
        tool_result("call_a", "body a"),
        msg("user", "Summarise both."),
    ];
    check(
        "enc-toolresults-order",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

/// A historical assistant's `reasoning_content` never reaches the prompt in
/// chat mode, where neither side renders it.  This case pins that it does not
/// leak; the thinking-mode case below is the one that decides the drop rule.
#[test]
fn historical_reasoning_not_rendered_in_chat_matches_official_encoding() {
    let mut history = assistant("Hi.", Vec::new());
    history.reasoning = "HISTORICAL-THOUGHT-MARKER".into();
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hi."),
        history,
        msg("user", "Say it again."),
    ];
    check("enc-drop-thinking", &msgs, "", ThinkMode::None);
}

/// A user-like message carrying only tool results does not start a turn
/// (encoding.py `find_last_user_index`:562-582).  The client of record sends
/// tool results as their own messages, so a tool turn with no following user
/// message ends on exactly this shape.
#[test]
fn tool_result_only_tail_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file."),
        assistant(
            "",
            vec![call(
                "call_1",
                "read_file",
                r#"{"target_file": "/tmp/x.txt"}"#,
            )],
        ),
        tool_result("call_1", "body"),
    ];
    check(
        "enc-toolresult-only",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}

/// The drop rule itself, measured in the mode that renders reasoning: with no
/// tools anywhere, `_drop_thinking_messages` (encoding.py:749-767) strips the
/// historical assistant's `reasoning_content`, so the `</think>` its
/// predecessor's transition token wrote is all the thinking that message has.
/// Same conversation as the chat-mode case above, so the two verdicts differ
/// only by the mode.
#[test]
fn historical_reasoning_is_dropped_matches_official_encoding() {
    let mut history = assistant("Hi.", Vec::new());
    history.reasoning = "HISTORICAL-THOUGHT-MARKER".into();
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hi."),
        history,
        msg("user", "Say it again."),
    ];
    check("enc-think-drop-thinking", &msgs, "", ThinkMode::High);
}

/// An assistant at or after the last user index keeps its reasoning: the
/// oracle writes `{reasoning}</think>` there (encoding.py:678-685, the
/// `index > last_user_idx` arm of the assistant branch).
#[test]
fn tail_reasoning_is_kept_matches_official_encoding() {
    let mut tail = assistant("Hi.", Vec::new());
    tail.reasoning = "TAIL-THOUGHT-MARKER".into();
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hi."),
        tail,
    ];
    check("enc-think-tail-reasoning", &msgs, "", ThinkMode::High);
}

/// Two assistant messages in a row share one `<Assistant>` header, so the
/// second has no transition token to open its thinking.  The oracle still
/// writes that message's reasoning and its closer inline, because
/// `thinking_part` belongs to the message body and not to the transition token
/// (encoding.py render_message:678-712).
#[test]
fn consecutive_assistant_reasoning_matches_official_encoding() {
    let mut second = assistant("Two.", Vec::new());
    second.reasoning = "SECOND-THOUGHT-MARKER".into();
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hi."),
        assistant("One.", Vec::new()),
        second,
    ];
    check("enc-think-two-assistants", &msgs, "", ThinkMode::High);
}
