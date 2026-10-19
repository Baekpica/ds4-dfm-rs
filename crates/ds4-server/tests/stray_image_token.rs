//! A user-supplied image placeholder in a textual field is refused.
//!
//! The carriage's authority (the oracle, encoding.py:355-367) refuses the
//! literal `<|deepseek_image|>` in `content` and in `reasoning_content`, and
//! refuses it inside a text content block (encoding.py:346-352, reachable
//! through nested tool-result content too).  An image is an image content
//! block; the literal in text is not a way to place one.
//!
//! Without the refusal the literal tokenizes to the placeholder id
//! (crates/ds4-core/src/tok.rs:2603) and a request that carries no image never
//! reaches the V4.1 expansion that counts placeholders
//! (crates/ds4-server/src/generate.rs:3107), so the model would be handed a
//! lone image token at a text position.

use ds4_server::render::DSML_IMAGE;
use ds4_server::{parse_anthropic_request, parse_chat_request, parse_responses_request, ParseEnv};

fn env() -> ParseEnv {
    ParseEnv::default()
}

fn refused(err: String, field: &str) -> bool {
    err.contains(DSML_IMAGE) && err.contains(field)
}

#[test]
fn openai_string_content_is_refused() {
    let body =
        format!(r#"{{"messages":[{{"role":"user","content":"look at {DSML_IMAGE} this"}}]}}"#);
    let err = parse_chat_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn openai_text_block_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"look at {DSML_IMAGE} this"}}]}}]}}"#
    );
    let err = parse_chat_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn openai_reasoning_content_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"assistant","reasoning_content":"thought {DSML_IMAGE} more","content":"hi"}}]}}"#
    );
    let err = parse_chat_request(&env(), &body).unwrap_err();
    assert!(
        refused(err.clone(), "reasoning_content"),
        "unexpected error: {err}"
    );
}

#[test]
fn openai_tool_result_text_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"shot","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c1","content":"shot {DSML_IMAGE} taken"}}]}}"#
    );
    let err = parse_chat_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn anthropic_text_block_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"look at {DSML_IMAGE} this"}}]}}]}}"#
    );
    let err = parse_anthropic_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn anthropic_thinking_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"assistant","content":[{{"type":"thinking","thinking":"thought {DSML_IMAGE} more"}}]}}]}}"#
    );
    let err = parse_anthropic_request(&env(), &body).unwrap_err();
    assert!(
        refused(err.clone(), "reasoning_content"),
        "unexpected error: {err}"
    );
}

#[test]
fn anthropic_tool_result_text_is_refused() {
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"t1","content":"shot {DSML_IMAGE} taken"}}]}}]}}"#
    );
    let err = parse_anthropic_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn responses_input_text_is_refused() {
    let body = format!(
        r#"{{"input":[{{"type":"message","role":"user","content":[{{"type":"input_text","text":"look at {DSML_IMAGE} this"}}]}}]}}"#
    );
    let err = parse_responses_request(&env(), &body).unwrap_err();
    assert!(refused(err.clone(), "content"), "unexpected error: {err}");
}

#[test]
fn the_servers_own_placeholders_still_parse() {
    // The server writes the placeholder itself for a tool message whose image
    // is the pixels (parse.rs write_tool_image_placeholders): that insertion
    // happens after the client's text has been validated and must not be
    // mistaken for one.
    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );
    let body = format!(
        r#"{{"messages":[{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"shot","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c1","content":[{{"type":"text","text":"shot:"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{PNG}"}}}},{{"type":"text","text":"done"}}]}}]}}"#
    );
    let parsed = parse_chat_request(&env(), &body).unwrap();
    assert_eq!(parsed.messages[1].content, format!("shot:{DSML_IMAGE}done"));

    // A plain text turn is untouched, and a user image block still admits.
    let plain = r#"{"messages":[{"role":"user","content":"say hello"}]}"#;
    assert!(parse_chat_request(&env(), plain).is_ok());
    let with_image = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"what colour"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{PNG}"}}}}]}}]}}"#
    );
    assert!(parse_chat_request(&env(), &with_image).is_ok());
}
