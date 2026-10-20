//! The constants leg: every literal the two sides must agree on, read out of
//! the vendored oracle's own source instead of copied by hand.
//!
//! `docs/ds41-dsml-function-map.md` section 3 carries one row per literal of
//! `tests/ds41_ref/oracle/encoding.py`.  A row that says "pass" because a case
//! happens to contain the literal is weaker than it reads: the case pins the
//! bytes of that one prompt, not the constant.  This test closes that gap for
//! the literals that have a counterpart in this crate, by extracting the value
//! the oracle assigns and comparing it with ours -- so a literal that moves on
//! either side fails here with both spellings, and a hand-copied constant can
//! no longer drift from the file it claims to mirror.
//!
//! The extracted literals are the oracle's: the same file
//! `tests/ds41_ref/ref_drive.py` pins by sha256, so a swapped oracle is caught
//! by the reference legs first and by this test second.

use ds4_server::render::{
    DSML_ASSISTANT, DSML_BOS, DSML_EOS, DSML_IMAGE, DSML_SYSTEM_TOKEN, DSML_THINK_END,
    DSML_THINK_START, DSML_TOOL_CALLS, DSML_USER, V41_EFFORT_TEMPLATE, V41_THINK_HIGH_PREFIX,
    V41_THINK_MAX_PREFIX,
};
use ds4_server::tools::{
    DSML_INVOKE_END_SPACED, DSML_INVOKE_START_SPACED, DSML_PARAM_END_SPACED,
    DSML_PARAM_START_SPACED, DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START_SPACED,
};

/// The oracle's bytes, compiled in.  Reading the same file the driver pins is
/// the point: the test compares against the authority, not against a copy.
const ORACLE: &str = include_str!("../../../tests/ds41_ref/oracle/encoding.py");

/// The source text of the statement that declares `name`.
///
/// A python declaration ends at the end of the line that closes the last open
/// bracket, unless the line continues with `+` (the oracle composes
/// `assistant_msg_template` that way).  Bracket depth is counted outside
/// string literals, because the effort template carries a `(range 1-100)`
/// parenthesis inside its own text.
fn declaration(name: &str) -> String {
    let mut at = None;
    let mut offset = 0;
    for line in ORACLE.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('#') {
            if let Some(rest) = trimmed.strip_prefix(name) {
                if rest.starts_with(':') || rest.starts_with(' ') {
                    at = Some(offset + (line.len() - trimmed.len()));
                    break;
                }
            }
        }
        offset += line.len();
    }
    let src = &ORACLE[at.unwrap_or_else(|| panic!("the oracle no longer declares {name}"))..];

    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    let mut end = src.len();
    for (i, c) in src.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_str {
            match c {
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth -= 1,
            '\n' if depth <= 0 => {
                let line = src[..i].rsplit('\n').next().unwrap_or("");
                let open = line.trim_end().ends_with('+') || line.trim_end().ends_with('\\');
                if !open {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    src[..end].to_string()
}

/// `name`'s value, with adjacent string literals concatenated as python does
/// and the escapes decoded.  A declaration that carries no string literal is a
/// test bug, not an oracle change, so it panics.
fn literal(name: &str) -> String {
    let text = declaration(name);
    let mut out = String::new();
    let mut chars = text.chars();
    let mut seen = false;
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        seen = true;
        while let Some(c) = chars.next() {
            match c {
                '\\' => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => break,
                },
                '"' => break,
                other => out.push(other),
            }
        }
    }
    assert!(seen, "{name} no longer holds a string literal");
    out
}

/// One integer member of an oracle dict, e.g. `REASONING_EFFORT_MAPPINGS["low"]`.
fn mapping_int(name: &str, key: &str) -> i64 {
    let text = declaration(name);
    let needle = format!("\"{key}\":");
    let rest = text
        .split_once(&needle)
        .unwrap_or_else(|| panic!("{name} no longer maps {key}"))
        .1;
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    digits
        .parse()
        .unwrap_or_else(|e| panic!("{name}[{key}] is not an integer: {e}"))
}

/// The five special tokens our renderer writes (encoding.py:29-39, :426).
#[test]
fn special_tokens_match_the_oracle() {
    assert_eq!(DSML_BOS, literal("bos_token"), "bos_token");
    assert_eq!(DSML_EOS, literal("eos_token"), "eos_token");
    assert_eq!(
        DSML_THINK_START,
        literal("thinking_start_token"),
        "thinking_start_token"
    );
    assert_eq!(
        DSML_THINK_END,
        literal("thinking_end_token"),
        "thinking_end_token"
    );
    assert_eq!(DSML_USER, literal("USER_SP_TOKEN"), "USER_SP_TOKEN");
    assert_eq!(
        DSML_ASSISTANT,
        literal("ASSISTANT_SP_TOKEN"),
        "ASSISTANT_SP_TOKEN"
    );
    assert_eq!(
        DSML_SYSTEM_TOKEN,
        literal("SYSTEM_SP_TOKEN"),
        "SYSTEM_SP_TOKEN"
    );
    assert_eq!(
        DSML_IMAGE,
        literal("IMAGE_PLACEHOLDER"),
        "IMAGE_PLACEHOLDER"
    );
}

/// The DSML tags, composed from the oracle's own fragments
/// (encoding.py:33, :428-435) so the separators are the oracle's and not
/// ours.
///
/// `DSML_TOOL_CALLS` is not in this set: it is the artifact's older unspaced
/// root (`｜DSML｜tool_calls`, no space before the name), a spelling the oracle
/// does not define and the map's section 4 lists beside the short one.  What
/// matters about it here is that the renderer's block tag and the parser's
/// block tag are the same string: a renderer writing a tag the parser does not
/// recognize would lose the call.
#[test]
fn tool_tags_match_the_oracle() {
    let dsml = literal("dsml_token");
    let block = literal("tool_calls_block_name");
    let call = literal("tool_call_tag_name");
    let param = literal("tool_parameter_tag_name");

    assert_eq!(DSML_TOOL_CALLS_START_SPACED, format!("<{dsml}{block}>"));
    assert_eq!(DSML_TOOL_CALLS_END_SPACED, format!("</{dsml}{block}>"));
    assert_eq!(DSML_INVOKE_START_SPACED, format!("<{dsml}{call}"));
    assert_eq!(DSML_INVOKE_END_SPACED, format!("</{dsml}{call}>"));
    assert_eq!(DSML_PARAM_START_SPACED, format!("<{dsml}{param}"));
    assert_eq!(DSML_PARAM_END_SPACED, format!("</{dsml}{param}>"));

    assert_eq!(DSML_TOOL_CALLS, ds4_server::tools::DSML_TOOL_CALLS_START);
}

/// The effort line and its budget table (encoding.py:443-453).  The 75 and 100
/// constants are the template with the mapping's number substituted, which is
/// what `render_reasoning_effort` does before it formats.
#[test]
fn effort_line_and_budgets_match_the_oracle() {
    let template = literal("REASONING_EFFORT_TEMPLATE");
    assert_eq!(V41_EFFORT_TEMPLATE, template, "REASONING_EFFORT_TEMPLATE");

    let with = |budget: i64| template.replace("{budget}", &budget.to_string());
    assert_eq!(
        V41_THINK_HIGH_PREFIX,
        with(mapping_int("REASONING_EFFORT_MAPPINGS", "high")),
        "high"
    );
    assert_eq!(
        V41_THINK_MAX_PREFIX,
        with(mapping_int("REASONING_EFFORT_MAPPINGS", "max")),
        "max"
    );
    assert_eq!(
        literal("DEFAULT_REASONING_EFFORT"),
        "high",
        "default effort"
    );
}

/// The two templates that define a served turn's tail: the tool-result
/// envelope (encoding.py:68) and the assistant composition (encoding.py:60).
/// The assistant template is `{reasoning}{content}{tool_calls}` plus the EOS
/// token, so its own body is what the renderer must write before the EOS.
#[test]
fn turn_templates_match_the_oracle() {
    assert_eq!(
        literal("tool_output_template"),
        "<tool_result>{content}</tool_result>"
    );

    let assistant = literal("assistant_msg_template");
    assert_eq!(assistant, "{reasoning}{content}{tool_calls}");
    assert!(
        declaration("assistant_msg_template").contains("+ eos_token"),
        "the oracle's assistant template no longer appends the EOS token"
    );
    assert_eq!(DSML_EOS, literal("eos_token"));
}
