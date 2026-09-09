//! Trailing-newline handling must match the `transformers` reference exactly.
//!
//! Jinja2's `keep_trailing_newline` defaults to `false` and `transformers` never overrides it, so
//! the reference strips exactly one newline from the end of the template source. We set the same
//! default. This is easy to get wrong in the invisible direction: a stray `\n` at the end of every
//! prompt does not look like a bug, it just silently is one.
//!
//! Real templates mask the difference, which is why this test exists rather than a corpus model:
//! they all end in a block tag, and `trim_blocks` eats that newline first, so both settings agree.
//! A template ending in `{{ expr }}\n` is what exposes it.
//!
//! Expected values are the literal output of Python
//! `tok.apply_chat_template(msgs, chat_template=<src>, tokenize=False)` on `transformers` 5.14.1
//! (`jinja2` 3.1.6), with `msgs = [{"role": "user", "content": "hi"}]`.

use hf_chat_template::{ChatTemplate, Message};

/// `(name, template source, expected output)` — expected values captured from Python.
const CASES: &[(&str, &str, &str)] = &[
    // The everyday shape: newline belongs to the loop body, so it survives.
    (
        "expr_then_newline",
        "{% for m in messages %}{{ m.content }}\n{% endfor %}",
        "hi\n",
    ),
    // Bare text ending in a newline: the one trailing newline is stripped.
    ("bare_text_newline", "start\n", "start"),
    // The shape that exposed the bug: an expression, then the source's final newline.
    ("expr_newline_end", "A{{ messages[0].content }}\n", "Ahi"),
    // Ends in a block tag, so trim_blocks consumes the newline before the setting can matter.
    // This is why all 20 corpus models agreed under either setting.
    (
        "block_then_newline",
        "{% for m in messages %}{{ m.content }}{% endfor %}\n",
        "hi",
    ),
    // Only *one* newline is stripped, never more.
    ("two_newlines", "A{{ messages[0].content }}\n\n", "Ahi\n"),
    // Nothing to strip; output is untouched.
    ("no_trailing", "A{{ messages[0].content }}", "Ahi"),
    // A trailing CRLF is stripped whole, matching Jinja2's newline handling.
    ("crlf_end", "A{{ messages[0].content }}\r\n", "Ahi"),
    // The newline is not final (a space follows), so nothing is stripped.
    ("trailing_space", "A{{ messages[0].content }}\n ", "Ahi\n "),
    // Degenerate: a source that is only a newline renders empty.
    ("only_newline", "\n", ""),
];

#[test]
fn trailing_newline_matches_the_python_reference() {
    let messages = [Message::user("hi")];
    for (name, source, expected) in CASES {
        let rendered = ChatTemplate::from_str(source)
            .unwrap_or_else(|e| panic!("{name}: template failed to compile: {e}"))
            .render_messages(&messages, false)
            .unwrap_or_else(|e| panic!("{name}: render failed: {e}"));
        assert_eq!(
            &rendered, expected,
            "{name}: diverged from transformers (source {source:?})"
        );
    }
}

/// The default must stay `false`; opting back in is possible but is not the reference behavior.
/// Guards against someone "fixing" a missing newline by flipping the global default.
#[test]
fn keeping_the_trailing_newline_is_not_the_default() {
    let rendered = ChatTemplate::from_str("A{{ messages[0].content }}\n")
        .unwrap()
        .render_messages(&[Message::user("hi")], false)
        .unwrap();
    assert_eq!(
        rendered, "Ahi",
        "the default must strip one trailing newline"
    );
}
