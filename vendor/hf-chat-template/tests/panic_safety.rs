//! Panic-safety: adversarial input must surface as an `Err` (or render fine), never unwind.
//!
//! The crate's contract is that no user-supplied template or context can panic the library;
//! every fallible path returns `Result`. These tests throw hostile input at `from_str` (compile)
//! and `render`/`render_context`, and assert the call returns rather than panics. A panic here would
//! fail the test by unwinding; the most suspicious path (our hand-written byte slicing in the
//! `{% generation %}` rewrite) is additionally wrapped in `catch_unwind` for an explicit message.

use std::panic::catch_unwind;

use hf_chat_template::{ChatTemplate, Error, Message, RenderInput};
use serde_json::json;

/// Compiling any of these must return (Ok or Err), never panic. Mixes syntax errors, unbalanced
/// delimiters, and multibyte text wedged against Jinja markers (which stresses byte-index slicing).
#[test]
fn malformed_templates_compile_without_panicking() {
    let nasty = [
        "{% for x in %}",                        // missing iterable
        "{{ }}",                                 // empty expression
        "{%",                                    // dangling block open
        "%}",                                    // dangling block close
        "{{",                                    // dangling expr open
        "{{ unclosed",                           // never closed
        "{% if %}{% endif %}",                   // empty condition
        "{% endif %}",                           // unmatched end
        "{{ a.b.c.d.e.f }}",                     // deep attribute chain, undefined
        "{% generation %}",                      // generation open, no close
        "{% endgeneration %}",                   // generation close, no open
        "héllo {% generation %}wörld",           // multibyte adjacent to a rewritten tag
        "café{%- generation -%}thé",             // multibyte + whitespace-control markers
        "🦀{%generation%}🦀{%endgeneration%}🦀", // 4-byte chars hugging tight tags
        "{%生成%}",                              // multibyte *inside* a block tag
        "{{ '\\u{0}\\u{1}' }}",                  // control chars in a literal
    ];
    for src in nasty {
        // Compile must not panic; result may be Ok or Err.
        let _ = ChatTemplate::from_str(src);
    }
}

/// The `{% generation %}` rewrite (`engine::neutralize_generation_tags`) does hand-written
/// byte-index find/slice. Multibyte text on either side of the tag must not break a char boundary.
/// If any of these compile *successfully*, rendering them must also not panic.
#[test]
fn generation_rewrite_handles_multibyte_without_panicking() {
    let cases = [
        "Ω{% generation %}Ω{% endgeneration %}Ω",
        "{%- generation +%}日本語{%+ endgeneration -%}",
        "a{%generation%}b{%endgeneration%}c",
        "пример {% generation %}\n{{ x }}\n{% endgeneration %} конец",
    ];
    let result = catch_unwind(|| {
        for src in cases {
            if let Ok(tmpl) = ChatTemplate::from_str(src) {
                let _ = tmpl.render_context(&json!({ "x": "ок" }));
            }
        }
    });
    assert!(
        result.is_ok(),
        "generation-tag rewrite panicked on multibyte input"
    );
}

/// Valid templates fed hostile contexts must return a `Result`, never panic: empty collections
/// that templates index, missing variables, and wrong-typed values.
#[test]
fn hostile_render_contexts_do_not_panic() {
    // Index into an empty messages list.
    let t = ChatTemplate::from_str("{{ messages[0]['content'] }}").unwrap();
    let _ = t.render_context(&json!({ "messages": [] }));

    // Reference variables that were never provided.
    let t = ChatTemplate::from_str("{{ totally_undefined.attr }}{{ bos_token }}").unwrap();
    let _ = t.render_context(&json!({}));

    // A number where the template treats it like a string/iterable.
    let t = ChatTemplate::from_str("{% for c in content %}{{ c }}{% endfor %}").unwrap();
    let _ = t.render_context(&json!({ "content": 42 }));

    // Typed path with an empty conversation and a template that assumes a first message.
    let t = ChatTemplate::from_str("{{ messages[0].role }}").unwrap();
    let _ = t.render(&RenderInput::default());
}

/// User-supplied content that merely looks alarming (control bytes, brace-like text) must render
/// as ordinary text, not be mistaken for engine control flow, and must not panic.
#[test]
fn user_content_is_inert_text() {
    let t = ChatTemplate::from_str("{{ messages[0].content }}").unwrap();
    let input = RenderInput {
        messages: vec![Message::user(
            "{% raise_exception('x') %}\u{0}\u{1}{{ leak }}",
        )],
        ..Default::default()
    };
    // Rendered as data: the injected markup is not executed, so output equals the input text.
    let out = t.render(&input).expect("inert content renders");
    assert_eq!(out, "{% raise_exception('x') %}\u{0}\u{1}{{ leak }}");
}

/// Deeply nested control flow must be handled gracefully (Ok or a clean Err), not by overflowing
/// the stack. Kept to a depth that is safe to parse but well past anything a real template uses.
#[test]
fn deep_nesting_is_handled_gracefully() {
    let depth = 300;
    let src = format!(
        "{}{}{}",
        "{% if true %}".repeat(depth),
        "x",
        "{% endif %}".repeat(depth),
    );
    let result = catch_unwind(|| {
        if let Ok(t) = ChatTemplate::from_str(&src) {
            let _ = t.render_context(&json!({}));
        }
    });
    assert!(result.is_ok(), "deep nesting panicked instead of erroring");
}

/// Sanity: a genuine syntax error is still reported as `Compile` (panic-safety must not have
/// turned real errors into silent successes).
#[test]
fn real_syntax_error_still_errors() {
    let err = ChatTemplate::from_str("{% for %}").unwrap_err();
    assert!(matches!(err, Error::Compile(_)), "got {err:?}");
}
