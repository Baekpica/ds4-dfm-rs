//! Guards on the 1.0 public API. Each of these is a promise that cannot be walked back without a
//! 2.0, so they are asserted rather than left to review.
//!
//! Note what this file does *not* import: `minijinja`. Everything below compiles without naming the
//! engine, which is the point of the 1.0 surface. If a future change leaks an engine type into a
//! signature, callers would be forced to depend on a matching engine version, and an engine major
//! release would drag this crate's major along with it.

use std::error::Error as StdError;

use hf_chat_template::{
    ChatTemplate, ChatTemplateBuilder, Civil, Clock, Content, Error, FixedClock, Message,
    RenderInput, SystemClock, TokenizerConfig,
};
use serde_json::json;

fn assert_send_sync_static<T: Send + Sync + 'static>() {}

/// `ChatTemplate` is documented as compile-once, share-everywhere. Server callers put it in an
/// `Arc` and render from many threads, which only works while these auto traits hold.
#[test]
fn public_types_are_send_sync_static() {
    assert_send_sync_static::<ChatTemplate>();
    assert_send_sync_static::<ChatTemplateBuilder>();
    assert_send_sync_static::<Error>();
    assert_send_sync_static::<RenderInput>();
    assert_send_sync_static::<Message>();
    assert_send_sync_static::<Content>();
    assert_send_sync_static::<TokenizerConfig>();
    assert_send_sync_static::<Civil>();
}

/// `?` into `anyhow`/`eyre` and `Box<dyn Error + Send + Sync>` is the normal way callers propagate
/// our errors. That requires the error to be `Send + Sync + 'static`, so pin it.
#[test]
fn error_boxes_like_a_normal_error() {
    let err = ChatTemplate::from_str("{% for x in %}").unwrap_err();
    let boxed: Box<dyn StdError + Send + Sync + 'static> = Box::new(err);
    assert!(boxed.to_string().contains("failed to compile"));
}

/// The engine's detail must survive being wrapped in an opaque type: message via `Display`, and
/// the chain via `source()`. Losing either would make compile errors undebuggable.
#[test]
fn template_error_keeps_engine_detail() {
    let err = ChatTemplate::from_str("{% for x in %}").unwrap_err();
    let Error::Compile(inner) = &err else {
        panic!("expected Compile, got {err:?}");
    };
    assert!(
        !inner.to_string().is_empty(),
        "the engine message must survive wrapping"
    );
    assert!(
        inner.line().is_some(),
        "a syntax error should report its line"
    );
    assert!(
        err.source().is_some(),
        "Error must chain to the underlying template error"
    );
}

/// A third-party clock must be writable without reimplementing strftime. If this ever stops
/// compiling, the `Clock` trait has grown a requirement that breaks external implementors.
#[test]
fn a_custom_clock_is_three_lines() {
    struct PinnedClock;
    impl Clock for PinnedClock {
        fn now(&self) -> Civil {
            Civil::from_ymd_hms(2026, 6, 13, 9, 30, 0).expect("valid date")
        }
    }

    let tmpl = ChatTemplate::builder("{{ strftime_now('%d %B %Y %H:%M') }}")
        .clock(PinnedClock)
        .build()
        .unwrap();
    // Formatting is the crate's, so a custom clock cannot drift from the reference.
    assert_eq!(
        tmpl.render_context(&json!({})).unwrap(),
        "13 June 2026 09:30"
    );
}

/// The shipped clocks and a hand-rolled one must format identically, since formatting lives in one
/// place. This is what makes the byte-identical claim safe under a custom clock.
#[test]
fn shipped_and_custom_clocks_format_identically() {
    struct SameInstant;
    impl Clock for SameInstant {
        fn now(&self) -> Civil {
            Civil::from_ymd_hms(2024, 7, 4, 0, 0, 0).expect("valid date")
        }
    }
    let fmt = "{{ strftime_now('%A %d %B %Y %j') }}";
    let render = |clock: Box<dyn Clock>| {
        ChatTemplate::builder(fmt)
            .clock(clock)
            .build()
            .unwrap()
            .render_context(&json!({}))
            .unwrap()
    };
    assert_eq!(
        render(Box::new(FixedClock::from_ymd(2024, 7, 4).unwrap())),
        render(Box::new(SameInstant)),
    );
    // SystemClock is exercised for wiring only; its value is the wall clock, so nothing is asserted
    // about the instant itself.
    let _ = SystemClock.now();
}

/// `render_context` accepts anything `Serialize`, not just a JSON value, and preserves key order
/// through `tojson` — the property the whole corpus rests on.
#[test]
fn render_context_takes_any_serialize_and_keeps_key_order() {
    #[derive(serde::Serialize)]
    struct Ctx<'a> {
        name: &'a str,
    }
    let tmpl = ChatTemplate::from_str("Hi {{ name }}").unwrap();
    assert_eq!(tmpl.render_context(&Ctx { name: "Ada" }).unwrap(), "Hi Ada");

    let tmpl = ChatTemplate::from_str("{{ t | tojson }}").unwrap();
    let out = tmpl
        .render_context(&json!({ "t": { "z": 1, "a": 2 } }))
        .unwrap();
    assert_eq!(out, r#"{"z": 1, "a": 2}"#, "insertion order must survive");
}

/// Role constructors cover the four roles a chat template branches on.
#[test]
fn role_constructors_cover_the_chat_roles() {
    assert_eq!(Message::system("s").role, "system");
    assert_eq!(Message::user("u").role, "user");
    assert_eq!(Message::assistant("a").role, "assistant");
    assert_eq!(Message::tool("t").role, "tool");
}
