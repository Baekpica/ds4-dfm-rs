//! M0 de-risking spikes. These are throwaway exploratory tests that resolve the
//! load-bearing "VERIFY" items from SPEC.md §15 against the real minijinja 2.20 engine.
//! The compiler + these assertions are ground truth; they gate the go/no-go on the design.
//!
//! Run with: `cargo test --test m0_spikes -- --nocapture`

use minijinja::{context, Environment, Error, Value};

/// §15.1 — THE critical one. Do `namespace()` objects support cross-scope mutation
/// (`{% set ns.x = ... %}`) accumulating across `{% for %}` iterations? Gemma and most
/// tool-calling templates depend on this. If this fails, the engine choice is in question.
#[test]
fn spike_namespace_mutation_across_loop() {
    let mut env = Environment::new();
    let src = "{% set ns = namespace(x=0) %}\
               {% for i in range(4) %}{% set ns.x = ns.x + i %}{% endfor %}\
               {{ ns.x }}";
    env.add_template("t", src).expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! {})
        .expect("render");
    // 0+1+2+3 = 6
    assert_eq!(out, "6", "namespace mutation across loop must accumulate");
}

/// §15.1 cont. — a more realistic "found_first" flag pattern straight out of real templates.
#[test]
fn spike_namespace_bool_flag_pattern() {
    let mut env = Environment::new();
    let src = "{% set ns = namespace(found=false) %}\
               {% for m in items %}\
               {%- if m == 'system' and not ns.found %}{% set ns.found = true %}SYS:{{ m }};{% endif -%}\
               {% endfor %}\
               found={{ ns.found }}";
    env.add_template("t", src).expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! { items => vec!["user", "system", "system"] })
        .expect("render");
    assert_eq!(out, "SYS:system;found=True");
}

/// §15.3 — pycompat: register the unknown-method callback and confirm Python string methods
/// that templates actually call (`strip`, `split`, `startswith`, `title`, `endswith`).
///
/// Gated: `minijinja-contrib` is an optional dependency, so without the `pycompat` feature this
/// test would not compile and would take the whole `--no-default-features` test run with it.
#[cfg(feature = "pycompat")]
#[test]
fn spike_pycompat_string_methods() {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template(
        "t",
        "{{ '  hi '.strip() }}|{{ 'a,b,c'.split(',') | join('-') }}|\
         {{ 'hello'.startswith('he') }}|{{ 'hello world'.title() }}|\
         {{ 'foo.txt'.endswith('.txt') }}",
    )
    .expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! {})
        .expect("render");
    assert_eq!(out, "hi|a-b-c|True|Hello World|True");
}

/// §15.4 — does the `json` feature give us a `tojson` filter, and what does it emit?
/// We need to know its default formatting to compare against Python's tojson later.
#[test]
fn spike_tojson_filter_output() {
    let mut env = Environment::new();
    env.add_template("t", "{{ obj | tojson }}")
        .expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! { obj => context!{ b => 1, a => "x" } })
        .expect("render");
    // Print so we can eyeball separators / key ordering / spacing vs Python.
    println!("TOJSON_OUTPUT='{out}'");
    assert!(out.contains("\"a\""), "tojson should serialize keys");
}

/// §15.2 / §15.6 — confirm the env whitespace-control knobs exist with these names and
/// observably change output (this is the transformers baseline we must match).
#[test]
fn spike_trim_and_lstrip_blocks_exist() {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // With trim_blocks, the newline after a block tag is removed.
    env.add_template("t", "{% if true %}\nX{% endif %}")
        .expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! {})
        .expect("render");
    println!("TRIM_OUTPUT='{out}'");
    // Just asserting it renders and the knobs compile; exact transformers parity verified in corpus.
    assert!(out.contains('X'));
}

/// §15.5 / §6.2 — `raise_exception(msg)` registered as a global function that aborts the
/// render with an error carrying the message. Confirm add_function + error propagation,
/// and that we can recover the message from the returned Error.
#[test]
fn spike_raise_exception_global() {
    let mut env = Environment::new();
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template(
        "t",
        "{% if not ok %}{{ raise_exception('bad conversation') }}{% endif %}done",
    )
    .expect("compile");

    // ok=true: no raise
    let good = env
        .get_template("t")
        .unwrap()
        .render(context! { ok => true })
        .expect("render");
    assert_eq!(good, "done");

    // ok=false: must error, and the message must be recoverable somewhere in the chain.
    let err = env
        .get_template("t")
        .unwrap()
        .render(context! { ok => false })
        .unwrap_err();
    println!("RAISE_ERR kind={:?} display='{}'", err.kind(), err);
    let mut found = err.to_string().contains("bad conversation");
    let mut src = std::error::Error::source(&err);
    while let Some(s) = src {
        if s.to_string().contains("bad conversation") {
            found = true;
        }
        src = s.source();
    }
    assert!(
        found,
        "raise_exception message must be recoverable from the error chain"
    );
}

/// §3.6 — loop variables used pervasively (loop.index0, loop.last, loop.first).
#[test]
fn spike_loop_variables() {
    let mut env = Environment::new();
    env.add_template(
        "t",
        "{% for m in items %}{{ loop.index0 }}:{{ m }}{% if not loop.last %},{% endif %}{% endfor %}",
    )
    .expect("compile");
    let out = env
        .get_template("t")
        .unwrap()
        .render(context! { items => vec!["a", "b", "c"] })
        .expect("render");
    assert_eq!(out, "0:a,1:b,2:c");
}

/// §3 — the string-or-list content branch (`content is string`) that multimodal templates use.
#[test]
fn spike_content_is_string_test() {
    let mut env = Environment::new();
    env.add_template(
        "t",
        "{% if content is string %}STR:{{ content }}{% else %}\
         {% for p in content %}PART:{{ p.text }};{% endfor %}{% endif %}",
    )
    .expect("compile");

    let s = env
        .get_template("t")
        .unwrap()
        .render(context! { content => "hello" })
        .expect("render string case");
    assert_eq!(s, "STR:hello");

    let parts = vec![context! { text => "a" }, context! { text => "b" }];
    let l = env
        .get_template("t")
        .unwrap()
        .render(context! { content => parts })
        .expect("render list case");
    assert_eq!(l, "PART:a;PART:b;");
}
