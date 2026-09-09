//! Builds a `minijinja::Environment` configured to match the `transformers` chat-template
//! environment: HF globals, the Python-compatible `tojson`, pycompat method shim, the
//! `strftime_now` clock hook, and the whitespace-control flags.

use std::borrow::Cow;
use std::sync::Arc;

use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior, Value};

use crate::clock::Clock;
use crate::error::RAISE_SENTINEL;
use crate::json::tojson_filter;

/// Knobs that affect rendering semantics. Defaults mirror the `transformers` reference.
#[derive(Clone)]
pub(crate) struct EngineConfig {
    pub trim_blocks: bool,
    pub lstrip_blocks: bool,
    pub keep_trailing_newline: bool,
    pub undefined: UndefinedBehavior,
    pub pycompat: bool,
    pub clock: Arc<dyn Clock>,
}

/// Install all HF-compat globals, filters and settings onto a fresh environment, then add the
/// template source under the fixed name `"chat"`.
///
/// `'static` source: we own the template string for the lifetime of the environment, so we
/// build an `Environment<'static>` by handing it an owned `String`.
pub(crate) fn build(source: String, cfg: &EngineConfig) -> Result<Environment<'static>, Error> {
    let mut env = Environment::new();

    env.set_trim_blocks(cfg.trim_blocks);
    env.set_lstrip_blocks(cfg.lstrip_blocks);
    env.set_keep_trailing_newline(cfg.keep_trailing_newline);
    env.set_undefined_behavior(cfg.undefined);

    // Chat templates are not HTML; escaping would corrupt the prompt. Force no auto-escape.
    env.set_auto_escape_callback(|_name| AutoEscape::None);

    // Python string/list/dict methods (.strip(), .split(), .items(), ...). The shim lives in
    // minijinja-contrib, which is an optional dependency gated by the `pycompat` feature.
    let _pycompat = cfg.pycompat;
    #[cfg(feature = "pycompat")]
    if _pycompat {
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    }

    // raise_exception(msg): abort the render carrying `msg`, marked so we can recover it as a
    // distinct TemplateRaised error (not an engine bug). The sentinel never reaches output.
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("{RAISE_SENTINEL}{msg}{RAISE_SENTINEL}"),
        ))
    });

    // strftime_now(fmt): current date/time formatted via the injected clock.
    let clock = cfg.clock.clone();
    env.add_function("strftime_now", move |fmt: String| -> Result<Value, Error> {
        Ok(Value::from(clock.now().strftime(&fmt)))
    });

    // Python-compatible tojson (overrides minijinja's sorted/space-less builtin).
    env.add_filter("tojson", tojson_filter);

    // Tool schemas also print defaults outside JSON. Keep plain float output
    // and the explicit string filter consistent with Python's shortest repr.
    env.set_formatter(|out, state, value| {
        if value.is_number() && !value.is_integer() {
            let text = crate::json::float_repr(f64::try_from(value.clone())?);
            return minijinja::escape_formatter(out, state, &Value::from(text));
        }
        minijinja::escape_formatter(out, state, value)
    });
    env.add_filter("string", |value: Value| -> Result<String, Error> {
        if value.is_number() && !value.is_integer() {
            return Ok(crate::json::float_repr(f64::try_from(value)?));
        }
        Ok(value.to_string())
    });

    let source = match neutralize_generation_tags(&source) {
        Cow::Borrowed(_) => source,
        Cow::Owned(rewritten) => rewritten,
    };
    env.add_template_owned("chat", source)?;

    Ok(env)
}

/// Rewrite `transformers`' `{% generation %}` / `{% endgeneration %}` block into an unconditional
/// `{% if true %}` / `{% endif %}` before compilation.
///
/// `transformers` registers a `generation` block (its `AssistantTracker` Jinja extension) to record
/// which output bytes are assistant-generated, for `return_assistant_tokens_mask=True`. minijinja
/// has no custom-statement API and rejects the tag (`unknown statement generation`). The block has
/// no effect on the rendered string — it only marks token offsets — so rewriting it to an
/// always-true `if` reproduces the body verbatim. Both are block tags, and `trim_blocks` /
/// `lstrip_blocks` act at the lexer level before tag dispatch, so the surrounding whitespace is
/// handled identically. Any whitespace-control markers (`-` / `+`) on the original tag are carried
/// over so trimming stays byte-identical.
///
/// Returns `Cow::Borrowed` (no allocation) when the source contains no such tag — the common case.
fn neutralize_generation_tags(source: &str) -> Cow<'_, str> {
    let mut out: Option<String> = None;
    let mut flushed = 0; // bytes of `source` already copied into `out`
    let mut cursor = 0; // scan position

    while let Some(rel) = source[cursor..].find("{%") {
        let open = cursor + rel;
        let Some(crel) = source[open + 2..].find("%}") else {
            break; // no closing delimiter; let minijinja report the syntax error
        };
        let close = open + 2 + crel + 2; // index just past "%}"
        let inner = &source[open + 2..close - 2]; // text between "{%" and "%}"

        // Peel optional whitespace-control markers, then match the bare keyword.
        let lead = inner.starts_with(['-', '+']);
        let trail = inner.ends_with(['-', '+']);
        let keyword = inner
            .trim_start_matches(['-', '+'])
            .trim_end_matches(['-', '+'])
            .trim();

        let replacement_kw = match keyword {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };

        if let Some(kw) = replacement_kw {
            let buf = out.get_or_insert_with(String::new);
            buf.push_str(&source[flushed..open]);
            buf.push_str("{%");
            if lead {
                buf.push_str(&inner[..1]);
            }
            buf.push(' ');
            buf.push_str(kw);
            buf.push(' ');
            if trail {
                buf.push_str(&inner[inner.len() - 1..]);
            }
            buf.push_str("%}");
            flushed = close;
        }
        cursor = close;
    }

    match out {
        Some(mut buf) => {
            buf.push_str(&source[flushed..]);
            Cow::Owned(buf)
        }
        None => Cow::Borrowed(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> EngineConfig {
        EngineConfig {
            trim_blocks: true,
            lstrip_blocks: true,
            keep_trailing_newline: false,
            undefined: UndefinedBehavior::Lenient,
            pycompat: false,
            clock: Arc::new(crate::clock::SystemClock),
        }
    }

    #[test]
    fn leaves_sources_without_the_tag_untouched() {
        // "generation" appears as a substring (add_generation_prompt) but there is no block tag,
        // so the source must be returned borrowed (no allocation, no rewrite).
        let src = "{% if add_generation_prompt %}<gen>{% endif %}";
        assert!(matches!(neutralize_generation_tags(src), Cow::Borrowed(_)));
    }

    #[test]
    fn rewrites_plain_and_marked_tags() {
        assert_eq!(
            neutralize_generation_tags("a{% generation %}b{% endgeneration %}c"),
            "a{% if true %}b{% endif %}c"
        );
        // Whitespace-control markers are preserved on both ends.
        assert_eq!(
            neutralize_generation_tags("a{%- generation -%}b{%- endgeneration -%}c"),
            "a{%- if true -%}b{%- endif -%}c"
        );
        // Tight delimiters with no inner spaces.
        assert_eq!(
            neutralize_generation_tags("{%generation%}x{%endgeneration%}"),
            "{% if true %}x{% endif %}"
        );
    }

    #[test]
    fn rewritten_template_renders_body_byte_identically() {
        // The generation block must render exactly like a hand-written always-true if, including
        // the whitespace trim_blocks applies to a block tag sitting on its own line.
        let with_gen = "x\n{% generation %}\nbody\n{% endgeneration %}\ny\n".to_string();
        let with_if = "x\n{% if true %}\nbody\n{% endif %}\ny\n".to_string();
        let cfg = default_cfg();
        let gen_env = build(with_gen, &cfg).expect("generation block compiles after rewrite");
        let if_env = build(with_if, &cfg).expect("control compiles");
        let ctx = Value::from(());
        let gen_out = gen_env.get_template("chat").unwrap().render(&ctx).unwrap();
        let if_out = if_env.get_template("chat").unwrap().render(&ctx).unwrap();
        assert_eq!(gen_out, if_out);
    }
}
