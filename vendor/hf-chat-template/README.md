# hf-chat-template

[![CI](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml/badge.svg)](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hf-chat-template.svg)](https://crates.io/crates/hf-chat-template)
[![docs.rs](https://img.shields.io/docsrs/hf-chat-template)](https://docs.rs/hf-chat-template)
[![license](https://img.shields.io/crates/l/hf-chat-template.svg)](#license)

Render a Hugging Face `chat_template` into a prompt string, byte-for-byte identical to Python's
`transformers.apply_chat_template`. The template is the Jinja2 string a model ships, either inside
`tokenizer_config.json` or as a standalone `chat_template.jinja` file.

```toml
[dependencies]
hf-chat-template = "1.0"
```

## Example

```rust
use hf_chat_template::{ChatTemplate, Message};

let tmpl = ChatTemplate::from_str(
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
     {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
)?;

let prompt = tmpl.render_messages(&[Message::user("Hello!")], true)?;
assert_eq!(prompt, "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
# Ok::<(), hf_chat_template::Error>(())
```

Load a real model's config to inject its special tokens and resolve named templates:

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

let json = std::fs::read_to_string("tokenizer_config.json")?;
let cfg: TokenizerConfig = serde_json::from_str(&json)?;
let tmpl = ChatTemplate::from_tokenizer_config(&cfg)?;

let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Newer models ship the template as a standalone `chat_template.jinja` file instead of inside
`tokenizer_config.json`. Load that with `from_template_and_config`, passing the template string
and the config the special tokens come from.

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

let jinja = std::fs::read_to_string("chat_template.jinja")?;
let cfg: TokenizerConfig = serde_json::from_str(&std::fs::read_to_string("tokenizer_config.json")?)?;
let tmpl = ChatTemplate::from_template_and_config(&jinja, &cfg)?;

let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For tools, documents, or model-specific kwargs, build a [`RenderInput`]. For an arbitrary
context, call `render_context` with anything that implements `Serialize`, such as a
`serde_json::Value` or your own struct.

## What it does

The Jinja engine is [`minijinja`](https://crates.io/crates/minijinja). This crate adds the
`transformers` compatibility layer on top of it, plus a corpus that checks byte-identical output
against real models on every commit.

It installs the globals that templates use: `raise_exception`, `strftime_now`, and a
Python-compatible `tojson` that matches the separators and key order of
`json.dumps(..., ensure_ascii=False)`. Python string, list, and dict methods come from `pycompat`.

It handles both `chat_template` shapes (a single string, or a named list with entries like
`tool_use` and `rag`), special-token injection, and the string-or-parts multimodal `content`.

It emits a prompt string. Turning that into token IDs stays the caller's job (`tokenizers`,
`tiktoken-rs`).

## Verified compatibility

These models render byte-identical to `transformers` in CI. See
[`COMPATIBILITY.md`](COMPATIBILITY.md) and the [corpus](tests/corpus/).

| Model | Notes |
|---|---|
| Qwen2.5, Qwen3, QwQ-32B | ChatML, tool calling (`tojson`), reasoning |
| Llama-3.1 | `<\|start_header_id\|>` format, tools in the user turn, `date_string` |
| SmolLM2 | ChatML |
| Phi-3 | `<\|user\|>` / `<\|end\|>` markers |
| Hermes-3-Llama-3.1 | named `tool_use` sub-template, Jinja macros and recursion |
| Mistral-7B-Instruct-v0.3 | `[INST]` / `[AVAILABLE_TOOLS]`, tool calling |
| DeepSeek-R1-Distill, deepseek-llm | reasoning (`<think>`), `User:` / `Assistant:` |
| OpenChat-3.5, Zephyr, Yi-1.5, Falcon | varied prompt formats, pycompat methods |
| LFM2 | standalone `chat_template.jinja` file, tool list (`tojson`) |
| SmolLM3 | standalone file, `{% generation %}` reasoning block |
| Granite-3.1 | `strftime_now` date stamp (clock pinned for a reproducible match) |
| Gemma-2, Gemma-3 | `<start_of_turn>` format, no system role (Gemma-2), system merge and content parts (Gemma-3) |
| Command-R | named `default` / `tool_use` / `rag` templates, Jinja macros and recursion |

Twenty models, sixty-eight cases, all byte-identical in CI.

## Loading from the Hub

The `hub` feature adds `from_hub`, which fetches a model's config and template and compiles it in
one call. It loads `tokenizer_config.json`, plus a standalone `chat_template.jinja` when the model
ships one (the standalone file wins over an inline `chat_template`, matching `transformers`). It
uses the synchronous `hf-hub` client with rustls, so there is no system OpenSSL dependency.
Authentication follows `hf-hub`: the `HF_TOKEN` env var or the token from `huggingface-cli login`,
which gated repos need.

```toml
hf-chat-template = { version = "1.0", features = ["hub"] }
```

```rust,no_run
use hf_chat_template::{ChatTemplate, Message};

let tmpl = ChatTemplate::from_hub("Qwen/Qwen2.5-0.5B-Instruct")?;
let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), hf_chat_template::Error>(())
```

## Tokenizing

This crate emits a prompt string. It does not tokenize, and it does not depend on a tokenizer
crate. Encode the rendered prompt with your own tokenizer, and pass `add_special_tokens = false`,
because the template already emits the model's special tokens (`bos_token`, end-of-turn markers).
That is what `transformers.apply_chat_template(..., tokenize=True)` does, and it avoids a doubled
BOS.

```rust,ignore
let prompt = tmpl.render(&input)?;
let ids = tokenizer.encode(&prompt, false)?.get_ids().to_vec(); // tokenizers crate
```

## Feature flags

`pycompat` is on by default. It adds Python methods on values (`.strip()`, `.split()`, `| items`)
through `minijinja-contrib`. Disable it to drop that dependency when your templates do not use
those methods.

`hub` is off by default. It adds `from_hub` and `from_hub_revision`, pulling in `hf-hub` and a
TLS stack that the core string-rendering path does not need.

`strftime` is off by default. It adds `LocalClock`, a `strftime_now` clock that reads local wall
time to match `transformers`, pulling in `chrono`. The default `SystemClock` is UTC and needs no
extra dependency.

## Minimum supported Rust version

The MSRV is declared in `Cargo.toml` (`rust-version`) and checked in CI. Raising it is treated as
a breaking change. It tracks what the dependency tree requires, not this crate's own code.

## Upgrading from 0.3

Version 1.0 removes every third-party type from the public API, so no dependency of this crate can
force a major version on you. Three things changed.

`render_value`, which took a `minijinja::Value`, is now `render_context`, which takes anything that
implements `Serialize`.

```rust
use hf_chat_template::ChatTemplate;
use serde_json::json;

let tmpl = ChatTemplate::from_str("Hi {{ name }}")?;
let out = tmpl.render_context(&json!({ "name": "Ada" }))?;
# assert_eq!(out, "Hi Ada");
# Ok::<(), hf_chat_template::Error>(())
```

`Error::Compile` and `Error::Render` now carry a `TemplateError` instead of a `minijinja::Error`.
It prints the same message, reports the line with `line()`, and chains with `source()`.

The `Clock` trait now returns the current time instead of formatting it. Implement `now` and return
a `Civil`, and the crate does the formatting, so a custom clock cannot drift from `transformers`.

```rust
use hf_chat_template::{Civil, Clock};

struct MyClock;
impl Clock for MyClock {
    fn now(&self) -> Civil {
        Civil::from_unix_secs(1_720_051_200)
    }
}
```

The `minijinja` re-export is gone. If you need the engine directly, depend on it yourself.

## Caveats

This crate does not add or strip a BOS token. If a template emits `{{ bos_token }}`, set
`add_special_tokens = false` at encode time so the tokenizer does not add BOS a second time. It
renders what the template says.

`strftime_now` defaults to UTC, while `transformers` uses local time. Enable the `strftime`
feature and inject `LocalClock` to match local time, or inject a `FixedClock` to pin a specific
date.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Files under `tests/corpus/` are trimmed excerpts of upstream model configs, redistributed under
each model's own license. See `tests/corpus/README.md`.

[`RenderInput`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/struct.RenderInput.html
