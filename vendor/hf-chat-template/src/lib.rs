//! Render Hugging Face `chat_template` (Jinja2) strings the way Python
//! `transformers.apply_chat_template` does.
//!
//! This crate emits a **prompt string**. Turning it into token IDs is the caller's job
//! (`tokenizers`, `tiktoken-rs`, …) — keeping that boundary is deliberate.
//!
//! ```
//! use hf_chat_template::{ChatTemplate, Message};
//!
//! let tmpl = ChatTemplate::from_str(
//!     "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}\
//!      {% if add_generation_prompt %}<|assistant|>{% endif %}",
//! ).unwrap();
//!
//! let out = tmpl.render_messages(&[Message::user("hi")], true).unwrap();
//! assert_eq!(out, "<|user|>hi\n<|assistant|>");
//! ```
//!
//! For template variables the typed model doesn't cover, pass any `Serialize` value to
//! [`ChatTemplate::render_context`].
//!
//! ## Special tokens & BOS doubling
//! Templates that emit `{{ bos_token }}` expect you to pass `bos_token` in the context, and
//! to set `add_special_tokens = false` at encode time so the tokenizer does not add BOS a
//! second time. This crate renders exactly what the template says and never strips silently.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod clock;
mod config;
mod engine;
mod error;
#[cfg(feature = "hub")]
mod hub;
mod json;
mod model;
mod template;

#[cfg(feature = "strftime")]
pub use clock::LocalClock;
pub use clock::{Civil, Clock, FixedClock, SystemClock};
pub use config::{ChatTemplateField, NamedTemplate, TokenField, TokenizerConfig};
pub use error::{Error, TemplateError};
pub use model::{Content, Message, RenderInput};
pub use template::{ChatTemplate, ChatTemplateBuilder};

/// Compiles every Rust block in `README.md` as a doctest, so the first thing a potential user
/// reads cannot silently rot. Exists only under `cargo test --doc`.
///
/// Gated on `hub` because the README shows `from_hub`. Gating here rather than annotating the
/// block keeps `#[cfg]` noise out of a file people read; CI covers it via `--all-features`.
#[cfg(all(doctest, feature = "hub"))]
#[doc = include_str!("../README.md")]
pub struct ReadmeExamples;
