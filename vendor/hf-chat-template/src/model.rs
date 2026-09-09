//! Typed input model — the data you hand to [`ChatTemplate::render`](crate::ChatTemplate::render).
//!
//! These mirror the OpenAI/HF message shape and are fully `serde`-(de)serializable, so callers
//! can deserialize their existing request JSON straight in. The model is deliberately *thin*:
//! the stable parts (`role`) are typed, while everything genuinely open-ended — tool schemas,
//! tool-call payloads, multimodal content parts, and per-model extra kwargs — stays as
//! [`serde_json::Value`] so we never lose a field or impose a shape a template doesn't expect.
//!
//! Key order is load-bearing (it must survive into `| tojson`); this is why the crate enables
//! `serde_json`'s `preserve_order` feature. See `src/json.rs`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// Everything a chat template renders from: the conversation plus optional tools/documents
/// and the generation-prompt flag. Unmodeled keys land in [`extra`](RenderInput::extra) and
/// are passed to the Jinja context verbatim.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RenderInput {
    /// The conversation, in order.
    pub messages: Vec<Message>,

    /// Tool/function definitions (JSON-schema), serialized by templates via `tools | tojson`.
    /// Left as raw JSON because the schema shape is open-ended and key order matters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Json>,

    /// RAG documents for grounded-generation templates (e.g. Command-R).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<Json>,

    /// Whether to append the assistant generation prefix. The template branches on this;
    /// we never synthesize the prefix ourselves.
    #[serde(default)]
    pub add_generation_prompt: bool,

    /// Arbitrary extra template kwargs some models read (`enable_thinking`, `builtin_tools`,
    /// …). Flattened into the top-level Jinja context. Order-preserving.
    #[serde(default, flatten)]
    pub extra: Map<String, Json>,
}

/// A single chat message. `role` is typed; `content` is string-or-parts; all other keys
/// (`name`, `tool_call_id`, …) flow through [`extra`](Message::extra).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Message {
    /// `"system" | "user" | "assistant" | "tool" | …` — open by design.
    pub role: String,

    /// Message body: a plain string, or a list of multimodal parts. `None` omits the key
    /// entirely (faithful to messages that carry only `tool_calls`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,

    /// Assistant tool-call payloads, raw JSON (shape varies by model:
    /// `function.arguments` is sometimes a dict, sometimes a JSON string).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Json>,

    /// Any other per-message keys (`name`, `tool_call_id`, model-specific fields).
    #[serde(default, flatten)]
    pub extra: Map<String, Json>,
}

impl Message {
    /// Convenience constructor for the common `{role, content: <text>}` message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: role.into(),
            content: Some(Content::Text(content.into())),
            tool_calls: Vec::new(),
            extra: Map::new(),
        }
    }

    /// A `system` message with text content. Shorthand for `Message::new("system", text)`.
    pub fn system(content: impl Into<String>) -> Self {
        Message::new("system", content)
    }

    /// A `user` message with text content. Shorthand for `Message::new("user", text)`.
    pub fn user(content: impl Into<String>) -> Self {
        Message::new("user", content)
    }

    /// An `assistant` message with text content. Shorthand for `Message::new("assistant", text)`.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new("assistant", content)
    }

    /// A `tool` message carrying a tool's result. Shorthand for `Message::new("tool", text)`.
    ///
    /// Templates that correlate results with calls read per-model keys such as `tool_call_id` or
    /// `name`; set those on [`extra`](Message::extra), since their names vary by model.
    pub fn tool(content: impl Into<String>) -> Self {
        Message::new("tool", content)
    }
}

/// Message content: a plain string, or a list of multimodal parts.
///
/// Serialized untagged, so `Text` becomes a JSON string and `Parts` a JSON array — preserving
/// the distinction templates probe with `content is string`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// A plain text body.
    Text(String),
    /// `[{ "type": "text", "text": … }, { "type": "image", … }, …]` — raw JSON parts.
    Parts(Vec<Json>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}
