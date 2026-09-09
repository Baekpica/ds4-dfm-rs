//! `tokenizer_config.json` loading and template-source resolution.
//!
//! A [`TokenizerConfig`] is a tolerant view over the parts of a model's `tokenizer_config.json`
//! we need: the `chat_template` (in either of its shapes) plus the special-token fields
//! the templates reference as context variables (`{{ bos_token }}`). Unknown fields are kept in
//! `extra`, never rejected.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::error::Error;

/// A tolerant view over a parsed `tokenizer_config.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// The chat template, in whichever shape this config uses (see [`ChatTemplateField`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<ChatTemplateField>,

    /// Beginning-of-sequence token, if the model defines one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<TokenField>,
    /// End-of-sequence token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<TokenField>,
    /// Padding token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token: Option<TokenField>,
    /// Unknown token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unk_token: Option<TokenField>,

    /// Everything else in the file, preserved.
    #[serde(default, flatten)]
    pub extra: Map<String, Json>,
}

/// The two shapes a `chat_template` field takes. A bare object mapping names to templates is not
/// accepted, and deserializing one fails as "did not match any variant".
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateField {
    /// A single Jinja string: `"chat_template": "{% for … %}"`.
    Single(String),
    /// A list of named templates: `[{"name":"default","template":"…"}, …]`.
    Named(Vec<NamedTemplate>),
}

/// One entry of the list-of-named-templates form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedTemplate {
    /// The template's name (e.g. `"default"`, `"tool_use"`, `"rag"`).
    pub name: String,
    /// The Jinja source.
    pub template: String,
}

/// A special-token field: either a bare string, or an `AddedToken` object with a `content` key.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenField {
    /// `"bos_token": "<s>"`.
    Str(String),
    /// `"bos_token": {"content": "<s>", "lstrip": false, …}`.
    Obj(Map<String, Json>),
}

impl TokenField {
    /// The token's string value (`content` for the object form).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            TokenField::Str(s) => Some(s),
            TokenField::Obj(m) => m.get("content").and_then(Json::as_str),
        }
    }
}

impl TokenizerConfig {
    /// The special tokens to inject into the render context, as `(name, value)` pairs.
    /// Only fields that are present and resolve to a string are included.
    pub(crate) fn special_tokens(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (name, field) in [
            ("bos_token", &self.bos_token),
            ("eos_token", &self.eos_token),
            ("pad_token", &self.pad_token),
            ("unk_token", &self.unk_token),
        ] {
            if let Some(s) = field.as_ref().and_then(TokenField::as_str) {
                out.push((name.to_string(), s.to_string()));
            }
        }
        out
    }
}

/// Resolve the Jinja source to use from a `chat_template` field, honoring an optionally
/// requested name (§8 rules):
/// - explicit `name` requested → that template, else [`Error::Config`];
/// - else a template named `"default"`;
/// - else, if exactly one exists, that one;
/// - else ambiguous → [`Error::Config`] listing the available names.
pub(crate) fn resolve_template<'a>(
    field: &'a ChatTemplateField,
    name: Option<&str>,
) -> Result<&'a str, Error> {
    match field {
        // A single string ignores any requested name — there is nothing to disambiguate.
        ChatTemplateField::Single(s) => Ok(s),
        ChatTemplateField::Named(list) => {
            if list.is_empty() {
                return Err(Error::Config("chat_template list is empty".into()));
            }
            if let Some(want) = name {
                list.iter()
                    .find(|t| t.name == want)
                    .map(|t| t.template.as_str())
                    .ok_or_else(|| {
                        Error::Config(format!(
                            "no chat_template named '{want}'; available: {}",
                            available_names(list)
                        ))
                    })
            } else if let Some(t) = list.iter().find(|t| t.name == "default") {
                Ok(&t.template)
            } else if list.len() == 1 {
                Ok(&list[0].template)
            } else {
                Err(Error::Config(format!(
                    "ambiguous chat_template: specify a name via builder. available: {}",
                    available_names(list)
                )))
            }
        }
    }
}

fn available_names(list: &[NamedTemplate]) -> String {
    list.iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}
