//! Local adapter for model-provided input grammar. Tokenization, media
//! processing and generated-output parsing remain separate host contracts.

use serde_json::{json, Map, Value};
use std::path::Path;

use crate::{ChatThinkMode, Error, GgufFile, Result};

#[derive(Debug, Clone, Copy)]
pub enum RenderClock {
    System,
    Fixed(i64),
}

#[derive(Debug)]
pub struct Template {
    renderer: hf_chat_template::ChatTemplate,
    defaults: Map<String, Value>,
    source: String,
}

/// Translate request options to template parameters, never role/tool grammar.
/// Forced-off prefixes close an output reasoning channel in templates whose
/// official generation mode always starts thinking.
#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    mode: ChatThinkMode,
    effort: &'static str,
    prefill: &'static str,
}

impl ChatOptions {
    pub fn new(model_id: i32, mode: ChatThinkMode) -> Self {
        use ChatThinkMode::{High, Low, Max, None};
        let effort = match (model_id, mode) {
            (6, High | Max) => "xhigh",
            (2, Low) => "medium",
            (2, Max) => "xhigh",
            (7, None | Low | Max) => "max",
            // The existing K2 output parser handles its default think channel.
            (8, _) => "high",
            (_, None) => "none",
            (_, Low) => "low",
            (_, High) => "high",
            (_, Max) => "max",
        };
        let prefill = match (model_id, mode) {
            (7, None) => "</think>",
            (8, None) => "</ifm|think>\n",
            _ => "",
        };
        Self {
            mode,
            effort,
            prefill,
        }
    }
}

impl Template {
    pub fn compile(source: &str, clock: RenderClock) -> Result<Self> {
        let mut builder = hf_chat_template::ChatTemplate::builder(source);
        if let RenderClock::Fixed(seconds) = clock {
            builder = builder.clock(hf_chat_template::FixedClock::from_unix_secs(seconds));
        }
        Ok(Self {
            renderer: builder.build().map_err(template_error)?,
            defaults: Map::new(),
            source: "inline".into(),
        })
    }

    /// Read only artifact metadata; never load weights or a GPU model.
    pub fn from_model(path: &Path) -> Result<Option<Self>> {
        let gguf = GgufFile::open(path).map_err(asset_error)?;
        Self::load(path, &gguf)
    }

    pub(crate) fn load(path: &Path, gguf: &GgufFile) -> Result<Option<Self>> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let config_path = dir.join("tokenizer_config.json");
        let config: Value = read_optional(&config_path)?
            .map(|text| serde_json::from_str(&text).map_err(asset_error))
            .transpose()?
            .unwrap_or(Value::Null);
        let sidecar = dir.join("chat_template.jinja");
        let (source, origin) = if let Some(text) = read_optional(&sidecar)? {
            (text, sidecar.display().to_string())
        } else if let Some(text) = config.get("chat_template").filter(|v| !v.is_null()) {
            let text = text
                .as_str()
                .or_else(|| text.get("default").and_then(Value::as_str))
                .ok_or_else(|| {
                    asset_error("tokenizer_config chat_template needs a string or default template")
                })?;
            (text.into(), config_path.display().to_string())
        } else {
            // DeepSeek V4 publishes a Python encoder, not Jinja. Its historical
            // GGUF template is not that encoder; do not silently select it.
            if gguf.get_string("general.architecture") == Some(b"deepseek4") {
                return Ok(None);
            }
            let Some(text) = gguf.get_string("tokenizer.chat_template") else {
                return Ok(None);
            };
            (
                std::str::from_utf8(text).map_err(asset_error)?.into(),
                format!("{}:tokenizer.chat_template", path.display()),
            )
        };
        if source.trim().is_empty() {
            return Err(asset_error(format!("empty template at {origin}")));
        }
        let mut template = Self::compile(&source, RenderClock::System)?;
        template.source = origin;
        let tokens = gguf
            .get_array("tokenizer.ggml.tokens")
            .map(|array| gguf.array_strings(&array))
            .transpose()
            .map_err(asset_error)?
            .unwrap_or_default();
        for name in ["bos", "eos", "pad", "unk", "sep", "cls", "mask"] {
            let key = format!("{name}_token");
            let config_token = config.get(&key).and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("content").and_then(Value::as_str))
            });
            let index = gguf.get_token_id(&format!("tokenizer.ggml.{name}_token_id"));
            let gguf_token = index
                .and_then(|id| usize::try_from(id).ok())
                .and_then(|id| tokens.get(id))
                .and_then(|bytes| std::str::from_utf8(bytes).ok());
            if let Some(token) = config_token.or(gguf_token) {
                template.defaults.insert(key, token.into());
            }
        }
        Ok(Some(template))
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Do not add or remove special tokens. Tokenization must not add a BOS.
    pub fn render(&self, context: &Value) -> Result<String> {
        let mut complete = self.defaults.clone();
        complete.extend(
            context
                .as_object()
                .ok_or_else(|| asset_error("context must be an object"))?
                .clone(),
        );
        self.renderer
            .render_context(&complete)
            .map_err(template_error)
    }

    pub fn render_chat(
        &self,
        messages: &[Value],
        tools: &[Value],
        options: ChatOptions,
    ) -> Result<String> {
        let mut messages = messages.to_vec();
        for message in &mut messages {
            if message["role"] != "assistant" {
                continue;
            }
            // APIs may omit hidden reasoning. Supply an empty canonical field,
            // while preserving any explicit template-specific thinking value.
            const THINKING_FIELDS: &[&str] = &[
                "reasoning_content",
                "reasoning",
                "think",
                "think_fast",
                "think_faster",
            ];
            if !THINKING_FIELDS
                .iter()
                .any(|field| message.get(field).is_some())
            {
                message["reasoning_content"] = "".into();
            }
        }
        let context = json!({
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": true,
            "enable_thinking": options.mode != ChatThinkMode::None,
            "reasoning_effort": options.effort,
        });
        let mut rendered = self.render(&context)?;
        rendered.push_str(options.prefill);
        Ok(rendered)
    }
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(asset_error(format!("{}: {error}", path.display()))),
    }
}

fn asset_error(error: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: format!("chat template: {error}"),
    }
}

fn template_error(error: hf_chat_template::Error) -> Error {
    asset_error(error)
}
