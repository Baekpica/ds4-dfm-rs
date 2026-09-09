//! API normalization for the shared template adapter. This module carries
//! structured messages and tool values; model Jinja owns input grammar.

use ds4_core::chat_template::{ChatOptions, Template};
use ds4_core::ChatThinkMode;
use serde_json::{json, Value};

use crate::generate::GenerateError;
use crate::parse::{ChatMsg, ChatPart, ParsedRequest};
use crate::route::{Api, ReqKind, ThinkMode};

pub(crate) fn render_model(
    template: Option<&Template>,
    model_id: i32,
    parsed: &ParsedRequest,
) -> Result<Vec<u8>, GenerateError> {
    if let Some(template) = template {
        return render(template, model_id, parsed);
    }
    if model_id == 0 || parsed.kind == ReqKind::Completion {
        return crate::generate::render_prompt(parsed, model_id);
    }
    Err(input_error(
        "missing chat_template.jinja or tokenizer.chat_template",
    ))
}

pub fn render(
    template: &Template,
    model_id: i32,
    parsed: &ParsedRequest,
) -> Result<Vec<u8>, GenerateError> {
    if parsed.kind == ReqKind::Completion {
        return Ok(parsed.prompt_text.clone().unwrap_or_default().into_bytes());
    }
    let mut messages = messages(parsed)?;
    prepare_image_text(model_id, &mut messages);
    let tools = serde_json::Deserializer::from_str(&parsed.tool_schemas)
        .into_iter::<Value>()
        .map(|value| {
            value.map(|function| {
                let function = match function {
                    Value::Object(fields) => Value::Object(
                        fields
                            .into_iter()
                            .map(|(key, value)| {
                                (
                                    if key == "input_schema" {
                                        "parameters".into()
                                    } else {
                                        key
                                    },
                                    value,
                                )
                            })
                            .collect(),
                    ),
                    other => other,
                };
                json!({"type":"function", "function":function})
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(input_error)?;
    let mode = match parsed.think_mode {
        ThinkMode::None => ChatThinkMode::None,
        ThinkMode::Low => ChatThinkMode::Low,
        ThinkMode::High => ChatThinkMode::High,
        ThinkMode::Max => ChatThinkMode::Max,
    };
    template
        .render_chat(&messages, &tools, ChatOptions::new(model_id, mode))
        .map(String::into_bytes)
        .map_err(input_error)
}

fn prepare_image_text(model_id: i32, messages: &mut [Value]) {
    use crate::render::{
        syntax_for_model_id, ModelSyntax, GLM_IMAGE, GLM_VISION_END, GLM_VISION_START,
    };

    if syntax_for_model_id(model_id) != ModelSyntax::Glm53 {
        return;
    }
    // GLM's text template does not consume media arrays. The image processor
    // supplies placeholders; native preparation later expands their token spans.
    for message in messages {
        let Some(parts) = message["content"].as_array() else {
            continue;
        };
        let mut text = String::new();
        for part in parts {
            if part["type"] == "image" {
                text.push_str(GLM_VISION_START);
                text.push_str(GLM_IMAGE);
                text.push_str(GLM_VISION_END);
            } else if let Some(value) = part["text"].as_str() {
                text.push_str(value);
            }
        }
        message["content"] = text.into();
    }
}

pub(crate) fn messages(parsed: &ParsedRequest) -> Result<Vec<Value>, GenerateError> {
    let mut out = Vec::new();
    // Anthropic's system field is top-level; the legacy parser stores it last.
    let ordered: Vec<_> = if parsed.api == Api::Anthropic {
        parsed
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .chain(parsed.messages.iter().filter(|m| m.role != "system"))
            .collect()
    } else {
        parsed.messages.iter().collect()
    };
    for msg in ordered {
        if !msg
            .parts
            .iter()
            .any(|p| matches!(p, ChatPart::ToolResult { .. }))
        {
            out.push(message(msg, &msg.parts, &msg.content)?);
            continue;
        }
        let mut parts = Vec::new();
        for part in &msg.parts {
            if let ChatPart::ToolResult { id, content } = part {
                if !parts.is_empty() {
                    out.push(message(msg, &parts, "")?);
                    parts.clear();
                }
                out.push(json!({"role":"tool", "tool_call_id":id, "content":content}));
            } else {
                parts.push(part.clone());
            }
        }
        if !parts.is_empty() {
            out.push(message(msg, &parts, "")?);
        }
    }
    Ok(out)
}

fn message(msg: &ChatMsg, parts: &[ChatPart], fallback: &str) -> Result<Value, GenerateError> {
    let role = if msg.role == "developer" {
        "system"
    } else {
        &msg.role
    };
    let mut out = json!({"role":role, "content":content(parts, fallback)});
    if !msg.name.is_empty() {
        out["name"] = msg.name.clone().into();
    }
    if !msg.reasoning.is_empty() {
        out["reasoning_content"] = msg.reasoning.clone().into();
    }
    if !msg.tool_call_id.is_empty() && role == "tool" {
        out["tool_call_id"] = msg.tool_call_id.clone().into();
    }
    if !msg.calls.is_empty() {
        let calls = msg.calls.iter().map(|call| {
            let arguments: Value = serde_json::from_str(if call.arguments.is_empty() { "{}" } else { &call.arguments }).map_err(input_error)?;
            if !arguments.is_object() {
                return Err(input_error("tool call arguments must be a JSON object"));
            }
            Ok(json!({"id":call.id, "type":"function", "function":{"name":call.name,"arguments":arguments}}))
        }).collect::<Result<Vec<_>, GenerateError>>()?;
        out["tool_calls"] = calls.into();
    }
    Ok(out)
}

fn content(parts: &[ChatPart], fallback: &str) -> Value {
    if parts.is_empty() {
        return fallback.into();
    }
    if !parts
        .iter()
        .any(|p| matches!(p, ChatPart::Image(_) | ChatPart::Audio(_)))
    {
        return parts
            .iter()
            .filter_map(|part| match part {
                ChatPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>()
            .into();
    }
    parts
        .iter()
        .filter_map(|part| match part {
            ChatPart::Text(text) => Some(json!({"type":"text", "text":text})),
            ChatPart::Image(_) => Some(json!({"type":"image"})),
            ChatPart::Audio(_) => Some(json!({"type":"audio"})),
            ChatPart::ToolResult { .. } => None,
        })
        .collect::<Vec<_>>()
        .into()
}

fn input_error(error: impl std::fmt::Display) -> GenerateError {
    GenerateError::Engine(format!("chat input: {error}"))
}

#[cfg(test)]
#[path = "chat_history_test.rs"]
mod history_tests;

/// One retained tool frontier. The registry owns liveness/TTL; this owns the
/// structured context needed when a client sends only tool results.
#[derive(Clone, Debug)]
pub(crate) struct History {
    request: ParsedRequest,
    calls: Vec<String>,
}

impl History {
    pub(crate) fn capture(
        mut request: ParsedRequest,
        generated: &crate::tools::ParsedGenerated,
    ) -> Self {
        let calls = generated.calls.iter().map(|call| call.id.clone()).collect();
        request.messages.push(ChatMsg {
            role: "assistant".into(),
            content: String::from_utf8_lossy(&generated.content).into_owned(),
            reasoning: String::from_utf8_lossy(&generated.reasoning).into_owned(),
            calls: generated.calls.clone(),
            raw_dsml: generated.raw_dsml.clone(),
            raw_tool_text: generated.raw_tool_text.clone(),
            ..ChatMsg::default()
        });
        Self { request, calls }
    }

    pub(crate) fn matches(&self, parsed: &ParsedRequest) -> bool {
        parsed.api == self.request.api
            && !parsed.live_call_ids.is_empty()
            && parsed
                .live_call_ids
                .iter()
                .all(|id| self.calls.contains(id))
    }

    pub(crate) fn restore(&self, parsed: &mut ParsedRequest) -> Result<bool, GenerateError> {
        if parsed.live_call_ids.is_empty() {
            return Ok(false);
        }
        if !self.matches(parsed) {
            return Err(input_error(
                "tool continuation does not match retained chat history",
            ));
        }
        let has_call = parsed
            .messages
            .iter()
            .any(|msg| msg.calls.iter().any(|call| self.calls.contains(&call.id)));
        if has_call {
            // Full caller history is authoritative, including edits. Fill only
            // reasoning omitted by a protocol that refers to our live frontier.
            if let Some(saved) = self.request.messages.last() {
                for msg in &mut parsed.messages {
                    if msg.reasoning.is_empty()
                        && msg.calls.iter().any(|call| self.calls.contains(&call.id))
                    {
                        msg.reasoning.clone_from(&saved.reasoning);
                    }
                }
            }
        } else {
            let mut history = self.request.messages.clone();
            let mut tail = parsed.messages.clone();
            let current_system: Vec<_> = tail
                .iter()
                .filter(|m| m.role == "system" || m.role == "developer")
                .cloned()
                .collect();
            if !current_system.is_empty() {
                history.retain(|m| m.role != "system" && m.role != "developer");
                history.splice(0..0, current_system);
            }
            tail.retain(|m| m.role != "system" && m.role != "developer");
            for msg in &mut tail {
                for part in &mut msg.parts {
                    match part {
                        ChatPart::Image(index) => *index += self.request.images.len(),
                        ChatPart::Audio(index) => *index += self.request.audios.len(),
                        _ => {}
                    }
                }
            }
            history.extend(tail);
            parsed.messages = history;
            parsed
                .images
                .splice(0..0, self.request.images.iter().cloned());
            parsed
                .audios
                .splice(0..0, self.request.audios.iter().cloned());
        }
        if parsed.tool_schemas.is_empty() && parsed.tool_choice != crate::parse::ToolChoice::None {
            parsed.tool_schemas.clone_from(&self.request.tool_schemas);
            parsed.tool_orders.clone_from(&self.request.tool_orders);
            parsed.has_tools = !parsed.tool_schemas.is_empty();
        }
        Ok(true)
    }
}
