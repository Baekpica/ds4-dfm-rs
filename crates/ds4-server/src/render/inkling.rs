//! Pinned Inkling message framing and sorted source-template JSON.

use serde_json::{json, Value};

use super::{
    collect_tool_result_message, role_is_system, ChatMsg, ChatPart, RenderError, ThinkMode,
};

pub(crate) const MODEL: &str = "<|message_model|>";
pub(crate) const TEXT: &str = "<|content_text|>";
pub(crate) const THINK: &str = "<|content_thinking|>";
pub(crate) const END: &str = "<|end_message|>";
pub(crate) const EOS: &str = "<|content_model_end_sampling|>";
pub(crate) const INVOKE: &str = "<|content_invoke_tool_json|>";
const SYSTEM: &str = "<|message_system|>";
const USER: &str = "<|message_user|>";
const TOOL: &str = "<|message_tool|>";
const IMAGE: &str = "<|content_image|><|unused_200054|>";
const AUDIO: &str = "<|content_audio_input|><|unused_200053|><|audio_end|>";

fn source_json(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(source_json).collect::<Vec<_>>().join(",")
        ),
        Value::Object(values) => {
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort();
            let members: Vec<_> = keys
                .into_iter()
                .map(|key| format!("{}:{}", json!(key), source_json(&values[key])))
                .collect();
            format!("{{{}}}", members.join(","))
        }
        Value::Number(n) if n.is_f64() => {
            let value = n.as_f64().unwrap();
            // Python's JSON encoder uses repr(float), including signed,
            // two-digit exponents and .0 for integral finite floats.
            if value != 0.0 && (value.abs() < 1e-4 || value.abs() >= 1e16) {
                let scientific = format!("{value:e}");
                let (mantissa, exponent) = scientific.split_once('e').unwrap();
                let exponent: i32 = exponent.parse().unwrap();
                format!("{mantissa}e{exponent:+03}")
            } else {
                let text = value.to_string();
                if text.contains('.') {
                    text
                } else {
                    format!("{text}.0")
                }
            }
        }
        _ => value.to_string(),
    }
}

fn message(out: &mut String, role: &str, channel: &str, text: &str) {
    out.push_str(role);
    out.push_str(channel);
    out.push_str(text);
    out.push_str(END);
}

fn effort(out: &mut String, think: ThinkMode) {
    let level = match think {
        ThinkMode::None => "0",
        ThinkMode::Low => "0.2",
        ThinkMode::High => "0.9",
        ThinkMode::Max => "0.99",
    };
    message(
        out,
        SYSTEM,
        TEXT,
        &format!("Thinking effort level: {level}"),
    );
}

fn declare(out: &mut String, schemas: &str) -> Result<(), RenderError> {
    if schemas.is_empty() {
        return Ok(());
    }
    // The shared request parser stores consecutive schema objects, including
    // multiline JSON. A line split would corrupt nested parameter schemas.
    let tools: Vec<Value> = serde_json::Deserializer::from_str(schemas)
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(|_| RenderError("invalid Inkling tool schemas"))?;
    if tools.is_empty() {
        return Ok(());
    }
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let function = tool.get("function").unwrap_or(&tool);
        let name = function["name"]
            .as_str()
            .ok_or(RenderError("invalid Inkling tool name"))?;
        specs.push(json!({
            "name": name,
            "description": function["description"].as_str().unwrap_or(""),
            "parameters": function.get("parameters").or_else(|| function.get("input_schema")).filter(|p| !p.is_null()).cloned().unwrap_or(json!({})),
            "type": tool["type"].as_str().filter(|s| !s.is_empty()).unwrap_or("function"),
        }));
    }
    message(
        out,
        SYSTEM,
        "tool_declare<|content_xml|>",
        &source_json(&json!(specs)),
    );
    Ok(())
}

fn append_message(out: &mut String, msg: &ChatMsg, history: &[ChatMsg]) -> Result<(), RenderError> {
    if super::chat_msg_is_model_tool_result(msg) {
        let mut views = Vec::new();
        collect_tool_result_message(msg, &mut views);
        for view in views {
            let name = if !msg.name.is_empty() {
                msg.name.as_str()
            } else {
                history
                    .iter()
                    .flat_map(|m| &m.calls)
                    .find(|call| !view.id.is_empty() && call.id == view.id)
                    .map(|call| call.name.as_str())
                    .unwrap_or("")
            };
            message(
                out,
                TOOL,
                &format!("{name}{TEXT}"),
                &String::from_utf8_lossy(view.text),
            );
        }
        return Ok(());
    }
    let role = match msg.role.as_str() {
        "system" | "developer" => SYSTEM,
        "user" => USER,
        "assistant" => MODEL,
        _ => return Err(RenderError("invalid Inkling message role")),
    };
    if msg.role == "assistant" && !msg.reasoning.is_empty() {
        message(out, MODEL, THINK, &msg.reasoning);
    }
    if !msg.parts.is_empty() {
        for part in &msg.parts {
            match part {
                ChatPart::Text(text) => message(out, role, TEXT, text),
                ChatPart::Image(_) => message(out, role, IMAGE, ""),
                ChatPart::Audio(_) => message(out, role, AUDIO, ""),
                ChatPart::ToolResult { .. } => {}
            }
        }
    } else if !msg.content.is_empty() || (msg.calls.is_empty() && msg.reasoning.is_empty()) {
        message(out, role, TEXT, &msg.content);
    }
    if msg.role == "assistant" {
        for call in &msg.calls {
            let args: Value = serde_json::from_str(if call.arguments.is_empty() {
                "{}"
            } else {
                &call.arguments
            })
            .map_err(|_| RenderError("invalid Inkling tool arguments"))?;
            if !args.is_object() {
                return Err(RenderError("Inkling tool arguments must be an object"));
            }
            // The outer envelope has a source-defined order; args sort recursively.
            let body = format!(
                "{{\"name\":{},\"args\":{}}}",
                json!(call.name),
                source_json(&args)
            );
            message(out, MODEL, &format!("{}{INVOKE}", call.name), &body);
        }
        out.push_str(EOS);
    }
    Ok(())
}

pub(crate) fn render(
    msgs: &[ChatMsg],
    schemas: &str,
    think: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let mut out = String::new();
    declare(&mut out, schemas)?;
    let mut emitted = false;
    for msg in msgs {
        if !emitted && !role_is_system(&msg.role) {
            effort(&mut out, think);
            emitted = true;
        }
        append_message(&mut out, msg, msgs)?;
    }
    if !emitted {
        effort(&mut out, think);
    }
    out.push_str(MODEL);
    Ok(out.into_bytes())
}

pub(crate) fn live_tail(tail: &[ChatMsg], history: &[ChatMsg]) -> Result<Vec<u8>, RenderError> {
    let mut out = EOS.to_string();
    for msg in tail {
        append_message(&mut out, msg, history)?;
    }
    out.push_str(MODEL);
    Ok(out.into_bytes())
}

pub(crate) fn recovery(raw: &[u8], detail: &str) -> Vec<u8> {
    let mut out = String::new();
    // EOS is sampled without being evaluated. Close an unfinished channel
    // before the source-template tool result and next model turn.
    if !raw.ends_with(EOS.as_bytes()) {
        if !raw.ends_with(END.as_bytes()) {
            out.push_str(END);
        }
        out.push_str(EOS);
    }
    let error = format!(
        "Tool error: invalid Inkling tool call: {detail}\n\
The previous assistant output was not executed because its tool syntax was malformed. \
Emit a new valid native Inkling tool call, or answer normally if no tool is needed."
    );
    message(&mut out, TOOL, TEXT, &error);
    out.push_str(MODEL);
    out.into_bytes()
}

pub(crate) fn system_region(prompt: &[u8]) -> Vec<u8> {
    let Ok(mut rest) = std::str::from_utf8(prompt) else {
        return Vec::new();
    };
    while let Some(body) = rest.strip_prefix(SYSTEM) {
        let Some((body, tail)) = body.split_once(END) else {
            break;
        };
        if let Some(text) = body.strip_prefix(TEXT) {
            return text.as_bytes().to_vec();
        }
        rest = tail;
    }
    Vec::new()
}

pub(crate) fn parse(raw: &[u8]) -> Option<crate::tools::ParsedGenerated> {
    let mut out = crate::tools::ParsedGenerated {
        ok: true,
        ..Default::default()
    };
    let mut pos = 0;
    while pos < raw.len() {
        if raw[pos..].starts_with(MODEL.as_bytes()) {
            pos += MODEL.len();
        }
        if raw[pos..].starts_with(EOS.as_bytes()) {
            pos += EOS.len();
            continue;
        }
        let Some((start, marker)) = [TEXT, THINK, INVOKE]
            .into_iter()
            .filter_map(|marker| {
                raw[pos..]
                    .windows(marker.len())
                    .position(|s| s == marker.as_bytes())
                    .map(|offset| (pos + offset, marker))
            })
            .min_by_key(|(at, _)| *at)
        else {
            // A generation budget can end inside the next channel header.
            // Never publish a pending tool name or partial control marker.
            break;
        };
        let name = std::str::from_utf8(&raw[pos..start]).ok()?;
        if marker != INVOKE && !name.is_empty() {
            return None;
        }
        let body_start = start + marker.len();
        let end = raw[body_start..]
            .windows(END.len())
            .position(|s| s == END.as_bytes())
            .map(|offset| body_start + offset);
        let body_end = end.unwrap_or(raw.len());
        let body = &raw[body_start..body_end];
        match marker {
            TEXT => out
                .content
                .extend_from_slice(crate::stream::utf8_trim_tail(body)),
            THINK => out
                .reasoning
                .extend_from_slice(crate::stream::utf8_trim_tail(body)),
            INVOKE if end.is_some() => {
                let value: Value = serde_json::from_slice(body).ok()?;
                let function = value["name"].as_str().filter(|s| !s.is_empty())?;
                let args = value.get("args").filter(|v| v.is_object())?;
                if !name.is_empty() && name != function {
                    return None;
                }
                out.calls.push(crate::parse::ToolCall {
                    id: String::new(),
                    name: function.into(),
                    arguments: source_json(args),
                });
                out.raw_tool_text.push_str(MODEL);
                out.raw_tool_text
                    .push_str(std::str::from_utf8(&raw[pos..body_end + END.len()]).ok()?);
            }
            INVOKE => return None,
            _ => {}
        }
        pos = end.map(|end| end + END.len()).unwrap_or(raw.len());
    }
    Some(out)
}
