//! IQuest XML values follow its official parser's schema-aware JSON decoding.
use serde_json::{Map, Value};

use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::render::{IQUEST_TOOL_CALL_END, IQUEST_TOOL_CALL_START};

use super::{find_substr, tool_schema_order_prop_type, tool_schema_orders_find, ParsedGenerated};

const KEY_START: &str = "<arg_key>";
const KEY_END: &str = "</arg_key>";
const VALUE_START: &str = "<arg_value>";
const VALUE_END: &str = "</arg_value>";

fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub(crate) fn call_body(body: &[u8], orders: &[ToolSchemaOrder]) -> Option<ToolCall> {
    let body = std::str::from_utf8(body).ok()?;
    let first = body.find(KEY_START).unwrap_or(body.len());
    let name = body[..first].trim_matches(python_space);
    if name.is_empty() {
        return None;
    }
    let order = tool_schema_orders_find(orders, name);
    let mut tail = &body[first..];
    let mut args = Map::new();
    while !tail.trim_start_matches(python_space).is_empty() {
        tail = tail
            .trim_start_matches(python_space)
            .strip_prefix(KEY_START)?;
        let end = tail.find(KEY_END)?;
        let key = tail[..end].trim_matches(python_space);
        if key.is_empty() {
            return None;
        }
        tail = tail[end + KEY_END.len()..]
            .trim_start_matches(python_space)
            .strip_prefix(VALUE_START)?;
        let end = tail.find(VALUE_END)?;
        let raw = &tail[..end];
        let value = if tool_schema_order_prop_type(order, key) == Some("string") {
            Value::String(raw.to_owned())
        } else {
            let trimmed = raw.trim_matches(python_space);
            serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(trimmed.to_owned()))
        };
        args.insert(key.to_owned(), value);
        tail = &tail[end + VALUE_END.len()..];
    }
    Some(ToolCall {
        name: name.to_owned(),
        arguments: Value::Object(args).to_string(),
        ..Default::default()
    })
}

pub(super) fn split(text: &[u8], thinking: bool) -> (&[u8], Vec<u8>) {
    if thinking {
        let start = b"<think>";
        let raw = find_substr(text, start)
            .map(|at| &text[at + start.len()..])
            .unwrap_or(text);
        let Some(end) = find_substr(raw, b"</think>") else {
            return (&[], raw.to_vec());
        };
        (&raw[end + b"</think>".len()..], raw[..end].to_vec())
    } else {
        (text, Vec::new())
    }
}

pub(super) fn parse(text: &[u8], thinking: bool, orders: &[ToolSchemaOrder]) -> ParsedGenerated {
    let (body, reasoning) = split(text, thinking);
    let start = IQUEST_TOOL_CALL_START.as_bytes();
    let end = IQUEST_TOOL_CALL_END.as_bytes();
    let mut cursor = 0;
    let mut kept = 0;
    let mut out = ParsedGenerated {
        reasoning,
        ok: true,
        ..Default::default()
    };
    while let Some(rel) = find_substr(&body[cursor..], start) {
        let open = cursor + rel;
        let first = open + start.len();
        let Some(close) = find_substr(&body[first..], end).map(|at| first + at) else {
            break;
        };
        cursor = close + end.len();
        if let Some(call) = call_body(&body[first..close], orders) {
            out.content.extend_from_slice(&body[kept..open]);
            out.raw_tool_text
                .push_str(&String::from_utf8_lossy(&body[open..cursor]));
            out.calls.push(call);
            kept = cursor;
        }
    }
    out.content.extend_from_slice(&body[kept..]);
    out
}
