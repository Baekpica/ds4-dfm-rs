//! Tencent Hy4's namespaced tool group, as emitted by its source template.
use super::{find_substr, split_reasoning_response, ParsedGenerated};
use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::stream::ChatFormat;

pub(crate) const START: &str = "<tool_calls:opensource>";
pub(crate) const END: &str = "</tool_calls:opensource>";
const CALL: &str = "<tool_call:opensource>";
const CLOSE: &str = "</tool_call:opensource>";
const ARGS: [&str; 4] = [
    "<arg_key:opensource>",
    "</arg_key:opensource>",
    "<arg_value:opensource>",
    "</arg_value:opensource>",
];

pub(crate) fn group(bytes: &[u8], orders: &[ToolSchemaOrder]) -> Option<Vec<ToolCall>> {
    let mut tail = std::str::from_utf8(bytes).ok()?.trim();
    let mut calls = Vec::new();
    while !tail.is_empty() {
        let body = tail.strip_prefix(CALL)?;
        let (call, rest) = super::iquest::read_call(body, orders, ARGS, CLOSE)?;
        calls.push(call);
        tail = rest.trim_start();
    }
    (!calls.is_empty()).then_some(calls)
}

pub(super) fn parse(text: &[u8], thinking: bool, orders: &[ToolSchemaOrder]) -> ParsedGenerated {
    let (body, reasoning) = split_reasoning_response(text, ChatFormat::Hy4, thinking);
    let mut out = ParsedGenerated {
        reasoning,
        ok: true,
        ..Default::default()
    };
    let mut tail = body.as_slice();
    while let Some(at) = find_substr(tail, START.as_bytes()) {
        out.content.extend_from_slice(&tail[..at]);
        let block = &tail[at..];
        let first = &block[START.len()..];
        if let Some((end, calls)) = find_substr(first, END.as_bytes())
            .and_then(|end| group(&first[..end], orders).map(|c| (end, c)))
        {
            let used = START.len() + end + END.len();
            out.raw_tool_text
                .push_str(&String::from_utf8_lossy(&block[..used]));
            out.calls.extend(calls);
            tail = &block[used..];
        } else {
            out.ok = false;
            out.content.extend_from_slice(block);
            return out;
        }
    }
    out.content.extend_from_slice(tail);
    out
}
