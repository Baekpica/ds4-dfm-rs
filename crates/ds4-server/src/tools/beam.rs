//! Beam-1.0 Harmony output (tokenizer_config.json `response_template`).
//!
//! The prompt ends at `<|start|>assistant`. Reasoning is an `analysis`
//! message, the answer a `final` one, and each tool call a
//! ` to=NAME<|channel|>analysis <|constrain|>xml<|message|>` message of
//! `<rfl:parameter name="K">\nV\n</rfl:parameter>` parameters joined by
//! newlines. Messages end at `<|end|>` (then `<|start|>assistant` opens the
//! next); `<|return|>` and `<|call|>` are stop tokens, and the decode loop
//! feeds `<|call|>` so the last call keeps its terminator.

use serde_json::{Map, Value};

use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::stream::ChatFormat;

use super::{tool_schema_order_prop_type, tool_schema_orders_find, ParsedGenerated};

pub(crate) const ASSISTANT: &str = "<|start|>assistant";
pub(crate) const ANALYSIS: &str = "<|channel|>analysis<|message|>";
pub(crate) const FINAL: &str = "<|channel|>final<|message|>";
pub(crate) const TOOL: &str = "<|channel|>analysis <|constrain|>xml<|message|>";
pub(crate) const RECIPIENT: &str = " to=";
pub(crate) const END: &str = "<|end|>";
pub(crate) const CALL: &str = "<|call|>";
pub(crate) const RETURN: &str = "<|return|>";
// The template's analysis close also accepts a bare `<|start|>assistant`
// and `<|constrain|><|start|>assistant`.
const CONSTRAIN: &str = "<|constrain|>";
const PARAM: &str = "<rfl:parameter name=\"";
const PARAM_HEAD_END: &str = "\">\n";
const PARAM_END: &str = "\n</rfl:parameter>";

// Header tokens of the release tokenizer (pinned with the release; ds4-core's
// beam_vq test encodes the header pieces against them).
const T_START: i32 = 151669;
const T_ASSISTANT: i32 = 77091;
const T_CHANNEL: i32 = 151672;
const T_MESSAGE: i32 = 151671;
const T_CONSTRAIN: i32 = 151673;
const T_ANALYSIS: i32 = 34484;
const T_FINAL: i32 = 11822;
const T_SPACE: i32 = 220;
const T_XML: i32 = 6455;

fn rfind(hay: &[u8], needle: &str) -> Option<usize> {
    hay.windows(needle.len())
        .rposition(|w| w == needle.as_bytes())
}

/// The token Harmony's header grammar admits alone after the reply so far.
/// The released VQ weights drift at header positions (the H100 goldens put
/// "final" or `<|channel|>` where the teacher writes "analysis" with a 17
/// logit margin, and replies open `<|channel|> final`), so headers are forced:
/// a reply opens with analysis, a message after `<|end|>` restarts as
/// `<|start|>assistant`, a second channel is final, and a call's header ends
/// ` <|constrain|>xml<|message|>`. Bodies, the recipient and the choice between
/// an answer and a call (when tools exist) stay sampled.
pub(crate) fn forced_header(text: &[u8], tools: bool) -> Option<i32> {
    let ends = |s: &str| text.ends_with(s.as_bytes());
    if text.is_empty() {
        return Some(T_CHANNEL);
    }
    if ends(END) {
        return Some(T_START);
    }
    if ends("<|start|>") {
        return Some(T_ASSISTANT);
    }
    if ends(ASSISTANT) {
        return (!tools).then_some(T_CHANNEL);
    }
    // The current message's header follows its `<|start|>assistant`.
    let header = &text[rfind(text, ASSISTANT).map_or(0, |at| at + ASSISTANT.len())..];
    let recipient = header.starts_with(RECIPIENT.as_bytes());
    if ends("<|channel|>") {
        return Some(if recipient || find(text, ANALYSIS).is_none() {
            T_ANALYSIS
        } else {
            T_FINAL
        });
    }
    if recipient {
        return if ends("<|channel|>analysis") {
            Some(T_SPACE)
        } else if ends("<|channel|>analysis ") {
            Some(T_CONSTRAIN)
        } else if ends("<|constrain|>") {
            Some(T_XML)
        } else if ends("<|constrain|>xml") {
            Some(T_MESSAGE)
        } else {
            None
        };
    }
    (ends("<|channel|>analysis") || ends("<|channel|>final")).then_some(T_MESSAGE)
}

/// `<|call|>` is a stop token that also closes the last call: the decode
/// lanes feed its text so parsers and streams see the terminator.
pub(crate) fn stop_text(format: ChatFormat, token: i32, eot: i32) -> Option<&'static [u8]> {
    (format == ChatFormat::Beam && token >= 0 && token == eot).then_some(CALL.as_bytes())
}

fn find(hay: &[u8], needle: &str) -> Option<usize> {
    hay.windows(needle.len())
        .position(|w| w == needle.as_bytes())
}

/// The earliest of `markers` in `hay`: (offset, marker).
pub(crate) fn earliest(hay: &[u8], markers: &[&'static str]) -> Option<(usize, &'static str)> {
    markers
        .iter()
        .filter_map(|&m| find(hay, m).map(|at| (at, m)))
        .min_by_key(|&(at, _)| at)
}

// A declared string stays text (the official value parser keeps it
// unstripped); anything else is the template's JSON with a text fallback.
fn call_body(name: &str, body: &str, orders: &[ToolSchemaOrder]) -> Option<ToolCall> {
    let order = tool_schema_orders_find(orders, name);
    let mut args = Map::new();
    let mut rest = body;
    loop {
        rest = rest.trim_start_matches('\n');
        if rest.is_empty() {
            break;
        }
        let tail = rest.strip_prefix(PARAM)?;
        let head = tail.find(PARAM_HEAD_END)?;
        let key = &tail[..head];
        if key.is_empty() || key.contains('"') {
            return None;
        }
        let value_start = &tail[head + PARAM_HEAD_END.len()..];
        // A missing close tag is legal when the next parameter follows.
        let (raw, next) = match value_start.find(PARAM_END) {
            Some(end) => (&value_start[..end], &value_start[end + PARAM_END.len()..]),
            None => {
                let end = value_start.find(&format!("\n{PARAM}"))?;
                (&value_start[..end], &value_start[end..])
            }
        };
        let value = if tool_schema_order_prop_type(order, key) == Some("string") {
            Value::String(raw.to_owned())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_owned()))
        };
        args.insert(key.to_owned(), value);
        rest = next;
    }
    Some(ToolCall {
        name: name.to_owned(),
        arguments: Value::Object(args).to_string(),
        ..Default::default()
    })
}

// ` to=NAME` before a tool header: NAME has no whitespace or `<`.
fn recipient(head: &[u8]) -> Option<&str> {
    let name = std::str::from_utf8(head).ok()?.strip_prefix(RECIPIENT)?;
    (!name.is_empty() && !name.contains(|c: char| c.is_whitespace() || c == '<')).then_some(name)
}

pub(crate) fn parse(raw: &[u8], orders: &[ToolSchemaOrder]) -> Option<ParsedGenerated> {
    let mut out = ParsedGenerated {
        ok: true,
        ..Default::default()
    };
    let (mut pos, mut framed) = (0, false);
    while pos < raw.len() {
        if raw[pos..].starts_with(ASSISTANT.as_bytes()) {
            pos += ASSISTANT.len();
        }
        // Unframed output is not Harmony (surface it raw); a budget can end
        // inside the next header: publish nothing of it.
        let Some((at, marker)) = earliest(&raw[pos..], &[FINAL, ANALYSIS, TOOL]) else {
            if !framed && !raw[pos..].starts_with(b"<|") {
                return None;
            }
            break;
        };
        framed = true;
        let head = &raw[pos..pos + at];
        let name = if marker == TOOL {
            Some(recipient(head)?)
        } else if head.is_empty() {
            None
        } else {
            return None;
        };
        let body_start = pos + at + marker.len();
        let closers: &[&'static str] = match marker {
            ANALYSIS => &[END, ASSISTANT, CALL, RETURN],
            FINAL => &[END, RETURN, CALL],
            _ => &[END, CALL],
        };
        let end = earliest(&raw[body_start..], closers).map(|(at, m)| (body_start + at, m));
        let body_end = end.map_or(raw.len(), |(at, _)| at);
        let mut body = &raw[body_start..body_end];
        match marker {
            ANALYSIS => {
                if end.is_some_and(|(_, m)| m == ASSISTANT) {
                    body = body.strip_suffix(CONSTRAIN.as_bytes()).unwrap_or(body);
                }
                out.reasoning.extend_from_slice(body);
            }
            FINAL => out.content.extend_from_slice(body),
            _ => {
                let (close_at, close) = end?;
                let call = call_body(name?, std::str::from_utf8(body).ok()?, orders)?;
                out.calls.push(call);
                out.raw_tool_text.push_str(ASSISTANT);
                out.raw_tool_text
                    .push_str(std::str::from_utf8(&raw[pos..close_at + close.len()]).ok()?);
            }
        }
        match end {
            // A bare `<|start|>assistant` closer opens the next message.
            Some((at, ASSISTANT)) => pos = at,
            Some((_, CALL | RETURN)) => break,
            Some((at, closer)) => pos = at + closer.len(),
            None => break,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_answer_and_calls() {
        let raw = b"<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>hi";
        let p = parse(raw, &[]).unwrap();
        assert_eq!(p.reasoning, b"think");
        assert_eq!(p.content, b"hi");
        assert!(p.calls.is_empty());

        let raw = "<|channel|>analysis<|message|>plan<|end|><|start|>assistant to=get_weather<|channel|>analysis \
                   <|constrain|>xml<|message|><rfl:parameter name=\"city\">\nSeoul\n</rfl:parameter>\n\
                   <rfl:parameter name=\"days\">\n3\n</rfl:parameter><|end|><|start|>assistant to=time\
                   <|channel|>analysis <|constrain|>xml<|message|><|call|>";
        let p = parse(raw.as_bytes(), &[]).unwrap();
        assert_eq!(p.reasoning, b"plan");
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[0].name, "get_weather");
        assert_eq!(p.calls[0].arguments, r#"{"city":"Seoul","days":3}"#);
        assert_eq!(p.calls[1].name, "time");
        assert_eq!(p.calls[1].arguments, "{}");
    }

    // Walk the grammar from an empty reply, appending each forced token's text.
    fn walk(mut text: String, tools: bool, steps: usize) -> String {
        let piece = |t: i32| match t {
            T_START => "<|start|>",
            T_ASSISTANT => "assistant",
            T_CHANNEL => "<|channel|>",
            T_MESSAGE => "<|message|>",
            T_CONSTRAIN => "<|constrain|>",
            T_ANALYSIS => "analysis",
            T_FINAL => "final",
            T_SPACE => " ",
            T_XML => "xml",
            _ => unreachable!(),
        };
        for _ in 0..steps {
            match forced_header(text.as_bytes(), tools) {
                Some(t) => text.push_str(piece(t)),
                None => break,
            }
        }
        text
    }

    #[test]
    fn forced_headers() {
        assert_eq!(walk(String::new(), false, 8), ANALYSIS);
        // After analysis: restart the message, then final without tools.
        let answer = walk(format!("{ANALYSIS}plan{END}"), false, 8);
        assert_eq!(answer, format!("{ANALYSIS}plan{END}{ASSISTANT}{FINAL}"));
        // With tools the model picks ` to=` or `<|channel|>` itself.
        assert_eq!(
            walk(format!("{ANALYSIS}plan{END}"), true, 8),
            format!("{ANALYSIS}plan{END}{ASSISTANT}")
        );
        let call = walk(
            format!("{ANALYSIS}plan{END}{ASSISTANT} to=f<|channel|>"),
            true,
            8,
        );
        assert_eq!(call, format!("{ANALYSIS}plan{END}{ASSISTANT} to=f{TOOL}"));
        // Bodies stay sampled.
        assert_eq!(
            forced_header(format!("{ANALYSIS}pl").as_bytes(), false),
            None
        );
    }

    #[test]
    fn truncated_and_malformed() {
        // A call cut by the budget has no terminator.
        let raw = b"<|channel|>analysis<|message|>x<|end|><|start|>assistant to=f<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"a\">\n1";
        assert!(parse(raw, &[]).is_none());
        // A header cut by the budget publishes nothing of it.
        let p = parse(
            b"<|channel|>analysis<|message|>x<|end|><|start|>assistant<|chan",
            &[],
        )
        .unwrap();
        assert_eq!(p.reasoning, b"x");
        assert!(p.content.is_empty());
        // Text outside any channel is not Harmony.
        assert!(parse(b"hello", &[]).is_none());
        assert!(parse(b"junk<|channel|>final<|message|>hi", &[]).is_none());
        // The template's bare closer of an analysis message.
        let p = parse(
            b"<|channel|>analysis<|message|>x<|start|>assistant<|channel|>final<|message|>y",
            &[],
        )
        .unwrap();
        assert_eq!(
            (p.reasoning.as_slice(), p.content.as_slice()),
            (&b"x"[..], &b"y"[..])
        );
    }
}
