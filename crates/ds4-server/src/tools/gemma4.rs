//! Gemma-4 output: reasoning in the `<|channel>thought` channel, then
//! content, then zero or more `<|tool_call>call:NAME{args}<tool_call|>`
//! blocks. Arguments use the template's own syntax (chat_template.jinja):
//! bare keys, `<|"|>`-quoted strings, objects, arrays and JSON literals.
//! Input is rendered by the artifact's Jinja template; this module only
//! reads what the model wrote.

use crate::json::json_escape_bytes;
use crate::parse::ToolCall;
use crate::stream::ChatFormat;

use super::{find_substr, split_reasoning_response, ParsedGenerated};

pub const THOUGHT_OPEN: &str = "<|channel>thought\n";
pub const THOUGHT_CLOSE: &str = "<channel|>";
pub const CALL_OPEN: &str = "<|tool_call>";
pub const CALL_CLOSE: &str = "<tool_call|>";
/// What a forced call starts with: the open marker and the call keyword.
pub const FORCED_CALL: &str = "<|tool_call>call:";

const CALL_KEYWORD: &str = "call:";
const QUOTE: &[u8] = b"<|\"|>";
/// Nesting bound for one argument: a real call never nests this deep, and a
/// runaway emission must not recurse without limit.
const MAX_DEPTH: u32 = 64;

/// Template markers that can only be a degenerate emission once the channel
/// and the calls are taken: a call close with no open, and a turn close the
/// stop list missed.
const STRAY: [&str; 5] = ["<|tool_call|>", CALL_CLOSE, CALL_OPEN, "<turn|>", "<|turn>"];

/// The channel is the reasoning field, so it is split off first; calls are
/// then taken from the rest. A call that does not parse reports `ok: false`
/// with its text, so the caller's malformed-call handling sees it.
pub(crate) fn parse(text: &[u8], require_thinking_closed: bool) -> ParsedGenerated {
    let (body, reasoning) =
        split_reasoning_response(text, ChatFormat::Gemma4, require_thinking_closed);
    let mut content = Vec::new();
    let mut calls = Vec::new();
    let mut pos = 0usize;
    while let Some(at) = find_substr(&body[pos..], CALL_OPEN.as_bytes()) {
        let start = pos + at;
        let args_at = start + CALL_OPEN.len();
        let parsed = find_substr(&body[args_at..], CALL_CLOSE.as_bytes()).and_then(|close| {
            let call = String::from_utf8_lossy(&body[args_at..args_at + close]);
            call_body(&call).map(|call| (call, args_at + close + CALL_CLOSE.len()))
        });
        let Some(((name, arguments), next)) = parsed else {
            content.extend_from_slice(&body[pos..start]);
            return ParsedGenerated {
                content: drop_markers(&content),
                reasoning,
                ok: false,
                raw_tool_text: String::from_utf8_lossy(&body[start..]).into_owned(),
                ..Default::default()
            };
        };
        content.extend_from_slice(&body[pos..start]);
        calls.push(ToolCall {
            id: String::new(),
            name,
            arguments,
        });
        pos = next;
    }
    content.extend_from_slice(&body[pos..]);
    ParsedGenerated {
        content: drop_markers(&content),
        reasoning,
        calls,
        ok: true,
        ..Default::default()
    }
}

fn drop_markers(content: &[u8]) -> Vec<u8> {
    let mut out = String::from_utf8_lossy(content).into_owned();
    for marker in STRAY {
        if out.contains(marker) {
            out = out.replace(marker, "");
        }
    }
    out.into_bytes()
}

/// One call body (`call:NAME{args}`, the text between the call markers) into
/// (name, arguments as JSON). None when the body is not the template's syntax.
pub(crate) fn call_body(body: &str) -> Option<(String, String)> {
    let body = body.strip_prefix(CALL_KEYWORD).unwrap_or(body);
    let (name, rest) = body.split_once('{')?;
    let mut args = Args {
        s: rest.as_bytes(),
        i: 0,
    };
    let value = args.object(0)?;
    args.ws();
    if args.i != args.s.len() {
        return None;
    }
    Some((name.trim().to_string(), to_json(&value)))
}

/// An argument value; object members keep the key order `to_json` sorts.
enum Value {
    Str(String),
    Object(Vec<(String, Value)>),
    Array(Vec<Value>),
    /// A JSON literal or number, already validated.
    Literal(String),
}

struct Args<'a> {
    s: &'a [u8],
    i: usize,
}

impl Args<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    /// `<|"|>...<|"|>`; None if the closing quote never comes.
    fn quoted(&mut self) -> Option<String> {
        self.i += QUOTE.len();
        let len = find_substr(&self.s[self.i..], QUOTE)?;
        let out = String::from_utf8_lossy(&self.s[self.i..self.i + len]).into_owned();
        self.i += len + QUOTE.len();
        Some(out)
    }

    /// The members after an object's `{`, through its `}`.
    fn object(&mut self, depth: u32) -> Option<Value> {
        let mut members = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Some(Value::Object(members));
        }
        loop {
            self.ws();
            let start = self.i;
            while self
                .peek()
                .is_some_and(|c| !matches!(c, b':' | b'}' | b','))
            {
                self.i += 1;
            }
            if self.peek() != Some(b':') || self.i == start {
                return None;
            }
            let key = String::from_utf8_lossy(&self.s[start..self.i])
                .trim()
                .to_owned();
            self.i += 1;
            members.push((key, self.value(depth + 1)?));
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(Value::Object(members));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self, depth: u32) -> Option<Value> {
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(Value::Array(items));
                }
                _ => return None,
            }
        }
    }

    fn value(&mut self, depth: u32) -> Option<Value> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        if self.s[self.i..].starts_with(QUOTE) {
            return self.quoted().map(Value::Str);
        }
        match self.peek()? {
            b'{' => {
                self.i += 1;
                self.object(depth)
            }
            b'[' => {
                self.i += 1;
                self.array(depth)
            }
            _ => self.literal(),
        }
    }

    /// A number, `true`, `false` or `null`: the run up to the next separator,
    /// kept only when it is valid JSON on its own.
    fn literal(&mut self) -> Option<Value> {
        let start = self.i;
        while self
            .peek()
            .is_some_and(|c| !matches!(c, b',' | b'}' | b']') && !c.is_ascii_whitespace())
        {
            self.i += 1;
        }
        let token = std::str::from_utf8(&self.s[start..self.i]).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(token).ok()?;
        if parsed.is_object() || parsed.is_array() || parsed.is_string() {
            return None;
        }
        Some(Value::Literal(token.to_owned()))
    }
}

fn json_string(s: &str) -> String {
    String::from_utf8_lossy(&json_escape_bytes(s.as_bytes())).into_owned()
}

/// JSON text with object keys in sorted order (the template's `dictsort`).
fn to_json(value: &Value) -> String {
    match value {
        Value::Str(s) => json_string(s),
        Value::Literal(token) => token.clone(),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(to_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Object(members) => {
            let mut sorted: Vec<&(String, Value)> = members.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let members: Vec<String> = sorted
                .iter()
                .map(|(key, value)| format!("{}:{}", json_string(key), to_json(value)))
                .collect();
            format!("{{{}}}", members.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{call_body, parse};

    #[test]
    fn call_body_round_trips() {
        let (name, args) = call_body("call:lookup{q:<|\"|>a b<|\"|>,n:2}").unwrap();
        assert_eq!(name, "lookup");
        assert_eq!(args, "{\"n\":2,\"q\":\"a b\"}");
    }

    #[test]
    fn nested_values_keep_their_types() {
        let (_, args) = call_body("call:f{o:{z:[1,true,null],a:<|\"|>x<|\"|>},e:{},l:[]}").unwrap();
        assert_eq!(
            args,
            "{\"e\":{},\"l\":[],\"o\":{\"a\":\"x\",\"z\":[1,true,null]}}"
        );
    }

    #[test]
    fn malformed_bodies_end_without_a_call() {
        // Each of these used to spin forever looking for a separator.
        for body in [
            "call:f{a}",
            "call:f{<|\"|>a<|\"|>}",
            "call:f{a:1 b}",
            "call:f{a:{bc}",
            "call:f{a:trueish}",
            "call:f{a:1.2.3}",
            "call:f{a:-}",
            "call:f{a:<|\"|>open}",
            "call:f{:1}",
            "call:f{a:1}}",
        ] {
            assert!(call_body(body).is_none(), "{body}");
        }
    }

    #[test]
    fn deep_nesting_is_refused() {
        let body = format!("call:f{{a:{}1{}}}", "[".repeat(100), "]".repeat(100));
        assert!(call_body(&body).is_none());
    }

    #[test]
    fn a_malformed_call_reports_not_ok_and_keeps_the_channel_off() {
        let parsed = parse(
            b"<|channel>thought\nplan\n<channel|>Sure.<|tool_call>call:f{a}<tool_call|>",
            true,
        );
        assert!(!parsed.ok);
        assert_eq!(parsed.content, b"Sure.");
        assert_eq!(parsed.reasoning, b"plan\n");
        assert_eq!(parsed.raw_tool_text, "<|tool_call>call:f{a}<tool_call|>");
    }
}
