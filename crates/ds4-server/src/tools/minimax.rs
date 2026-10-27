//! MiniMax-M3.1 tool calls: the inverse of chat_template.jinja's `to_xml`.
//! Every tag carries the namespace token; a call is
//! `NS<invoke name="N">NS<k>v NS</k>…NS</invoke>`, dicts nest as `NS<key>`,
//! lists as `NS<item>`, booleans are JSON and other scalars print bare.
//! Top-level values follow the declared schema type; nested leaves only
//! become JSON scalars (numbers, booleans, null), anything else stays text.
use serde_json::{Map, Value};

use crate::parse::{ToolCall, ToolSchemaOrder};

use super::{
    find_substr, tool_schema_order_prop_type, tool_schema_order_prop_type_first,
    tool_schema_orders_find, ParsedGenerated,
};

pub const NS: &str = "]<]minimax[>[";
pub const TOOL_CALL_START: &str = "]<]minimax[>[<tool_call>";
pub const TOOL_CALL_END: &str = "]<]minimax[>[</tool_call>";
pub const THINK: &str = "<mm:think>";
pub const THINK_END: &str = "</mm:think>";
const INVOKE: &str = "]<]minimax[>[<invoke name=\"";

enum Node {
    Text(String),
    Tags(Vec<(String, Node)>),
}

// The body after `NS<name>` through its `NS</name>`, and what follows.
fn node<'a>(s: &'a str, name: &str) -> Option<(Node, &'a str)> {
    let close = format!("{NS}</{name}>");
    let lead = s.trim_start();
    if !lead.starts_with(NS) || lead[NS.len()..].starts_with("</") {
        let at = s.find(&close)?;
        return Some((Node::Text(s[..at].to_owned()), &s[at + close.len()..]));
    }
    let mut tags = Vec::new();
    let mut rest = lead;
    loop {
        if let Some(after) = rest.strip_prefix(close.as_str()) {
            return Some((Node::Tags(tags), after));
        }
        let open = rest.strip_prefix(NS)?.strip_prefix('<')?;
        let end = open.find('>')?;
        let tag = &open[..end];
        if tag.is_empty() || tag.starts_with('/') {
            return None;
        }
        let (child, after) = node(&open[end + 1..], tag)?;
        tags.push((tag.to_owned(), child));
        rest = after.trim_start();
    }
}

fn value(node: Node, ty: Option<&str>) -> Value {
    match node {
        Node::Text(text) => match ty {
            Some("string") => Value::String(text),
            Some("array") if text.trim().is_empty() => Value::Array(Vec::new()),
            Some("object") if text.trim().is_empty() => Value::Object(Map::new()),
            Some(_) => serde_json::from_str(text.trim()).unwrap_or(Value::String(text)),
            None => match serde_json::from_str(text.trim()) {
                Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Null)) => v,
                _ => Value::String(text),
            },
        },
        Node::Tags(tags) => {
            if ty != Some("object") && !tags.is_empty() && tags.iter().all(|(t, _)| t == "item") {
                Value::Array(tags.into_iter().map(|(_, n)| value(n, None)).collect())
            } else {
                Value::Object(tags.into_iter().map(|(t, n)| (t, value(n, None))).collect())
            }
        }
    }
}

// One `NS<invoke name="…">…NS</invoke>`; `s` starts after the opening quote.
fn invoke<'a>(s: &'a str, orders: &[ToolSchemaOrder]) -> Option<(ToolCall, &'a str)> {
    let end = s.find("\">")?;
    let name = s[..end].trim();
    if name.is_empty() {
        return None;
    }
    let (body, rest) = node(&s[end + 2..], "invoke")?;
    let order = tool_schema_orders_find(orders, name);
    let mut args = Map::new();
    match body {
        Node::Tags(tags) => {
            for (key, n) in tags {
                let ty = tool_schema_order_prop_type(order, &key)
                    .filter(|t| !t.is_empty())
                    .or_else(|| tool_schema_order_prop_type_first(order, &key));
                args.insert(key, value(n, ty));
            }
        }
        Node::Text(text) if text.trim().is_empty() => {}
        Node::Text(_) => return None,
    }
    let call = ToolCall {
        name: name.to_owned(),
        arguments: Value::Object(args).to_string(),
        ..Default::default()
    };
    Some((call, rest))
}

// One tool block after its start marker: the calls and the bytes consumed.
fn block(s: &str, orders: &[ToolSchemaOrder]) -> Option<(Vec<ToolCall>, usize)> {
    let mut calls = Vec::new();
    let mut rest = s.trim_start();
    loop {
        if let Some(after) = rest.strip_prefix(TOOL_CALL_END) {
            return Some((calls, s.len() - after.len()));
        }
        let (call, after) = invoke(rest.strip_prefix(INVOKE)?, orders)?;
        calls.push(call);
        rest = after.trim_start();
    }
}

pub(super) fn parse(
    text: &[u8],
    require_thinking_closed: bool,
    orders: &[ToolSchemaOrder],
) -> ParsedGenerated {
    let mut out = ParsedGenerated {
        ok: true,
        ..Default::default()
    };
    let mut body = text;
    if require_thinking_closed {
        let raw = body.strip_prefix(THINK.as_bytes()).unwrap_or(body);
        let Some(end) = find_substr(raw, THINK_END.as_bytes()) else {
            out.reasoning = raw.to_vec();
            return out;
        };
        out.reasoning = raw[..end].to_vec();
        body = &raw[end + THINK_END.len()..];
    }
    let Ok(body) = std::str::from_utf8(body) else {
        out.ok = false;
        return out;
    };
    let mut kept = 0;
    while let Some(rel) = body[kept..].find(TOOL_CALL_START) {
        let open = kept + rel;
        let first = open + TOOL_CALL_START.len();
        let Some((calls, used)) = block(&body[first..], orders) else {
            out.ok = false;
            return out;
        };
        out.content.extend_from_slice(body[kept..open].as_bytes());
        out.raw_tool_text.push_str(&body[open..first + used]);
        out.calls.extend(calls);
        kept = first + used;
    }
    out.content.extend_from_slice(body[kept..].as_bytes());
    out
}

#[derive(Debug)]
enum StreamState {
    Block,
    Invoke,
    Param,
    String(String),
    Failed,
}

/// M3 XML decoding only; the shared sink owns IDs' wire shape and SSE framing.
#[derive(Debug)]
pub(crate) struct Stream {
    pos: usize,
    state: StreamState,
    name: String,
    first: bool,
}

impl Stream {
    pub(crate) fn new(pos: usize) -> Self {
        Self {
            pos,
            state: StreamState::Block,
            name: String::new(),
            first: true,
        }
    }

    pub(crate) fn update(
        &mut self,
        sink: &mut impl crate::tool_stream::ToolSink,
        ids: &mut crate::tool_stream::DsmlToolStream,
        raw: &[u8],
        orders: &[ToolSchemaOrder],
    ) -> bool {
        while self.pos < raw.len() {
            let tail = &raw[self.pos..];
            match &self.state {
                StreamState::Block => {
                    let Some(rel) = find_substr(tail, TOOL_CALL_START.as_bytes()) else {
                        return true;
                    };
                    self.pos += rel + TOOL_CALL_START.len();
                    self.state = StreamState::Invoke;
                }
                StreamState::Invoke | StreamState::Param => {
                    if tail[0].is_ascii_whitespace() {
                        self.pos += 1;
                        continue;
                    }
                    if matches!(self.state, StreamState::Invoke) {
                        if tail.starts_with(TOOL_CALL_END.as_bytes()) {
                            self.pos += TOOL_CALL_END.len();
                            self.state = StreamState::Block;
                            continue;
                        }
                        if INVOKE.as_bytes().starts_with(tail)
                            || TOOL_CALL_END.as_bytes().starts_with(tail)
                        {
                            return true;
                        }
                        let Some(head) = tail.strip_prefix(INVOKE.as_bytes()) else {
                            self.state = StreamState::Failed;
                            continue;
                        };
                        let Some(end) = find_substr(head, b"\">") else {
                            return true;
                        };
                        let Ok(name) = std::str::from_utf8(&head[..end]) else {
                            return true;
                        };
                        self.name = name.trim().to_owned();
                        if self.name.is_empty() {
                            self.state = StreamState::Failed;
                            continue;
                        }
                        let id = ids.id_at(ids.index);
                        if !sink.start_invoke(ids.index, &id, self.name.as_bytes())
                            || !sink.args_fragment(ids.index, b"{")
                        {
                            return false;
                        }
                        ids.emitted_any = true;
                        self.first = true;
                        self.pos += INVOKE.len() + end + 2;
                        self.state = StreamState::Param;
                        continue;
                    }
                    let close = format!("{NS}</invoke>");
                    if tail.starts_with(close.as_bytes()) {
                        if !sink.args_fragment(ids.index, b"}") || !sink.close_invoke(ids.index) {
                            return false;
                        }
                        ids.index += 1;
                        self.pos += close.len();
                        self.state = StreamState::Invoke;
                        continue;
                    }
                    let Some(open) = tail
                        .strip_prefix(NS.as_bytes())
                        .and_then(|s| s.strip_prefix(b"<"))
                    else {
                        return true;
                    };
                    let Some(end) = open.iter().position(|&c| c == b'>') else {
                        return true;
                    };
                    let Ok(key) = std::str::from_utf8(&open[..end]) else {
                        return true;
                    };
                    let header = NS.len() + 1 + end + 1;
                    if key.is_empty() || key.starts_with('/') {
                        self.state = StreamState::Failed;
                        continue;
                    }
                    let order = tool_schema_orders_find(orders, &self.name);
                    let ty = tool_schema_order_prop_type(order, key)
                        .filter(|t| !t.is_empty())
                        .or_else(|| tool_schema_order_prop_type_first(order, key));
                    let mut string = ty == Some("string");
                    if string {
                        let limit = crate::stream::utf8_stream_safe_len(
                            raw,
                            self.pos + header,
                            raw.len(),
                            false,
                        );
                        let Ok(body) = std::str::from_utf8(&raw[self.pos + header..limit]) else {
                            return true;
                        };
                        let lead = body.trim_start().as_bytes();
                        if lead.is_empty() || NS.as_bytes().starts_with(lead) {
                            return true;
                        }
                        if let Some(tag) = lead.strip_prefix(NS.as_bytes()) {
                            if b"</".starts_with(tag) {
                                return true;
                            }
                            string = tag.starts_with(b"</");
                        }
                    }
                    let mut fragment = if self.first { Vec::new() } else { vec![b','] };
                    fragment.extend(crate::json::json_escape_bytes(key.as_bytes()));
                    fragment.push(b':');
                    if string {
                        fragment.push(b'"');
                        self.pos += header;
                        self.state = StreamState::String(format!("{NS}</{key}>"));
                    } else {
                        // Complex and untyped values use the final parser's conversion.
                        let Ok(text) = std::str::from_utf8(&raw[self.pos + header..]) else {
                            return true;
                        };
                        let Some((n, rest)) = node(text, key) else {
                            return true;
                        };
                        fragment.extend(value(n, ty).to_string().as_bytes());
                        self.pos = raw.len() - rest.len();
                    }
                    if !sink.args_fragment(ids.index, &fragment) {
                        return false;
                    }
                    self.first = false;
                }
                StreamState::String(close) => {
                    let end = find_substr(tail, close.as_bytes());
                    // Keep both a split close tag and a split UTF-8 character off the wire.
                    let keep = (1..close.len().min(tail.len() + 1))
                        .rev()
                        .find(|&n| tail.ends_with(&close.as_bytes()[..n]))
                        .unwrap_or(0);
                    let limit = self.pos + end.unwrap_or(tail.len() - keep);
                    let limit = crate::stream::utf8_stream_safe_len(raw, self.pos, limit, false);
                    if limit > self.pos {
                        let escaped = crate::json::json_escape_bytes(&raw[self.pos..limit]);
                        if !sink.args_fragment(ids.index, &escaped[1..escaped.len() - 1]) {
                            return false;
                        }
                        self.pos = limit;
                    }
                    if end.is_none() {
                        return true;
                    }
                    if !sink.args_fragment(ids.index, b"\"") {
                        return false;
                    }
                    self.pos += close.len();
                    self.state = StreamState::Param;
                }
                StreamState::Failed => return true,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(props: &[(&str, &str)]) -> ToolSchemaOrder {
        ToolSchemaOrder {
            name: "f".into(),
            prop: props.iter().map(|p| p.0.to_owned()).collect(),
            prop_type: props.iter().map(|p| p.1.to_owned()).collect(),
            prop_type_first: props.iter().map(|_| String::new()).collect(),
            ..Default::default()
        }
    }

    // Bytes rendered by chat_template.jinja for
    // f(city="Seoul", n=3, tags=["a", 2], opts={"deep": true, "s": "7"}, code="42").
    const CALL: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">\
]<]minimax[>[<city>Seoul]<]minimax[>[</city>]<]minimax[>[<n>3]<]minimax[>[</n>\
]<]minimax[>[<tags>]<]minimax[>[<item>a]<]minimax[>[</item>]<]minimax[>[<item>2]<]minimax[>[</item>]<]minimax[>[</tags>\
]<]minimax[>[<opts>]<]minimax[>[<deep>true]<]minimax[>[</deep>]<]minimax[>[<s>x y]<]minimax[>[</s>]<]minimax[>[</opts>\
]<]minimax[>[<code>42]<]minimax[>[</code>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";

    #[test]
    fn template_round_trip() {
        let orders = [order(&[
            ("city", "string"),
            ("n", "integer"),
            ("tags", "array"),
            ("opts", "object"),
            ("code", "string"),
        ])];
        let text = format!("<mm:think>plan</mm:think>Let me check.{CALL}");
        let p = parse(text.as_bytes(), true, &orders);
        assert!(p.ok);
        assert_eq!(p.reasoning, b"plan");
        assert_eq!(p.content, b"Let me check.");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "f");
        let args: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"city": "Seoul", "n": 3, "tags": ["a", 2],
                "opts": {"deep": true, "s": "x y"}, "code": "42"})
        );
        assert_eq!(p.raw_tool_text, CALL);
    }

    #[test]
    fn two_invokes_no_args_and_bare_close() {
        let text = "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"a\">]<]minimax[>[</invoke>\n\
]<]minimax[>[<invoke name=\"b\">]<]minimax[>[<x>]<]minimax[>[<item>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[</item>]<]minimax[>[</x>]<]minimax[>[</invoke>\n\
]<]minimax[>[</tool_call>";
        let p = parse(text.as_bytes(), true, &[]);
        assert!(p.ok);
        assert!(p.reasoning.is_empty() && p.content.is_empty());
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[0].arguments, "{}");
        assert_eq!(p.calls[1].arguments, r#"{"x":[[1]]}"#);
    }

    #[test]
    fn malformed_block_and_open_thinking() {
        let bad = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"a\">]<]minimax[>[<x>1";
        assert!(!parse(bad.as_bytes(), false, &[]).ok);
        let open = parse(b"<mm:think>still thinking", true, &[]);
        assert!(open.ok && open.content.is_empty());
        assert_eq!(open.reasoning, b"still thinking");
    }

    #[test]
    fn malformed_prefix_is_not_a_call() {
        // vLLM, SGLang and TRT-LLM require the full wrapper too (#116).
        for text in [
            "]<]minimax[>[ slide",
            "]<]minimax[>[<invoke native pptx table",
            "]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke>",
        ] {
            let p = parse(text.as_bytes(), false, &[]);
            assert!(p.ok && p.calls.is_empty());
            assert_eq!(p.content, text.as_bytes());
        }
    }

    #[test]
    fn stream_matches_final_values() {
        #[derive(Default)]
        struct Sink(Vec<ToolCall>);
        impl crate::tool_stream::ToolSink for Sink {
            fn start_invoke(&mut self, index: i32, id: &str, name: &[u8]) -> bool {
                assert_eq!(index as usize, self.0.len());
                self.0.push(ToolCall {
                    id: id.into(),
                    name: std::str::from_utf8(name).unwrap().into(),
                    ..Default::default()
                });
                true
            }
            fn args_fragment(&mut self, index: i32, bytes: &[u8]) -> bool {
                self.0[index as usize]
                    .arguments
                    .push_str(std::str::from_utf8(bytes).unwrap());
                true
            }
            fn close_invoke(&mut self, _: i32) -> bool {
                true
            }
        }
        let call = CALL
            .replace("Seoul", "서울")
            .replace(">42", ">a\"\\\n]<]mini");
        let nested = call.replace("서울", "]<]minimax[>[<child>7]<]minimax[>[</child>");
        let raw = format!("{call}\n{nested}");
        let orders = [order(&[
            ("city", "string"),
            ("n", "integer"),
            ("tags", "array"),
            ("opts", "object"),
            ("code", "string"),
        ])];
        let expected = parse(raw.as_bytes(), false, &orders);
        assert!(expected.ok && expected.calls.len() == 2);
        for width in [1, 2, 7, raw.len()] {
            let mut stream = Stream::new(0);
            let mut ids = crate::tool_stream::DsmlToolStream::default();
            let mut sink = Sink::default();
            for end in (width..raw.len()).step_by(width).chain([raw.len()]) {
                assert!(stream.update(&mut sink, &mut ids, &raw.as_bytes()[..end], &orders));
            }
            assert_eq!(sink.0.len(), expected.calls.len());
            for (actual, expected) in sink.0.iter().zip(&expected.calls) {
                assert_eq!(actual.name, expected.name);
                assert_eq!(
                    serde_json::from_str::<Value>(&actual.arguments).unwrap(),
                    serde_json::from_str::<Value>(&expected.arguments).unwrap()
                );
            }
            assert_ne!(sink.0[0].id, sink.0[1].id);
        }
    }

    // The namespace token arrives whole; the stream must hold it back until
    // the next token shows whether a tool block opens.
    #[test]
    fn stream_holds_the_namespace_token() {
        use crate::stream::{text_stream_safe_limit, ChatFormat};
        let raw = b"Sure. ]<]minimax[>[";
        let at = text_stream_safe_limit(raw, 0, raw.len(), true, false, ChatFormat::Minimax, &[]);
        assert_eq!(&raw[..at], b"Sure.");
    }
}
