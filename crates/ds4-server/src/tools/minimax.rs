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
