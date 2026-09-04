//! Generated-message tool parse + serial semantic accumulator from
//! `ds4_server.c` at v0.6.5-dfm. DSML greedy / required-prefix sampling
//! and corrective retry (`retry`) are host-owned.

use crate::dsml::{DsmlDecodeState, DsmlDecodeTracker, SampleOverride, SamplePolicy};
use crate::json::{json_escape_bytes, json_minify_raw_value, json_raw_value, json_string, Json};
use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::render::{
    syntax_for_model_id, ModelSyntax, GLM_TOOL_CALL_END, GLM_TOOL_CALL_START, QWEN_TOOL_CALL_END,
    QWEN_TOOL_CALL_START, SOLAR_THINK_END, SOLAR_THINK_START, SOLAR_TOOL_ARG_END,
    SOLAR_TOOL_ARG_START, SOLAR_TOOL_ARG_VALUE, SOLAR_TOOL_CALLS, SOLAR_TOOL_CALL_END,
};
use crate::stream::{think_end, think_start, ChatFormat};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read as _;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DSML_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
pub const DSML_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
pub const DSML_INVOKE_START: &str = "<｜DSML｜invoke";
pub const DSML_INVOKE_END: &str = "</｜DSML｜invoke>";
pub const DSML_PARAM_START: &str = "<｜DSML｜parameter";
pub const DSML_PARAM_END: &str = "</｜DSML｜parameter>";
pub const DSML_TOOL_CALLS_START_SHORT: &str = "<DSML｜tool_calls>";
pub const DSML_TOOL_CALLS_END_SHORT: &str = "</DSML｜tool_calls>";
pub const DSML_INVOKE_START_SHORT: &str = "<DSML｜invoke";
pub const DSML_INVOKE_END_SHORT: &str = "</DSML｜invoke>";
pub const DSML_PARAM_START_SHORT: &str = "<DSML｜parameter";
pub const DSML_PARAM_END_SHORT: &str = "</DSML｜parameter>";

#[derive(Debug, Clone, Default)]
pub struct ParsedGenerated {
    pub content: Vec<u8>,
    pub reasoning: Vec<u8>,
    pub calls: Vec<ToolCall>,
    pub raw_dsml: String,
    pub raw_tool_text: String,
    pub ok: bool,
    pub recovered: bool,
}

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn find_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn find_last_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len())
        .enumerate()
        .rev()
        .find(|(_, w)| *w == needle)
        .map(|(i, _)| i)
}

fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    i
}

fn trim_tool_separator_ws(raw: &[u8], start: usize, mut limit: usize) -> usize {
    while limit > start && is_c_space(raw[limit - 1]) {
        limit -= 1;
    }
    limit
}

fn trim_ascii_span(s: &[u8]) -> &[u8] {
    let mut a = 0;
    let mut b = s.len();
    while a < b && is_c_space(s[a]) {
        a += 1;
    }
    while b > a && is_c_space(s[b - 1]) {
        b -= 1;
    }
    &s[a..b]
}

pub(crate) fn dsml_unescape_text(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if s[i] != b'&' {
            out.push(s[i]);
            i += 1;
            continue;
        }
        if s[i..].starts_with(b"&amp;") {
            out.push(b'&');
            i += 5;
        } else if s[i..].starts_with(b"&lt;") {
            out.push(b'<');
            i += 4;
        } else if s[i..].starts_with(b"&gt;") {
            out.push(b'>');
            i += 4;
        } else if s[i..].starts_with(b"&quot;") {
            out.push(b'"');
            i += 6;
        } else if s[i..].starts_with(b"&apos;") {
            out.push(b'\'');
            i += 6;
        } else {
            out.push(b'&');
            i += 1;
        }
    }
    out
}

pub(crate) fn dsml_attr(tag: &[u8], name: &str) -> Option<Vec<u8>> {
    let mut pat = name.as_bytes().to_vec();
    pat.extend_from_slice(b"=\"");
    let p = find_substr(tag, &pat)?;
    let start = p + pat.len();
    let rel = tag[start..].iter().position(|&c| c == b'"')?;
    Some(dsml_unescape_text(&tag[start..start + rel]))
}

fn tool_call_json_args_add(args: &mut Vec<u8>, name: &[u8], value: &[u8], is_string: bool) {
    if !args.is_empty() {
        args.extend_from_slice(b", ");
    }
    args.extend(json_escape_bytes(name));
    args.extend_from_slice(b": ");
    if is_string {
        args.extend(json_escape_bytes(value));
    } else {
        let min = json_minify_raw_value(&String::from_utf8_lossy(value));
        if min.is_empty() {
            args.extend_from_slice(b"null");
        } else {
            args.extend_from_slice(min.as_bytes());
        }
    }
}

fn split_reasoning_content(text: &[u8], n: usize, format: ChatFormat) -> (Vec<u8>, Vec<u8>) {
    let s = &text[..n.min(text.len())];
    let start = think_start(format).as_bytes();
    let end = think_end(format).as_bytes();
    let body = if s.starts_with(start) {
        &s[start.len()..]
    } else {
        s
    };
    if let Some(i) = find_substr(body, end) {
        (body[i + end.len()..].to_vec(), body[..i].to_vec())
    } else {
        (s.to_vec(), Vec::new())
    }
}

fn unterminated_reasoning(text: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let body = if text.starts_with(b"<think>") {
        &text[7..]
    } else {
        text
    };
    (Vec::new(), body.to_vec())
}

fn unterminated_reasoning_before_tool(text: &[u8], mut prefix_len: usize) -> (Vec<u8>, Vec<u8>) {
    prefix_len = prefix_len.min(text.len());
    let (body, plen) = if prefix_len >= 7 && text.starts_with(b"<think>") {
        (&text[7..], prefix_len - 7)
    } else {
        (text, prefix_len)
    };
    (Vec::new(), body[..plen.min(body.len())].to_vec())
}

fn dsml_parse_leaf_param(
    s: &[u8],
    i: &mut usize,
    param_start: &[u8],
    param_end: &[u8],
    out: &mut Vec<u8>,
) -> bool {
    if !s[*i..].starts_with(param_start) {
        return false;
    }
    let Some(gt) = s[*i..].iter().position(|&c| c == b'>') else {
        return false;
    };
    let tag_end = *i + gt;
    let tag = &s[*i..=tag_end];
    let Some(name) = dsml_attr(tag, "name") else {
        return false;
    };
    let is_string_attr = dsml_attr(tag, "string");
    let value_start = tag_end + 1;
    let Some(rel) = find_substr(&s[value_start..], param_end) else {
        return false;
    };
    let raw = &s[value_start..value_start + rel];
    let is_string = is_string_attr
        .as_deref()
        .map(|v| v == b"true")
        .unwrap_or(true);
    let value = if is_string {
        dsml_unescape_text(raw)
    } else {
        raw.to_vec()
    };
    tool_call_json_args_add(out, &name, &value, is_string);
    *i = value_start + rel + param_end.len();
    true
}

fn dsml_parse_nested_params_object(
    s: &[u8],
    i: &mut usize,
    param_start: &[u8],
    param_end: &[u8],
) -> Option<Vec<u8>> {
    let mut members = Vec::new();
    let mut any = false;
    loop {
        *i = skip_ws(s, *i);
        if !s[*i..].starts_with(param_start) {
            break;
        }
        if !dsml_parse_leaf_param(s, i, param_start, param_end, &mut members) {
            return None;
        }
        any = true;
    }
    if !any {
        return None;
    }
    let mut out = vec![b'{'];
    out.extend_from_slice(&members);
    out.push(b'}');
    Some(out)
}

struct DsmlStyle {
    tool_calls_start: &'static [u8],
    tool_calls_end: &'static [u8],
    invoke_start: &'static [u8],
    invoke_end: &'static [u8],
    param_start: &'static [u8],
    param_end: &'static [u8],
}

fn parse_dsml_generated(text: &[u8], require_thinking_closed: bool) -> Option<ParsedGenerated> {
    let mut tool_search = 0usize;
    if require_thinking_closed {
        match find_last_substr(text, b"</think>") {
            None => {
                let (content, reasoning) = unterminated_reasoning(text);
                return Some(ParsedGenerated {
                    content,
                    reasoning,
                    ok: true,
                    ..Default::default()
                });
            }
            Some(i) => tool_search = i + 8,
        }
    }
    let search = &text[tool_search..];
    let nl_full = format!("\n\n{DSML_TOOL_CALLS_START}");
    let nl_short = format!("\n\n{DSML_TOOL_CALLS_START_SHORT}");
    let styles: [(&[u8], i32); 6] = [
        (nl_full.as_bytes(), 0),
        (DSML_TOOL_CALLS_START.as_bytes(), 0),
        (nl_short.as_bytes(), 2),
        (DSML_TOOL_CALLS_START_SHORT.as_bytes(), 2),
        (b"\n\n<tool_calls>", 1),
        (b"<tool_calls>", 1),
    ];
    let mut start_rel = None;
    let mut style = 0i32;
    for (pat, st) in styles {
        if let Some(i) = find_substr(search, pat) {
            start_rel = Some(i);
            style = st;
            break;
        }
    }
    let Some(rel) = start_rel else {
        let (content, reasoning) = split_reasoning_content(text, text.len(), ChatFormat::DeepSeek);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let content_len = trim_tool_separator_ws(text, 0, start);
    let sty = match style {
        1 => DsmlStyle {
            tool_calls_start: b"<tool_calls>",
            tool_calls_end: b"</tool_calls>",
            invoke_start: b"<invoke",
            invoke_end: b"</invoke>",
            param_start: b"<parameter",
            param_end: b"</parameter>",
        },
        2 => DsmlStyle {
            tool_calls_start: DSML_TOOL_CALLS_START_SHORT.as_bytes(),
            tool_calls_end: DSML_TOOL_CALLS_END_SHORT.as_bytes(),
            invoke_start: DSML_INVOKE_START_SHORT.as_bytes(),
            invoke_end: DSML_INVOKE_END_SHORT.as_bytes(),
            param_start: DSML_PARAM_START_SHORT.as_bytes(),
            param_end: DSML_PARAM_END_SHORT.as_bytes(),
        },
        _ => DsmlStyle {
            tool_calls_start: DSML_TOOL_CALLS_START.as_bytes(),
            tool_calls_end: DSML_TOOL_CALLS_END.as_bytes(),
            invoke_start: DSML_INVOKE_START.as_bytes(),
            invoke_end: DSML_INVOKE_END.as_bytes(),
            param_start: DSML_PARAM_START.as_bytes(),
            param_end: DSML_PARAM_END.as_bytes(),
        },
    };
    let Some(p0) = find_substr(&text[start..], sty.tool_calls_start) else {
        return None;
    };
    let mut p = start + p0 + sty.tool_calls_start.len();
    let mut calls = Vec::new();
    loop {
        p = skip_ws(text, p);
        if text[p..].starts_with(sty.tool_calls_end) {
            let raw_end = p + sty.tool_calls_end.len();
            let (content, reasoning) =
                split_reasoning_content(text, content_len, ChatFormat::DeepSeek);
            return Some(ParsedGenerated {
                content,
                reasoning,
                calls,
                raw_dsml: String::from_utf8_lossy(&text[start..raw_end]).into_owned(),
                ok: true,
                ..Default::default()
            });
        }
        if !text[p..].starts_with(sty.invoke_start) {
            return None;
        }
        let Some(gt) = text[p..].iter().position(|&c| c == b'>') else {
            return None;
        };
        let tag = &text[p..=p + gt];
        let Some(name) = dsml_attr(tag, "name") else {
            return None;
        };
        p = p + gt + 1;
        let mut args = Vec::new();
        loop {
            p = skip_ws(text, p);
            if text[p..].starts_with(sty.invoke_end) {
                p += sty.invoke_end.len();
                break;
            }
            if !text[p..].starts_with(sty.param_start) {
                return None;
            }
            let Some(gt) = text[p..].iter().position(|&c| c == b'>') else {
                return None;
            };
            let tag = &text[p..=p + gt];
            let Some(param_name) = dsml_attr(tag, "name") else {
                return None;
            };
            let param_is_string = dsml_attr(tag, "string");
            let value_start = p + gt + 1;
            let after_ws = skip_ws(text, value_start);
            if param_is_string.is_none() && text[after_ws..].starts_with(sty.param_start) {
                let mut nested_p = value_start;
                let Some(nested) = dsml_parse_nested_params_object(
                    text,
                    &mut nested_p,
                    sty.param_start,
                    sty.param_end,
                ) else {
                    return None;
                };
                tool_call_json_args_add(&mut args, &param_name, &nested, false);
                p = skip_ws(text, nested_p);
                if text[p..].starts_with(sty.param_end) {
                    p += sty.param_end.len();
                }
                continue;
            }
            let Some(rel) = find_substr(&text[value_start..], sty.param_end) else {
                return None;
            };
            let raw = &text[value_start..value_start + rel];
            let is_string = param_is_string
                .as_deref()
                .map(|v| v == b"true")
                .unwrap_or(true);
            let value = if is_string {
                dsml_unescape_text(raw)
            } else {
                raw.to_vec()
            };
            tool_call_json_args_add(&mut args, &param_name, &value, is_string);
            p = value_start + rel + sty.param_end.len();
        }
        let mut wrapped = vec![b'{'];
        wrapped.extend_from_slice(&args);
        wrapped.push(b'}');
        calls.push(ToolCall {
            name: String::from_utf8_lossy(&name).into_owned(),
            arguments: String::from_utf8_lossy(&wrapped).into_owned(),
            ..Default::default()
        });
    }
}

fn parse_hermes_tool_call_json(json: &str) -> Option<ToolCall> {
    let mut p = Json::new(json);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    p.ws();
    let mut tc = ToolCall::default();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(&mut p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "name" {
            tc.name = json_string(&mut p)?;
        } else if key == "arguments" {
            tc.arguments = json_raw_value(&mut p)?;
        } else if key == "id" {
            tc.id = json_string(&mut p)?;
        } else if !crate::json::json_skip_value(&mut p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        return None;
    }
    p.ws();
    if p.peek().is_some() || tc.name.is_empty() {
        return None;
    }
    if tc.arguments.is_empty() {
        tc.arguments = "null".into();
    }
    Some(tc)
}

fn parse_hermes_generated(text: &[u8], require_thinking_closed: bool) -> Option<ParsedGenerated> {
    let tool_start = b"<tool_call>";
    let tool_end = b"</tool_call>";
    let mut tool_search = 0usize;
    let mut recovered = false;
    if require_thinking_closed {
        match find_last_substr(text, b"</think>") {
            None => {
                let candidate = find_substr(text, tool_start);
                if candidate.is_none()
                    || find_substr(&text[candidate.unwrap()..], tool_end).is_none()
                {
                    let (content, reasoning) = unterminated_reasoning(text);
                    return Some(ParsedGenerated {
                        content,
                        reasoning,
                        ok: true,
                        ..Default::default()
                    });
                }
                tool_search = candidate.unwrap();
                recovered = true;
            }
            Some(i) => tool_search = i + 8,
        }
    }
    let Some(rel) = find_substr(&text[tool_search..], tool_start) else {
        let (content, reasoning) = split_reasoning_content(text, text.len(), ChatFormat::DeepSeek);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let raw_block_start = if start > 0 && text[start - 1] == b'\n' {
        start - 1
    } else {
        start
    };
    let content_len = trim_tool_separator_ws(text, 0, raw_block_start);
    let mut p = start;
    let mut calls = Vec::new();
    loop {
        p = skip_ws(text, p);
        if !text[p..].starts_with(tool_start) {
            break;
        }
        p += tool_start.len();
        let Some(rel) = find_substr(&text[p..], tool_end) else {
            return None;
        };
        let json = String::from_utf8_lossy(&text[p..p + rel]).into_owned();
        calls.push(parse_hermes_tool_call_json(&json)?);
        p = p + rel + tool_end.len();
    }
    if calls.is_empty() {
        return None;
    }
    let (content, reasoning) = if recovered {
        unterminated_reasoning_before_tool(text, content_len)
    } else {
        split_reasoning_content(text, content_len, ChatFormat::DeepSeek)
    };
    Some(ParsedGenerated {
        content,
        reasoning,
        calls,
        raw_tool_text: String::from_utf8_lossy(&text[raw_block_start..p]).into_owned(),
        ok: true,
        ..Default::default()
    })
}

fn parse_dots3_invoke(body: &[u8]) -> Option<ToolCall> {
    let name_open = b"<invoke name=\"";
    let param_open = b"<parameter name=\"";
    let param_close = b"</parameter>";
    let p0 = find_substr(body, name_open)?;
    let name_s = p0 + name_open.len();
    let name_e = body[name_s..].iter().position(|&c| c == b'"')?;
    let mut tc = ToolCall {
        name: String::from_utf8_lossy(&body[name_s..name_s + name_e]).into_owned(),
        ..Default::default()
    };
    let mut p = name_s + name_e + 1;
    if p < body.len() && body[p] == b'>' {
        p += 1;
    }
    let mut args = vec![b'{'];
    let mut n_args = 0;
    loop {
        let Some(open) = find_substr(&body[p..], param_open) else {
            break;
        };
        let key_s = p + open + param_open.len();
        let Some(key_e) = body[key_s..].iter().position(|&c| c == b'"') else {
            break;
        };
        let mut value = key_s + key_e + 1;
        if value < body.len() && body[value] == b'>' {
            value += 1;
        }
        if value < body.len() && body[value] == b'\n' {
            value += 1;
        }
        let Some(close) = find_substr(&body[value..], param_close) else {
            break;
        };
        let mut value_end = value + close;
        if value_end > value && body[value_end - 1] == b'\n' {
            value_end -= 1;
        }
        if n_args > 0 {
            args.extend_from_slice(b", ");
        }
        args.extend(json_escape_bytes(&body[key_s..key_s + key_e]));
        args.extend_from_slice(b": ");
        let raw = &body[value..value_end];
        let raw_s = String::from_utf8_lossy(raw);
        let mut probe = Json::new(&raw_s);
        probe.ws();
        if let Some(v) = json_raw_value(&mut probe) {
            probe.ws();
            if probe.peek().is_none() {
                args.extend_from_slice(v.as_bytes());
            } else {
                args.extend(json_escape_bytes(raw));
            }
        } else {
            args.extend(json_escape_bytes(raw));
        }
        n_args += 1;
        p = value + close + param_close.len();
    }
    args.push(b'}');
    if tc.name.is_empty() {
        return None;
    }
    tc.arguments = String::from_utf8_lossy(&args).into_owned();
    Some(tc)
}

fn parse_dots3_generated(text: &[u8], require_thinking_closed: bool) -> Option<ParsedGenerated> {
    let tool_start = b"<dots_function_call>";
    let tool_end = b"</dots_function_call>";
    let mut tool_search = 0usize;
    let mut recovered = false;
    if require_thinking_closed {
        match find_last_substr(text, b"</think>") {
            None => {
                let candidate = find_substr(text, tool_start);
                if candidate.is_none()
                    || find_substr(&text[candidate.unwrap()..], tool_end).is_none()
                {
                    let (content, reasoning) = unterminated_reasoning(text);
                    return Some(ParsedGenerated {
                        content,
                        reasoning,
                        ok: true,
                        ..Default::default()
                    });
                }
                tool_search = candidate.unwrap();
                recovered = true;
            }
            Some(i) => tool_search = i + 8,
        }
    }
    let Some(rel) = find_substr(&text[tool_search..], tool_start) else {
        let (content, reasoning) = split_reasoning_content(text, text.len(), ChatFormat::DeepSeek);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let raw_block_start = if start > 0 && text[start - 1] == b'\n' {
        start - 1
    } else {
        start
    };
    let content_len = trim_tool_separator_ws(text, 0, raw_block_start);
    let mut p = start;
    let mut calls = Vec::new();
    loop {
        p = skip_ws(text, p);
        if !text[p..].starts_with(tool_start) {
            break;
        }
        p += tool_start.len();
        let Some(rel) = find_substr(&text[p..], tool_end) else {
            return None;
        };
        let close = p + rel;
        let mut invoke = p;
        while invoke < close {
            let Some(next) = find_substr(&text[invoke..close], b"<invoke name=\"") else {
                break;
            };
            let next = invoke + next;
            let invoke_end = find_substr(&text[next..close], b"</invoke>")
                .map(|i| next + i)
                .unwrap_or(close);
            calls.push(parse_dots3_invoke(&text[next..invoke_end])?);
            invoke = invoke_end;
        }
        p = close + tool_end.len();
    }
    if calls.is_empty() {
        return None;
    }
    let (content, reasoning) = if recovered {
        unterminated_reasoning_before_tool(text, content_len)
    } else {
        split_reasoning_content(text, content_len, ChatFormat::DeepSeek)
    };
    Some(ParsedGenerated {
        content,
        reasoning,
        calls,
        raw_tool_text: String::from_utf8_lossy(&text[raw_block_start..p]).into_owned(),
        ok: true,
        ..Default::default()
    })
}

fn json_value_is_complete(value: &str) -> Option<String> {
    let mut p = Json::new(value);
    p.ws();
    if p.peek().is_none() {
        return None;
    }
    let raw = json_raw_value(&mut p)?;
    p.ws();
    if p.peek().is_some() {
        return None;
    }
    Some(json_minify_raw_value(&raw))
}

fn solar_tool_arg_json_add(
    args: &mut Vec<u8>,
    name: &[u8],
    raw: &[u8],
    expected_type: Option<&str>,
) {
    let trimmed = trim_ascii_span(raw);
    if !args.is_empty() {
        args.extend_from_slice(b", ");
    }
    args.extend(json_escape_bytes(name));
    args.extend_from_slice(b": ");
    if expected_type == Some("string") {
        args.extend(json_escape_bytes(raw));
    } else if let Some(min) = json_value_is_complete(&String::from_utf8_lossy(trimmed)) {
        args.extend_from_slice(min.as_bytes());
    } else {
        args.extend(json_escape_bytes(raw));
    }
}

fn tool_schema_orders_find<'a>(
    orders: &'a [ToolSchemaOrder],
    name: &str,
) -> Option<&'a ToolSchemaOrder> {
    orders.iter().find(|o| o.name == name)
}

fn tool_schema_order_prop_type<'a>(
    order: Option<&'a ToolSchemaOrder>,
    name: &str,
) -> Option<&'a str> {
    let order = order?;
    order
        .prop
        .iter()
        .zip(order.prop_type.iter())
        .find(|(p, _)| p.as_str() == name)
        .map(|(_, t)| t.as_str())
}

fn parse_solar_generated(
    text: &[u8],
    require_thinking_closed: bool,
    orders: &[ToolSchemaOrder],
) -> Option<ParsedGenerated> {
    let think_end = SOLAR_THINK_END.as_bytes();
    let think_start = SOLAR_THINK_START.as_bytes();
    let mut tool_search = 0usize;
    if require_thinking_closed {
        match find_last_substr(text, think_end) {
            None => {
                let body = if text.starts_with(think_start) {
                    &text[think_start.len()..]
                } else {
                    text
                };
                return Some(ParsedGenerated {
                    content: Vec::new(),
                    reasoning: body.to_vec(),
                    ok: true,
                    ..Default::default()
                });
            }
            Some(i) => tool_search = i + think_end.len(),
        }
    }
    let Some(rel) = find_substr(&text[tool_search..], SOLAR_TOOL_CALLS.as_bytes()) else {
        let (content, reasoning) =
            split_reasoning_content(text, text.len(), ChatFormat::SolarOpen2);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let content_len = trim_tool_separator_ws(text, 0, start);
    let mut p = start;
    let mut raw_end = None;
    let mut calls = Vec::new();
    loop {
        if !text[p..].starts_with(SOLAR_TOOL_CALLS.as_bytes()) {
            return None;
        }
        p += SOLAR_TOOL_CALLS.len();
        let Some(nl) = text[p..].iter().position(|&c| c == b'\n') else {
            return None;
        };
        let name = trim_ascii_span(&text[p..p + nl]);
        if name.is_empty() {
            return None;
        }
        let name_s = String::from_utf8_lossy(name).into_owned();
        let order = tool_schema_orders_find(orders, &name_s);
        p = p + nl + 1;
        let mut args = Vec::new();
        let mut call_closed = false;
        while p < text.len() {
            while p < text.len() && (text[p] == b'\r' || text[p] == b'\n') {
                p += 1;
            }
            if text[p..].starts_with(SOLAR_TOOL_CALL_END.as_bytes()) {
                p += SOLAR_TOOL_CALL_END.len();
                raw_end = Some(p);
                call_closed = true;
                break;
            }
            if !text[p..].starts_with(SOLAR_TOOL_ARG_START.as_bytes()) {
                return None;
            }
            p += SOLAR_TOOL_ARG_START.len();
            let Some(vm) = find_substr(&text[p..], SOLAR_TOOL_ARG_VALUE.as_bytes()) else {
                return None;
            };
            let arg_name = trim_ascii_span(&text[p..p + vm]);
            p = p + vm + SOLAR_TOOL_ARG_VALUE.len();
            let Some(ae) = find_substr(&text[p..], SOLAR_TOOL_ARG_END.as_bytes()) else {
                return None;
            };
            if arg_name.is_empty() {
                return None;
            }
            let arg_name_s = String::from_utf8_lossy(arg_name).into_owned();
            solar_tool_arg_json_add(
                &mut args,
                arg_name,
                &text[p..p + ae],
                tool_schema_order_prop_type(order, &arg_name_s),
            );
            p = p + ae + SOLAR_TOOL_ARG_END.len();
        }
        if !call_closed {
            return None;
        }
        let mut wrapped = vec![b'{'];
        wrapped.extend_from_slice(&args);
        wrapped.push(b'}');
        calls.push(ToolCall {
            name: name_s,
            arguments: String::from_utf8_lossy(&wrapped).into_owned(),
            ..Default::default()
        });
        let next = skip_ws(text, p);
        if !text[next..].starts_with(SOLAR_TOOL_CALLS.as_bytes()) {
            break;
        }
        p = next;
    }
    if calls.is_empty() {
        return None;
    }
    let (content, reasoning) = split_reasoning_content(text, content_len, ChatFormat::SolarOpen2);
    Some(ParsedGenerated {
        content,
        reasoning,
        calls,
        raw_dsml: String::from_utf8_lossy(&text[start..raw_end.unwrap_or(p)]).into_owned(),
        ok: true,
        ..Default::default()
    })
}

fn parse_k2_generated(text: &[u8], require_thinking_closed: bool) -> Option<ParsedGenerated> {
    let (content, reasoning) =
        split_reasoning_response(text, ChatFormat::K2Horizon, require_thinking_closed);
    Some(ParsedGenerated {
        content,
        reasoning,
        ok: true,
        ..Default::default()
    })
}

fn parse_qwen_generated(
    text: &[u8],
    require_thinking_closed: bool,
    orders: &[ToolSchemaOrder],
) -> Option<ParsedGenerated> {
    let mut tool_search = 0usize;
    if require_thinking_closed {
        match find_last_substr(text, b"</think>") {
            None => {
                let body = text.strip_prefix(b"<think>").unwrap_or(text);
                return Some(ParsedGenerated {
                    content: Vec::new(),
                    reasoning: body.to_vec(),
                    ok: true,
                    ..Default::default()
                });
            }
            Some(i) => tool_search = i + "</think>".len(),
        }
    }
    let Some(rel) = find_substr(&text[tool_search..], QWEN_TOOL_CALL_START.as_bytes()) else {
        let (content, reasoning) = split_reasoning_content(text, text.len(), ChatFormat::Qwen4Exp);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let content_len = trim_tool_separator_ws(text, 0, start);
    let mut p = start;
    let mut raw_end = None;
    let mut calls = Vec::new();
    while text[p..].starts_with(QWEN_TOOL_CALL_START.as_bytes()) {
        p += QWEN_TOOL_CALL_START.len();
        p = skip_ws(text, p);
        if !text[p..].starts_with(b"<function=") {
            return None;
        }
        p += "<function=".len();
        let name_end = text[p..].iter().position(|&c| c == b'>')?;
        let name = trim_ascii_span(&text[p..p + name_end]);
        if name.is_empty() {
            return None;
        }
        let name_s = String::from_utf8_lossy(name).into_owned();
        let order = tool_schema_orders_find(orders, &name_s);
        p += name_end + 1;

        let mut args = Vec::new();
        let mut function_closed = false;
        while p < text.len() {
            p = skip_ws(text, p);
            if text[p..].starts_with(b"</function>") {
                p += "</function>".len();
                function_closed = true;
                break;
            }
            if !text[p..].starts_with(b"<parameter=") {
                return None;
            }
            p += "<parameter=".len();
            let arg_name_end = text[p..].iter().position(|&c| c == b'>')?;
            let arg_name = trim_ascii_span(&text[p..p + arg_name_end]);
            if arg_name.is_empty() {
                return None;
            }
            p += arg_name_end + 1;
            if text.get(p) == Some(&b'\r') {
                p += 1;
            }
            if text.get(p) == Some(&b'\n') {
                p += 1;
            }
            let arg_end = find_substr(&text[p..], b"</parameter>")?;
            let mut value_end = p + arg_end;
            if value_end > p && text[value_end - 1] == b'\n' {
                value_end -= 1;
            }
            if value_end > p && text[value_end - 1] == b'\r' {
                value_end -= 1;
            }
            let arg_name_s = String::from_utf8_lossy(arg_name).into_owned();
            solar_tool_arg_json_add(
                &mut args,
                arg_name,
                &text[p..value_end],
                tool_schema_order_prop_type(order, &arg_name_s),
            );
            p += arg_end + "</parameter>".len();
        }
        if !function_closed {
            return None;
        }
        p = skip_ws(text, p);
        if !text[p..].starts_with(QWEN_TOOL_CALL_END.as_bytes()) {
            return None;
        }
        p += QWEN_TOOL_CALL_END.len();
        raw_end = Some(p);

        let mut wrapped = vec![b'{'];
        wrapped.extend_from_slice(&args);
        wrapped.push(b'}');
        calls.push(ToolCall {
            name: name_s,
            arguments: String::from_utf8_lossy(&wrapped).into_owned(),
            ..Default::default()
        });
        let next = skip_ws(text, p);
        if !text[next..].starts_with(QWEN_TOOL_CALL_START.as_bytes()) {
            break;
        }
        p = next;
    }
    if calls.is_empty() {
        return None;
    }
    let (content, reasoning) = split_reasoning_content(text, content_len, ChatFormat::Qwen4Exp);
    Some(ParsedGenerated {
        content,
        reasoning,
        calls,
        raw_dsml: String::from_utf8_lossy(&text[start..raw_end.unwrap_or(p)]).into_owned(),
        ok: true,
        ..Default::default()
    })
}

fn parse_glm_generated(text: &[u8], require_thinking_closed: bool) -> Option<ParsedGenerated> {
    const ARG_KEY_START: &[u8] = b"<arg_key>";
    const ARG_KEY_END: &[u8] = b"</arg_key>";
    const ARG_VALUE_START: &[u8] = b"<arg_value>";
    const ARG_VALUE_END: &[u8] = b"</arg_value>";
    let tool_start = GLM_TOOL_CALL_START.as_bytes();
    let tool_end = GLM_TOOL_CALL_END.as_bytes();

    let mut tool_search = 0usize;
    let mut recovered_unclosed_tool = false;
    if require_thinking_closed {
        if let Some(end) = find_last_substr(text, b"</think>") {
            tool_search = end + "</think>".len();
        } else if let Some(candidate) = find_substr(text, tool_start) {
            if find_substr(&text[candidate..], tool_end).is_none() {
                let (content, reasoning) = unterminated_reasoning(text);
                return Some(ParsedGenerated {
                    content,
                    reasoning,
                    ok: true,
                    ..Default::default()
                });
            }
            tool_search = candidate;
            recovered_unclosed_tool = true;
        } else {
            let (content, reasoning) = unterminated_reasoning(text);
            return Some(ParsedGenerated {
                content,
                reasoning,
                ok: true,
                ..Default::default()
            });
        }
    }

    let Some(rel) = find_substr(&text[tool_search..], tool_start) else {
        let (content, reasoning) = split_reasoning_content(text, text.len(), ChatFormat::DeepSeek);
        return Some(ParsedGenerated {
            content,
            reasoning,
            ok: true,
            ..Default::default()
        });
    };
    let start = tool_search + rel;
    let raw_start = if start >= 2 && &text[start - 2..start] == b"\n\n" {
        start - 2
    } else {
        start
    };
    let content_len = trim_tool_separator_ws(text, 0, raw_start);
    let mut p = start;
    let mut calls = Vec::new();

    loop {
        p = skip_ws(text, p);
        if !text[p..].starts_with(tool_start) {
            break;
        }
        p += tool_start.len();
        let close = p + find_substr(&text[p..], tool_end)?;
        let arg = find_substr(&text[p..close], ARG_KEY_START).map(|at| p + at);
        let name_end = arg.unwrap_or(close);
        let name = trim_ascii_span(&text[p..name_end]);
        if name.is_empty() {
            return None;
        }
        let name = String::from_utf8_lossy(name).into_owned();
        p = name_end;
        let mut args = Vec::new();

        loop {
            p = skip_ws(text, p);
            if text[p..].starts_with(tool_end) {
                p += tool_end.len();
                break;
            }
            if !text[p..].starts_with(ARG_KEY_START) {
                return None;
            }
            p += ARG_KEY_START.len();
            let key_end = p + find_substr(&text[p..], ARG_KEY_END)?;
            if key_end > close {
                return None;
            }
            let key = dsml_unescape_text(trim_ascii_span(&text[p..key_end]));
            if key.is_empty() {
                return None;
            }
            p = skip_ws(text, key_end + ARG_KEY_END.len());
            if !text[p..].starts_with(ARG_VALUE_START) {
                return None;
            }
            p += ARG_VALUE_START.len();
            let value_end = p + find_substr(&text[p..], ARG_VALUE_END)?;
            if value_end > close {
                return None;
            }
            let value = dsml_unescape_text(&text[p..value_end]);
            tool_call_json_args_add(&mut args, &key, &value, true);
            p = value_end + ARG_VALUE_END.len();
        }

        let mut arguments = vec![b'{'];
        arguments.extend(args);
        arguments.push(b'}');
        calls.push(ToolCall {
            name,
            arguments: String::from_utf8_lossy(&arguments).into_owned(),
            ..Default::default()
        });
        let next = skip_ws(text, p);
        if !text[next..].starts_with(tool_start) {
            p = next;
            break;
        }
        p = next;
    }
    if calls.is_empty() {
        return None;
    }
    let (content, reasoning) = if recovered_unclosed_tool {
        unterminated_reasoning_before_tool(text, content_len)
    } else {
        split_reasoning_content(text, content_len, ChatFormat::DeepSeek)
    };
    Some(ParsedGenerated {
        content,
        reasoning,
        calls,
        raw_dsml: String::from_utf8_lossy(&text[raw_start..p]).into_owned(),
        ok: true,
        recovered: recovered_unclosed_tool,
        ..Default::default()
    })
}

pub fn parse_generated_message(
    syntax: ModelSyntax,
    text: &[u8],
    require_thinking_closed: bool,
    format: ChatFormat,
    orders: &[ToolSchemaOrder],
) -> ParsedGenerated {
    let parsed = match syntax {
        ModelSyntax::Motif3 | ModelSyntax::Exaone => {
            parse_hermes_generated(text, require_thinking_closed)
        }
        ModelSyntax::Dots3 => parse_dots3_generated(text, require_thinking_closed),
        ModelSyntax::SolarOpen2 => parse_solar_generated(text, require_thinking_closed, orders),
        ModelSyntax::Qwen4Exp => parse_qwen_generated(text, require_thinking_closed, orders),
        ModelSyntax::K2Horizon => parse_k2_generated(text, require_thinking_closed),
        ModelSyntax::Glm53 => parse_glm_generated(text, require_thinking_closed),
        ModelSyntax::DeepSeek => {
            if format == ChatFormat::SolarOpen2 {
                parse_solar_generated(text, require_thinking_closed, orders)
            } else if format == ChatFormat::Qwen4Exp {
                parse_qwen_generated(text, require_thinking_closed, orders)
            } else if format == ChatFormat::K2Horizon {
                parse_k2_generated(text, require_thinking_closed)
            } else {
                parse_dsml_generated(text, require_thinking_closed)
            }
        }
    };
    parsed.unwrap_or(ParsedGenerated {
        content: text.to_vec(),
        ok: false,
        ..Default::default()
    })
}

pub fn parse_generated_for_model_id(
    model_id: i32,
    text: &[u8],
    require_thinking_closed: bool,
    orders: &[ToolSchemaOrder],
) -> ParsedGenerated {
    let syntax = syntax_for_model_id(model_id);
    let format = match syntax {
        ModelSyntax::SolarOpen2 => ChatFormat::SolarOpen2,
        ModelSyntax::Exaone => ChatFormat::Exaone,
        ModelSyntax::Qwen4Exp => ChatFormat::Qwen4Exp,
        ModelSyntax::K2Horizon => ChatFormat::K2Horizon,
        _ => ChatFormat::DeepSeek,
    };
    parse_generated_message(syntax, text, require_thinking_closed, format, orders)
}

fn split_reasoning_response(
    text: &[u8],
    format: ChatFormat,
    require_thinking_closed: bool,
) -> (Vec<u8>, Vec<u8>) {
    let end = think_end(format).as_bytes();
    if require_thinking_closed && find_last_substr(text, end).is_none() {
        let start = think_start(format).as_bytes();
        let body = if text.starts_with(start) {
            &text[start.len()..]
        } else {
            text
        };
        return (Vec::new(), body.to_vec());
    }
    split_reasoning_content(text, text.len(), format)
}

/// C `parse_generated_message_for_response_model`. No-tools never
/// extracts calls. Parse failure with a seen start becomes raw-text
/// fallback; finish becomes `stop` unless it was `length` or `error`.
pub fn parse_generated_for_response(
    syntax: ModelSyntax,
    text: &[u8],
    has_tools: bool,
    saw_tool_start: bool,
    require_thinking_closed: bool,
    format: ChatFormat,
    orders: &[ToolSchemaOrder],
    finish: &str,
) -> (ParsedGenerated, &'static str) {
    if !has_tools {
        let (content, reasoning) = split_reasoning_response(text, format, require_thinking_closed);
        return (
            ParsedGenerated {
                content,
                reasoning,
                ok: true,
                ..Default::default()
            },
            intern_finish(finish),
        );
    }
    let parsed = parse_generated_message(syntax, text, require_thinking_closed, format, orders);
    if parsed.ok {
        return (parsed, intern_finish(finish));
    }
    let recovered = has_tools && saw_tool_start && finish != "error";
    let out_finish = if recovered {
        if finish == "length" {
            "length"
        } else {
            "stop"
        }
    } else {
        intern_finish(finish)
    };
    (
        ParsedGenerated {
            content: text.to_vec(),
            ok: false,
            recovered,
            ..Default::default()
        },
        out_finish,
    )
}

fn intern_finish(finish: &str) -> &'static str {
    match finish {
        "tool_calls" => "tool_calls",
        "length" => "length",
        "error" => "error",
        _ => "stop",
    }
}

pub fn find_tool_start(s: &[u8], format: ChatFormat) -> Option<usize> {
    match format {
        ChatFormat::SolarOpen2 => find_substr(s, SOLAR_TOOL_CALLS.as_bytes()),
        ChatFormat::Exaone => find_substr(s, b"<tool_call>"),
        ChatFormat::Qwen4Exp => find_substr(s, QWEN_TOOL_CALL_START.as_bytes()),
        ChatFormat::K2Horizon => find_substr(s, crate::render::K2_TOOL_CALLS_START.as_bytes()),
        ChatFormat::DeepSeek => {
            let cands = [
                DSML_TOOL_CALLS_START.as_bytes(),
                DSML_TOOL_CALLS_START_SHORT.as_bytes(),
                b"<tool_calls>",
                b"<tool_call>",
                b"<dots_function_call>",
            ];
            cands.iter().filter_map(|p| find_substr(s, p)).min()
        }
    }
}

pub fn find_tool_end(s: &[u8], format: ChatFormat) -> Option<usize> {
    match format {
        ChatFormat::SolarOpen2 => find_substr(s, SOLAR_TOOL_CALL_END.as_bytes()),
        ChatFormat::Exaone => find_substr(s, b"</tool_call>"),
        ChatFormat::Qwen4Exp => find_substr(s, QWEN_TOOL_CALL_END.as_bytes()),
        ChatFormat::K2Horizon => find_substr(s, crate::render::K2_TOOL_CALLS_END.as_bytes()),
        ChatFormat::DeepSeek => {
            let cands = [
                DSML_TOOL_CALLS_END.as_bytes(),
                DSML_TOOL_CALLS_END_SHORT.as_bytes(),
                b"</tool_calls>",
                b"</tool_call>",
                b"</dots_function_call>",
            ];
            cands.iter().filter_map(|p| find_substr(s, p)).min()
        }
    }
}

#[allow(dead_code)]
pub fn observe_tool_markers(
    scan: &[u8],
    saw_start: &mut bool,
    saw_end: &mut bool,
    format: ChatFormat,
) -> (bool, bool) {
    let had_start = *saw_start;
    let start = find_tool_start(scan, format);
    if start.is_some() {
        *saw_start = true;
    }
    let end_scan = if had_start { Some(0) } else { start };
    let end = end_scan.and_then(|off| find_tool_end(&scan[off..], format));
    if end.is_some() {
        *saw_end = true;
    }
    let entered = *saw_start && !had_start;
    let closed = *saw_end && end.is_some();
    // closed should be "became true this observe". Caller tracks old_end.
    (entered, closed)
}

struct ToolIdState {
    next: u128,
    blocked: HashMap<(u8, u128), usize>,
}

static TOOL_ID_STATE: OnceLock<Mutex<ToolIdState>> = OnceLock::new();

fn tool_id_seed() -> u128 {
    let mut bytes = [0u8; 16];
    if File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .is_ok()
    {
        return u128::from_le_bytes(bytes);
    }
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    time ^ u128::from(std::process::id())
}

fn tool_id_kind(prefix: &str) -> u8 {
    u8::from(prefix == "toolu_")
}

#[cfg(any(feature = "native", test))]
fn tool_id_key(id: &str) -> Option<(u8, u128)> {
    let (kind, hex) = if let Some(hex) = id.strip_prefix("call_") {
        (0, hex)
    } else if let Some(hex) = id.strip_prefix("toolu_") {
        (1, hex)
    } else {
        return None;
    };
    if hex.len() != 32 {
        return None;
    }
    Some((kind, u128::from_str_radix(hex, 16).ok()?))
}

fn mint_tool_id_from(state: &mut ToolIdState, prefix: &str) -> String {
    loop {
        let value = state.next;
        state.next = state.next.wrapping_add(1);
        if !state.blocked.contains_key(&(tool_id_kind(prefix), value)) {
            return format!("{prefix}{value:032x}");
        }
    }
}

pub(crate) fn mint_tool_id(prefix: &str) -> String {
    let state = TOOL_ID_STATE.get_or_init(|| {
        Mutex::new(ToolIdState {
            next: tool_id_seed(),
            blocked: HashMap::new(),
        })
    });
    mint_tool_id_from(
        &mut state.lock().unwrap_or_else(|error| error.into_inner()),
        prefix,
    )
}

#[cfg(any(feature = "native", test))]
pub(crate) fn reserve_tool_id(id: &str) {
    let Some(key) = tool_id_key(id) else {
        return;
    };
    let state = TOOL_ID_STATE.get_or_init(|| {
        Mutex::new(ToolIdState {
            next: tool_id_seed(),
            blocked: HashMap::new(),
        })
    });
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    *state.blocked.entry(key).or_default() += 1;
}

#[cfg(any(feature = "native", test))]
pub(crate) fn release_tool_id(id: &str) {
    let Some(key) = tool_id_key(id) else {
        return;
    };
    let Some(state) = TOOL_ID_STATE.get() else {
        return;
    };
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(refs) = state.blocked.get_mut(&key) {
        if *refs <= 1 {
            state.blocked.remove(&key);
        } else {
            *refs -= 1;
        }
    }
}

fn assign_tool_ids_with(
    calls: &mut [ToolCall],
    prefix: &str,
    mut mint: impl FnMut(&str) -> String,
) {
    let mut used: HashSet<String> = calls
        .iter()
        .filter(|call| !call.id.is_empty())
        .map(|call| call.id.clone())
        .collect();
    for tc in calls {
        while tc.id.is_empty() {
            let id = mint(prefix);
            if used.insert(id.clone()) {
                tc.id = id;
            }
        }
    }
}

pub fn assign_tool_ids(calls: &mut [ToolCall], prefix: &str) {
    assign_tool_ids_with(calls, prefix, mint_tool_id);
}

fn stop_list_find_from(stops: &[String], text: &[u8], from: usize) -> Option<(usize, usize)> {
    if stops.is_empty() || from > text.len() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        let needle = s.as_bytes();
        if from + needle.len() > text.len() {
            continue;
        }
        if let Some(rel) = find_substr(&text[from..], needle) {
            let pos = from + rel;
            if best.map(|(p, _)| pos < p).unwrap_or(true) {
                best = Some((pos, needle.len()));
            }
        }
    }
    best
}

fn stop_list_max_len(stops: &[String]) -> usize {
    stops.iter().map(|s| s.len()).max().unwrap_or(0)
}

fn stop_list_stream_safe_len(stops: &[String], text_len: usize) -> usize {
    let max = stop_list_max_len(stops);
    if max <= 1 || text_len <= max - 1 {
        return if max <= 1 { text_len } else { 0 };
    }
    text_len - (max - 1)
}

fn tool_marker_stream_safe_len(text: &[u8], format: ChatFormat) -> usize {
    let marks: &[&[u8]] = match format {
        ChatFormat::SolarOpen2 => &[SOLAR_TOOL_CALLS.as_bytes()],
        ChatFormat::Exaone => &[b"<tool_call>"],
        ChatFormat::Qwen4Exp => &[QWEN_TOOL_CALL_START.as_bytes()],
        ChatFormat::K2Horizon => &[crate::render::K2_TOOL_CALLS_START.as_bytes()],
        ChatFormat::DeepSeek => &[
            DSML_TOOL_CALLS_START.as_bytes(),
            DSML_TOOL_CALLS_START_SHORT.as_bytes(),
            b"<tool_call>",
            b"<dots_function_call>",
        ],
    };
    if text.is_empty() {
        return 0;
    }
    let mut safe = text.len();
    for m in marks {
        if m.len() <= 1 {
            continue;
        }
        let mut k = (m.len() - 1).min(text.len());
        while k >= 1 {
            if text[text.len() - k..].starts_with(&m[..k]) {
                let pos = text.len() - k;
                if pos < safe {
                    safe = pos;
                }
                break;
            }
            k -= 1;
        }
    }
    safe
}

#[derive(Debug, Clone)]
pub struct SemAccum {
    pub text: Vec<u8>,
    pub track_tools: bool,
    cut_tool_syntax: bool,
    think_gates: bool,
    chat_format: ChatFormat,
    thinking_inside: bool,
    think_tail: Vec<u8>,
    stop_scan_from: usize,
    tool_scan_from: usize,
    tool_scan_waiting: bool,
    pub saw_tool_start: bool,
    pub saw_tool_end: bool,
    pub verdict: Option<&'static str>,
    pub matched_stop: Option<String>,
    pub completion: i32,
    pub reasoning_tokens: i32,
    pub dsml: DsmlDecodeTracker,
    required_tool_prefix_pos: i32,
    required_think_end_prefix_pos: i32,
}

#[derive(Debug, Clone, Default)]
pub struct SemFeed {
    pub emit_limit: usize,
    pub hit_stop: bool,
    pub tool_block_closed: bool,
    pub entered_tool_block: bool,
    pub tool_syntax_cut: bool,
}

impl SemAccum {
    pub fn init(
        kind_chat: bool,
        has_tools: bool,
        think_enabled: bool,
        format: ChatFormat,
        prompt: &[u8],
    ) -> Self {
        let mut a = Self {
            text: Vec::new(),
            track_tools: kind_chat && has_tools,
            cut_tool_syntax: kind_chat && !has_tools,
            think_gates: think_enabled,
            chat_format: format,
            thinking_inside: false,
            think_tail: Vec::new(),
            stop_scan_from: 0,
            tool_scan_from: 0,
            tool_scan_waiting: think_enabled,
            saw_tool_start: false,
            saw_tool_end: false,
            verdict: None,
            matched_stop: None,
            completion: 0,
            reasoning_tokens: 0,
            dsml: DsmlDecodeTracker::default(),
            required_tool_prefix_pos: 0,
            required_think_end_prefix_pos: 0,
        };
        if !prompt.is_empty() {
            a.feed_thinking(prompt);
        } else if think_enabled {
            a.thinking_inside = true;
        }
        a.tool_scan_waiting = a.think_gates && a.thinking_inside;
        a
    }

    fn feed_thinking(&mut self, piece: &[u8]) {
        for &c in piece {
            if self.think_tail.len() == 16 {
                self.think_tail.remove(0);
            }
            self.think_tail.push(c);
            if tail_ends_with(&self.think_tail, b"<think>")
                || tail_ends_with(&self.think_tail, SOLAR_THINK_START.as_bytes())
            {
                self.thinking_inside = true;
            } else if tail_ends_with(&self.think_tail, b"</think>")
                || tail_ends_with(&self.think_tail, SOLAR_THINK_END.as_bytes())
            {
                self.thinking_inside = false;
            }
        }
    }

    pub fn feed(&mut self, piece: &[u8], stops: &[String]) -> SemFeed {
        let mut f = SemFeed::default();
        if self.think_gates && self.thinking_inside {
            self.reasoning_tokens += 1;
        }
        self.feed_thinking(piece);
        self.text.extend_from_slice(piece);
        self.completion += 1;
        if self.track_tools && self.chat_format == ChatFormat::DeepSeek {
            self.dsml.update(&self.text);
        }

        let hit = stop_list_find_from(stops, &self.text, self.stop_scan_from);
        f.emit_limit = if let Some((pos, _)) = hit {
            pos
        } else {
            stop_list_stream_safe_len(stops, self.text.len())
        };
        if f.emit_limit > self.text.len() {
            f.emit_limit = self.text.len();
        }
        if hit.is_none() {
            let max = stop_list_max_len(stops);
            if max > 1 {
                let hold = max - 1;
                self.stop_scan_from = if self.text.len() > hold {
                    self.text.len() - hold
                } else {
                    0
                };
            }
        }
        if self.cut_tool_syntax {
            let safe = tool_marker_stream_safe_len(&self.text, self.chat_format);
            if f.emit_limit > safe {
                f.emit_limit = safe;
            }
        }

        let mut cut_pos = None;
        if self.track_tools || self.cut_tool_syntax {
            if self.think_gates && self.thinking_inside {
                self.tool_scan_waiting = true;
                self.tool_scan_from = self.text.len();
            } else {
                if self.tool_scan_waiting {
                    let marker = think_end(self.chat_format).as_bytes();
                    self.tool_scan_from = find_last_substr(&self.text, marker)
                        .map(|i| i + marker.len())
                        .unwrap_or(self.text.len())
                        .min(self.text.len());
                    self.tool_scan_waiting = false;
                }
                if self.tool_scan_from > self.text.len() {
                    self.tool_scan_from = self.text.len();
                }
                let scan = &self.text[self.tool_scan_from..];
                let old_start = self.saw_tool_start;
                let old_end = self.saw_tool_end;
                let start = find_tool_start(scan, self.chat_format);
                if start.is_some() {
                    self.saw_tool_start = true;
                }
                let end_off = if old_start { Some(0) } else { start };
                if end_off
                    .and_then(|o| find_tool_end(&scan[o..], self.chat_format))
                    .is_some()
                {
                    self.saw_tool_end = true;
                }
                f.entered_tool_block = self.saw_tool_start && !old_start;
                f.tool_block_closed = self.saw_tool_end && !old_end;
                if f.entered_tool_block && self.cut_tool_syntax {
                    if let Some(m) = find_tool_start(scan, self.chat_format) {
                        cut_pos = Some(self.tool_scan_from + m);
                    }
                }
                let marker_hold = 80;
                let hold_from = if self.text.len() > marker_hold {
                    self.text.len() - marker_hold
                } else {
                    0
                };
                if hold_from > self.tool_scan_from {
                    self.tool_scan_from = hold_from;
                }
            }
        }

        if let Some((pos, len)) = hit {
            self.matched_stop =
                Some(String::from_utf8_lossy(&self.text[pos..pos + len]).into_owned());
            self.text.truncate(pos);
            self.verdict = Some("stop");
            f.hit_stop = true;
            f.emit_limit = pos;
        } else if self.cut_tool_syntax && cut_pos.is_some() && self.verdict.is_none() {
            let pos = cut_pos.unwrap();
            self.text.truncate(pos);
            self.verdict = Some("stop");
            f.hit_stop = true;
            f.tool_syntax_cut = true;
            if f.emit_limit > pos {
                f.emit_limit = pos;
            }
        } else if self.track_tools
            && self.saw_tool_end
            && self.verdict.is_none()
            && self.chat_format == ChatFormat::DeepSeek
        {
            // C only auto-verdicts DSML format, not Solar/EXAONE.
            self.verdict = Some("tool_calls");
        }
        f
    }

    pub fn thinking_inside(&self) -> bool {
        self.thinking_inside
    }

    pub fn dsml_state(&self) -> DsmlDecodeState {
        if self.track_tools && self.chat_format == ChatFormat::DeepSeek {
            self.dsml.decode
        } else {
            DsmlDecodeState::Outside
        }
    }

    pub fn sampling_override(&mut self, p: &SamplePolicy<'_>) -> SampleOverride {
        crate::dsml::sampling_override(
            self.track_tools,
            self.saw_tool_start,
            self.thinking_inside,
            self.completion,
            self.chat_format,
            self.dsml.decode,
            &mut self.required_tool_prefix_pos,
            &mut self.required_think_end_prefix_pos,
            p,
        )
    }
}

fn tail_ends_with(tail: &[u8], s: &[u8]) -> bool {
    tail.len() >= s.len() && tail[tail.len() - s.len()..] == s[..]
}

#[cfg(test)]
mod tool_id_tests {
    use super::{assign_tool_ids_with, mint_tool_id_from, ToolIdState};
    use crate::parse::ToolCall;
    use std::collections::{HashMap, VecDeque};

    #[test]
    fn assignment_retries_ids_already_present_in_the_turn() {
        let mut calls = vec![
            ToolCall {
                id: "call_taken".into(),
                ..Default::default()
            },
            ToolCall::default(),
        ];
        let mut minted = VecDeque::from(["call_taken".to_string(), "call_fresh".to_string()]);
        assign_tool_ids_with(&mut calls, "call_", |_| minted.pop_front().unwrap());
        assert_eq!(calls[1].id, "call_fresh");
    }

    #[test]
    fn allocator_skips_reserved_checkpoint_ids() {
        let mut state = ToolIdState {
            next: 7,
            blocked: HashMap::from([((0, 7), 1)]),
        };
        assert_eq!(
            mint_tool_id_from(&mut state, "call_"),
            "call_00000000000000000000000000000008"
        );
    }
}
