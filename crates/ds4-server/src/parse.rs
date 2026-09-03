//! Four surface parsers from `ds4_server.c`. No prompt tokenize / generation.

use crate::format::{
    parse_output_config_effort, parse_output_format_value, parse_reasoning_effort_value,
    parse_responses_text_value,
};
use crate::json::{
    json_bool, json_content, json_escape, json_int, json_number, json_raw_value, json_skip_value,
    json_string, Json,
};
use crate::models::{model_alias_disables_thinking, model_alias_enables_thinking};
use crate::route::{
    compute_needs, think_mode_from_enabled, Api, NeedInput, ReqKind, ThinkMode, WireSurface,
};
use std::sync::Arc;

pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const DEFAULT_MIN_P: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto = 0,
    None = 1,
    Required = 2,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatPart {
    Text(String),
    Image(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMime {
    Png,
    Jpeg,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestImage {
    pub mime: ImageMime,
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub tool_call_id: String,
    pub tool_call_ids: Vec<String>,
    pub calls: Vec<ToolCall>,
    pub raw_dsml: String,
    pub raw_tool_text: String,
    pub parts: Vec<ChatPart>,
}

#[derive(Debug, Clone, Default)]
pub struct ToolSchemaOrder {
    pub name: String,
    pub wire_name: String,
    pub namespace: String,
    pub responses_tool_search: bool,
    pub prop: Vec<String>,
    pub prop_type: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParseEnv {
    pub default_model: String,
    pub default_tokens: i32,
    pub default_effort: ThinkMode,
    pub default_temp: f32,
    /// LIVE call ids for this surface at parse time (C `*_live_has_call_id`).
    pub live_ids: Vec<String>,
}

impl Default for ParseEnv {
    fn default() -> Self {
        Self {
            default_model: "deepseek-v4-flash".into(),
            default_tokens: 393216,
            default_effort: ThinkMode::Low,
            default_temp: default_temperature(),
            live_ids: Vec::new(),
        }
    }
}

pub fn default_temperature() -> f32 {
    std::env::var("DS4_SERVER_DEFAULT_TEMP")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TEMPERATURE)
}

#[derive(Debug, Clone)]
pub struct ParsedRequest {
    pub kind: ReqKind,
    pub api: Api,
    pub model: String,
    pub model_from_request: bool,
    pub max_tokens: i32,
    pub max_tokens_set: bool,
    pub top_k: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub return_token_ids: bool,
    pub think_mode: ThinkMode,
    pub has_tools: bool,
    pub has_tool_results: bool,
    pub tool_choice: ToolChoice,
    pub required_tool_prefix: Vec<i32>,
    pub required_think_end_prefix: Vec<i32>,
    pub stops: Vec<String>,
    pub reasoning_summary_emit: bool,
    pub responses_requires_live_tool_state: bool,
    pub responses_requires_live_reasoning: bool,
    pub anthropic_requires_live_tool_state: bool,
    pub live_state_bank_owned: bool,
    pub directed_bank: Option<i32>,
    pub live_call_ids: Vec<String>,
    pub messages: Vec<ChatMsg>,
    pub images: Vec<RequestImage>,
    pub tool_schemas: String,
    pub tool_orders: Vec<ToolSchemaOrder>,
    pub prompt_text: Option<String>,
    pub needs: u32,
}

impl ParsedRequest {
    fn init(kind: ReqKind, env: &ParseEnv) -> Self {
        Self {
            kind,
            api: Api::Openai,
            model: env.default_model.clone(),
            model_from_request: false,
            max_tokens: env.default_tokens,
            max_tokens_set: false,
            top_k: 0,
            temperature: env.default_temp,
            top_p: DEFAULT_TOP_P,
            min_p: DEFAULT_MIN_P,
            seed: 0,
            stream: false,
            stream_include_usage: false,
            return_token_ids: false,
            think_mode: ThinkMode::Low,
            has_tools: false,
            has_tool_results: false,
            tool_choice: ToolChoice::Auto,
            required_tool_prefix: Vec::new(),
            required_think_end_prefix: Vec::new(),
            stops: Vec::new(),
            reasoning_summary_emit: false,
            responses_requires_live_tool_state: false,
            responses_requires_live_reasoning: false,
            anthropic_requires_live_tool_state: false,
            live_state_bank_owned: false,
            directed_bank: None,
            live_call_ids: Vec::new(),
            messages: Vec::new(),
            images: Vec::new(),
            tool_schemas: String::new(),
            tool_orders: Vec::new(),
            prompt_text: None,
            needs: 0,
        }
    }

    pub(crate) fn finish_needs(&mut self) {
        self.needs = compute_needs(&NeedInput {
            api: self.api,
            kind: self.kind,
            stream: self.stream,
            temperature: self.temperature,
            think: crate::route::think_mode_enabled(self.think_mode),
            stop_count: self.stops.len() as u32,
            has_tools: self.has_tools,
            return_token_ids: self.return_token_ids,
            responses_requires_live_tool_state: self.responses_requires_live_tool_state,
            responses_requires_live_reasoning: self.responses_requires_live_reasoning,
            anthropic_requires_live_tool_state: self.anthropic_requires_live_tool_state,
            live_state_bank_owned: self.live_state_bank_owned,
            max_tokens_set: self.max_tokens_set,
            max_tokens: self.max_tokens,
            has_images: !self.images.is_empty(),
        });
    }
}

const CHAT_IMAGE_MAX_COUNT: usize = 4;
const CHAT_IMAGE_MAX_BYTES: usize = 10 * 1024 * 1024;
const CHAT_IMAGE_TOTAL_MAX: usize = 20 * 1024 * 1024;

fn base64_value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((26 + c - b'a') as u32),
        b'0'..=b'9' => Some((52 + c - b'0') as u32),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn base64_decode_image(src: &str, request_remaining: usize) -> Result<Vec<u8>, String> {
    let src = src.as_bytes();
    if src.is_empty() || src.len() & 3 != 0 {
        return Err("invalid image base64 length".into());
    }
    let padding = usize::from(src[src.len() - 1] == b'=') + usize::from(src[src.len() - 2] == b'=');
    let decoded = src.len() / 4 * 3 - padding;
    if decoded == 0 || decoded > CHAT_IMAGE_MAX_BYTES {
        return Err("image exceeds 10 MiB decoded limit".into());
    }
    if decoded > request_remaining {
        return Err("images exceed 20 MiB request limit".into());
    }
    let mut out = Vec::with_capacity(decoded);
    for (i, chunk) in src.chunks_exact(4).enumerate() {
        let last = i + 1 == src.len() / 4;
        let a = base64_value(chunk[0]);
        let b = base64_value(chunk[1]);
        let c = (chunk[2] != b'=').then(|| base64_value(chunk[2])).flatten();
        let d = (chunk[3] != b'=').then(|| base64_value(chunk[3])).flatten();
        if a.is_none()
            || b.is_none()
            || (chunk[2] != b'=' && c.is_none())
            || (chunk[3] != b'=' && d.is_none())
            || (!last && (c.is_none() || d.is_none()))
            || (chunk[2] == b'=' && chunk[3] != b'=')
            || (chunk[2] == b'=' && b.unwrap() & 15 != 0)
            || (chunk[3] == b'=' && c.is_some_and(|v| v & 3 != 0))
        {
            return Err("invalid image base64 data".into());
        }
        let v = a.unwrap() << 18 | b.unwrap() << 12 | c.unwrap_or(0) << 6 | d.unwrap_or(0);
        if out.len() < decoded {
            out.push((v >> 16) as u8);
        }
        if out.len() < decoded {
            out.push((v >> 8) as u8);
        }
        if out.len() < decoded {
            out.push(v as u8);
        }
    }
    if out.len() != decoded {
        return Err("invalid image base64 padding".into());
    }
    Ok(out)
}

fn add_image_base64(
    images: &mut Vec<RequestImage>,
    mime_type: &str,
    payload: &str,
) -> Result<usize, String> {
    let mime = match mime_type {
        "image/png" => ImageMime::Png,
        "image/jpeg" => ImageMime::Jpeg,
        _ => return Err(format!("unsupported image media type: {mime_type}")),
    };
    if images.len() >= CHAT_IMAGE_MAX_COUNT {
        return Err("at most 4 images are supported".into());
    }
    let total = images.iter().map(|image| image.data.len()).sum::<usize>();
    if total > CHAT_IMAGE_TOTAL_MAX {
        return Err("images exceed 20 MiB request limit".into());
    }
    let data = base64_decode_image(payload, CHAT_IMAGE_TOTAL_MAX - total)?;
    let magic_matches = match mime {
        ImageMime::Png => data.starts_with(b"\x89PNG\r\n\x1a\n"),
        ImageMime::Jpeg => data.starts_with(&[0xff, 0xd8, 0xff]),
    };
    if !magic_matches {
        return Err("image bytes do not match declared media type".into());
    }
    let index = images.len();
    images.push(RequestImage {
        mime,
        data: data.into(),
    });
    Ok(index)
}

fn add_image_data_uri(images: &mut Vec<RequestImage>, uri: &str) -> Result<usize, String> {
    let Some(rest) = uri.strip_prefix("data:") else {
        return Err("image_url must be a base64 data URI".into());
    };
    let Some((mime, payload)) = rest.split_once(";base64,") else {
        return Err("image_url must be a base64 data URI".into());
    };
    if mime.is_empty() {
        return Err("image_url must be a base64 data URI".into());
    }
    add_image_base64(images, mime, payload)
}

fn append_text_part(msg: &mut ChatMsg, text: String) {
    msg.content.push_str(&text);
    msg.parts.push(ChatPart::Text(text));
}

fn validate_image_references(msgs: &[ChatMsg], images: &[RequestImage]) -> Result<(), String> {
    if images.is_empty() {
        return Ok(());
    }
    let mut seen = vec![false; images.len()];
    for msg in msgs {
        for part in &msg.parts {
            let ChatPart::Image(index) = part else {
                continue;
            };
            if msg.role != "user" {
                return Err("images are allowed only in user messages".into());
            }
            let Some(slot) = seen.get_mut(*index) else {
                return Err("invalid image reference".into());
            };
            *slot = true;
        }
    }
    if seen.iter().any(|seen| !seen) {
        return Err("unreferenced image payload".into());
    }
    Ok(())
}

fn bad<T>(err: &mut String, fallback: &str) -> Result<T, String> {
    if err.is_empty() {
        *err = fallback.into();
    }
    Err(err.clone())
}

fn json_skip_ok(p: &mut Json<'_>) -> Option<()> {
    if json_skip_value(p) {
        Some(())
    } else {
        None
    }
}

fn parse_budget(p: &mut Json<'_>, key: &str, r: &mut ParsedRequest, err: &mut String) -> bool {
    let Some(budget) = json_number(p) else {
        return false;
    };
    if budget < 0.0 {
        *err = format!("{key} must be >= 0");
        return false;
    }
    r.max_tokens = if budget > i32::MAX as f64 {
        i32::MAX
    } else {
        budget as i32
    };
    r.max_tokens_set = true;
    true
}

fn parse_stop(p: &mut Json<'_>) -> Option<Vec<String>> {
    p.ws();
    let mut out = Vec::new();
    if p.peek() == Some(b'"') {
        let s = json_string(p)?;
        if !s.is_empty() {
            out.push(s);
        }
        return Some(out);
    }
    if p.peek() != Some(b'[') {
        json_skip_ok(p)?;
        return Some(out);
    }
    p.i += 1;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            let s = json_string(p)?;
            if !s.is_empty() {
                out.push(s);
            }
        } else if !json_skip_value(p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(out)
}

fn parse_stream_options(p: &mut Json<'_>) -> Option<bool> {
    p.ws();
    if p.peek() != Some(b'{') {
        json_skip_ok(p)?;
        return Some(false);
    }
    p.i += 1;
    p.ws();
    let mut include = false;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "include_usage" {
            include = json_bool(p)?;
        } else if !json_skip_value(p) {
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
    Some(include)
}

fn parse_parallel_tool_calls(p: &mut Json<'_>, err: &mut String) -> bool {
    match json_bool(p) {
        Some(true) => true,
        Some(false) => {
            *err = "parallel_tool_calls=false is not supported".into();
            false
        }
        None => false,
    }
}

pub fn parse_thinking_control_value(p: &mut Json<'_>) -> Option<Option<bool>> {
    p.ws();
    if p.lit("null") {
        return Some(None);
    }
    if matches!(p.peek(), Some(b't' | b'f')) {
        return Some(Some(json_bool(p)?));
    }
    if p.peek() != Some(b'{') {
        json_skip_ok(p)?;
        return Some(None);
    }
    p.i += 1;
    p.ws();
    let mut enabled = None;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "type" {
            let typ = json_string(p)?;
            if typ == "enabled" {
                enabled = Some(true);
            } else if typ == "disabled" {
                enabled = Some(false);
            }
        } else if !json_skip_value(p) {
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
    Some(enabled)
}

fn parse_openai_tool_choice(p: &mut Json<'_>, err: &mut String) -> Option<ToolChoice> {
    p.ws();
    if p.peek() != Some(b'"') {
        *err = if p.peek() == Some(b'{') {
            "forced tool_choice not supported".into()
        } else {
            "invalid tool_choice".into()
        };
        return None;
    }
    let choice = json_string(p)?;
    match choice.as_str() {
        "auto" => Some(ToolChoice::Auto),
        "none" => Some(ToolChoice::None),
        "required" => Some(ToolChoice::Required),
        other => {
            *err = format!("tool_choice={other} not supported");
            None
        }
    }
}

fn parse_anthropic_tool_choice(p: &mut Json<'_>, err: &mut String) -> Option<ToolChoice> {
    p.ws();
    if p.peek() != Some(b'{') {
        *err = "invalid tool_choice".into();
        return None;
    }
    p.i += 1;
    p.ws();
    let mut typ = None;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            if err.is_empty() {
                *err = "invalid tool_choice".into();
            }
            return None;
        }
        if key == "type" {
            typ = Some(json_string(p)?);
        } else if !json_skip_value(p) {
            if err.is_empty() {
                *err = "invalid tool_choice".into();
            }
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        if err.is_empty() {
            *err = "invalid tool_choice".into();
        }
        return None;
    }
    match typ.as_deref() {
        Some("auto") => Some(ToolChoice::Auto),
        Some("none") => Some(ToolChoice::None),
        Some("any") => Some(ToolChoice::Required),
        Some("tool") => {
            *err = "forced tool_choice not supported".into();
            None
        }
        Some(t) => {
            *err = format!("tool_choice type={t} not supported");
            None
        }
        None => {
            if err.is_empty() {
                *err = "invalid tool_choice".into();
            }
            None
        }
    }
}

fn add_tool_call_id(msg: &mut ChatMsg, id: &str) {
    if id.is_empty() {
        return;
    }
    if msg.tool_call_id.is_empty() {
        msg.tool_call_id = id.to_string();
    }
    if !msg.tool_call_ids.iter().any(|x| x == id) {
        msg.tool_call_ids.push(id.to_string());
    }
}

fn parse_function_call(p: &mut Json<'_>, tc: &mut ToolCall) -> bool {
    p.ws();
    if p.bump() != Some(b'{') {
        return false;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(p) else {
            return false;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if key == "name" {
            let Some(n) = json_string(p) else {
                return false;
            };
            tc.name = n;
        } else if key == "arguments" {
            p.ws();
            if p.peek() == Some(b'"') {
                let Some(a) = json_string(p) else {
                    return false;
                };
                tc.arguments = a;
            } else {
                let Some(a) = json_raw_value(p) else {
                    return false;
                };
                tc.arguments = a;
            }
        } else if !json_skip_value(p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    p.bump() == Some(b'}')
}

fn parse_tool_calls_value(p: &mut Json<'_>) -> Option<Vec<ToolCall>> {
    p.ws();
    if p.lit("null") {
        return Some(Vec::new());
    }
    if p.bump() != Some(b'[') {
        return None;
    }
    let mut calls = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.bump() != Some(b'{') {
            return None;
        }
        let mut tc = ToolCall::default();
        p.ws();
        while p.peek().is_some() && p.peek() != Some(b'}') {
            let key = json_string(p)?;
            p.ws();
            if p.bump() != Some(b':') {
                return None;
            }
            if key == "id" {
                tc.id = json_string(p)?;
            } else if key == "function" {
                if !parse_function_call(p, &mut tc) {
                    return None;
                }
            } else if !json_skip_value(p) {
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
        if !tc.name.is_empty() && !tc.arguments.is_empty() {
            calls.push(tc);
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(calls)
}

fn parse_chat_image_url(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    if p.peek() == Some(b'"') {
        return json_string(p);
    }
    if p.bump() != Some(b'{') {
        return None;
    }
    let mut url = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "url" {
            url = Some(json_string(p)?);
        } else if !json_skip_value(p) {
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
    url
}

fn parse_openai_message_content(
    p: &mut Json<'_>,
    msg: &mut ChatMsg,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> bool {
    p.ws();
    if p.peek() == Some(b'"') {
        return json_string(p).map(|s| msg.content = s).is_some();
    }
    if p.lit("null") {
        msg.content.clear();
        return true;
    }
    if p.bump() != Some(b'[') {
        return false;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            let Some(text) = json_string(p) else {
                return false;
            };
            append_text_part(msg, text);
        } else if p.bump() == Some(b'{') {
            let mut typ = None;
            let mut text = None;
            let mut image_url = None;
            p.ws();
            while p.peek().is_some() && p.peek() != Some(b'}') {
                let Some(key) = json_string(p) else {
                    return false;
                };
                p.ws();
                if p.bump() != Some(b':') {
                    return false;
                }
                if key == "type" {
                    typ = json_string(p);
                    if typ.is_none() {
                        return false;
                    }
                } else if key == "text" {
                    text = json_string(p);
                    if text.is_none() {
                        return false;
                    }
                } else if key == "image_url" {
                    image_url = parse_chat_image_url(p);
                    if image_url.is_none() {
                        return false;
                    }
                } else if !json_skip_value(p) {
                    return false;
                }
                p.ws();
                if p.peek() == Some(b',') {
                    p.i += 1;
                }
                p.ws();
            }
            if p.bump() != Some(b'}') {
                return false;
            }
            match (typ.as_deref(), text, image_url) {
                (Some("text"), Some(text), _) => append_text_part(msg, text),
                (Some("image_url"), _, Some(url)) => match add_image_data_uri(images, &url) {
                    Ok(index) => msg.parts.push(ChatPart::Image(index)),
                    Err(error) => {
                        *err = error;
                        return false;
                    }
                },
                _ => {
                    *err = "unsupported chat content block".into();
                    return false;
                }
            }
        } else {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    p.bump() == Some(b']')
}

fn parse_messages(
    p: &mut Json<'_>,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> Option<Vec<ChatMsg>> {
    p.ws();
    if p.bump() != Some(b'[') {
        return None;
    }
    let mut msgs = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.bump() != Some(b'{') {
            return None;
        }
        let mut msg = ChatMsg::default();
        p.ws();
        while p.peek().is_some() && p.peek() != Some(b'}') {
            let key = json_string(p)?;
            p.ws();
            if p.bump() != Some(b':') {
                return None;
            }
            if key == "role" {
                msg.role = json_string(p)?;
            } else if key == "content" {
                msg.content.clear();
                msg.parts.clear();
                if !parse_openai_message_content(p, &mut msg, images, err) {
                    if err.is_empty() {
                        *err = "invalid chat content".into();
                    }
                    return None;
                }
            } else if key == "reasoning_content" {
                msg.reasoning = json_content(p)?;
            } else if key == "tool_call_id" {
                let id = json_string(p)?;
                add_tool_call_id(&mut msg, &id);
            } else if key == "tool_calls" {
                msg.calls = parse_tool_calls_value(p)?;
            } else if !json_skip_value(p) {
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
        if msg.role.is_empty() {
            msg.role = "user".into();
        }
        msgs.push(msg);
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(msgs)
}

fn append_tool_result_text(out: &mut String, s: &str) {
    let end = "</tool_result>";
    let mut rest = s;
    while !rest.is_empty() {
        if rest.starts_with(end) {
            out.push_str("&lt;");
            rest = &rest[1..];
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
}

fn parse_anthropic_image_source(p: &mut Json<'_>) -> Option<(String, String, String)> {
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    let mut source_type = None;
    let mut media_type = None;
    let mut data = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "type" {
            source_type = Some(json_string(p)?);
        } else if key == "media_type" {
            media_type = Some(json_string(p)?);
        } else if key == "data" {
            data = Some(json_string(p)?);
        } else if !json_skip_value(p) {
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
    Some((
        source_type.unwrap_or_default(),
        media_type.unwrap_or_default(),
        data.unwrap_or_default(),
    ))
}

fn parse_anthropic_content_block(
    p: &mut Json<'_>,
    msg: &mut ChatMsg,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> bool {
    if p.bump() != Some(b'{') {
        return false;
    }
    let mut typ = None;
    let mut text = None;
    let mut thinking = None;
    let mut id = None;
    let mut name = None;
    let mut input = None;
    let mut tool_result = None;
    let mut source = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(p) else {
            return false;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if key == "type" {
            typ = json_string(p);
            if typ.is_none() {
                return false;
            }
        } else if key == "text" {
            text = json_content(p);
            if text.is_none() {
                return false;
            }
        } else if key == "thinking" {
            thinking = json_content(p);
            if thinking.is_none() {
                return false;
            }
        } else if key == "id" || key == "tool_use_id" {
            id = json_string(p);
            if id.is_none() {
                return false;
            }
        } else if key == "name" {
            name = json_string(p);
            if name.is_none() {
                return false;
            }
        } else if key == "input" {
            input = json_raw_value(p);
            if input.is_none() {
                return false;
            }
        } else if key == "content" {
            tool_result = json_content(p);
            if tool_result.is_none() {
                return false;
            }
        } else if key == "source" {
            source = parse_anthropic_image_source(p);
            if source.is_none() {
                return false;
            }
        } else if !json_skip_value(p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        return false;
    }
    match typ.as_deref() {
        Some("image") => {
            let Some((source_type, media_type, data)) = source else {
                *err = "Anthropic image source must use base64 data".into();
                return false;
            };
            if source_type != "base64" || media_type.is_empty() || data.is_empty() {
                *err = "Anthropic image source must use base64 data".into();
                return false;
            }
            match add_image_base64(images, &media_type, &data) {
                Ok(index) => msg.parts.push(ChatPart::Image(index)),
                Err(error) => {
                    *err = error;
                    return false;
                }
            }
        }
        Some("tool_use") => {
            msg.calls.push(ToolCall {
                id: id.unwrap_or_default(),
                name: name.unwrap_or_default(),
                arguments: input.unwrap_or_else(|| "{}".into()),
            });
        }
        Some("tool_result") => {
            if let Some(ref i) = id {
                add_tool_call_id(msg, i);
            }
            let mut b = msg.content.clone();
            b.push_str("<tool_result>");
            append_tool_result_text(&mut b, tool_result.as_deref().unwrap_or(""));
            b.push_str("</tool_result>");
            msg.content = b;
        }
        _ => {
            if let Some(t) = text {
                append_text_part(msg, t);
            }
            if let Some(t) = thinking {
                msg.reasoning.push_str(&t);
            }
        }
    }
    true
}

fn parse_anthropic_content(
    p: &mut Json<'_>,
    msg: &mut ChatMsg,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> bool {
    p.ws();
    if p.peek() == Some(b'"') {
        return json_string(p).map(|s| msg.content = s).is_some();
    }
    if p.lit("null") {
        msg.content = String::new();
        return true;
    }
    if p.peek() != Some(b'[') {
        return json_skip_value(p);
    }
    p.i += 1;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            if let Some(s) = json_string(p) {
                append_text_part(msg, s);
            } else {
                return false;
            }
        } else if p.peek() == Some(b'{') {
            if !parse_anthropic_content_block(p, msg, images, err) {
                return false;
            }
        } else if !json_skip_value(p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    p.bump() == Some(b']')
}

fn parse_anthropic_messages(
    p: &mut Json<'_>,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> Option<Vec<ChatMsg>> {
    p.ws();
    if p.bump() != Some(b'[') {
        return None;
    }
    let mut msgs = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.bump() != Some(b'{') {
            return None;
        }
        let mut msg = ChatMsg::default();
        p.ws();
        while p.peek().is_some() && p.peek() != Some(b'}') {
            let key = json_string(p)?;
            p.ws();
            if p.bump() != Some(b':') {
                return None;
            }
            if key == "role" {
                msg.role = json_string(p)?;
            } else if key == "content" {
                msg.content.clear();
                msg.parts.clear();
                if !parse_anthropic_content(p, &mut msg, images, err) {
                    return None;
                }
            } else if !json_skip_value(p) {
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
        if msg.role.is_empty() {
            msg.role = "user".into();
        }
        msgs.push(msg);
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(msgs)
}

fn anthropic_system_part_is_private(s: &str) -> bool {
    s.starts_with("x-anthropic-")
}

fn append_anthropic_system_part(b: &mut String, s: &str) {
    if s.is_empty() || anthropic_system_part_is_private(s) {
        return;
    }
    if !b.is_empty() && !b.ends_with('\n') {
        b.push('\n');
    }
    b.push_str(s);
}

fn parse_anthropic_system_object(p: &mut Json<'_>, out: &mut String) -> bool {
    if p.bump() != Some(b'{') {
        return false;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(p) else {
            return false;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if key == "text" {
            let Some(text) = json_string(p) else {
                return false;
            };
            append_anthropic_system_part(out, &text);
        } else if !json_skip_value(p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    p.bump() == Some(b'}')
}

fn parse_anthropic_system(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    let mut b = String::new();
    if p.peek() == Some(b'"') {
        let text = json_string(p)?;
        append_anthropic_system_part(&mut b, &text);
        return Some(b);
    }
    if p.lit("null") {
        return Some(String::new());
    }
    if p.peek() != Some(b'[') {
        json_skip_ok(p)?;
        return Some(String::new());
    }
    p.i += 1;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            let text = json_string(p)?;
            append_anthropic_system_part(&mut b, &text);
        } else if p.peek() == Some(b'{') {
            if !parse_anthropic_system_object(p, &mut b) {
                return None;
            }
        } else if !json_skip_value(p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(b)
}

fn parse_prompt(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    if p.peek() == Some(b'"') {
        return json_string(p);
    }
    if p.peek() != Some(b'[') {
        json_skip_ok(p)?;
        return Some(String::new());
    }
    p.i += 1;
    p.ws();
    let out = if p.peek() == Some(b'"') {
        json_string(p)?
    } else {
        if p.peek().is_some() && p.peek() != Some(b']') && !json_skip_value(p) {
            return None;
        }
        String::new()
    };
    while p.peek().is_some() && p.peek() != Some(b']') {
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
            if !json_skip_value(p) {
                return None;
            }
        } else {
            break;
        }
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(out)
}

fn openai_function_schema_from_tool(raw: &str) -> Option<String> {
    let mut p = Json::new(raw);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(&mut p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "function" {
            return json_raw_value(&mut p);
        }
        json_skip_ok(&mut p)?;
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    None
}

fn responses_special_schema_from_tool(raw: &str) -> Option<String> {
    let mut p = Json::new(raw);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    let mut typ = None;
    let mut description = None;
    let mut parameters = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(&mut p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "type" {
            typ = Some(json_string(&mut p)?);
        } else if key == "description" {
            description = Some(json_string(&mut p)?);
        } else if key == "parameters" {
            parameters = Some(json_raw_value(&mut p)?);
        } else if !json_skip_value(&mut p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if typ.as_deref() == Some("tool_search") {
        let desc = description.unwrap_or_else(|| "Search available tools.".into());
        let params = parameters.unwrap_or_else(|| "{\"type\":\"object\",\"properties\":{}}".into());
        return Some(format!(
            "{{\"name\":\"tool_search\",\"description\":{},\"parameters\":{params}}}",
            json_escape(&desc)
        ));
    }
    None
}

fn schema_property_type(json: &str) -> Option<String> {
    let mut p = Json::new(json);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(&mut p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "type" {
            return json_string(&mut p);
        }
        json_skip_ok(&mut p)?;
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    None
}

fn parse_schema_properties(json: &str, order: &mut ToolSchemaOrder) -> bool {
    let mut p = Json::new(json);
    p.ws();
    if p.bump() != Some(b'{') {
        return false;
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return false;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if key == "properties" {
            p.ws();
            if p.bump() != Some(b'{') {
                return false;
            }
            p.ws();
            while p.peek().is_some() && p.peek() != Some(b'}') {
                let Some(prop) = json_string(&mut p) else {
                    return false;
                };
                p.ws();
                if p.bump() != Some(b':') {
                    return false;
                }
                let Some(property) = json_raw_value(&mut p) else {
                    return false;
                };
                let typ = schema_property_type(&property).unwrap_or_default();
                order.prop.push(prop);
                order.prop_type.push(typ);
                p.ws();
                if p.peek() == Some(b',') {
                    p.i += 1;
                }
                p.ws();
            }
            if p.bump() != Some(b'}') {
                return false;
            }
        } else if !json_skip_value(&mut p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    p.peek() == Some(b'}')
}

fn tool_schema_orders_add_json_wire(
    orders: &mut Vec<ToolSchemaOrder>,
    json: &str,
    namespace: Option<&str>,
    wire_name: Option<&str>,
    responses_tool_search: bool,
) {
    let mut p = Json::new(json);
    p.ws();
    if p.bump() != Some(b'{') {
        return;
    }
    let mut order = ToolSchemaOrder {
        responses_tool_search,
        ..Default::default()
    };
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return;
        }
        if key == "name" {
            if let Some(n) = json_string(&mut p) {
                order.name = n;
            } else {
                return;
            }
        } else if key == "input_schema" || key == "parameters" {
            if let Some(schema) = json_raw_value(&mut p) {
                parse_schema_properties(&schema, &mut order);
            } else {
                return;
            }
        } else if !json_skip_value(&mut p) {
            return;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if !order.name.is_empty() {
        if let Some(ns) = namespace {
            if !ns.is_empty() {
                order.namespace = ns.into();
            }
        }
        if let Some(w) = wire_name {
            if !w.is_empty() {
                order.wire_name = w.into();
            }
        }
        orders.push(order);
    }
}

fn responses_namespace_function_schema(raw: &str, namespace: &str) -> Option<(String, String)> {
    let mut p = Json::new(raw);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    let mut typ = None;
    let mut name = None;
    let mut description = None;
    let mut parameters = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(&mut p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "type" {
            typ = Some(json_string(&mut p)?);
        } else if key == "name" {
            name = Some(json_string(&mut p)?);
        } else if key == "description" {
            description = Some(json_string(&mut p)?);
        } else if key == "parameters" || key == "input_schema" {
            parameters = Some(json_raw_value(&mut p)?);
        } else if !json_skip_value(&mut p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    let name = name?;
    if name.is_empty() {
        return None;
    }
    if typ.as_deref().is_some_and(|t| t != "function") {
        return None;
    }
    let prompt_name = format!("{namespace}{name}");
    let desc = description.unwrap_or_default();
    let params = parameters.unwrap_or_else(|| "{\"type\":\"object\",\"properties\":{}}".into());
    Some((
        format!(
            "{{\"name\":{},\"description\":{},\"parameters\":{params}}}",
            json_escape(&prompt_name),
            json_escape(&desc)
        ),
        name,
    ))
}

fn append_responses_namespace_tool_schemas(
    schemas: &mut String,
    orders: &mut Vec<ToolSchemaOrder>,
    raw: &str,
) -> bool {
    let mut p = Json::new(raw);
    p.ws();
    if p.bump() != Some(b'{') {
        return false;
    }
    let mut typ = None;
    let mut name = None;
    let mut tools = None;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return false;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if key == "type" {
            typ = json_string(&mut p);
        } else if key == "name" {
            name = json_string(&mut p);
        } else if key == "tools" {
            tools = json_raw_value(&mut p);
        } else if !json_skip_value(&mut p) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if typ.as_deref() != Some("namespace") || name.is_none() || tools.is_none() {
        return false;
    }
    let name = name.unwrap();
    let mut tp = Json::new(tools.as_ref().unwrap());
    tp.ws();
    if tp.bump() != Some(b'[') {
        return false;
    }
    tp.ws();
    let mut appended = false;
    while tp.peek().is_some() && tp.peek() != Some(b']') {
        let Some(tool_raw) = json_raw_value(&mut tp) else {
            return false;
        };
        if let Some((schema, wire)) = responses_namespace_function_schema(&tool_raw, &name) {
            if !schemas.is_empty() {
                schemas.push('\n');
            }
            schemas.push_str(&schema);
            tool_schema_orders_add_json_wire(orders, &schema, Some(&name), Some(&wire), false);
            appended = true;
        }
        tp.ws();
        if tp.peek() == Some(b',') {
            tp.i += 1;
        }
        tp.ws();
    }
    appended
}

fn parse_tools_value(p: &mut Json<'_>) -> Option<(String, Vec<ToolSchemaOrder>)> {
    p.ws();
    if p.lit("null") {
        return Some((String::new(), Vec::new()));
    }
    if p.bump() != Some(b'[') {
        return None;
    }
    let mut schemas = String::new();
    let mut orders = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        let raw = json_raw_value(p)?;
        if let Some(function) = openai_function_schema_from_tool(&raw) {
            if !schemas.is_empty() {
                schemas.push('\n');
            }
            schemas.push_str(&function);
            tool_schema_orders_add_json_wire(&mut orders, &function, None, None, false);
        } else if !append_responses_namespace_tool_schemas(&mut schemas, &mut orders, &raw) {
            if let Some(special) = responses_special_schema_from_tool(&raw) {
                if !schemas.is_empty() {
                    schemas.push('\n');
                }
                schemas.push_str(&special);
                tool_schema_orders_add_json_wire(&mut orders, &special, None, None, true);
            } else {
                if !schemas.is_empty() {
                    schemas.push('\n');
                }
                schemas.push_str(&raw);
                tool_schema_orders_add_json_wire(&mut orders, &raw, None, None, false);
            }
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some((schemas, orders))
}

fn chat_msg_is_model_tool_result(m: &ChatMsg) -> bool {
    m.role == "tool"
        || m.role == "function"
        || (m.role == "user" && (!m.tool_call_id.is_empty() || !m.tool_call_ids.is_empty()))
}

fn chat_history_has_pending_tool_results(msgs: &[ChatMsg]) -> bool {
    let mut saw = false;
    for m in msgs.iter().rev() {
        if m.role == "system" || m.role == "developer" {
            continue;
        }
        if chat_msg_is_model_tool_result(m) {
            saw = true;
            continue;
        }
        if m.role == "assistant" {
            return saw;
        }
    }
    saw
}

fn msg_has_call_id(m: &ChatMsg, id: &str) -> bool {
    m.role == "assistant" && m.calls.iter().any(|c| c.id == id)
}

fn find_prior_call<'a>(msgs: &'a [ChatMsg], before: usize, id: &str) -> Option<&'a ChatMsg> {
    if id.is_empty() {
        return None;
    }
    let end = before.min(msgs.len());
    msgs[..end].iter().rev().find(|m| msg_has_call_id(m, id))
}

fn collect_tool_call_ids(m: &ChatMsg) -> Vec<String> {
    let mut ids = Vec::new();
    if !m.tool_call_id.is_empty() {
        ids.push(m.tool_call_id.clone());
    }
    for id in &m.tool_call_ids {
        if !ids.iter().any(|x| x == id) {
            ids.push(id.clone());
        }
    }
    ids
}

fn prepare_tool_choice(
    r: &mut ParsedRequest,
    schemas: &mut String,
    orders: &mut Vec<ToolSchemaOrder>,
    err: &mut String,
) -> bool {
    if r.tool_choice == ToolChoice::None {
        schemas.clear();
        orders.clear();
    }
    r.has_tools = !schemas.is_empty();
    if r.tool_choice == ToolChoice::Required && !r.has_tools {
        *err = "tool_choice=required requires at least one tool".into();
        return false;
    }
    true
}

fn apply_think_aliases(r: &ParsedRequest, got_thinking: bool, enabled: bool) -> bool {
    let mut e = enabled;
    if !got_thinking && model_alias_disables_thinking(&r.model) {
        e = false;
    }
    if !got_thinking && model_alias_enables_thinking(&r.model) {
        e = true;
    }
    e
}

fn id_is_live(live: &[String], id: &str) -> bool {
    live.iter().any(|x| x == id)
}

fn anthropic_validate_tool_results(
    msgs: &[ChatMsg],
    live: &[String],
    err: &mut String,
) -> Result<bool, ()> {
    let mut requires_live = false;
    for (i, m) in msgs.iter().enumerate() {
        if m.role != "user" || (m.tool_call_id.is_empty() && m.tool_call_ids.is_empty()) {
            continue;
        }
        for id in collect_tool_call_ids(m) {
            let prior = find_prior_call(msgs, i, &id);
            if prior.is_none() && !id_is_live(live, &id) {
                *err = format!(
                    "Anthropic continuation state is not available for tool_use_id {id}; retry by replaying the full messages history"
                );
                return Err(());
            }
            if prior.is_none() {
                requires_live = true;
            }
        }
    }
    Ok(requires_live)
}

fn responses_validate_tool_outputs(
    msgs: &[ChatMsg],
    think: ThinkMode,
    live: &[String],
    err: &mut String,
) -> Result<(bool, bool), ()> {
    let mut live_tool = false;
    let mut live_reason = false;
    let needs_reasoning = crate::route::think_mode_enabled(think);
    for (i, m) in msgs.iter().enumerate() {
        if m.role != "tool" && m.role != "function" {
            continue;
        }
        for id in collect_tool_call_ids(m) {
            let prior = find_prior_call(msgs, i, &id);
            if prior.is_none() && !id_is_live(live, &id) {
                *err = format!(
                    "Responses continuation state is not available for call_id {id}; retry by replaying the full input history"
                );
                return Err(());
            }
            if prior.is_none() {
                live_tool = true;
                continue;
            }
            if needs_reasoning && prior.unwrap().reasoning.is_empty() {
                live_reason = true;
            }
        }
    }
    Ok((live_tool, live_reason))
}

fn parse_responses_content_array_mm(
    p: &mut Json<'_>,
    mut images: Option<&mut Vec<RequestImage>>,
    err: &mut String,
) -> Option<(String, Vec<ChatPart>)> {
    p.ws();
    if p.peek() == Some(b'"') {
        let text = json_string(p)?;
        return Some((text.clone(), vec![ChatPart::Text(text)]));
    }
    if p.lit("null") {
        return Some((String::new(), Vec::new()));
    }
    if p.peek() != Some(b'[') {
        return None;
    }
    p.i += 1;
    let mut b = String::new();
    let mut parts = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            let text = json_string(p)?;
            b.push_str(&text);
            parts.push(ChatPart::Text(text));
        } else if p.peek() == Some(b'{') {
            p.i += 1;
            let mut typ = None;
            let mut text = None;
            let mut image_url = None;
            p.ws();
            while p.peek().is_some() && p.peek() != Some(b'}') {
                let key = json_string(p)?;
                p.ws();
                if p.bump() != Some(b':') {
                    return None;
                }
                if key == "type" {
                    typ = Some(json_string(p)?);
                } else if key == "text" {
                    p.ws();
                    if p.lit("null") {
                        text = Some(String::new());
                    } else {
                        text = Some(json_string(p)?);
                    }
                } else if key == "image_url" {
                    image_url = Some(json_string(p)?);
                } else if !json_skip_value(p) {
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
            let is_text = matches!(
                typ.as_deref(),
                Some("input_text" | "output_text" | "text" | "summary_text" | "reasoning_text")
            );
            if is_text {
                let text = text?;
                b.push_str(&text);
                parts.push(ChatPart::Text(text));
            } else if typ.as_deref() == Some("input_image") {
                let Some(images) = images.as_deref_mut() else {
                    *err = "unsupported Responses content block".into();
                    return None;
                };
                let Some(image_url) = image_url else {
                    *err = "unsupported Responses content block".into();
                    return None;
                };
                match add_image_data_uri(images, &image_url) {
                    Ok(index) => parts.push(ChatPart::Image(index)),
                    Err(error) => {
                        *err = error;
                        return None;
                    }
                }
            } else {
                *err = "unsupported Responses content block".into();
                return None;
            }
        } else {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some((b, parts))
}

fn parse_responses_content_array(p: &mut Json<'_>) -> Option<String> {
    parse_responses_content_array_mm(p, None, &mut String::new()).map(|(text, _)| text)
}

fn parse_responses_reasoning(
    p: &mut Json<'_>,
    effort: &mut ThinkMode,
    summary_opted: &mut bool,
) -> Option<bool> {
    p.ws();
    if p.lit("null") {
        return Some(false);
    }
    if p.peek() != Some(b'{') {
        json_skip_ok(p)?;
        return Some(false);
    }
    p.i += 1;
    p.ws();
    let mut effort_seen = false;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p)?;
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        if key == "effort" {
            p.ws();
            if p.lit("null") {
            } else {
                *effort = parse_reasoning_effort_value(p).ok()??;
                effort_seen = true;
            }
        } else if key == "summary" {
            p.ws();
            if p.lit("null") {
            } else if p.peek() == Some(b'"') {
                let mode = json_string(p)?;
                if matches!(mode.as_str(), "auto" | "concise" | "detailed") {
                    *summary_opted = true;
                }
            } else if !json_skip_value(p) {
                return None;
            }
        } else if !json_skip_value(p) {
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
    Some(effort_seen)
}

fn parse_responses_input(
    p: &mut Json<'_>,
    loaded: &mut String,
    orders: &mut Vec<ToolSchemaOrder>,
    images: &mut Vec<RequestImage>,
    err: &mut String,
) -> Option<Vec<ChatMsg>> {
    p.ws();
    if p.bump() != Some(b'[') {
        return None;
    }
    let mut msgs = Vec::new();
    let mut pending = String::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.bump() != Some(b'{') {
            return None;
        }
        let mut typ = None;
        let mut role = None;
        let mut content = None;
        let mut content_parts = Vec::new();
        let mut name = None;
        let mut namespace = None;
        let mut call_id = None;
        let mut item_id = None;
        let mut arguments = None;
        let mut output = None;
        let mut input_str = None;
        let mut summary = None;
        let mut action = None;
        let mut result = None;
        let mut tools_json = None;
        let mut status_str = None;
        p.ws();
        while p.peek().is_some() && p.peek() != Some(b'}') {
            let key = json_string(p)?;
            p.ws();
            if p.bump() != Some(b':') {
                return None;
            }
            if key == "type" {
                typ = Some(json_string(p)?);
            } else if key == "role" {
                role = Some(json_string(p)?);
            } else if key == "content" {
                let parsed = parse_responses_content_array_mm(p, Some(images), err)?;
                content = Some(parsed.0);
                content_parts = parsed.1;
            } else if key == "name" {
                name = Some(json_string(p)?);
            } else if key == "namespace" {
                namespace = Some(json_string(p)?);
            } else if key == "call_id" {
                call_id = Some(json_string(p)?);
            } else if key == "id" {
                item_id = Some(json_string(p)?);
            } else if key == "arguments" {
                p.ws();
                arguments = if p.peek() == Some(b'"') {
                    Some(json_string(p)?)
                } else {
                    Some(json_raw_value(p)?)
                };
            } else if key == "output" {
                p.ws();
                output = if p.peek() == Some(b'[') {
                    Some(parse_responses_content_array(p)?)
                } else if p.peek() == Some(b'"') {
                    Some(json_string(p)?)
                } else {
                    Some(json_raw_value(p)?)
                };
            } else if key == "input" {
                p.ws();
                input_str = if p.peek() == Some(b'"') {
                    Some(json_string(p)?)
                } else {
                    Some(json_raw_value(p)?)
                };
            } else if key == "summary" {
                summary = Some(parse_responses_content_array(p)?);
            } else if key == "action" {
                action = Some(json_raw_value(p)?);
            } else if key == "result" {
                p.ws();
                result = if p.peek() == Some(b'"') {
                    Some(json_string(p)?)
                } else {
                    Some(json_raw_value(p)?)
                };
            } else if key == "status" {
                status_str = Some(json_string(p)?);
            } else if key == "tools" {
                tools_json = Some(json_raw_value(p)?);
            } else if !json_skip_value(p) {
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
        if let Some(st) = status_str.as_deref() {
            if !st.is_empty() && st != "completed" {
                return None;
            }
        }
        let t = typ.as_deref().unwrap_or("message");
        let consumes = (t == "message" && role.as_deref() == Some("assistant"))
            || matches!(
                t,
                "function_call"
                    | "custom_tool_call"
                    | "local_shell_call"
                    | "web_search_call"
                    | "tool_search_call"
                    | "image_generation_call"
            );
        let bookkeeping = t == "compaction" || t == "context_compaction";
        if !consumes && !bookkeeping && !pending.is_empty() {
            msgs.push(ChatMsg {
                role: "assistant".into(),
                content: String::new(),
                reasoning: std::mem::take(&mut pending),
                ..Default::default()
            });
        }
        if t == "message" {
            let mut msg = ChatMsg {
                role: role.unwrap_or_else(|| "user".into()),
                content: content.take().unwrap_or_default(),
                parts: std::mem::take(&mut content_parts),
                ..Default::default()
            };
            if msg.role == "assistant" && !pending.is_empty() {
                msg.reasoning = std::mem::take(&mut pending);
            }
            msgs.push(msg);
        } else if t == "function_call" || t == "custom_tool_call" {
            let args = arguments
                .as_deref()
                .or(input_str.as_deref())
                .unwrap_or("{}");
            let tc_name = if t != "custom_tool_call"
                && namespace.as_ref().is_some_and(|n| !n.is_empty())
                && name.as_ref().is_some_and(|n| !n.is_empty())
            {
                format!("{}{}", namespace.as_ref().unwrap(), name.as_ref().unwrap())
            } else {
                name.clone().unwrap_or_default()
            };
            let tc = ToolCall {
                id: call_id.clone().or(item_id.clone()).unwrap_or_default(),
                name: tc_name,
                arguments: args.to_string(),
            };
            if let Some(last) = msgs.last_mut().filter(|m| m.role == "assistant") {
                if !pending.is_empty() && last.reasoning.is_empty() {
                    last.reasoning = std::mem::take(&mut pending);
                }
                last.calls.push(tc);
            } else {
                let mut msg = ChatMsg {
                    role: "assistant".into(),
                    ..Default::default()
                };
                if !pending.is_empty() {
                    msg.reasoning = std::mem::take(&mut pending);
                }
                msg.calls.push(tc);
                msgs.push(msg);
            }
        } else if t == "function_call_output" || t == "custom_tool_call_output" {
            let mut msg = ChatMsg {
                role: "tool".into(),
                content: output.take().unwrap_or_default(),
                ..Default::default()
            };
            if let Some(id) = call_id.as_deref().or(item_id.as_deref()) {
                add_tool_call_id(&mut msg, id);
            }
            msgs.push(msg);
        } else if t == "reasoning" {
            if let Some(s) = summary.as_deref() {
                if !s.is_empty() {
                    if !pending.is_empty() {
                        pending.push('\n');
                    }
                    pending.push_str(s);
                }
            }
            if let Some(s) = content.as_deref() {
                if !s.is_empty() {
                    if !pending.is_empty() {
                        pending.push('\n');
                    }
                    pending.push_str(s);
                }
            }
        } else if matches!(
            t,
            "local_shell_call" | "web_search_call" | "tool_search_call" | "image_generation_call"
        ) {
            let tc = ToolCall {
                id: call_id.clone().or(item_id.clone()).unwrap_or_default(),
                name: match t {
                    "tool_search_call" => "tool_search".into(),
                    "local_shell_call" => "local_shell".into(),
                    other => other.into(),
                },
                arguments: action
                    .as_deref()
                    .or(arguments.as_deref())
                    .or(input_str.as_deref())
                    .unwrap_or("{}")
                    .to_string(),
            };
            if let Some(last) = msgs.last_mut().filter(|m| m.role == "assistant") {
                if !pending.is_empty() && last.reasoning.is_empty() {
                    last.reasoning = std::mem::take(&mut pending);
                }
                last.calls.push(tc);
            } else {
                let mut msg = ChatMsg {
                    role: "assistant".into(),
                    ..Default::default()
                };
                if !pending.is_empty() {
                    msg.reasoning = std::mem::take(&mut pending);
                }
                msg.calls.push(tc);
                msgs.push(msg);
            }
        } else if matches!(
            t,
            "local_shell_call_output"
                | "web_search_call_output"
                | "tool_search_output"
                | "tool_search_call_output"
                | "image_generation_call_output"
        ) {
            if t == "tool_search_output" {
                if let Some(ref tj) = tools_json {
                    let mut tp = Json::new(tj);
                    if let Some((schemas, ords)) = parse_tools_value(&mut tp) {
                        if !schemas.is_empty() {
                            if !loaded.is_empty() {
                                loaded.push('\n');
                            }
                            loaded.push_str(&schemas);
                        }
                        orders.extend(ords);
                    } else {
                        return None;
                    }
                }
            }
            let body = output
                .as_deref()
                .or(result.as_deref())
                .or(tools_json.as_deref())
                .unwrap_or("");
            let mut msg = ChatMsg {
                role: "tool".into(),
                content: body.to_string(),
                ..Default::default()
            };
            if let Some(id) = call_id.as_deref().or(item_id.as_deref()) {
                add_tool_call_id(&mut msg, id);
            }
            msgs.push(msg);
        } else if !bookkeeping {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    if !pending.is_empty() {
        msgs.push(ChatMsg {
            role: "assistant".into(),
            reasoning: pending,
            ..Default::default()
        });
    }
    Some(msgs)
}

fn apply_think(r: &mut ParsedRequest, got: bool, mut enabled: bool, effort: ThinkMode) {
    enabled = apply_think_aliases(r, got, enabled);
    r.think_mode = think_mode_from_enabled(enabled, effort);
}

pub fn parse_chat_request(env: &ParseEnv, body: &str) -> Result<ParsedRequest, String> {
    let mut r = ParsedRequest::init(ReqKind::Chat, env);
    let mut err = String::new();
    let mut p = Json::new(body);
    let mut got_messages = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = env.default_effort;
    let mut msgs = Vec::new();
    let mut images = Vec::new();
    let mut tool_schemas = String::new();
    let mut orders = Vec::new();

    p.ws();
    if p.bump() != Some(b'{') {
        return bad(&mut err, "invalid JSON request");
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return bad(&mut err, "invalid JSON request");
        };
        p.ws();
        if p.bump() != Some(b':') {
            return bad(&mut err, "invalid JSON request");
        }
        let ok = if key == "messages" {
            match parse_messages(&mut p, &mut images, &mut err) {
                Some(m) => {
                    msgs = m;
                    got_messages = true;
                    true
                }
                None => false,
            }
        } else if key == "tools" {
            match parse_tools_value(&mut p) {
                Some((s, o)) => {
                    tool_schemas = s;
                    orders = o;
                    true
                }
                None => false,
            }
        } else if key == "tool_choice" {
            match parse_openai_tool_choice(&mut p, &mut err) {
                Some(c) => {
                    r.tool_choice = c;
                    true
                }
                None => false,
            }
        } else if key == "parallel_tool_calls" {
            parse_parallel_tool_calls(&mut p, &mut err)
        } else if key == "model" {
            match json_string(&mut p) {
                Some(m) => {
                    r.model = m;
                    r.model_from_request = true;
                    true
                }
                None => false,
            }
        } else if key == "max_tokens" || key == "max_completion_tokens" {
            parse_budget(&mut p, &key, &mut r, &mut err)
        } else if key == "temperature" {
            json_number(&mut p)
                .map(|v| r.temperature = v as f32)
                .is_some()
        } else if key == "top_p" {
            json_number(&mut p).map(|v| r.top_p = v as f32).is_some()
        } else if key == "min_p" {
            json_number(&mut p).map(|v| r.min_p = v as f32).is_some()
        } else if key == "top_k" {
            json_int(&mut p).map(|v| r.top_k = v).is_some()
        } else if key == "seed" {
            json_number(&mut p)
                .map(|v| r.seed = if v > 0.0 { v as u64 } else { 0 })
                .is_some()
        } else if key == "stream" {
            json_bool(&mut p).map(|v| r.stream = v).is_some()
        } else if key == "stream_options" {
            parse_stream_options(&mut p)
                .map(|v| r.stream_include_usage = v)
                .is_some()
        } else if key == "return_token_ids" {
            json_bool(&mut p).map(|v| r.return_token_ids = v).is_some()
        } else if key == "thinking" {
            match parse_thinking_control_value(&mut p) {
                Some(v) => {
                    if let Some(e) = v {
                        thinking_enabled = e;
                    }
                    got_thinking = true;
                    true
                }
                None => false,
            }
        } else if key == "reasoning_effort" {
            match parse_reasoning_effort_value(&mut p) {
                Ok(Some(e)) => {
                    reasoning_effort = e;
                    true
                }
                Ok(None) => true,
                Err(_) => false,
            }
        } else if key == "think" || key == "enable_thinking" {
            match json_bool(&mut p) {
                Some(v) => {
                    thinking_enabled = v;
                    got_thinking = true;
                    true
                }
                None => false,
            }
        } else if key == "stop" {
            match parse_stop(&mut p) {
                Some(s) => {
                    r.stops = s;
                    true
                }
                None => false,
            }
        } else if key == "response_format" {
            parse_output_format_value(&mut p, "response_format")
                .map_err(|e| err = e)
                .is_ok()
        } else {
            json_skip_value(&mut p)
        };
        if !ok {
            return bad(&mut err, "invalid JSON request");
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.peek() != Some(b'}') {
        return bad(&mut err, "invalid JSON request");
    }
    if !got_messages {
        return Err("missing messages".into());
    }
    validate_image_references(&msgs, &images)?;
    r.has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if !prepare_tool_choice(&mut r, &mut tool_schemas, &mut orders, &mut err) {
        return Err(err);
    }
    apply_think(&mut r, got_thinking, thinking_enabled, reasoning_effort);
    r.messages = msgs;
    r.images = images;
    r.tool_schemas = tool_schemas;
    r.tool_orders = orders;
    r.finish_needs();
    Ok(r)
}

pub fn parse_completion_request(env: &ParseEnv, body: &str) -> Result<ParsedRequest, String> {
    let mut r = ParsedRequest::init(ReqKind::Completion, env);
    let mut err = String::new();
    let mut p = Json::new(body);
    let mut prompt = None;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = env.default_effort;

    p.ws();
    if p.bump() != Some(b'{') {
        return bad(&mut err, "invalid JSON request");
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return bad(&mut err, "invalid JSON request");
        };
        p.ws();
        if p.bump() != Some(b':') {
            return bad(&mut err, "invalid JSON request");
        }
        let ok = if key == "prompt" {
            match parse_prompt(&mut p) {
                Some(s) => {
                    prompt = Some(s);
                    true
                }
                None => false,
            }
        } else if key == "model" {
            match json_string(&mut p) {
                Some(m) => {
                    r.model = m;
                    r.model_from_request = true;
                    true
                }
                None => false,
            }
        } else if key == "max_tokens" {
            parse_budget(&mut p, &key, &mut r, &mut err)
        } else if key == "temperature" {
            json_number(&mut p)
                .map(|v| r.temperature = v as f32)
                .is_some()
        } else if key == "top_p" {
            json_number(&mut p).map(|v| r.top_p = v as f32).is_some()
        } else if key == "min_p" {
            json_number(&mut p).map(|v| r.min_p = v as f32).is_some()
        } else if key == "top_k" {
            json_int(&mut p).map(|v| r.top_k = v).is_some()
        } else if key == "seed" {
            json_number(&mut p)
                .map(|v| r.seed = if v > 0.0 { v as u64 } else { 0 })
                .is_some()
        } else if key == "stream" {
            json_bool(&mut p).map(|v| r.stream = v).is_some()
        } else if key == "stream_options" {
            parse_stream_options(&mut p)
                .map(|v| r.stream_include_usage = v)
                .is_some()
        } else if key == "return_token_ids" {
            json_bool(&mut p).map(|v| r.return_token_ids = v).is_some()
        } else if key == "thinking" {
            match parse_thinking_control_value(&mut p) {
                Some(v) => {
                    if let Some(e) = v {
                        thinking_enabled = e;
                    }
                    got_thinking = true;
                    true
                }
                None => false,
            }
        } else if key == "reasoning_effort" {
            match parse_reasoning_effort_value(&mut p) {
                Ok(Some(e)) => {
                    reasoning_effort = e;
                    true
                }
                Ok(None) => true,
                Err(_) => false,
            }
        } else if key == "think" || key == "enable_thinking" {
            match json_bool(&mut p) {
                Some(v) => {
                    thinking_enabled = v;
                    got_thinking = true;
                    true
                }
                None => false,
            }
        } else if key == "stop" {
            match parse_stop(&mut p) {
                Some(s) => {
                    r.stops = s;
                    true
                }
                None => false,
            }
        } else if key == "response_format" {
            parse_output_format_value(&mut p, "response_format")
                .map_err(|e| err = e)
                .is_ok()
        } else {
            json_skip_value(&mut p)
        };
        if !ok {
            return bad(&mut err, "invalid JSON request");
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.peek() != Some(b'}') {
        return bad(&mut err, "invalid JSON request");
    }
    let Some(prompt) = prompt else {
        return Err("missing prompt".into());
    };
    apply_think(&mut r, got_thinking, thinking_enabled, reasoning_effort);
    r.prompt_text = Some(prompt);
    r.finish_needs();
    Ok(r)
}

pub fn parse_anthropic_request(env: &ParseEnv, body: &str) -> Result<ParsedRequest, String> {
    let mut r = ParsedRequest::init(ReqKind::Chat, env);
    r.api = Api::Anthropic;
    let mut err = String::new();
    let mut p = Json::new(body);
    let mut got_messages = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = env.default_effort;
    let mut msgs = Vec::new();
    let mut images = Vec::new();
    let mut system = None;
    let mut tool_schemas = String::new();
    let mut orders = Vec::new();

    p.ws();
    if p.bump() != Some(b'{') {
        return bad(&mut err, "invalid JSON request");
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return bad(&mut err, "invalid JSON request");
        };
        p.ws();
        if p.bump() != Some(b':') {
            return bad(&mut err, "invalid JSON request");
        }
        let ok = if key == "messages" {
            match parse_anthropic_messages(&mut p, &mut images, &mut err) {
                Some(m) => {
                    msgs = m;
                    got_messages = true;
                    true
                }
                None => false,
            }
        } else if key == "system" {
            match parse_anthropic_system(&mut p) {
                Some(s) => {
                    system = Some(s);
                    true
                }
                None => false,
            }
        } else if key == "tools" {
            match parse_tools_value(&mut p) {
                Some((s, o)) => {
                    tool_schemas = s;
                    orders = o;
                    true
                }
                None => false,
            }
        } else if key == "tool_choice" {
            match parse_anthropic_tool_choice(&mut p, &mut err) {
                Some(c) => {
                    r.tool_choice = c;
                    true
                }
                None => false,
            }
        } else if key == "model" {
            match json_string(&mut p) {
                Some(m) => {
                    r.model = m;
                    r.model_from_request = true;
                    true
                }
                None => false,
            }
        } else if key == "max_tokens" {
            parse_budget(&mut p, &key, &mut r, &mut err)
        } else if key == "temperature" {
            json_number(&mut p)
                .map(|v| r.temperature = v as f32)
                .is_some()
        } else if key == "top_p" {
            json_number(&mut p).map(|v| r.top_p = v as f32).is_some()
        } else if key == "top_k" {
            json_int(&mut p).map(|v| r.top_k = v).is_some()
        } else if key == "stream" {
            json_bool(&mut p).map(|v| r.stream = v).is_some()
        } else if key == "stop_sequences" {
            match parse_stop(&mut p) {
                Some(s) => {
                    r.stops = s;
                    true
                }
                None => false,
            }
        } else if key == "thinking" {
            match parse_thinking_control_value(&mut p) {
                Some(v) => {
                    if let Some(e) = v {
                        thinking_enabled = e;
                    }
                    got_thinking = true;
                    true
                }
                None => false,
            }
        } else if key == "output_config" {
            match parse_output_config_effort(&mut p) {
                Ok(e) => {
                    if let Some(v) = e {
                        reasoning_effort = v;
                    }
                    true
                }
                Err(e) => {
                    err = e;
                    false
                }
            }
        } else if key == "output_format" {
            parse_output_format_value(&mut p, "output_format")
                .map_err(|e| err = e)
                .is_ok()
        } else if key == "reasoning_effort" {
            match parse_reasoning_effort_value(&mut p) {
                Ok(Some(e)) => {
                    reasoning_effort = e;
                    true
                }
                Ok(None) => true,
                Err(_) => false,
            }
        } else {
            json_skip_value(&mut p)
        };
        if !ok {
            return bad(&mut err, "invalid JSON request");
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.peek() != Some(b'}') {
        return bad(&mut err, "invalid JSON request");
    }
    if !got_messages {
        return Err("missing messages".into());
    }
    if let Some(sys) = system {
        if !sys.is_empty() {
            msgs.push(ChatMsg {
                role: "system".into(),
                content: sys,
                ..Default::default()
            });
        }
    }
    validate_image_references(&msgs, &images)?;
    r.has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if !prepare_tool_choice(&mut r, &mut tool_schemas, &mut orders, &mut err) {
        return Err(err);
    }
    apply_think(&mut r, got_thinking, thinking_enabled, reasoning_effort);
    match anthropic_validate_tool_results(&msgs, &env.live_ids, &mut err) {
        Ok(live) => r.anthropic_requires_live_tool_state = live,
        Err(()) => return Err(err),
    }
    r.live_call_ids = crate::cont::live_tool_result_ids(Api::Anthropic, &msgs);
    r.messages = msgs;
    r.images = images;
    r.tool_schemas = tool_schemas;
    r.tool_orders = orders;
    r.finish_needs();
    Ok(r)
}

pub fn parse_responses_request(env: &ParseEnv, body: &str) -> Result<ParsedRequest, String> {
    let mut r = ParsedRequest::init(ReqKind::Chat, env);
    r.api = Api::Responses;
    let mut err = String::new();
    let mut p = Json::new(body);
    let mut got_input = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = env.default_effort;
    let mut msgs = Vec::new();
    let mut images = Vec::new();
    let mut loaded = String::new();
    let mut orders = Vec::new();
    let mut instructions = None;
    let mut tool_schemas = String::new();

    p.ws();
    if p.bump() != Some(b'{') {
        return bad(&mut err, "invalid JSON request");
    }
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return bad(&mut err, "invalid JSON request");
        };
        p.ws();
        if p.bump() != Some(b':') {
            return bad(&mut err, "invalid JSON request");
        }
        let ok = if key == "input" {
            p.ws();
            if p.peek() == Some(b'"') {
                match json_string(&mut p) {
                    Some(plain) => {
                        msgs = vec![ChatMsg {
                            role: "user".into(),
                            content: plain,
                            ..Default::default()
                        }];
                        got_input = true;
                        true
                    }
                    None => false,
                }
            } else {
                match parse_responses_input(&mut p, &mut loaded, &mut orders, &mut images, &mut err)
                {
                    Some(m) => {
                        msgs = m;
                        got_input = true;
                        true
                    }
                    None => false,
                }
            }
        } else if key == "instructions" {
            p.ws();
            if p.lit("null") {
                instructions = Some(String::new());
                true
            } else {
                match json_string(&mut p) {
                    Some(s) => {
                        instructions = Some(s);
                        true
                    }
                    None => false,
                }
            }
        } else if key == "tools" {
            match parse_tools_value(&mut p) {
                Some((s, o)) => {
                    tool_schemas = s;
                    orders.extend(o);
                    true
                }
                None => false,
            }
        } else if key == "tool_choice" {
            match parse_openai_tool_choice(&mut p, &mut err) {
                Some(c) => {
                    r.tool_choice = c;
                    true
                }
                None => false,
            }
        } else if key == "parallel_tool_calls" {
            parse_parallel_tool_calls(&mut p, &mut err)
        } else if key == "model" {
            match json_string(&mut p) {
                Some(m) => {
                    r.model = m;
                    r.model_from_request = true;
                    true
                }
                None => false,
            }
        } else if key == "max_output_tokens" || key == "max_tokens" {
            parse_budget(&mut p, &key, &mut r, &mut err)
        } else if key == "temperature" {
            json_number(&mut p)
                .map(|v| r.temperature = v as f32)
                .is_some()
        } else if key == "top_p" {
            json_number(&mut p).map(|v| r.top_p = v as f32).is_some()
        } else if key == "stream" {
            json_bool(&mut p).map(|v| r.stream = v).is_some()
        } else if key == "reasoning" {
            match parse_responses_reasoning(
                &mut p,
                &mut reasoning_effort,
                &mut r.reasoning_summary_emit,
            ) {
                Some(seen) => {
                    if seen {
                        got_thinking = true;
                        if reasoning_effort == ThinkMode::None {
                            thinking_enabled = false;
                        }
                    }
                    true
                }
                None => false,
            }
        } else if key == "text" {
            parse_responses_text_value(&mut p)
                .map_err(|e| err = e)
                .is_ok()
        } else if key == "previous_response_id" || key == "conversation" {
            p.ws();
            if p.lit("null") {
                true
            } else {
                err = format!("{key} is not supported; replay full input instead");
                false
            }
        } else {
            json_skip_value(&mut p)
        };
        if !ok {
            return bad(&mut err, "invalid JSON request");
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.peek() != Some(b'}') {
        return bad(&mut err, "invalid JSON request");
    }
    if !got_input {
        return Err("missing input".into());
    }
    if let Some(ins) = instructions {
        if !ins.is_empty() {
            msgs.push(ChatMsg {
                role: "system".into(),
                content: ins,
                ..Default::default()
            });
            if msgs.len() > 1 {
                let last = msgs.pop().unwrap();
                msgs.insert(0, last);
            }
        }
    }
    validate_image_references(&msgs, &images)?;
    let mut combined = tool_schemas.clone();
    if !loaded.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&loaded);
    }
    r.has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if !prepare_tool_choice(&mut r, &mut combined, &mut orders, &mut err) {
        return Err(err);
    }
    apply_think(&mut r, got_thinking, thinking_enabled, reasoning_effort);
    match responses_validate_tool_outputs(&msgs, r.think_mode, &env.live_ids, &mut err) {
        Ok((t, reason)) => {
            r.responses_requires_live_tool_state = t;
            r.responses_requires_live_reasoning = reason;
        }
        Err(()) => return Err(err),
    }
    r.live_call_ids = crate::cont::live_tool_result_ids(Api::Responses, &msgs);
    r.messages = msgs;
    r.images = images;
    r.tool_schemas = combined;
    r.tool_orders = orders;
    r.finish_needs();
    Ok(r)
}

pub fn parse_request(
    surface: WireSurface,
    env: &ParseEnv,
    body: &str,
) -> Result<ParsedRequest, String> {
    match surface {
        WireSurface::OpenaiChat => parse_chat_request(env, body),
        WireSurface::OpenaiCompletion => parse_completion_request(env, body),
        WireSurface::Anthropic => parse_anthropic_request(env, body),
        WireSurface::Responses => parse_responses_request(env, body),
    }
}
