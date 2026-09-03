//! Four-surface stream projectors and buffered finals from `ds4_server.c`
//! at v0.6.5-dfm. Incremental tool projection is host-owned
//! (`tool_stream`); GPU decode still native.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{cors_headers, http_response_bytes, wire_stream_error_bytes};
use crate::json::{json_args_parse, json_escape_bytes};
use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::route::{think_mode_enabled, Api, ReqKind, ThinkMode};
use crate::tool_stream::{DsmlToolStream, ToolSink};
use crate::tools::find_tool_start;

/// Frozen epoch used by the C stream oracle and tape tests.
pub const CREATED_TEST: i64 = 1_767_225_600;

pub const TEST_RESP_ID: &str = "resp_aaaaaaaaaaaaaaaaaaaaaaaa";
pub const TEST_RS_ID: &str = "rs_aaaaaaaaaaaaaaaaaaaaaaaa";
pub const TEST_MSG_ID: &str = "msg_aaaaaaaaaaaaaaaaaaaaaaaa";

pub const TAPE_PLAIN: &[&str] = &["Hel", "lo", " wor", "ld."];
pub const TAPE_THINKING: &[&str] = &["plan", "</think>", "Answer", " done."];
pub const TAPE_UTF8: &[&[u8]] = &[b"caf", b"\xc3", b"\xa9", b" ok"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    DeepSeek,
    SolarOpen2,
    Exaone,
    Qwen4Exp,
}

pub fn think_start(fmt: ChatFormat) -> &'static str {
    if fmt == ChatFormat::SolarOpen2 {
        "<|think:start|>"
    } else {
        "<think>"
    }
}

pub fn think_end(fmt: ChatFormat) -> &'static str {
    if fmt == ChatFormat::SolarOpen2 {
        "<|think:end|>"
    } else {
        "</think>"
    }
}

pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn utf8_expected_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if (0xc2..=0xdf).contains(&c) {
        2
    } else if (0xe0..=0xef).contains(&c) {
        3
    } else if (0xf0..=0xf4).contains(&c) {
        4
    } else {
        1
    }
}

/// Hold a trailing incomplete UTF-8 character. `final_` is documentation
/// only, matching C (`(void)final`).
pub fn utf8_stream_safe_len(s: &[u8], start: usize, limit: usize, _final: bool) -> usize {
    if limit <= start {
        return limit;
    }
    let mut p = limit;
    let mut cont = 0;
    while p > start && cont < 4 && (s[p - 1] & 0xc0) == 0x80 {
        p -= 1;
        cont += 1;
    }
    if p == limit {
        return if utf8_expected_len(s[limit - 1]) > 1 {
            limit - 1
        } else {
            limit
        };
    }
    if p == start && (s[p] & 0xc0) == 0x80 {
        return start;
    }
    let lead = p - 1;
    let need = utf8_expected_len(s[lead]);
    if limit - lead < need {
        lead
    } else {
        limit
    }
}

pub fn utf8_trim_tail(s: &[u8]) -> &[u8] {
    let n = utf8_stream_safe_len(s, 0, s.len(), true);
    &s[..n]
}

pub fn append_json_object_or_empty(out: &mut Vec<u8>, json: &str) {
    let Some(args) = json_args_parse(json) else {
        out.extend_from_slice(b"{}");
        return;
    };
    out.push(b'{');
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend(json_escape_bytes(a.key.as_bytes()));
        out.push(b':');
        if a.is_string {
            out.extend(json_escape_bytes(a.value.as_bytes()));
        } else {
            out.extend_from_slice(a.value.as_bytes());
        }
    }
    out.push(b'}');
}

pub fn append_json_object_string(out: &mut Vec<u8>, json: &str) {
    let mut tmp = Vec::new();
    append_json_object_or_empty(&mut tmp, json);
    out.extend(json_escape_bytes(&tmp));
}

pub fn append_tool_calls_json(out: &mut Vec<u8>, calls: &[ToolCall], id_prefix: &str) {
    out.push(b'[');
    for (i, tc) in calls.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        let fallback = format!("{id_prefix}_tool_{i}");
        let id = if tc.id.is_empty() {
            fallback.as_str()
        } else {
            tc.id.as_str()
        };
        out.extend_from_slice(b"{\"id\":");
        out.extend(json_escape_bytes(id.as_bytes()));
        out.extend_from_slice(b",\"type\":\"function\",\"function\":{\"name\":");
        out.extend(json_escape_bytes(tc.name.as_bytes()));
        out.extend_from_slice(b",\"arguments\":");
        append_json_object_string(out, &tc.arguments);
        out.extend_from_slice(b"}}");
    }
    out.push(b']');
}

pub fn append_tool_call_deltas_json(out: &mut Vec<u8>, calls: &[ToolCall], id_prefix: &str) {
    out.push(b'[');
    for (i, tc) in calls.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        let fallback = format!("{id_prefix}_tool_{i}");
        let id = if tc.id.is_empty() {
            fallback.as_str()
        } else {
            tc.id.as_str()
        };
        out.extend_from_slice(format!("{{\"index\":{i},\"id\":").as_bytes());
        out.extend(json_escape_bytes(id.as_bytes()));
        out.extend_from_slice(b",\"type\":\"function\",\"function\":{\"name\":");
        out.extend(json_escape_bytes(tc.name.as_bytes()));
        out.extend_from_slice(b",\"arguments\":");
        append_json_object_string(out, &tc.arguments);
        out.extend_from_slice(b"}}");
    }
    out.push(b']');
}

fn tool_schema_orders_find<'a>(
    orders: &'a [ToolSchemaOrder],
    name: &str,
) -> Option<&'a ToolSchemaOrder> {
    orders.iter().find(|o| o.name == name)
}

fn responses_tool_call_is_tool_search(tc: &ToolCall, order: Option<&ToolSchemaOrder>) -> bool {
    tc.name == "tool_search" && order.map(|o| o.responses_tool_search).unwrap_or(true)
}

fn responses_tool_wire_name<'a>(tc: &'a ToolCall, order: Option<&'a ToolSchemaOrder>) -> &'a str {
    order
        .and_then(|o| (!o.wire_name.is_empty()).then_some(o.wire_name.as_str()))
        .unwrap_or(tc.name.as_str())
}

pub fn append_anthropic_tool_use(out: &mut Vec<u8>, tc: &ToolCall, id_prefix: &str, i: usize) {
    let fallback = format!("toolu_{id_prefix}_{i}");
    let id = if tc.id.is_empty() {
        fallback.as_str()
    } else {
        tc.id.as_str()
    };
    out.extend_from_slice(b"{\"type\":\"tool_use\",\"id\":");
    out.extend(json_escape_bytes(id.as_bytes()));
    out.extend_from_slice(b",\"name\":");
    out.extend(json_escape_bytes(tc.name.as_bytes()));
    out.extend_from_slice(b",\"input\":");
    append_json_object_or_empty(out, &tc.arguments);
    out.push(b'}');
}

fn append_responses_function_call_item(
    out: &mut Vec<u8>,
    tc: &ToolCall,
    fc_id: &str,
    call_id: &str,
    item_status: &str,
    orders: &[ToolSchemaOrder],
) {
    let order = tool_schema_orders_find(orders, &tc.name);
    if responses_tool_call_is_tool_search(tc, order) {
        out.extend_from_slice(
            format!(
                "{{\"id\":\"{fc_id}\",\"type\":\"tool_search_call\",\"status\":\"{item_status}\",\"call_id\":"
            )
            .as_bytes(),
        );
        out.extend(json_escape_bytes(call_id.as_bytes()));
        out.extend_from_slice(b",\"execution\":\"client\",\"arguments\":");
        append_json_object_or_empty(out, &tc.arguments);
        out.push(b'}');
        return;
    }
    let name = responses_tool_wire_name(tc, order);
    out.extend_from_slice(
        format!(
            "{{\"id\":\"{fc_id}\",\"type\":\"function_call\",\"status\":\"{item_status}\",\"name\":"
        )
        .as_bytes(),
    );
    out.extend(json_escape_bytes(name.as_bytes()));
    if let Some(ns) = order
        .map(|o| o.namespace.as_str())
        .filter(|s| !s.is_empty())
    {
        out.extend_from_slice(b",\"namespace\":");
        out.extend(json_escape_bytes(ns.as_bytes()));
    }
    out.extend_from_slice(b",\"call_id\":");
    out.extend(json_escape_bytes(call_id.as_bytes()));
    out.extend_from_slice(b",\"arguments\":");
    append_json_object_string(out, &tc.arguments);
    out.push(b'}');
}

fn trim_tool_separator_ws(raw: &[u8], start: usize, mut limit: usize) -> usize {
    while limit > start && raw[limit - 1].is_ascii_whitespace() {
        limit -= 1;
    }
    limit
}

pub fn text_stream_safe_limit(
    raw: &[u8],
    start: usize,
    raw_len: usize,
    has_tools: bool,
    final_: bool,
    format: ChatFormat,
) -> usize {
    if raw_len <= start {
        return raw_len;
    }
    let mut limit = raw_len;
    if has_tools {
        if let Some(rel) = find_tool_start(&raw[start..], format) {
            limit = trim_tool_separator_ws(raw, start, start + rel);
            return utf8_stream_safe_len(raw, start, limit, true);
        }
        if !final_ {
            while limit > start && raw[limit - 1].is_ascii_whitespace() {
                limit -= 1;
            }
            let max_marker = 80;
            let scan = if raw_len - start > max_marker {
                raw_len - max_marker
            } else {
                start
            };
            for i in (scan + 1..=raw_len).rev() {
                if raw[i - 1] == b'<' {
                    if i - 1 < limit {
                        limit = i - 1;
                    }
                    break;
                }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    utf8_stream_safe_len(raw, start, limit, final_)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReqTimings {
    pub valid: bool,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prefill_tokens: i32,
    pub prefill_cached: i32,
    pub decode_tokens: i32,
    pub decode_steps: i32,
}

impl ReqTimings {
    pub fn json_suffix(self) -> String {
        if !self.valid {
            return String::new();
        }
        let mut b = format!(
            ",\"timings\":{{\"ttft_ms\":{:.1},\"prefill_tokens\":{},\"prefill_cached_tokens\":{}",
            self.ttft_ms, self.prefill_tokens, self.prefill_cached
        );
        if self.prefill_tokens > 0 && self.prefill_ms > 0.0 {
            b.push_str(&format!(
                ",\"prefill_tok_s\":{:.1}",
                self.prefill_tokens as f64 / (self.prefill_ms / 1e3)
            ));
        }
        if self.decode_tokens > 1 && self.decode_ms > 0.0 {
            b.push_str(&format!(
                ",\"decode_tok_s\":{:.1}",
                (self.decode_tokens - 1) as f64 / (self.decode_ms / 1e3)
            ));
        }
        if self.decode_steps > 0 {
            b.push_str(&format!(
                ",\"tok_per_step\":{:.2}",
                self.decode_tokens as f64 / self.decode_steps as f64
            ));
        }
        b.push('}');
        b
    }
}

#[derive(Debug, Clone)]
pub struct StreamReq {
    pub kind: ReqKind,
    pub api: Api,
    pub model: String,
    pub think_mode: ThinkMode,
    pub has_tools: bool,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub reasoning_summary_emit: bool,
    pub chat_format: ChatFormat,
    pub cache_read_tokens: i32,
    pub cache_write_tokens: i32,
    pub timings: ReqTimings,
    pub tool_orders: Vec<ToolSchemaOrder>,
}

impl Default for StreamReq {
    fn default() -> Self {
        Self {
            kind: ReqKind::Chat,
            api: Api::Openai,
            model: "deepseek-v4-flash".into(),
            think_mode: ThinkMode::Low,
            has_tools: false,
            stream: true,
            stream_include_usage: false,
            reasoning_summary_emit: false,
            chat_format: ChatFormat::DeepSeek,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            timings: ReqTimings::default(),
            tool_orders: Vec::new(),
        }
    }
}

pub struct Writer {
    pub created: i64,
    pub out: Vec<u8>,
}

impl Writer {
    pub fn new(created: i64) -> Self {
        Self {
            created,
            out: Vec::new(),
        }
    }

    fn put(&mut self, s: &[u8]) {
        self.out.extend_from_slice(s);
    }

    fn put_str(&mut self, s: &str) {
        self.put(s.as_bytes());
    }

    fn esc(&mut self, s: &[u8]) {
        self.out.extend(json_escape_bytes(s));
    }

    fn esc_str(&mut self, s: &str) {
        self.esc(s.as_bytes());
    }
}

fn clamp_usage(value: i32, max: i32) -> i32 {
    if value < 0 {
        0
    } else if max >= 0 && value > max {
        max
    } else {
        value
    }
}

fn openai_usage(r: &StreamReq, prompt: i32, completion: i32) -> String {
    let cached = clamp_usage(r.cache_read_tokens, prompt);
    let write = clamp_usage(r.cache_write_tokens, prompt - cached);
    format!(
        "{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{cached},\"cache_write_tokens\":{write}}}}}",
        prompt + completion
    )
}

fn anthropic_usage(r: &StreamReq, prompt: i32, completion: i32) -> String {
    let cache_read = clamp_usage(r.cache_read_tokens, prompt);
    let cache_write = clamp_usage(r.cache_write_tokens, prompt - cache_read);
    let mut input = prompt - cache_read - cache_write;
    if input < 0 {
        input = 0;
    }
    format!(
        "{{\"input_tokens\":{input},\"output_tokens\":{completion},\"cache_read_input_tokens\":{cache_read},\"cache_creation_input_tokens\":{cache_write}}}"
    )
}

fn responses_usage(r: &StreamReq, input: i32, output: i32, reasoning: i32) -> String {
    let cached = clamp_usage(r.cache_read_tokens, input);
    let write = clamp_usage(r.cache_write_tokens, input - cached);
    let reasoning = clamp_usage(reasoning, output);
    format!(
        "{{\"input_tokens\":{input},\"input_tokens_details\":{{\"cached_tokens\":{cached},\"cache_write_tokens\":{write}}},\"output_tokens\":{output},\"output_tokens_details\":{{\"reasoning_tokens\":{reasoning}}},\"total_tokens\":{}}}",
        input + output
    )
}

pub fn sse_headers(cors: bool) -> Vec<u8> {
    let mut h =
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
            .to_vec();
    if cors {
        h.extend_from_slice(cors_headers().as_bytes());
    }
    h.extend_from_slice(b"Connection: close\r\n\r\n");
    h
}

pub fn sse_chunk(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    text: Option<&[u8]>,
    finish: Option<&str>,
) {
    if r.kind == ReqKind::Chat {
        w.put_str(&format!(
            "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
            w.created
        ));
        w.esc_str(&r.model);
        w.put_str(",\"choices\":[{\"index\":0,\"delta\":");
        if let Some(t) = text {
            w.put_str("{\"content\":");
            w.esc(t);
            w.put_str("}");
        } else if finish.is_some() {
            w.put_str("{}");
        } else {
            w.put_str("{\"role\":\"assistant\"}");
        }
        w.put_str(",\"finish_reason\":");
        if let Some(f) = finish {
            w.esc_str(f);
        } else {
            w.put_str("null");
        }
        w.put_str("}]}\n\n");
    } else {
        w.put_str(&format!(
            "data: {{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{},\"model\":",
            w.created
        ));
        w.esc_str(&r.model);
        w.put_str(",\"choices\":[{\"text\":");
        w.esc(text.unwrap_or(b""));
        w.put_str(",\"index\":0,\"finish_reason\":");
        if let Some(f) = finish {
            w.esc_str(f);
        } else {
            w.put_str("null");
        }
        w.put_str("}]}\n\n");
    }
}

fn sse_usage_chunk(w: &mut Writer, r: &StreamReq, id: &str, prompt: i32, completion: i32) {
    if !r.stream_include_usage {
        return;
    }
    let obj = if r.kind == ReqKind::Chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    w.put_str(&format!(
        "data: {{\"id\":\"{id}\",\"object\":\"{obj}\",\"created\":{},\"model\":",
        w.created
    ));
    w.esc_str(&r.model);
    w.put_str(",\"choices\":[],\"usage\":");
    w.put_str(&openai_usage(r, prompt, completion));
    w.put_str("}\n\n");
}

pub fn sse_done(w: &mut Writer, r: &StreamReq, id: &str, prompt: i32, completion: i32) {
    sse_usage_chunk(w, r, id, prompt, completion);
    w.put_str("data: [DONE]\n\n");
}

fn sse_event(w: &mut Writer, event: &str, data: &[u8]) {
    w.put_str("event: ");
    w.put_str(event);
    w.put_str("\ndata: ");
    w.put(data);
    w.put_str("\n\n");
}

fn sse_chat_delta_n(w: &mut Writer, r: &StreamReq, id: &str, field: &str, text: &[u8]) {
    if text.is_empty() {
        return;
    }
    w.put_str(&format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
        w.created
    ));
    w.esc_str(&r.model);
    w.put_str(",\"choices\":[{\"index\":0,\"delta\":{");
    w.esc_str(field);
    w.put_str(":");
    w.esc(text);
    w.put_str("},\"finish_reason\":null}]}\n\n");
}

struct OpenaiToolSink<'a> {
    w: &'a mut Writer,
    r: &'a StreamReq,
    job: &'a str,
}

impl ToolSink for OpenaiToolSink<'_> {
    fn start_invoke(&mut self, index: i32, id: &str, name: &[u8]) -> bool {
        self.w.put_str(&format!(
            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
            self.job, self.w.created
        ));
        self.w.esc_str(&self.r.model);
        self.w.put_str(&format!(
            ",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"id\":"
        ));
        self.w.esc_str(id);
        self.w
            .put_str(",\"type\":\"function\",\"function\":{\"name\":");
        self.w.esc(name);
        self.w
            .put_str(",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
        true
    }

    fn args_fragment(&mut self, index: i32, text: &[u8]) -> bool {
        if text.is_empty() {
            return true;
        }
        self.w.put_str(&format!(
            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
            self.job, self.w.created
        ));
        self.w.esc_str(&self.r.model);
        self.w.put_str(&format!(
            ",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"function\":{{\"arguments\":"
        ));
        self.w.esc(text);
        self.w.put_str("}}]},\"finish_reason\":null}]}\n\n");
        true
    }

    fn close_invoke(&mut self, _index: i32) -> bool {
        true
    }
}

struct AnthropicToolSink<'a> {
    w: &'a mut Writer,
    next_index: &'a mut i32,
    open_block: &'a mut AnthBlock,
    job: &'a str,
}

impl ToolSink for AnthropicToolSink<'_> {
    fn start_invoke(&mut self, _index: i32, id: &str, name: &[u8]) -> bool {
        if *self.open_block == AnthBlock::Tool {
            return true;
        }
        if *self.open_block != AnthBlock::None {
            return false;
        }
        let mut start = format!(
            "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"tool_use\",\"id\":",
            *self.next_index
        )
        .into_bytes();
        start.extend(json_escape_bytes(id.as_bytes()));
        start.extend_from_slice(b",\"name\":");
        start.extend(json_escape_bytes(name));
        start.extend_from_slice(b",\"input\":{}}}");
        sse_event(self.w, "content_block_start", &start);
        *self.open_block = AnthBlock::Tool;
        true
    }

    fn args_fragment(&mut self, _index: i32, text: &[u8]) -> bool {
        if text.is_empty() {
            return true;
        }
        let mut b = format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":",
            *self.next_index
        )
        .into_bytes();
        b.extend(json_escape_bytes(text));
        b.extend_from_slice(b"}}");
        sse_event(self.w, "content_block_delta", &b);
        true
    }

    fn close_invoke(&mut self, _index: i32) -> bool {
        if *self.open_block == AnthBlock::None {
            return true;
        }
        if *self.open_block == AnthBlock::Thinking {
            let mut b = format!(
                "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"signature_delta\",\"signature\":",
                *self.next_index
            )
            .into_bytes();
            b.extend(json_escape_bytes(self.job.as_bytes()));
            b.extend_from_slice(b"}}");
            sse_event(self.w, "content_block_delta", &b);
        }
        let stop = format!(
            "{{\"type\":\"content_block_stop\",\"index\":{}}}",
            *self.next_index
        );
        sse_event(self.w, "content_block_stop", stop.as_bytes());
        *self.open_block = AnthBlock::None;
        *self.next_index += 1;
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenaiMode {
    Thinking,
    Text,
    Tool,
    Suppress,
}

#[derive(Debug)]
pub struct OpenaiStream {
    pub mode: OpenaiMode,
    pub emit_pos: usize,
    pub active: bool,
    pub checked_think_prefix: bool,
    pub tool: DsmlToolStream,
}

pub fn openai_stream_start(r: &StreamReq) -> OpenaiStream {
    OpenaiStream {
        mode: if think_mode_enabled(r.think_mode) {
            OpenaiMode::Thinking
        } else {
            OpenaiMode::Text
        },
        emit_pos: 0,
        active: true,
        checked_think_prefix: false,
        tool: DsmlToolStream::with_prefix("call_"),
    }
}

pub fn openai_sse_stream_update(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    st: &mut OpenaiStream,
    raw: &[u8],
    final_: bool,
) -> bool {
    if !st.active {
        return true;
    }
    if st.mode == OpenaiMode::Thinking {
        if !st.checked_think_prefix {
            let open = think_start(r.chat_format).as_bytes();
            if raw.len() < open.len() && raw == &open[..raw.len()] && !final_ {
                return true;
            }
            if raw.len() >= open.len() && raw.starts_with(open) {
                st.emit_pos = open.len();
            }
            st.checked_think_prefix = true;
        }
        let close_s = think_end(r.chat_format);
        let close = find_substr(raw, st.emit_pos, close_s.as_bytes());
        let limit = if let Some(c) = close {
            c
        } else if final_ {
            utf8_stream_safe_len(raw, st.emit_pos, raw.len(), true)
        } else {
            let hold = close_s.len() - 1;
            let lim = if raw.len() > hold {
                raw.len() - hold
            } else {
                st.emit_pos
            };
            utf8_stream_safe_len(raw, st.emit_pos, lim, false)
        };
        if limit > st.emit_pos {
            sse_chat_delta_n(w, r, id, "reasoning_content", &raw[st.emit_pos..limit]);
            st.emit_pos = limit;
        }
        if let Some(c) = close {
            st.emit_pos = c + close_s.len();
            st.mode = OpenaiMode::Text;
        } else if final_ {
            st.mode = OpenaiMode::Suppress;
            return true;
        } else {
            return true;
        }
    }
    if st.mode == OpenaiMode::Text {
        let tool = if r.has_tools {
            find_tool_start(&raw[st.emit_pos..], r.chat_format).map(|rel| st.emit_pos + rel)
        } else {
            None
        };
        let limit = text_stream_safe_limit(
            raw,
            st.emit_pos,
            raw.len(),
            r.has_tools,
            final_,
            r.chat_format,
        );
        if limit > st.emit_pos {
            sse_chat_delta_n(w, r, id, "content", &raw[st.emit_pos..limit]);
            st.emit_pos = limit;
        }
        if let Some(pos) = tool {
            st.emit_pos = pos;
            if st.tool.init(raw, st.emit_pos) {
                st.mode = OpenaiMode::Tool;
            } else {
                st.mode = OpenaiMode::Suppress;
            }
        } else if final_ {
            st.mode = OpenaiMode::Suppress;
        }
    }
    if st.mode == OpenaiMode::Tool {
        let mut sink = OpenaiToolSink { w, r, job: id };
        if !st.tool.update(&mut sink, raw) {
            return false;
        }
        if !st.tool.active {
            st.mode = OpenaiMode::Suppress;
        }
    }
    true
}

pub fn openai_sse_finish_live(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    st: &mut OpenaiStream,
    raw: &[u8],
    finish: &str,
    prompt: i32,
    completion: i32,
    calls: &[ToolCall],
) -> bool {
    if !openai_sse_stream_update(w, r, id, st, raw, true) {
        return false;
    }
    if !calls.is_empty() && !st.tool.emitted_any {
        w.put_str(&format!(
            "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
            w.created
        ));
        w.esc_str(&r.model);
        w.put_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        let mut deltas = Vec::new();
        append_tool_call_deltas_json(&mut deltas, calls, id);
        w.put(&deltas);
        w.put_str("},\"finish_reason\":null}]}\n\n");
    }
    w.put_str(&format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":",
        w.created
    ));
    w.esc_str(&r.model);
    w.put_str(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    w.esc_str(finish);
    w.put_str("}]}\n\n");
    sse_done(w, r, id, prompt, completion);
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnthMode {
    Thinking,
    Text,
    Tool,
    Suppress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnthBlock {
    None,
    Thinking,
    Text,
    Tool,
}

#[derive(Debug)]
pub struct AnthropicStream {
    mode: AnthMode,
    open_block: AnthBlock,
    next_index: i32,
    emit_pos: usize,
    active: bool,
    checked_think_prefix: bool,
    sent_thinking: bool,
    sent_text: bool,
    pub tool: DsmlToolStream,
}

fn anthropic_sse_open_block(w: &mut Writer, st: &mut AnthropicStream, ty: AnthBlock) -> bool {
    if st.open_block == ty {
        return true;
    }
    if st.open_block != AnthBlock::None {
        return false;
    }
    let body = if ty == AnthBlock::Thinking {
        format!(
            "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}}}",
            st.next_index
        )
    } else {
        format!(
            "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}",
            st.next_index
        )
    };
    sse_event(w, "content_block_start", body.as_bytes());
    st.open_block = ty;
    true
}

fn anthropic_sse_delta_live(w: &mut Writer, st: &AnthropicStream, ty: AnthBlock, text: &[u8]) {
    if text.is_empty() {
        return;
    }
    let mut b = if ty == AnthBlock::Thinking {
        format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":",
            st.next_index
        )
        .into_bytes()
    } else {
        format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":",
            st.next_index
        )
        .into_bytes()
    };
    b.extend(json_escape_bytes(text));
    b.extend_from_slice(b"}}");
    sse_event(w, "content_block_delta", &b);
}

fn anthropic_sse_close_block_live(w: &mut Writer, id: &str, st: &mut AnthropicStream) -> bool {
    if st.open_block == AnthBlock::None {
        return true;
    }
    if st.open_block == AnthBlock::Thinking {
        let mut b = format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"signature_delta\",\"signature\":",
            st.next_index
        )
        .into_bytes();
        b.extend(json_escape_bytes(id.as_bytes()));
        b.extend_from_slice(b"}}");
        sse_event(w, "content_block_delta", &b);
    }
    let stop = format!(
        "{{\"type\":\"content_block_stop\",\"index\":{}}}",
        st.next_index
    );
    sse_event(w, "content_block_stop", stop.as_bytes());
    st.open_block = AnthBlock::None;
    st.next_index += 1;
    true
}

pub fn anthropic_sse_start_live(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    prompt_tokens: i32,
) -> AnthropicStream {
    let mut msg = format!(
        "{{\"type\":\"message_start\",\"message\":{{\"id\":\"{id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":"
    )
    .into_bytes();
    msg.extend(json_escape_bytes(r.model.as_bytes()));
    msg.extend_from_slice(
        b",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":",
    );
    msg.extend_from_slice(anthropic_usage(r, prompt_tokens, 0).as_bytes());
    msg.extend_from_slice(b"}}");
    sse_event(w, "message_start", &msg);
    AnthropicStream {
        mode: if think_mode_enabled(r.think_mode) {
            AnthMode::Thinking
        } else {
            AnthMode::Text
        },
        open_block: AnthBlock::None,
        next_index: 0,
        emit_pos: 0,
        active: true,
        checked_think_prefix: false,
        sent_thinking: false,
        sent_text: false,
        tool: DsmlToolStream::with_prefix("toolu_"),
    }
}

pub fn anthropic_sse_stream_update(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    st: &mut AnthropicStream,
    raw: &[u8],
    final_: bool,
) -> bool {
    if !st.active {
        return true;
    }
    if st.mode == AnthMode::Thinking {
        if !st.checked_think_prefix {
            let open = think_start(r.chat_format).as_bytes();
            if raw.len() < open.len() && raw == &open[..raw.len()] && !final_ {
                return true;
            }
            if raw.len() >= open.len() && raw.starts_with(open) {
                st.emit_pos = open.len();
            }
            st.checked_think_prefix = true;
        }
        let close_s = think_end(r.chat_format);
        let close = find_substr(raw, st.emit_pos, close_s.as_bytes());
        let limit = if let Some(c) = close {
            c
        } else if final_ {
            utf8_stream_safe_len(raw, st.emit_pos, raw.len(), true)
        } else {
            let hold = close_s.len() - 1;
            let lim = if raw.len() > hold {
                raw.len() - hold
            } else {
                st.emit_pos
            };
            utf8_stream_safe_len(raw, st.emit_pos, lim, false)
        };
        if limit > st.emit_pos {
            if !anthropic_sse_open_block(w, st, AnthBlock::Thinking) {
                return false;
            }
            anthropic_sse_delta_live(w, st, AnthBlock::Thinking, &raw[st.emit_pos..limit]);
            st.sent_thinking = true;
            st.emit_pos = limit;
        }
        if close.is_some() || final_ {
            if !anthropic_sse_close_block_live(w, id, st) {
                return false;
            }
            if let Some(c) = close {
                st.emit_pos = c + close_s.len();
                st.mode = AnthMode::Text;
            } else {
                st.mode = AnthMode::Suppress;
                return true;
            }
        } else {
            return true;
        }
    }
    if st.mode == AnthMode::Text {
        let tool = if r.has_tools {
            find_tool_start(&raw[st.emit_pos..], r.chat_format).map(|rel| st.emit_pos + rel)
        } else {
            None
        };
        let limit = text_stream_safe_limit(
            raw,
            st.emit_pos,
            raw.len(),
            r.has_tools,
            final_,
            r.chat_format,
        );
        if limit > st.emit_pos {
            if !anthropic_sse_open_block(w, st, AnthBlock::Text) {
                return false;
            }
            anthropic_sse_delta_live(w, st, AnthBlock::Text, &raw[st.emit_pos..limit]);
            st.sent_text = true;
            st.emit_pos = limit;
        }
        if let Some(pos) = tool {
            if !anthropic_sse_close_block_live(w, id, st) {
                return false;
            }
            st.emit_pos = pos;
            if !final_ && st.tool.init(raw, st.emit_pos) {
                st.mode = AnthMode::Tool;
            } else {
                st.mode = AnthMode::Suppress;
            }
        } else if final_ {
            if !anthropic_sse_close_block_live(w, id, st) {
                return false;
            }
            st.mode = AnthMode::Suppress;
        }
    }
    if st.mode == AnthMode::Tool {
        let mut sink = AnthropicToolSink {
            w,
            next_index: &mut st.next_index,
            open_block: &mut st.open_block,
            job: id,
        };
        if !st.tool.update(&mut sink, raw) {
            return false;
        }
        if !st.tool.active {
            st.mode = AnthMode::Suppress;
        }
    }
    true
}

fn anthropic_stop_reason(finish: &str, matched_stop: Option<&str>) -> &'static str {
    if finish == "tool_calls" {
        "tool_use"
    } else if finish == "length" {
        "max_tokens"
    } else if matched_stop.map(|s| !s.is_empty()).unwrap_or(false) && finish == "stop" {
        "stop_sequence"
    } else {
        "end_turn"
    }
}

fn append_anthropic_stop_fields(out: &mut Vec<u8>, finish: &str, matched_stop: Option<&str>) {
    let reason = anthropic_stop_reason(finish, matched_stop);
    out.extend_from_slice(b"\"stop_reason\":");
    out.extend(json_escape_bytes(reason.as_bytes()));
    out.extend_from_slice(b",\"stop_sequence\":");
    if reason == "stop_sequence" {
        out.extend(json_escape_bytes(matched_stop.unwrap_or("").as_bytes()));
    } else {
        out.extend_from_slice(b"null");
    }
}

fn anthropic_sse_stop_live(
    w: &mut Writer,
    finish: &str,
    matched_stop: Option<&str>,
    completion: i32,
) {
    let mut b = b"{\"type\":\"message_delta\",\"delta\":{".to_vec();
    append_anthropic_stop_fields(&mut b, finish, matched_stop);
    b.extend_from_slice(format!("}},\"usage\":{{\"output_tokens\":{completion}}}}}").as_bytes());
    sse_event(w, "message_delta", &b);
    sse_event(w, "message_stop", b"{\"type\":\"message_stop\"}");
}

pub fn anthropic_sse_finish_live(
    w: &mut Writer,
    r: &StreamReq,
    id: &str,
    st: &mut AnthropicStream,
    raw: &[u8],
    finish: &str,
    matched_stop: Option<&str>,
    completion: i32,
    calls: &[ToolCall],
) -> bool {
    if !anthropic_sse_stream_update(w, r, id, st, raw, true) {
        return false;
    }
    if st.sent_thinking && !st.sent_text && calls.is_empty() {
        if !anthropic_sse_open_block(w, st, AnthBlock::Text) {
            return false;
        }
        if !anthropic_sse_close_block_live(w, id, st) {
            return false;
        }
    }
    anthropic_sse_tool_blocks_live(w, id, st, calls);
    anthropic_sse_stop_live(w, finish, matched_stop, completion);
    true
}

fn anthropic_sse_tool_blocks_live(
    w: &mut Writer,
    id: &str,
    st: &mut AnthropicStream,
    calls: &[ToolCall],
) {
    let already = if st.tool.emitted_any {
        st.tool.index.max(0) as usize
    } else {
        0
    };
    for (i, tc) in calls.iter().enumerate().skip(already) {
        let fallback = format!("toolu_{id}_{i}");
        let tool_id = if tc.id.is_empty() {
            fallback.as_str()
        } else {
            tc.id.as_str()
        };
        let mut start = format!(
            "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"tool_use\",\"id\":",
            st.next_index
        )
        .into_bytes();
        start.extend(json_escape_bytes(tool_id.as_bytes()));
        start.extend_from_slice(b",\"name\":");
        start.extend(json_escape_bytes(tc.name.as_bytes()));
        start.extend_from_slice(b",\"input\":{}}}");
        sse_event(w, "content_block_start", &start);

        let mut delta = format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":",
            st.next_index
        )
        .into_bytes();
        append_json_object_string(&mut delta, &tc.arguments);
        delta.extend_from_slice(b"}}");
        sse_event(w, "content_block_delta", &delta);

        let stop = format!(
            "{{\"type\":\"content_block_stop\",\"index\":{}}}",
            st.next_index
        );
        sse_event(w, "content_block_stop", stop.as_bytes());
        st.next_index += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RespMode {
    Thinking,
    Text,
    Suppress,
}

#[derive(Debug)]
pub struct ResponsesStream {
    mode: RespMode,
    emit_pos: usize,
    active: bool,
    checked_think_prefix: bool,
    reasoning_item_opened: bool,
    reasoning_item_closed: bool,
    reasoning_summary_started: bool,
    reasoning_closed_naturally: bool,
    message_item_opened: bool,
    message_text_part_open: bool,
    message_item_closed: bool,
    reasoning_emitted_any: bool,
    message_emitted_any: bool,
    reasoning_start: usize,
    reasoning_end: usize,
    message_start: usize,
    message_end: usize,
    message_tail_start: usize,
    message_tail_end: usize,
    response_id: String,
    reasoning_id: String,
    message_id: String,
    reasoning_index: i32,
    message_index: i32,
    next_output_index: i32,
    sequence: i32,
    created_at: i64,
}

pub fn responses_stream_init(
    r: &StreamReq,
    response_id: &str,
    reasoning_id: &str,
    message_id: &str,
) -> ResponsesStream {
    ResponsesStream {
        mode: if think_mode_enabled(r.think_mode) {
            RespMode::Thinking
        } else {
            RespMode::Text
        },
        emit_pos: 0,
        active: false,
        checked_think_prefix: false,
        reasoning_item_opened: false,
        reasoning_item_closed: false,
        reasoning_summary_started: false,
        reasoning_closed_naturally: false,
        message_item_opened: false,
        message_text_part_open: false,
        message_item_closed: false,
        reasoning_emitted_any: false,
        message_emitted_any: false,
        reasoning_start: 0,
        reasoning_end: 0,
        message_start: 0,
        message_end: 0,
        message_tail_start: 0,
        message_tail_end: 0,
        response_id: response_id.into(),
        reasoning_id: reasoning_id.into(),
        message_id: message_id.into(),
        reasoning_index: -1,
        message_index: -1,
        next_output_index: 0,
        sequence: 0,
        created_at: 0,
    }
}

fn responses_sse_emit_event(w: &mut Writer, st: &mut ResponsesStream, body: &[u8]) {
    w.put_str("data: ");
    if let Some(end) = find_type_close(body) {
        w.put(&body[..end]);
        w.put_str(&format!(",\"sequence_number\":{}", st.sequence));
        st.sequence += 1;
        w.put(&body[end..]);
    } else {
        w.put(body);
    }
    w.put_str("\n\n");
}

pub(crate) const STREAM_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

fn responses_sse_in_progress(w: &mut Writer, r: &StreamReq, st: &mut ResponsesStream) {
    let mut b = format!(
        "{{\"type\":\"response.in_progress\",\"response\":{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{},\"status\":\"in_progress\",\"model\":",
        st.response_id, st.created_at
    )
    .into_bytes();
    b.extend(json_escape_bytes(r.model.as_bytes()));
    b.extend_from_slice(b",\"output\":[]}}");
    responses_sse_emit_event(w, st, &b);
}

pub(crate) fn stream_heartbeat_if_due(
    w: &mut Writer,
    r: &StreamReq,
    responses: Option<&mut ResponsesStream>,
    last: &mut Instant,
    now: Instant,
    comment: &str,
) -> bool {
    if !r.stream
        || now
            .checked_duration_since(*last)
            .is_none_or(|elapsed| elapsed < STREAM_HEARTBEAT_INTERVAL)
    {
        return false;
    }
    match r.api {
        Api::Anthropic => w.put_str("event: ping\ndata: {\"type\": \"ping\"}\n\n"),
        Api::Responses => match responses {
            Some(st) if st.created_at != 0 => responses_sse_in_progress(w, r, st),
            _ => w.put_str(comment),
        },
        Api::Openai => w.put_str(comment),
    }
    *last = now;
    true
}

fn find_type_close(body: &[u8]) -> Option<usize> {
    if body.first() != Some(&b'{') {
        return None;
    }
    let p = &body[1..];
    if !p.starts_with(b"\"type\":\"") {
        return None;
    }
    let mut q = 1 + 8;
    while q < body.len() {
        if body[q] == b'\\' && q + 1 < body.len() {
            q += 2;
            continue;
        }
        if body[q] == b'"' {
            return Some(q + 1);
        }
        q += 1;
    }
    None
}

pub fn responses_sse_created(
    w: &mut Writer,
    r: &StreamReq,
    st: &mut ResponsesStream,
    created_at: i64,
) {
    st.active = true;
    let mut b = format!(
        "{{\"type\":\"response.created\",\"response\":{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{created_at},\"status\":\"in_progress\",\"model\":",
        st.response_id
    )
    .into_bytes();
    b.extend(json_escape_bytes(r.model.as_bytes()));
    b.extend_from_slice(b",\"output\":[]}}");
    responses_sse_emit_event(w, st, &b);
    st.created_at = created_at;
}

pub fn stream_error(
    w: &mut Writer,
    r: &StreamReq,
    responses: Option<&mut ResponsesStream>,
    msg: &str,
) {
    let sequence = if r.api == Api::Responses {
        responses.map_or(0, |st| {
            let sequence = st.sequence;
            st.sequence += 1;
            sequence
        })
    } else {
        0
    };
    let surface = match r.api {
        Api::Anthropic => crate::route::WireSurface::Anthropic,
        Api::Responses => crate::route::WireSurface::Responses,
        Api::Openai if r.kind == ReqKind::Completion => crate::route::WireSurface::OpenaiCompletion,
        Api::Openai => crate::route::WireSurface::OpenaiChat,
    };
    w.put(&wire_stream_error_bytes(surface, msg, sequence));
}

fn responses_sse_reasoning_added(w: &mut Writer, st: &mut ResponsesStream) {
    let b = format!(
        "{{\"type\":\"response.output_item.added\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}}}}",
        st.reasoning_index, st.reasoning_id
    );
    responses_sse_emit_event(w, st, b.as_bytes());
}

fn responses_sse_reasoning_summary_part_added(w: &mut Writer, st: &mut ResponsesStream) {
    let b = format!(
        "{{\"type\":\"response.reasoning_summary_part.added\",\"item_id\":\"{}\",\"output_index\":{},\"summary_index\":0,\"part\":{{\"type\":\"summary_text\",\"text\":\"\"}}}}",
        st.reasoning_id, st.reasoning_index
    );
    responses_sse_emit_event(w, st, b.as_bytes());
}

fn responses_sse_reasoning_delta(w: &mut Writer, st: &mut ResponsesStream, text: &[u8]) {
    if text.is_empty() {
        return;
    }
    let mut b = format!(
        "{{\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"{}\",\"output_index\":{},\"summary_index\":0,\"delta\":",
        st.reasoning_id, st.reasoning_index
    )
    .into_bytes();
    b.extend(json_escape_bytes(text));
    b.push(b'}');
    responses_sse_emit_event(w, st, &b);
}

fn responses_item_status_for_finish(finish: &str) -> &'static str {
    if finish == "length" || finish == "error" {
        "incomplete"
    } else {
        "completed"
    }
}

fn responses_status_for_finish(finish: &str) -> &'static str {
    if finish == "length" {
        "incomplete"
    } else if finish == "error" {
        "failed"
    } else {
        "completed"
    }
}

fn responses_sse_reasoning_done(
    w: &mut Writer,
    st: &mut ResponsesStream,
    raw: &[u8],
    _finish: &str,
) -> bool {
    let item_status = if st.reasoning_closed_naturally {
        "completed"
    } else {
        "incomplete"
    };
    let rtext = if st.reasoning_end > st.reasoning_start {
        &raw[st.reasoning_start..st.reasoning_end]
    } else {
        b""
    };
    let mut b = format!(
        "{{\"type\":\"response.reasoning_summary_text.done\",\"item_id\":\"{}\",\"output_index\":{},\"summary_index\":0,\"text\":",
        st.reasoning_id, st.reasoning_index
    )
    .into_bytes();
    b.extend(json_escape_bytes(rtext));
    b.push(b'}');
    responses_sse_emit_event(w, st, &b);
    if st.reasoning_summary_started {
        let mut b = format!(
            "{{\"type\":\"response.reasoning_summary_part.done\",\"item_id\":\"{}\",\"output_index\":{},\"summary_index\":0,\"part\":{{\"type\":\"summary_text\",\"text\":",
            st.reasoning_id, st.reasoning_index
        )
        .into_bytes();
        b.extend(json_escape_bytes(rtext));
        b.extend_from_slice(b"}}");
        responses_sse_emit_event(w, st, &b);
    }
    let mut b = format!(
        "{{\"type\":\"response.output_item.done\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"reasoning\",\"status\":\"{item_status}\",\"summary\":[",
        st.reasoning_index, st.reasoning_id
    )
    .into_bytes();
    if !rtext.is_empty() {
        b.extend_from_slice(b"{\"type\":\"summary_text\",\"text\":");
        b.extend(json_escape_bytes(rtext));
        b.push(b'}');
    }
    b.extend_from_slice(b"]}}");
    responses_sse_emit_event(w, st, &b);
    true
}

fn responses_sse_message_added(w: &mut Writer, st: &mut ResponsesStream) {
    let b = format!(
        "{{\"type\":\"response.output_item.added\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}}}",
        st.message_index, st.message_id
    );
    responses_sse_emit_event(w, st, b.as_bytes());
}

fn responses_sse_message_text_part_added(w: &mut Writer, st: &mut ResponsesStream) {
    let b = format!(
        "{{\"type\":\"response.content_part.added\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}}}",
        st.message_id, st.message_index
    );
    responses_sse_emit_event(w, st, b.as_bytes());
}

fn responses_sse_output_text_delta(w: &mut Writer, st: &mut ResponsesStream, text: &[u8]) {
    if text.is_empty() {
        return;
    }
    let mut b = format!(
        "{{\"type\":\"response.output_text.delta\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":0,\"delta\":",
        st.message_id, st.message_index
    )
    .into_bytes();
    b.extend(json_escape_bytes(text));
    b.push(b'}');
    responses_sse_emit_event(w, st, &b);
}

fn json_escape_fragment(s: &[u8]) -> Vec<u8> {
    let full = json_escape_bytes(s);
    full[1..full.len() - 1].to_vec()
}

fn responses_message_text_escape_fixed(st: &ResponsesStream, raw: &[u8]) -> Vec<u8> {
    let mut b = vec![b'"'];
    if st.message_end > st.message_start {
        b.extend(json_escape_fragment(&raw[st.message_start..st.message_end]));
    }
    if st.message_tail_end > st.message_tail_start {
        b.extend(json_escape_fragment(
            &raw[st.message_tail_start..st.message_tail_end],
        ));
    }
    b.push(b'"');
    b
}

fn responses_sse_message_done(
    w: &mut Writer,
    st: &mut ResponsesStream,
    raw: &[u8],
    finish: &str,
) -> bool {
    let item_status = responses_item_status_for_finish(finish);
    let text = responses_message_text_escape_fixed(st, raw);
    let mut b = format!(
        "{{\"type\":\"response.output_text.done\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":0,\"text\":",
        st.message_id, st.message_index
    )
    .into_bytes();
    b.extend(&text);
    b.push(b'}');
    responses_sse_emit_event(w, st, &b);

    let mut b = format!(
        "{{\"type\":\"response.content_part.done\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":",
        st.message_id, st.message_index
    )
    .into_bytes();
    b.extend(&text);
    b.extend_from_slice(b",\"annotations\":[]}}");
    responses_sse_emit_event(w, st, &b);

    let mut b = format!(
        "{{\"type\":\"response.output_item.done\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"message\",\"status\":\"{item_status}\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":",
        st.message_index, st.message_id
    )
    .into_bytes();
    b.extend(&text);
    b.extend_from_slice(b",\"annotations\":[]}]}}");
    responses_sse_emit_event(w, st, &b);
    true
}

fn responses_sse_function_calls(
    w: &mut Writer,
    r: &StreamReq,
    st: &mut ResponsesStream,
    calls: &[ToolCall],
    item_status: &str,
) {
    for (i, tc) in calls.iter().enumerate() {
        let output_index = st.next_output_index;
        st.next_output_index += 1;
        let item_id = format!("fc_{}_{i}", st.response_id);
        let call_id = if tc.id.is_empty() {
            format!("{}_tool_{i}", st.response_id)
        } else {
            tc.id.clone()
        };
        let order = tool_schema_orders_find(&r.tool_orders, &tc.name);

        let mut added = format!(
            "{{\"type\":\"response.output_item.added\",\"output_index\":{output_index},\"item\":"
        )
        .into_bytes();
        append_responses_function_call_item(
            &mut added,
            tc,
            &item_id,
            &call_id,
            "in_progress",
            &r.tool_orders,
        );
        added.push(b'}');
        responses_sse_emit_event(w, st, &added);

        if !responses_tool_call_is_tool_search(tc, order) {
            let mut delta = format!(
                "{{\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"{item_id}\",\"output_index\":{output_index},\"delta\":"
            )
            .into_bytes();
            append_json_object_string(&mut delta, &tc.arguments);
            delta.push(b'}');
            responses_sse_emit_event(w, st, &delta);

            let mut done = format!(
                "{{\"type\":\"response.function_call_arguments.done\",\"item_id\":\"{item_id}\",\"output_index\":{output_index},\"name\":"
            )
            .into_bytes();
            done.extend(json_escape_bytes(
                responses_tool_wire_name(tc, order).as_bytes(),
            ));
            done.extend_from_slice(b",\"arguments\":");
            append_json_object_string(&mut done, &tc.arguments);
            done.push(b'}');
            responses_sse_emit_event(w, st, &done);
        }

        let mut done = format!(
            "{{\"type\":\"response.output_item.done\",\"output_index\":{output_index},\"item\":"
        )
        .into_bytes();
        append_responses_function_call_item(
            &mut done,
            tc,
            &item_id,
            &call_id,
            item_status,
            &r.tool_orders,
        );
        done.push(b'}');
        responses_sse_emit_event(w, st, &done);
    }
}

fn responses_sse_completed(
    w: &mut Writer,
    r: &StreamReq,
    st: &mut ResponsesStream,
    raw: &[u8],
    finish: &str,
    prompt: i32,
    completion: i32,
    reasoning_tokens: i32,
    created_at: i64,
    calls: &[ToolCall],
) {
    let event_type = if finish == "error" {
        "response.failed"
    } else if finish == "length" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    let status = responses_status_for_finish(finish);
    let item_status = responses_item_status_for_finish(finish);
    let mut b = format!(
        "{{\"type\":\"{event_type}\",\"response\":{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{created_at},\"status\":\"{status}\",\"model\":",
        st.response_id
    )
    .into_bytes();
    b.extend(json_escape_bytes(r.model.as_bytes()));
    if event_type == "response.failed" {
        b.extend_from_slice(
            b",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}",
        );
    } else if event_type == "response.incomplete" {
        b.extend_from_slice(b",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    }
    b.extend_from_slice(b",\"output\":[");
    let mut wrote = false;
    if st.reasoning_emitted_any {
        let reasoning_status = if st.reasoning_closed_naturally {
            "completed"
        } else {
            "incomplete"
        };
        let rtext = if st.reasoning_end > st.reasoning_start {
            &raw[st.reasoning_start..st.reasoning_end]
        } else {
            b""
        };
        b.extend_from_slice(
            format!(
                "{{\"id\":\"{}\",\"type\":\"reasoning\",\"status\":\"{reasoning_status}\",\"summary\":[",
                st.reasoning_id
            )
            .as_bytes(),
        );
        if !rtext.is_empty() {
            b.extend_from_slice(b"{\"type\":\"summary_text\",\"text\":");
            b.extend(json_escape_bytes(rtext));
            b.push(b'}');
        }
        b.extend_from_slice(b"]}");
        wrote = true;
    }
    if st.message_emitted_any {
        if wrote {
            b.push(b',');
        }
        b.extend_from_slice(
            format!(
                "{{\"id\":\"{}\",\"type\":\"message\",\"status\":\"{item_status}\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":",
                st.message_id
            )
            .as_bytes(),
        );
        b.extend(responses_message_text_escape_fixed(st, raw));
        b.extend_from_slice(b",\"annotations\":[]}]}");
        wrote = true;
    }
    for (i, tc) in calls.iter().enumerate() {
        if wrote {
            b.push(b',');
        }
        let fc_id = format!("fc_{}_{i}", st.response_id);
        let call_id = if tc.id.is_empty() {
            format!("{}_tool_{i}", st.response_id)
        } else {
            tc.id.clone()
        };
        append_responses_function_call_item(
            &mut b,
            tc,
            &fc_id,
            &call_id,
            item_status,
            &r.tool_orders,
        );
        wrote = true;
    }
    b.push(b']');
    b.extend_from_slice(b",\"usage\":");
    b.extend_from_slice(responses_usage(r, prompt, completion, reasoning_tokens).as_bytes());
    b.extend_from_slice(b"}}");
    responses_sse_emit_event(w, st, &b);
}

pub fn responses_sse_stream_update(
    w: &mut Writer,
    r: &StreamReq,
    st: &mut ResponsesStream,
    raw: &[u8],
    final_: bool,
) -> bool {
    if !st.active {
        return true;
    }
    let emit_reasoning = r.reasoning_summary_emit;
    if st.mode == RespMode::Thinking {
        if !st.checked_think_prefix {
            let open = think_start(r.chat_format).as_bytes();
            if raw.len() < open.len() && raw == &open[..raw.len()] && !final_ {
                return true;
            }
            if raw.len() >= open.len() && raw.starts_with(open) {
                st.emit_pos = open.len();
            }
            st.checked_think_prefix = true;
        }
        let close_s = think_end(r.chat_format);
        let close = find_substr(raw, st.emit_pos, close_s.as_bytes());
        let limit = if let Some(c) = close {
            c
        } else if final_ {
            utf8_stream_safe_len(raw, st.emit_pos, raw.len(), true)
        } else {
            let hold = close_s.len() - 1;
            let lim = if raw.len() > hold {
                raw.len() - hold
            } else {
                st.emit_pos
            };
            utf8_stream_safe_len(raw, st.emit_pos, lim, false)
        };
        if limit > st.emit_pos {
            if emit_reasoning {
                if !st.reasoning_item_opened {
                    st.reasoning_index = st.next_output_index;
                    st.next_output_index += 1;
                    responses_sse_reasoning_added(w, st);
                    st.reasoning_item_opened = true;
                }
                if !st.reasoning_summary_started {
                    responses_sse_reasoning_summary_part_added(w, st);
                    st.reasoning_summary_started = true;
                }
                responses_sse_reasoning_delta(w, st, &raw[st.emit_pos..limit]);
                if !st.reasoning_emitted_any {
                    st.reasoning_start = st.emit_pos;
                }
                st.reasoning_end = limit;
                st.reasoning_emitted_any = true;
            }
            st.emit_pos = limit;
        }
        if let Some(c) = close {
            st.emit_pos = c + close_s.len();
            st.mode = RespMode::Text;
            st.reasoning_closed_naturally = true;
        } else if final_ {
            st.mode = RespMode::Suppress;
            return true;
        } else {
            return true;
        }
    }
    if st.mode == RespMode::Text {
        let limit = text_stream_safe_limit(
            raw,
            st.emit_pos,
            raw.len(),
            r.has_tools,
            final_,
            r.chat_format,
        );
        if limit > st.emit_pos {
            if !st.message_item_opened {
                st.message_index = st.next_output_index;
                st.next_output_index += 1;
                responses_sse_message_added(w, st);
                st.message_item_opened = true;
            }
            if !st.message_text_part_open {
                responses_sse_message_text_part_added(w, st);
                st.message_text_part_open = true;
            }
            responses_sse_output_text_delta(w, st, &raw[st.emit_pos..limit]);
            if !st.message_emitted_any {
                st.message_start = st.emit_pos;
            }
            st.message_end = limit;
            st.message_emitted_any = true;
            st.emit_pos = limit;
        }
        if final_ {
            st.mode = RespMode::Suppress;
        }
    }
    true
}

pub fn responses_sse_finish_live(
    w: &mut Writer,
    r: &StreamReq,
    st: &mut ResponsesStream,
    raw: &[u8],
    finish: &str,
    prompt: i32,
    completion: i32,
    reasoning_tokens: i32,
    created_at: i64,
    calls: &[ToolCall],
) -> bool {
    if !responses_sse_stream_update(w, r, st, raw, true) {
        return false;
    }
    if st.reasoning_end > raw.len() {
        st.reasoning_end = raw.len();
    }
    if st.reasoning_start > st.reasoning_end {
        st.reasoning_start = st.reasoning_end;
    }
    if st.message_end > raw.len() {
        st.message_end = raw.len();
    }
    if st.message_start > st.message_end {
        st.message_start = st.message_end;
    }
    if st.reasoning_item_opened && !st.reasoning_item_closed {
        if !responses_sse_reasoning_done(w, st, raw, finish) {
            return false;
        }
        st.reasoning_item_closed = true;
    }
    if st.message_item_opened && !st.message_item_closed {
        if !responses_sse_message_done(w, st, raw, finish) {
            return false;
        }
        st.message_item_closed = true;
    }
    responses_sse_function_calls(w, r, st, calls, responses_item_status_for_finish(finish));
    responses_sse_completed(
        w,
        r,
        st,
        raw,
        finish,
        prompt,
        completion,
        reasoning_tokens,
        created_at,
        calls,
    );
    true
}

fn append_anthropic_content(
    out: &mut Vec<u8>,
    text: &[u8],
    reasoning: &[u8],
    calls: &[ToolCall],
    id_prefix: &str,
) {
    out.push(b'[');
    let mut wrote = false;
    let mut wrote_after_thinking = false;
    if !reasoning.is_empty() {
        out.extend_from_slice(b"{\"type\":\"thinking\",\"thinking\":");
        out.extend(json_escape_bytes(reasoning));
        out.extend_from_slice(b",\"signature\":\"\"}");
        wrote = true;
    }
    if !text.is_empty() {
        if wrote {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"type\":\"text\",\"text\":");
        out.extend(json_escape_bytes(text));
        out.push(b'}');
        wrote = true;
        wrote_after_thinking = true;
    }
    for (i, tc) in calls.iter().enumerate() {
        if wrote {
            out.push(b',');
        }
        append_anthropic_tool_use(out, tc, id_prefix, i);
        wrote = true;
        wrote_after_thinking = true;
    }
    if !wrote || (!reasoning.is_empty() && !wrote_after_thinking) {
        if wrote {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"type\":\"text\",\"text\":\"\"}");
    }
    out.push(b']');
}

pub fn final_response(
    r: &StreamReq,
    id: &str,
    text: &[u8],
    reasoning: Option<&[u8]>,
    finish: &str,
    prompt: i32,
    completion: i32,
    created: i64,
    cors: bool,
    calls: &[ToolCall],
) -> Vec<u8> {
    let text = utf8_trim_tail(text);
    let reasoning = reasoning.map(utf8_trim_tail).unwrap_or(b"");
    let mut b = Vec::new();
    if r.kind == ReqKind::Chat {
        b.extend(
            format!(
                "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\"model\":"
            )
            .as_bytes(),
        );
        b.extend(json_escape_bytes(r.model.as_bytes()));
        b.extend_from_slice(
            b",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
        );
        b.extend(json_escape_bytes(text));
        if !reasoning.is_empty() {
            b.extend_from_slice(b",\"reasoning_content\":");
            b.extend(json_escape_bytes(reasoning));
        }
        if !calls.is_empty() {
            b.extend_from_slice(b",\"tool_calls\":");
            append_tool_calls_json(&mut b, calls, id);
        }
        b.extend_from_slice(b"},\"finish_reason\":");
        b.extend(json_escape_bytes(finish.as_bytes()));
        b.extend_from_slice(b"}],\"usage\":");
    } else {
        b.extend(
            format!(
                "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":"
            )
            .as_bytes(),
        );
        b.extend(json_escape_bytes(r.model.as_bytes()));
        b.extend_from_slice(b",\"choices\":[{\"text\":");
        b.extend(json_escape_bytes(text));
        b.extend_from_slice(b",\"index\":0,\"finish_reason\":");
        b.extend(json_escape_bytes(finish.as_bytes()));
        b.extend_from_slice(b"}],\"usage\":");
    }
    b.extend_from_slice(openai_usage(r, prompt, completion).as_bytes());
    b.extend_from_slice(r.timings.json_suffix().as_bytes());
    b.extend_from_slice(b"}\n");
    let body = String::from_utf8(b).expect("final_response utf8");
    http_response_bytes(200, Some("application/json"), None, cors, &body)
}

pub fn anthropic_final_response(
    r: &StreamReq,
    id: &str,
    text: &[u8],
    reasoning: Option<&[u8]>,
    finish: &str,
    matched_stop: Option<&str>,
    prompt: i32,
    completion: i32,
    cors: bool,
    calls: &[ToolCall],
) -> Vec<u8> {
    let text = utf8_trim_tail(text);
    let reasoning = reasoning.map(utf8_trim_tail).unwrap_or(b"");
    let mut b = format!("{{\"id\":\"{id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":")
        .into_bytes();
    b.extend(json_escape_bytes(r.model.as_bytes()));
    b.extend_from_slice(b",\"content\":");
    append_anthropic_content(&mut b, text, reasoning, calls, id);
    b.push(b',');
    append_anthropic_stop_fields(&mut b, finish, matched_stop);
    b.extend_from_slice(b",\"usage\":");
    b.extend_from_slice(anthropic_usage(r, prompt, completion).as_bytes());
    b.extend_from_slice(r.timings.json_suffix().as_bytes());
    b.extend_from_slice(b"}\n");
    let body = String::from_utf8(b).expect("anthropic_final utf8");
    http_response_bytes(200, Some("application/json"), None, cors, &body)
}

pub fn responses_final_response(
    r: &StreamReq,
    text: &[u8],
    reasoning: Option<&[u8]>,
    finish: &str,
    prompt: i32,
    completion: i32,
    reasoning_tokens: i32,
    created: i64,
    cors: bool,
    response_id: &str,
    reasoning_id: &str,
    message_id: &str,
    calls: &[ToolCall],
) -> Vec<u8> {
    let text = utf8_trim_tail(text);
    let reasoning = reasoning.map(utf8_trim_tail).unwrap_or(b"");
    let status = responses_status_for_finish(finish);
    let item_status = responses_item_status_for_finish(finish);
    let mut b = format!(
        "{{\"id\":\"{response_id}\",\"object\":\"response\",\"created_at\":{created},\"status\":\"{status}\",\"model\":"
    )
    .into_bytes();
    b.extend(json_escape_bytes(r.model.as_bytes()));
    if finish == "error" {
        b.extend_from_slice(
            b",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}",
        );
    } else if finish == "length" {
        b.extend_from_slice(b",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    }
    b.extend_from_slice(b",\"output\":[");
    let mut wrote = false;
    if !reasoning.is_empty() && r.reasoning_summary_emit {
        b.extend_from_slice(
            format!(
                "{{\"id\":\"{reasoning_id}\",\"type\":\"reasoning\",\"status\":\"{item_status}\",\"summary\":[{{\"type\":\"summary_text\",\"text\":"
            )
            .as_bytes(),
        );
        b.extend(json_escape_bytes(reasoning));
        b.extend_from_slice(b"}]}");
        wrote = true;
    }
    if !text.is_empty() {
        if wrote {
            b.push(b',');
        }
        b.extend_from_slice(
            format!(
                "{{\"id\":\"{message_id}\",\"type\":\"message\",\"status\":\"{item_status}\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":"
            )
            .as_bytes(),
        );
        b.extend(json_escape_bytes(text));
        b.extend_from_slice(b",\"annotations\":[]}]}");
        wrote = true;
    }
    for (i, tc) in calls.iter().enumerate() {
        if wrote {
            b.push(b',');
        }
        let fc_id = format!("fc_{response_id}_{i}");
        let call_id = if tc.id.is_empty() {
            format!("{response_id}_tool_{i}")
        } else {
            tc.id.clone()
        };
        append_responses_function_call_item(
            &mut b,
            tc,
            &fc_id,
            &call_id,
            item_status,
            &r.tool_orders,
        );
        wrote = true;
    }
    b.push(b']');
    b.extend_from_slice(b",\"usage\":");
    b.extend_from_slice(responses_usage(r, prompt, completion, reasoning_tokens).as_bytes());
    b.extend_from_slice(r.timings.json_suffix().as_bytes());
    b.push(b'}');
    let body = String::from_utf8(b).expect("responses_final utf8");
    http_response_bytes(200, Some("application/json"), None, cors, &body)
}

fn find_substr(hay: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    if start > hay.len() {
        return None;
    }
    hay[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| start + i)
}

pub fn project_openai_chat_thinking(created: i64) -> Vec<u8> {
    let mut r = StreamReq::default();
    r.api = Api::Openai;
    r.kind = ReqKind::Chat;
    r.stream = true;
    r.think_mode = ThinkMode::Low;
    let mut w = Writer::new(created);
    w.put(&sse_headers(false));
    sse_chunk(&mut w, &r, "chatcmpl_tape", None, None);
    let mut st = openai_stream_start(&r);
    let mut raw = Vec::new();
    for piece in TAPE_THINKING {
        raw.extend_from_slice(piece.as_bytes());
        openai_sse_stream_update(&mut w, &r, "chatcmpl_tape", &mut st, &raw, false);
    }
    openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_tape",
        &mut st,
        &raw,
        "stop",
        7,
        4,
        &[],
    );
    w.out
}

pub fn project_openai_chat_utf8(created: i64) -> Vec<u8> {
    let mut r = StreamReq::default();
    r.think_mode = ThinkMode::None;
    let mut w = Writer::new(created);
    w.put(&sse_headers(false));
    let mut st = openai_stream_start(&r);
    let mut raw = Vec::new();
    for piece in TAPE_UTF8 {
        raw.extend_from_slice(piece);
        openai_sse_stream_update(&mut w, &r, "chatcmpl_tape8", &mut st, &raw, false);
    }
    openai_sse_finish_live(
        &mut w,
        &r,
        "chatcmpl_tape8",
        &mut st,
        &raw,
        "stop",
        4,
        4,
        &[],
    );
    w.out
}

pub fn project_openai_completion(created: i64) -> Vec<u8> {
    let mut r = StreamReq::default();
    r.kind = ReqKind::Completion;
    r.api = Api::Openai;
    r.stream = true;
    let mut w = Writer::new(created);
    w.put(&sse_headers(false));
    for piece in TAPE_PLAIN {
        sse_chunk(&mut w, &r, "cmpl_tape", Some(piece.as_bytes()), None);
    }
    sse_chunk(&mut w, &r, "cmpl_tape", None, Some("stop"));
    sse_done(&mut w, &r, "cmpl_tape", 4, 4);
    w.out
}

pub fn project_anthropic_thinking(created: i64) -> Vec<u8> {
    let mut r = StreamReq::default();
    r.api = Api::Anthropic;
    r.stream = true;
    r.think_mode = ThinkMode::Low;
    let mut w = Writer::new(created);
    let mut st = anthropic_sse_start_live(&mut w, &r, "msg_tape", 7);
    let mut raw = Vec::new();
    for piece in TAPE_THINKING {
        raw.extend_from_slice(piece.as_bytes());
        anthropic_sse_stream_update(&mut w, &r, "msg_tape", &mut st, &raw, false);
    }
    anthropic_sse_finish_live(&mut w, &r, "msg_tape", &mut st, &raw, "stop", None, 4, &[]);
    w.out
}

pub fn project_responses_thinking(created: i64) -> Vec<u8> {
    let mut r = StreamReq::default();
    r.api = Api::Responses;
    r.stream = true;
    r.think_mode = ThinkMode::Low;
    r.reasoning_summary_emit = true;
    let mut w = Writer::new(created);
    let mut st = responses_stream_init(&r, TEST_RESP_ID, TEST_RS_ID, TEST_MSG_ID);
    st.active = true;
    responses_sse_created(&mut w, &r, &mut st, created);
    let mut raw = Vec::new();
    for piece in TAPE_THINKING {
        raw.extend_from_slice(piece.as_bytes());
        responses_sse_stream_update(&mut w, &r, &mut st, &raw, false);
    }
    responses_sse_finish_live(&mut w, &r, &mut st, &raw, "stop", 7, 4, 1, created, &[]);
    w.out
}

#[cfg(test)]
mod stream_failure_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn responses_stream_error_continues_the_live_sequence() {
        let mut req = StreamReq {
            api: Api::Responses,
            ..StreamReq::default()
        };
        req.stream = true;
        let mut state = responses_stream_init(&req, "resp_test", "rs_test", "msg_test");
        let mut writer = Writer::new(7);
        responses_sse_created(&mut writer, &req, &mut state, 7);
        writer.out.clear();

        stream_error(&mut writer, &req, Some(&mut state), "boom");

        assert_eq!(
            writer.out,
            b"data: {\"type\":\"error\",\"sequence_number\":1,\"code\":\"server_error\",\"message\":\"boom\",\"param\":null}\n\n"
        );
        assert_eq!(state.sequence, 2);
    }

    #[test]
    fn heartbeat_uses_native_surface_and_preserves_responses_sequence() {
        let now = Instant::now();
        let mut last = now;
        let mut writer = Writer::new(7);
        let mut req = StreamReq {
            stream: true,
            model: "model".into(),
            ..StreamReq::default()
        };

        assert!(!stream_heartbeat_if_due(
            &mut writer,
            &req,
            None,
            &mut last,
            now + Duration::from_secs(4),
            ": decode\n\n",
        ));
        assert!(writer.out.is_empty());

        assert!(stream_heartbeat_if_due(
            &mut writer,
            &req,
            None,
            &mut last,
            now + Duration::from_secs(5),
            ": decode\n\n",
        ));
        assert_eq!(writer.out, b": decode\n\n");

        writer.out.clear();
        req.api = Api::Anthropic;
        assert!(stream_heartbeat_if_due(
            &mut writer,
            &req,
            None,
            &mut last,
            now + Duration::from_secs(10),
            ": decode\n\n",
        ));
        assert_eq!(writer.out, b"event: ping\ndata: {\"type\": \"ping\"}\n\n");

        writer.out.clear();
        req.api = Api::Responses;
        let mut state = responses_stream_init(&req, "resp_test", "rs_test", "msg_test");
        responses_sse_created(&mut writer, &req, &mut state, 7);
        writer.out.clear();
        assert!(stream_heartbeat_if_due(
            &mut writer,
            &req,
            Some(&mut state),
            &mut last,
            now + Duration::from_secs(15),
            ": decode\n\n",
        ));
        assert_eq!(
            writer.out,
            b"data: {\"type\":\"response.in_progress\",\"sequence_number\":1,\"response\":{\"id\":\"resp_test\",\"object\":\"response\",\"created_at\":7,\"status\":\"in_progress\",\"model\":\"model\",\"output\":[]}}\n\n"
        );
        assert_eq!(state.sequence, 2);
    }

    #[test]
    fn responses_function_calls_emit_item_and_argument_lifecycle() {
        let req = StreamReq {
            api: Api::Responses,
            stream: true,
            model: "model".into(),
            ..StreamReq::default()
        };
        let mut state = responses_stream_init(&req, "resp_test", "rs_test", "msg_test");
        let mut writer = Writer::new(7);
        responses_sse_created(&mut writer, &req, &mut state, 7);
        let calls = [ToolCall {
            id: "call_test".into(),
            name: "bash".into(),
            arguments: r#"{"command":"true"}"#.into(),
        }];

        responses_sse_finish_live(
            &mut writer,
            &req,
            &mut state,
            b"",
            "tool_calls",
            3,
            1,
            0,
            7,
            &calls,
        );

        let wire = String::from_utf8(writer.out).unwrap();
        let added = wire
            .find("\"type\":\"response.output_item.added\"")
            .unwrap();
        let delta = wire
            .find("\"type\":\"response.function_call_arguments.delta\"")
            .unwrap();
        let done = wire
            .find("\"type\":\"response.function_call_arguments.done\"")
            .unwrap();
        let item_done = wire.find("\"type\":\"response.output_item.done\"").unwrap();
        let completed = wire.find("\"type\":\"response.completed\"").unwrap();
        assert!(added < delta && delta < done && done < item_done && item_done < completed);
        assert!(wire.contains("\"item_id\":\"fc_resp_test_0\""), "{wire}");
        assert!(wire.contains("\"call_id\":\"call_test\""), "{wire}");
    }
}
