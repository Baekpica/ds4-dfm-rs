//! Incremental DSML → OpenAI / Anthropic live tool projection.
//! Copied from `ds4_server.c` at v0.6.3-dfm. Final parse stays authoritative.

use crate::json::json_escape_bytes;
use crate::parse::ToolCall;
use crate::stream::utf8_stream_safe_len;
use crate::tools::{
    dsml_attr, mint_tool_id, DSML_INVOKE_END, DSML_INVOKE_END_SHORT, DSML_INVOKE_END_SPACED,
    DSML_INVOKE_START, DSML_INVOKE_START_SHORT, DSML_INVOKE_START_SPACED, DSML_PARAM_END,
    DSML_PARAM_END_SHORT, DSML_PARAM_END_SPACED, DSML_PARAM_START, DSML_PARAM_START_SHORT,
    DSML_PARAM_START_SPACED, DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SHORT,
    DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START, DSML_TOOL_CALLS_START_SHORT,
    DSML_TOOL_CALLS_START_SPACED,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsmlToolState {
    BetweenInvokes,
    BetweenParams,
    ParamValue,
    Done,
    Error,
}

#[derive(Clone, Copy, Debug)]
pub struct DsmlSyntax {
    pub tool_calls_start: &'static [u8],
    pub tool_calls_end: &'static [u8],
    pub invoke_start: &'static [u8],
    pub invoke_end: &'static [u8],
    pub param_start: &'static [u8],
    pub param_end: &'static [u8],
    /// Old generations write `&`/`<` in string values as XML entities (the
    /// stream holds back a chunk tail that would split one); the official
    /// spaced generation writes values raw and its only escape is the render
    /// guard (engine server_dsml_stream.c dsml_value_unescape / :384-406).
    pub entity_values: bool,
}

pub const DSML_SYNTAXES: [DsmlSyntax; 4] = [
    DsmlSyntax {
        tool_calls_start: DSML_TOOL_CALLS_START.as_bytes(),
        tool_calls_end: DSML_TOOL_CALLS_END.as_bytes(),
        invoke_start: DSML_INVOKE_START.as_bytes(),
        invoke_end: DSML_INVOKE_END.as_bytes(),
        param_start: DSML_PARAM_START.as_bytes(),
        param_end: DSML_PARAM_END.as_bytes(),
        entity_values: true,
    },
    DsmlSyntax {
        tool_calls_start: DSML_TOOL_CALLS_START_SHORT.as_bytes(),
        tool_calls_end: DSML_TOOL_CALLS_END_SHORT.as_bytes(),
        invoke_start: DSML_INVOKE_START_SHORT.as_bytes(),
        invoke_end: DSML_INVOKE_END_SHORT.as_bytes(),
        param_start: DSML_PARAM_START_SHORT.as_bytes(),
        param_end: DSML_PARAM_END_SHORT.as_bytes(),
        entity_values: true,
    },
    DsmlSyntax {
        tool_calls_start: b"<tool_calls>",
        tool_calls_end: b"</tool_calls>",
        invoke_start: b"<invoke",
        invoke_end: b"</invoke>",
        param_start: b"<parameter",
        param_end: b"</parameter>",
        entity_values: true,
    },
    // The official V4.1 encoding.py spelling (4294136).
    DsmlSyntax {
        tool_calls_start: DSML_TOOL_CALLS_START_SPACED.as_bytes(),
        tool_calls_end: DSML_TOOL_CALLS_END_SPACED.as_bytes(),
        invoke_start: DSML_INVOKE_START_SPACED.as_bytes(),
        invoke_end: DSML_INVOKE_END_SPACED.as_bytes(),
        param_start: DSML_PARAM_START_SPACED.as_bytes(),
        param_end: DSML_PARAM_END_SPACED.as_bytes(),
        entity_values: false,
    },
];

#[derive(Debug, Clone)]
pub struct DsmlToolStream {
    pub state: DsmlToolState,
    pub syn: Option<DsmlSyntax>,
    pub parse_pos: usize,
    pub index: i32,
    pub active: bool,
    pub emitted_any: bool,
    pub args_open: bool,
    first_param: bool,
    param_is_string: bool,
    ids: Vec<String>,
    id_prefix: &'static str,
    random_ids: bool,
}

impl Default for DsmlToolStream {
    fn default() -> Self {
        Self {
            state: DsmlToolState::BetweenInvokes,
            syn: None,
            parse_pos: 0,
            index: 0,
            active: false,
            emitted_any: false,
            args_open: false,
            first_param: false,
            param_is_string: false,
            ids: Vec::new(),
            id_prefix: "call_",
            random_ids: false,
        }
    }
}

impl DsmlToolStream {
    pub fn with_prefix(prefix: &'static str) -> Self {
        Self {
            id_prefix: prefix,
            ..Self::default()
        }
    }

    pub(crate) fn use_random_ids(&mut self) {
        self.random_ids = true;
    }

    pub fn init(&mut self, raw: &[u8], pos: usize) -> bool {
        let prefix = self.id_prefix;
        let random_ids = self.random_ids;
        *self = Self::with_prefix(prefix);
        self.random_ids = random_ids;
        self.active = true;
        self.state = DsmlToolState::BetweenInvokes;
        for syn in DSML_SYNTAXES {
            if raw_full_lit(raw, pos, syn.tool_calls_start) {
                self.syn = Some(syn);
                self.parse_pos = pos + syn.tool_calls_start.len();
                return true;
            }
        }
        self.active = false;
        self.state = DsmlToolState::Error;
        false
    }

    pub fn id_at(&mut self, index: i32) -> String {
        if index < 0 {
            return String::new();
        }
        let i = index as usize;
        while self.ids.len() <= i {
            self.ids.push(String::new());
        }
        if self.ids[i].is_empty() {
            self.ids[i] = if self.random_ids {
                mint_tool_id(self.id_prefix)
            } else {
                format!("{}{index}", self.id_prefix)
            };
        }
        self.ids[i].clone()
    }

    pub fn apply_ids(&self, calls: &mut [ToolCall]) {
        let n = calls.len().min(self.ids.len());
        for i in 0..n {
            if calls[i].id.is_empty() && !self.ids[i].is_empty() {
                calls[i].id = self.ids[i].clone();
            }
        }
    }

    /// A call whose name went on the wire but whose arguments did not: the
    /// live stream started this invoke and then failed mid-block on the
    /// V4.1 artifact's mixed tag spellings (docs 6.23).  The finish
    /// completes this index with an arguments-only delta -- the client
    /// merges deltas by index.  None when the live stream finished
    /// cleanly, emitted no call, or already streamed part of the
    /// arguments (nothing safe to append).
    pub fn partial_invoke_index(&self) -> Option<i32> {
        if !self.emitted_any || self.state == DsmlToolState::Done || self.args_open {
            return None;
        }
        Some(self.index)
    }

    fn fail(&mut self) -> bool {
        self.active = false;
        self.state = DsmlToolState::Error;
        true
    }

    pub fn update<S: ToolSink>(&mut self, sink: &mut S, raw: &[u8]) -> bool {
        let Some(syn) = self.syn else {
            return true;
        };
        while self.active && self.parse_pos < raw.len() {
            match self.state {
                DsmlToolState::BetweenInvokes => {
                    while self.parse_pos < raw.len() && raw[self.parse_pos].is_ascii_whitespace() {
                        self.parse_pos += 1;
                    }
                    if self.parse_pos >= raw.len() {
                        return true;
                    }
                    if raw_full_lit(raw, self.parse_pos, syn.tool_calls_end) {
                        self.parse_pos += syn.tool_calls_end.len();
                        self.active = false;
                        self.state = DsmlToolState::Done;
                        return true;
                    }
                    if raw_partial_any(raw, self.parse_pos, syn.tool_calls_end, syn.invoke_start) {
                        return true;
                    }
                    if raw_full_lit(raw, self.parse_pos, syn.invoke_start) {
                        let before = self.parse_pos;
                        let before_state = self.state;
                        if !self.start_invoke(sink, raw) {
                            return false;
                        }
                        if self.parse_pos == before && self.state == before_state {
                            return true;
                        }
                        continue;
                    }
                    return self.fail();
                }
                DsmlToolState::BetweenParams => {
                    while self.parse_pos < raw.len() && raw[self.parse_pos].is_ascii_whitespace() {
                        self.parse_pos += 1;
                    }
                    if self.parse_pos >= raw.len() {
                        return true;
                    }
                    if raw_full_lit(raw, self.parse_pos, syn.invoke_end) {
                        if self.args_open && !sink.args_fragment(self.index, b"}") {
                            return false;
                        }
                        self.args_open = false;
                        if !sink.close_invoke(self.index) {
                            return false;
                        }
                        self.parse_pos += syn.invoke_end.len();
                        self.index += 1;
                        self.state = DsmlToolState::BetweenInvokes;
                        continue;
                    }
                    if raw_partial_lit(raw, self.parse_pos, syn.invoke_end)
                        || param_open_partial(&syn, raw, self.parse_pos)
                    {
                        return true;
                    }
                    if param_open_full(&syn, raw, self.parse_pos) {
                        let before = self.parse_pos;
                        let before_state = self.state;
                        if !self.start_param(sink, raw) {
                            return false;
                        }
                        if self.parse_pos == before && self.state == before_state {
                            return true;
                        }
                        continue;
                    }
                    return self.fail();
                }
                DsmlToolState::ParamValue => {
                    if let Some((end, end_len)) = param_close_find(&syn, raw, self.parse_pos) {
                        if !self.finish_param(sink, raw, self.parse_pos + end, end_len) {
                            return false;
                        }
                        continue;
                    }
                    let limit = tool_param_value_stream_safe_len(
                        raw,
                        self.parse_pos,
                        syn.param_end,
                        bare_param_alias(&syn).then_some(BARE_PARAM_END),
                        self.param_is_string,
                        syn.entity_values,
                    );
                    if limit > self.parse_pos {
                        let ok = if self.param_is_string {
                            emit_string_value(sink, self.index, &syn, &raw[self.parse_pos..limit])
                        } else {
                            sink.args_fragment(self.index, &raw[self.parse_pos..limit])
                        };
                        if !ok {
                            return false;
                        }
                        self.parse_pos = limit;
                    }
                    return true;
                }
                DsmlToolState::Done | DsmlToolState::Error => return true,
            }
        }
        true
    }

    fn start_invoke<S: ToolSink>(&mut self, sink: &mut S, raw: &[u8]) -> bool {
        let Some(rel) = raw[self.parse_pos..].iter().position(|&c| c == b'>') else {
            return true;
        };
        let tag = &raw[self.parse_pos..self.parse_pos + rel + 1];
        let Some(name) = dsml_attr(tag, "name") else {
            return self.fail();
        };
        let id = self.id_at(self.index);
        if !sink.start_invoke(self.index, &id, &name) {
            return false;
        }
        if !sink.args_fragment(self.index, b"{") {
            return false;
        }
        self.emitted_any = true;
        self.args_open = true;
        self.first_param = true;
        self.parse_pos = self.parse_pos + rel + 1;
        self.state = DsmlToolState::BetweenParams;
        true
    }

    fn start_param<S: ToolSink>(&mut self, sink: &mut S, raw: &[u8]) -> bool {
        let Some(rel) = raw[self.parse_pos..].iter().position(|&c| c == b'>') else {
            return true;
        };
        let tag = &raw[self.parse_pos..self.parse_pos + rel + 1];
        let Some(name) = dsml_attr(tag, "name") else {
            return self.fail();
        };
        // A bare parameter carries no string attribute; the engine defaults
        // it to string, matching the final parser (f302be7).
        let string_value = dsml_attr(tag, "string")
            .map(|v| v == b"true")
            .unwrap_or(true);
        if !emit_param_prefix(sink, self.index, &name, string_value, self.first_param) {
            return false;
        }
        self.first_param = false;
        self.param_is_string = string_value;
        self.parse_pos = self.parse_pos + rel + 1;
        self.state = DsmlToolState::ParamValue;
        true
    }

    fn finish_param<S: ToolSink>(
        &mut self,
        sink: &mut S,
        raw: &[u8],
        value_end: usize,
        end_len: usize,
    ) -> bool {
        let syn = self.syn.unwrap();
        if value_end > self.parse_pos {
            let ok = if self.param_is_string {
                emit_string_value(sink, self.index, &syn, &raw[self.parse_pos..value_end])
            } else {
                sink.args_fragment(self.index, &raw[self.parse_pos..value_end])
            };
            if !ok {
                return false;
            }
        }
        if self.param_is_string && !sink.args_fragment(self.index, b"\"") {
            return false;
        }
        self.parse_pos = value_end + end_len;
        self.state = DsmlToolState::BetweenParams;
        true
    }
}

pub trait ToolSink {
    fn start_invoke(&mut self, index: i32, id: &str, name: &[u8]) -> bool;
    fn args_fragment(&mut self, index: i32, text: &[u8]) -> bool;
    fn close_invoke(&mut self, index: i32) -> bool;
}

fn emit_param_prefix<S: ToolSink>(
    sink: &mut S,
    index: i32,
    name: &[u8],
    is_string: bool,
    first: bool,
) -> bool {
    let mut frag = Vec::new();
    if !first {
        frag.push(b',');
    }
    frag.extend(json_escape_bytes(name));
    frag.push(b':');
    if is_string {
        frag.push(b'"');
    }
    sink.args_fragment(index, &frag)
}

fn emit_string_value<S: ToolSink>(sink: &mut S, index: i32, syn: &DsmlSyntax, text: &[u8]) -> bool {
    if text.is_empty() {
        return true;
    }
    let unescaped =
        crate::tools::dsml_value_unescape_syntax(syn.entity_values, syn.param_end, text);
    let frag = json_escape_fragment(&unescaped);
    sink.args_fragment(index, &frag)
}

pub fn json_escape_fragment(s: &[u8]) -> Vec<u8> {
    let full = json_escape_bytes(s);
    if full.len() >= 2 {
        full[1..full.len() - 1].to_vec()
    } else {
        Vec::new()
    }
}

fn raw_full_lit(raw: &[u8], pos: usize, lit: &[u8]) -> bool {
    pos <= raw.len() && raw.len() - pos >= lit.len() && raw[pos..].starts_with(lit)
}

fn raw_partial_lit(raw: &[u8], pos: usize, lit: &[u8]) -> bool {
    if pos > raw.len() || raw.len() - pos >= lit.len() {
        return false;
    }
    raw[pos..] == lit[..raw.len() - pos]
}

fn raw_partial_any(raw: &[u8], pos: usize, a: &[u8], b: &[u8]) -> bool {
    raw_partial_lit(raw, pos, a) || raw_partial_lit(raw, pos, b)
}

/// The engine's bare-alias rule (f302be7, server_dsml_parse.c
/// dsml_param_open_len/close_len): the quantized artifact, with a system
/// text and the tool instructions in context, writes the parameter tags
/// bare -- `<parameter name="x">v</parameter>` -- while the wrapper and
/// invoke opens stay DSML (fable5.md, 10-11/10-12; measured 5/5 with a
/// Chinese system, 0/5 without).  A syntax whose own spelling is not
/// already the bare one accepts the bare form as an alias.  The open alias
/// keeps the trailing space: `<parameters>` is not an open tag.  The
/// engine routed the final parse, the live streams, and the KV syntax
/// tracking through the same entry points; these are those entry points.
const BARE_PARAM_START: &[u8] = b"<parameter ";
const BARE_PARAM_END: &[u8] = b"</parameter>";

fn bare_param_alias(syn: &DsmlSyntax) -> bool {
    syn.param_start != &b"<parameter"[..]
}

fn param_open_full(syn: &DsmlSyntax, raw: &[u8], pos: usize) -> bool {
    raw_full_lit(raw, pos, syn.param_start)
        || (bare_param_alias(syn) && raw_full_lit(raw, pos, BARE_PARAM_START))
}

fn param_open_partial(syn: &DsmlSyntax, raw: &[u8], pos: usize) -> bool {
    raw_partial_lit(raw, pos, syn.param_start)
        || (bare_param_alias(syn) && raw_partial_lit(raw, pos, BARE_PARAM_START))
}

/// The earliest close from `from` in either spelling: (offset, length)
/// (f302be7's dsml_param_close_find takes the earlier of the two).
fn param_close_find(syn: &DsmlSyntax, raw: &[u8], from: usize) -> Option<(usize, usize)> {
    let own = find_lit_bounded(&raw[from..], syn.param_end).map(|r| (r, syn.param_end.len()));
    if !bare_param_alias(syn) {
        return own;
    }
    let bare = find_lit_bounded(&raw[from..], BARE_PARAM_END).map(|r| (r, BARE_PARAM_END.len()));
    match (own, bare) {
        (Some(a), Some(b)) => Some(if b.0 < a.0 { b } else { a }),
        (a, b) => a.or(b),
    }
}

fn find_lit_bounded(s: &[u8], lit: &[u8]) -> Option<usize> {
    if lit.is_empty() {
        return Some(0);
    }
    s.windows(lit.len()).position(|w| w == lit)
}

/// The engine's `dsml_entity_stream_safe_len` (server_dsml_stream.c:384-406):
/// a chunk tail that would split an escape sequence is held back.  The old
/// generations' escapes are the XML entities; the spaced generation writes
/// values raw and its only escape is the render guard "&lt;" + the closing
/// tag minus its leading '<' (dsml_value_unescape builds the same sentinel).
fn dsml_entity_stream_safe_len(
    raw: &[u8],
    start: usize,
    limit: usize,
    param_end: &[u8],
    entity_values: bool,
) -> usize {
    const ENTS: &[&[u8]] = &[b"&amp;", b"&lt;", b"&gt;", b"&quot;", b"&apos;"];
    let sentinel = [b"&lt;", &param_end[1..]].concat();
    let (list, max_ent): (&[&[u8]], usize) = if entity_values {
        (ENTS, 6)
    } else {
        (&[sentinel.as_slice()], sentinel.len())
    };
    let scan = if limit > start + max_ent {
        limit - max_ent
    } else {
        start
    };
    for i in (scan + 1..=limit).rev() {
        if raw[i - 1] != b'&' {
            continue;
        }
        let amp = i - 1;
        let tail = limit - amp;
        for ent in list {
            if tail < ent.len() && raw[amp..].starts_with(&ent[..tail]) {
                return amp;
            }
        }
        break;
    }
    limit
}

/// A chunk tail that is a proper prefix of `close` is held back; the hold
/// runs once per accepted spelling (f302be7's close_partial).
fn partial_close_hold(raw: &[u8], start: usize, close: &[u8]) -> usize {
    let raw_len = raw.len();
    let end_len = close.len();
    let scan = if raw_len > start + end_len {
        raw_len - end_len
    } else {
        start
    };
    for i in (scan + 1..=raw_len).rev() {
        if raw[i - 1] != b'<' {
            continue;
        }
        let marker = i - 1;
        let tail = raw_len - marker;
        if tail < end_len && raw[marker..].starts_with(&close[..tail]) {
            return marker;
        }
        break;
    }
    raw_len
}

fn tool_param_value_stream_safe_len(
    raw: &[u8],
    start: usize,
    param_end: &[u8],
    param_end_alias: Option<&[u8]>,
    is_string: bool,
    entity_values: bool,
) -> usize {
    let mut limit = partial_close_hold(raw, start, param_end);
    if let Some(alias) = param_end_alias {
        limit = limit.min(partial_close_hold(raw, start, alias));
    }
    if is_string {
        limit = dsml_entity_stream_safe_len(raw, start, limit, param_end, entity_values);
    }
    utf8_stream_safe_len(raw, start, limit, false)
}

/// Tape dumps matching `tests/parity/tool_stream_c_oracle`.
pub fn dump_script(name: &str) -> Vec<u8> {
    use crate::parse::ToolCall;
    use crate::route::{Api, ReqKind, ThinkMode};
    use crate::stream::{
        anthropic_sse_finish_live, anthropic_sse_start_live, anthropic_sse_stream_update,
        openai_sse_finish_live, openai_sse_stream_update, openai_stream_start, sse_chunk,
        ChatFormat, StreamReq, Writer, CREATED_TEST,
    };

    let swapped_bash = [ToolCall {
        name: "bash".into(),
        arguments: "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}".into(),
        ..Default::default()
    }];

    let mut r = StreamReq {
        kind: ReqKind::Chat,
        api: Api::Openai,
        model: "deepseek-v4-flash".into(),
        think_mode: ThinkMode::None,
        has_tools: true,
        stream: true,
        chat_format: ChatFormat::DeepSeek,
        ..StreamReq::default()
    };

    match name {
        "openai-partial" => {
            let mut w = Writer::new(CREATED_TEST);
            sse_chunk(&mut w, &r, "chatcmpl_partial_tool", None, None);
            let mut st = openai_stream_start(&r);
            let raw1 = format!(
                "Before.\n\n{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo partial",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_partial_tool",
                &mut st,
                raw1.as_bytes(),
                false,
            );
            let raw2 = format!(
                "Before.\n\n{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo partial done{}\n{}\n{}",
                DSML_TOOL_CALLS_START,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_TOOL_CALLS_END
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_partial_tool",
                &mut st,
                raw2.as_bytes(),
                false,
            );
            openai_sse_finish_live(
                &mut w,
                &r,
                "chatcmpl_partial_tool",
                &mut st,
                raw2.as_bytes(),
                "tool_calls",
                10,
                4,
                &[],
            );
            w.out
        }
        "openai-raw" => {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let raw = format!(
                "{}\n{} name=\"edit\">\n{} name=\"edits\" string=\"false\">[1,2,3",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_raw_tool",
                &mut st,
                raw.as_bytes(),
                false,
            );
            w.out
        }
        "openai-wait-tag" => {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let raw1 = format!("{}\n{}", DSML_TOOL_CALLS_START, DSML_INVOKE_START);
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_incomplete_tool",
                &mut st,
                raw1.as_bytes(),
                false,
            );
            let raw2 = format!(
                "{}\n{} name=\"bash\">\n{}",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_incomplete_tool",
                &mut st,
                raw2.as_bytes(),
                false,
            );
            w.out
        }
        "openai-entity" => {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let raw1 = format!(
                "{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo &amp",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_entity_tool",
                &mut st,
                raw1.as_bytes(),
                false,
            );
            let raw2 = format!(
                "{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo &amp; done{}\n{}\n{}",
                DSML_TOOL_CALLS_START,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_TOOL_CALLS_END
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_entity_tool",
                &mut st,
                raw2.as_bytes(),
                false,
            );
            w.out
        }
        "openai-think-tool" => {
            r.think_mode = ThinkMode::Low;
            let mut w = Writer::new(CREATED_TEST);
            sse_chunk(&mut w, &r, "chatcmpl_test", None, None);
            let mut st = openai_stream_start(&r);
            let raw1 = "<think>need a tool</think>Hello.\n\n";
            openai_sse_stream_update(&mut w, &r, "chatcmpl_test", &mut st, raw1.as_bytes(), false);
            let raw2 = format!(
                "<think>need a tool</think>Hello.\n\n{}\n",
                DSML_TOOL_CALLS_START
            );
            openai_sse_stream_update(&mut w, &r, "chatcmpl_test", &mut st, raw2.as_bytes(), false);
            openai_sse_finish_live(
                &mut w,
                &r,
                "chatcmpl_test",
                &mut st,
                raw2.as_bytes(),
                "tool_calls",
                10,
                8,
                &swapped_bash,
            );
            w.out
        }
        "anthropic-partial" => {
            r.api = Api::Anthropic;
            let mut w = Writer::new(CREATED_TEST);
            let mut st = anthropic_sse_start_live(&mut w, &r, "msg_tool", 7);
            let raw1 = format!(
                "Before.\n\n{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo partial",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            );
            anthropic_sse_stream_update(&mut w, &r, "msg_tool", &mut st, raw1.as_bytes(), false);
            let raw2 = format!(
                "Before.\n\n{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">echo partial done{}\n{}\n{}",
                DSML_TOOL_CALLS_START,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_TOOL_CALLS_END
            );
            anthropic_sse_stream_update(&mut w, &r, "msg_tool", &mut st, raw2.as_bytes(), false);
            anthropic_sse_finish_live(
                &mut w,
                &r,
                "msg_tool",
                &mut st,
                raw2.as_bytes(),
                "tool_calls",
                None,
                5,
                &[],
            );
            w.out
        }
        "anthropic-think-tool" => {
            r.api = Api::Anthropic;
            r.think_mode = ThinkMode::Low;
            let mut w = Writer::new(CREATED_TEST);
            let mut st = anthropic_sse_start_live(&mut w, &r, "msg_test", 10);
            let raw1 = "need a tool</think>Hello.\n\n";
            anthropic_sse_stream_update(&mut w, &r, "msg_test", &mut st, raw1.as_bytes(), false);
            let raw2 = format!("need a tool</think>Hello.\n\n{}\n", DSML_TOOL_CALLS_START);
            anthropic_sse_stream_update(&mut w, &r, "msg_test", &mut st, raw2.as_bytes(), false);
            anthropic_sse_finish_live(
                &mut w,
                &r,
                "msg_test",
                &mut st,
                raw2.as_bytes(),
                "tool_calls",
                None,
                8,
                &swapped_bash,
            );
            w.out
        }
        "openai-utf8" => {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let mut raw1 = format!(
                "{}\n{} name=\"write\">\n{} name=\"content\" string=\"true\">flag ",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            )
            .into_bytes();
            raw1.extend_from_slice(&[0xf0, 0x9f]);
            openai_sse_stream_update(&mut w, &r, "chatcmpl_utf8_tool", &mut st, &raw1, false);
            let mut raw2 = format!(
                "{}\n{} name=\"write\">\n{} name=\"content\" string=\"true\">flag ",
                DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
            )
            .into_bytes();
            raw2.extend_from_slice(&[0xf0, 0x9f, 0x9a, 0xa9]);
            raw2.extend_from_slice(
                format!(
                    " done{}\n{}\n{}",
                    DSML_PARAM_END, DSML_INVOKE_END, DSML_TOOL_CALLS_END
                )
                .as_bytes(),
            );
            openai_sse_stream_update(&mut w, &r, "chatcmpl_utf8_tool", &mut st, &raw2, false);
            w.out
        }
        "openai-multi" => {
            let mut w = Writer::new(CREATED_TEST);
            let mut st = openai_stream_start(&r);
            let raw = format!(
                "{}\n{} name=\"read\">\n{} name=\"path\" string=\"true\">a.c{}\n{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">wc -l a.c{}\n{}\n{}",
                DSML_TOOL_CALLS_START,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_TOOL_CALLS_END
            );
            openai_sse_stream_update(
                &mut w,
                &r,
                "chatcmpl_multi_tool",
                &mut st,
                raw.as_bytes(),
                false,
            );
            w.out
        }
        _ => b"ERROR unknown-script\n".to_vec(),
    }
}

#[cfg(test)]
mod id_tests {
    use super::{DsmlToolStream, DSML_TOOL_CALLS_START};
    use crate::parse::ToolCall;

    #[test]
    fn random_stream_ids_are_unique_and_inherited_by_final_calls() {
        let mut first = DsmlToolStream::with_prefix("call_");
        first.use_random_ids();
        assert!(first.init(DSML_TOOL_CALLS_START.as_bytes(), 0));
        let id = first.id_at(0);
        assert_eq!(id.len(), 37);
        assert!(id.starts_with("call_"));
        assert!(id[5..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));

        let mut calls = vec![ToolCall::default()];
        first.apply_ids(&mut calls);
        assert_eq!(calls[0].id, id);

        let mut second = DsmlToolStream::with_prefix("call_");
        second.use_random_ids();
        assert_ne!(second.id_at(0), id);
    }
}
