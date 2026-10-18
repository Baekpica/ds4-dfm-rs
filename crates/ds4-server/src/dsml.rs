//! DSML decode tracker + `sem_accum_sampling_override` from `ds4_server.c`
//! at v0.6.3-dfm. Structural tool syntax is greedy; payload keeps the
//! request sampler. Required tool / think-end prefixes force exact tokens.

use crate::parse::ToolChoice;
use crate::route::ThinkMode;
use crate::stream::ChatFormat;
use crate::tool_stream::DsmlSyntax;
use crate::tools::{
    dsml_attr, DSML_INVOKE_END, DSML_INVOKE_END_SHORT, DSML_INVOKE_END_SPACED, DSML_INVOKE_START,
    DSML_INVOKE_START_SHORT, DSML_INVOKE_START_SPACED, DSML_PARAM_END, DSML_PARAM_END_SHORT,
    DSML_PARAM_END_SPACED, DSML_PARAM_START, DSML_PARAM_START_SHORT, DSML_PARAM_START_SPACED,
    DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SHORT, DSML_TOOL_CALLS_END_SPACED,
    DSML_TOOL_CALLS_START, DSML_TOOL_CALLS_START_SHORT, DSML_TOOL_CALLS_START_SPACED,
};

pub const SAMPLE_OVERRIDE_NONE: i32 = 0;
pub const SAMPLE_OVERRIDE_GREEDY: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsmlDecodeState {
    Outside = 0,
    Structural = 1,
    StringBody = 2,
    JsonStructural = 3,
    JsonString = 4,
}

impl DsmlDecodeState {
    pub fn is_tool(self) -> bool {
        self != DsmlDecodeState::Outside
    }
    pub fn uses_payload_sampling(self) -> bool {
        matches!(
            self,
            DsmlDecodeState::StringBody | DsmlDecodeState::JsonString
        )
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Outside => "outside",
            Self::Structural => "structural",
            Self::StringBody => "string-body",
            Self::JsonStructural => "json-structural",
            Self::JsonString => "json-string",
        }
    }
}

/// The engine's dsml_decode_state_is_syntax (server_generate_v41.c): under
/// sampling the tool-call syntax positions (the tag / parameter head / JSON
/// punctuation) take the raw argmax while the string value body and the JSON
/// string keep the request's draw.  True = greedy at this position.
pub fn dsml_decode_state_is_syntax(state: DsmlDecodeState) -> bool {
    !matches!(
        state,
        DsmlDecodeState::Outside | DsmlDecodeState::StringBody | DsmlDecodeState::JsonString
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackMode {
    Search,
    Structural,
    StringBody,
    JsonParam,
    Done,
}

/// The tool dialect the tracker recognizes: C `dsml_decode_tracker::model_syntax`
/// (ds4_server.c:7364).  GLM's `<tool_call>/<arg_key>/<arg_value>` block is
/// deliberately NOT a member of the DSML syntax table; the reference selects it
/// by model syntax (ds4_server.c:7595-7611) so the dialects never recognize each
/// other's bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolSyntax {
    #[default]
    Dsml,
    Glm,
}

/// GLM's tool block (ds4_server.c:7376 `glm_tool_syntax`): the function name
/// sits between `<tool_call>` and the first `<arg_key>`, and argument values are
/// text, so every value body is a string body.
const GLM_SYNTAX: DsmlSyntax = DsmlSyntax {
    tool_calls_start: b"<tool_call>",
    tool_calls_end: b"</tool_call>",
    invoke_start: b"<arg_key>",
    invoke_end: b"</arg_key>",
    param_start: b"<arg_value>",
    param_end: b"</arg_value>",
    entity_values: false,
};

const SYNTAXES: [DsmlSyntax; 4] = [
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
    // The official V4.1 encoding.py spelling (4294136): the block name and the
    // tag names open with a space (engine server_types.h DS4_*_SPACED).
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
pub struct DsmlDecodeTracker {
    mode: TrackMode,
    pub decode: DsmlDecodeState,
    syn: Option<DsmlSyntax>,
    pos: usize,
    json_in_string: bool,
    json_escaped: bool,
    syntax: ToolSyntax,
}

impl Default for DsmlDecodeTracker {
    fn default() -> Self {
        Self {
            mode: TrackMode::Search,
            decode: DsmlDecodeState::Outside,
            syn: None,
            pos: 0,
            json_in_string: false,
            json_escaped: false,
            syntax: ToolSyntax::Dsml,
        }
    }
}

impl DsmlDecodeTracker {
    /// C: `tracker.model_syntax = ...`, assigned before the first update
    /// (ds4_server.c:20591).  A dialect must be chosen before the block opens:
    /// the start search is dialect-specific.
    pub fn set_tool_syntax(&mut self, syntax: ToolSyntax) {
        self.syntax = syntax;
    }

    /// The `<parameter ` / `</parameter>` bare spellings (f302be7) belong to the
    /// DSML dialects; GLM's block has `<arg_key>`/`<arg_value>` only, so its
    /// dialect must not accept the aliases.
    fn alias(&self) -> bool {
        self.syntax == ToolSyntax::Dsml
    }

    /// C `tracker->syn` (ds4_server.c:7758): a tool block has been opened since
    /// the last reset.
    pub fn block_started(&self) -> bool {
        self.syn.is_some()
    }

    /// The dialect the tracker was told to recognize.
    pub fn tool_syntax(&self) -> ToolSyntax {
        self.syntax
    }

    /// C `tracker->mode == DSML_TRACK_DONE` (ds4_server.c:7759): the block that
    /// was opened has been closed by its own end marker.
    pub fn block_closed(&self) -> bool {
        self.mode == TrackMode::Done
    }

    pub fn update(&mut self, raw: &[u8]) {
        loop {
            if self.mode == TrackMode::Done {
                self.decode = DsmlDecodeState::Outside;
                return;
            }
            if self.mode == TrackMode::Search {
                let found = match self.syntax {
                    // The GLM dialect searches only its own start marker
                    // (ds4_server.c:7595-7611): `<tool_call>` is not one of the
                    // DSML spellings, and the DSML list must not see GLM bytes.
                    ToolSyntax::Glm => {
                        let from = self.pos.min(raw.len());
                        find_lit_bounded(&raw[from..], GLM_SYNTAX.tool_calls_start)
                            .map(|rel| (from + rel + GLM_SYNTAX.tool_calls_start.len(), GLM_SYNTAX))
                    }
                    ToolSyntax::Dsml => dsml_find_tool_start_from(raw, self.pos),
                };
                match found {
                    None => {
                        let hold = dsml_max_tool_start_len();
                        self.pos = if raw.len() > hold {
                            raw.len() - hold
                        } else {
                            0
                        };
                        self.decode = DsmlDecodeState::Outside;
                        return;
                    }
                    Some((pos, syn)) => {
                        self.syn = Some(syn);
                        self.pos = pos;
                        self.mode = TrackMode::Structural;
                        self.decode = DsmlDecodeState::Structural;
                    }
                }
            }
            if self.mode == TrackMode::StringBody {
                let syn = self.syn.unwrap();
                while self.pos < raw.len() {
                    let end_len = param_close_len(&syn, raw, self.pos, self.alias());
                    if end_len > 0 {
                        self.pos += end_len;
                        self.mode = TrackMode::Structural;
                        self.decode = DsmlDecodeState::Structural;
                        break;
                    }
                    if param_close_partial_any(&syn, raw, self.pos, self.alias()) {
                        // A tail that is a proper prefix of the close spelling
                        // holds the position: the stream may cut the marker
                        // between two tokens, and the next update re-reads it
                        // (ds4_server.c:7635-7641).  A lone '<' is payload.
                        self.decode = if raw.len() - self.pos >= 2 {
                            DsmlDecodeState::Structural
                        } else {
                            DsmlDecodeState::StringBody
                        };
                        return;
                    }
                    self.pos += 1;
                }
                if self.mode == TrackMode::StringBody {
                    self.decode = DsmlDecodeState::StringBody;
                    return;
                }
            }
            if self.mode == TrackMode::JsonParam {
                let syn = self.syn.unwrap();
                while self.pos < raw.len() {
                    if !self.json_in_string {
                        let end_len = param_close_len(&syn, raw, self.pos, self.alias());
                        if end_len > 0 {
                            self.pos += end_len;
                            self.mode = TrackMode::Structural;
                            self.decode = DsmlDecodeState::Structural;
                            break;
                        }
                        if param_close_partial(&syn, raw, self.pos, 2, self.alias()) {
                            self.decode = DsmlDecodeState::Structural;
                            return;
                        }
                    }
                    let c = raw[self.pos];
                    self.pos += 1;
                    if self.json_in_string {
                        if self.json_escaped {
                            self.json_escaped = false;
                        } else if c == b'\\' {
                            self.json_escaped = true;
                        } else if c == b'"' {
                            self.json_in_string = false;
                        }
                    } else if c == b'"' {
                        self.json_in_string = true;
                    }
                }
                if self.mode == TrackMode::JsonParam {
                    self.decode = if self.json_in_string {
                        DsmlDecodeState::JsonString
                    } else {
                        DsmlDecodeState::JsonStructural
                    };
                    return;
                }
            }
            while self.mode == TrackMode::Structural {
                let syn = self.syn.unwrap();
                while self.pos < raw.len() && raw[self.pos].is_ascii_whitespace() {
                    self.pos += 1;
                }
                if self.pos >= raw.len() {
                    self.decode = DsmlDecodeState::Structural;
                    return;
                }
                if raw_full_lit(raw, self.pos, syn.tool_calls_end) {
                    self.mode = TrackMode::Done;
                    self.pos += syn.tool_calls_end.len();
                    self.decode = DsmlDecodeState::Outside;
                    return;
                }
                if raw_full_lit(raw, self.pos, syn.invoke_end) {
                    self.pos += syn.invoke_end.len();
                    continue;
                }
                if raw_full_lit(raw, self.pos, syn.invoke_start) {
                    match raw[self.pos..].iter().position(|&c| c == b'>') {
                        None => {
                            self.decode = DsmlDecodeState::Structural;
                            return;
                        }
                        Some(rel) => {
                            self.pos += rel + 1;
                            continue;
                        }
                    }
                }
                if param_open_full(&syn, raw, self.pos, self.alias()) {
                    let tag_start = self.pos;
                    let Some(rel) = raw[self.pos..].iter().position(|&c| c == b'>') else {
                        self.decode = DsmlDecodeState::Structural;
                        return;
                    };
                    let tag_after = self.pos + rel + 1;
                    // GLM argument values carry no type attribute: every value
                    // body is text (ds4_server.c:7731-7739 over glm_tool_syntax).
                    let string_value = self.syntax == ToolSyntax::Glm
                        || dsml_attr_is_string_true(raw, tag_start, tag_after);
                    self.pos = tag_after;
                    if string_value {
                        self.mode = TrackMode::StringBody;
                        self.decode = DsmlDecodeState::StringBody;
                    } else {
                        self.mode = TrackMode::JsonParam;
                        self.json_in_string = false;
                        self.json_escaped = false;
                        self.decode = DsmlDecodeState::JsonStructural;
                    }
                    break;
                }
                if raw_partial_lit(raw, self.pos, syn.tool_calls_end)
                    || raw_partial_lit(raw, self.pos, syn.invoke_start)
                    || raw_partial_lit(raw, self.pos, syn.invoke_end)
                    || param_open_partial(&syn, raw, self.pos, self.alias())
                    || param_close_partial(&syn, raw, self.pos, 1, self.alias())
                {
                    self.decode = DsmlDecodeState::Structural;
                    return;
                }
                if self.syntax == ToolSyntax::Glm {
                    // GLM puts the function and argument names between tags;
                    // unknown structural bytes advance (ds4_server.c:7746-7752).
                    self.pos += 1;
                    continue;
                }
                self.decode = DsmlDecodeState::Structural;
                return;
            }
        }
    }
}

fn dsml_attr_is_string_true(raw: &[u8], tag_start: usize, tag_end: usize) -> bool {
    if tag_end <= tag_start || tag_end > raw.len() {
        return false;
    }
    dsml_attr(&raw[tag_start..tag_end], "string").is_some_and(|v| v == b"true")
}

fn dsml_max_tool_start_len() -> usize {
    SYNTAXES
        .iter()
        .map(|s| s.tool_calls_start.len())
        .max()
        .unwrap_or(0)
}

fn dsml_find_tool_start(raw: &[u8]) -> Option<(usize, DsmlSyntax)> {
    let mut best: Option<(usize, DsmlSyntax)> = None;
    for syn in SYNTAXES {
        if let Some(i) = find_lit_bounded(raw, syn.tool_calls_start) {
            if best.map(|(p, _)| i < p).unwrap_or(true) {
                best = Some((i + syn.tool_calls_start.len(), syn));
            }
        }
    }
    best
}

fn dsml_find_tool_start_from(raw: &[u8], start: usize) -> Option<(usize, DsmlSyntax)> {
    if start > raw.len() {
        return None;
    }
    dsml_find_tool_start(&raw[start..]).map(|(rel, syn)| (start + rel, syn))
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

fn raw_partial_lit_min(raw: &[u8], pos: usize, lit: &[u8], min_len: usize) -> bool {
    if pos > raw.len() || raw.len() - pos >= lit.len() {
        return false;
    }
    let avail = raw.len() - pos;
    avail >= min_len && raw[pos..] == lit[..avail]
}

fn find_lit_bounded(s: &[u8], lit: &[u8]) -> Option<usize> {
    if lit.is_empty() {
        return Some(0);
    }
    s.windows(lit.len()).position(|w| w == lit)
}

/// The engine's bare-parameter aliases (f302be7, server_dsml_parse.c:5-47): a
/// decorated or spaced block also accepts the plain spellings, because the
/// artifact mixes them inside one block (decorated invoke, bare parameters).
/// The plain syntax already is the bare form, so the alias never stacks on it.
/// The open alias carries the trailing space: "<parameters>" is not an open.
const BARE_PARAM_START: &[u8] = b"<parameter ";
const BARE_PARAM_END: &[u8] = b"</parameter>";

fn bare_param_alias(syn: &DsmlSyntax) -> bool {
    syn.param_start != &b"<parameter"[..]
}

fn param_open_full(syn: &DsmlSyntax, raw: &[u8], pos: usize, alias: bool) -> bool {
    raw_full_lit(raw, pos, syn.param_start) || (alias && raw_full_lit(raw, pos, BARE_PARAM_START))
}

fn param_open_partial(syn: &DsmlSyntax, raw: &[u8], pos: usize, alias: bool) -> bool {
    raw_partial_lit(raw, pos, syn.param_start)
        || (alias && raw_partial_lit(raw, pos, BARE_PARAM_START))
}

/// Length of the close at pos in either spelling (f302be7,
/// server_dsml_parse.c:31-38); 0 when neither matches.
fn param_close_len(syn: &DsmlSyntax, raw: &[u8], pos: usize, alias: bool) -> usize {
    if raw_full_lit(raw, pos, syn.param_end) {
        return syn.param_end.len();
    }
    if alias && raw_full_lit(raw, pos, BARE_PARAM_END) {
        return BARE_PARAM_END.len();
    }
    0
}

fn param_close_partial(
    syn: &DsmlSyntax,
    raw: &[u8],
    pos: usize,
    min_len: usize,
    alias: bool,
) -> bool {
    raw_partial_lit_min(raw, pos, syn.param_end, min_len)
        || (alias && raw_partial_lit_min(raw, pos, BARE_PARAM_END, min_len))
}

/// Any proper prefix of the close spelling (C `raw_partial_lit`, ds4_server.c:
/// 7636): the tracker holds its position on one, so a marker cut between tokens
/// is re-read by the next update.  `param_close_partial` is the >= min_len form
/// used where only the decoded state matters.
fn param_close_partial_any(syn: &DsmlSyntax, raw: &[u8], pos: usize, alias: bool) -> bool {
    raw_partial_lit(raw, pos, syn.param_end) || (alias && raw_partial_lit(raw, pos, BARE_PARAM_END))
}

/// Earliest close in either spelling from `from`, as (offset, len)
/// (f302be7, server_dsml_parse.c:42-47).
fn param_close_find(syn: &DsmlSyntax, raw: &[u8], from: usize) -> Option<(usize, usize)> {
    let rest = &raw[from..];
    let own = find_lit_bounded(rest, syn.param_end).map(|r| (r, syn.param_end.len()));
    if !bare_param_alias(syn) {
        return own;
    }
    let bare = find_lit_bounded(&rest[..own.map_or(rest.len(), |(r, _)| r)], BARE_PARAM_END)
        .map(|r| (r, BARE_PARAM_END.len()));
    bare.or(own)
}

/// The buffer tail is a proper prefix of either close spelling (>= min_len
/// bytes): the stream holds it back until the next token proves it (f302be7,
/// server_dsml_stream.c:104-110).
fn raw_suffix_partial_close(syn: &DsmlSyntax, raw: &[u8], min_len: usize, alias: bool) -> bool {
    let mut n = min_len;
    while n <= raw.len() && n < 32 {
        if param_close_partial(syn, raw, raw.len() - n, min_len, alias) {
            return true;
        }
        n += 1;
    }
    false
}

fn dsml_decode_scan_json_param(
    raw: &[u8],
    mut pos: usize,
    syn: DsmlSyntax,
    alias: bool,
) -> DsmlDecodeState {
    let mut in_string = false;
    let mut escaped = false;
    while pos < raw.len() {
        if !in_string && param_close_len(&syn, raw, pos, alias) > 0 {
            return DsmlDecodeState::Structural;
        }
        let c = raw[pos];
        pos += 1;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
        } else if c == b'"' {
            in_string = true;
        }
    }
    if !in_string && raw_suffix_partial_close(&syn, raw, 2, bare_param_alias(&syn)) {
        return DsmlDecodeState::Structural;
    }
    if in_string {
        DsmlDecodeState::JsonString
    } else {
        DsmlDecodeState::JsonStructural
    }
}

/// Slow reference recognizer used by C tests (`dsml_decode_state_for_text`).
pub fn dsml_decode_state_for_text(raw: &[u8]) -> DsmlDecodeState {
    if raw.is_empty() {
        return DsmlDecodeState::Outside;
    }
    let Some((mut pos, syn)) = dsml_find_tool_start(raw) else {
        return DsmlDecodeState::Outside;
    };
    loop {
        while pos < raw.len() && raw[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= raw.len() {
            return DsmlDecodeState::Structural;
        }
        if raw_full_lit(raw, pos, syn.tool_calls_end) {
            return DsmlDecodeState::Outside;
        }
        if raw_full_lit(raw, pos, syn.invoke_end) {
            pos += syn.invoke_end.len();
            continue;
        }
        if raw_full_lit(raw, pos, syn.invoke_start) {
            match raw[pos..].iter().position(|&c| c == b'>') {
                None => return DsmlDecodeState::Structural,
                Some(rel) => {
                    pos += rel + 1;
                    continue;
                }
            }
        }
        if param_open_full(&syn, raw, pos, bare_param_alias(&syn)) {
            let tag_start = pos;
            let Some(rel) = raw[pos..].iter().position(|&c| c == b'>') else {
                return DsmlDecodeState::Structural;
            };
            let tag_end = pos + rel + 1;
            let string_value = dsml_attr_is_string_true(raw, tag_start, tag_end);
            pos = tag_end;
            if string_value {
                match param_close_find(&syn, raw, pos) {
                    None => {
                        if raw_suffix_partial_close(&syn, raw, 2, bare_param_alias(&syn)) {
                            return DsmlDecodeState::Structural;
                        }
                        return DsmlDecodeState::StringBody;
                    }
                    Some((end, end_len)) => {
                        pos += end + end_len;
                        continue;
                    }
                }
            }
            let json_state = dsml_decode_scan_json_param(raw, pos, syn, bare_param_alias(&syn));
            if json_state == DsmlDecodeState::Structural {
                match param_close_find(&syn, raw, pos) {
                    None => return DsmlDecodeState::Structural,
                    Some((end, end_len)) => {
                        pos += end + end_len;
                        continue;
                    }
                }
            }
            return json_state;
        }
        for s in SYNTAXES {
            if raw_partial_lit(raw, pos, s.tool_calls_end)
                || raw_partial_lit(raw, pos, s.invoke_start)
                || raw_partial_lit(raw, pos, s.invoke_end)
                || raw_partial_lit(raw, pos, s.param_start)
                || raw_partial_lit(raw, pos, s.param_end)
            {
                return DsmlDecodeState::Structural;
            }
        }
        return DsmlDecodeState::Structural;
    }
}

pub fn agent_turn_reasoning_cap(think_mode: ThinkMode, max_tokens: i32) -> i32 {
    let reserve = 64;
    let budget_cap = if max_tokens > reserve {
        max_tokens - reserve
    } else {
        0
    };
    let effort_cap = match think_mode {
        ThinkMode::High => 256,
        ThinkMode::Max => 512,
        _ => 64,
    };
    budget_cap.min(effort_cap)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleOverride {
    None,
    Greedy,
    Token(i32),
}

impl SampleOverride {
    pub fn as_c_int(self) -> i32 {
        match self {
            Self::None => SAMPLE_OVERRIDE_NONE,
            Self::Greedy => SAMPLE_OVERRIDE_GREEDY,
            Self::Token(t) => t + 2,
        }
    }
}

pub struct SamplePolicy<'a> {
    pub tool_choice: ToolChoice,
    pub has_tool_results: bool,
    pub think_mode: ThinkMode,
    pub max_tokens: i32,
    pub required_tool_prefix: &'a [i32],
    pub required_think_end_prefix: &'a [i32],
}

pub fn sampling_override(
    track_tools: bool,
    saw_tool_start: bool,
    thinking_inside: bool,
    completion: i32,
    chat_format: ChatFormat,
    dsml: DsmlDecodeState,
    tool_pos: &mut i32,
    think_pos: &mut i32,
    p: &SamplePolicy<'_>,
) -> SampleOverride {
    let prefix_in_flight = *tool_pos > 0 && (*tool_pos as usize) < p.required_tool_prefix.len();
    let required_pending = p.tool_choice == ToolChoice::Required
        && track_tools
        && (!saw_tool_start || prefix_in_flight);
    // MiMo auto chooses the marker; only then complete its XML head.
    let auto_body_pending = p.tool_choice == ToolChoice::Auto
        && track_tools
        && saw_tool_start
        && !thinking_inside
        && (*tool_pos as usize) < p.required_tool_prefix.len();
    let reserve_post_thinking = required_pending || p.has_tool_results;
    if reserve_post_thinking && thinking_inside {
        if completion >= agent_turn_reasoning_cap(p.think_mode, p.max_tokens)
            && (*think_pos as usize) < p.required_think_end_prefix.len()
        {
            let token = p.required_think_end_prefix[*think_pos as usize];
            *think_pos += 1;
            return SampleOverride::Token(token);
        }
        return SampleOverride::None;
    }
    if (required_pending || auto_body_pending)
        && (*tool_pos as usize) < p.required_tool_prefix.len()
    {
        let token = p.required_tool_prefix[*tool_pos as usize];
        *tool_pos += 1;
        return SampleOverride::Token(token);
    }
    let state = if track_tools && chat_format == ChatFormat::DeepSeek {
        dsml
    } else {
        DsmlDecodeState::Outside
    };
    if state.is_tool() && !state.uses_payload_sampling() {
        return SampleOverride::Greedy;
    }
    SampleOverride::None
}

pub fn dump_script(name: &str) -> String {
    match name {
        "state-prefix" => dump_state(state_prefix()),
        "state-path-param" => dump_state(state_path_param()),
        "state-path-closing" => dump_state(state_path_closing()),
        "state-json-struct" => dump_state(state_json_struct()),
        "state-json-string" => dump_state(state_json_string()),
        "state-done" => dump_state(state_done()),
        "override-required" => dump_override_required(),
        "override-think-cap" => dump_override_think_cap(),
        "override-tool-result" => dump_override_tool_result(),
        "cap-low-128" => format!("{}\n", agent_turn_reasoning_cap(ThinkMode::Low, 128)),
        "cap-high-128" => format!("{}\n", agent_turn_reasoning_cap(ThinkMode::High, 128)),
        "cap-max-600" => format!("{}\n", agent_turn_reasoning_cap(ThinkMode::Max, 600)),
        "glm-boundaries-token" => dump_glm_boundaries(false),
        "glm-boundaries-byte" => dump_glm_boundaries(true),
        "glm-control-text" => dump_glm_control_text(),
        _ => "ERROR unknown-script\n".into(),
    }
}

/// The reference's GLM fragments, ds4_server.c:20572
/// (`test_glm_decode_tracker_boundaries` in /data/ds4).
fn glm_parts() -> [&'static str; 7] {
    [
        "prose <tool_call>bash<arg_key>command</arg_key>",
        "<arg_value>printf '<html>'",
        "<",
        "/arg_",
        "value>",
        "<arg_key>description</arg_key><arg_value>test",
        "</arg_value></tool_call>",
    ]
}

/// Feed the accumulated text fragment by fragment against the GLM dialect and
/// print the state after each one, so a dialect drift shows as a tape line.
fn dump_glm_boundaries(bytewise: bool) -> String {
    let mut tracker = DsmlDecodeTracker::default();
    tracker.set_tool_syntax(ToolSyntax::Glm);
    let mut raw = Vec::new();
    let mut states = Vec::new();
    for part in glm_parts() {
        if bytewise {
            for byte in part.bytes() {
                raw.push(byte);
                tracker.update(&raw);
            }
        } else {
            raw.extend_from_slice(part.as_bytes());
            tracker.update(&raw);
        }
        states.push(tracker.decode.name());
    }
    format!("states={}\n", states.join(","))
}

/// ds4_server.c:20608 (`test_tool_control_text_inside_arguments`): control
/// markers written into an argument value are payload, so the block stays open.
fn dump_glm_control_text() -> String {
    let raw = format!(
        "<think>reason</think><tool_call>write<arg_key>content</arg_key><arg_value>\
         literal </tool_call> {DSML_TOOL_CALLS_END} </think> <think>"
    );
    let mut tracker = DsmlDecodeTracker::default();
    tracker.set_tool_syntax(ToolSyntax::Glm);
    for n in 1..=raw.len() {
        tracker.update(&raw.as_bytes()[..n]);
    }
    format!(
        "tracker={} start={} end={}\n",
        tracker.decode.name(),
        u8::from(tracker.block_started()),
        u8::from(tracker.block_closed())
    )
}

fn state_prefix() -> String {
    format!(
        "{}\n{} name=\"edit\">\n",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START
    )
}
fn state_path_param() -> String {
    format!(
        "{}\n{} name=\"edit\">\n{} name=\"path\" string=\"true\">/tmp/a.py",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
    )
}
fn state_path_closing() -> String {
    format!(
        "{}\n{} name=\"edit\">\n{} name=\"path\" string=\"true\">/tmp/a.py</",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
    )
}
fn state_json_struct() -> String {
    format!(
        "{}\n{} name=\"edit\">\n{} name=\"edits\" string=\"false\">[{{",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
    )
}
fn state_json_string() -> String {
    format!(
        "{}\n{} name=\"edit\">\n{} name=\"edits\" string=\"false\">[{{\"newText\":\"for i in",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
    )
}
fn state_done() -> String {
    format!(
        "{}\n{} name=\"edit\">\n{} name=\"edits\" string=\"false\">[]{}\n{}\n{}",
        DSML_TOOL_CALLS_START,
        DSML_INVOKE_START,
        DSML_PARAM_START,
        DSML_PARAM_END,
        DSML_INVOKE_END,
        DSML_TOOL_CALLS_END
    )
}

fn dump_state(raw: String) -> String {
    let ref_state = dsml_decode_state_for_text(raw.as_bytes());
    let mut tr = DsmlDecodeTracker::default();
    tr.update(raw.as_bytes());
    format!("ref={} tracker={}\n", ref_state.name(), tr.decode.name())
}

fn dump_override_required() -> String {
    let p = SamplePolicy {
        tool_choice: ToolChoice::Required,
        has_tool_results: false,
        think_mode: ThinkMode::None,
        max_tokens: 128,
        required_tool_prefix: &[101, 202],
        required_think_end_prefix: &[],
    };
    let mut tool_pos = 0;
    let mut think_pos = 0;
    let a = sampling_override(
        true,
        false,
        false,
        0,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    let b = sampling_override(
        true,
        false,
        false,
        0,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    format!("{} {}\n", a.as_c_int(), b.as_c_int())
}

fn dump_override_think_cap() -> String {
    let p = SamplePolicy {
        tool_choice: ToolChoice::Required,
        has_tool_results: false,
        think_mode: ThinkMode::Low,
        max_tokens: 128,
        required_tool_prefix: &[],
        required_think_end_prefix: &[303],
    };
    let mut tool_pos = 0;
    let mut think_pos = 0;
    let none = sampling_override(
        true,
        false,
        true,
        0,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    let forced = sampling_override(
        true,
        false,
        true,
        64,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    format!("{} {}\n", none.as_c_int(), forced.as_c_int())
}

fn dump_override_tool_result() -> String {
    let p = SamplePolicy {
        tool_choice: ToolChoice::Auto,
        has_tool_results: true,
        think_mode: ThinkMode::Low,
        max_tokens: 128,
        required_tool_prefix: &[],
        required_think_end_prefix: &[303],
    };
    let mut tool_pos = 0;
    let mut think_pos = 0;
    let a = sampling_override(
        false,
        false,
        true,
        63,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    let b = sampling_override(
        false,
        false,
        true,
        64,
        ChatFormat::DeepSeek,
        DsmlDecodeState::Outside,
        &mut tool_pos,
        &mut think_pos,
        &p,
    );
    format!("{} {}\n", a.as_c_int(), b.as_c_int())
}

#[cfg(test)]
mod override_tests {
    use super::{sampling_override, DsmlDecodeState, SampleOverride, SamplePolicy};
    use crate::parse::ToolChoice;
    use crate::route::ThinkMode;
    use crate::stream::ChatFormat;

    #[test]
    fn required_prefix_finishes_after_tool_start() {
        let policy = SamplePolicy {
            tool_choice: ToolChoice::Required,
            has_tool_results: false,
            think_mode: ThinkMode::None,
            max_tokens: 128,
            required_tool_prefix: &[101, 202],
            required_think_end_prefix: &[],
        };
        let mut tool_pos = 0;
        let mut think_pos = 0;
        let mut sample = |saw_tool_start| {
            sampling_override(
                true,
                saw_tool_start,
                false,
                tool_pos,
                ChatFormat::Qwen4Exp,
                DsmlDecodeState::Outside,
                &mut tool_pos,
                &mut think_pos,
                &policy,
            )
        };
        assert_eq!(sample(false), SampleOverride::Token(101));
        assert_eq!(sample(true), SampleOverride::Token(202));
    }

    #[test]
    fn auto_tool_start_forces_function_prefix() {
        let policy = SamplePolicy {
            tool_choice: ToolChoice::Auto,
            has_tool_results: false,
            think_mode: ThinkMode::None,
            max_tokens: 128,
            required_tool_prefix: &[202],
            required_think_end_prefix: &[],
        };
        let mut tool_pos = 0;
        let mut think_pos = 0;
        let mut sample = |saw_tool_start| {
            sampling_override(
                true,
                saw_tool_start,
                false,
                0,
                ChatFormat::Qwen4Exp,
                DsmlDecodeState::Outside,
                &mut tool_pos,
                &mut think_pos,
                &policy,
            )
        };
        assert_eq!(sample(false), SampleOverride::None);
        assert_eq!(sample(true), SampleOverride::Token(202));
        assert_eq!(sample(true), SampleOverride::None);
    }
}
