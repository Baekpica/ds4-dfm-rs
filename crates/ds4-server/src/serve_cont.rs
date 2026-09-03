//! Continuous-lane execution: a push-based per-token stepper (pure, tape
//! testable) plus the native `ContLane` that drives the engine's rolling
//! scheduler through `ds4-core::BatchCtx`. The accept loop admits another
//! request while one is generating; the persistent batch context owns every
//! configured bank. The engine-side bank/admit machinery is the same one C's
//! continuous lane uses. Corrective retry (`decode_again`) and continuation
//! publish route serial by the needs word, so this path never re-decodes.

use std::io::Write;
#[cfg(any(feature = "native", test))]
use std::path::Path;
use std::time::Instant;

use ds4_kv::Store as KvStore;
#[cfg(feature = "native")]
use ds4_kv::{bank_checkpoint_due_from_host, HostKvView};
#[cfg(any(feature = "native", test))]
use ds4_kv::{bank_persist_ext_flags, Reason as KvReason, EXT_IMAGE_PIXELS_V2};

use crate::dsml::{SampleOverride, SamplePolicy};
use crate::generate::{
    render_prompt, responses_ids, stream_req_from_parsed, GenerateError, GenerateOutcome,
};
use crate::parse::{ParsedRequest, ToolCall, ToolChoice};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
#[cfg(feature = "native")]
use crate::render::render_live_tool_tail;
use crate::render::syntax_for_model_id;
#[cfg(any(feature = "native", test))]
use crate::render::ModelSyntax;
use crate::retry::{terminal_finish, truncation_outcome, TruncationOutcome};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind, NEED_BANK_FRONTIER};
#[cfg(feature = "native")]
use crate::stream::stream_error;
#[cfg(any(feature = "native", test))]
use crate::stream::stream_heartbeat_if_due;
use crate::stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, responses_final_response, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init, sse_chunk,
    sse_done, sse_headers, AnthropicStream, OpenaiStream, ReqTimings, ResponsesStream, StreamReq,
    Writer,
};
#[cfg(any(feature = "native", test))]
use crate::stream::{think_end, think_start, ChatFormat};
use crate::tools::{assign_tool_ids, parse_generated_for_response, SemAccum};

#[cfg(any(feature = "native", test))]
const DEFAULT_BANK_PERSIST_MIN_TOKENS: i32 = 8_192;

#[cfg(any(feature = "native", test))]
fn bank_persist_eligible(committed: i32, persist_min: i32) -> bool {
    persist_min > 0 && committed >= persist_min
}

/// Pure continuous-request stepper. The caller feeds decoded pieces and
/// receives wire bytes; the engine (or a tape) owns token production.
/// The four frozen wire surfaces share this stepper; route eligibility still
/// decides which request shapes may enter the continuous lane.
pub struct ContStepper {
    pub req: StreamReq,
    pub job_id: String,
    pub model_id: i32,
    pub prompt: Vec<u8>,
    #[cfg(feature = "native")]
    cache_prompt: Option<Vec<u8>>,
    #[cfg(feature = "native")]
    image_cache_spans: Vec<ImageCacheSpan>,
    pub prompt_n: i32,
    pub max_tokens: i32,
    acc: SemAccum,
    w: Writer,
    oa: Option<OpenaiStream>,
    anth: Option<AnthropicStream>,
    resp: Option<ResponsesStream>,
    finish: &'static str,
    stops: Vec<String>,
    think_mode: crate::route::ThinkMode,
    tool_choice: ToolChoice,
    has_tool_results: bool,
    effective_usage_frame: bool,
    #[cfg(any(feature = "native", test))]
    cors: bool,
    #[cfg(feature = "native")]
    prompt_preserves_reasoning: bool,
    parsed_max_tokens: i32,
    required_tool_prefix: Vec<i32>,
    required_think_end_prefix: Vec<i32>,
    tool_replay: Option<(Vec<ToolCall>, String)>,
    started: bool,
    #[cfg(any(feature = "native", test))]
    last_heartbeat: Instant,
}

pub struct ContStep {
    pub bytes: Vec<u8>,
    /// Set when the host wants the sequence aborted (stop hit / tool block
    /// closed). The engine's own EOS/budget finish arrives via `finalize`.
    pub done: bool,
}

impl ContStepper {
    pub fn new(
        parsed: &ParsedRequest,
        model_id: i32,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        prompt: Vec<u8>,
        prompt_n: i32,
        seq_room: i32,
    ) -> (Self, Vec<u8>) {
        let req = stream_req_from_parsed(parsed, model_id);
        let mut max_tokens =
            decode_budget(parsed.max_tokens_set, parsed.max_tokens, default_tokens);
        let room = seq_room - prompt_n;
        if room >= 0 && max_tokens > room {
            max_tokens = room;
        }
        let acc = SemAccum::init(
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            think_mode_enabled(parsed.think_mode),
            req.chat_format,
            &prompt,
        );
        let mut w = Writer::new(created);
        let mut oa = None;
        let mut anth = None;
        let mut resp = None;
        if req.stream {
            w.out.extend_from_slice(&sse_headers(cors));
            match req.api {
                Api::Openai if req.kind == ReqKind::Chat => {
                    let mut st = openai_stream_start(&req);
                    st.tool.use_random_ids();
                    oa = Some(st);
                    sse_chunk(&mut w, &req, job_id, None, None);
                }
                Api::Anthropic => {
                    let mut st = anthropic_sse_start_live(&mut w, &req, job_id, prompt_n);
                    st.tool.use_random_ids();
                    anth = Some(st);
                }
                Api::Responses => {
                    let (response_id, reasoning_id, message_id) = responses_ids(job_id);
                    let mut st =
                        responses_stream_init(&req, &response_id, &reasoning_id, &message_id);
                    responses_sse_created(&mut w, &req, &mut st, created);
                    resp = Some(st);
                }
                Api::Openai => {}
            }
        }
        let head = std::mem::take(&mut w.out);
        (
            Self {
                req,
                job_id: job_id.to_string(),
                model_id,
                prompt,
                #[cfg(feature = "native")]
                cache_prompt: None,
                #[cfg(feature = "native")]
                image_cache_spans: Vec::new(),
                prompt_n,
                max_tokens,
                acc,
                w,
                oa,
                anth,
                resp,
                finish: "length",
                stops: parsed.stops.clone(),
                think_mode: parsed.think_mode,
                tool_choice: parsed.tool_choice,
                has_tool_results: parsed.has_tool_results,
                effective_usage_frame: parsed.needs & NEED_BANK_FRONTIER != 0,
                #[cfg(any(feature = "native", test))]
                cors,
                #[cfg(feature = "native")]
                prompt_preserves_reasoning: prompt_preserves_reasoning(parsed),
                parsed_max_tokens: parsed.max_tokens,
                required_tool_prefix: parsed.required_tool_prefix.clone(),
                required_think_end_prefix: parsed.required_think_end_prefix.clone(),
                tool_replay: None,
                started: true,
                #[cfg(any(feature = "native", test))]
                last_heartbeat: Instant::now(),
            },
            head,
        )
    }

    fn apply_engine_usage(&mut self, n_cached: i32, n_computed: i32) {
        let cached = n_cached.max(0);
        let computed = n_computed.max(0);
        if self.effective_usage_frame {
            self.prompt_n = cached.saturating_add(computed);
            self.req.cache_read_tokens = cached;
            self.req.cache_write_tokens = computed;
            return;
        }
        let prompt = self.prompt_n.max(0);
        let write = computed.min(prompt);
        self.req.cache_write_tokens = write;
        self.req.cache_read_tokens = prompt - write;
    }

    #[cfg(any(feature = "native", test))]
    fn admitted_head(&mut self, head: Vec<u8>, n_cached: i32, n_computed: i32) -> Vec<u8> {
        self.apply_engine_usage(n_cached, n_computed);
        if !self.req.stream || self.req.api != Api::Anthropic {
            return head;
        }
        let mut w = Writer::new(self.w.created);
        w.out.extend_from_slice(&sse_headers(self.cors));
        let mut stream = anthropic_sse_start_live(&mut w, &self.req, &self.job_id, self.prompt_n);
        stream.tool.use_random_ids();
        self.anth = Some(stream);
        w.out
    }

    /// Per-token sampling override, same policy the serial `decode_pass`
    /// consults (required prefixes + DSML structural greedy).
    pub fn sample_override(&mut self) -> SampleOverride {
        let policy = SamplePolicy {
            tool_choice: self.tool_choice,
            has_tool_results: self.has_tool_results,
            think_mode: self.think_mode,
            max_tokens: self.parsed_max_tokens,
            required_tool_prefix: &self.required_tool_prefix,
            required_think_end_prefix: &self.required_think_end_prefix,
        };
        self.acc.sampling_override(&policy)
    }

    /// Effective sampling block for the engine's per-seq sampler. Thinking
    /// requests pin the serial defaults, mirroring `decode_pass`.
    pub fn sampling(&self, parsed: &ParsedRequest) -> (f32, i32, f32, f32) {
        if think_mode_enabled(parsed.think_mode) {
            (DEFAULT_TEMPERATURE, 0, DEFAULT_TOP_P, DEFAULT_MIN_P)
        } else {
            (parsed.temperature, parsed.top_k, parsed.top_p, parsed.min_p)
        }
    }

    pub fn feed(&mut self, piece: &[u8]) -> ContStep {
        let feed = self.acc.feed(piece, &self.stops);
        if self.req.stream {
            let view = &self.acc.text[..feed.emit_limit.min(self.acc.text.len())];
            match (self.req.api, self.req.kind) {
                (Api::Openai, ReqKind::Completion) => {
                    if let Some(delta) = last_delta(&self.acc.text, feed.emit_limit, piece.len()) {
                        sse_chunk(&mut self.w, &self.req, &self.job_id, Some(delta), None);
                    }
                }
                (Api::Openai, ReqKind::Chat) => {
                    if let Some(st) = self.oa.as_mut() {
                        openai_sse_stream_update(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            view,
                            false,
                        );
                    }
                }
                (Api::Anthropic, _) => {
                    if let Some(st) = self.anth.as_mut() {
                        anthropic_sse_stream_update(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            view,
                            false,
                        );
                    }
                }
                (Api::Responses, _) => {
                    if let Some(st) = self.resp.as_mut() {
                        responses_sse_stream_update(&mut self.w, &self.req, st, view, false);
                    }
                }
            }
        }
        let mut done = false;
        if feed.hit_stop {
            self.finish = "stop";
            done = true;
        } else if self.acc.track_tools
            && self.acc.saw_tool_end
            && self.req.chat_format == crate::stream::ChatFormat::DeepSeek
        {
            self.finish = "tool_calls";
            done = true;
        } else if self.acc.completion >= self.max_tokens {
            self.finish = "length";
            done = true;
        }
        ContStep {
            bytes: std::mem::take(&mut self.w.out),
            done,
        }
    }

    #[cfg(any(feature = "native", test))]
    fn heartbeat(&mut self, now: Instant) -> Vec<u8> {
        stream_heartbeat_if_due(
            &mut self.w,
            &self.req,
            self.resp.as_mut(),
            &mut self.last_heartbeat,
            now,
            ": keep-alive\n\n",
        );
        std::mem::take(&mut self.w.out)
    }

    #[cfg(feature = "native")]
    fn fail(&mut self, message: &str) -> Vec<u8> {
        if self.req.stream {
            stream_error(&mut self.w, &self.req, self.resp.as_mut(), message);
        }
        std::mem::take(&mut self.w.out)
    }

    /// Engine finished (EOS = 1, budget/abort = 0). Runs the serial tail:
    /// tag-completion repair (no re-decode on this lane), generated-message
    /// parse, tool ids, usage/timings, stream finish or buffered final.
    pub fn finalize(
        &mut self,
        engine_eos: bool,
        n_cached: i32,
        n_computed: i32,
        timings: ReqTimings,
        cors: bool,
    ) -> (Vec<u8>, GenerateOutcome) {
        assert!(self.started);
        if self.finish == "length" && engine_eos {
            self.finish = "stop";
        }
        let syntax = syntax_for_model_id(self.model_id);
        self.finish = terminal_finish(self.acc.thinking_inside(), self.finish);
        if let TruncationOutcome::Repair(text) = truncation_outcome(
            syntax,
            self.req.chat_format,
            self.req.kind == ReqKind::Chat,
            self.acc.track_tools,
            self.acc.saw_tool_start,
            self.acc.saw_tool_end,
            self.finish,
            self.req.stream,
            true, /* recovery_attempted: decode_again routes serial */
            &self.acc.text,
            &self.req.tool_orders,
        ) {
            self.acc.text = text;
            self.acc.saw_tool_end = true;
        }
        let mut parsed_gen = if self.req.kind == ReqKind::Chat {
            let (pg, recovered_finish) = parse_generated_for_response(
                syntax,
                &self.acc.text,
                self.acc.track_tools,
                self.acc.saw_tool_start,
                crate::route::think_mode_enabled(self.think_mode),
                self.req.chat_format,
                &self.req.tool_orders,
                self.finish,
            );
            self.finish = recovered_finish;
            pg
        } else {
            crate::tools::ParsedGenerated {
                content: self.acc.text.clone(),
                ok: true,
                ..Default::default()
            }
        };
        let completion = self.acc.completion;
        if !parsed_gen.calls.is_empty() {
            if let Some(st) = self.oa.as_ref() {
                st.tool.apply_ids(&mut parsed_gen.calls);
            }
            if let Some(st) = self.anth.as_ref() {
                st.tool.apply_ids(&mut parsed_gen.calls);
            }
            assign_tool_ids(
                &mut parsed_gen.calls,
                if self.req.api == Api::Anthropic {
                    "toolu_"
                } else {
                    "call_"
                },
            );
            let exact = if parsed_gen.raw_dsml.is_empty() {
                &parsed_gen.raw_tool_text
            } else {
                &parsed_gen.raw_dsml
            };
            if !exact.is_empty() {
                self.tool_replay = Some((parsed_gen.calls.clone(), exact.clone()));
            }
            self.finish = "tool_calls";
        }
        self.apply_engine_usage(n_cached, n_computed);
        self.req.timings = timings;
        if self.req.stream {
            match (self.req.api, self.req.kind) {
                (Api::Openai, ReqKind::Completion) => {
                    sse_chunk(
                        &mut self.w,
                        &self.req,
                        &self.job_id,
                        None,
                        Some(self.finish),
                    );
                    sse_done(
                        &mut self.w,
                        &self.req,
                        &self.job_id,
                        self.prompt_n,
                        completion,
                    );
                }
                (Api::Openai, ReqKind::Chat) => {
                    if let Some(st) = self.oa.as_mut() {
                        openai_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.prompt_n,
                            completion,
                            &parsed_gen.calls,
                        );
                    }
                }
                (Api::Anthropic, _) => {
                    if let Some(st) = self.anth.as_mut() {
                        anthropic_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.acc.matched_stop.as_deref(),
                            completion,
                            &parsed_gen.calls,
                        );
                    }
                }
                (Api::Responses, _) => {
                    if let Some(st) = self.resp.as_mut() {
                        let created = self.w.created;
                        responses_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.prompt_n,
                            completion,
                            self.acc.reasoning_tokens,
                            created,
                            &parsed_gen.calls,
                        );
                    }
                }
            }
        } else {
            let bytes = match self.req.api {
                Api::Anthropic => anthropic_final_response(
                    &self.req,
                    &self.job_id,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    self.finish,
                    self.acc.matched_stop.as_deref(),
                    self.prompt_n,
                    completion,
                    cors,
                    &parsed_gen.calls,
                ),
                Api::Responses => {
                    let (response_id, reasoning_id, message_id) = responses_ids(&self.job_id);
                    responses_final_response(
                        &self.req,
                        &parsed_gen.content,
                        Some(&parsed_gen.reasoning),
                        self.finish,
                        self.prompt_n,
                        completion,
                        self.acc.reasoning_tokens,
                        self.w.created,
                        cors,
                        &response_id,
                        &reasoning_id,
                        &message_id,
                        &parsed_gen.calls,
                    )
                }
                Api::Openai => final_response(
                    &self.req,
                    &self.job_id,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    self.finish,
                    self.prompt_n,
                    completion,
                    self.w.created,
                    cors,
                    &parsed_gen.calls,
                ),
            };
            self.w.out.extend_from_slice(&bytes);
        }
        let outcome = GenerateOutcome {
            tool_ids: parsed_gen
                .calls
                .iter()
                .map(|c| c.id.clone())
                .filter(|id| !id.is_empty())
                .collect(),
            bank: None,
            generation: 0,
            frontier: self.prompt_n + completion,
            finish: self.finish.to_string(),
        };
        (std::mem::take(&mut self.w.out), outcome)
    }

    #[cfg(any(feature = "native", test))]
    fn take_tool_replay(&mut self) -> Option<(Vec<ToolCall>, String)> {
        self.tool_replay.take()
    }

    #[cfg(feature = "native")]
    fn has_complete_tool_turn(&self) -> bool {
        self.acc.saw_tool_end
    }
}

#[cfg(any(feature = "native", test))]
const QWEN_IMAGE_CACHE_PREFIX: &[u8] = b"\xffDS4IMG2:";
#[cfg(any(feature = "native", test))]
const QWEN_IMAGE_CACHE_MARKER_LEN: usize = 53;
#[cfg(any(feature = "native", test))]
const QWEN_IMAGE_PAD: &[u8] = b"<|image_pad|>";

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageCacheSpan {
    marker_start: usize,
    marker_end: usize,
    token_offset: u32,
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy)]
struct ImageCacheIdentity {
    pixel_hash: u64,
    grid_h: u32,
    grid_w: u32,
    token_count: u32,
    token_offset: u32,
}

#[cfg(any(feature = "native", test))]
fn qwen_image_cache_text_build(
    text: &[u8],
    images: &[ImageCacheIdentity],
) -> Result<(Vec<u8>, Vec<ImageCacheSpan>), &'static str> {
    if images.is_empty() || images.len() > 4 {
        return Err("image cache key requires an image");
    }
    let capacity = text
        .len()
        .checked_add(
            images
                .len()
                .checked_mul(QWEN_IMAGE_CACHE_MARKER_LEN)
                .ok_or("image cache key overflow")?,
        )
        .ok_or("image cache key overflow")?;
    let mut key = Vec::with_capacity(capacity);
    let mut spans = Vec::with_capacity(images.len());
    let mut cursor = 0;
    for image in images {
        let pad = text[cursor..]
            .windows(QWEN_IMAGE_PAD.len())
            .position(|window| window == QWEN_IMAGE_PAD)
            .map(|offset| cursor + offset)
            .ok_or("image cache key is missing an image placeholder")?;
        let through_pad = pad + QWEN_IMAGE_PAD.len();
        key.extend_from_slice(&text[cursor..through_pad]);
        let marker_start = key.len();
        key.extend_from_slice(QWEN_IMAGE_CACHE_PREFIX);
        key.extend_from_slice(
            format!(
                "{:016x}:{:08x}:{:08x}:{:08x}",
                image.pixel_hash, image.grid_h, image.grid_w, image.token_count
            )
            .as_bytes(),
        );
        key.push(0xfe);
        let marker_end = key.len();
        if marker_end - marker_start != QWEN_IMAGE_CACHE_MARKER_LEN {
            return Err("image cache key overflow");
        }
        spans.push(ImageCacheSpan {
            marker_start,
            marker_end,
            token_offset: image.token_offset,
        });
        cursor = through_pad;
    }
    key.extend_from_slice(&text[cursor..]);
    Ok((key, spans))
}

#[cfg(any(feature = "native", test))]
fn qwen_image_cache_marker_at(bytes: &[u8]) -> bool {
    bytes.len() >= QWEN_IMAGE_CACHE_MARKER_LEN
        && bytes.starts_with(QWEN_IMAGE_CACHE_PREFIX)
        && bytes[25] == b':'
        && bytes[34] == b':'
        && bytes[43] == b':'
        && bytes[52] == 0xfe
        && (QWEN_IMAGE_CACHE_PREFIX.len()..52)
            .all(|i| matches!(i, 25 | 34 | 43) || bytes[i].is_ascii_hexdigit())
}

#[cfg(any(feature = "native", test))]
fn qwen_image_cache_text_strip(key: &[u8]) -> Option<Vec<u8>> {
    let mut text = Vec::with_capacity(key.len());
    let mut markers = 0;
    let mut i = 0;
    while i < key.len() {
        if qwen_image_cache_marker_at(&key[i..]) {
            i += QWEN_IMAGE_CACHE_MARKER_LEN;
            markers += 1;
        } else {
            text.push(key[i]);
            i += 1;
        }
    }
    (markers > 0).then_some(text)
}

#[cfg(any(feature = "native", test))]
fn qwen_image_cache_token_cap(spans: &[ImageCacheSpan], cache_lcp: usize) -> usize {
    spans
        .iter()
        .find(|span| cache_lcp >= span.marker_start && cache_lcp < span.marker_end)
        .map_or(usize::MAX, |span| span.token_offset as usize)
}

fn last_delta(raw: &[u8], emit_limit: usize, piece_len: usize) -> Option<&[u8]> {
    if emit_limit == 0 {
        return None;
    }
    let start = raw.len().saturating_sub(piece_len).min(emit_limit);
    if start >= emit_limit {
        return None;
    }
    Some(&raw[start..emit_limit])
}

#[cfg(any(feature = "native", test))]
fn bank_scope(parsed: &ParsedRequest) -> bool {
    parsed.kind == ReqKind::Chat
}

#[cfg(any(feature = "native", test))]
fn prompt_preserves_reasoning(parsed: &ParsedRequest) -> bool {
    parsed.has_tools
        || parsed.messages.iter().any(|message| {
            (message.role == "assistant" && !message.calls.is_empty())
                || message.role == "tool"
                || message.role == "function"
        })
}

#[cfg(any(feature = "native", test))]
fn thinking_visible_key(
    prompt: &[u8],
    content: &[u8],
    syntax: ModelSyntax,
    format: ChatFormat,
) -> Option<Vec<u8>> {
    if format == ChatFormat::Qwen4Exp || syntax == ModelSyntax::Exaone {
        if !prompt.ends_with(b"<think>\n") {
            return None;
        }
        let content = content.trim_ascii();
        let mut visible = Vec::with_capacity(prompt.len() + 12 + content.len());
        visible.extend_from_slice(prompt);
        visible.extend_from_slice(b"\n</think>\n\n");
        visible.extend_from_slice(content);
        return Some(visible);
    }

    let start = think_start(format).as_bytes();
    if !prompt.ends_with(start) {
        return None;
    }
    let prefix = if format == ChatFormat::SolarOpen2 {
        prompt
    } else {
        &prompt[..prompt.len() - start.len()]
    };
    let mut visible = Vec::with_capacity(prefix.len() + think_end(format).len() + content.len());
    visible.extend_from_slice(prefix);
    visible.extend_from_slice(think_end(format).as_bytes());
    visible.extend_from_slice(content);
    Some(visible)
}

#[cfg(any(feature = "native", test))]
#[derive(Debug, Default)]
pub(crate) struct WarmBank {
    pub(crate) record: Option<WarmRecord>,
    /// Exact native committed frontier recorded at retire/restore. Banks are
    /// idle between owner calls, so this is also the C victim-picker depth.
    pub(crate) committed_tokens: i32,
    pub(crate) stored_tokens: i32,
    pub(crate) last_use: u64,
}

#[cfg(any(feature = "native", test))]
#[derive(Debug)]
pub(crate) struct WarmRecord {
    pub(crate) text: Vec<u8>,
    pub(crate) cache_text: Option<Vec<u8>>,
    pub(crate) exact_text: Option<Vec<u8>>,
    pub(crate) exact_cache_text: Option<Vec<u8>>,
    pub(crate) partial_only: bool,
    pub(crate) generation: u64,
    pub(crate) ext_flags: u8,
    pub(crate) trailer: Vec<u8>,
}

/// Keep Responses/title bits across retire so the next persist still
/// writes them. Trailer is rebuilt at persist time.
#[cfg(any(feature = "native", test))]
fn retired_record(
    prev: Option<&WarmRecord>,
    key: Vec<u8>,
    cache_text: Option<Vec<u8>>,
    generation: u64,
) -> Option<WarmRecord> {
    let mut ext_flags = prev.map(|record| record.ext_flags).unwrap_or(0) & !EXT_IMAGE_PIXELS_V2;
    if cache_text.is_some() {
        ext_flags |= EXT_IMAGE_PIXELS_V2;
    }
    (!key.is_empty()).then(|| WarmRecord {
        text: key,
        cache_text,
        exact_text: None,
        exact_cache_text: None,
        partial_only: false,
        generation,
        ext_flags,
        trailer: Vec::new(),
    })
}

#[cfg(any(feature = "native", test))]
fn restored_record(key: Vec<u8>, generation: u64, ext_flags: u8) -> Option<WarmRecord> {
    if ext_flags & EXT_IMAGE_PIXELS_V2 != 0 {
        Some(WarmRecord {
            text: qwen_image_cache_text_strip(&key)?,
            cache_text: Some(key),
            exact_text: None,
            exact_cache_text: None,
            partial_only: false,
            generation,
            ext_flags,
            trailer: Vec::new(),
        })
    } else {
        Some(WarmRecord {
            text: key,
            cache_text: None,
            exact_text: None,
            exact_cache_text: None,
            partial_only: false,
            generation,
            ext_flags,
            trailer: Vec::new(),
        })
    }
}

#[cfg(any(feature = "native", test))]
fn extended_image_cache_key(prompt: &[u8], cache_prompt: &[u8], key: &[u8]) -> Option<Vec<u8>> {
    let suffix = key.strip_prefix(prompt)?;
    let mut cache_key = Vec::with_capacity(cache_prompt.len().checked_add(suffix.len())?);
    cache_key.extend_from_slice(cache_prompt);
    cache_key.extend_from_slice(suffix);
    Some(cache_key)
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WarmMatch {
    bank: usize,
    exact: bool,
}

#[cfg(any(feature = "native", test))]
fn warm_match_pick(
    banks: &[WarmBank],
    prompt: &[u8],
    cache_prompt: Option<&[u8]>,
) -> Option<WarmMatch> {
    let request_key = cache_prompt.unwrap_or(prompt);
    let mut best = None;
    let mut best_len = 0;
    for (bank, state) in banks.iter().enumerate() {
        let Some(record) = state.record.as_ref() else {
            continue;
        };
        if record.partial_only || record.cache_text.is_some() != cache_prompt.is_some() {
            continue;
        }
        let primary = record.cache_text.as_deref().unwrap_or(&record.text);
        let exact = if cache_prompt.is_some() {
            record.exact_cache_text.as_deref()
        } else {
            record.exact_text.as_deref()
        };
        for (key, is_exact) in [(Some(primary), false), (exact, true)] {
            let Some(key) = key else { continue };
            if !key.is_empty()
                && key.len() < request_key.len()
                && key.len() > best_len
                && request_key.starts_with(key)
            {
                best = Some(WarmMatch {
                    bank,
                    exact: is_exact,
                });
                best_len = key.len();
            }
        }
    }
    best
}

#[cfg(any(feature = "native", test))]
fn warm_partial_match_pick(
    banks: &[WarmBank],
    prompt: &[u8],
    cache_prompt: Option<&[u8]>,
    min_prefix: usize,
) -> Option<(usize, usize)> {
    let request_key = cache_prompt.unwrap_or(prompt);
    if min_prefix == 0 || request_key.len() < min_prefix {
        return None;
    }
    let mut best = None;
    let mut best_len = 0;
    for (bank, state) in banks.iter().enumerate() {
        let Some(record) = state.record.as_ref() else {
            continue;
        };
        if record.cache_text.is_some() != cache_prompt.is_some() {
            continue;
        }
        let key = record.cache_text.as_deref().unwrap_or(&record.text);
        let prefix = request_key
            .iter()
            .zip(key)
            .take_while(|(left, right)| left == right)
            .count();
        if prefix >= min_prefix && prefix > best_len {
            best = Some((bank, prefix));
            best_len = prefix;
        }
    }
    best
}

#[cfg(any(feature = "native", test))]
fn warm_partial_token_cut(
    source: &[i32],
    prompt: &[i32],
    min_tokens: i32,
    token_cap: usize,
) -> Option<i32> {
    let cap = source
        .len()
        .min(prompt.len().saturating_sub(1))
        .min(token_cap);
    let cut = source
        .iter()
        .zip(prompt)
        .take(cap)
        .take_while(|(left, right)| left == right)
        .count();
    i32::try_from(cut)
        .ok()
        .filter(|cut| *cut >= min_tokens.max(1))
}

#[cfg(any(feature = "native", test))]
fn warm_victim_pick(
    banks: &[WarmBank],
    protected: &[bool],
    exclude: Option<usize>,
    pin_min: i32,
    evict_lru: bool,
) -> Option<usize> {
    debug_assert!(protected.is_empty() || protected.len() == banks.len());
    let mut superseded = None;
    let mut superseded_use = u64::MAX;
    let mut shallow = None;
    let mut shallow_use = u64::MAX;
    let mut deep = None;
    let mut deep_use = u64::MAX;

    for (bank, state) in banks.iter().enumerate() {
        if Some(bank) == exclude || protected.get(bank).copied().unwrap_or(false) {
            continue;
        }
        if state.record.is_none() {
            return Some(bank);
        }
        let is_deep = pin_min > 0 && state.committed_tokens >= pin_min;
        if is_deep {
            if state.last_use < deep_use {
                deep = Some(bank);
                deep_use = state.last_use;
            }
        } else if state.last_use < shallow_use {
            shallow = Some(bank);
            shallow_use = state.last_use;
        }
        if state.last_use < superseded_use && warm_record_superseded(banks, bank) {
            superseded = Some(bank);
            superseded_use = state.last_use;
        }
    }
    superseded.or_else(|| evict_lru.then_some(shallow.or(deep)).flatten())
}

#[cfg(any(feature = "native", test))]
fn warm_record_superseded(banks: &[WarmBank], bank: usize) -> bool {
    let Some(record) = banks.get(bank).and_then(|state| state.record.as_ref()) else {
        return false;
    };
    let key = record.cache_text.as_deref().unwrap_or(&record.text);
    if key.is_empty() {
        return false;
    }
    banks.iter().enumerate().any(|(other_bank, state)| {
        other_bank != bank
            && state.record.as_ref().is_some_and(|other| {
                let other_key = other.cache_text.as_deref().unwrap_or(&other.text);
                !other.partial_only
                    && other.cache_text.is_some() == record.cache_text.is_some()
                    && !other_key.is_empty()
                    && other_key.starts_with(key)
            })
    })
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WarmPlacement {
    source: usize,
    target: usize,
    fork: bool,
}

#[cfg(any(feature = "native", test))]
fn warm_placement(
    banks: &[WarmBank],
    protected: &[bool],
    source: usize,
    cached: i32,
    pin_min: i32,
    fork_enabled: bool,
) -> Option<WarmPlacement> {
    let target = (fork_enabled && (pin_min <= 0 || cached < pin_min))
        .then(|| warm_victim_pick(banks, protected, Some(source), pin_min, false))
        .flatten();
    if let Some(target) = target {
        return Some(WarmPlacement {
            source,
            target,
            fork: true,
        });
    }
    if protected.get(source).copied().unwrap_or(false) {
        return None;
    }
    Some(WarmPlacement {
        source,
        target: source,
        fork: false,
    })
}

fn motif3_history_retire_prompt(prompt: &[u8]) -> &[u8] {
    // Motif none-think generation ends with an empty think pair; official
    // history replay omits it. Bank keys must use the history form or the
    // next tool-result turn diverges at <|assistant|>.
    prompt.strip_suffix(b"<think></think>").unwrap_or(prompt)
}

fn committed_key(
    prompt: &[u8],
    tokens: &[i32],
    mut token_text: impl FnMut(i32) -> Vec<u8>,
) -> Vec<u8> {
    let mut key = motif3_history_retire_prompt(prompt).to_vec();
    for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
        key.extend(token_text(token));
    }
    key
}

#[cfg(any(feature = "native", test))]
#[derive(Debug, PartialEq, Eq)]
struct BankRetireKey {
    text: Vec<u8>,
    exact_text: Option<Vec<u8>>,
    partial_only: bool,
    retained_existing: bool,
}

#[cfg(any(feature = "native", test))]
fn text_ends_with_proper_prefix(text: &[u8], marker: &[u8]) -> bool {
    (1..marker.len())
        .rev()
        .any(|len| text.ends_with(&marker[..len]))
}

#[cfg(any(feature = "native", test))]
fn thinking_bank_retire_key(
    prompt: &[u8],
    done_tokens: &[i32],
    engine_finished: bool,
    semantic_or_transport_cut: bool,
    prompt_preserves_reasoning: bool,
    has_tools: bool,
    saw_tool_start: bool,
    syntax: ModelSyntax,
    format: ChatFormat,
    mut token_text: impl FnMut(i32) -> Vec<u8>,
) -> Option<BankRetireKey> {
    if done_tokens.is_empty() {
        return None;
    }
    let mut generation = Vec::new();
    for &token in done_tokens.iter().take(done_tokens.len().saturating_sub(1)) {
        generation.extend(token_text(token));
    }
    let mut exact = prompt.to_vec();
    exact.extend_from_slice(&generation);

    let close = think_end(format).as_bytes();
    let close_at = generation
        .windows(close.len())
        .position(|window| window == close);

    if prompt_preserves_reasoning {
        if engine_finished && has_tools && !saw_tool_start {
            if let Some(at) = close_at {
                if let Some(visible) =
                    thinking_visible_key(prompt, &generation[at + close.len()..], syntax, format)
                {
                    return Some(BankRetireKey {
                        text: visible,
                        exact_text: Some(exact),
                        partial_only: false,
                        retained_existing: false,
                    });
                }
            }
        }
        return Some(BankRetireKey {
            text: exact,
            exact_text: None,
            partial_only: false,
            retained_existing: false,
        });
    }

    if engine_finished {
        let at = close_at?;
        return Some(BankRetireKey {
            text: thinking_visible_key(prompt, &generation[at + close.len()..], syntax, format)?,
            exact_text: None,
            partial_only: false,
            retained_existing: false,
        });
    }

    let partial_only = done_tokens.len() > 1
        && (semantic_or_transport_cut || text_ends_with_proper_prefix(&generation, close));
    if !partial_only {
        if let Some(at) = close_at {
            return Some(BankRetireKey {
                text: thinking_visible_key(
                    prompt,
                    &generation[at + close.len()..],
                    syntax,
                    format,
                )?,
                exact_text: None,
                partial_only: false,
                retained_existing: false,
            });
        }
    }
    Some(BankRetireKey {
        text: prompt.to_vec(),
        exact_text: None,
        partial_only,
        retained_existing: false,
    })
}

#[cfg(any(feature = "native", test))]
fn bank_retire_key(
    prompt: &[u8],
    snapshot_tokens: &[i32],
    done_tokens: &[i32],
    allow_generated_snapshot: bool,
    mut token_text: impl FnMut(i32) -> Vec<u8>,
) -> Option<(Vec<u8>, bool)> {
    if !done_tokens.is_empty() {
        return Some((committed_key(prompt, done_tokens, token_text), false));
    }
    let retained = committed_key(&[], snapshot_tokens, &mut token_text);
    if retained.is_empty() {
        return None;
    }
    if prompt.starts_with(&retained) {
        return Some((retained, true));
    }
    (allow_generated_snapshot && retained.starts_with(prompt)).then_some((retained, false))
}

#[cfg(any(feature = "native", test))]
fn warm_admit_tokens(
    warm: &WarmBank,
    prompt: &[u8],
    cache_prompt: Option<&[u8]>,
    snapshot_tokens: &[i32],
    generation: u64,
    seq_cap: i32,
    exact: bool,
    tokenize_suffix: impl FnOnce(&[u8]) -> Vec<i32>,
) -> Option<(Vec<i32>, i32)> {
    let record = warm.record.as_ref()?;
    if record.generation != generation || record.partial_only {
        return None;
    }
    if record.cache_text.is_some() != cache_prompt.is_some() {
        return None;
    }
    let text = if exact {
        record.exact_text.as_deref()?
    } else {
        &record.text
    };
    let key = if cache_prompt.is_some() {
        if exact {
            record.exact_cache_text.as_deref()?
        } else {
            record.cache_text.as_deref()?
        }
    } else {
        text
    };
    let request_key = cache_prompt.unwrap_or(prompt);
    if key.is_empty()
        || key.len() >= request_key.len()
        || !request_key.starts_with(key)
        || text.len() >= prompt.len()
        || !prompt.starts_with(text)
    {
        return None;
    }
    let cached = i32::try_from(snapshot_tokens.len())
        .ok()
        .filter(|n| *n > 0)?;
    let mut tokens = snapshot_tokens.to_vec();
    tokens.extend(tokenize_suffix(&prompt[text.len()..]));
    if tokens.len() <= snapshot_tokens.len()
        || i32::try_from(tokens.len())
            .ok()
            .filter(|n| *n <= seq_cap)
            .is_none()
    {
        return None;
    }
    Some((tokens, cached))
}

#[cfg(any(feature = "native", test))]
fn live_continuation_tokens(
    exact_prefix: &[i32],
    suffix: &[u8],
    seq_cap: i32,
    tokenize_suffix: impl FnOnce(&[u8]) -> Vec<i32>,
) -> Option<(Vec<i32>, i32)> {
    let cached = i32::try_from(exact_prefix.len()).ok().filter(|n| *n > 0)?;
    let mut tokens = exact_prefix.to_vec();
    tokens.extend(tokenize_suffix(suffix));
    (tokens.len() > exact_prefix.len() && i32::try_from(tokens.len()).is_ok_and(|n| n <= seq_cap))
        .then_some((tokens, cached))
}

#[cfg(any(feature = "native", test))]
fn warm_partial_admit_tokens(
    warm: &WarmBank,
    prompt_tokens: &[i32],
    snapshot_tokens: &[i32],
    generation: u64,
    min_tokens: i32,
    seq_cap: i32,
    token_cap: usize,
) -> Option<(Vec<i32>, i32)> {
    let record = warm.record.as_ref()?;
    if record.generation != generation
        || i32::try_from(prompt_tokens.len())
            .ok()
            .filter(|tokens| *tokens <= seq_cap)
            .is_none()
    {
        return None;
    }
    let cached = warm_partial_token_cut(snapshot_tokens, prompt_tokens, min_tokens, token_cap)?;
    Some((prompt_tokens.to_vec(), cached))
}

#[cfg(any(feature = "native", test))]
fn disk_restore_allowed(
    warm: &WarmBank,
    live: Option<(u64, usize)>,
    disk_tokens: u32,
    pin_min: i32,
) -> bool {
    let Some(record) = warm.record.as_ref() else {
        return true;
    };
    let Some((generation, live_tokens)) = live else {
        return true;
    };
    if record.generation != generation {
        return true;
    }
    let Ok(pin_min) = usize::try_from(pin_min) else {
        return false;
    };
    pin_min > 0 && disk_tokens as usize >= pin_min && live_tokens < pin_min
}

#[cfg(any(feature = "native", test))]
fn disk_restore_target_allowed(
    banks: &[WarmBank],
    target: usize,
    disk_tokens: u32,
    pin_min: i32,
) -> bool {
    let Some(state) = banks.get(target) else {
        return false;
    };
    if warm_record_superseded(banks, target) {
        return true;
    }
    let live = state
        .record
        .as_ref()
        .map(|record| (record.generation, state.committed_tokens.max(0) as usize));
    disk_restore_allowed(state, live, disk_tokens, pin_min)
}

#[cfg(any(feature = "native", test))]
fn bank_retire_allowed(bank_enabled: bool, admitted: bool, done_called: bool) -> bool {
    bank_enabled && admitted && done_called
}

#[cfg(any(feature = "native", test))]
fn retired_bank(
    bank_enabled: bool,
    admitted: bool,
    done_called: bool,
    actual_bank: Option<i32>,
    max_seq: i32,
) -> Option<i32> {
    if !bank_retire_allowed(bank_enabled, admitted, done_called) {
        return None;
    }
    actual_bank.filter(|bank| *bank >= 0 && *bank < max_seq)
}

#[cfg(any(feature = "native", test))]
pub(crate) fn save_bank_record(
    store: &mut KvStore,
    warm: &mut WarmBank,
    identity: (u8, u8, u32),
    committed: i32,
    generation: u64,
    reason: KvReason,
    save_payload: impl FnOnce(&Path) -> Result<(), GenerateError>,
) -> Result<bool, GenerateError> {
    if committed <= 0 {
        return Ok(false);
    }
    let Some(record) = warm.record.as_ref().filter(|record| {
        record.generation == generation && !record.text.is_empty() && !record.partial_only
    }) else {
        return Ok(false);
    };
    let text = record.cache_text.as_deref().unwrap_or(&record.text);
    let tokens = u32::try_from(committed)
        .map_err(|_| GenerateError::Engine("bank token count exceeds u32".into()))?;
    let keep_ext = record.ext_flags;
    let trailer = &record.trailer;
    let (model_id, quant_bits, ctx) = identity;
    let mut header = crate::generate::kv_header(model_id, quant_bits, ctx, tokens);
    header.reason = reason;
    header.ext_flags = bank_persist_ext_flags(keep_ext, false, !trailer.is_empty());
    let payload = store
        .payload_temp()
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    save_payload(payload.path())?;
    store
        .write_payload_file(header, text, payload.path(), trailer)
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    warm.stored_tokens = committed;
    Ok(true)
}

/// One continuous work item for a rolling admit.
#[derive(Clone, Debug)]
pub struct ContPreparedPrompt {
    pub prompt: Vec<u8>,
    pub tokens: Vec<i32>,
}

pub struct ContWork<'a> {
    pub parsed: &'a ParsedRequest,
    pub prepared: Option<&'a ContPreparedPrompt>,
    pub job_id: &'a str,
    pub created: i64,
    pub cors: bool,
    pub default_tokens: i32,
    pub t_arrive: Instant,
    pub out: &'a mut dyn Write,
}

/// Owned request handed to the native rolling loop. The owner keeps the
/// original job for settlement; `out` is a clone of its shared sink.
pub struct ContOwnedWork {
    pub key: usize,
    pub parsed: ParsedRequest,
    pub prepared: Option<ContPreparedPrompt>,
    pub job_id: String,
    pub created: i64,
    pub cors: bool,
    pub default_tokens: i32,
    pub t_arrive: Instant,
    pub out: Box<dyn Write>,
}

pub struct ContOwnedResult {
    pub key: usize,
    pub result: Result<GenerateOutcome, GenerateError>,
}

/// Read-only engine view used by the owner while native generation is active.
pub trait ContProbe {
    fn prompt_tokens(&self, parsed: &ParsedRequest) -> Result<(Vec<u8>, Vec<i32>), GenerateError>;
    fn seq_cap(&self) -> i32;
    fn bank_live(&self, bank: i32) -> Option<(u64, i32)>;
}

/// FIFO source polled again whenever the native scheduler exposes a free bank.
pub trait ContSource {
    fn next(&mut self, probe: &dyn ContProbe) -> Option<ContOwnedWork>;

    /// Called after the bank record is retired and before terminal bytes are
    /// appended to the shared client sink.
    fn publish(&mut self, _key: usize, _outcome: &GenerateOutcome) {}

    /// Called after terminal bytes have been appended. The owner can release
    /// completed clients without waiting for the whole rolling epoch to drain.
    fn settled(&mut self, _key: usize, _result: &Result<GenerateOutcome, GenerateError>) {}
}

/// Trait seam so `handle_client_inner` can drive a continuous lane without
/// the native feature (tests supply a scripted implementation).
pub trait ContExec {
    fn model_id(&self) -> i32;
    fn seq_cap(&self) -> i32;
    fn set_stop_requested(&mut self, _stop_requested: Option<fn() -> bool>) {}
    /// Number of persistent continuous banks available to this executor.
    fn max_seq(&self) -> i32 {
        1
    }
    /// One C trim (`ds4_batch_ctx_trim_free`). Host does not implement trim.
    fn trim_idle_banks(&mut self, _want_bytes: u64) -> u64 {
        0
    }
    fn encode_chat(&self, rendered: &[u8]) -> Vec<i32>;
    fn encode_text(&self, text: &str) -> Vec<i32>;
    fn prepare_tokens(
        &self,
        parsed: &ParsedRequest,
        _prompt: &[u8],
        tokens: Vec<i32>,
    ) -> Result<Vec<i32>, GenerateError> {
        if parsed.images.is_empty() {
            Ok(tokens)
        } else {
            Err(GenerateError::Unsupported(
                "image input requires native continuous runtime",
            ))
        }
    }
    /// `bank_hold_retry` keeps registry locking outside the native owner; a
    /// missing live reference asks the registry to preserve C's fail-closed rule.
    fn generate(
        &mut self,
        parsed: &ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError>;

    fn generate_prepared(
        &mut self,
        parsed: &ParsedRequest,
        _prepared: Option<&ContPreparedPrompt>,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        self.generate(
            parsed,
            job_id,
            created,
            cors,
            default_tokens,
            t_arrive,
            bank_hold_retry,
            store,
            out,
        )
    }

    /// Drive up to `max_seq()` jobs in one rolling admit loop. The default
    /// keeps non-native implementations correct by running them sequentially.
    fn generate_batch(
        &mut self,
        jobs: Vec<ContWork<'_>>,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        mut store: Option<&mut KvStore>,
    ) -> Vec<Result<GenerateOutcome, GenerateError>> {
        jobs.into_iter()
            .map(|job| {
                self.generate_prepared(
                    job.parsed,
                    job.prepared,
                    job.job_id,
                    job.created,
                    job.cors,
                    job.default_tokens,
                    job.t_arrive,
                    bank_hold_retry,
                    store.as_deref_mut(),
                    job.out,
                )
            })
            .collect()
    }

    /// Native implementations return `Some` and keep polling `source` while
    /// any bank remains active. Other executors retain the bounded batch path.
    fn generate_rolling(
        &mut self,
        _source: &mut dyn ContSource,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
    ) -> Option<Vec<ContOwnedResult>> {
        None
    }

    fn shutdown(&mut self, _store: Option<&mut KvStore>) {}

    fn bank_live(&self, _bank: i32) -> Option<(u64, i32)> {
        None
    }

    /// Static owner when this lane also holds a `BatchCtx`.
    fn as_static(&mut self) -> Option<&mut dyn crate::serve_static::StaticExec> {
        None
    }
}

/// Render + tokenize a request for routing (`prompt_len` feeds
/// `route_decide` before any lane is entered), mirroring the C server's
/// job-prep order.
pub fn cont_prompt_tokens(
    exec: &dyn ContExec,
    parsed: &ParsedRequest,
) -> Result<(Vec<u8>, Vec<i32>), GenerateError> {
    let prompt = render_prompt(parsed, exec.model_id())?;
    let tokens = match parsed.kind {
        ReqKind::Completion => exec.encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
        ReqKind::Chat => exec.encode_chat(&prompt),
    };
    let tokens = exec.prepare_tokens(parsed, &prompt, tokens)?;
    Ok((prompt, tokens))
}

#[cfg(any(feature = "native", test))]
fn can_reuse_cont_prompt(attached_tool_blocks: usize, has_images: bool) -> bool {
    attached_tool_blocks == 0 && !has_images
}

#[cfg(feature = "native")]
pub use native::ContLane;

#[cfg(feature = "native")]
mod native {
    use super::*;

    use ds4_core::{
        qwen_image_pixel_hash, qwen_image_probe, BatchCtx, ContAdmit, ContDriver, QwenImageInput,
        Vocab, CONT_SAMPLE_GREEDY, CONT_SAMPLE_NONE,
    };

    use crate::serve_static::{BatchStatic, CoalesceLimits, StaticExec, StaticJob, StaticRow};
    use crate::tool_memory::ToolMemory;

    /// Native continuous lane: a rolling `ContDriver` over every bank
    /// exposed by the persistent native batch context.
    pub struct ContLane<'m> {
        batch: BatchCtx<'m>,
        host: ContHost<'m>,
    }

    struct ContHost<'m> {
        vocab: &'m Vocab,
        model_id: i32,
        quant_bits: i32,
        ctx: i32,
        /// Family EOT for the per-seq stop, like the C server's job prep;
        /// the engine's `-1` default is the base EOS, not the family EOT.
        eos: i32,
        warm: Vec<WarmBank>,
        warm_clock: u64,
        warm_fork: bool,
        warm_fork_partial: bool,
        warm_disk_partial: bool,
        warm_partial_min: i32,
        warm_pin_min: i32,
        warm_persist_min: i32,
        warm_checkpoint: bool,
        tool_memory: ToolMemory,
        memgov: Box<dyn crate::serve_cont_roll::ContMemGov>,
        stop_requested: Option<fn() -> bool>,
    }

    struct WarmAdmitPlan {
        source: usize,
        tokens: Vec<i32>,
        cached: i32,
        partial: bool,
    }

    struct PreparedSlot {
        admit: ContAdmit,
        head: Vec<u8>,
        stepper: ContStepper,
        capture_done: bool,
        t_arrive: Instant,
        stop_requested: Option<fn() -> bool>,
    }

    struct PreparedImages {
        tokens: Vec<i32>,
        images: Vec<QwenImageInput>,
        cache_prompt: Option<Vec<u8>>,
        cache_spans: Vec<ImageCacheSpan>,
    }

    fn prepare_qwen_images(
        model_id: i32,
        parsed: &ParsedRequest,
        prompt: Option<&[u8]>,
        tokens: Vec<i32>,
    ) -> Result<PreparedImages, GenerateError> {
        const IMAGE_PAD_TOKEN: i32 = 248056;
        if parsed.images.is_empty() {
            return Ok(PreparedImages {
                tokens,
                images: Vec::new(),
                cache_prompt: None,
                cache_spans: Vec::new(),
            });
        }
        if model_id != 6 {
            return Err(GenerateError::Unsupported(
                "image input is supported only by Qwen4Exp",
            ));
        }
        let mut probed = Vec::with_capacity(parsed.images.len());
        let mut expanded_len = tokens.len();
        for image in &parsed.images {
            let info = qwen_image_probe(&image.data)
                .map_err(|error| GenerateError::Engine(error.to_string()))?;
            if info.token_count == 0 {
                return Err(GenerateError::Engine(
                    "Qwen image probe returned zero tokens".into(),
                ));
            }
            expanded_len = expanded_len
                .checked_add(info.token_count as usize - 1)
                .ok_or_else(|| {
                    GenerateError::Engine("expanded image prompt is too large".into())
                })?;
            probed.push(info);
        }
        if expanded_len > i32::MAX as usize {
            return Err(GenerateError::Engine(
                "expanded image prompt is too large".into(),
            ));
        }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut images = Vec::with_capacity(parsed.images.len());
        let mut image_index = 0;
        for token in tokens {
            if token != IMAGE_PAD_TOKEN {
                expanded.push(token);
                continue;
            }
            let Some((image, info)) = parsed.images.get(image_index).zip(probed.get(image_index))
            else {
                return Err(GenerateError::Engine(
                    "ambiguous literal <|image_pad|> in prompt".into(),
                ));
            };
            let token_offset = u32::try_from(expanded.len())
                .map_err(|_| GenerateError::Engine("expanded image prompt is too large".into()))?;
            expanded.extend(std::iter::repeat_n(
                IMAGE_PAD_TOKEN,
                info.token_count as usize,
            ));
            images.push(QwenImageInput {
                data: image.data.clone(),
                token_offset,
                grid_h: info.grid_h,
                grid_w: info.grid_w,
            });
            image_index += 1;
        }
        if image_index != parsed.images.len() || expanded.len() != expanded_len {
            return Err(GenerateError::Engine(
                "Qwen image placeholder count does not match payloads".into(),
            ));
        }
        let (cache_prompt, cache_spans) = if let Some(prompt) = prompt {
            let identities = images
                .iter()
                .zip(&probed)
                .map(|(image, info)| {
                    Ok(ImageCacheIdentity {
                        pixel_hash: qwen_image_pixel_hash(&image.data)
                            .map_err(|error| GenerateError::Engine(error.to_string()))?,
                        grid_h: info.grid_h,
                        grid_w: info.grid_w,
                        token_count: info.token_count,
                        token_offset: image.token_offset,
                    })
                })
                .collect::<Result<Vec<_>, GenerateError>>()?;
            let (key, spans) = qwen_image_cache_text_build(prompt, &identities)
                .map_err(|error| GenerateError::Engine(error.into()))?;
            (Some(key), spans)
        } else {
            (None, Vec::new())
        };
        Ok(PreparedImages {
            tokens: expanded,
            images,
            cache_prompt,
            cache_spans,
        })
    }

    impl PreparedSlot {
        fn into_slot<'a>(self, out: Box<dyn Write + 'a>) -> JobSlot<'a> {
            JobSlot {
                admit: Some(self.admit),
                head: if self.head.is_empty() {
                    None
                } else {
                    Some(self.head)
                },
                stepper: self.stepper,
                out,
                admitted: false,
                bank: None,
                io_failed: false,
                host_abort: false,
                engine_eos: false,
                capture_done: self.capture_done,
                stop_requested: self.stop_requested,
                done_tokens: Vec::new(),
                n_cached: 0,
                n_computed: 0,
                t_arrive: self.t_arrive,
                t_admit: None,
                t_first: None,
                t_done: None,
                decode_ms: 0.0,
                decode_tokens: 0,
                decode_steps: 0,
            }
        }
    }

    struct JobSlot<'a> {
        admit: Option<ContAdmit>,
        /* Client-visible transport is committed at on_admitted, the C
         * cont_stream_start point: a rejected request never sees bytes and
         * can fall back to the serial lane transport-clean. */
        head: Option<Vec<u8>>,
        stepper: ContStepper,
        out: Box<dyn Write + 'a>,
        admitted: bool,
        bank: Option<i32>,
        io_failed: bool,
        host_abort: bool,
        engine_eos: bool,
        capture_done: bool,
        stop_requested: Option<fn() -> bool>,
        done_tokens: Vec<i32>,
        n_cached: i32,
        n_computed: i32,
        t_arrive: Instant,
        t_admit: Option<Instant>,
        t_first: Option<Instant>,
        t_done: Option<Instant>,
        decode_ms: f64,
        decode_tokens: i32,
        decode_steps: i32,
    }

    impl JobSlot<'_> {
        fn transport_alive(&mut self) -> bool {
            if self.stepper.stop_for_shutdown(self.stop_requested) {
                self.host_abort = true;
                return false;
            }
            if self.admitted {
                let heartbeat = self.stepper.heartbeat(Instant::now());
                self.push(&heartbeat);
            }
            if !self.io_failed && self.out.flush().is_err() {
                self.io_failed = true;
            }
            !self.io_failed
        }

        fn push(&mut self, bytes: &[u8]) {
            if bytes.is_empty() || self.io_failed {
                return;
            }
            if self.out.write_all(bytes).is_err() || self.out.flush().is_err() {
                self.io_failed = true;
            }
        }

        fn on_token(&mut self, vocab: &Vocab, token: i32) -> bool {
            if !self.transport_alive() {
                self.host_abort = true;
                return false;
            }
            if self.t_first.is_none() {
                self.t_first = Some(Instant::now());
            }
            if vocab.is_stop(token) {
                self.stepper.mark_stop();
                self.host_abort = true;
                return false;
            }
            let step = self.stepper.feed(&vocab.token_text(token));
            self.push(&step.bytes);
            if self.io_failed || step.done {
                self.host_abort = true;
                return false;
            }
            true
        }

        fn sample_override(&mut self) -> i32 {
            match self.stepper.sample_override() {
                SampleOverride::None => CONT_SAMPLE_NONE,
                SampleOverride::Greedy => CONT_SAMPLE_GREEDY,
                SampleOverride::Token(token) => ds4_core::cont_sample_token(token),
            }
        }

        fn admitted(&mut self, n_cached: i32, n_computed: i32, bank: i32) -> bool {
            self.n_cached = n_cached;
            self.n_computed = n_computed;
            self.bank = Some(bank);
            self.t_admit = Some(Instant::now());
            self.admitted = true;
            let head = self.stepper.admitted_head(
                self.head.take().unwrap_or_default(),
                n_cached,
                n_computed,
            );
            self.push(&head);
            self.transport_alive()
        }

        fn done(
            &mut self,
            tokens: &[i32],
            finish: i32,
            decode_ms: f64,
            decode_tokens: i32,
            decode_steps: i32,
        ) {
            self.engine_eos = finish == 1;
            if self.capture_done {
                self.done_tokens.extend_from_slice(tokens);
            }
            self.decode_ms = decode_ms;
            self.decode_tokens = decode_tokens;
            self.decode_steps = decode_steps;
            self.t_done = Some(Instant::now());
        }
    }

    struct RollDriver<'a> {
        roll: crate::serve_cont_roll::ContRoll,
        slots: std::collections::HashMap<usize, JobSlot<'a>>,
        vocab: &'a Vocab,
        next_user: usize,
    }

    impl<'a> RollDriver<'a> {
        fn new(vocab: &'a Vocab) -> Self {
            Self {
                roll: crate::serve_cont_roll::ContRoll::new(),
                slots: std::collections::HashMap::new(),
                vocab,
                next_user: 1,
            }
        }

        fn push(&mut self, mut slot: JobSlot<'a>) -> usize {
            let user = self.next_user;
            self.next_user += 1;
            if let Some(admit) = slot.admit.as_mut() {
                admit.user = user;
            }
            self.roll.enqueue(user);
            self.slots.insert(user, slot);
            user
        }
    }

    impl ContDriver for RollDriver<'_> {
        fn admit(&mut self) -> Option<ContAdmit> {
            let user = self.roll.admit()?;
            self.slots.get_mut(&user)?.admit.take()
        }

        fn on_token(&mut self, user: usize, token: i32) -> bool {
            let Some(slot) = self.slots.get_mut(&user) else {
                return false;
            };
            slot.on_token(self.vocab, token)
        }

        fn on_done(
            &mut self,
            user: usize,
            tokens: &[i32],
            finish: i32,
            decode_ms: f64,
            decode_tokens: i32,
            decode_steps: i32,
        ) {
            let Some(slot) = self.slots.get_mut(&user) else {
                return;
            };
            slot.done(tokens, finish, decode_ms, decode_tokens, decode_steps);
            self.roll.complete(user);
        }

        fn sample_override(&mut self, user: usize) -> i32 {
            let Some(slot) = self.slots.get_mut(&user) else {
                return CONT_SAMPLE_NONE;
            };
            slot.sample_override()
        }

        fn alive(&mut self, user: usize) -> bool {
            self.slots
                .get_mut(&user)
                .is_none_or(|slot| slot.transport_alive())
        }

        fn on_admitted(&mut self, user: usize, n_cached: i32, n_computed: i32, bank: i32) -> bool {
            let Some(slot) = self.slots.get_mut(&user) else {
                return false;
            };
            slot.admitted(n_cached, n_computed, bank)
        }
    }

    struct LaneProbe<'a, 'm> {
        batch: &'a BatchCtx<'m>,
        host: &'a ContHost<'m>,
    }

    impl ContProbe for LaneProbe<'_, '_> {
        fn prompt_tokens(
            &self,
            parsed: &ParsedRequest,
        ) -> Result<(Vec<u8>, Vec<i32>), GenerateError> {
            let prompt = render_prompt(parsed, self.host.model_id)?;
            let tokens = match parsed.kind {
                ReqKind::Completion => self
                    .host
                    .vocab
                    .encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
                ReqKind::Chat => self.host.vocab.encode_rendered_bytes(&prompt),
            };
            let tokens = prepare_qwen_images(self.host.model_id, parsed, None, tokens)?.tokens;
            Ok((prompt, tokens))
        }

        fn seq_cap(&self) -> i32 {
            self.batch.seq_cap()
        }

        fn bank_live(&self, bank: i32) -> Option<(u64, i32)> {
            let snapshot = self.batch.bank_snapshot(bank).ok()?;
            let frontier = i32::try_from(snapshot.tokens.len()).ok()?;
            (frontier > 0).then_some((snapshot.generation, frontier))
        }
    }

    struct RollingSlot {
        key: usize,
        cors: bool,
        job: JobSlot<'static>,
    }

    struct RollingDriver<'a, 'm> {
        host: &'a mut ContHost<'m>,
        batch: &'a BatchCtx<'m>,
        source: &'a mut dyn ContSource,
        bank_hold_retry: &'a mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&'a mut KvStore>,
        slots: std::collections::HashMap<usize, RollingSlot>,
        pending: Option<ContOwnedWork>,
        results: Vec<ContOwnedResult>,
        next_user: usize,
    }

    impl RollingDriver<'_, '_> {
        fn reserve(&self) -> crate::serve_cont_roll::RollReserve {
            let mut reserve = crate::serve_cont_roll::RollReserve::new();
            for slot in self.slots.values() {
                if let Some(bank) = slot.job.bank {
                    reserve.note_place(bank.saturating_add(1));
                }
            }
            reserve
        }

        fn finish_stranded(&mut self, error: String) {
            if let Some(work) = self.pending.take() {
                self.results.push(ContOwnedResult {
                    key: work.key,
                    result: Err(GenerateError::Engine(error.clone())),
                });
            }
            for (_, slot) in std::mem::take(&mut self.slots) {
                let result = self.host.finish_driven(
                    self.batch,
                    slot.job,
                    Some(error.clone()),
                    self.store.as_deref_mut(),
                    slot.cors,
                    None,
                );
                self.results.push(ContOwnedResult {
                    key: slot.key,
                    result,
                });
            }
        }
    }

    impl ContDriver for RollingDriver<'_, '_> {
        fn admit(&mut self) -> Option<ContAdmit> {
            loop {
                let work = match self.pending.take() {
                    Some(work) => work,
                    None => {
                        let probe = LaneProbe {
                            batch: self.batch,
                            host: self.host,
                        };
                        self.source.next(&probe)?
                    }
                };
                let reserve = self.reserve();
                if work
                    .parsed
                    .directed_bank
                    .is_some_and(|bank| reserve.contains(bank))
                {
                    self.pending = Some(work);
                    return None;
                }
                let key = work.key;
                let cors = work.cors;
                let mut work = work;
                let prepared = {
                    let borrowed = ContWork {
                        parsed: &work.parsed,
                        prepared: work.prepared.as_ref(),
                        job_id: &work.job_id,
                        created: work.created,
                        cors: work.cors,
                        default_tokens: work.default_tokens,
                        t_arrive: work.t_arrive,
                        out: &mut *work.out,
                    };
                    self.host.prepare_slot(
                        self.batch,
                        &borrowed,
                        self.bank_hold_retry,
                        self.store.as_deref_mut(),
                        &reserve,
                    )
                };
                let slot = match prepared {
                    Ok(slot) => slot,
                    Err(error) => {
                        self.results.push(ContOwnedResult {
                            key,
                            result: Err(error),
                        });
                        continue;
                    }
                };
                let user = self.next_user;
                self.next_user += 1;
                let mut job = slot.into_slot(work.out);
                let admit = job.admit.as_mut()?;
                admit.user = user;
                let result = job.admit.take();
                self.slots.insert(user, RollingSlot { key, cors, job });
                return result;
            }
        }

        fn on_token(&mut self, user: usize, token: i32) -> bool {
            self.slots
                .get_mut(&user)
                .is_some_and(|slot| slot.job.on_token(self.host.vocab, token))
        }

        fn on_done(
            &mut self,
            user: usize,
            tokens: &[i32],
            finish: i32,
            decode_ms: f64,
            decode_tokens: i32,
            decode_steps: i32,
        ) {
            let Some(mut slot) = self.slots.remove(&user) else {
                return;
            };
            slot.job
                .done(tokens, finish, decode_ms, decode_tokens, decode_steps);
            let key = slot.key;
            let source = &mut self.source;
            let mut publish = |outcome: &GenerateOutcome| source.publish(key, outcome);
            let result = self.host.finish_driven(
                self.batch,
                slot.job,
                None,
                self.store.as_deref_mut(),
                slot.cors,
                Some(&mut publish),
            );
            drop(publish);
            self.source.settled(key, &result);
            self.results.push(ContOwnedResult { key, result });
        }

        fn sample_override(&mut self, user: usize) -> i32 {
            self.slots
                .get_mut(&user)
                .map_or(CONT_SAMPLE_NONE, |slot| slot.job.sample_override())
        }

        fn alive(&mut self, user: usize) -> bool {
            self.slots
                .get_mut(&user)
                .is_none_or(|slot| slot.job.transport_alive())
        }

        fn on_admitted(&mut self, user: usize, n_cached: i32, n_computed: i32, bank: i32) -> bool {
            self.slots
                .get_mut(&user)
                .is_some_and(|slot| slot.job.admitted(n_cached, n_computed, bank))
        }
    }

    impl<'m> ContLane<'m> {
        pub fn new(
            batch: BatchCtx<'m>,
            vocab: &'m Vocab,
            model_id: i32,
            quant_bits: i32,
            ctx: i32,
            eos: i32,
        ) -> Self {
            let max_seq = usize::try_from(batch.max_seq().max(0)).unwrap_or(0);
            let warm_pin_min = crate::serve::env_i32_bound("DS4_SERVER_PIN_MIN_TOKENS", 65536);
            let warm_persist_min = crate::serve::env_i32_bound(
                "DS4_SERVER_PERSIST_MIN_TOKENS",
                DEFAULT_BANK_PERSIST_MIN_TOKENS,
            );
            let warm_fork = std::env::var_os("DS4_SERVER_FORK").is_none_or(|value| value != "0");
            let partial_requested =
                std::env::var_os("DS4_SERVER_FORK_PARTIAL").is_none_or(|value| value != "0");
            let warm_fork_partial = partial_requested && batch.supports_partial_reuse();
            let warm_disk_partial = warm_fork_partial
                && std::env::var_os("DS4_SERVER_DISK_PARTIAL").is_none_or(|value| value != "0");
            let warm_partial_min =
                crate::serve::env_i32_bound("DS4_SERVER_FORK_PARTIAL_MIN", 192).max(136);
            let warm_checkpoint =
                std::env::var_os("DS4_SERVER_BANK_CHECKPOINT").is_none_or(|value| value != "0");
            if partial_requested && !warm_fork_partial {
                eprintln!("ds4-server-rs: partial bank reuse disabled by model runtime");
            }
            Self {
                batch,
                host: ContHost {
                    vocab,
                    model_id,
                    quant_bits,
                    ctx,
                    eos,
                    warm: (0..max_seq).map(|_| WarmBank::default()).collect(),
                    warm_clock: 0,
                    warm_fork,
                    warm_fork_partial,
                    warm_disk_partial,
                    warm_partial_min,
                    warm_pin_min,
                    warm_persist_min,
                    warm_checkpoint,
                    tool_memory: ToolMemory::default(),
                    memgov: Box::new(crate::serve_cont_roll::AdmitAlways),
                    stop_requested: None,
                },
            }
        }
    }

    impl ContHost<'_> {
        fn identity(&self) -> Option<(u8, u8, u32)> {
            crate::generate::kv_identity(self.model_id, self.quant_bits, self.ctx)
        }

        fn note_use(&mut self, bank: usize) {
            self.warm_clock = self.warm_clock.wrapping_add(1);
            if let Some(state) = self.warm.get_mut(bank) {
                state.last_use = self.warm_clock;
            }
        }

        fn warm_full_plan(
            &mut self,
            batch: &BatchCtx<'_>,
            prompt: &[u8],
            cache_prompt: Option<&[u8]>,
        ) -> Option<WarmAdmitPlan> {
            let matched = warm_match_pick(&self.warm, prompt, cache_prompt)?;
            let source = matched.bank;
            let snapshot = batch.bank_snapshot(i32::try_from(source).ok()?).ok()?;
            let (tokens, cached) = warm_admit_tokens(
                self.warm.get(source)?,
                prompt,
                cache_prompt,
                &snapshot.tokens,
                snapshot.generation,
                batch.seq_cap(),
                matched.exact,
                |suffix| self.vocab.encode_rendered_bytes(suffix),
            )?;
            self.warm[source].committed_tokens = cached;
            Some(WarmAdmitPlan {
                source,
                tokens,
                cached,
                partial: false,
            })
        }

        fn warm_partial_plan(
            &mut self,
            batch: &BatchCtx<'_>,
            prompt: &[u8],
            cache_prompt: Option<&[u8]>,
            cache_spans: &[ImageCacheSpan],
            prompt_tokens: &[i32],
        ) -> Option<WarmAdmitPlan> {
            if !self.warm_fork_partial {
                return None;
            }
            let min_prefix = usize::try_from(self.warm_partial_min).ok()?;
            let (source, cache_lcp) =
                warm_partial_match_pick(&self.warm, prompt, cache_prompt, min_prefix)?;
            let snapshot = batch.bank_snapshot(i32::try_from(source).ok()?).ok()?;
            let (tokens, cached) = warm_partial_admit_tokens(
                self.warm.get(source)?,
                prompt_tokens,
                &snapshot.tokens,
                snapshot.generation,
                self.warm_partial_min,
                batch.seq_cap(),
                qwen_image_cache_token_cap(cache_spans, cache_lcp),
            )?;
            self.warm[source].committed_tokens = i32::try_from(snapshot.tokens.len()).ok()?;
            Some(WarmAdmitPlan {
                source,
                tokens,
                cached,
                partial: true,
            })
        }

        fn warm_plan(
            &mut self,
            batch: &BatchCtx<'_>,
            prompt: &[u8],
            cache_prompt: Option<&[u8]>,
            cache_spans: &[ImageCacheSpan],
            prompt_tokens: &[i32],
        ) -> Option<WarmAdmitPlan> {
            let full = self.warm_full_plan(batch, prompt, cache_prompt);
            let partial =
                self.warm_partial_plan(batch, prompt, cache_prompt, cache_spans, prompt_tokens);
            match (full, partial) {
                (Some(full), Some(partial)) if partial.cached > full.cached => Some(partial),
                (Some(full), _) => Some(full),
                (None, partial) => partial,
            }
        }

        fn protected_banks(
            &self,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        ) -> (Vec<bool>, Option<i32>) {
            let mut protected = Vec::with_capacity(self.warm.len());
            let mut retry_min = None;
            for (bank, state) in self.warm.iter().enumerate() {
                let live = state
                    .record
                    .as_ref()
                    .map(|record| (record.generation, state.committed_tokens));
                let retry = bank_hold_retry(i32::try_from(bank).unwrap_or(-1), live);
                protected.push(retry.is_some());
                if let Some(retry) = retry {
                    retry_min = Some(retry_min.map_or(retry, |current: i32| current.min(retry)));
                }
            }
            (protected, retry_min)
        }

        fn disk_victim(&self, protected: &[bool], disk_tokens: u32) -> Option<usize> {
            if let Some(bank) =
                warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, false)
            {
                return Some(bank);
            }
            if self.warm_pin_min <= 0 || disk_tokens < self.warm_pin_min as u32 {
                return None;
            }
            let bank = warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, true)?;
            (self.warm[bank].committed_tokens < self.warm_pin_min).then_some(bank)
        }

        fn disk_plan(
            &mut self,
            batch: &BatchCtx<'_>,
            store: &mut KvStore,
            prompt: &[u8],
            cache_prompt: Option<&[u8]>,
            protected: &[bool],
        ) -> Option<WarmAdmitPlan> {
            let identity = self.identity()?;
            let request_key = cache_prompt.unwrap_or(prompt);
            let identity_flags = cache_prompt
                .is_some()
                .then_some(EXT_IMAGE_PIXELS_V2)
                .unwrap_or(0);
            let (path, envelope) = store
                .bank_text_prefix_candidate_identity(
                    request_key,
                    identity.0,
                    identity.1,
                    identity.2,
                    identity_flags,
                )
                .ok()??;
            let target = self.disk_victim(protected, envelope.header.tokens)?;
            if !disk_restore_target_allowed(
                &self.warm,
                target,
                envelope.header.tokens,
                self.warm_pin_min,
            ) {
                return None;
            }
            let mut record = restored_record(envelope.text, 0, envelope.header.ext_flags)?;
            let snapshot = match batch.load_bank_payload_range(
                i32::try_from(target).ok()?,
                &path,
                envelope.payload_offset,
                envelope.header.payload_bytes,
            ) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    eprintln!("ds4-server-rs: bank restore skipped: {error}");
                    return None;
                }
            };
            if usize::try_from(envelope.header.tokens).ok() != Some(snapshot.tokens.len()) {
                let _ = store.discard_bank(&path);
                return None;
            }
            let committed = i32::try_from(snapshot.tokens.len()).ok()?;
            record.generation = snapshot.generation;
            self.warm[target].record = Some(record);
            self.warm[target].committed_tokens = committed;
            self.warm[target].stored_tokens = committed;
            self.note_use(target);
            let _ = store.touch_hit(&path);
            let (tokens, cached) = warm_admit_tokens(
                &self.warm[target],
                prompt,
                cache_prompt,
                &snapshot.tokens,
                snapshot.generation,
                batch.seq_cap(),
                false,
                |suffix| self.vocab.encode_rendered_bytes(suffix),
            )?;
            Some(WarmAdmitPlan {
                source: target,
                tokens,
                cached,
                partial: false,
            })
        }

        fn disk_partial_plan(
            &mut self,
            batch: &BatchCtx<'_>,
            store: &mut KvStore,
            prompt: &[u8],
            cache_prompt: Option<&[u8]>,
            cache_spans: &[ImageCacheSpan],
            prompt_tokens: &[i32],
            protected: &[bool],
        ) -> Option<WarmAdmitPlan> {
            if !self.warm_disk_partial {
                return None;
            }
            let identity = self.identity()?;
            let min_prefix = usize::try_from(self.warm_partial_min).ok()?;
            let request_key = cache_prompt.unwrap_or(prompt);
            let identity_flags = cache_prompt
                .is_some()
                .then_some(EXT_IMAGE_PIXELS_V2)
                .unwrap_or(0);
            let (path, envelope, cache_lcp) = store
                .bank_text_lcp_candidate_identity(
                    request_key,
                    identity.0,
                    identity.1,
                    identity.2,
                    min_prefix,
                    identity_flags,
                )
                .ok()??;
            let target = self.disk_victim(protected, envelope.header.tokens)?;
            if !disk_restore_target_allowed(
                &self.warm,
                target,
                envelope.header.tokens,
                self.warm_pin_min,
            ) {
                return None;
            }
            let mut record = restored_record(envelope.text, 0, envelope.header.ext_flags)?;
            let snapshot = match batch.load_bank_payload_range(
                i32::try_from(target).ok()?,
                &path,
                envelope.payload_offset,
                envelope.header.payload_bytes,
            ) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    eprintln!("ds4-server-rs: partial bank restore skipped: {error}");
                    return None;
                }
            };
            if usize::try_from(envelope.header.tokens).ok() != Some(snapshot.tokens.len()) {
                let _ = store.discard_bank(&path);
                return None;
            }
            let committed = i32::try_from(snapshot.tokens.len()).ok()?;
            record.generation = snapshot.generation;
            self.warm[target].record = Some(record);
            self.warm[target].committed_tokens = committed;
            self.warm[target].stored_tokens = committed;
            self.note_use(target);
            let _ = store.touch_hit(&path);
            let (tokens, cached) = warm_partial_admit_tokens(
                &self.warm[target],
                prompt_tokens,
                &snapshot.tokens,
                snapshot.generation,
                self.warm_partial_min,
                batch.seq_cap(),
                qwen_image_cache_token_cap(cache_spans, cache_lcp),
            )?;
            Some(WarmAdmitPlan {
                source: target,
                tokens,
                cached,
                partial: true,
            })
        }

        fn retire(
            &mut self,
            batch: &BatchCtx<'_>,
            bank: i32,
            stepper: &ContStepper,
            done_tokens: &[i32],
            engine_finished: bool,
            transport_failed: bool,
            allow_generated_snapshot: bool,
        ) {
            let Ok(bank) = usize::try_from(bank) else {
                return;
            };
            if bank >= self.warm.len() {
                return;
            }
            let Ok(snapshot) = batch.bank_snapshot(bank as i32) else {
                self.warm[bank].record = None;
                return;
            };
            self.warm[bank].committed_tokens = snapshot.tokens.len() as i32;
            self.note_use(bank);
            if snapshot.tokens.is_empty() {
                self.warm[bank].record = None;
                return;
            }
            let plan = if think_mode_enabled(stepper.think_mode) && !done_tokens.is_empty() {
                thinking_bank_retire_key(
                    &stepper.prompt,
                    done_tokens,
                    engine_finished,
                    transport_failed || stepper.acc.verdict.is_some(),
                    stepper.prompt_preserves_reasoning,
                    stepper.acc.track_tools,
                    stepper.acc.saw_tool_start,
                    syntax_for_model_id(stepper.model_id),
                    stepper.req.chat_format,
                    |token| self.vocab.token_text(token),
                )
            } else {
                bank_retire_key(
                    &stepper.prompt,
                    &snapshot.tokens,
                    done_tokens,
                    allow_generated_snapshot,
                    |token| self.vocab.token_text(token),
                )
                .map(|(text, retained_existing)| BankRetireKey {
                    text,
                    exact_text: None,
                    partial_only: false,
                    retained_existing,
                })
            };
            let Some(plan) = plan else {
                self.warm[bank].record = None;
                return;
            };
            if plan.partial_only && !self.warm_fork_partial {
                self.warm[bank].record = None;
                return;
            }
            if plan.retained_existing {
                self.warm[bank].stored_tokens = self.warm[bank]
                    .stored_tokens
                    .min(snapshot.tokens.len() as i32);
            }
            let cache_text = stepper.cache_prompt.as_deref().and_then(|cache_prompt| {
                extended_image_cache_key(&stepper.prompt, cache_prompt, &plan.text)
            });
            let exact_cache_text = stepper.cache_prompt.as_deref().and_then(|cache_prompt| {
                plan.exact_text.as_deref().and_then(|exact| {
                    extended_image_cache_key(&stepper.prompt, cache_prompt, exact)
                })
            });
            if stepper.cache_prompt.is_some()
                && (cache_text.is_none()
                    || (plan.exact_text.is_some() && exact_cache_text.is_none()))
            {
                self.warm[bank].record = None;
                return;
            }
            let mut record = retired_record(
                self.warm[bank].record.as_ref(),
                plan.text,
                cache_text,
                snapshot.generation,
            );
            if let Some(record) = record.as_mut() {
                record.exact_text = plan.exact_text;
                record.exact_cache_text = exact_cache_text;
                record.partial_only = plan.partial_only;
            }
            self.warm[bank].record = record;
        }

        /// Live evict: same `save_bank_record` path as `persist_bank`,
        /// then drop the warm record. Pinned banks are left untouched.
        fn evict_bank(
            &mut self,
            batch: &BatchCtx<'_>,
            store: Option<&mut KvStore>,
            bank: usize,
            pinned: bool,
        ) -> bool {
            if pinned {
                return false;
            }
            if let Some(store) = store {
                self.persist_bank(
                    batch,
                    store,
                    bank,
                    KvReason::BankEvict,
                    self.warm_persist_min,
                    false,
                );
            }
            if let Some(warm) = self.warm.get_mut(bank) {
                warm.record = None;
            }
            true
        }

        fn persist_bank(
            &mut self,
            batch: &BatchCtx<'_>,
            store: &mut KvStore,
            bank: usize,
            reason: KvReason,
            min_committed: i32,
            due_only: bool,
        ) {
            if bank >= self.warm.len() {
                return;
            }
            let Some(identity) = self.identity() else {
                return;
            };
            let Ok(bank_i32) = i32::try_from(bank) else {
                return;
            };
            let Ok(snapshot) = batch.bank_snapshot(bank_i32) else {
                return;
            };
            let Ok(committed) = i32::try_from(snapshot.tokens.len()) else {
                return;
            };
            if self.warm[bank].record.as_ref().is_none_or(|record| {
                record.generation != snapshot.generation || record.partial_only
            }) || !bank_persist_eligible(committed, min_committed)
                || (due_only
                    && !bank_checkpoint_due_from_host(
                        &store.opt,
                        HostKvView {
                            live_tokens: committed,
                            stored_tokens: self.warm[bank].stored_tokens,
                        },
                    ))
            {
                return;
            }
            let Some(trailer) = self.warm[bank]
                .record
                .as_ref()
                .and_then(|record| self.tool_memory.checkpoint(&record.text))
            else {
                eprintln!(
                    "ds4-server-rs: bank checkpoint skipped bank={bank}: tool-map exceeds bound"
                );
                return;
            };
            if let Some(record) = self.warm[bank].record.as_mut() {
                record.trailer = trailer;
            }
            let state = &mut self.warm[bank];
            if let Err(error) = save_bank_record(
                store,
                state,
                identity,
                committed,
                snapshot.generation,
                reason,
                |path| {
                    batch
                        .save_bank_payload(bank_i32, path)
                        .map_err(|error| GenerateError::Engine(error.to_string()))
                },
            ) {
                eprintln!(
                    "ds4-server-rs: bank checkpoint failed bank={bank} reason={reason:?}: {error}"
                );
            }
        }

        fn place_warm(
            &mut self,
            batch: &BatchCtx<'_>,
            plan: &WarmAdmitPlan,
            protected: &[bool],
            mut store: Option<&mut KvStore>,
        ) -> Option<WarmPlacement> {
            let placement = warm_placement(
                &self.warm,
                protected,
                plan.source,
                plan.cached,
                self.warm_pin_min,
                self.warm_fork,
            )?;
            if placement.fork && protected.get(placement.target).copied().unwrap_or(false) {
                return None;
            }
            self.note_use(placement.source);
            if placement.fork {
                let _ = self.evict_bank(batch, store.as_deref_mut(), placement.target, false);
                let stored = if plan.partial {
                    self.warm[placement.source].stored_tokens.min(plan.cached)
                } else {
                    self.warm[placement.source].stored_tokens
                };
                self.warm[placement.target].stored_tokens = stored;
            } else {
                if plan.partial {
                    let _ = self.evict_bank(batch, store.as_deref_mut(), placement.source, false);
                    self.warm[placement.source].stored_tokens =
                        self.warm[placement.source].stored_tokens.min(plan.cached);
                } else {
                    self.warm[placement.source].record = None;
                }
            }
            Some(placement)
        }

        fn place_cold(
            &mut self,
            batch: &BatchCtx<'_>,
            protected: &[bool],
            store: Option<&mut KvStore>,
        ) -> Option<usize> {
            let target = warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, true)?;
            if !self.evict_bank(
                batch,
                store,
                target,
                protected.get(target).copied().unwrap_or(false),
            ) {
                return None;
            }
            self.warm[target].committed_tokens = 0;
            self.warm[target].stored_tokens = 0;
            Some(target)
        }

        fn shutdown_banks(&mut self, batch: &BatchCtx<'_>, store: &mut KvStore) {
            for bank in 0..self.warm.len() {
                self.persist_bank(batch, store, bank, KvReason::BankShutdown, 1, false);
            }
        }

        fn prepare_slot(
            &mut self,
            batch: &BatchCtx<'_>,
            work: &ContWork<'_>,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            mut store: Option<&mut KvStore>,
            reserve: &crate::serve_cont_roll::RollReserve,
        ) -> Result<PreparedSlot, GenerateError> {
            let mut parsed = work.parsed.clone();
            let attached_tool_blocks = if parsed.kind == ReqKind::Chat {
                if let Ok(model_id) = u8::try_from(self.model_id) {
                    if let Some(store) = store.as_deref() {
                        self.tool_memory
                            .restore_store(store, model_id, &parsed.messages);
                    }
                }
                self.tool_memory.attach(&mut parsed.messages)
            } else {
                0
            };
            let parsed = &parsed;
            let prepared = work
                .prepared
                .filter(|_| can_reuse_cont_prompt(attached_tool_blocks, !parsed.images.is_empty()));
            let (prompt, tokens) = match prepared {
                Some(prepared) => (prepared.prompt.clone(), prepared.tokens.clone()),
                None => {
                    let prompt = render_prompt(parsed, self.model_id)?;
                    let tokens = match parsed.kind {
                        ReqKind::Completion => self
                            .vocab
                            .encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
                        ReqKind::Chat => self.vocab.encode_rendered_bytes(&prompt),
                    };
                    (prompt, tokens)
                }
            };
            let directed = parsed.directed_bank.filter(|bank| *bank >= 0);
            let (tokens, images, cache_prompt, cache_spans, directed_cached) = if let Some(bank) =
                directed
            {
                if reserve.contains(bank) {
                    return Err(GenerateError::Unsupported(
                        "continuation bank is already in flight",
                    ));
                }
                let suffix = render_live_tool_tail(
                    syntax_for_model_id(self.model_id),
                    parsed.api,
                    &parsed.messages,
                    parsed.think_mode,
                )?;
                let snapshot = batch.bank_snapshot(bank).map_err(|_| {
                    GenerateError::Unsupported("continuation bank is no longer live")
                })?;
                let (tokens, cached) = live_continuation_tokens(
                    &snapshot.tokens,
                    &suffix,
                    batch.seq_cap(),
                    |suffix| self.vocab.encode_rendered_bytes(suffix),
                )
                .ok_or(GenerateError::Unsupported(
                    "continuation suffix is empty or exceeds the sequence capacity",
                ))?;
                (tokens, Vec::new(), None, Vec::new(), Some(cached))
            } else {
                let prepared = prepare_qwen_images(self.model_id, parsed, Some(&prompt), tokens)?;
                (
                    prepared.tokens,
                    prepared.images,
                    prepared.cache_prompt,
                    prepared.cache_spans,
                    None,
                )
            };
            let prompt_n = tokens.len() as i32;
            let (mut stepper, head) = ContStepper::new(
                parsed,
                self.model_id,
                work.job_id,
                work.created,
                work.cors,
                work.default_tokens,
                prompt,
                prompt_n,
                batch.seq_cap(),
            );
            stepper.cache_prompt = cache_prompt;
            stepper.image_cache_spans = cache_spans;
            let (temperature, top_k, top_p, min_p) = stepper.sampling(parsed);
            let capture_done = bank_scope(parsed);
            crate::serve_cont_roll::charge_roll_admit(
                &mut *self.memgov,
                prompt_n,
                stepper.max_tokens,
            )
            .map_err(|_| GenerateError::Unsupported(crate::serve_cont_roll::CONT_ADMIT_REFUSED))?;
            let mut admit = if let Some(bank) = directed {
                let mut admit = ContAdmit::cold(1, tokens, stepper.max_tokens.max(1));
                admit.place_bank = bank.saturating_add(1);
                admit.n_cached = directed_cached.unwrap_or(0);
                admit
            } else {
                let (hold, hold_retry) = self.protected_banks(bank_hold_retry);
                let protected = reserve.protect(&hold);
                let warm = if capture_done {
                    self.warm_plan(
                        batch,
                        &stepper.prompt,
                        stepper.cache_prompt.as_deref(),
                        &stepper.image_cache_spans,
                        &tokens,
                    )
                    .or_else(|| {
                        store.as_deref_mut().and_then(|store| {
                            self.disk_plan(
                                batch,
                                store,
                                &stepper.prompt,
                                stepper.cache_prompt.as_deref(),
                                &protected,
                            )
                        })
                    })
                    .or_else(|| {
                        store.as_deref_mut().and_then(|store| {
                            self.disk_partial_plan(
                                batch,
                                store,
                                &stepper.prompt,
                                stepper.cache_prompt.as_deref(),
                                &stepper.image_cache_spans,
                                &tokens,
                                &protected,
                            )
                        })
                    })
                } else {
                    None
                };
                let placement = warm.as_ref().and_then(|plan| {
                    self.place_warm(batch, plan, &protected, store.as_deref_mut())
                });
                if let (Some(plan), Some(placement)) = (warm, placement) {
                    let mut admit = ContAdmit::cold(1, plan.tokens, stepper.max_tokens.max(1));
                    admit.place_bank = i32::try_from(placement.target + 1).unwrap_or(0);
                    admit.n_cached = plan.cached;
                    if placement.fork || plan.partial {
                        admit.fork_bank = i32::try_from(placement.source + 1).unwrap_or(0);
                    }
                    admit
                } else {
                    let target = self
                        .place_cold(batch, &protected, store.as_deref_mut())
                        .ok_or(GenerateError::ContinuationHold {
                            retry_after: hold_retry.unwrap_or(1),
                        })?;
                    let mut admit = ContAdmit::cold(1, tokens, stepper.max_tokens.max(1));
                    admit.place_bank = i32::try_from(target + 1).unwrap_or(0);
                    admit
                }
            };
            admit.eos = self.eos;
            admit.temperature = temperature;
            admit.top_k = top_k;
            admit.top_p = top_p;
            admit.min_p = min_p;
            admit.seed = parsed.seed;
            admit.images = images;
            Ok(PreparedSlot {
                admit,
                head,
                stepper,
                capture_done,
                t_arrive: work.t_arrive,
                stop_requested: self.stop_requested,
            })
        }

        fn finish_driven(
            &mut self,
            batch: &BatchCtx<'_>,
            mut job: JobSlot<'_>,
            native_err: Option<String>,
            mut store: Option<&mut KvStore>,
            cors: bool,
            publish: Option<&mut dyn FnMut(&GenerateOutcome)>,
        ) -> Result<GenerateOutcome, GenerateError> {
            let timings = {
                let completion = job.stepper.completion();
                match job.t_first {
                    Some(first) if completion > 0 => ReqTimings {
                        valid: true,
                        ttft_ms: first.duration_since(job.t_arrive).as_secs_f64() * 1e3,
                        prefill_ms: first
                            .duration_since(job.t_admit.unwrap_or(job.t_arrive))
                            .as_secs_f64()
                            * 1e3,
                        decode_ms: job.decode_ms,
                        prefill_tokens: job.n_computed,
                        prefill_cached: job.n_cached,
                        decode_tokens: job.decode_tokens,
                        decode_steps: job.decode_steps,
                    },
                    _ => ReqTimings::default(),
                }
            };
            let engine_eos = job.engine_eos && !job.host_abort;
            let n_cached = job.n_cached;
            let n_computed = job.n_computed;
            let io_ok = !job.io_failed;
            let done_called = job.t_done.is_some();
            let done_tokens = std::mem::take(&mut job.done_tokens);
            let admitted = job.admitted;
            let actual_bank = job.bank;
            let capture_done = job.capture_done;
            let allow_generated_snapshot =
                native_err.is_none() && io_ok && job.stepper.has_complete_tool_turn();
            let retired = retired_bank(
                capture_done,
                admitted,
                done_called,
                actual_bank,
                batch.max_seq(),
            );
            if let Some(bank) = retired {
                self.retire(
                    batch,
                    bank,
                    &job.stepper,
                    &done_tokens,
                    job.engine_eos,
                    !io_ok,
                    allow_generated_snapshot,
                );
                if self.warm_checkpoint {
                    if let Some(store) = store.as_deref_mut() {
                        self.persist_bank(
                            batch,
                            store,
                            bank as usize,
                            KvReason::BankCheckpoint,
                            self.warm_persist_min,
                            true,
                        );
                    }
                }
            }
            if let Some(error) = native_err {
                if admitted && io_ok && job.stepper.req.stream {
                    let failure = job.stepper.fail(&error);
                    job.push(&failure);
                    return if job.io_failed {
                        Err(GenerateError::Io)
                    } else {
                        Err(GenerateError::Streamed(error))
                    };
                }
                return Err(GenerateError::Engine(error));
            }
            if !admitted {
                return Err(GenerateError::Unsupported(
                    "continuous admission rejected; serial fallback",
                ));
            }
            if !io_ok {
                return Err(GenerateError::Io);
            }
            let (tail, mut outcome) = job
                .stepper
                .finalize(engine_eos, n_cached, n_computed, timings, cors);
            if let Some(bank) = actual_bank {
                if let Ok(snapshot) = batch.bank_snapshot(bank) {
                    outcome.bank = Some(bank);
                    outcome.generation = snapshot.generation;
                    outcome.frontier = i32::try_from(snapshot.tokens.len()).unwrap_or(0);
                }
            }
            let mut tool_remembered = false;
            if let Some((calls, exact)) = job.stepper.take_tool_replay() {
                tool_remembered = self.tool_memory.remember(&calls, &exact) > 0;
            }
            if tool_remembered && self.warm_checkpoint {
                if let (Some(store), Some(bank)) = (
                    store.as_deref_mut(),
                    retired.and_then(|bank| usize::try_from(bank).ok()),
                ) {
                    self.persist_bank(
                        batch,
                        store,
                        bank,
                        KvReason::BankCheckpoint,
                        self.warm_persist_min,
                        false,
                    );
                }
            }
            if let Some(publish) = publish {
                publish(&outcome);
            }
            if !tail.is_empty() {
                job.out.write_all(&tail).map_err(|_| GenerateError::Io)?;
                let _ = job.out.flush();
            }
            Ok(outcome)
        }
    }

    impl StaticExec for ContLane<'_> {
        fn generate_static(
            &mut self,
            jobs: &[StaticJob<'_>],
        ) -> Result<Vec<StaticRow>, GenerateError> {
            BatchStatic::new(&mut self.batch).generate_static(jobs)
        }

        fn ctx_max_seq(&self) -> i32 {
            self.batch.max_seq()
        }

        fn coalesce_limits(&self) -> CoalesceLimits {
            CoalesceLimits {
                cap: self.batch.max_seq().max(1) as usize,
                max_tok_total: 0,
            }
        }
    }

    impl ContExec for ContLane<'_> {
        fn model_id(&self) -> i32 {
            self.host.model_id
        }

        fn set_stop_requested(&mut self, stop_requested: Option<fn() -> bool>) {
            self.host.stop_requested = stop_requested;
        }

        fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
            Some(self)
        }

        fn seq_cap(&self) -> i32 {
            self.batch.seq_cap()
        }

        fn max_seq(&self) -> i32 {
            self.batch.max_seq()
        }

        fn trim_idle_banks(&mut self, want_bytes: u64) -> u64 {
            self.batch.trim_free(want_bytes)
        }

        fn encode_chat(&self, rendered: &[u8]) -> Vec<i32> {
            self.host.vocab.encode_rendered_bytes(rendered)
        }

        fn encode_text(&self, text: &str) -> Vec<i32> {
            self.host.vocab.encode_text(text)
        }

        fn prepare_tokens(
            &self,
            parsed: &ParsedRequest,
            _prompt: &[u8],
            tokens: Vec<i32>,
        ) -> Result<Vec<i32>, GenerateError> {
            prepare_qwen_images(self.host.model_id, parsed, None, tokens)
                .map(|prepared| prepared.tokens)
        }

        fn bank_live(&self, bank: i32) -> Option<(u64, i32)> {
            let snapshot = self.batch.bank_snapshot(bank).ok()?;
            let frontier = i32::try_from(snapshot.tokens.len()).ok()?;
            if frontier <= 0 {
                return None;
            }
            Some((snapshot.generation, frontier))
        }

        fn generate(
            &mut self,
            parsed: &ParsedRequest,
            job_id: &str,
            created: i64,
            cors: bool,
            default_tokens: i32,
            t_arrive: Instant,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            store: Option<&mut KvStore>,
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            self.generate_prepared(
                parsed,
                None,
                job_id,
                created,
                cors,
                default_tokens,
                t_arrive,
                bank_hold_retry,
                store,
                out,
            )
        }

        fn generate_prepared(
            &mut self,
            parsed: &ParsedRequest,
            prepared: Option<&ContPreparedPrompt>,
            job_id: &str,
            created: i64,
            cors: bool,
            default_tokens: i32,
            t_arrive: Instant,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            mut store: Option<&mut KvStore>,
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            let work = ContWork {
                parsed,
                prepared,
                job_id,
                created,
                cors,
                default_tokens,
                t_arrive,
                out,
            };
            let prepared = self.host.prepare_slot(
                &self.batch,
                &work,
                bank_hold_retry,
                store.as_deref_mut(),
                &crate::serve_cont_roll::RollReserve::new(),
            )?;
            let mut driver = RollDriver::new(self.host.vocab);
            let user = driver.push(prepared.into_slot(Box::new(work.out)));
            let native_err = self
                .batch
                .continuous_generate(&mut driver)
                .err()
                .map(|error| error.to_string());
            let job = driver.slots.remove(&user).ok_or_else(|| {
                GenerateError::Engine("continuous driver lost the admitted job".into())
            })?;
            self.host
                .finish_driven(&self.batch, job, native_err, store, cors, None)
        }

        fn generate_batch(
            &mut self,
            jobs: Vec<ContWork<'_>>,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            mut store: Option<&mut KvStore>,
        ) -> Vec<Result<GenerateOutcome, GenerateError>> {
            let count = jobs.len();
            let mut results: Vec<Option<Result<GenerateOutcome, GenerateError>>> =
                (0..count).map(|_| None).collect();
            let mut reserve = crate::serve_cont_roll::RollReserve::new();
            let mut prepared = Vec::with_capacity(count);
            for (index, work) in jobs.into_iter().enumerate() {
                let cors = work.cors;
                match self.host.prepare_slot(
                    &self.batch,
                    &work,
                    bank_hold_retry,
                    store.as_deref_mut(),
                    &reserve,
                ) {
                    Ok(slot) => {
                        reserve.note_place(slot.admit.place_bank);
                        prepared.push((index, cors, slot, work.out));
                    }
                    Err(error) => results[index] = Some(Err(error)),
                }
            }
            let mut driver = RollDriver::new(self.host.vocab);
            let order: Vec<_> = prepared
                .into_iter()
                .map(|(index, cors, slot, out)| {
                    let user = driver.push(slot.into_slot(Box::new(out)));
                    (index, user, cors)
                })
                .collect();
            let native_err = (!order.is_empty()).then(|| {
                self.batch
                    .continuous_generate(&mut driver)
                    .err()
                    .map(|error| error.to_string())
            });
            let native_err = native_err.flatten();
            for (index, user, cors) in order {
                results[index] = Some(match driver.slots.remove(&user) {
                    Some(job) => self.host.finish_driven(
                        &self.batch,
                        job,
                        native_err.clone(),
                        store.as_deref_mut(),
                        cors,
                        None,
                    ),
                    None => Err(GenerateError::Engine(
                        "continuous driver lost an admitted job".into(),
                    )),
                });
            }
            results
                .into_iter()
                .map(|result| {
                    result.unwrap_or_else(|| {
                        Err(GenerateError::Engine(
                            "continuous batch lost a job result".into(),
                        ))
                    })
                })
                .collect()
        }

        fn generate_rolling(
            &mut self,
            source: &mut dyn ContSource,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            store: Option<&mut KvStore>,
        ) -> Option<Vec<ContOwnedResult>> {
            let mut driver = RollingDriver {
                host: &mut self.host,
                batch: &self.batch,
                source,
                bank_hold_retry,
                store,
                slots: std::collections::HashMap::new(),
                pending: None,
                results: Vec::new(),
                next_user: 1,
            };
            loop {
                match self.batch.continuous_generate(&mut driver) {
                    Ok(()) if driver.pending.is_some() => continue,
                    Ok(()) => break,
                    Err(error) => {
                        driver.finish_stranded(error.to_string());
                        break;
                    }
                }
            }
            Some(driver.results)
        }

        fn shutdown(&mut self, store: Option<&mut KvStore>) {
            if let Some(store) = store {
                self.host.shutdown_banks(&self.batch, store);
            }
        }
    }
}

impl ContStepper {
    pub fn completion(&self) -> i32 {
        self.acc.completion
    }

    /// Host stop-token verdict (family EOT / eos / role starts), the same
    /// pre-eval check the serial `decode_pass` runs; the stop token's text
    /// is never fed.
    pub fn mark_stop(&mut self) {
        self.finish = "stop";
    }

    #[cfg(any(feature = "native", test))]
    fn stop_for_shutdown(&mut self, stop_requested: Option<fn() -> bool>) -> bool {
        if !stop_requested.is_some_and(|stop| stop()) {
            return false;
        }
        self.finish = "error";
        true
    }
}

#[cfg(test)]
mod bank_tests {
    use super::*;
    use crate::parse::{
        parse_anthropic_request, parse_chat_request, parse_responses_request, ChatMsg, ParseEnv,
    };
    use crate::route::{Api, ReqKind, ThinkMode};
    use ds4_kv::{Options, Reason, Store, EXT_BANK_REPLAY_V1, EXT_TOOL_MAP};
    use std::fs;

    fn shutdown_requested() -> bool {
        true
    }

    #[test]
    fn prepared_prompt_is_reused_only_when_native_inputs_are_unchanged() {
        assert!(can_reuse_cont_prompt(0, false));
        assert!(!can_reuse_cont_prompt(1, false));
        assert!(!can_reuse_cont_prompt(0, true));
    }

    #[test]
    fn shallow_agent_sessions_persist_without_becoming_resident_pinned() {
        let session_tokens = 58_000;
        assert!(bank_persist_eligible(
            session_tokens,
            DEFAULT_BANK_PERSIST_MIN_TOKENS
        ));
        assert!(session_tokens < 65_536);
        assert!(!bank_persist_eligible(session_tokens, 65_536));
    }

    #[test]
    fn anthropic_stream_starts_with_the_admitted_cache_split() {
        let parsed = parse_anthropic_request(
            &ParseEnv::default(),
            r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true}"#,
        )
        .unwrap();
        let (mut stepper, head) = ContStepper::new(
            &parsed,
            0,
            "msg-cache-usage",
            7,
            false,
            16,
            b"prompt".to_vec(),
            278,
            512,
        );

        let head = String::from_utf8(stepper.admitted_head(head, 260, 18)).unwrap();
        assert!(head.contains("\"input_tokens\":0"), "{head}");
        assert!(head.contains("\"cache_read_input_tokens\":260"), "{head}");
        assert!(
            head.contains("\"cache_creation_input_tokens\":18"),
            "{head}"
        );
    }

    #[test]
    fn output_only_usage_reports_the_full_effective_input() {
        let mut parsed =
            parse_responses_request(&ParseEnv::default(), r#"{"input":"result"}"#).unwrap();
        parsed.needs |= NEED_BANK_FRONTIER;
        let (mut stepper, _) = ContStepper::new(
            &parsed,
            0,
            "resp-cache-usage",
            7,
            false,
            16,
            b"suffix".to_vec(),
            16,
            512,
        );

        stepper.feed(b"answer");
        let (body, _) = stepper.finalize(true, 260, 18, ReqTimings::default(), false);
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("\"input_tokens\":278"), "{body}");
        assert!(body.contains("\"cached_tokens\":260"), "{body}");
        assert!(body.contains("\"cache_write_tokens\":18"), "{body}");
    }

    #[test]
    fn continuous_responses_heartbeat_advances_the_live_sequence() {
        let parsed =
            parse_responses_request(&ParseEnv::default(), r#"{"input":"hello","stream":true}"#)
                .unwrap();
        let (mut stepper, head) = ContStepper::new(
            &parsed,
            0,
            "heartbeat",
            7,
            false,
            16,
            b"prompt".to_vec(),
            1,
            32,
        );
        assert!(String::from_utf8_lossy(&head).contains("\"sequence_number\":0"));

        let due = stepper.last_heartbeat + crate::stream::STREAM_HEARTBEAT_INTERVAL;
        let heartbeat = stepper.heartbeat(due);

        let heartbeat = String::from_utf8(heartbeat).unwrap();
        assert!(heartbeat.contains("\"type\":\"response.in_progress\""));
        assert!(heartbeat.contains("\"sequence_number\":1"));
    }

    #[test]
    fn continuous_shutdown_marks_a_protocol_failure_before_row_abort() {
        let parsed =
            parse_responses_request(&ParseEnv::default(), r#"{"input":"hello","stream":true}"#)
                .unwrap();
        let (mut stepper, _) = ContStepper::new(
            &parsed,
            0,
            "shutdown",
            7,
            false,
            16,
            b"prompt".to_vec(),
            1,
            32,
        );

        assert!(stepper.stop_for_shutdown(Some(shutdown_requested)));
        assert_eq!(stepper.finish, "error");
    }

    #[test]
    fn bank_continuation_keeps_exact_prefix_and_tokenizes_only_the_suffix() {
        let (tokens, cached) = live_continuation_tokens(&[10, 11], b" suffix", 4, |suffix| {
            assert_eq!(suffix, b" suffix");
            vec![20, 21]
        })
        .unwrap();
        assert_eq!(tokens, [10, 11, 20, 21]);
        assert_eq!(cached, 2);
        assert!(live_continuation_tokens(&[10, 11], b" suffix", 3, |_| vec![20, 21]).is_none());
    }

    fn temp_store(tag: &str) -> (std::path::PathBuf, Store) {
        let dir =
            std::env::temp_dir().join(format!("ds4-server-bank-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::open(&dir, 16, false, Options::default()).unwrap();
        (dir, store)
    }

    fn warm(text: &str, last_use: u64) -> WarmBank {
        WarmBank {
            record: Some(WarmRecord {
                text: text.as_bytes().to_vec(),
                cache_text: None,
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 1,
                ext_flags: 0,
                trailer: Vec::new(),
            }),
            committed_tokens: 10,
            stored_tokens: 0,
            last_use,
        }
    }

    #[test]
    fn bank_scope_accepts_chat_turns_including_thinking() {
        let mut env = ParseEnv::default();
        env.default_effort = ThinkMode::None;
        let mut parsed = parse_chat_request(
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
        )
        .unwrap();
        assert!(bank_scope(&parsed));

        parsed.api = Api::Anthropic;
        assert!(bank_scope(&parsed));
        parsed.api = Api::Responses;
        assert!(bank_scope(&parsed));
        parsed.api = Api::Openai;
        parsed.kind = ReqKind::Completion;
        assert!(!bank_scope(&parsed));
        parsed.kind = ReqKind::Chat;
        parsed.think_mode = ThinkMode::Low;
        assert!(bank_scope(&parsed));
        parsed.think_mode = ThinkMode::None;
        parsed.has_tools = true;
        assert!(bank_scope(&parsed));
        parsed.has_tools = false;
        parsed.has_tool_results = true;
        assert!(bank_scope(&parsed));
        parsed.has_tool_results = false;
        parsed.live_call_ids.push("call_1".into());
        assert!(bank_scope(&parsed));
    }

    #[test]
    fn completed_tool_history_preserves_reasoning_without_active_tools() {
        let parsed = parse_chat_request(
            &ParseEnv::default(),
            r#"{"messages":[{"role":"user","content":"call f"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"},{"role":"assistant","content":"done"},{"role":"user","content":"next"}],"reasoning_effort":"high"}"#,
        )
        .unwrap();
        assert!(!parsed.has_tools);
        assert!(prompt_preserves_reasoning(&parsed));
    }

    #[test]
    fn qwen_thinking_bank_key_matches_the_next_visible_history() {
        let user = ChatMsg {
            role: "user".into(),
            content: "hello".into(),
            ..Default::default()
        };
        let prompt = crate::render::render_chat(
            ModelSyntax::Qwen4Exp,
            std::slice::from_ref(&user),
            "",
            ThinkMode::High,
        )
        .unwrap();
        let key = thinking_visible_key(
            &prompt,
            b"  visible answer  ",
            ModelSyntax::Qwen4Exp,
            ChatFormat::Qwen4Exp,
        )
        .unwrap();
        let next = crate::render::render_chat(
            ModelSyntax::Qwen4Exp,
            &[
                user,
                ChatMsg {
                    role: "assistant".into(),
                    content: "visible answer".into(),
                    ..Default::default()
                },
                ChatMsg {
                    role: "user".into(),
                    content: "next".into(),
                    ..Default::default()
                },
            ],
            "",
            ThinkMode::High,
        )
        .unwrap();
        assert!(next.starts_with(&key));
        assert!(!key.ends_with(b"<|im_end|>\n"));
    }

    #[test]
    fn qwen_thinking_retire_keeps_visible_key_and_tools_alias() {
        let prompt = b"<|im_start|>assistant\n<think>\n";
        let pieces = [
            b"hidden".as_slice(),
            b"</think>".as_slice(),
            b"\n\nanswer".as_slice(),
            b"<|im_end|>".as_slice(),
        ];
        let retired = thinking_bank_retire_key(
            prompt,
            &[0, 1, 2, 3],
            true,
            false,
            true,
            true,
            false,
            crate::render::ModelSyntax::Qwen4Exp,
            crate::stream::ChatFormat::Qwen4Exp,
            |token| pieces[token as usize].to_vec(),
        )
        .unwrap();
        assert_eq!(
            retired.text,
            b"<|im_start|>assistant\n<think>\n\n</think>\n\nanswer"
        );
        assert_eq!(
            retired.exact_text.as_deref(),
            Some(b"<|im_start|>assistant\n<think>\nhidden</think>\n\nanswer".as_slice())
        );
        assert!(!retired.partial_only);
    }

    #[test]
    fn completed_tool_history_without_active_tools_keeps_the_raw_key() {
        let prompt = b"<|im_start|>assistant\n<think>\n";
        let pieces = [
            b"hidden".as_slice(),
            b"</think>".as_slice(),
            b"\n\nanswer".as_slice(),
            b"<|im_end|>".as_slice(),
        ];
        let retired = thinking_bank_retire_key(
            prompt,
            &[0, 1, 2, 3],
            true,
            false,
            true,
            false,
            false,
            ModelSyntax::Qwen4Exp,
            ChatFormat::Qwen4Exp,
            |token| pieces[token as usize].to_vec(),
        )
        .unwrap();
        assert_eq!(
            retired.text,
            b"<|im_start|>assistant\n<think>\nhidden</think>\n\nanswer"
        );
        assert!(retired.exact_text.is_none());
    }

    #[test]
    fn interrupted_thinking_retire_is_rewind_only_after_a_committed_sample() {
        let prompt = b"<|im_start|>assistant\n<think>\n";
        let retired = thinking_bank_retire_key(
            prompt,
            &[0, 1],
            false,
            true,
            false,
            false,
            false,
            crate::render::ModelSyntax::Qwen4Exp,
            crate::stream::ChatFormat::Qwen4Exp,
            |token| [b"hidden".as_slice(), b" tail".as_slice()][token as usize].to_vec(),
        )
        .unwrap();
        assert_eq!(retired.text, prompt);
        assert!(retired.partial_only);
    }

    #[test]
    fn cont_stepper_retains_exact_tool_text_for_bank_checkpoint() {
        let mut env = ParseEnv::default();
        env.default_effort = ThinkMode::None;
        let parsed = parse_chat_request(
            &env,
            r#"{"messages":[{"role":"user","content":"call f"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"thinking":{"type":"disabled"}}"#,
        )
        .unwrap();
        let (mut stepper, _) = ContStepper::new(
            &parsed,
            0,
            "tool-bank",
            1,
            false,
            16,
            b"prompt".to_vec(),
            1,
            32,
        );
        let exact =
            "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert!(stepper.feed(exact.as_bytes()).done);
        let (_, outcome) = stepper.finalize(false, 0, 1, ReqTimings::default(), false);
        assert_eq!(outcome.tool_ids.len(), 1);
        let (calls, remembered) = stepper.take_tool_replay().unwrap();
        assert_eq!(calls[0].id, outcome.tool_ids[0]);
        assert_eq!(remembered, exact);
    }

    #[test]
    fn warm_key_drops_the_uncommitted_last_sample() {
        assert_eq!(
            committed_key(b"prompt:", &[1, 2, 3], |token| vec![b'0' + token as u8]),
            b"prompt:12"
        );
        assert_eq!(
            committed_key(b"prompt:", &[], |_| unreachable!()),
            b"prompt:"
        );
    }

    #[test]
    fn bank_retire_key_accepts_a_complete_tool_turn_from_the_snapshot() {
        let token_text = |token| vec![b'0' + token as u8];
        assert_eq!(
            bank_retire_key(b"12", &[1, 2, 3, 4], &[], false, token_text),
            None
        );
        assert_eq!(
            bank_retire_key(b"12", &[1, 2, 3, 4], &[], true, token_text),
            Some((b"123".to_vec(), false))
        );
        assert_eq!(
            bank_retire_key(b"1234", &[1, 2, 3, 4], &[], false, token_text),
            Some((b"123".to_vec(), true))
        );
        assert_eq!(
            bank_retire_key(b"12", &[], &[3, 4], false, token_text),
            Some((b"123".to_vec(), false))
        );
        assert_eq!(
            bank_retire_key(
                b"<|assistant|><think></think>",
                &[],
                &[1, 2],
                false,
                |token| match token {
                    1 => b"I need".to_vec(),
                    2 => b" it".to_vec(),
                    _ => Vec::new(),
                }
            ),
            Some((b"<|assistant|>I need".to_vec(), false))
        );
    }

    #[test]
    fn warm_match_requires_a_strict_suffix_and_exact_snapshot_tokens() {
        let warm = WarmBank {
            record: Some(WarmRecord {
                text: b"shared".to_vec(),
                cache_text: None,
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 7,
                ext_flags: 0,
                trailer: Vec::new(),
            }),
            committed_tokens: 2,
            stored_tokens: 0,
            last_use: 0,
        };
        assert!(
            warm_admit_tokens(&warm, b"shared", None, &[10, 11], 7, 8, false, |_| vec![12])
                .is_none()
        );
        assert!(
            warm_admit_tokens(&warm, b"other", None, &[10, 11], 7, 8, false, |_| vec![12])
                .is_none()
        );
        assert!(warm_admit_tokens(
            &warm,
            b"shared suffix",
            None,
            &[10, 11],
            8,
            8,
            false,
            |_| vec![12],
        )
        .is_none());
        let (tokens, cached) = warm_admit_tokens(
            &warm,
            b"shared suffix",
            None,
            &[10, 11],
            7,
            8,
            false,
            |suffix| {
                assert_eq!(suffix, b" suffix");
                vec![20, 21]
            },
        )
        .unwrap();
        assert_eq!(tokens, vec![10, 11, 20, 21]);
        assert_eq!(cached, 2);
        assert!(warm_admit_tokens(
            &warm,
            b"shared suffix",
            None,
            &[10, 11],
            7,
            3,
            false,
            |_| vec![20, 21],
        )
        .is_none());
    }

    #[test]
    fn multi_bank_match_uses_the_longest_strict_prefix() {
        let banks = vec![warm("shared", 3), warm("shared turn", 2), warm("other", 1)];
        assert_eq!(
            warm_match_pick(&banks, b"shared turn suffix", None),
            Some(WarmMatch {
                bank: 1,
                exact: false,
            })
        );
        assert_eq!(
            warm_match_pick(&banks, b"shared turn", None),
            Some(WarmMatch {
                bank: 0,
                exact: false,
            })
        );
        assert_eq!(warm_match_pick(&banks, b"unrelated", None), None);
    }

    #[test]
    fn warm_full_match_accepts_visible_and_exact_reasoning_keys() {
        let banks = [WarmBank {
            record: Some(WarmRecord {
                text: b"visible".to_vec(),
                cache_text: None,
                exact_text: Some(b"raw reasoning".to_vec()),
                exact_cache_text: None,
                partial_only: false,
                generation: 7,
                ext_flags: 0,
                trailer: Vec::new(),
            }),
            committed_tokens: 2,
            stored_tokens: 0,
            last_use: 0,
        }];
        assert_eq!(
            warm_match_pick(&banks, b"visible tail", None),
            Some(WarmMatch {
                bank: 0,
                exact: false,
            })
        );
        assert_eq!(
            warm_match_pick(&banks, b"raw reasoning tail", None),
            Some(WarmMatch {
                bank: 0,
                exact: true,
            })
        );
        let (tokens, cached) = warm_admit_tokens(
            &banks[0],
            b"raw reasoning tail",
            None,
            &[10, 11],
            7,
            8,
            true,
            |suffix| {
                assert_eq!(suffix, b" tail");
                vec![12]
            },
        )
        .unwrap();
        assert_eq!(tokens, [10, 11, 12]);
        assert_eq!(cached, 2);
    }

    #[test]
    fn partial_match_uses_the_longest_divergent_record() {
        let mut banks = vec![
            warm("shared system alpha", 3),
            warm("shared beta", 2),
            warm("other", 1),
        ];
        assert_eq!(
            warm_partial_match_pick(&banks, b"shared system gamma", None, 5),
            Some((0, b"shared system ".len()))
        );
        assert_eq!(
            warm_partial_match_pick(&banks, b"shared system gamma", None, 15),
            None
        );
        banks[0].record.as_mut().unwrap().partial_only = true;
        assert_eq!(
            warm_partial_match_pick(&banks, b"shared system gamma", None, 5),
            Some((0, b"shared system ".len()))
        );
        assert_eq!(
            warm_match_pick(&banks, b"shared system alpha tail", None),
            None
        );
    }

    #[test]
    fn rewind_only_record_cannot_supersede_an_exact_record() {
        let mut banks = vec![warm("prefix", 1), warm("prefix plus tail", 2)];
        banks[1].record.as_mut().unwrap().partial_only = true;
        assert!(!warm_record_superseded(&banks, 0));
        banks[0].record.as_mut().unwrap().partial_only = true;
        banks[1].record.as_mut().unwrap().partial_only = false;
        assert!(warm_record_superseded(&banks, 0));
    }

    #[test]
    fn partial_cut_is_a_canonical_token_prefix_with_a_suffix_left_to_prefill() {
        assert_eq!(
            warm_partial_token_cut(&[10, 11, 12, 13], &[10, 11, 12, 20], 3, usize::MAX),
            Some(3)
        );
        assert_eq!(
            warm_partial_token_cut(&[10, 11, 12, 13], &[10, 11, 12, 20], 4, usize::MAX),
            None
        );
        assert_eq!(
            warm_partial_token_cut(&[10, 11, 12], &[10, 11, 12], 2, usize::MAX),
            Some(2)
        );
    }

    #[test]
    fn qwen_image_cache_key_separates_pixels_and_caps_partial_reuse() {
        let prompt = b"before<|image_pad|>after";
        let identity = |pixel_hash| ImageCacheIdentity {
            pixel_hash,
            grid_h: 16,
            grid_w: 16,
            token_count: 64,
            token_offset: 17,
        };
        let (red, spans) = qwen_image_cache_text_build(prompt, &[identity(1)]).unwrap();
        let (same, _) = qwen_image_cache_text_build(prompt, &[identity(1)]).unwrap();
        let (blue, _) = qwen_image_cache_text_build(prompt, &[identity(2)]).unwrap();
        assert_eq!(
            red,
            b"before<|image_pad|>\xffDS4IMG2:0000000000000001:00000010:00000010:00000040\xfeafter"
        );
        assert_eq!(red, same);
        assert_ne!(red, blue);
        assert_eq!(qwen_image_cache_text_strip(&red).unwrap(), prompt);
        let extended =
            extended_image_cache_key(prompt, &red, b"before<|image_pad|>after answer").unwrap();
        assert_eq!(&extended[..red.len()], red);
        assert_eq!(&extended[red.len()..], b" answer");
        let lcp = red.iter().zip(&blue).take_while(|(a, b)| a == b).count();
        assert_eq!(qwen_image_cache_token_cap(&spans, lcp), 17);

        let mut record_key = red.clone();
        record_key.extend_from_slice(b" answer");
        let banks = [WarmBank {
            record: Some(WarmRecord {
                text: b"before<|image_pad|>after answer".to_vec(),
                cache_text: Some(record_key.clone()),
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 1,
                ext_flags: EXT_IMAGE_PIXELS_V2,
                trailer: Vec::new(),
            }),
            committed_tokens: 80,
            stored_tokens: 0,
            last_use: 1,
        }];
        let mut same_request = record_key;
        same_request.extend_from_slice(b" tail");
        let mut different_request = blue;
        different_request.extend_from_slice(b" answer tail");
        assert_eq!(
            warm_match_pick(
                &banks,
                b"before<|image_pad|>after answer tail",
                Some(&same_request),
            ),
            Some(WarmMatch {
                bank: 0,
                exact: false,
            })
        );
        assert_eq!(
            warm_match_pick(
                &banks,
                b"before<|image_pad|>after answer tail",
                Some(&different_request),
            ),
            None
        );
        assert_eq!(
            warm_match_pick(&banks, b"before<|image_pad|>after answer tail", None),
            None
        );
    }

    #[test]
    fn partial_admit_uses_canonical_prompt_tokens_and_validates_generation() {
        let warm = warm("shared system alpha", 7);
        let (tokens, cached) = warm_partial_admit_tokens(
            &warm,
            &[10, 11, 12, 20, 21],
            &[10, 11, 12, 13],
            1,
            3,
            8,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(tokens, [10, 11, 12, 20, 21]);
        assert_eq!(cached, 3);
        assert!(warm_partial_admit_tokens(
            &warm,
            &[10, 11, 12, 20],
            &[10, 11, 12, 13],
            2,
            3,
            8,
            usize::MAX,
        )
        .is_none());
        assert!(warm_partial_admit_tokens(
            &warm,
            &[10, 11, 12, 20],
            &[10, 11, 12, 13],
            1,
            3,
            3,
            usize::MAX,
        )
        .is_none());
    }

    #[test]
    fn cold_victim_matches_c_no_value_superseded_and_depth_tiers() {
        let mut banks = vec![warm("trunk", 30), WarmBank::default(), warm("tenant", 10)];
        banks[0].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(1));

        banks[1] = warm("trunk child", 20);
        banks[1].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(0));

        banks[0] = warm("alpha", 5);
        banks[1] = warm("beta", 4);
        banks[2] = warm("gamma", 10);
        banks[0].committed_tokens = 70_000;
        banks[1].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(2));
        banks[2].committed_tokens = 90_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(1));
    }

    #[test]
    fn fork_target_never_spends_a_plain_tenant_record() {
        let banks = vec![warm("source", 3), warm("tenant-a", 1), warm("tenant-b", 2)];
        assert_eq!(warm_victim_pick(&banks, &[], Some(0), 65_536, false), None);

        let mut spare = banks;
        spare[2] = WarmBank::default();
        assert_eq!(
            warm_victim_pick(&spare, &[], Some(0), 65_536, false),
            Some(2)
        );
    }

    #[test]
    fn warm_fork_preserves_the_trunk_only_when_a_safe_target_exists() {
        let mut with_spare = vec![warm("trunk", 3), warm("tenant", 1), WarmBank::default()];
        assert_eq!(
            warm_placement(&with_spare, &[], 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 2,
                fork: true,
            })
        );

        let no_spare = vec![warm("trunk", 3), warm("tenant-a", 1), warm("tenant-b", 2)];
        assert_eq!(
            warm_placement(&no_spare, &[], 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 0,
                fork: false,
            })
        );
        with_spare[0].committed_tokens = 70_000;
        assert!(
            !warm_placement(&with_spare, &[], 0, 70_000, 65_536, true)
                .unwrap()
                .fork
        );
    }

    #[test]
    fn rolling_fork_reuses_prefix_without_spending_the_live_target() {
        let mut banks = vec![warm("trunk", 3), WarmBank::default(), WarmBank::default()];
        let first = warm_placement(&banks, &[], 0, 10, 65_536, true).unwrap();
        assert!(first.fork && first.target == 1);
        banks[1].record = None;

        let mut reserve = crate::serve_cont_roll::RollReserve::new();
        reserve.note_place(i32::try_from(first.target + 1).unwrap());
        let protected = reserve.protect(&[false, false, false]);
        assert_eq!(
            warm_victim_pick(&banks, &protected, None, 65_536, true),
            Some(2)
        );
        assert_eq!(
            warm_placement(&banks, &protected, 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 2,
                fork: true,
            })
        );
    }

    #[test]
    fn rolling_pin_prevents_evict_of_a_deep_fork_target() {
        let mut banks = vec![warm("trunk", 3), warm("tenant", 1), WarmBank::default()];
        banks[1].committed_tokens = 70_000;
        let mut reserve = crate::serve_cont_roll::RollReserve::new();
        reserve.note_place(3);
        let protected = reserve.protect(&[false, false, false]);
        assert_eq!(
            warm_victim_pick(&banks, &protected, Some(0), 65_536, false),
            None
        );
        assert_eq!(
            warm_placement(&banks, &protected, 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 0,
                fork: false,
            })
        );
    }

    #[test]
    fn rolling_protected_saturation_refuses_when_hold_and_live_fill_the_table() {
        let banks = vec![warm("trunk", 3), warm("tenant", 1)];
        let mut reserve = crate::serve_cont_roll::RollReserve::new();
        reserve.note_place(2);
        let protected = reserve.protect(&[true, false]);
        assert_eq!(
            warm_victim_pick(&banks, &protected, None, 65_536, true),
            None
        );
        assert_eq!(
            warm_placement(&banks, &protected, 0, 10, 65_536, true),
            None
        );
    }

    #[test]
    fn protected_trunk_forks_safely_or_refuses_in_place_extension() {
        let with_spare = vec![warm("trunk", 3), WarmBank::default()];
        assert!(
            warm_placement(&with_spare, &[true, false], 0, 10, 65_536, true)
                .is_some_and(|placement| placement.fork && placement.target == 1)
        );

        let no_spare = vec![warm("trunk", 3), warm("tenant", 2)];
        assert_eq!(
            warm_placement(&no_spare, &[true, false], 0, 10, 65_536, true),
            None
        );
    }

    #[test]
    fn victim_selection_skips_a_protected_bank_at_every_tier() {
        let mut banks = vec![WarmBank::default(), warm("shallow", 1), warm("deep", 2)];
        banks[2].committed_tokens = 70_000;
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, true),
            Some(1)
        );

        banks[0] = warm("trunk", 1);
        banks[1] = warm("trunk child", 2);
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, false),
            None
        );

        banks[0] = warm("shallow", 1);
        banks[1].committed_tokens = 70_000;
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, true),
            Some(1)
        );

        let all_protected = [true, true, true];
        assert_eq!(
            warm_victim_pick(&banks, &all_protected, None, 65_536, true),
            None
        );
    }

    #[test]
    fn disk_restore_replaces_only_empty_stale_or_shallow_live_banks() {
        let empty = WarmBank::default();
        assert!(disk_restore_allowed(&empty, Some((7, 70_000)), 1, 65_536));

        let live = WarmBank {
            record: Some(WarmRecord {
                text: b"live".to_vec(),
                cache_text: None,
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 7,
                ext_flags: 0,
                trailer: Vec::new(),
            }),
            committed_tokens: 70_000,
            stored_tokens: 0,
            last_use: 0,
        };
        assert!(disk_restore_allowed(&live, Some((8, 70_000)), 1, 65_536));
        assert!(disk_restore_allowed(
            &live,
            Some((7, 32_000)),
            70_000,
            65_536,
        ));
        assert!(!disk_restore_allowed(
            &live,
            Some((7, 70_000)),
            70_000,
            65_536,
        ));
        assert!(!disk_restore_allowed(
            &live,
            Some((7, 32_000)),
            32_000,
            65_536,
        ));
    }

    #[test]
    fn disk_restore_may_replace_a_superseded_deep_record() {
        let mut banks = vec![warm("trunk", 1), warm("trunk child", 2)];
        banks[0].committed_tokens = 70_000;
        banks[1].committed_tokens = 70_000;
        assert!(disk_restore_target_allowed(&banks, 0, 10, 65_536));
        assert!(!disk_restore_target_allowed(&banks, 1, 10, 65_536));
    }

    #[test]
    fn bank_retires_only_after_native_done() {
        assert!(bank_retire_allowed(true, true, true));
        assert!(!bank_retire_allowed(false, true, true));
        assert!(!bank_retire_allowed(true, false, true));
        assert!(!bank_retire_allowed(true, true, false));
    }

    #[test]
    fn bank_retirement_uses_the_native_reported_bank() {
        assert_eq!(retired_bank(true, true, true, Some(2), 3), Some(2));
        assert_eq!(retired_bank(true, true, true, None, 3), None);
        assert_eq!(retired_bank(true, true, true, Some(3), 3), None);
    }

    #[test]
    fn bank_retire_keeps_prior_warm_ext_flags() {
        use ds4_kv::{EXT_RESPONSES_VISIBLE, EXT_SESSION_TITLE};
        let prev = WarmRecord {
            text: b"old".to_vec(),
            cache_text: None,
            exact_text: None,
            exact_cache_text: None,
            partial_only: false,
            generation: 1,
            ext_flags: EXT_RESPONSES_VISIBLE | EXT_SESSION_TITLE,
            trailer: b"ktm".to_vec(),
        };
        let retired = retired_record(Some(&prev), b"new-key".to_vec(), None, 9).unwrap();
        assert_eq!(retired.text, b"new-key");
        assert_eq!(retired.generation, 9);
        assert_eq!(retired.ext_flags, EXT_RESPONSES_VISIBLE | EXT_SESSION_TITLE);
        assert!(retired_record(Some(&prev), Vec::new(), None, 9).is_none());
        assert_eq!(
            retired_record(None, b"fresh".to_vec(), None, 2)
                .unwrap()
                .ext_flags,
            0
        );
    }

    #[test]
    fn bank_save_stages_before_reuse_and_advances_marker_only_on_success() {
        let (dir, mut store) = temp_store("save-order");
        let trailer =
            crate::tool_memory::encode_ktm(&[(b"call_1", b"<tool_call>x</tool_call>")]).unwrap();
        let mut warm = WarmBank {
            record: Some(WarmRecord {
                text: b"shared prefix".to_vec(),
                cache_text: None,
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 7,
                ext_flags: 0,
                trailer: trailer.clone(),
            }),
            committed_tokens: 3,
            stored_tokens: 1,
            last_use: 0,
        };
        assert!(!save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            8,
            Reason::BankShutdown,
            |_| panic!("stale records must not stage payloads"),
        )
        .unwrap());
        assert!(store.entries().is_empty());
        assert_eq!(warm.stored_tokens, 1);
        assert!(save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            7,
            Reason::BankShutdown,
            |path| fs::write(path, b"opaque").map_err(|e| GenerateError::Engine(e.to_string())),
        )
        .unwrap());
        assert_eq!(warm.stored_tokens, 3);
        let path = store.entries()[0].path.clone();
        let record = store.read(&path).unwrap();
        assert_eq!(record.header.reason, Reason::BankShutdown);
        assert_eq!(record.header.ext_flags, EXT_BANK_REPLAY_V1 | EXT_TOOL_MAP);
        assert_eq!(record.header.tokens, 3);
        assert_eq!(record.header.ctx_size, 8192);
        assert_eq!(record.text, b"shared prefix");
        assert_eq!(record.payload, b"opaque");
        assert_eq!(record.trailer, trailer);

        warm.stored_tokens = 1;
        let error = save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            7,
            Reason::BankShutdown,
            |_| Err(GenerateError::Engine("stage failed".into())),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "stage failed");
        assert_eq!(warm.stored_tokens, 1);
        assert_eq!(store.read(&path).unwrap().payload, b"opaque");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bank_save_skips_rewind_only_thinking_records() {
        let (dir, mut store) = temp_store("partial-only");
        let mut warm = warm("prompt", 1);
        warm.record.as_mut().unwrap().partial_only = true;
        assert!(!save_bank_record(
            &mut store,
            &mut warm,
            (6, 4, 262_144),
            2,
            1,
            Reason::BankCheckpoint,
            |_| panic!("rewind-only records must not be persisted"),
        )
        .unwrap());
        assert!(store.entries().is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bank_save_persists_visible_key_without_the_in_memory_alias() {
        let (dir, mut store) = temp_store("thinking-visible");
        let mut warm = warm("visible answer", 1);
        warm.record.as_mut().unwrap().exact_text = Some(b"hidden reasoning answer".to_vec());
        assert!(save_bank_record(
            &mut store,
            &mut warm,
            (6, 4, 262_144),
            3,
            1,
            Reason::BankCheckpoint,
            |path| fs::write(path, b"opaque").map_err(|e| GenerateError::Engine(e.to_string())),
        )
        .unwrap());
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.text, b"visible answer");
        assert_eq!(record.header.ext_flags, EXT_BANK_REPLAY_V1);
        assert!(!record.text.windows(6).any(|window| window == b"hidden"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bank_save_persists_the_image_identity_key() {
        let (dir, mut store) = temp_store("image-key");
        let visible = b"before<|image_pad|>after".to_vec();
        let (cache_key, _) = qwen_image_cache_text_build(
            &visible,
            &[ImageCacheIdentity {
                pixel_hash: 7,
                grid_h: 16,
                grid_w: 16,
                token_count: 64,
                token_offset: 5,
            }],
        )
        .unwrap();
        let mut warm = WarmBank {
            record: Some(WarmRecord {
                text: visible.clone(),
                cache_text: Some(cache_key.clone()),
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 3,
                ext_flags: EXT_IMAGE_PIXELS_V2,
                trailer: Vec::new(),
            }),
            committed_tokens: 64,
            stored_tokens: 0,
            last_use: 0,
        };
        assert!(save_bank_record(
            &mut store,
            &mut warm,
            (6, 2, 8192),
            64,
            3,
            Reason::BankCheckpoint,
            |path| fs::write(path, b"opaque").map_err(|e| GenerateError::Engine(e.to_string())),
        )
        .unwrap());
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.text, cache_key);
        assert_eq!(
            record.header.ext_flags,
            EXT_BANK_REPLAY_V1 | EXT_IMAGE_PIXELS_V2
        );
        assert_eq!(
            restored_record(record.text, 4, record.header.ext_flags)
                .unwrap()
                .text,
            visible
        );
        let _ = fs::remove_dir_all(dir);
    }
}
