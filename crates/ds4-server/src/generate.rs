//! Serial decode driver: render (including tool-schema / invoke reconstruct)
//! → host Vocab tokenize (FFI fallback) → host SessionLedger pos/generation
//! → native prefill/eval → SemAccum + generated-message parse → stream
//! projectors.
//! Incremental live DSML tool projection, required-prefix / structural
//! greedy sampling, and corrective retry (`decode_again` / model-visible
//! tool error) are host-owned. Continuation publish/hold/resolve is
//! host-owned (`cont`).

use std::io::Write;
#[cfg(any(feature = "native", test))]
use std::path::Path;
use std::time::{Duration, Instant};
#[cfg(any(feature = "native", test))]
use std::time::{SystemTime, UNIX_EPOCH};

use ds4_kv::Store as KvStore;
#[cfg(any(feature = "native", test))]
use ds4_kv::{
    chat_anchor_pos as kv_chat_anchor_pos, continued_store_target_from_host,
    store_len as kv_store_len, Header as KvHeader, HostKvView, Reason as KvReason,
    EXT_THINKING_VISIBLE, EXT_TOOL_MAP,
};

use crate::dsml::{SampleOverride, SamplePolicy};
use crate::parse::{ChatMsg, ParsedRequest, ToolCall, ToolChoice};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use crate::render::{render_chat_choice, syntax_for_model_id, ModelSyntax, RenderError};
use crate::retry::{
    build_invalid_tool_error_suffix, parse_failure_should_retry, terminal_finish,
    truncation_outcome, TruncationOutcome,
};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, responses_final_response, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init, sse_chunk,
    sse_done, sse_headers, stream_error, think_end, think_start, AnthropicStream, ChatFormat,
    OpenaiStream, ReqTimings, ResponsesStream, StreamReq, Writer,
};
#[cfg(feature = "native")]
use crate::tool_memory::ToolMemory;
use crate::tools::{assign_tool_ids, parse_generated_for_response, SemAccum};

#[derive(Debug)]
pub enum GenerateError {
    Unsupported(&'static str),
    ContinuationHold { retry_after: i32 },
    Engine(String),
    Streamed(String),
    Io,
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Unsupported(s) => f.write_str(s),
            GenerateError::ContinuationHold { .. } => {
                f.write_str("batch capacity is reserved for live tool continuations")
            }
            GenerateError::Engine(s) => write!(f, "{s}"),
            GenerateError::Streamed(s) => write!(f, "{s}"),
            GenerateError::Io => f.write_str("client stream write failed"),
        }
    }
}

impl From<RenderError> for GenerateError {
    fn from(e: RenderError) -> Self {
        GenerateError::Unsupported(e.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGraphFit {
    /// C quote verdict at the probed ctx (`ds4_engine_session_graph_fits`).
    /// Family fit checks (EXAONE/Solar/Motif/dots3) report only this bit;
    /// their byte fields stay zero.
    pub fits: bool,
    pub need_bytes: u64,
    pub avail_bytes: u64,
    pub headroom_bytes: u64,
    pub deficit_bytes: u64,
    pub fail_open: bool,
}

/// C `serial_session_ensure_fit` view of the serial session lane.
/// `None` from [`DecodeIo::serial_session_probe`] means the engine has no
/// native serial session (stub/test engines): the host must pass native and
/// never invent a rightsize or a refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialSessionProbe {
    /// Current session ctx; the boot `-c` when no session exists yet (the
    /// Rust host creates the serial session lazily, which is C's boot-shape
    /// lazy session by another name).
    pub cur_ctx: i32,
    /// C `ds4_session_graph_pending`: true when the graph alloc is still
    /// deferred (including "no session yet"). A pending session can be
    /// re-created at another ctx for free.
    pub graph_pending: bool,
}

pub trait DecodeIo {
    fn model_id(&self) -> i32;
    fn kv_store_mut(&mut self) -> Option<&mut KvStore> {
        None
    }
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError>;
    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError>;
    fn tokenizes_control_literals(&self) -> bool {
        true
    }
    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError>;
    fn token_is_stop(&self, token: i32) -> bool;
    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError>;
    fn sync_prompt(
        &mut self,
        _prompt: &[u8],
        tokens: &[i32],
        _disk_eligible: bool,
        _thinking_visible_eligible: bool,
    ) -> Result<i32, GenerateError> {
        self.sync(tokens)?;
        Ok(0)
    }
    fn prompt_sync_elapsed(&self) -> Option<Duration> {
        None
    }
    fn restore_tool_replay(&mut self, _messages: &mut [ChatMsg]) {}
    fn sync_tool_replay_prompt(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
    ) -> Result<i32, GenerateError> {
        self.sync_prompt(prompt, tokens, true, false)
    }
    fn remember_tool_replay(&mut self, _calls: &[ToolCall], _raw_dsml: &str) {}
    fn maybe_store_continued(&mut self) -> Result<(), GenerateError> {
        Ok(())
    }
    fn shutdown(&mut self) -> Result<(), GenerateError> {
        Ok(())
    }
    fn eval(&mut self, token: i32) -> Result<(), GenerateError>;
    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32;
    fn native_graph_fit(&self, _ctx: i32) -> Option<NativeGraphFit> {
        None
    }
    /// C `serial_session_ensure_fit` inputs. `None` = no native serial
    /// session lane; the host ensure-fit passes native.
    fn serial_session_probe(&self) -> Option<SerialSessionProbe> {
        None
    }
    /// C `ds4_session_free` + `ds4_session_create(target)`: replace the
    /// serial session with a right-sized one. The old session's live
    /// records die with it (the caller demotes the registry).
    fn serial_session_rightsize(&mut self, _target_ctx: i32) -> Result<(), GenerateError> {
        Ok(())
    }
    /// C refusal shape: free the session so the server keeps its boot
    /// invariant and later requests re-probe fresh (the Rust host restores
    /// lazily on next use instead of eagerly re-creating at boot ctx).
    fn serial_session_reset(&mut self) {}
    fn pos(&self) -> i32;
    fn ctx(&self) -> i32;
    fn generation(&self) -> u64;
    fn session_tokens(&self) -> Vec<i32> {
        Vec::new()
    }
    fn remember_thinking_visible_checkpoint(&mut self, _text: Vec<u8>) {}
    fn invalidate(&mut self);
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy)]
struct DiskSyncPolicy {
    save_current: bool,
    load: bool,
}

#[cfg(any(feature = "native", test))]
struct ThinkingVisibleCheckpoint {
    text: Vec<u8>,
    frontier: i32,
}

#[cfg(any(feature = "native", test))]
fn settle_thinking_visible_checkpoint(
    checkpoint: &mut Option<ThinkingVisibleCheckpoint>,
    sync_succeeded: bool,
) {
    if sync_succeeded {
        *checkpoint = None;
    }
}

#[cfg(any(feature = "native", test))]
trait SerialKvIo {
    fn ctx(&self) -> i32;
    fn chat_token_ids(&self) -> (i32, i32) {
        (-1, -1)
    }
    fn live_len(&self) -> i32;
    fn live_tokens(&self) -> Vec<i32>;
    fn render_tokens(&self, tokens: &[i32]) -> Result<Vec<u8>, GenerateError>;
    fn checkpoint_trailer(&self, _text: &[u8]) -> Option<Vec<u8>> {
        Some(Vec::new())
    }
    fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError>;
    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError>;
    fn sync_with_prefill_checkpoints(
        &mut self,
        tokens: &[i32],
        _store: &mut KvStore,
        _identity: (u8, u8, u32),
        _cached_floor: i32,
    ) -> Result<(), GenerateError> {
        self.sync(tokens)
    }
    fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError>;
    fn load_payload_range(
        &mut self,
        path: &Path,
        offset: u64,
        length: u64,
    ) -> Result<(), GenerateError>;
    fn invalidate(&mut self);
}

#[cfg(any(feature = "native", test))]
fn write_checkpoint(
    store: &mut KvStore,
    header: KvHeader,
    text: &[u8],
    trailer: &[u8],
    save_payload: impl FnOnce(&Path) -> Result<(), GenerateError>,
) -> Result<(), GenerateError> {
    if store
        .reuse_compatible(header.clone(), text, trailer)
        .map_err(|error| GenerateError::Engine(error.to_string()))?
        .is_some()
    {
        return Ok(());
    }
    let payload = store
        .payload_temp()
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    save_payload(payload.path())?;
    store
        .write_payload_file(header, text, payload.path(), trailer)
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    Ok(())
}

#[cfg(any(feature = "native", test))]
fn kv_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(any(feature = "native", test))]
pub(crate) fn kv_identity(model_id: i32, quant_bits: i32, ctx: i32) -> Option<(u8, u8, u32)> {
    let model_id = u8::try_from(model_id).ok()?;
    let quant_bits = match quant_bits {
        2 | 4 => quant_bits as u8,
        _ => return None,
    };
    let ctx = u32::try_from(ctx).ok().filter(|ctx| *ctx > 0)?;
    Some((model_id, quant_bits, ctx))
}

#[cfg(any(feature = "native", test))]
fn intermediate_prefill_eligible(
    deepseek: bool,
    cuda: bool,
    disk_eligible: bool,
    tool_replay: bool,
) -> bool {
    disk_eligible && !tool_replay && deepseek && cuda
}

#[cfg(any(feature = "native", test))]
pub(crate) fn kv_header(model_id: u8, quant_bits: u8, ctx: u32, tokens: u32) -> KvHeader {
    let now = kv_now();
    KvHeader {
        quant_bits,
        reason: KvReason::Evict,
        ext_flags: 0,
        model_id,
        tokens,
        hits: 0,
        ctx_size: ctx,
        created_at: now,
        last_used: now,
        payload_bytes: 0,
        text_bytes: 0,
    }
}

#[cfg(any(feature = "native", test))]
fn try_store_live(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    model_id: u8,
    quant_bits: u8,
    ctx: u32,
    reason: KvReason,
    checkpoint: Option<&ThinkingVisibleCheckpoint>,
) -> Result<bool, GenerateError> {
    let live = io.live_tokens();
    if live.len() < store.opt.min_tokens.max(0) as usize {
        return Ok(false);
    }
    let tokens = u32::try_from(live.len())
        .map_err(|_| GenerateError::Engine("KVC token count exceeds u32".into()))?;
    let (text, ext_flags) = match checkpoint {
        Some(checkpoint) if usize::try_from(checkpoint.frontier).ok() == Some(live.len()) => {
            (checkpoint.text.clone(), EXT_THINKING_VISIBLE)
        }
        _ => (io.render_tokens(&live)?, 0),
    };
    let trailer = if ext_flags == 0 {
        let Some(trailer) = io.checkpoint_trailer(&text) else {
            return Ok(false);
        };
        trailer
    } else {
        Vec::new()
    };
    let mut header = kv_header(model_id, quant_bits, ctx, tokens);
    header.reason = reason;
    header.ext_flags = ext_flags;
    write_checkpoint(store, header, &text, &trailer, |path| io.save_payload(path))?;
    Ok(true)
}

#[cfg(any(feature = "native", test))]
fn continued_target(store: &KvStore, live_tokens: i32) -> i32 {
    continued_store_target_from_host(
        &store.opt,
        HostKvView {
            live_tokens,
            stored_tokens: store.continued_last_store_tokens,
        },
    )
}

#[cfg(any(feature = "native", test))]
fn try_store_continued(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    identity: (u8, u8, u32),
) -> Result<bool, GenerateError> {
    let target = continued_target(store, io.live_len());
    if target == 0 {
        return Ok(false);
    }
    let (model_id, quant_bits, ctx) = identity;
    if !try_store_live(
        io,
        store,
        model_id,
        quant_bits,
        ctx,
        KvReason::Continued,
        None,
    )? {
        return Ok(false);
    }
    store.continued_last_store_tokens = store.continued_last_store_tokens.max(target);
    Ok(true)
}

fn store_continued_best_effort(engine: &mut dyn DecodeIo) {
    if let Err(error) = engine.maybe_store_continued() {
        eprintln!("ds4-server-rs: continued KV checkpoint failed: {error}");
    }
}

fn continued_decode_allowed(acc: &SemAccum) -> bool {
    !(acc.track_tools && (acc.saw_tool_start || acc.dsml_state().is_tool()))
}

#[cfg(any(feature = "native", test))]
fn sync_maybe_checkpoint(
    io: &mut impl SerialKvIo,
    tokens: &[i32],
    store: Option<&mut KvStore>,
    identity: Option<(u8, u8, u32)>,
    cached_floor: i32,
    enabled: bool,
) -> Result<(), GenerateError> {
    match (enabled, store, identity) {
        (true, Some(store), Some(identity)) => {
            io.sync_with_prefill_checkpoints(tokens, store, identity, cached_floor)
        }
        _ => io.sync(tokens),
    }
}

#[cfg(any(feature = "native", test))]
fn suppress_continued(store: &mut KvStore, target: i32) -> Option<i32> {
    if continued_target(store, target) != target {
        return None;
    }
    let old = store.continued_last_store_tokens;
    store.continued_last_store_tokens = target;
    Some(old)
}

#[cfg(any(feature = "native", test))]
fn restore_suppressed_continued(store: &mut KvStore, old: Option<i32>, target: i32) {
    if let Some(old) = old {
        if store.continued_last_store_tokens == target {
            store.continued_last_store_tokens = old;
        }
    }
}

#[cfg(any(feature = "native", test))]
fn cold_sync(io: &mut impl SerialKvIo, tokens: &[i32]) -> Result<i32, GenerateError> {
    io.sync(tokens)?;
    Ok(0)
}

#[cfg(any(feature = "native", test))]
fn cold_sync_and_store(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    identity: (u8, u8, u32),
    tokens: &[i32],
    prefill_checkpoints: bool,
) -> Result<i32, GenerateError> {
    let (model_id, quant_bits, ctx) = identity;
    let Ok(full_len) = i32::try_from(tokens.len()) else {
        sync_maybe_checkpoint(
            io,
            tokens,
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )?;
        return Ok(0);
    };
    if full_len < store.opt.min_tokens
        || store.opt.cold_max_tokens <= 0
        || full_len > store.opt.cold_max_tokens
    {
        sync_maybe_checkpoint(
            io,
            tokens,
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )?;
        return Ok(0);
    }
    let (user_id, assistant_id) = io.chat_token_ids();
    let anchor = kv_chat_anchor_pos(&store.opt, tokens, user_id, assistant_id);
    let target = if anchor >= store.opt.min_tokens {
        anchor
    } else {
        kv_store_len(&store.opt, full_len)
    };
    if target < store.opt.min_tokens || target > full_len {
        sync_maybe_checkpoint(
            io,
            tokens,
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )?;
        return Ok(0);
    }
    let target_i32 = target;
    let target = target_i32 as usize;
    let suppressed = suppress_continued(store, target_i32);
    let first_sync = if target < tokens.len() {
        sync_maybe_checkpoint(
            io,
            &tokens[..target],
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )
    } else {
        sync_maybe_checkpoint(
            io,
            tokens,
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )
    };
    if let Err(error) = first_sync {
        restore_suppressed_continued(store, suppressed, target_i32);
        return Err(error);
    }
    let cold_stored = matches!(
        try_store_live(io, store, model_id, quant_bits, ctx, KvReason::Cold, None,),
        Ok(true)
    );
    if cold_stored {
        store.continued_last_store_tokens = store.continued_last_store_tokens.max(target_i32);
    } else {
        restore_suppressed_continued(store, suppressed, target_i32);
    }
    if target < tokens.len() {
        sync_maybe_checkpoint(
            io,
            tokens,
            Some(store),
            Some(identity),
            0,
            prefill_checkpoints,
        )?;
    }
    Ok(0)
}

#[cfg(any(feature = "native", test))]
fn discard_loaded(store: &mut KvStore, io: &mut impl SerialKvIo, path: &Path) {
    store.continued_last_store_tokens = 0;
    let _ = store.discard(path);
    io.invalidate();
}

#[cfg(any(feature = "native", test))]
fn disk_sync_prompt(
    io: &mut impl SerialKvIo,
    store: Option<&mut KvStore>,
    model_id: i32,
    quant_bits: i32,
    prompt: &[u8],
    canonical_tokens: &[i32],
    checkpoint: Option<&ThinkingVisibleCheckpoint>,
    thinking_visible_eligible: bool,
    policy: DiskSyncPolicy,
) -> Result<i32, GenerateError> {
    disk_sync_prompt_impl(
        io,
        store,
        model_id,
        quant_bits,
        prompt,
        canonical_tokens,
        checkpoint,
        thinking_visible_eligible,
        policy,
        false,
    )
}

#[cfg(any(feature = "native", test))]
fn disk_sync_tool_replay(
    io: &mut impl SerialKvIo,
    store: Option<&mut KvStore>,
    model_id: i32,
    quant_bits: i32,
    prompt: &[u8],
    canonical_tokens: &[i32],
    policy: DiskSyncPolicy,
) -> Result<i32, GenerateError> {
    disk_sync_prompt_impl(
        io,
        store,
        model_id,
        quant_bits,
        prompt,
        canonical_tokens,
        None,
        false,
        policy,
        true,
    )
}

#[cfg(any(feature = "native", test))]
fn disk_sync_prompt_impl(
    io: &mut impl SerialKvIo,
    mut store: Option<&mut KvStore>,
    model_id: i32,
    quant_bits: i32,
    prompt: &[u8],
    canonical_tokens: &[i32],
    checkpoint: Option<&ThinkingVisibleCheckpoint>,
    thinking_visible_eligible: bool,
    policy: DiskSyncPolicy,
    allow_tool_map: bool,
) -> Result<i32, GenerateError> {
    let identity = kv_identity(model_id, quant_bits, io.ctx());
    let prefill_checkpoints = policy.load && !allow_tool_map;
    let live = io.live_tokens();
    if !live.is_empty() && canonical_tokens.starts_with(&live) {
        let cached = live.len() as i32;
        sync_maybe_checkpoint(
            io,
            canonical_tokens,
            store.as_deref_mut(),
            identity,
            cached,
            prefill_checkpoints,
        )?;
        return Ok(cached);
    }
    if thinking_visible_eligible {
        if let Some(checkpoint) = checkpoint {
            if !live.is_empty()
                && usize::try_from(checkpoint.frontier).ok() == Some(live.len())
                && checkpoint.text.len() < prompt.len()
                && prompt.starts_with(&checkpoint.text)
            {
                let cached = live.len() as i32;
                let mut effective = live;
                effective.extend(io.tokenize_suffix(&prompt[checkpoint.text.len()..])?);
                sync_maybe_checkpoint(
                    io,
                    &effective,
                    store.as_deref_mut(),
                    identity,
                    cached,
                    prefill_checkpoints,
                )?;
                return Ok(cached);
            }
        }
    }
    if !live.is_empty() {
        let rendered = io.render_tokens(&live)?;
        if prompt.starts_with(&rendered) {
            let cached = live.len() as i32;
            let mut effective = live;
            effective.extend(io.tokenize_suffix(&prompt[rendered.len()..])?);
            sync_maybe_checkpoint(
                io,
                &effective,
                store.as_deref_mut(),
                identity,
                cached,
                prefill_checkpoints,
            )?;
            return Ok(cached);
        }
    }

    let Some(store) = store else {
        return cold_sync(io, canonical_tokens);
    };
    store.continued_last_store_tokens = 0;
    let Some((model_id, quant_bits, ctx)) = identity else {
        return cold_sync(io, canonical_tokens);
    };
    if policy.save_current {
        let _ = try_store_live(
            io,
            store,
            model_id,
            quant_bits,
            ctx,
            KvReason::Evict,
            checkpoint,
        );
    }
    if !policy.load {
        return cold_sync(io, canonical_tokens);
    }
    let candidate = match store.text_prefix_candidate(prompt, model_id, quant_bits, ctx) {
        Ok(candidate) => candidate,
        Err(_) => {
            return cold_sync_and_store(
                io,
                store,
                (model_id, quant_bits, ctx),
                canonical_tokens,
                prefill_checkpoints,
            )
        }
    };
    let Some((path, envelope)) = candidate else {
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    };
    let extension_ok = match envelope.header.ext_flags {
        0 => envelope.trailer_bytes == 0,
        EXT_THINKING_VISIBLE => envelope.trailer_bytes == 0,
        EXT_TOOL_MAP => allow_tool_map && envelope.trailer_bytes > 0,
        _ => false,
    };
    if !extension_ok {
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    }
    if io
        .load_payload_range(
            &path,
            envelope.payload_offset,
            envelope.header.payload_bytes,
        )
        .is_err()
    {
        io.invalidate();
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    }
    let loaded = io.live_tokens();
    if loaded.len() != envelope.header.tokens as usize {
        io.invalidate();
        let _ = store.discard(&path);
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    }
    let cached = loaded.len() as i32;
    store.continued_last_store_tokens = cached;
    let _ = store.touch_hit(&path);
    let mut effective = loaded;
    let suffix = &prompt[envelope.text.len()..];
    let suffix_tokens = match io.tokenize_suffix(suffix) {
        Ok(tokens) => tokens,
        Err(error) => {
            discard_loaded(store, io, &path);
            return Err(error);
        }
    };
    effective.extend(suffix_tokens);
    if let Err(error) = sync_maybe_checkpoint(
        io,
        &effective,
        Some(store),
        Some((model_id, quant_bits, ctx)),
        cached,
        prefill_checkpoints,
    ) {
        discard_loaded(store, io, &path);
        return Err(error);
    }
    Ok(cached)
}

#[derive(Debug, Clone, Default)]
pub struct GenerateOutcome {
    pub tool_ids: Vec<String>,
    pub bank: Option<i32>,
    pub generation: u64,
    pub frontier: i32,
    pub finish: String,
}

pub fn generation_blocked(parsed: &ParsedRequest, model_id: i32) -> Option<&'static str> {
    if parsed.images.is_empty() {
        None
    } else if model_id != ModelSyntax::Qwen4Exp as i32 {
        Some("image input is supported only by Qwen4Exp")
    } else {
        Some("image input requires continuous runtime")
    }
}

pub fn chat_format_for_syntax(syntax: ModelSyntax) -> ChatFormat {
    match syntax {
        ModelSyntax::SolarOpen2 => ChatFormat::SolarOpen2,
        ModelSyntax::Exaone => ChatFormat::Exaone,
        ModelSyntax::Qwen4Exp => ChatFormat::Qwen4Exp,
        ModelSyntax::DeepSeek | ModelSyntax::Motif3 | ModelSyntax::Dots3 => ChatFormat::DeepSeek,
    }
}

pub fn stream_req_from_parsed(parsed: &ParsedRequest, model_id: i32) -> StreamReq {
    StreamReq {
        kind: parsed.kind,
        api: parsed.api,
        model: parsed.model.clone(),
        think_mode: parsed.think_mode,
        has_tools: parsed.has_tools,
        stream: parsed.stream,
        stream_include_usage: parsed.stream_include_usage,
        reasoning_summary_emit: parsed.reasoning_summary_emit,
        chat_format: chat_format_for_syntax(syntax_for_model_id(model_id)),
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        timings: ReqTimings::default(),
        tool_orders: parsed.tool_orders.clone(),
    }
}

pub fn render_prompt(parsed: &ParsedRequest, model_id: i32) -> Result<Vec<u8>, GenerateError> {
    match parsed.kind {
        ReqKind::Completion => Ok(parsed.prompt_text.clone().unwrap_or_default().into_bytes()),
        ReqKind::Chat => Ok(render_chat_choice(
            syntax_for_model_id(model_id),
            &parsed.messages,
            &parsed.tool_schemas,
            &parsed.tool_orders,
            parsed.think_mode,
            parsed.tool_choice,
        )?),
    }
}

pub(crate) fn ordinary_disk_cache_eligible(parsed: &ParsedRequest) -> bool {
    parsed.api == Api::Openai
        && !think_mode_enabled(parsed.think_mode)
        && !parsed.has_tools
        && !parsed.has_tool_results
        && parsed.live_call_ids.is_empty()
}

fn tool_replay_scope(parsed: &ParsedRequest, syntax: ModelSyntax) -> bool {
    parsed.kind == ReqKind::Chat
        && parsed.api == Api::Openai
        && !think_mode_enabled(parsed.think_mode)
        && parsed.live_call_ids.is_empty()
        && matches!(
            syntax,
            ModelSyntax::DeepSeek | ModelSyntax::SolarOpen2 | ModelSyntax::Qwen4Exp
        )
}

fn tool_replay_disk_cache_eligible(parsed: &ParsedRequest, syntax: ModelSyntax) -> bool {
    tool_replay_scope(parsed, syntax)
        && parsed.messages.iter().any(|message| {
            !message.calls.is_empty()
                || !message.tool_call_id.is_empty()
                || !message.tool_call_ids.is_empty()
        })
}

fn tool_replay_producer_eligible(parsed: &ParsedRequest, syntax: ModelSyntax) -> bool {
    tool_replay_scope(parsed, syntax) && parsed.has_tools
}

fn thinking_visible_cache_eligible(parsed: &ParsedRequest) -> bool {
    parsed.kind == ReqKind::Chat && parsed.api != Api::Responses
}

fn motif3_no_think_visible_checkpoint(
    parsed: &ParsedRequest,
    syntax: ModelSyntax,
    prompt: &[u8],
    content: &[u8],
    finish: &str,
) -> Option<Vec<u8>> {
    if parsed.kind != ReqKind::Chat
        || syntax != ModelSyntax::Motif3
        || !ordinary_disk_cache_eligible(parsed)
        || finish == "error"
        || finish == "length"
    {
        return None;
    }
    let prefix = prompt.strip_suffix(b"<think></think>")?;
    let content = content.trim_ascii();
    let mut visible = Vec::with_capacity(prefix.len() + content.len());
    visible.extend_from_slice(prefix);
    visible.extend_from_slice(content);
    Some(visible)
}

fn prepare_required_prefixes(
    engine: &dyn DecodeIo,
    parsed: &mut ParsedRequest,
    format: ChatFormat,
) -> Result<(), GenerateError> {
    if !engine.tokenizes_control_literals() {
        return Ok(());
    }
    if parsed.tool_choice != ToolChoice::Required && !parsed.has_tool_results {
        return Ok(());
    }
    if parsed.required_think_end_prefix.is_empty() {
        let toks = engine.tokenize_rendered_chat(think_end(format).as_bytes())?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize thinking control prefix".into(),
            ));
        }
        parsed.required_think_end_prefix = toks;
    }
    if parsed.tool_choice == ToolChoice::Required && parsed.required_tool_prefix.is_empty() {
        let marker = match format {
            ChatFormat::SolarOpen2 => crate::render::SOLAR_TOOL_CALLS,
            ChatFormat::Exaone => "<tool_call>",
            ChatFormat::Qwen4Exp => crate::render::QWEN_TOOL_CALL_START,
            ChatFormat::DeepSeek => crate::tools::DSML_TOOL_CALLS_START,
        };
        let toks = engine.tokenize_rendered_chat(marker.as_bytes())?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize required tool control prefix".into(),
            ));
        }
        parsed.required_tool_prefix = toks;
    }
    Ok(())
}

fn find_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

pub fn stop_list_find_from(stops: &[String], text: &[u8], from: usize) -> Option<(usize, usize)> {
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

pub fn stop_list_max_len(stops: &[String]) -> usize {
    stops.iter().map(|s| s.len()).max().unwrap_or(0)
}

pub fn stop_list_stream_safe_len(stops: &[String], text_len: usize) -> usize {
    let max = stop_list_max_len(stops);
    if max <= 1 || text_len <= max - 1 {
        return if max <= 1 { text_len } else { 0 };
    }
    text_len - (max - 1)
}

#[allow(dead_code)]
fn split_think(raw: &[u8], think: bool, fmt: ChatFormat) -> (Vec<u8>, Vec<u8>) {
    if !think {
        return (raw.to_vec(), Vec::new());
    }
    let start = think_start(fmt).as_bytes();
    let end = think_end(fmt).as_bytes();
    let body = if raw.starts_with(start) {
        &raw[start.len()..]
    } else {
        raw
    };
    if let Some(i) = find_substr(body, end) {
        (body[i + end.len()..].to_vec(), body[..i].to_vec())
    } else {
        (Vec::new(), body.to_vec())
    }
}

pub(crate) fn responses_ids(job_id: &str) -> (String, String, String) {
    let mut h = 2_166_136_261u32;
    for b in job_id.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(16_777_619);
    }
    let hex = format!("{h:08x}{h:08x}{h:08x}");
    (
        format!("resp_{hex}"),
        format!("rs_{hex}"),
        format!("msg_{hex}"),
    )
}

fn flush(w: &mut Writer, out: &mut impl Write) -> Result<(), GenerateError> {
    if !w.out.is_empty() {
        out.write_all(&w.out).map_err(|_| GenerateError::Io)?;
        w.out.clear();
    }
    out.flush().map_err(|_| GenerateError::Io)
}

fn append_recovery_suffix(engine: &mut dyn DecodeIo, suffix: &[u8]) -> Result<i32, GenerateError> {
    if suffix.is_empty() {
        return Ok(0);
    }
    let before = engine.pos();
    let mut target = engine.session_tokens();
    let extra = engine.tokenize_rendered_chat(suffix)?;
    target.extend(extra);
    engine.sync(&target)?;
    let delta = engine.pos() - before;
    Ok(if delta > 0 { delta } else { 0 })
}

fn decode_pass(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    req: &StreamReq,
    job_id: &str,
    acc: &mut SemAccum,
    finish: &mut &'static str,
    max_tokens: i32,
    rng: &mut u64,
    w: &mut Writer,
    out: &mut impl Write,
    mut oa: Option<&mut OpenaiStream>,
    mut anth: Option<&mut AnthropicStream>,
    mut resp: Option<&mut ResponsesStream>,
    first_tok: &mut Option<Instant>,
    decode_steps: &mut i32,
) -> Result<(), GenerateError> {
    while acc.completion < max_tokens && engine.pos() < engine.ctx() {
        out.flush().map_err(|_| GenerateError::Io)?;
        if continued_decode_allowed(acc) {
            store_continued_best_effort(engine);
        }
        let mut temperature = parsed.temperature;
        let mut top_k = parsed.top_k;
        let mut top_p = parsed.top_p;
        let mut min_p = parsed.min_p;
        if think_mode_enabled(parsed.think_mode) {
            temperature = DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DEFAULT_TOP_P;
            min_p = DEFAULT_MIN_P;
        }
        let policy = SamplePolicy {
            tool_choice: parsed.tool_choice,
            has_tool_results: parsed.has_tool_results,
            think_mode: parsed.think_mode,
            max_tokens: parsed.max_tokens,
            required_tool_prefix: &parsed.required_tool_prefix,
            required_think_end_prefix: &parsed.required_think_end_prefix,
        };
        let ov = acc.sampling_override(&policy);
        if matches!(ov, SampleOverride::Greedy) {
            temperature = 0.0;
        }
        let token = if let SampleOverride::Token(t) = ov {
            t
        } else {
            engine.sample(temperature, top_k, top_p, min_p, rng)
        };
        if token < 0 || engine.token_is_stop(token) {
            *finish = "stop";
            break;
        }
        engine.eval(token)?;
        *decode_steps += 1;
        if first_tok.is_none() {
            *first_tok = Some(Instant::now());
        }
        let piece = engine.token_text(token)?;
        let feed = acc.feed(&piece, &parsed.stops);

        if req.stream {
            let view = &acc.text[..feed.emit_limit.min(acc.text.len())];
            match req.api {
                Api::Openai if req.kind == ReqKind::Completion => {
                    if let Some(delta) = last_delta(&acc.text, feed.emit_limit, piece.len()) {
                        sse_chunk(w, req, job_id, Some(delta), None);
                    }
                }
                Api::Openai => {
                    if let Some(st) = oa.as_mut() {
                        openai_sse_stream_update(w, req, job_id, st, view, false);
                    }
                }
                Api::Anthropic => {
                    if let Some(st) = anth.as_mut() {
                        if !anthropic_sse_stream_update(w, req, job_id, st, view, false) {
                            return Err(GenerateError::Io);
                        }
                    }
                }
                Api::Responses => {
                    if let Some(st) = resp.as_mut() {
                        if !responses_sse_stream_update(w, req, st, view, false) {
                            return Err(GenerateError::Io);
                        }
                    }
                }
            }
            flush(w, out)?;
        }

        if feed.hit_stop {
            *finish = "stop";
            engine.invalidate();
            break;
        }
        if acc.track_tools && acc.saw_tool_end && req.chat_format == ChatFormat::DeepSeek {
            *finish = "tool_calls";
            break;
        }
    }
    Ok(())
}

pub fn generate_and_write(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    out: &mut impl Write,
) -> Result<GenerateOutcome, GenerateError> {
    generate_and_write_at(
        engine,
        parsed,
        job_id,
        created,
        cors,
        default_tokens,
        Instant::now(),
        out,
    )
}

/// Queue-aware serial entry point. `t_arrive` is captured by the HTTP owner;
/// callers outside the queued server should use [`generate_and_write`].
pub fn generate_and_write_at(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    t_arrive: Instant,
    out: &mut impl Write,
) -> Result<GenerateOutcome, GenerateError> {
    let (outcome, terminal) = generate_terminal_at(
        engine,
        parsed,
        job_id,
        created,
        cors,
        default_tokens,
        t_arrive,
        out,
    )?;
    out.write_all(&terminal).map_err(|_| GenerateError::Io)?;
    Ok(outcome)
}

/// Serial phase 1: everything before the prompt sync — blocked check, tool
/// replay restore, required prefixes, render, tokenize. Split out so the
/// server's `ensure_serial_session_fit` (C `serial_session_ensure_fit`) can
/// size the session from the exact sync-bound token count without a second
/// render+tokenize pass.
pub(crate) struct PreparedSerialPrompt {
    pub(crate) parsed: ParsedRequest,
    pub(crate) tool_replay: bool,
    pub(crate) prompt: Vec<u8>,
    pub(crate) tokens: Vec<i32>,
}

pub(crate) fn prepare_serial_prompt(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
) -> Result<PreparedSerialPrompt, GenerateError> {
    if let Some(msg) = generation_blocked(parsed, engine.model_id()) {
        return Err(GenerateError::Unsupported(msg));
    }

    let mut parsed = parsed.clone();
    let syntax = syntax_for_model_id(engine.model_id());
    let tool_replay = tool_replay_disk_cache_eligible(&parsed, syntax);
    if tool_replay {
        engine.restore_tool_replay(&mut parsed.messages);
    }
    prepare_required_prefixes(
        engine,
        &mut parsed,
        chat_format_for_syntax(syntax_for_model_id(engine.model_id())),
    )?;

    let prompt = render_prompt(&parsed, engine.model_id())?;
    let tokens = match parsed.kind {
        ReqKind::Completion => {
            let text = std::str::from_utf8(&prompt).unwrap_or("");
            engine.tokenize_text(text)?
        }
        ReqKind::Chat => engine.tokenize_rendered_chat(&prompt)?,
    };
    Ok(PreparedSerialPrompt {
        parsed,
        tool_replay,
        prompt,
        tokens,
    })
}

/// Runs serial generation while withholding only the final wire terminal.
/// Streaming headers/deltas still flow through `out` as they are produced.
pub(crate) fn generate_terminal_at(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    t_arrive: Instant,
    out: &mut impl Write,
) -> Result<(GenerateOutcome, Vec<u8>), GenerateError> {
    let prep = prepare_serial_prompt(engine, parsed)?;
    generate_terminal_prepared(
        engine,
        prep,
        job_id,
        created,
        cors,
        default_tokens,
        t_arrive,
        out,
    )
}

/// Serial phase 2: prompt sync onward, on a [`PreparedSerialPrompt`].
pub(crate) fn generate_terminal_prepared(
    engine: &mut dyn DecodeIo,
    prep: PreparedSerialPrompt,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    t_arrive: Instant,
    out: &mut impl Write,
) -> Result<(GenerateOutcome, Vec<u8>), GenerateError> {
    let PreparedSerialPrompt {
        parsed,
        tool_replay,
        prompt,
        tokens,
    } = prep;
    let syntax = syntax_for_model_id(engine.model_id());
    let mut req = stream_req_from_parsed(&parsed, engine.model_id());
    let mut w = Writer::new(created);
    if req.stream {
        w.out.extend_from_slice(&sse_headers(cors));
        flush(&mut w, out)?;
    }
    let t_prefill = Instant::now();
    let sync_result = if tool_replay {
        engine.sync_tool_replay_prompt(&prompt, &tokens)
    } else {
        engine.sync_prompt(
            &prompt,
            &tokens,
            ordinary_disk_cache_eligible(&parsed),
            thinking_visible_cache_eligible(&parsed),
        )
    };
    let cached = match sync_result {
        Ok(cached) => cached,
        Err(error) if req.stream => {
            let message = error.to_string();
            stream_error(&mut w, &req, None, &message);
            flush(&mut w, out)?;
            return Err(GenerateError::Streamed(message));
        }
        Err(error) => return Err(error),
    };
    store_continued_best_effort(engine);
    let decode_t0 = Instant::now();
    let prefill_elapsed = engine
        .prompt_sync_elapsed()
        .unwrap_or_else(|| decode_t0.duration_since(t_prefill));
    let mut first_tok = None;
    let mut decode_steps = 0i32;

    let prompt_n = engine.pos();
    let mut rng = parsed.seed;
    req.cache_read_tokens = cached.clamp(0, prompt_n);
    req.cache_write_tokens = prompt_n - req.cache_read_tokens;
    let mut acc;
    let mut finish;
    let mut recovery_attempted = false;

    let mut oa = if req.stream && req.api == Api::Openai && req.kind == ReqKind::Chat {
        let mut stream = openai_stream_start(&req);
        stream.tool.use_random_ids();
        Some(stream)
    } else {
        None
    };
    let mut anth = if req.stream && req.api == Api::Anthropic {
        let mut stream = anthropic_sse_start_live(&mut w, &req, job_id, prompt_n);
        stream.tool.use_random_ids();
        Some(stream)
    } else {
        None
    };
    let mut resp = if req.stream && req.api == Api::Responses {
        let (rid, rsid, mid) = responses_ids(job_id);
        let mut st = responses_stream_init(&req, &rid, &rsid, &mid);
        responses_sse_created(&mut w, &req, &mut st, created);
        Some(st)
    } else {
        None
    };

    if req.stream {
        if req.api == Api::Openai && req.kind == ReqKind::Chat {
            sse_chunk(&mut w, &req, job_id, None, None);
        }
        flush(&mut w, out)?;
    }

    let mut parsed_gen;
    loop {
        let mut max_tokens =
            decode_budget(parsed.max_tokens_set, parsed.max_tokens, default_tokens);
        let room = engine.ctx() - engine.pos();
        if room >= 0 && max_tokens > room {
            max_tokens = room;
        }
        acc = SemAccum::init(
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            think_mode_enabled(parsed.think_mode),
            req.chat_format,
            &prompt,
        );
        finish = "length";

        let decoded = decode_pass(
            engine,
            &parsed,
            &req,
            job_id,
            &mut acc,
            &mut finish,
            max_tokens,
            &mut rng,
            &mut w,
            out,
            oa.as_mut(),
            anth.as_mut(),
            resp.as_mut(),
            &mut first_tok,
            &mut decode_steps,
        );
        if let Err(error) = decoded {
            if !req.stream || matches!(&error, GenerateError::Io) {
                return Err(error);
            }
            let message = error.to_string();
            stream_error(&mut w, &req, resp.as_mut(), &message);
            flush(&mut w, out)?;
            return Err(GenerateError::Streamed(message));
        }

        finish = terminal_finish(acc.thinking_inside(), finish);
        match truncation_outcome(
            syntax,
            req.chat_format,
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            acc.saw_tool_start,
            acc.saw_tool_end,
            finish,
            parsed.stream,
            recovery_attempted,
            &acc.text,
            &parsed.tool_orders,
        ) {
            TruncationOutcome::Repair(text) => {
                acc.text = text;
                acc.saw_tool_end = true;
            }
            TruncationOutcome::RetryUnterminated => {
                if append_recovery_suffix(
                    engine,
                    &build_invalid_tool_error_suffix(
                        req.chat_format,
                        parsed.think_mode,
                        acc.thinking_inside(),
                        &prompt,
                        "unterminated tool call",
                    ),
                )
                .is_ok()
                {
                    recovery_attempted = true;
                    continue;
                }
                finish = "error";
            }
            TruncationOutcome::ErrorUnterminated => {
                finish = "error";
            }
            TruncationOutcome::None => {}
        }

        parsed_gen = if parsed.kind == ReqKind::Chat {
            let (pg, recovered_finish) = parse_generated_for_response(
                syntax,
                &acc.text,
                parsed.has_tools,
                acc.saw_tool_start,
                think_mode_enabled(parsed.think_mode),
                req.chat_format,
                &parsed.tool_orders,
                finish,
            );
            finish = recovered_finish;
            pg
        } else {
            crate::tools::ParsedGenerated {
                content: acc.text.clone(),
                ok: true,
                ..Default::default()
            }
        };
        if !parsed_gen.ok
            && parse_failure_should_retry(
                syntax,
                parsed.stream,
                recovery_attempted,
                finish,
                parsed_gen.recovered,
                parsed.has_tools,
                acc.saw_tool_start,
            )
        {
            if append_recovery_suffix(
                engine,
                &build_invalid_tool_error_suffix(
                    req.chat_format,
                    parsed.think_mode,
                    acc.thinking_inside(),
                    &prompt,
                    "invalid tool call",
                ),
            )
            .is_ok()
            {
                recovery_attempted = true;
                continue;
            }
            finish = "error";
        }
        break;
    }

    let completion = acc.completion;
    if completion > 0 {
        if let Some(t_first) = first_tok {
            req.timings = ReqTimings {
                valid: true,
                ttft_ms: t_first.duration_since(t_arrive).as_secs_f64() * 1e3,
                prefill_ms: prefill_elapsed.as_secs_f64() * 1e3,
                decode_ms: Instant::now().duration_since(t_first).as_secs_f64() * 1e3,
                prefill_tokens: prompt_n - req.cache_read_tokens,
                prefill_cached: req.cache_read_tokens,
                decode_tokens: completion,
                decode_steps,
            };
        }
    }
    if !parsed_gen.calls.is_empty() {
        if let Some(st) = oa.as_ref() {
            st.tool.apply_ids(&mut parsed_gen.calls);
        }
        if let Some(st) = anth.as_ref() {
            st.tool.apply_ids(&mut parsed_gen.calls);
        }
        assign_tool_ids(
            &mut parsed_gen.calls,
            if parsed.api == Api::Anthropic {
                "toolu_"
            } else {
                "call_"
            },
        );
        if tool_replay_producer_eligible(&parsed, syntax) {
            engine.remember_tool_replay(&parsed_gen.calls, &parsed_gen.raw_dsml);
        }
        finish = "tool_calls";
    }
    if let Some(visible) =
        motif3_no_think_visible_checkpoint(&parsed, syntax, &prompt, &parsed_gen.content, finish)
    {
        engine.remember_thinking_visible_checkpoint(visible);
    }
    let matched_stop = acc.matched_stop.clone();
    let terminal = if req.stream {
        match req.api {
            Api::Openai if req.kind == ReqKind::Completion => {
                sse_chunk(&mut w, &req, job_id, None, Some(finish));
                sse_done(&mut w, &req, job_id, prompt_n, completion);
            }
            Api::Openai => {
                if let Some(st) = oa.as_mut() {
                    openai_sse_finish_live(
                        &mut w,
                        &req,
                        job_id,
                        st,
                        &acc.text,
                        finish,
                        prompt_n,
                        completion,
                        &parsed_gen.calls,
                    );
                }
            }
            Api::Anthropic => {
                if let Some(st) = anth.as_mut() {
                    if !anthropic_sse_finish_live(
                        &mut w,
                        &req,
                        job_id,
                        st,
                        &acc.text,
                        finish,
                        matched_stop.as_deref(),
                        completion,
                        &parsed_gen.calls,
                    ) {
                        return Err(GenerateError::Io);
                    }
                }
            }
            Api::Responses => {
                if let Some(st) = resp.as_mut() {
                    if !responses_sse_finish_live(
                        &mut w,
                        &req,
                        st,
                        &acc.text,
                        finish,
                        prompt_n,
                        completion,
                        0,
                        created,
                        &parsed_gen.calls,
                    ) {
                        return Err(GenerateError::Io);
                    }
                }
            }
        }
        std::mem::take(&mut w.out)
    } else {
        let bytes = match req.api {
            Api::Anthropic => anthropic_final_response(
                &req,
                job_id,
                &parsed_gen.content,
                Some(&parsed_gen.reasoning),
                finish,
                matched_stop.as_deref(),
                prompt_n,
                completion,
                cors,
                &parsed_gen.calls,
            ),
            Api::Responses => {
                let (rid, rsid, mid) = responses_ids(job_id);
                responses_final_response(
                    &req,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    finish,
                    prompt_n,
                    completion,
                    0,
                    created,
                    cors,
                    &rid,
                    &rsid,
                    &mid,
                    &parsed_gen.calls,
                )
            }
            Api::Openai => final_response(
                &req,
                job_id,
                &parsed_gen.content,
                Some(&parsed_gen.reasoning),
                finish,
                prompt_n,
                completion,
                created,
                cors,
                &parsed_gen.calls,
            ),
        };
        bytes
    };
    let outcome = GenerateOutcome {
        tool_ids: parsed_gen
            .calls
            .iter()
            .map(|c| c.id.clone())
            .filter(|id| !id.is_empty())
            .collect(),
        bank: None,
        generation: engine.generation(),
        frontier: engine.pos(),
        finish: finish.to_string(),
    };
    Ok((outcome, terminal))
}

fn last_delta(raw: &[u8], emit_limit: usize, piece_len: usize) -> Option<&[u8]> {
    if emit_limit == 0 {
        return None;
    }
    let start = raw.len().saturating_sub(piece_len);
    if start >= emit_limit {
        return None;
    }
    Some(&raw[start..emit_limit])
}

/// Tape engine for tests. Does not open a GGUF.
pub struct ScriptedDecode {
    pub model_id: i32,
    pub prompt_tokens: Vec<i32>,
    pub steps: Vec<ScriptedStep>,
    pub idx: usize,
    pub pos: i32,
    pub ctx: i32,
    pub generation: u64,
    pub live: Vec<i32>,
    pub suffix_tokens: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct ScriptedStep {
    pub token: i32,
    pub piece: Vec<u8>,
    pub stop: bool,
}

impl ScriptedDecode {
    pub fn from_pieces(pieces: &[&[u8]]) -> Self {
        let steps = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| ScriptedStep {
                token: (i as i32) + 1,
                piece: p.to_vec(),
                stop: false,
            })
            .chain(std::iter::once(ScriptedStep {
                token: 99,
                piece: Vec::new(),
                stop: true,
            }))
            .collect();
        Self {
            model_id: 0,
            prompt_tokens: vec![1],
            steps,
            idx: 0,
            pos: 0,
            ctx: 8192,
            generation: 1,
            live: Vec::new(),
            suffix_tokens: Vec::new(),
        }
    }
}

impl DecodeIo for ScriptedDecode {
    fn model_id(&self) -> i32 {
        self.model_id
    }

    fn tokenize_text(&self, _text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(self.prompt_tokens.clone())
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        if find_substr(text, b"Tool error:").is_some() {
            if self.suffix_tokens.is_empty() {
                Ok(vec![7])
            } else {
                Ok(self.suffix_tokens.clone())
            }
        } else {
            Ok(self.prompt_tokens.clone())
        }
    }

    fn tokenizes_control_literals(&self) -> bool {
        false
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        Ok(self
            .steps
            .iter()
            .find(|s| s.token == token)
            .map(|s| s.piece.clone())
            .unwrap_or_default())
    }

    fn token_is_stop(&self, token: i32) -> bool {
        self.steps.iter().any(|s| s.token == token && s.stop)
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        self.live = tokens.to_vec();
        self.pos = tokens.len() as i32;
        Ok(())
    }

    fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
        self.pos += 1;
        Ok(())
    }

    fn sample(
        &mut self,
        _temperature: f32,
        _top_k: i32,
        _top_p: f32,
        _min_p: f32,
        _rng: &mut u64,
    ) -> i32 {
        if self.idx >= self.steps.len() {
            return -1;
        }
        let t = self.steps[self.idx].token;
        self.idx += 1;
        t
    }

    fn pos(&self) -> i32 {
        self.pos
    }

    fn ctx(&self) -> i32 {
        self.ctx
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn session_tokens(&self) -> Vec<i32> {
        if self.live.is_empty() {
            self.prompt_tokens.clone()
        } else {
            self.live.clone()
        }
    }

    fn invalidate(&mut self) {
        self.live.clear();
        self.pos = 0;
    }
}

#[cfg(feature = "native")]
struct NativeSerialKvIo<'s, 'm, 'v, 't> {
    session: &'s mut ds4_core::Session<'m>,
    vocab: &'v ds4_core::Vocab,
    tool_memory: &'t ToolMemory,
    prefill_checkpoints: bool,
    sync_elapsed: Duration,
}

#[cfg(feature = "native")]
impl SerialKvIo for NativeSerialKvIo<'_, '_, '_, '_> {
    fn ctx(&self) -> i32 {
        self.session.ctx()
    }

    fn chat_token_ids(&self) -> (i32, i32) {
        (self.vocab.user_id, self.vocab.assistant_id)
    }

    fn live_len(&self) -> i32 {
        self.session.host().live_len()
    }

    fn live_tokens(&self) -> Vec<i32> {
        if self.session.host().valid {
            self.session.host().tokens().to_vec()
        } else {
            Vec::new()
        }
    }

    fn render_tokens(&self, tokens: &[i32]) -> Result<Vec<u8>, GenerateError> {
        let mut text = Vec::new();
        for &token in tokens {
            text.extend(self.vocab.token_text(token));
        }
        Ok(text)
    }

    fn checkpoint_trailer(&self, text: &[u8]) -> Option<Vec<u8>> {
        self.tool_memory.checkpoint(text)
    }

    fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError> {
        Ok(self.vocab.encode_rendered_bytes(suffix))
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        let tokens = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        let started = Instant::now();
        let result = self
            .session
            .sync(&tokens)
            .map_err(|error| GenerateError::Engine(error.to_string()));
        self.sync_elapsed += started.elapsed();
        result
    }

    fn sync_with_prefill_checkpoints(
        &mut self,
        tokens: &[i32],
        store: &mut KvStore,
        identity: (u8, u8, u32),
        cached_floor: i32,
    ) -> Result<(), GenerateError> {
        if !self.prefill_checkpoints {
            return self.sync(tokens);
        }
        let tokens = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        let (model_id, quant_bits, ctx) = identity;
        let vocab = self.vocab;
        let tool_memory = self.tool_memory;
        let started = Instant::now();
        let result = self
            .session
            .sync_progress(&tokens, |checkpoint| {
                let current = i32::try_from(checkpoint.current()).unwrap_or(0);
                if current <= cached_floor {
                    return;
                }
                let target = continued_target(store, current);
                if target == 0 {
                    return;
                }
                let mut text = Vec::new();
                for &token in checkpoint.tokens() {
                    text.extend(vocab.token_text(token));
                }
                let Some(trailer) = tool_memory.checkpoint(&text) else {
                    return;
                };
                let Ok(token_count) = u32::try_from(checkpoint.current()) else {
                    return;
                };
                let mut header = kv_header(model_id, quant_bits, ctx, token_count);
                header.reason = KvReason::Continued;
                match write_checkpoint(store, header, &text, &trailer, |path| {
                    checkpoint
                        .save_payload(path)
                        .map_err(|error| GenerateError::Engine(error.to_string()))
                }) {
                    Ok(()) => {
                        store.continued_last_store_tokens =
                            store.continued_last_store_tokens.max(target);
                    }
                    Err(error) => {
                        eprintln!(
                            "ds4-server-rs: intermediate continued KV checkpoint failed: {error}"
                        );
                    }
                }
            })
            .map_err(|error| GenerateError::Engine(error.to_string()));
        self.sync_elapsed += started.elapsed();
        result
    }

    fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError> {
        self.session
            .save_payload(path)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn load_payload_range(
        &mut self,
        path: &Path,
        offset: u64,
        length: u64,
    ) -> Result<(), GenerateError> {
        self.session
            .load_payload_range(path, offset, length)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn invalidate(&mut self) {
        self.session.invalidate();
    }
}

#[cfg(feature = "native")]
pub struct NativeDecode<'a> {
    model: &'a ds4_core::Model,
    vocab: Option<&'a ds4_core::Vocab>,
    session: Option<ds4_core::Session<'a>>,
    store: Option<KvStore>,
    session_disk_storable: bool,
    thinking_visible: Option<ThinkingVisibleCheckpoint>,
    tool_memory: ToolMemory,
    prompt_sync_elapsed: Option<Duration>,
    ctx: i32,
}

#[cfg(feature = "native")]
impl<'a> NativeDecode<'a> {
    pub fn new(model: &'a ds4_core::Model, ctx: i32) -> Self {
        Self {
            model,
            vocab: None,
            session: None,
            store: None,
            session_disk_storable: false,
            thinking_visible: None,
            tool_memory: ToolMemory::default(),
            prompt_sync_elapsed: None,
            ctx,
        }
    }

    pub fn with_vocab(mut self, vocab: &'a ds4_core::Vocab) -> Self {
        self.vocab = Some(vocab);
        self
    }

    pub fn with_store(mut self, store: KvStore) -> Self {
        self.store = Some(store);
        self
    }

    fn sync_prompt_inner(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
        disk_eligible: bool,
        thinking_visible_eligible: bool,
        tool_replay: bool,
    ) -> Result<i32, GenerateError> {
        self.prompt_sync_elapsed = None;
        let model_id = self.model.model_id();
        let quant_bits = self.model.routed_quant_bits();
        let vocab = self.vocab.unwrap_or_else(|| self.model.vocab());
        let save_current = self.session_disk_storable;
        let has_store = self.store.is_some();
        let prefill_checkpoints = intermediate_prefill_eligible(
            self.model.family() == ds4_core::ModelFamily::DeepSeek4,
            self.model.backend() == ds4_core::Backend::Cuda,
            disk_eligible,
            tool_replay,
        );
        self.session()?;
        let (session, store, checkpoint, tool_memory) = (
            &mut self.session,
            &mut self.store,
            &self.thinking_visible,
            &self.tool_memory,
        );
        let session = session
            .as_mut()
            .ok_or_else(|| GenerateError::Engine("native session was not created".into()))?;
        let mut io = NativeSerialKvIo {
            session,
            vocab,
            tool_memory,
            prefill_checkpoints,
            sync_elapsed: Duration::ZERO,
        };
        let policy = DiskSyncPolicy {
            save_current,
            load: disk_eligible,
        };
        let result = if tool_replay {
            disk_sync_tool_replay(
                &mut io,
                store.as_mut(),
                model_id,
                quant_bits,
                prompt,
                tokens,
                policy,
            )
        } else {
            disk_sync_prompt(
                &mut io,
                store.as_mut(),
                model_id,
                quant_bits,
                prompt,
                tokens,
                checkpoint.as_ref(),
                thinking_visible_eligible,
                policy,
            )
        };
        if result.is_ok() {
            self.prompt_sync_elapsed = Some(io.sync_elapsed);
        }
        self.session_disk_storable = result.is_ok() && disk_eligible && has_store;
        settle_thinking_visible_checkpoint(&mut self.thinking_visible, result.is_ok());
        result
    }

    fn session(&mut self) -> Result<&mut ds4_core::Session<'a>, GenerateError> {
        if self.session.is_none() {
            let s = self
                .model
                .session(self.ctx)
                .map_err(|e| GenerateError::Engine(e.to_string()))?;
            self.session = Some(s);
        }
        Ok(self.session.as_mut().unwrap())
    }
}

#[cfg(feature = "native")]
impl DecodeIo for NativeDecode<'_> {
    fn model_id(&self) -> i32 {
        self.model.model_id()
    }

    fn kv_store_mut(&mut self) -> Option<&mut KvStore> {
        self.store.as_mut()
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.encode_text(text));
        }
        self.model
            .tokenize_text(text)
            .map(|b| b.as_slice().to_vec())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.encode_rendered_bytes(text));
        }
        let s = std::str::from_utf8(text)
            .map_err(|_| GenerateError::Engine("prompt not utf8".into()))?;
        self.model
            .tokenize_rendered_chat(s)
            .map(|b| b.as_slice().to_vec())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.token_text(token));
        }
        self.model
            .token_text(token)
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn token_is_stop(&self, token: i32) -> bool {
        if let Some(v) = self.vocab {
            return v.is_stop(token);
        }
        self.model.token_is_stop(token)
    }

    fn native_graph_fit(&self, ctx: i32) -> Option<NativeGraphFit> {
        let quote = self.model.session_graph_fit_quote(ctx)?;
        Some(NativeGraphFit {
            fits: quote.fits,
            need_bytes: quote.need_bytes,
            avail_bytes: quote.avail_bytes,
            headroom_bytes: quote.headroom_bytes,
            deficit_bytes: quote.deficit_bytes,
            fail_open: quote.fail_open,
        })
    }

    fn serial_session_probe(&self) -> Option<SerialSessionProbe> {
        Some(match &self.session {
            Some(s) => SerialSessionProbe {
                cur_ctx: s.ctx(),
                graph_pending: s.graph_pending(),
            },
            // No session yet: the C boot-shape lazy session (pending at -c).
            None => SerialSessionProbe {
                cur_ctx: self.ctx,
                graph_pending: true,
            },
        })
    }

    fn serial_session_rightsize(&mut self, target_ctx: i32) -> Result<(), GenerateError> {
        // Free BEFORE creating so a committed right-sized graph's own GiBs
        // count as available for its replacement (the C regrow case).
        self.session = None;
        self.session_disk_storable = false;
        self.thinking_visible = None;
        let s = self
            .model
            .session(target_ctx)
            .map_err(|e| GenerateError::Engine(e.to_string()))?;
        self.session = Some(s);
        Ok(())
    }

    fn serial_session_reset(&mut self) {
        self.session = None;
        self.session_disk_storable = false;
        self.thinking_visible = None;
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        let buf = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        self.session()?
            .sync(&buf)
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn sync_prompt(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
        disk_eligible: bool,
        thinking_visible_eligible: bool,
    ) -> Result<i32, GenerateError> {
        self.sync_prompt_inner(
            prompt,
            tokens,
            disk_eligible,
            thinking_visible_eligible,
            false,
        )
    }

    fn prompt_sync_elapsed(&self) -> Option<Duration> {
        self.prompt_sync_elapsed
    }

    fn restore_tool_replay(&mut self, messages: &mut [ChatMsg]) {
        let Ok(model_id) = u8::try_from(self.model.model_id()) else {
            return;
        };
        if let Some(store) = &self.store {
            self.tool_memory.restore_store(store, model_id, messages);
        }
        self.tool_memory.attach(messages);
    }

    fn sync_tool_replay_prompt(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
    ) -> Result<i32, GenerateError> {
        self.sync_prompt_inner(prompt, tokens, true, false, true)
    }

    fn remember_tool_replay(&mut self, calls: &[ToolCall], raw_dsml: &str) {
        if self.tool_memory.remember(calls, raw_dsml) > 0 && self.store.is_some() {
            self.session_disk_storable = true;
        }
    }

    fn maybe_store_continued(&mut self) -> Result<(), GenerateError> {
        if !self.session_disk_storable {
            return Ok(());
        }
        let Some(identity) = kv_identity(
            self.model.model_id(),
            self.model.routed_quant_bits(),
            self.ctx(),
        ) else {
            return Ok(());
        };
        let vocab = self.vocab.unwrap_or_else(|| self.model.vocab());
        let (Some(session), Some(store), tool_memory) =
            (&mut self.session, &mut self.store, &self.tool_memory)
        else {
            return Ok(());
        };
        let mut io = NativeSerialKvIo {
            session,
            vocab,
            tool_memory,
            prefill_checkpoints: false,
            sync_elapsed: Duration::ZERO,
        };
        try_store_continued(&mut io, store, identity)?;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), GenerateError> {
        let Some(identity) = kv_identity(
            self.model.model_id(),
            self.model.routed_quant_bits(),
            self.ctx(),
        ) else {
            return Ok(());
        };
        let vocab = self.vocab.unwrap_or_else(|| self.model.vocab());
        let (Some(session), Some(store), checkpoint, tool_memory) = (
            &mut self.session,
            &mut self.store,
            &self.thinking_visible,
            &self.tool_memory,
        ) else {
            return Ok(());
        };
        let mut io = NativeSerialKvIo {
            session,
            vocab,
            tool_memory,
            prefill_checkpoints: false,
            sync_elapsed: Duration::ZERO,
        };
        let (model_id, quant_bits, ctx) = identity;
        try_store_live(
            &mut io,
            store,
            model_id,
            quant_bits,
            ctx,
            KvReason::Shutdown,
            checkpoint.as_ref(),
        )?;
        Ok(())
    }

    fn eval(&mut self, token: i32) -> Result<(), GenerateError> {
        self.session()?
            .eval(token)
            .map(|_| ())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        match self.session() {
            Ok(s) => s.sample(temperature, top_k, top_p, min_p, rng),
            Err(_) => -1,
        }
    }

    fn pos(&self) -> i32 {
        self.session.as_ref().map(|s| s.pos()).unwrap_or(0)
    }

    fn ctx(&self) -> i32 {
        self.session.as_ref().map(|s| s.ctx()).unwrap_or(self.ctx)
    }

    fn generation(&self) -> u64 {
        self.session.as_ref().map(|s| s.generation()).unwrap_or(0)
    }

    fn session_tokens(&self) -> Vec<i32> {
        self.session
            .as_ref()
            .map(|s| s.host().tokens().to_vec())
            .unwrap_or_default()
    }

    fn remember_thinking_visible_checkpoint(&mut self, text: Vec<u8>) {
        self.thinking_visible = Some(ThinkingVisibleCheckpoint {
            text,
            frontier: self.pos(),
        });
    }

    fn invalidate(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.invalidate();
        }
    }
}

#[cfg(test)]
mod disk_sync_tests {
    use super::{
        continued_decode_allowed, discard_loaded, disk_sync_prompt, disk_sync_tool_replay,
        intermediate_prefill_eligible, ordinary_disk_cache_eligible,
        settle_thinking_visible_checkpoint, thinking_visible_cache_eligible,
        tool_replay_disk_cache_eligible, tool_replay_producer_eligible, try_store_continued,
        try_store_live, DiskSyncPolicy, GenerateError, SerialKvIo, ThinkingVisibleCheckpoint,
    };
    use crate::parse::{parse_request, ChatMsg, ParseEnv, ToolCall};
    use crate::render::{render_motif3_chat_ex, ModelSyntax};
    use crate::route::{Api, ThinkMode, WireSurface};
    use crate::stream::ChatFormat;
    use crate::tools::SemAccum;
    use ds4_kv::{
        read_envelope, Header, Options, Reason, Record, Store, EXT_THINKING_VISIBLE, EXT_TOOL_MAP,
    };
    use std::cell::Cell;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    struct FakeSerial {
        ctx: i32,
        live: Vec<i32>,
        rendered_live: Vec<u8>,
        trailer: Option<Vec<u8>>,
        suffix_tokens: Vec<i32>,
        loaded_tokens: Vec<i32>,
        fail_load: bool,
        fail_sync: bool,
        fail_sync_at: Option<usize>,
        fail_save: bool,
        fail_save_at: Option<usize>,
        save_calls: usize,
        progress_frontiers: Vec<usize>,
        invalidations: usize,
        syncs: Vec<Vec<i32>>,
        suffixes: Vec<Vec<u8>>,
        loads: Vec<(PathBuf, u64, u64)>,
        events: Vec<&'static str>,
        user_token_id: i32,
        assistant_token_id: i32,
        live_token_reads: Cell<usize>,
    }

    impl FakeSerial {
        fn new(live: &[i32], rendered_live: &[u8]) -> Self {
            Self {
                ctx: 4096,
                live: live.to_vec(),
                rendered_live: rendered_live.to_vec(),
                trailer: Some(Vec::new()),
                suffix_tokens: Vec::new(),
                loaded_tokens: Vec::new(),
                fail_load: false,
                fail_sync: false,
                fail_sync_at: None,
                fail_save: false,
                fail_save_at: None,
                save_calls: 0,
                progress_frontiers: Vec::new(),
                invalidations: 0,
                syncs: Vec::new(),
                suffixes: Vec::new(),
                loads: Vec::new(),
                events: Vec::new(),
                user_token_id: -1,
                assistant_token_id: -1,
                live_token_reads: Cell::new(0),
            }
        }
    }

    impl SerialKvIo for FakeSerial {
        fn ctx(&self) -> i32 {
            self.ctx
        }

        fn chat_token_ids(&self) -> (i32, i32) {
            (self.user_token_id, self.assistant_token_id)
        }

        fn live_len(&self) -> i32 {
            i32::try_from(self.live.len()).unwrap_or(0)
        }

        fn live_tokens(&self) -> Vec<i32> {
            self.live_token_reads.set(self.live_token_reads.get() + 1);
            self.live.clone()
        }

        fn render_tokens(&self, _tokens: &[i32]) -> Result<Vec<u8>, GenerateError> {
            Ok(self.rendered_live.clone())
        }

        fn checkpoint_trailer(&self, _text: &[u8]) -> Option<Vec<u8>> {
            self.trailer.clone()
        }

        fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError> {
            self.suffixes.push(suffix.to_vec());
            Ok(self.suffix_tokens.clone())
        }

        fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
            self.events.push("sync");
            self.syncs.push(tokens.to_vec());
            if self.fail_sync || self.fail_sync_at == Some(self.syncs.len()) {
                return Err(GenerateError::Engine("injected suffix sync failure".into()));
            }
            self.live = tokens.to_vec();
            Ok(())
        }

        fn sync_with_prefill_checkpoints(
            &mut self,
            tokens: &[i32],
            store: &mut Store,
            identity: (u8, u8, u32),
            cached_floor: i32,
        ) -> Result<(), GenerateError> {
            self.events.push("sync");
            self.syncs.push(tokens.to_vec());
            for frontier in self.progress_frontiers.clone() {
                if frontier <= tokens.len() {
                    self.live = tokens[..frontier].to_vec();
                    self.events.push("chunk");
                    if i32::try_from(frontier).unwrap_or(0) > cached_floor {
                        let _ = try_store_continued(self, store, identity);
                    }
                }
            }
            if self.fail_sync || self.fail_sync_at == Some(self.syncs.len()) {
                return Err(GenerateError::Engine("injected suffix sync failure".into()));
            }
            self.live = tokens.to_vec();
            Ok(())
        }

        fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError> {
            self.events.push("save");
            self.save_calls += 1;
            if self.fail_save || self.fail_save_at == Some(self.save_calls) {
                return Err(GenerateError::Engine(
                    "injected payload save failure".into(),
                ));
            }
            fs::write(path, b"current-payload").map_err(|e| GenerateError::Engine(e.to_string()))
        }

        fn load_payload_range(
            &mut self,
            path: &Path,
            offset: u64,
            length: u64,
        ) -> Result<(), GenerateError> {
            self.events.push("load");
            self.loads.push((path.to_path_buf(), offset, length));
            if self.fail_load {
                return Err(GenerateError::Engine(
                    "injected payload load failure".into(),
                ));
            }
            self.live = self.loaded_tokens.clone();
            Ok(())
        }

        fn invalidate(&mut self) {
            self.events.push("invalidate");
            self.invalidations += 1;
            self.live.clear();
        }
    }

    fn store(tag: &str) -> (PathBuf, Store) {
        let dir =
            std::env::temp_dir().join(format!("ds4-server-disk-sync-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let options = Options {
            min_tokens: 1,
            cold_max_tokens: 32,
            continued_interval_tokens: 8,
            boundary_trim_tokens: 0,
            boundary_align_tokens: 0,
        };
        let store = Store::open(&dir, 16, true, options).unwrap();
        (dir, store)
    }

    fn candidate(store: &mut Store, text: &[u8], tokens: u32) -> PathBuf {
        store
            .write(Record {
                header: Header {
                    quant_bits: 2,
                    reason: Reason::Evict,
                    ext_flags: 0,
                    model_id: 0,
                    tokens,
                    hits: 0,
                    ctx_size: 4096,
                    created_at: 1,
                    last_used: 1,
                    payload_bytes: 0,
                    text_bytes: 0,
                },
                text: text.to_vec(),
                payload: b"candidate-payload".to_vec(),
                trailer: Vec::new(),
            })
            .unwrap()
    }

    #[test]
    fn disk_sync_prompt_cold_miss_syncs_then_saves_full_prompt() {
        let (dir, mut store) = store("cold");
        let mut io = FakeSerial::new(&[], b"cold prompt");

        let cached = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"cold prompt",
            &[1, 2],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.syncs, [vec![1, 2]]);
        assert!(io.loads.is_empty());
        assert_eq!(io.events, ["sync", "save"]);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Cold);
        assert_eq!(store.entries()[0].header.tokens, 2);
        assert_eq!(store.entries()[0].header.ext_flags, 0);
        assert_eq!(store.continued_last_store_tokens, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_cold_miss_saves_stable_prefix_before_full_sync() {
        let (dir, mut store) = store("cold-prefix");
        store.opt.boundary_trim_tokens = 1;
        store.opt.boundary_align_tokens = 4;
        let mut io = FakeSerial::new(&[], b"stable prefix");
        let prompt_tokens: Vec<i32> = (1..=10).collect();

        let cached = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"cold prompt",
            &prompt_tokens,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.syncs, [prompt_tokens[..8].to_vec(), prompt_tokens]);
        assert_eq!(io.events, ["sync", "save", "sync"]);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Cold);
        assert_eq!(store.entries()[0].header.tokens, 8);
        assert_eq!(store.continued_last_store_tokens, 8);
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.text, b"stable prefix");
        assert_eq!(record.payload, b"current-payload");
        assert!(record.trailer.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_cold_miss_prefers_chat_anchor() {
        let (dir, mut store) = store("cold-anchor");
        store.opt.boundary_trim_tokens = 1;
        store.opt.boundary_align_tokens = 4;
        let mut io = FakeSerial::new(&[], b"chat anchor");
        io.user_token_id = 99;
        io.assistant_token_id = 100;
        let prompt_tokens = vec![10, 11, 99, 12, 13, 100, 14, 15, 16, 17];

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"cold prompt",
            &prompt_tokens,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt_tokens[..2].to_vec(), prompt_tokens]);
        assert_eq!(store.entries()[0].header.tokens, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_cold_policy_respects_disable_limit_and_lane_gate() {
        let prompt_tokens: Vec<i32> = (1..=10).collect();
        for (tag, min, cold_max, load) in [
            ("cold-below-min", 11, 32, true),
            ("cold-disabled", 1, 0, true),
            ("cold-above-max", 1, 4, true),
            ("cold-ineligible", 1, 32, false),
        ] {
            let (dir, mut store) = store(tag);
            store.opt.min_tokens = min;
            store.opt.cold_max_tokens = cold_max;
            let mut io = FakeSerial::new(&[], b"cold prompt");
            disk_sync_prompt(
                &mut io,
                Some(&mut store),
                0,
                2,
                b"cold prompt",
                &prompt_tokens,
                None,
                false,
                DiskSyncPolicy {
                    save_current: false,
                    load,
                },
            )
            .unwrap();
            assert_eq!(io.syncs, [prompt_tokens.clone()], "{tag}");
            assert_eq!(io.events, ["sync"], "{tag}");
            assert!(store.entries().is_empty(), "{tag}");
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn disk_sync_prompt_cold_save_is_nonfatal_but_prefix_sync_failure_is_fatal() {
        let prompt_tokens: Vec<i32> = (1..=10).collect();
        let (save_dir, mut save_store) = store("cold-save-fail");
        save_store.opt.boundary_trim_tokens = 1;
        save_store.opt.boundary_align_tokens = 4;
        let mut save_io = FakeSerial::new(&[], b"stable prefix");
        save_io.fail_save = true;
        disk_sync_prompt(
            &mut save_io,
            Some(&mut save_store),
            0,
            2,
            b"cold prompt",
            &prompt_tokens,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(
            save_io.syncs,
            [prompt_tokens[..8].to_vec(), prompt_tokens.clone()]
        );
        assert_eq!(save_io.events, ["sync", "save", "sync"]);
        assert!(save_store.entries().is_empty());
        assert_eq!(save_store.continued_last_store_tokens, 0);

        let (sync_dir, mut sync_store) = store("cold-sync-fail");
        sync_store.opt.boundary_trim_tokens = 1;
        sync_store.opt.boundary_align_tokens = 4;
        let mut sync_io = FakeSerial::new(&[], b"stable prefix");
        sync_io.fail_sync = true;
        assert!(disk_sync_prompt(
            &mut sync_io,
            Some(&mut sync_store),
            0,
            2,
            b"cold prompt",
            &prompt_tokens,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .is_err());
        assert_eq!(sync_io.syncs, [prompt_tokens[..8].to_vec()]);
        assert_eq!(sync_io.events, ["sync"]);
        assert!(sync_store.entries().is_empty());

        let (tail_dir, mut tail_store) = store("cold-tail-sync-fail");
        tail_store.opt.boundary_trim_tokens = 1;
        tail_store.opt.boundary_align_tokens = 4;
        let mut tail_io = FakeSerial::new(&[], b"stable prefix");
        tail_io.fail_sync_at = Some(2);
        assert!(disk_sync_prompt(
            &mut tail_io,
            Some(&mut tail_store),
            0,
            2,
            b"cold prompt",
            &prompt_tokens,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .is_err());
        assert_eq!(tail_io.events, ["sync", "save", "sync"]);
        assert_eq!(tail_store.entries().len(), 1);
        assert_eq!(tail_store.entries()[0].header.reason, Reason::Cold);
        assert_eq!(tail_store.continued_last_store_tokens, 8);
        let _ = fs::remove_dir_all(save_dir);
        let _ = fs::remove_dir_all(sync_dir);
        let _ = fs::remove_dir_all(tail_dir);
    }

    #[test]
    fn continued_store_requires_an_exact_frontier_and_advances_only_on_success() {
        let (dir, mut continued_store) = store("continued");
        let mut io = FakeSerial::new(&[1, 2, 3, 4, 5, 6, 7], b"continued");
        assert!(!try_store_continued(&mut io, &mut continued_store, (0, 2, 4096)).unwrap());
        assert!(io.events.is_empty());
        assert_eq!(io.live_token_reads.get(), 0);
        assert_eq!(continued_store.continued_last_store_tokens, 0);

        io.live.push(8);
        assert!(try_store_continued(&mut io, &mut continued_store, (0, 2, 4096)).unwrap());
        assert_eq!(io.events, ["save"]);
        assert_eq!(io.live_token_reads.get(), 1);
        assert_eq!(continued_store.continued_last_store_tokens, 8);
        assert_eq!(continued_store.entries().len(), 1);
        assert_eq!(
            continued_store.entries()[0].header.reason,
            Reason::Continued
        );
        assert_eq!(continued_store.entries()[0].header.tokens, 8);

        continued_store.continued_last_store_tokens = 0;
        io.events.clear();
        assert!(try_store_continued(&mut io, &mut continued_store, (0, 2, 4096)).unwrap());
        assert!(io.events.is_empty());
        assert_eq!(continued_store.continued_last_store_tokens, 8);

        let (fail_dir, mut fail_store) = store("continued-fail");
        let mut fail_io = FakeSerial::new(&[1, 2, 3, 4, 5, 6, 7, 8], b"continued-fail");
        fail_io.fail_save = true;
        assert!(try_store_continued(&mut fail_io, &mut fail_store, (0, 2, 4096)).is_err());
        assert_eq!(fail_io.events, ["save"]);
        assert_eq!(fail_store.continued_last_store_tokens, 0);
        assert!(fail_store.entries().is_empty());

        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(fail_dir);
    }

    #[test]
    fn intermediate_prefill_stores_due_frontier_from_one_sync() {
        let (dir, mut store) = store("intermediate-one-sync");
        store.opt.cold_max_tokens = 0;
        store.opt.continued_interval_tokens = 4;
        let prompt: Vec<i32> = (1..=6).collect();
        let mut io = FakeSerial::new(&[], b"prefix-four");
        io.progress_frontiers = vec![4];

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"full prompt",
            &prompt,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt]);
        assert_eq!(io.events, ["sync", "chunk", "save"]);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Continued);
        assert_eq!(store.entries()[0].header.tokens, 4);
        assert_eq!(store.entries()[0].header.ext_flags, 0);
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.text, b"prefix-four");
        assert_eq!(record.payload, b"current-payload");
        assert!(record.trailer.is_empty());
        assert_eq!(store.continued_last_store_tokens, 4);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn intermediate_prefill_ignores_frontiers_at_or_below_cached_floor() {
        let (dir, mut store) = store("intermediate-cached-floor");
        store.opt.cold_max_tokens = 0;
        store.opt.continued_interval_tokens = 4;
        let prompt: Vec<i32> = (1..=10).collect();
        let mut io = FakeSerial::new(&prompt[..6], b"prefix-eight");
        io.progress_frontiers = vec![4, 8];

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"full prompt",
            &prompt,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt]);
        assert_eq!(io.save_calls, 1);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Continued);
        assert_eq!(store.entries()[0].header.tokens, 8);
        assert_eq!(store.continued_last_store_tokens, 8);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cold_prefill_suppresses_duplicate_continued_at_same_frontier() {
        let (dir, mut store) = store("cold-suppresses-continued");
        store.opt.boundary_trim_tokens = 2;
        store.opt.continued_interval_tokens = 4;
        let prompt: Vec<i32> = (1..=6).collect();
        let mut io = FakeSerial::new(&[], b"prefix-four");
        io.progress_frontiers = vec![4];

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"full prompt",
            &prompt,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt[..4].to_vec(), prompt]);
        assert_eq!(io.save_calls, 1);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Cold);
        assert_eq!(store.entries()[0].header.tokens, 4);
        assert_eq!(store.continued_last_store_tokens, 4);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cold_save_failure_restores_continued_at_resumed_start() {
        let (dir, mut store) = store("cold-fail-restores-continued");
        store.opt.boundary_trim_tokens = 2;
        store.opt.continued_interval_tokens = 4;
        let prompt: Vec<i32> = (1..=6).collect();
        let mut io = FakeSerial::new(&[], b"prefix-four");
        io.progress_frontiers = vec![4];
        io.fail_save_at = Some(1);

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"full prompt",
            &prompt,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt[..4].to_vec(), prompt]);
        assert_eq!(io.save_calls, 2);
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Continued);
        assert_eq!(store.entries()[0].header.tokens, 4);
        assert_eq!(store.continued_last_store_tokens, 4);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn intermediate_save_failure_is_best_effort() {
        let (dir, mut store) = store("intermediate-save-fail");
        store.opt.cold_max_tokens = 0;
        store.opt.continued_interval_tokens = 4;
        let prompt: Vec<i32> = (1..=6).collect();
        let mut io = FakeSerial::new(&[], b"prefix-four");
        io.progress_frontiers = vec![4];
        io.fail_save = true;

        disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"full prompt",
            &prompt,
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(io.syncs, [prompt]);
        assert!(store.entries().is_empty());
        assert_eq!(store.continued_last_store_tokens, 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn intermediate_prefill_scope_is_cuda_deepseek_ordinary_disk_only() {
        assert!(intermediate_prefill_eligible(true, true, true, false));
        for case in [
            (false, true, true, false),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, true),
        ] {
            assert!(!intermediate_prefill_eligible(
                case.0, case.1, case.2, case.3,
            ));
        }
    }

    #[test]
    fn continued_decode_gate_stops_after_tool_syntax_begins() {
        let plain = SemAccum::init(true, false, false, ChatFormat::DeepSeek, b"");
        assert!(continued_decode_allowed(&plain));

        let mut tools = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
        assert!(continued_decode_allowed(&tools));
        tools.feed("<｜DSML｜tool_calls>\n".as_bytes(), &[]);
        assert!(tools.saw_tool_start);
        assert!(!continued_decode_allowed(&tools));

        let mut thinking = SemAccum::init(true, true, true, ChatFormat::DeepSeek, b"");
        thinking.feed("<｜DSML｜tool_calls>\n".as_bytes(), &[]);
        assert!(!thinking.saw_tool_start);
        assert!(thinking.dsml_state().is_tool());
        assert!(!continued_decode_allowed(&thinking));
    }

    fn chat_msg(role: &str, content: &str) -> ChatMsg {
        ChatMsg {
            role: role.into(),
            content: content.into(),
            ..ChatMsg::default()
        }
    }

    #[test]
    fn continued_prefix_hit_misses_motif_history_wrap() {
        let (dir, mut store) = store("continued-motif-prefix");
        let user = "summarize this";
        let live_decode = "The model serves five families.";
        let first =
            render_motif3_chat_ex(&[chat_msg("user", user)], "", &[], ThinkMode::None).unwrap();
        let mut live = first.clone();
        live.extend_from_slice(live_decode.as_bytes());

        store
            .write(Record {
                header: Header {
                    quant_bits: 2,
                    reason: Reason::Continued,
                    ext_flags: 0,
                    model_id: 0,
                    tokens: 8,
                    hits: 0,
                    ctx_size: 4096,
                    created_at: 1,
                    last_used: 1,
                    payload_bytes: 0,
                    text_bytes: 0,
                },
                text: live.clone(),
                payload: b"continued-payload".to_vec(),
                trailer: Vec::new(),
            })
            .unwrap();

        let mut prefix_follow = live.clone();
        prefix_follow.extend_from_slice(b"\nReply with exactly RESTORED_OK.");
        let (path, envelope) = store
            .text_prefix_candidate(&prefix_follow, 0, 2, 4096)
            .unwrap()
            .expect("continued live text is a prefix of a completions follow-up");
        assert_eq!(envelope.header.reason, Reason::Continued);
        assert_eq!(envelope.text, live);

        let mut hit = FakeSerial::new(&[], b"");
        hit.loaded_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        hit.suffix_tokens = vec![9];
        let cached = disk_sync_prompt(
            &mut hit,
            Some(&mut store),
            0,
            2,
            &prefix_follow,
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached, 8);
        assert_eq!(
            hit.suffixes,
            [b"\nReply with exactly RESTORED_OK.".to_vec()]
        );
        assert_eq!(hit.syncs, [vec![1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        assert_eq!(
            hit.loads,
            [(path, envelope.payload_offset, envelope.header.payload_bytes)]
        );

        let wrap = render_motif3_chat_ex(
            &[
                chat_msg("user", user),
                chat_msg("assistant", live_decode),
                chat_msg("user", "Reply with exactly RESTORED_OK."),
            ],
            "",
            &[],
            ThinkMode::None,
        )
        .unwrap();
        assert!(
            !wrap.starts_with(&live),
            "closed Motif assistant history is not live decode bytes"
        );
        assert!(
            live.windows(b"<think></think>".len())
                .any(|window| window == b"<think></think>"),
            "live frontier keeps the open-assistant think tags"
        );
        assert!(
            store
                .text_prefix_candidate(&wrap, 0, 2, 4096)
                .unwrap()
                .is_none(),
            "history wrap must miss the continued live-decode record"
        );

        let mut miss = FakeSerial::new(&[], b"");
        miss.loaded_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let cached = disk_sync_prompt(
            &mut miss,
            Some(&mut store),
            0,
            2,
            &wrap,
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert!(miss.loads.is_empty());
        assert_eq!(miss.syncs, [vec![90, 91]]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_reuses_current_exact_and_byte_prefixes() {
        let (dir, mut store) = store("current");
        let mut exact = FakeSerial::new(&[1, 2], b"unused");
        let cached = disk_sync_prompt(
            &mut exact,
            Some(&mut store),
            0,
            2,
            b"hello",
            &[1, 2, 3],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 2);
        assert_eq!(exact.syncs, [vec![1, 2, 3]]);
        assert!(exact.loads.is_empty());

        let mut byte_prefix = FakeSerial::new(&[9], b"hello ");
        byte_prefix.suffix_tokens = vec![10, 11];
        let cached = disk_sync_prompt(
            &mut byte_prefix,
            Some(&mut store),
            0,
            2,
            b"hello world",
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 1);
        assert_eq!(byte_prefix.suffixes, [b"world".to_vec()]);
        assert_eq!(byte_prefix.syncs, [vec![9, 10, 11]]);
        assert!(byte_prefix.loads.is_empty());

        let checkpoint = ThinkingVisibleCheckpoint {
            text: b"canonical assistant".to_vec(),
            frontier: 2,
        };
        let mut visible = FakeSerial::new(&[41, 42], b"generation <think></think> assistant");
        visible.suffix_tokens = vec![6, 43];
        let cached = disk_sync_prompt(
            &mut visible,
            None,
            0,
            2,
            b"canonical assistant<|endofturn|>next",
            &[90, 91],
            Some(&checkpoint),
            true,
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 2);
        assert_eq!(visible.suffixes, [b"<|endofturn|>next".to_vec()]);
        assert_eq!(visible.syncs, [vec![41, 42, 6, 43]]);

        let mut ineligible = FakeSerial::new(&[41, 42], b"not a prompt prefix");
        let cached = disk_sync_prompt(
            &mut ineligible,
            None,
            0,
            2,
            b"canonical assistant<|endofturn|>next",
            &[90, 91],
            Some(&checkpoint),
            false,
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert_eq!(ineligible.syncs, [vec![90, 91]]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_hit_loads_exact_tokens_retokenizes_suffix_and_touches_hit() {
        let (dir, mut store) = store("hit");
        let path = candidate(&mut store, b"hello ", 2);
        let envelope = read_envelope(&path).unwrap();
        let mut io = FakeSerial::new(&[7], b"old conversation");
        io.loaded_tokens = vec![41, 42];
        io.suffix_tokens = vec![43];

        let cached = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"hello world",
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: true,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 2);
        assert_eq!(io.suffixes, [b"world".to_vec()]);
        assert_eq!(io.syncs, [vec![41, 42, 43]]);
        assert_eq!(
            io.loads,
            [(
                path.clone(),
                envelope.payload_offset,
                envelope.header.payload_bytes
            )]
        );
        let save = io
            .events
            .iter()
            .position(|event| *event == "save")
            .expect("live checkpoint save event");
        let load = io
            .events
            .iter()
            .position(|event| *event == "load")
            .expect("candidate load event");
        assert!(
            save < load,
            "live checkpoint must be staged before a disk restore: {:?}",
            io.events
        );
        let (_, old) = store
            .text_prefix_candidate(b"old conversation", 0, 2, 4096)
            .unwrap()
            .expect("old conversation checkpoint");
        assert_eq!(old.header.reason, Reason::Evict);
        assert_eq!(store.read(&path).unwrap().header.hits, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_token_count_mismatch_discards_then_cold_stores() {
        let (dir, mut store) = store("count-mismatch");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"hello world");
        io.loaded_tokens = vec![41];
        io.suffix_tokens = vec![43];

        let cached = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"hello world",
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![90, 91]]);
        assert!(!path.exists());
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].header.reason, Reason::Cold);
        assert_eq!(store.entries()[0].header.tokens, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_load_failure_keeps_candidate_and_cold_syncs() {
        let (dir, mut store) = store("load-failure");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"");
        io.fail_load = true;

        let cached = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"hello world",
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![90, 91]]);
        assert!(path.exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_suffix_sync_failure_discards_and_returns_error() {
        let (dir, mut store) = store("sync-failure");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"");
        io.loaded_tokens = vec![41, 42];
        io.suffix_tokens = vec![43];
        io.fail_sync = true;

        let error = disk_sync_prompt(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"hello world",
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("injected suffix sync failure"));
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![41, 42, 43]]);
        assert!(!path.exists());
        assert!(store.entries().is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_saves_and_restores_thinking_visible_checkpoint() {
        let (dir, mut store) = store("thinking-visible");
        let checkpoint = ThinkingVisibleCheckpoint {
            text: b"canonical assistant".to_vec(),
            frontier: 2,
        };
        let mut saving = FakeSerial::new(&[41, 42], b"generation <think></think> assistant");
        saving.trailer = Some(b"must-not-mix".to_vec());

        disk_sync_prompt(
            &mut saving,
            Some(&mut store),
            0,
            2,
            b"unrelated prompt",
            &[90],
            Some(&checkpoint),
            false,
            DiskSyncPolicy {
                save_current: true,
                load: false,
            },
        )
        .unwrap();

        let prompt = b"canonical assistant<|endofturn|>next";
        let (path, envelope) = store
            .text_prefix_candidate(prompt, 0, 2, 4096)
            .unwrap()
            .expect("canonical visible checkpoint");
        assert_eq!(envelope.header.ext_flags, EXT_THINKING_VISIBLE);
        assert_eq!(envelope.trailer_bytes, 0);
        assert_eq!(envelope.header.tokens, 2);
        assert_eq!(envelope.text, checkpoint.text);

        let mut loading = FakeSerial::new(&[], b"");
        loading.loaded_tokens = vec![41, 42];
        loading.suffix_tokens = vec![6, 43];
        let cached = disk_sync_prompt(
            &mut loading,
            Some(&mut store),
            0,
            2,
            prompt,
            &[90, 91],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 2);
        assert_eq!(loading.suffixes, [b"<|endofturn|>next".to_vec()]);
        assert_eq!(loading.syncs, [vec![41, 42, 6, 43]]);
        assert_eq!(store.read(&path).unwrap().header.hits, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn live_store_writes_tool_map_trailer_only_for_plain_checkpoints() {
        let (dir, mut store) = store("tool-map-write");
        let mut io = FakeSerial::new(&[41, 42], b"plain sampled tool block");
        io.trailer = Some(b"KTM\x01\0\0\0\0".to_vec());

        try_store_live(&mut io, &mut store, 0, 2, 4096, Reason::Evict, None).unwrap();

        assert_eq!(store.entries().len(), 1);
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.header.ext_flags, EXT_TOOL_MAP);
        assert_eq!(Some(record.trailer), io.trailer);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn live_store_skips_checkpoint_when_tool_map_exceeds_its_bound() {
        let (dir, mut store) = store("tool-map-overflow");
        let mut io = FakeSerial::new(&[1, 2, 3, 4, 5, 6, 7, 8], b"plain sampled tool block");
        io.trailer = None;

        assert!(!try_store_live(&mut io, &mut store, 0, 2, 4096, Reason::Evict, None,).unwrap());
        assert!(!try_store_continued(&mut io, &mut store, (0, 2, 4096)).unwrap());

        assert!(store.entries().is_empty());
        assert_eq!(store.continued_last_store_tokens, 0);
        assert!(!io.events.contains(&"save"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tool_replay_sync_is_the_only_lane_that_loads_tool_map_records() {
        fn write_tool_candidate(store: &mut Store, ext_flags: u8, trailer: Vec<u8>) {
            store
                .write(Record {
                    header: Header {
                        quant_bits: 2,
                        reason: Reason::Evict,
                        ext_flags,
                        model_id: 0,
                        tokens: 2,
                        hits: 0,
                        ctx_size: 4096,
                        created_at: 1,
                        last_used: 1,
                        payload_bytes: 0,
                        text_bytes: 0,
                    },
                    text: b"tool prompt".to_vec(),
                    payload: b"payload".to_vec(),
                    trailer,
                })
                .unwrap();
        }

        let (ordinary_dir, mut ordinary_store) = store("tool-map-ordinary-reject");
        write_tool_candidate(
            &mut ordinary_store,
            EXT_TOOL_MAP,
            b"KTM\x01\0\0\0\0".to_vec(),
        );
        let mut ordinary = FakeSerial::new(&[], b"");
        ordinary.loaded_tokens = vec![41, 42];
        assert_eq!(
            disk_sync_prompt(
                &mut ordinary,
                Some(&mut ordinary_store),
                0,
                2,
                b"tool prompt tail",
                &[90, 91],
                None,
                false,
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap(),
            0
        );
        assert!(ordinary.loads.is_empty());

        let (tool_dir, mut tool_store) = store("tool-map-scoped-load");
        write_tool_candidate(&mut tool_store, EXT_TOOL_MAP, b"KTM\x01\0\0\0\0".to_vec());
        let mut tool = FakeSerial::new(&[], b"");
        tool.loaded_tokens = vec![41, 42];
        tool.suffix_tokens = vec![43];
        assert_eq!(
            disk_sync_tool_replay(
                &mut tool,
                Some(&mut tool_store),
                0,
                2,
                b"tool prompt tail",
                &[90, 91, 92],
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap(),
            2
        );
        assert_eq!(tool.loads.len(), 1);

        for (tag, flags, trailer) in [
            ("empty", EXT_TOOL_MAP, Vec::new()),
            (
                "combined",
                EXT_TOOL_MAP | EXT_THINKING_VISIBLE,
                b"KTM\x01\0\0\0\0".to_vec(),
            ),
            ("unknown", 1 << 7, b"KTM\x01\0\0\0\0".to_vec()),
        ] {
            let (bad_dir, mut bad_store) = store(&format!("tool-map-{tag}-reject"));
            write_tool_candidate(&mut bad_store, flags, trailer);
            let mut bad = FakeSerial::new(&[], b"");
            bad.loaded_tokens = vec![41, 42];
            assert_eq!(
                disk_sync_tool_replay(
                    &mut bad,
                    Some(&mut bad_store),
                    0,
                    2,
                    b"tool prompt tail",
                    &[90, 91],
                    DiskSyncPolicy {
                        save_current: false,
                        load: true,
                    },
                )
                .unwrap(),
                0
            );
            assert!(bad.loads.is_empty());
            let _ = fs::remove_dir_all(bad_dir);
        }

        let _ = fs::remove_dir_all(ordinary_dir);
        let _ = fs::remove_dir_all(tool_dir);
    }

    #[test]
    fn thinking_visible_checkpoint_clears_only_after_successful_sync() {
        let mut checkpoint = Some(ThinkingVisibleCheckpoint {
            text: b"visible".to_vec(),
            frontier: 2,
        });

        settle_thinking_visible_checkpoint(&mut checkpoint, false);
        assert_eq!(checkpoint.as_ref().unwrap().frontier, 2);

        settle_thinking_visible_checkpoint(&mut checkpoint, true);
        assert!(checkpoint.is_none());
    }

    #[test]
    fn discard_failure_still_resets_marker_and_invalidates_session() {
        let (dir, mut store) = store("discard-race");
        let mut io = FakeSerial::new(&[41, 42], b"hello");
        store.continued_last_store_tokens = 2;

        discard_loaded(&mut store, &mut io, &dir.join("missing.kvc"));

        assert_eq!(store.continued_last_store_tokens, 0);
        assert_eq!(io.invalidations, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_cache_eligibility_excludes_protocol_state() {
        let env = ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 16,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
        };
        let ordinary = parse_request(
            WireSurface::OpenaiChat,
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
        )
        .unwrap();
        assert!(ordinary_disk_cache_eligible(&ordinary));
        assert!(thinking_visible_cache_eligible(&ordinary));

        let mut anthropic = ordinary.clone();
        anthropic.api = Api::Anthropic;
        assert!(thinking_visible_cache_eligible(&anthropic));

        let mut completion = ordinary.clone();
        completion.kind = crate::route::ReqKind::Completion;
        assert!(!thinking_visible_cache_eligible(&completion));

        let mut responses = ordinary.clone();
        responses.api = Api::Responses;
        assert!(!thinking_visible_cache_eligible(&responses));

        let mut cases = Vec::new();
        let mut request = ordinary.clone();
        request.api = Api::Anthropic;
        cases.push(request);
        let mut request = ordinary.clone();
        request.api = Api::Responses;
        cases.push(request);
        let mut request = ordinary.clone();
        request.think_mode = ThinkMode::Low;
        cases.push(request);
        let mut request = ordinary.clone();
        request.has_tools = true;
        cases.push(request);
        let mut request = ordinary.clone();
        request.has_tool_results = true;
        cases.push(request);
        let mut request = ordinary;
        request.live_call_ids.push("call_1".into());
        cases.push(request);

        assert!(cases
            .iter()
            .all(|request| !ordinary_disk_cache_eligible(request)));

        let mut replay = cases[4].clone();
        replay.api = Api::Openai;
        replay.think_mode = ThinkMode::None;
        replay.live_call_ids.clear();
        replay.has_tools = true;
        replay.has_tool_results = false;
        replay.messages = vec![ChatMsg {
            role: "assistant".into(),
            calls: vec![ToolCall {
                id: "call_history".into(),
                ..Default::default()
            }],
            ..Default::default()
        }];
        assert!(tool_replay_disk_cache_eligible(
            &replay,
            ModelSyntax::DeepSeek
        ));
        assert!(tool_replay_disk_cache_eligible(
            &replay,
            ModelSyntax::SolarOpen2
        ));
        assert!(tool_replay_producer_eligible(
            &replay,
            ModelSyntax::DeepSeek
        ));
        for syntax in [ModelSyntax::Motif3, ModelSyntax::Exaone, ModelSyntax::Dots3] {
            assert!(!tool_replay_disk_cache_eligible(&replay, syntax));
        }
        for mutate in 0..4 {
            let mut rejected = replay.clone();
            match mutate {
                0 => rejected.api = Api::Anthropic,
                1 => rejected.think_mode = ThinkMode::Low,
                2 => rejected.live_call_ids.push("call_live".into()),
                _ => rejected.kind = crate::route::ReqKind::Completion,
            }
            assert!(!tool_replay_disk_cache_eligible(
                &rejected,
                ModelSyntax::DeepSeek
            ));
            assert!(!tool_replay_producer_eligible(
                &rejected,
                ModelSyntax::DeepSeek
            ));
        }
    }
}
