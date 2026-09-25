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
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(any(feature = "native", test))]
use std::time::{SystemTime, UNIX_EPOCH};

use ds4_core::{ReuseMiss, ReuseTaken};
use ds4_kv::PrefixAnswer;
use ds4_kv::Store as KvStore;
#[cfg(any(feature = "native", test))]
use ds4_kv::{
    chat_anchor_pos as kv_chat_anchor_pos, continued_store_target_from_host,
    store_len as kv_store_len, Header as KvHeader, HostKvView, Reason as KvReason,
    EXT_THINKING_VISIBLE, EXT_TOOL_MAP,
};

use crate::dsml::{SampleOverride, SamplePolicy};
use crate::parse::{ChatMsg, ChatPart, EosPolicy, ParsedRequest, ToolCall, ToolChoice};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use crate::render::{
    render_chat_choice, syntax_for_model_id, tool_start_marker, ModelSyntax, RenderError, DSML_EOS,
    QWEN_IM_END, SOLAR_IM_END,
};
use crate::retry::{
    build_recovery_suffix, parse_failure_should_retry, terminal_finish, truncation_outcome,
    TruncationOutcome,
};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, responses_final_response, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init, sse_chunk,
    sse_done, sse_headers, stream_error, stream_heartbeat_if_due, think_end, think_start,
    AnthropicStream, ChatFormat, OpenaiStream, ReqTimings, ResponsesStream, StreamReq, ThinkBlock,
    Writer,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionProbe {
    pub token_count: u32,
}

#[derive(Clone, Debug)]
pub struct VisionPromptInput {
    pub data: Arc<[u8]>,
    pub token_offset: u32,
    /// Packed MiMo frames from the prepare pass. Empty for still images
    /// and for a video the sync path still has to decode itself.
    pub frames: Vec<ds4_core::PackedVisual>,
}

#[derive(Clone, Debug)]
pub struct AudioPromptInput {
    pub data: Arc<[u8]>,
    pub token_offset: u32,
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
    fn template(&self) -> Option<&ds4_core::chat_template::Template> {
        None
    }
    fn render_request(&self, parsed: &ParsedRequest) -> Result<Vec<u8>, GenerateError> {
        if let Some(template) = self.template() {
            return crate::chat_input::render(template, self.model_id(), parsed);
        }
        render_prompt(parsed, self.model_id())
    }
    fn restore_chat(&self, _parsed: &mut ParsedRequest) -> Result<(), GenerateError> {
        Ok(())
    }
    fn remember_chat(
        &mut self,
        _parsed: &ParsedRequest,
        _generated: &crate::tools::ParsedGenerated,
    ) {
    }
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
    fn eos_id(&self) -> i32 {
        -1
    }
    fn eot_id(&self) -> i32 {
        -1
    }
    fn vision_probe(&self, _data: &[u8]) -> Result<VisionProbe, GenerateError> {
        Err(GenerateError::Unsupported("vision encoder is not loaded"))
    }
    fn vision_tokens(&self, data: &[u8]) -> Result<Vec<i32>, GenerateError> {
        let marker = match syntax_for_model_id(self.model_id()) {
            ModelSyntax::Glm53 => 154854,
            ModelSyntax::Inkling => 200054,
            _ => {
                return Err(GenerateError::Unsupported(
                    "image token format is unavailable",
                ))
            }
        };
        Ok(vec![marker; self.vision_probe(data)?.token_count as usize])
    }
    fn sync_vision_prompt(
        &mut self,
        _tokens: &[i32],
        _images: &[VisionPromptInput],
    ) -> Result<(), GenerateError> {
        Err(GenerateError::Unsupported("vision encoder is not loaded"))
    }
    fn audio_probe(&self, _data: &[u8]) -> Result<u32, GenerateError> {
        Err(GenerateError::Unsupported("audio encoder is not loaded"))
    }
    fn sync_media_prompt(
        &mut self,
        tokens: &[i32],
        images: &[VisionPromptInput],
        audios: &[AudioPromptInput],
    ) -> Result<(), GenerateError> {
        if audios.is_empty() {
            return self.sync_vision_prompt(tokens, images);
        }
        Err(GenerateError::Unsupported("audio encoder is not loaded"))
    }
    fn sync_mimo_prompt(
        &mut self,
        tokens: &[i32],
        images: &[VisionPromptInput],
        audios: &[AudioPromptInput],
        videos: &[VisionPromptInput],
    ) -> Result<(), GenerateError> {
        if !videos.is_empty() {
            return Err(GenerateError::Unsupported("video input requires MiMo"));
        }
        self.sync_media_prompt(tokens, images, audios)
    }
    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError>;
    /// Start this request's reuse trace. A corrective retry re-syncs inside
    /// the same request, so the sync itself must not erase what the first
    /// one reported.
    fn begin_trace(&mut self) {}
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
    /// Commit a nonempty greedy prefix bounded by the positive budget.
    /// Implementations may include EOS as the final accepted token.
    fn eval_greedy(&mut self, first: i32, _budget: i32) -> Result<Vec<i32>, GenerateError> {
        self.eval(first)?;
        Ok(vec![first])
    }
    fn last_eval_speculated(&self) -> bool {
        false
    }
    /// The mechanism the last prompt sync used, for the request trace.
    fn last_reuse(&self) -> ReuseTaken {
        ReuseTaken::Cold
    }
    /// Why the last prompt sync refused a candidate, for the request trace.
    fn last_miss(&self) -> ReuseMiss {
        ReuseMiss::None
    }
    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32;
    fn sample_excluding(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
        excluded_id: i32,
    ) -> i32 {
        if excluded_id >= 0 {
            return -1;
        }
        self.sample(temperature, top_k, top_p, min_p, rng)
    }
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
    /// The native plan may replay part or all of a matching token prefix.
    fn sync_start(&self, _tokens: &[i32]) -> i32 {
        self.live_len()
    }
    fn render_tokens(&self, tokens: &[i32]) -> Result<Vec<u8>, GenerateError>;
    fn checkpoint_trailer(&self, _text: &[u8]) -> Option<Vec<u8>> {
        Some(Vec::new())
    }
    fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError>;
    /// Record which mechanism produced the reuse. Cached/computed counts
    /// cannot tell an appended frontier turn from a checkpoint replay.
    fn note_reuse(&mut self, _taken: ReuseTaken) {}
    /// Record why a candidate was refused, so the trace can say it. The
    /// first reason wins: a later, broader one would mask it.
    fn note_miss(&mut self, _miss: ReuseMiss) {}
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
    io.note_reuse(ReuseTaken::Cold);
    io.sync(tokens)?;
    Ok(0)
}

#[cfg(any(feature = "native", test))]
fn record_prefix_reuse(io: &mut impl SerialKvIo, tokens: &[i32], prefix: usize) -> i32 {
    let prefix = i32::try_from(prefix.min(tokens.len())).unwrap_or(0);
    let cached = io.sync_start(tokens).clamp(0, prefix);
    if cached < prefix {
        io.note_miss(ReuseMiss::StateReplay);
    }
    io.note_reuse(if cached == 0 {
        ReuseTaken::Cold
    } else if cached < prefix {
        ReuseTaken::Partial
    } else {
        ReuseTaken::Exact
    });
    cached
}

#[cfg(any(feature = "native", test))]
fn cold_sync_and_store(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    identity: (u8, u8, u32),
    tokens: &[i32],
    prefill_checkpoints: bool,
) -> Result<i32, GenerateError> {
    io.note_reuse(ReuseTaken::Cold);
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
    if full_len < store.opt.min_tokens {
        io.note_miss(ReuseMiss::BelowThreshold);
    }
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
fn disk_sync_template(
    io: &mut impl SerialKvIo,
    mut store: Option<&mut KvStore>,
    model_id: i32,
    quant_bits: i32,
    prompt: &[u8],
    tokens: &[i32],
    policy: DiskSyncPolicy,
) -> Result<i32, GenerateError> {
    let history = policy
        .load
        .then(|| history_frontier(model_id, prompt, tokens, |text| io.tokenize_suffix(text)))
        .flatten()
        .filter(|(_, prefix)| {
            store
                .as_ref()
                .is_some_and(|store| prefix.len() >= store.opt.min_tokens.max(1) as usize)
        });
    if let Some((text, prefix)) = history {
        // Keep an already deeper, matching live frontier. Otherwise sync to
        // the history boundary before the generation-only think pair: both
        // the saved token ledger and the native KV now describe that prefix.
        let live = io.live_tokens();
        if live.len() <= prefix.len() || !tokens.starts_with(&live) {
            let cached = disk_sync_prompt_impl(
                io,
                store.as_deref_mut(),
                model_id,
                quant_bits,
                &text,
                &prefix,
                None,
                false,
                policy,
                false,
                PromptReuse::Tokens,
            )?;
            if let (Some(store), Some((model_id, quant_bits, ctx)), Some(trailer)) = (
                store,
                kv_identity(model_id, quant_bits, io.ctx()),
                io.checkpoint_trailer(&text),
            ) {
                let header = kv_header(model_id, quant_bits, ctx, prefix.len() as u32);
                if let Err(error) =
                    write_checkpoint(store, header, &text, &trailer, |path| io.save_payload(path))
                {
                    eprintln!("ds4-server-rs: history KV checkpoint failed: {error}");
                }
            }
            io.sync(tokens)?;
            return Ok(cached);
        }
    }
    // Rendered-text identity is not token identity: an official template
    // that drops a block still held in KV would continue from a sequence
    // the client never sent. Jinja families reuse on tokens only.
    disk_sync_prompt_impl(
        io,
        store,
        model_id,
        quant_bits,
        prompt,
        tokens,
        None,
        false,
        policy,
        false,
        PromptReuse::Tokens,
    )
}

#[cfg(any(feature = "native", test))]
pub(crate) fn history_frontier(
    model_id: i32,
    prompt: &[u8],
    tokens: &[i32],
    encode: impl FnOnce(&[u8]) -> Result<Vec<i32>, GenerateError>,
) -> Option<(Vec<u8>, Vec<i32>)> {
    let (suffix, boundary): (&[u8], &[u8]) = match syntax_for_model_id(model_id) {
        ModelSyntax::Step37 => (b"assistant\n<think>\n</think>\n", b"<|im_start|>"),
        ModelSyntax::Motif3 => (b"<think></think>", b"<|startofturn|><|assistant|>"),
        _ => return None,
    };
    let text = prompt.strip_suffix(suffix)?;
    if !text.ends_with(boundary) {
        return None;
    }
    let prefix = encode(text).ok()?;
    // Both official templates omit the empty think pair on history replay.
    // Stop at a control token: content whitespace may be trimmed (Motif) or
    // merge with the role newline (Step). Validate the full token prefix too.
    if prefix.is_empty() || prefix.len() >= tokens.len() || !tokens.starts_with(&prefix) {
        return None;
    }
    Some((text.to_vec(), prefix))
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PromptReuse {
    Tokens,
    LegacyText,
    Off,
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
        PromptReuse::LegacyText,
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
        PromptReuse::LegacyText,
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
    reuse: PromptReuse,
) -> Result<i32, GenerateError> {
    let identity = kv_identity(model_id, quant_bits, io.ctx());
    let prefill_checkpoints = policy.load && !allow_tool_map;
    if reuse == PromptReuse::Off {
        if policy.save_current {
            if let (Some(store), Some((model_id, quant_bits, ctx))) =
                (store.as_deref_mut(), identity)
            {
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
        }
        return cold_sync(io, canonical_tokens);
    }
    let live = io.live_tokens();
    if !live.is_empty() && canonical_tokens.starts_with(&live) {
        let cached = record_prefix_reuse(io, canonical_tokens, live.len());
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
    if reuse == PromptReuse::LegacyText && thinking_visible_eligible {
        if let Some(checkpoint) = checkpoint {
            if !live.is_empty()
                && usize::try_from(checkpoint.frontier).ok() == Some(live.len())
                && checkpoint.text.len() < prompt.len()
                && prompt.starts_with(&checkpoint.text)
            {
                let prefix = live.len();
                let mut effective = live;
                effective.extend(io.tokenize_suffix(&prompt[checkpoint.text.len()..])?);
                let cached = record_prefix_reuse(io, &effective, prefix);
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
    if reuse == PromptReuse::LegacyText && !live.is_empty() {
        let rendered = io.render_tokens(&live)?;
        if prompt.starts_with(&rendered) {
            let prefix = live.len();
            let mut effective = live;
            effective.extend(io.tokenize_suffix(&prompt[rendered.len()..])?);
            let cached = record_prefix_reuse(io, &effective, prefix);
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

    // A live session that holds this conversation, whose render moved: the
    // text still leads here, the token sequence no longer does. A session
    // about a different conversation is not that, and says nothing.
    if !live.is_empty() && prompt.starts_with(&io.render_tokens(&live)?) {
        io.note_miss(ReuseMiss::RenderedPrefix);
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
            // A record whose envelope will not read is refused by its
            // payload as surely as one whose tokens disagree.
            io.note_miss(ReuseMiss::PayloadMismatch);
            return cold_sync_and_store(
                io,
                store,
                (model_id, quant_bits, ctx),
                canonical_tokens,
                prefill_checkpoints,
            );
        }
    };
    let Some((path, envelope)) = candidate else {
        // Order matters: a record that exists but cannot be used, then a
        // conversation too short to have been stored, then plain absence.
        let short =
            i32::try_from(canonical_tokens.len()).unwrap_or(i32::MAX) < store.opt.min_tokens;
        io.note_miss(
            match store.prefix_answer(prompt, model_id, quant_bits, ctx) {
                PrefixAnswer::Mismatch => ReuseMiss::PayloadMismatch,
                // A record written before the minimum was raised is skipped
                // by every search since, whatever this prompt's length.
                PrefixAnswer::Shallow => ReuseMiss::BelowThreshold,
                _ if short => ReuseMiss::BelowThreshold,
                _ => ReuseMiss::NoCheckpoint,
            },
        );
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
        io.note_miss(ReuseMiss::PayloadMismatch);
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    }
    if store
        .open_payload(&path, &envelope)
        .map_err(|error| GenerateError::Engine(error.to_string()))
        .and_then(|payload| {
            io.load_payload_range(
                payload.path(),
                envelope.payload_offset,
                envelope.header.payload_bytes,
            )
        })
        .is_err()
    {
        // The candidate was chosen and then could not be read. That is a
        // refusal by its payload, not an absence.
        io.note_miss(ReuseMiss::PayloadMismatch);
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
        io.note_miss(ReuseMiss::PayloadMismatch);
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
    if reuse == PromptReuse::Tokens && !canonical_tokens.starts_with(&loaded) {
        // The template re-rendered this conversation differently, so the
        // stored tokens are no longer a prefix of the prompt.
        io.note_miss(ReuseMiss::RenderedPrefix);
        io.invalidate();
        return cold_sync_and_store(
            io,
            store,
            (model_id, quant_bits, ctx),
            canonical_tokens,
            prefill_checkpoints,
        );
    }
    let prefix = loaded.len();
    store.continued_last_store_tokens = prefix as i32;
    // Token identity permits restore, but native state may require replay
    // below that frontier (e.g. an unaligned Dots3 MTP append).
    if reuse == PromptReuse::Tokens {
        let cached = record_prefix_reuse(io, canonical_tokens, prefix);
        if cached > 0 {
            let _ = store.touch_hit(&path);
        }
        sync_maybe_checkpoint(
            io,
            canonical_tokens,
            Some(store),
            identity,
            cached,
            prefill_checkpoints,
        )?;
        return Ok(cached);
    }
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
    let cached = record_prefix_reuse(io, &effective, prefix);
    if cached > 0 {
        let _ = store.touch_hit(&path);
    }
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
    pub timings: ReqTimings,
    pub lane: Option<&'static str>,
    pub speculation_active: bool,
    pub reuse: ReuseTaken,
    pub reuse_miss: ReuseMiss,
    pub fallback_reason: Option<String>,
}

pub fn generation_blocked(parsed: &ParsedRequest, model_id: i32) -> Option<&'static str> {
    let syntax = syntax_for_model_id(model_id);
    if !parsed.videos.is_empty() && syntax != ModelSyntax::Mimo2 {
        return Some("video input requires MiMo");
    }
    if !parsed.audios.is_empty() && !matches!(syntax, ModelSyntax::Inkling | ModelSyntax::Mimo2) {
        return Some("audio input requires Inkling or MiMo");
    }
    if parsed.images.is_empty() {
        None
    } else {
        match syntax {
            ModelSyntax::Glm53
            | ModelSyntax::Inkling
            | ModelSyntax::Step37
            | ModelSyntax::Ling3Vl
            | ModelSyntax::Mimo2 => None,
            ModelSyntax::Qwen4Exp => Some("image input requires continuous runtime"),
            _ => Some(
                "image input is supported only by Qwen4Exp, GLM-5.3, Inkling, Step, Ling or MiMo",
            ),
        }
    }
}

pub fn chat_format_for_syntax(syntax: ModelSyntax) -> ChatFormat {
    match syntax {
        ModelSyntax::SolarOpen2 => ChatFormat::SolarOpen2,
        ModelSyntax::Exaone => ChatFormat::Exaone,
        // Only the generated thinking/tool envelope is shared with Qwen.
        ModelSyntax::Qwen4Exp | ModelSyntax::Step37 | ModelSyntax::Mimo2 => ChatFormat::Qwen4Exp,
        ModelSyntax::K2Horizon => ChatFormat::K2Horizon,
        ModelSyntax::Inkling => ChatFormat::Inkling,
        // Ling shares GLM's thinking and tool-call XML.
        ModelSyntax::DeepSeek
        | ModelSyntax::Motif3
        | ModelSyntax::Dots3
        | ModelSyntax::Glm53
        | ModelSyntax::Ling3Vl => ChatFormat::DeepSeek,
    }
}

pub fn stream_req_from_parsed(parsed: &ParsedRequest, model_id: i32) -> StreamReq {
    let syntax = syntax_for_model_id(model_id);
    StreamReq {
        kind: parsed.kind,
        api: parsed.api,
        model: parsed.model.clone(),
        think_mode: parsed.think_mode,
        has_tools: parsed.has_tools,
        stream: parsed.stream,
        stream_include_usage: parsed.stream_include_usage,
        reasoning_summary_emit: parsed.reasoning_summary_emit,
        chat_format: chat_format_for_syntax(syntax),
        syntax,
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

pub(crate) fn thinking_visible_key(
    prompt: &[u8],
    content: &[u8],
    syntax: ModelSyntax,
    format: ChatFormat,
    terminal: bool,
) -> Option<Vec<u8>> {
    if matches!(syntax, ModelSyntax::Step37 | ModelSyntax::Ling3Vl) {
        // Removing reasoning changes these families' history grammar. Re-render
        // the structured history with Jinja instead of inventing a prefix.
        return None;
    }
    let mut visible = if format == ChatFormat::K2Horizon {
        if !prompt.ends_with(b"<ifm|think>\n") {
            return None;
        }
        let content = content.trim_ascii();
        let mut visible = Vec::with_capacity(prompt.len() + 16 + content.len());
        visible.extend_from_slice(prompt);
        visible.extend_from_slice(b"</ifm|think>");
        visible.extend_from_slice(content);
        visible
    } else if format == ChatFormat::Qwen4Exp || syntax == ModelSyntax::Exaone {
        if !prompt.ends_with(b"<think>\n") {
            return None;
        }
        let content = content.trim_ascii();
        let mut visible = Vec::with_capacity(prompt.len() + 12 + content.len());
        visible.extend_from_slice(prompt);
        visible.extend_from_slice(b"\n</think>\n\n");
        visible.extend_from_slice(content);
        visible
    } else {
        let start = think_start(format).as_bytes();
        if !prompt.ends_with(start) {
            return None;
        }
        let prefix = if format == ChatFormat::SolarOpen2 {
            prompt
        } else {
            &prompt[..prompt.len() - start.len()]
        };
        let mut visible =
            Vec::with_capacity(prefix.len() + think_end(format).len() + content.len());
        visible.extend_from_slice(prefix);
        visible.extend_from_slice(think_end(format).as_bytes());
        visible.extend_from_slice(content);
        visible
    };
    if terminal {
        match syntax {
            ModelSyntax::Qwen4Exp => visible.extend_from_slice(QWEN_IM_END.as_bytes()),
            ModelSyntax::K2Horizon => {
                visible.extend_from_slice(crate::render::K2_IM_END.as_bytes())
            }
            ModelSyntax::Exaone => visible.extend_from_slice(b"<|endofturn|>"),
            ModelSyntax::Motif3 => visible.extend_from_slice(b"<|endofturn|>"),
            ModelSyntax::SolarOpen2 => visible.extend_from_slice(SOLAR_IM_END.as_bytes()),
            ModelSyntax::Glm53 => {}
            _ => visible.extend_from_slice(DSML_EOS.as_bytes()),
        }
        if matches!(
            syntax,
            ModelSyntax::Qwen4Exp | ModelSyntax::Exaone | ModelSyntax::SolarOpen2
        ) {
            visible.push(b'\n');
        }
    }
    Some(visible)
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

pub(crate) fn prepare_required_prefixes(
    parsed: &mut ParsedRequest,
    syntax: ModelSyntax,
    tokenize: impl Fn(&[u8]) -> Result<Vec<i32>, GenerateError>,
) -> Result<(), GenerateError> {
    let mimo_auto = syntax == ModelSyntax::Mimo2
        && parsed.has_tools
        && parsed.tool_choice == ToolChoice::Auto
        && !think_mode_enabled(parsed.think_mode);
    if parsed.tool_choice != ToolChoice::Required && !parsed.has_tool_results && !mimo_auto {
        return Ok(());
    }
    let format = chat_format_for_syntax(syntax);
    if !mimo_auto && parsed.required_think_end_prefix.is_empty() {
        let toks = tokenize(think_end(format).as_bytes())?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize thinking control prefix".into(),
            ));
        }
        parsed.required_think_end_prefix = toks;
    }
    if (parsed.tool_choice == ToolChoice::Required || mimo_auto)
        && parsed.required_tool_prefix.is_empty()
    {
        // MiMo can continue a bare tool marker as prose without thinking.
        let prefix = if mimo_auto {
            b"<function=".as_slice()
        } else if syntax == ModelSyntax::Mimo2 && !think_mode_enabled(parsed.think_mode) {
            b"<tool_call><function=".as_slice()
        } else {
            tool_start_marker(syntax).as_bytes()
        };
        let toks = tokenize(prefix)?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize required tool control prefix".into(),
            ));
        }
        parsed.required_tool_prefix = toks;
    }
    Ok(())
}

#[cfg(test)]
mod required_prefix_tests {
    use super::prepare_required_prefixes;
    use crate::parse::{parse_chat_request, ParseEnv};
    use crate::render::{ModelSyntax, GLM_TOOL_CALL_START};
    use crate::tools::DSML_TOOL_CALLS_START;

    #[test]
    fn ling_required_tool_prefix_is_glm_marker() {
        let mut parsed = parse_chat_request(
            &ParseEnv::default(),
            r#"{"messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"tool_choice":"required"}"#,
        )
        .unwrap();
        prepare_required_prefixes(&mut parsed, ModelSyntax::Ling3Vl, |literal| {
            Ok(if literal == GLM_TOOL_CALL_START.as_bytes() {
                vec![11]
            } else if literal == DSML_TOOL_CALLS_START.as_bytes() {
                vec![99]
            } else {
                vec![1]
            })
        })
        .unwrap();
        assert_eq!(parsed.required_tool_prefix, [11]);
    }

    #[test]
    fn mimo_no_think_required_starts_function() {
        let mut parsed = parse_chat_request(
            &ParseEnv::default(),
            r#"{"messages":[{"role":"user","content":"weather"}],"reasoning_effort":"none","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"tool_choice":"required"}"#,
        )
        .unwrap();
        prepare_required_prefixes(&mut parsed, ModelSyntax::Mimo2, |literal| {
            Ok(if literal == b"<tool_call><function=" {
                vec![17]
            } else {
                vec![99]
            })
        })
        .unwrap();
        assert_eq!(parsed.required_tool_prefix, [17]);
    }

    #[test]
    fn mimo_no_think_auto_prepares_function() {
        let mut parsed = parse_chat_request(
            &ParseEnv::default(),
            r#"{"messages":[{"role":"user","content":"weather"}],"reasoning_effort":"none","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}"#,
        )
        .unwrap();
        prepare_required_prefixes(&mut parsed, ModelSyntax::Mimo2, |literal| {
            Ok(if literal == b"<function=" {
                vec![17]
            } else {
                vec![99]
            })
        })
        .unwrap();
        assert_eq!(parsed.required_tool_prefix, [17]);
    }
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

fn retry_chat(
    engine: &mut dyn DecodeIo,
    parsed: &mut ParsedRequest,
    prompt: &mut Vec<u8>,
    acc: &SemAccum,
    detail: &str,
) -> Result<(), GenerateError> {
    if engine.template().is_none() {
        let format = chat_format_for_syntax(syntax_for_model_id(engine.model_id()));
        return append_recovery_suffix(
            engine,
            &build_recovery_suffix(format, parsed.think_mode, prompt, acc, detail),
        )
        .map(|_| ());
    }

    // Discard the invalid generation. Re-render the valid conversation with a
    // correction, so recovery never invents a model's role or tool delimiters.
    let mut retry = parsed.clone();
    let correction = format!("\n\nTool error: {detail}. The previous attempt was not executed. Emit a valid tool call matching the provided schema, or answer normally if no tool is needed.");
    if let Some(last) = retry.messages.last_mut().filter(|m| m.role == "user") {
        last.content.push_str(&correction);
        if !last.parts.is_empty() {
            last.parts.push(crate::parse::ChatPart::Text(correction));
        }
    } else {
        retry.messages.push(crate::parse::ChatMsg {
            role: "user".into(),
            content: correction,
            ..Default::default()
        });
    }
    let rendered = engine.render_request(&retry)?;
    let tokens = engine.tokenize_rendered_chat(&rendered)?;
    let media = prepare_media(engine, &retry, tokens)?;
    engine.invalidate();
    if media.vision.is_empty() && media.audios.is_empty() && media.videos.is_empty() {
        engine.sync(&media.tokens)?;
    } else {
        engine.sync_mimo_prompt(&media.tokens, &media.vision, &media.audios, &media.videos)?;
    }
    *parsed = retry;
    *prompt = rendered;
    Ok(())
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
    speculation: &mut bool,
    stop_requested: Option<fn() -> bool>,
    token_ids: &mut Vec<i32>,
) -> Result<(), GenerateError> {
    let mut last_heartbeat = Instant::now();
    'decode: while acc.completion < max_tokens && engine.pos() < engine.ctx() {
        if stop_requested.is_some_and(|stop| stop()) {
            *finish = "error";
            break;
        }
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
        let eos = engine.eos_id();
        if parsed.eos_policy != EosPolicy::Default && eos < 0 {
            return Err(GenerateError::Engine(
                "EOS policy has no model EOS token".into(),
            ));
        }
        let excluded = parsed.eos_policy.excluded(eos, acc.thinking_inside());
        let excluded_eot = excluded.and_then(|_| {
            let eot = engine.eot_id();
            (eot >= 0 && engine.token_is_stop(eot)).then_some(eot)
        });
        let token = if let SampleOverride::Token(t) = ov {
            if excluded == Some(t) || excluded_eot == Some(t) {
                return Err(GenerateError::Engine(
                    "forced token conflicts with EOS policy".into(),
                ));
            }
            t
        } else if let Some(excluded_id) = excluded {
            engine.sample_excluding(temperature, top_k, top_p, min_p, rng, excluded_id)
        } else {
            engine.sample(temperature, top_k, top_p, min_p, rng)
        };
        if token < 0 && excluded.is_some() {
            return Err(GenerateError::Engine(
                "EOS policy left no sample token".into(),
            ));
        }
        if token < 0 || engine.token_is_stop(token) {
            *finish = "stop";
            break;
        }
        let budget = (max_tokens - acc.completion).min(engine.ctx() - engine.pos());
        // Forced control prefixes can change policy between tokens. Only
        // an unconstrained greedy segment may be committed ahead of output.
        let accepted = if temperature <= 0.0
            && parsed.eos_policy == EosPolicy::Default
            && matches!(ov, SampleOverride::None)
            && parsed.required_tool_prefix.is_empty()
            && parsed.required_think_end_prefix.is_empty()
        {
            let accepted = engine.eval_greedy(token, budget)?;
            if engine.last_eval_speculated() {
                *speculation = true;
            }
            accepted
        } else {
            engine.eval(token)?;
            vec![token]
        };
        if accepted.is_empty() || accepted.len() > budget as usize || accepted[0] != token {
            engine.invalidate();
            return Err(GenerateError::Engine("invalid greedy prefix result".into()));
        }
        let count = accepted.len();
        for (index, token) in accepted.into_iter().enumerate() {
            if stop_requested.is_some_and(|stop| stop()) {
                *finish = "error";
                engine.invalidate();
                break 'decode;
            }
            if token < 0 {
                engine.invalidate();
                return Err(GenerateError::Engine("invalid greedy prefix token".into()));
            }
            if engine.token_is_stop(token) {
                *finish = "stop";
                if index + 1 < count {
                    engine.invalidate();
                }
                break 'decode;
            }
            if parsed.return_token_ids {
                token_ids.push(token);
            }
            // A speculative prefix advances multiple tokens in one decode step.
            if index == 0 {
                *decode_steps += 1;
            }
            if first_tok.is_none() {
                *first_tok = Some(Instant::now());
            }
            let piece = engine.token_text(token).inspect_err(|_| {
                engine.invalidate();
            })?;
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
                                engine.invalidate();
                                return Err(GenerateError::Io);
                            }
                        }
                    }
                    Api::Responses => {
                        if let Some(st) = resp.as_mut() {
                            if !responses_sse_stream_update(w, req, st, view, false) {
                                engine.invalidate();
                                return Err(GenerateError::Io);
                            }
                        }
                    }
                }
                stream_heartbeat_if_due(
                    w,
                    req,
                    resp.as_deref_mut(),
                    &mut last_heartbeat,
                    Instant::now(),
                    ": decode\n\n",
                );
                flush(w, out).inspect_err(|_| {
                    engine.invalidate();
                })?;
            }

            if feed.hit_stop {
                *finish = "stop";
                engine.invalidate();
                break 'decode;
            }
            if acc.track_tools && acc.saw_tool_end && req.chat_format == ChatFormat::DeepSeek {
                *finish = "tool_calls";
                if index + 1 < count {
                    engine.invalidate();
                }
                break 'decode;
            }
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
    pub(crate) vision: Vec<VisionPromptInput>,
    audios: Vec<AudioPromptInput>,
    videos: Vec<VisionPromptInput>,
}

struct MediaPrompt {
    tokens: Vec<i32>,
    vision: Vec<VisionPromptInput>,
    audios: Vec<AudioPromptInput>,
    videos: Vec<VisionPromptInput>,
}

fn prepare_media(
    engine: &dyn DecodeIo,
    parsed: &ParsedRequest,
    tokens: Vec<i32>,
) -> Result<MediaPrompt, GenerateError> {
    if parsed.images.is_empty() && parsed.audios.is_empty() && parsed.videos.is_empty() {
        return Ok(MediaPrompt {
            tokens,
            vision: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
        });
    }
    if syntax_for_model_id(engine.model_id()) == ModelSyntax::Mimo2 {
        return prepare_mimo(engine, parsed, tokens);
    }
    if !parsed.videos.is_empty() {
        return Err(GenerateError::Unsupported("video input requires MiMo"));
    }
    const GLM_IMAGE_TOKEN: i32 = 154854;
    const INKLING_IMAGE_TOKEN: i32 = 200054;
    const STEP_IMAGE_TOKEN: i32 = 128001;
    const LING3VL_IMAGE_TOKEN: i32 = 157157;
    const MIMO_IMAGE_TOKEN: i32 = 151655;
    const INKLING_AUDIO_TOKEN: i32 = 200053;
    const MIMO_AUDIO_TOKEN: i32 = 151669;
    let image_token = match syntax_for_model_id(engine.model_id()) {
        ModelSyntax::Glm53 => GLM_IMAGE_TOKEN,
        ModelSyntax::Inkling => INKLING_IMAGE_TOKEN,
        ModelSyntax::Step37 => STEP_IMAGE_TOKEN,
        ModelSyntax::Ling3Vl => LING3VL_IMAGE_TOKEN,
        ModelSyntax::Mimo2 => MIMO_IMAGE_TOKEN,
        _ => {
            return Err(GenerateError::Unsupported(
                "serial images require GLM-5.3, Inkling, Step or Ling",
            ))
        }
    };
    if parsed.images.len() + parsed.audios.len() + parsed.videos.len() > 4 {
        return Err(GenerateError::Unsupported(
            "serial media supports 1 to 4 inputs",
        ));
    }
    let mut image_spans = Vec::with_capacity(parsed.images.len());
    let mut expanded_len = tokens.len();
    for image in &parsed.images {
        let span = engine.vision_tokens(&image.data)?;
        if span.is_empty() {
            return Err(GenerateError::Engine(
                "image probe returned zero tokens".into(),
            ));
        }
        expanded_len = expanded_len
            .checked_add(span.len() - 1)
            .ok_or_else(|| GenerateError::Engine("expanded image prompt is too large".into()))?;
        image_spans.push(span);
    }
    let mut audio_counts = Vec::with_capacity(parsed.audios.len());
    for audio in &parsed.audios {
        let count = engine.audio_probe(&audio.data)?;
        if count == 0 {
            return Err(GenerateError::Engine(
                "audio probe returned zero tokens".into(),
            ));
        }
        expanded_len = expanded_len
            .checked_add(count as usize - 1)
            .ok_or_else(|| GenerateError::Engine("expanded media prompt is too large".into()))?;
        audio_counts.push(count);
    }
    let mut expanded = Vec::with_capacity(expanded_len);
    let mut images = Vec::with_capacity(parsed.images.len());
    let mut image_index = 0usize;
    let mut audios = Vec::with_capacity(parsed.audios.len());
    let mut audio_index = 0usize;
    for token in tokens {
        if (image_token == INKLING_IMAGE_TOKEN && token == INKLING_AUDIO_TOKEN)
            || (image_token == MIMO_IMAGE_TOKEN && token == MIMO_AUDIO_TOKEN)
        {
            let Some((audio, count)) = parsed
                .audios
                .get(audio_index)
                .zip(audio_counts.get(audio_index))
            else {
                return Err(GenerateError::Engine(
                    "ambiguous audio placeholder in prompt".into(),
                ));
            };
            let token_offset = u32::try_from(expanded.len())
                .map_err(|_| GenerateError::Engine("expanded media prompt is too large".into()))?;
            expanded.extend(std::iter::repeat_n(token, *count as usize));
            audios.push(AudioPromptInput {
                data: audio.data.clone(),
                token_offset,
            });
            audio_index += 1;
            continue;
        }
        if token != image_token {
            expanded.push(token);
            continue;
        }
        let Some((image, span)) = parsed
            .images
            .get(image_index)
            .zip(image_spans.get(image_index))
        else {
            return Err(GenerateError::Engine(
                "ambiguous image placeholder in prompt".into(),
            ));
        };
        let token_offset = u32::try_from(expanded.len())
            .map_err(|_| GenerateError::Engine("expanded image prompt is too large".into()))?;
        expanded.extend_from_slice(span);
        images.push(VisionPromptInput {
            data: image.data.clone(),
            token_offset,
            frames: Vec::new(),
        });
        image_index += 1;
    }
    if image_index != parsed.images.len()
        || audio_index != parsed.audios.len()
        || expanded.len() != expanded_len
    {
        return Err(GenerateError::Engine(
            "media placeholder count does not match payloads".into(),
        ));
    }
    Ok(MediaPrompt {
        tokens: expanded,
        vision: images,
        audios,
        videos: Vec::new(),
    })
}

/// 2 fps frames and a temporal patch of 2, so each pair is one second.
const MIMO_SECONDS_PER_PAIR: f32 = 1.0;

fn prepare_mimo(
    engine: &dyn DecodeIo,
    parsed: &ParsedRequest,
    tokens: Vec<i32>,
) -> Result<MediaPrompt, GenerateError> {
    if parsed.images.len() + parsed.audios.len() + parsed.videos.len() > 4 {
        return Err(GenerateError::Unsupported(
            "serial media supports 1 to 4 inputs",
        ));
    }

    let mut pieces = Vec::new();
    let mut vision = Vec::new();
    let mut audios = Vec::new();
    let mut videos = Vec::new();
    // Pair a video only with the audio part that follows it in the same
    // message. The next message's audio is a separate clip; the rendered
    // tokens have a boundary between the two stubs.
    for message in &parsed.messages {
        let parts = &message.parts;
        let mut index = 0usize;
        while index < parts.len() {
            match &parts[index] {
                ChatPart::Text(_) | ChatPart::ToolResult { .. } => index += 1,
                ChatPart::Image(slot) => {
                    let image = parsed.images.get(*slot).ok_or_else(|| {
                        GenerateError::Engine("image reference is missing".into())
                    })?;
                    let span = engine.vision_tokens(&image.data)?;
                    if span.is_empty() {
                        return Err(GenerateError::Engine(
                            "image probe returned zero tokens".into(),
                        ));
                    }
                    pieces.push(ds4_core::MediaPiece::Image {
                        count: span.len() as u32,
                    });
                    vision.push(VisionPromptInput {
                        data: image.data.clone(),
                        token_offset: 0,
                        frames: Vec::new(),
                    });
                    index += 1;
                }
                ChatPart::Audio(slot) => {
                    let audio = parsed.audios.get(*slot).ok_or_else(|| {
                        GenerateError::Engine("audio reference is missing".into())
                    })?;
                    let count = engine.audio_probe(&audio.data)?;
                    if count == 0 {
                        return Err(GenerateError::Engine(
                            "audio probe returned zero tokens".into(),
                        ));
                    }
                    pieces.push(ds4_core::MediaPiece::Audio { count });
                    audios.push(AudioPromptInput {
                        data: audio.data.clone(),
                        token_offset: 0,
                    });
                    index += 1;
                }
                ChatPart::Video(slot) => {
                    let video = parsed.videos.get(*slot).ok_or_else(|| {
                        GenerateError::Engine("video reference is missing".into())
                    })?;
                    let (_duration, packed) = ds4_core::load_video(&video.data)
                        .map_err(|error| GenerateError::Engine(error.to_string()))?;
                    if packed.is_empty() {
                        return Err(GenerateError::Engine("video produced no frames".into()));
                    }

                    // The audio that follows a video is that video's track. Text
                    // between them leaves a separate audio part.
                    let audio_len = match parts.get(index + 1) {
                        Some(ChatPart::Audio(audio_slot)) => {
                            let audio = parsed.audios.get(*audio_slot).ok_or_else(|| {
                                GenerateError::Engine("audio reference is missing".into())
                            })?;
                            let count = engine.audio_probe(&audio.data)?;
                            if count == 0 {
                                return Err(GenerateError::Engine(
                                    "audio probe returned zero tokens".into(),
                                ));
                            }
                            audios.push(AudioPromptInput {
                                data: audio.data.clone(),
                                token_offset: 0,
                            });
                            count
                        }
                        _ => 0,
                    };
                    let mut pairs = Vec::with_capacity(packed.len());
                    for (pair_index, visual) in packed.iter().enumerate() {
                        let start_s = pair_index as f32 * MIMO_SECONDS_PER_PAIR;
                        let timestamp_ids =
                            engine.tokenize_text(&ds4_core::format_timestamp(start_s))?;
                        if timestamp_ids.is_empty() {
                            return Err(GenerateError::Engine("video timestamp is empty".into()));
                        }
                        let audio_tokens = if audio_len == 0 {
                            0
                        } else {
                            // The last pair keeps the remaining codec rows. Duration
                            // times 6.25 Hz is not the feature length.
                            let end_s = if pair_index + 1 == packed.len() {
                                start_s + audio_len as f32
                            } else {
                                (pair_index as f32 + 1.0) * MIMO_SECONDS_PER_PAIR
                            };
                            ds4_core::audio_interval(start_s, end_s, audio_len)
                                .map_err(|error| GenerateError::Engine(error.to_string()))?
                        };
                        pairs.push(ds4_core::VideoPair {
                            timestamp_s: start_s,
                            timestamp_ids,
                            height: visual.height(),
                            width: visual.width(),
                            audio_tokens,
                        });
                    }
                    if audio_len == 0 {
                        pieces.push(ds4_core::MediaPiece::Video { pairs });
                        index += 1;
                    } else {
                        pieces.push(ds4_core::MediaPiece::Joint { pairs });
                        index += 2;
                    }
                    videos.push(VisionPromptInput {
                        data: video.data.clone(),
                        token_offset: 0,
                        frames: packed,
                    });
                }
            }
        }
    }

    let (expanded, spans) = ds4_core::expand_pieces(&tokens, &pieces)
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    ds4_core::check_span_budget(spans.len())
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    Ok(MediaPrompt {
        tokens: expanded,
        vision,
        audios,
        videos,
    })
}

pub(crate) fn prepare_serial_prompt(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
) -> Result<PreparedSerialPrompt, GenerateError> {
    if let Some(msg) = generation_blocked(parsed, engine.model_id()) {
        return Err(GenerateError::Unsupported(msg));
    }

    let mut parsed = parsed.clone();
    engine.restore_chat(&mut parsed)?;
    let syntax = syntax_for_model_id(engine.model_id());
    let tool_replay = parsed.images.is_empty()
        && parsed.audios.is_empty()
        && parsed.videos.is_empty()
        && tool_replay_disk_cache_eligible(&parsed, syntax);
    if tool_replay {
        engine.restore_tool_replay(&mut parsed.messages);
    }
    if engine.tokenizes_control_literals() {
        prepare_required_prefixes(&mut parsed, syntax, |literal| {
            engine.tokenize_rendered_chat(literal)
        })?;
    }

    let prompt = engine.render_request(&parsed)?;
    let tokens = match parsed.kind {
        ReqKind::Completion => {
            let text = std::str::from_utf8(&prompt).unwrap_or("");
            engine.tokenize_text(text)?
        }
        ReqKind::Chat => engine.tokenize_rendered_chat(&prompt)?,
    };
    let MediaPrompt {
        tokens,
        vision,
        audios,
        videos,
    } = prepare_media(engine, &parsed, tokens)?;
    Ok(PreparedSerialPrompt {
        parsed,
        tool_replay,
        prompt,
        tokens,
        vision,
        audios,
        videos,
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
        None,
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
    stop_requested: Option<fn() -> bool>,
    out: &mut impl Write,
) -> Result<(GenerateOutcome, Vec<u8>), GenerateError> {
    let PreparedSerialPrompt {
        mut parsed,
        tool_replay,
        mut prompt,
        tokens,
        vision,
        audios,
        videos,
    } = prep;
    let syntax = syntax_for_model_id(engine.model_id());
    let mut req = stream_req_from_parsed(&parsed, engine.model_id());
    let mut w = Writer::new(created);
    if req.stream {
        w.out.extend_from_slice(&sse_headers(cors));
        flush(&mut w, out)?;
    }
    engine.begin_trace();
    let t_prefill = Instant::now();
    let sync_result = if !vision.is_empty() || !audios.is_empty() || !videos.is_empty() {
        engine
            .sync_mimo_prompt(&tokens, &vision, &audios, &videos)
            .map(|()| 0)
    } else if tool_replay {
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
    let mut speculation = false;
    let mut token_ids = Vec::new();

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
        token_ids.clear();

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
            &mut speculation,
            stop_requested,
            &mut token_ids,
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

        finish = terminal_finish(finish);
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
                if retry_chat(
                    engine,
                    &mut parsed,
                    &mut prompt,
                    &acc,
                    "unterminated tool call",
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
            if retry_chat(engine, &mut parsed, &mut prompt, &acc, "invalid tool call").is_ok() {
                recovery_attempted = true;
                continue;
            }
            finish = "error";
        }
        break;
    }

    let completion = acc.completion;
    req.timings.prefill_tokens = prompt_n - req.cache_read_tokens;
    req.timings.prefill_cached = req.cache_read_tokens;
    req.timings.decode_tokens = completion;
    req.timings.decode_steps = decode_steps;
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
    let visible = if parsed_gen.calls.is_empty()
        && parsed.kind == ReqKind::Chat
        && think_mode_enabled(parsed.think_mode)
        && !matches!(finish, "error" | "length")
        && !acc.thinking_inside()
    {
        thinking_visible_key(&prompt, &parsed_gen.content, syntax, req.chat_format, true)
    } else {
        None
    }
    .or_else(|| {
        motif3_no_think_visible_checkpoint(&parsed, syntax, &prompt, &parsed_gen.content, finish)
    });
    if let Some(visible) = visible {
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
                        acc.reasoning_tokens,
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
                let think = if acc.thinking_inside() {
                    ThinkBlock::Open
                } else {
                    ThinkBlock::Closed
                };
                responses_final_response(
                    &req,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    finish,
                    think,
                    prompt_n,
                    completion,
                    acc.reasoning_tokens,
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
                if parsed.return_token_ids {
                    &token_ids
                } else {
                    &[]
                },
            ),
        };
        bytes
    };
    if finish != "error" && finish != "length" {
        engine.remember_chat(&parsed, &parsed_gen);
    }
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
        timings: req.timings,
        speculation_active: speculation,
        reuse: engine.last_reuse(),
        reuse_miss: engine.last_miss(),
        lane: None,
        fallback_reason: None,
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

    fn vision_probe(&self, _data: &[u8]) -> Result<VisionProbe, GenerateError> {
        if matches!(
            syntax_for_model_id(self.model_id),
            ModelSyntax::Glm53 | ModelSyntax::Inkling
        ) {
            Ok(VisionProbe { token_count: 16 })
        } else {
            Err(GenerateError::Unsupported("vision encoder is not loaded"))
        }
    }

    fn vision_tokens(&self, data: &[u8]) -> Result<Vec<i32>, GenerateError> {
        if syntax_for_model_id(self.model_id) == ModelSyntax::Step37 {
            return Ok([vec![128000], vec![128001; 169], vec![128002]].concat());
        }
        if syntax_for_model_id(self.model_id) == ModelSyntax::Mimo2 {
            let count = ds4_core::image_pad_count(data)
                .map_err(|error| GenerateError::Engine(error.to_string()))?;
            return Ok(vec![ds4_core::IMAGE_PAD; count as usize]);
        }
        let marker = if syntax_for_model_id(self.model_id) == ModelSyntax::Inkling {
            200054
        } else {
            154854
        };
        Ok(vec![marker; self.vision_probe(data)?.token_count as usize])
    }

    fn sync_vision_prompt(
        &mut self,
        tokens: &[i32],
        _images: &[VisionPromptInput],
    ) -> Result<(), GenerateError> {
        self.sync(tokens)
    }

    fn audio_probe(&self, data: &[u8]) -> Result<u32, GenerateError> {
        if syntax_for_model_id(self.model_id) == ModelSyntax::Inkling {
            return Ok(2);
        }
        if syntax_for_model_id(self.model_id) == ModelSyntax::Mimo2 {
            return ds4_core::audio_pad_count(data)
                .map_err(|error| GenerateError::Engine(error.to_string()));
        }
        Err(GenerateError::Unsupported("audio encoder is not loaded"))
    }

    fn sync_media_prompt(
        &mut self,
        tokens: &[i32],
        _images: &[VisionPromptInput],
        _audios: &[AudioPromptInput],
    ) -> Result<(), GenerateError> {
        self.sync(tokens)
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
    reuse: ReuseTaken,
    miss: ReuseMiss,
}

#[cfg(feature = "native")]
impl SerialKvIo for NativeSerialKvIo<'_, '_, '_, '_> {
    fn ctx(&self) -> i32 {
        self.session.ctx()
    }

    fn note_reuse(&mut self, taken: ReuseTaken) {
        self.reuse = taken;
    }

    fn note_miss(&mut self, miss: ReuseMiss) {
        if self.miss == ReuseMiss::None {
            self.miss = miss;
        }
    }

    fn chat_token_ids(&self) -> (i32, i32) {
        (self.vocab.user_id, self.vocab.assistant_id)
    }

    fn live_len(&self) -> i32 {
        self.session.host().live_len()
    }

    fn sync_start(&self, tokens: &[i32]) -> i32 {
        self.session.last_plan(tokens).start
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

#[cfg(any(feature = "native", test))]
fn serial_mtp_ready(family: ds4_core::ModelFamily, sidecar: bool, dots3: bool, draft: i32) -> bool {
    match family {
        ds4_core::ModelFamily::Dots3Note => dots3,
        ds4_core::ModelFamily::Inkling | ds4_core::ModelFamily::Step37 => sidecar,
        ds4_core::ModelFamily::Mimo2 => draft > 1,
        _ => false,
    }
}

#[test]
fn dots3_embedded_mtp_needs_session_state() {
    use ds4_core::ModelFamily::{Dots3Note, Inkling, Mimo2, Qwen4Exp, Step37};
    assert!(serial_mtp_ready(Dots3Note, false, true, 1));
    assert!(!serial_mtp_ready(Dots3Note, false, false, 1));
    assert!(!serial_mtp_ready(Dots3Note, true, false, 1));
    assert!(serial_mtp_ready(Inkling, true, false, 1));
    assert!(serial_mtp_ready(Step37, true, false, 1));
    assert!(!serial_mtp_ready(Qwen4Exp, true, false, 1));
    assert!(serial_mtp_ready(Mimo2, false, false, 3));
    assert!(!serial_mtp_ready(Mimo2, true, false, 1));
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
    chat_history: Option<crate::chat_input::History>,
    prompt_sync_elapsed: Option<Duration>,
    ctx: i32,
    prefix_reuse: ds4_core::ReuseKind,
    speculated: bool,
    reuse: ReuseTaken,
    miss: ReuseMiss,
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
            chat_history: None,
            prompt_sync_elapsed: None,
            ctx,
            prefix_reuse: ds4_core::ReuseKind::Exact,
            speculated: false,
            reuse: ReuseTaken::Cold,
            miss: ReuseMiss::None,
        }
    }

    pub fn with_prefix_reuse(mut self, reuse: ds4_core::ReuseKind) -> Self {
        self.prefix_reuse = reuse;
        self
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
            reuse: ReuseTaken::Cold,
            miss: ReuseMiss::None,
        };
        let policy = DiskSyncPolicy {
            save_current,
            load: disk_eligible,
        };
        let reuse_off = self.prefix_reuse == ds4_core::ReuseKind::None;
        let result = if reuse_off {
            disk_sync_prompt_impl(
                &mut io,
                store.as_mut(),
                model_id,
                quant_bits,
                prompt,
                tokens,
                checkpoint.as_ref(),
                thinking_visible_eligible,
                policy,
                false,
                PromptReuse::Off,
            )
        } else if self.model.chat_template().is_some() {
            disk_sync_template(
                &mut io,
                store.as_mut(),
                model_id,
                quant_bits,
                prompt,
                tokens,
                policy,
            )
        } else if tool_replay {
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
        self.reuse = if result.is_ok() {
            io.reuse
        } else {
            ReuseTaken::Cold
        };
        // A request that reused something is not a miss. The mechanism is
        // the answer; the reason says why nothing was taken.
        self.miss = if self.reuse == ReuseTaken::Cold {
            io.miss
        } else {
            ReuseMiss::None
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

    fn template(&self) -> Option<&ds4_core::chat_template::Template> {
        self.model.chat_template()
    }

    fn render_request(&self, parsed: &ParsedRequest) -> Result<Vec<u8>, GenerateError> {
        crate::chat_input::render_model(self.model.chat_template(), self.model_id(), parsed)
    }

    fn restore_chat(&self, parsed: &mut ParsedRequest) -> Result<(), GenerateError> {
        if self.model.chat_template().is_some() && !parsed.live_call_ids.is_empty() {
            if let Some(history) = &self.chat_history {
                history.restore(parsed)?;
            } else if parsed.responses_requires_live_tool_state
                || parsed.anthropic_requires_live_tool_state
            {
                return Err(GenerateError::Unsupported(
                    "retained chat history is unavailable",
                ));
            }
        }
        Ok(())
    }

    fn remember_chat(&mut self, parsed: &ParsedRequest, generated: &crate::tools::ParsedGenerated) {
        if self.model.chat_template().is_some() {
            self.chat_history = (!generated.calls.is_empty())
                .then(|| crate::chat_input::History::capture(parsed.clone(), generated));
        }
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

    fn eos_id(&self) -> i32 {
        self.model.vocab().eos_id
    }

    fn eot_id(&self) -> i32 {
        self.model.vocab().eot_id
    }

    fn vision_probe(&self, data: &[u8]) -> Result<VisionProbe, GenerateError> {
        self.model
            .vision_probe(data)
            .map(|info| VisionProbe {
                token_count: info.token_count,
            })
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn vision_tokens(&self, data: &[u8]) -> Result<Vec<i32>, GenerateError> {
        self.model
            .vision_tokens(data)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn audio_probe(&self, data: &[u8]) -> Result<u32, GenerateError> {
        self.model
            .audio_probe(data)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn sync_vision_prompt(
        &mut self,
        tokens: &[i32],
        images: &[VisionPromptInput],
    ) -> Result<(), GenerateError> {
        self.sync_media_prompt(tokens, images, &[])
    }

    fn sync_media_prompt(
        &mut self,
        tokens: &[i32],
        images: &[VisionPromptInput],
        audios: &[AudioPromptInput],
    ) -> Result<(), GenerateError> {
        self.sync_mimo_prompt(tokens, images, audios, &[])
    }

    fn sync_mimo_prompt(
        &mut self,
        tokens: &[i32],
        images: &[VisionPromptInput],
        audios: &[AudioPromptInput],
        videos: &[VisionPromptInput],
    ) -> Result<(), GenerateError> {
        if !videos.is_empty() && syntax_for_model_id(self.model_id()) != ModelSyntax::Mimo2 {
            return Err(GenerateError::Unsupported("video input requires MiMo"));
        }
        self.prompt_sync_elapsed = None;
        self.session_disk_storable = false;
        self.thinking_visible = None;
        self.reuse = ReuseTaken::Cold;
        let tokens = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        let images = images
            .iter()
            .map(|image| ds4_core::VisionInput {
                data: &image.data,
                token_offset: image.token_offset,
            })
            .collect::<Vec<_>>();
        let audios = audios
            .iter()
            .map(|audio| ds4_core::AudioInput {
                data: &audio.data,
                token_offset: audio.token_offset,
            })
            .collect::<Vec<_>>();
        let videos_in = videos
            .iter()
            .map(|video| ds4_core::VisionInput {
                data: &video.data,
                token_offset: video.token_offset,
            })
            .collect::<Vec<_>>();
        let packed = videos
            .iter()
            .map(|video| video.frames.as_slice())
            .collect::<Vec<_>>();
        let started = Instant::now();
        let session = self.session()?;
        let result = if videos_in.is_empty() {
            session.sync_media(&tokens, &images, &audios)
        } else {
            session.sync_mimo(&tokens, &images, &audios, &videos_in, &packed)
        }
        .map_err(|error| GenerateError::Engine(error.to_string()));
        if result.is_ok() {
            self.prompt_sync_elapsed = Some(started.elapsed());
        }
        result
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

    fn begin_trace(&mut self) {
        self.reuse = ReuseTaken::Cold;
        self.miss = ReuseMiss::None;
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        // A retry re-syncs mid-request: nothing was reused for what it
        // produces, but why the first sync refused a candidate still holds.
        self.reuse = ReuseTaken::Cold;
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
            reuse: ReuseTaken::Cold,
            miss: ReuseMiss::None,
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
            reuse: ReuseTaken::Cold,
            miss: ReuseMiss::None,
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

    fn eval_greedy(&mut self, first: i32, budget: i32) -> Result<Vec<i32>, GenerateError> {
        let family = self.model.family();
        let dots3 = family == ds4_core::ModelFamily::Dots3Note && self.session()?.has_dots3_mtp();
        if !serial_mtp_ready(
            family,
            self.model.mtp().is_some(),
            dots3,
            self.model.mtp_draft_tokens(),
        ) || std::env::var_os("DS4_MTP_SPEC_DISABLE").is_some()
        {
            self.speculated = false;
            self.eval(first)?;
            return Ok(vec![first]);
        }
        self.speculated = true;
        let eos = self.model.token_eos();
        self.session()?
            .eval_speculative_argmax(first, budget, eos)
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn last_eval_speculated(&self) -> bool {
        self.speculated
    }

    fn last_reuse(&self) -> ReuseTaken {
        self.reuse
    }

    fn last_miss(&self) -> ReuseMiss {
        self.miss
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

    fn sample_excluding(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
        excluded_id: i32,
    ) -> i32 {
        match self.session() {
            Ok(s) => s.sample_excluding(temperature, top_k, top_p, min_p, rng, excluded_id),
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
        settle_thinking_visible_checkpoint, thinking_visible_cache_eligible, thinking_visible_key,
        tool_replay_disk_cache_eligible, tool_replay_producer_eligible, try_store_continued,
        try_store_live, DiskSyncPolicy, GenerateError, ReuseMiss, ReuseTaken, SerialKvIo,
        ThinkingVisibleCheckpoint,
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
        saved_prefixes: Vec<Vec<i32>>,
        progress_frontiers: Vec<usize>,
        invalidations: usize,
        syncs: Vec<Vec<i32>>,
        suffixes: Vec<Vec<u8>>,
        loads: Vec<(PathBuf, u64, u64)>,
        events: Vec<&'static str>,
        user_token_id: i32,
        assistant_token_id: i32,
        live_token_reads: Cell<usize>,
        planned_start: Option<i32>,
        reuse: ReuseTaken,
        miss: ReuseMiss,
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
                saved_prefixes: Vec::new(),
                progress_frontiers: Vec::new(),
                invalidations: 0,
                syncs: Vec::new(),
                suffixes: Vec::new(),
                loads: Vec::new(),
                events: Vec::new(),
                user_token_id: -1,
                assistant_token_id: -1,
                live_token_reads: Cell::new(0),
                planned_start: None,
                reuse: ReuseTaken::Cold,
                miss: ReuseMiss::None,
            }
        }
    }

    impl SerialKvIo for FakeSerial {
        fn ctx(&self) -> i32 {
            self.ctx
        }

        fn note_reuse(&mut self, taken: ReuseTaken) {
            self.reuse = taken;
        }

        fn note_miss(&mut self, miss: ReuseMiss) {
            if self.miss == ReuseMiss::None {
                self.miss = miss;
            }
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

        fn sync_start(&self, _tokens: &[i32]) -> i32 {
            self.planned_start.unwrap_or_else(|| self.live_len())
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
            self.saved_prefixes.push(self.live.clone());
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
            self.loads.push((
                path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
                offset,
                length,
            ));
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

    #[test]
    fn template_rejects_text_prefix() {
        // Identical prompt bytes do not prove identical tokens: a BPE merge
        // can cross the old frontier. Only the complete tokenization is valid.
        let mut io = FakeSerial::new(&[1, 2], b"prefix");
        io.suffix_tokens = vec![99];
        let cached = super::disk_sync_template(
            &mut io,
            None,
            6,
            2,
            b"prefix suffix",
            &[1, 3, 4],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert_eq!(io.live, [1, 3, 4]);
        assert!(io.suffixes.is_empty());
    }

    #[test]
    fn template_keeps_token_prefix() {
        let mut io = FakeSerial::new(&[1, 2], b"prefix");
        let cached = super::disk_sync_template(
            &mut io,
            None,
            6,
            2,
            b"different text",
            &[1, 2, 4],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 2);
        assert_eq!(io.live, [1, 2, 4]);
        assert!(io.suffixes.is_empty());
    }

    #[test]
    fn template_checks_disk_tokens() {
        for (name, loaded, expected) in [
            ("jinja-disk-hit", vec![1, 2], 2),
            ("jinja-disk-miss", vec![1, 9], 0),
        ] {
            let (dir, mut store) = store(name);
            candidate(&mut store, b"prefix", 2);
            let mut io = FakeSerial::new(&[], b"prefix suffix");
            io.loaded_tokens = loaded;
            io.suffix_tokens = vec![99];
            let cached = super::disk_sync_template(
                &mut io,
                Some(&mut store),
                0,
                2,
                b"prefix suffix",
                &[1, 2, 3],
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap();
            assert_eq!(cached, expected, "{name}");
            assert_eq!(io.live, [1, 2, 3], "{name}");
            assert!(io.suffixes.is_empty(), "{name}");
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn step_restart_uses_history_kv() {
        use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
        use serde_json::json;

        let template = Template::compile(
            include_str!("../../../tests/fixtures/step37/chat_template.jinja"),
            RenderClock::Fixed(0),
        )
        .unwrap();
        let options = ChatOptions::new(10, ds4_core::ChatThinkMode::None);
        let first = template
            .render_chat(&[json!({"role":"user", "content":"Hello"})], &[], options)
            .unwrap();
        let follow = template
            .render_chat(
                &[
                    json!({"role":"user", "content":"Hello"}),
                    json!({"role":"assistant", "content":"4"}),
                    json!({"role":"user", "content":"Again"}),
                ],
                &[],
                options,
            )
            .unwrap();
        let history = first
            .strip_suffix("assistant\n<think>\n</think>\n")
            .unwrap();
        let tokens = |text: &str| text.bytes().map(i32::from).collect::<Vec<_>>();
        let (dir, mut store) = store("step-history-frontier");
        store.bind_identity([7; 32]);
        let mut saving = FakeSerial::new(&[], first.as_bytes());
        saving.suffix_tokens = tokens(history);
        super::disk_sync_template(
            &mut saving,
            Some(&mut store),
            10,
            2,
            first.as_bytes(),
            &tokens(&first),
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        let (_, envelope) = store
            .text_prefix_candidate(follow.as_bytes(), 10, 2, 4096)
            .unwrap()
            .expect("next official render must find a saved history frontier");
        assert_eq!(envelope.text, history.as_bytes());
        assert_eq!(envelope.header.tokens as usize, history.len());
        assert!(saving.saved_prefixes.contains(&tokens(history)));
        assert_eq!(saving.live, tokens(&first));

        let options = store.opt.clone();
        drop(store);
        let mut store = Store::open(&dir, 16, true, options).unwrap();
        store.bind_identity([7; 32]);
        let mut loading = FakeSerial::new(&[], follow.as_bytes());
        loading.loaded_tokens = tokens(history);
        loading.suffix_tokens = tokens(
            follow
                .strip_suffix("assistant\n<think>\n</think>\n")
                .unwrap(),
        );
        let cached = super::disk_sync_template(
            &mut loading,
            Some(&mut store),
            10,
            2,
            follow.as_bytes(),
            &tokens(&follow),
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached as usize, history.len());
        assert_eq!(loading.reuse, ReuseTaken::Exact);
        assert_eq!(loading.live, tokens(&follow));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn motif_restart_uses_history_kv() {
        use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
        use serde_json::json;

        let template = Template::compile(
            include_str!("../../../tests/fixtures/chat-template/models/motif/chat_template.jinja"),
            RenderClock::Fixed(0),
        )
        .unwrap();
        let options = ChatOptions::new(3, ds4_core::ChatThinkMode::None);
        let first = template
            .render_chat(
                &[json!({"role":"user", "content":"What is 2 + 2?"})],
                &[],
                options,
            )
            .unwrap();
        let follow = template
            .render_chat(
                &[
                    json!({"role":"user", "content":"What is 2 + 2?"}),
                    json!({"role":"assistant", "content":" 2 + 2 = 4.\n"}),
                    json!({"role":"user", "content":"What is 4 + 1?"}),
                ],
                &[],
                options,
            )
            .unwrap();
        let history = first.strip_suffix("<think></think>").unwrap();
        assert!(follow.starts_with(history));
        assert!(!follow.starts_with(&first));
        let tokens = |text: &str| text.bytes().map(i32::from).collect::<Vec<_>>();
        let (dir, mut store) = store("motif-history-frontier");
        store.bind_identity([7; 32]);
        let mut saving = FakeSerial::new(&[], first.as_bytes());
        saving.suffix_tokens = tokens(history);
        super::disk_sync_template(
            &mut saving,
            Some(&mut store),
            3,
            2,
            first.as_bytes(),
            &tokens(&first),
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        let (_, envelope) = store
            .text_prefix_candidate(follow.as_bytes(), 3, 2, 4096)
            .unwrap()
            .expect("Motif must save the real history frontier");
        assert_eq!(envelope.text, history.as_bytes());
        assert_eq!(envelope.header.tokens as usize, history.len());
        assert!(saving.saved_prefixes.contains(&tokens(history)));
        assert_eq!(saving.live, tokens(&first));

        let options = store.opt.clone();
        drop(store);
        let mut store = Store::open(&dir, 16, true, options).unwrap();
        store.bind_identity([7; 32]);
        let mut loading = FakeSerial::new(&[], follow.as_bytes());
        loading.loaded_tokens = tokens(history);
        loading.suffix_tokens = tokens(follow.strip_suffix("<think></think>").unwrap());
        let cached = super::disk_sync_template(
            &mut loading,
            Some(&mut store),
            3,
            2,
            follow.as_bytes(),
            &tokens(&follow),
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached as usize, history.len());
        assert_eq!(loading.reuse, ReuseTaken::Exact);
        assert_eq!(loading.live, tokens(&follow));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn motif_history_needs_token_cut() {
        let prompt = b"history<|startofturn|><|assistant|><think></think>";
        assert!(super::history_frontier(3, prompt, &[1, 2, 11, 12], |_| Ok(vec![1, 2])).is_some());
        assert!(super::history_frontier(3, prompt, &[1, 2, 11, 12], |_| Ok(vec![1, 9])).is_none());
        assert!(super::history_frontier(3, prompt, &[1, 2], |_| Ok(vec![1, 2])).is_none());
        for other in [b"user<think></think>".as_slice(), b"<|assistant|><think>"] {
            assert!(
                super::history_frontier(3, other, &[1, 2], |_| panic!("not a history cut"))
                    .is_none()
            );
        }
    }

    #[test]
    fn step_history_needs_token_cut() {
        let prompt = b"history<|im_start|>assistant\n<think>\n</think>\n";
        assert!(super::history_frontier(10, prompt, &[1, 2, 3], |_| Ok(vec![1, 9])).is_none());
        assert!(super::history_frontier(10, prompt, &[1, 2, 3], |_| Ok(vec![1, 2, 3])).is_none());
        assert!(
            super::history_frontier(6, prompt, &[1, 2, 3], |_| panic!("other family")).is_none()
        );
    }

    #[test]
    fn step_keeps_live_frontier() {
        let (dir, mut store) = store("step-history-live");
        let mut io = FakeSerial::new(
            &[1, 2, 3],
            b"history<|im_start|>assistant\n<think>\n</think>\n",
        );
        io.suffix_tokens = vec![1, 2];
        let cached = super::disk_sync_template(
            &mut io,
            Some(&mut store),
            10,
            2,
            b"history<|im_start|>assistant\n<think>\n</think>\n",
            &[1, 2, 3],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached, 3);
        assert_eq!(io.syncs, [vec![1, 2, 3]]);
        assert!(io.saved_prefixes.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    /// Step's official follow-up render drops the empty think pair that the
    /// stored KV still holds, so a text-prefix hit would continue from a
    /// token sequence the client never sent. A record without a real
    /// history-frontier payload must still fall back to a cold prefill.
    #[test]
    fn template_refuses_a_text_only_history_match() {
        let (dir, mut store) = store("step-history");
        let history =
            b"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n".to_vec();
        let mut saving = FakeSerial::new(&[41, 42], &history);
        saving.trailer = Some(Vec::new());
        super::disk_sync_prompt(
            &mut saving,
            Some(&mut store),
            0,
            2,
            &history,
            &[41, 42],
            None,
            false,
            DiskSyncPolicy {
                save_current: true,
                load: false,
            },
        )
        .unwrap();

        let follow =
            b"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n";
        let mut loading = FakeSerial::new(&[], b"");
        // The payload's KV still carries the generation-form think pair.
        loading.loaded_tokens = vec![41, 42, 90];
        let cached = super::disk_sync_template(
            &mut loading,
            Some(&mut store),
            0,
            2,
            follow,
            &[41, 42, 3, 4],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert_eq!(loading.reuse, ReuseTaken::Cold);
        assert!(loading.syncs.last().unwrap().starts_with(&[41, 42, 3, 4]));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn appended_turn_records_exact_reuse() {
        let mut io = FakeSerial::new(&[1, 2], b"prefix");
        let cached = super::disk_sync_template(
            &mut io,
            None,
            6,
            2,
            b"prefix suffix",
            &[1, 2, 4],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        // Both counters are positive: cached prefix plus a prefilled turn.
        assert_eq!(cached, 2);
        assert_eq!(io.reuse, ReuseTaken::Exact);
    }

    /// Candidate refusal reasons from `docs/serving-contract.md`, each from the
    /// decision that produced it.
    #[test]
    fn a_refused_candidate_says_why() {
        let (dir, mut store) = store("miss-reasons");
        candidate(&mut store, b"prefix", 2);

        // The template re-rendered the conversation: stored tokens are no
        // longer a prefix of the prompt.
        let mut io = FakeSerial::new(&[], b"prefix suffix");
        io.loaded_tokens = vec![41, 42];
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"prefix suffix",
            &[1, 2, 3],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::RenderedPrefix);

        // Nothing stored under a text this prompt starts with.
        let mut io = FakeSerial::new(&[], b"");
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"unrelated",
            &[7],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::NoCheckpoint);

        // The payload does not hold the token count its record claims.
        let mut io = FakeSerial::new(&[], b"prefix suffix");
        io.loaded_tokens = vec![41];
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"prefix suffix",
            &[41, 42, 3],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::PayloadMismatch);

        // The record was chosen and then could not be read back.
        let mut io = FakeSerial::new(&[], b"prefix suffix");
        io.fail_load = true;
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"prefix suffix",
            &[41, 42, 3],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::PayloadMismatch);

        // A live session holds this conversation and the render moved: the
        // text still leads here, the token sequence does not.
        let mut io = FakeSerial::new(&[41, 42], b"prefix");
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"prefix suffix",
            &[7, 8, 9],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::RenderedPrefix);

        // A live session about another conversation says nothing: its text
        // does not lead to this prompt either.
        let mut io = FakeSerial::new(&[41, 42], b"unrelated");
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            0,
            2,
            b"prefix suffix",
            &[7, 8, 9],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::None);

        // A broader reason never masks the specific one that came first.
        let mut io = FakeSerial::new(&[], b"");
        io.note_miss(ReuseMiss::RenderedPrefix);
        io.note_miss(ReuseMiss::BelowThreshold);
        assert_eq!(io.miss, ReuseMiss::RenderedPrefix);

        // A conversation too short to have been stored says so, rather than
        // reporting the absence that shortness caused.
        let short_dir = std::env::temp_dir().join(format!(
            "ds4-server-disk-sync-miss-threshold-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&short_dir);
        let mut short_store = Store::open(
            &short_dir,
            16,
            true,
            Options {
                min_tokens: 8,
                cold_max_tokens: 32,
                continued_interval_tokens: 8,
                boundary_trim_tokens: 0,
                boundary_align_tokens: 0,
            },
        )
        .unwrap();
        let mut io = FakeSerial::new(&[], b"hi");
        super::disk_sync_template(
            &mut io,
            Some(&mut short_store),
            0,
            2,
            b"hi",
            &[1],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::BelowThreshold);
        let _ = fs::remove_dir_all(short_dir);

        // A record for another model is a mismatch, not an absence: the
        // prompt-keyed entry is there, its identity rules it out.
        let mut io = FakeSerial::new(&[], b"prefix suffix");
        super::disk_sync_template(
            &mut io,
            Some(&mut store),
            9,
            2,
            b"prefix suffix",
            &[1, 2, 3],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();
        assert_eq!(io.miss, ReuseMiss::PayloadMismatch);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_cold_prompt_records_cold_reuse() {
        let mut io = FakeSerial::new(&[], b"");
        let cached = super::disk_sync_template(
            &mut io,
            None,
            6,
            2,
            b"prefix suffix",
            &[1, 2, 4],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert_eq!(io.reuse, ReuseTaken::Cold);
    }

    #[test]
    fn reuse_off_skips_live_prefix() {
        let mut io = FakeSerial::new(&[1, 2], b"prefix");
        let cached = super::disk_sync_prompt_impl(
            &mut io,
            None,
            6,
            2,
            b"prefix suffix",
            &[1, 2, 4],
            None,
            false,
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
            false,
            super::PromptReuse::Off,
        )
        .unwrap();
        assert_eq!(cached, 0);
        assert_eq!(io.live, [1, 2, 4]);
        assert_eq!(io.reuse, ReuseTaken::Cold);
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
    fn live_reuse_counts_actual_start() {
        // Dots3 MTP replays an unaligned append from zero; plain Dots3
        // replays its last partial chunk. Exact and aligned hits remain hits.
        for (live_len, prompt_len, start, cached, taken) in [
            (600, 620, 0, 0, ReuseTaken::Cold),
            (600, 620, 576, 576, ReuseTaken::Partial),
            (600, 600, 600, 600, ReuseTaken::Exact),
            (608, 620, 608, 608, ReuseTaken::Exact),
            (600, 620, 620, 600, ReuseTaken::Exact),
            (600, 620, -1, 0, ReuseTaken::Cold),
        ] {
            let tokens = vec![1; prompt_len];
            let mut io = FakeSerial::new(&tokens[..live_len], b"prefix");
            io.planned_start = Some(start);
            let actual = super::disk_sync_template(
                &mut io,
                None,
                0,
                2,
                b"prefix append",
                &tokens,
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap();
            assert_eq!(
                actual, cached,
                "live={live_len}, prompt={prompt_len}, start={start}"
            );
            assert_eq!(io.reuse, taken);
            assert_eq!(
                io.miss.as_str(),
                if cached < live_len as i32 {
                    "session state requires prefix replay"
                } else {
                    "none"
                }
            );
            assert_eq!(io.syncs, [tokens]);
        }
    }

    #[test]
    fn disk_reuse_counts_actual_start() {
        for (live_len, prompt_len, start, cached, taken) in [
            (600, 620, 0, 0, ReuseTaken::Cold),
            (600, 620, 576, 576, ReuseTaken::Partial),
            (600, 600, 600, 600, ReuseTaken::Exact),
            (608, 620, 608, 608, ReuseTaken::Exact),
        ] {
            let (dir, mut store) = store("actual-start");
            candidate(&mut store, b"prefix", live_len as u32);
            let tokens = vec![1; prompt_len];
            let mut io = FakeSerial::new(&[], b"");
            io.loaded_tokens = tokens[..live_len].to_vec();
            io.planned_start = Some(start);
            let actual = super::disk_sync_template(
                &mut io,
                Some(&mut store),
                0,
                2,
                b"prefix append",
                &tokens,
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap();
            assert_eq!(
                actual, cached,
                "loaded={live_len}, prompt={prompt_len}, start={start}"
            );
            assert_eq!(io.reuse, taken);
            assert_eq!(
                io.miss.as_str(),
                if cached < live_len as i32 {
                    "session state requires prefix replay"
                } else {
                    "none"
                }
            );
            assert_eq!(io.loads.len(), 1);
            assert_eq!(io.syncs, [tokens]);
            let _ = fs::remove_dir_all(dir);
        }
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
    fn k2_thinking_visible_key_matches_adjacent_ifm_turns() {
        let prompt = b"<|ifm|begin_of_text|><|ifm|im_start|>assistant\n<ifm|think>\n";
        assert_eq!(
            thinking_visible_key(
                prompt,
                b"answer",
                ModelSyntax::K2Horizon,
                ChatFormat::K2Horizon,
                true,
            )
            .unwrap(),
            b"<|ifm|begin_of_text|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>answer<|ifm|im_end|>"
        );
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

#[cfg(test)]
mod mimo_prepare {
    use super::{prepare_media, ScriptedDecode};
    use crate::parse::{parse_chat_request, ParseEnv};

    fn engine() -> ScriptedDecode {
        ScriptedDecode {
            model_id: 12,
            prompt_tokens: vec![77],
            steps: Vec::new(),
            idx: 0,
            pos: 0,
            ctx: 8192,
            generation: 1,
            live: Vec::new(),
            suffix_tokens: Vec::new(),
        }
    }

    fn pads(tokens: &[i32], pad: i32) -> usize {
        tokens.iter().filter(|token| **token == pad).count()
    }

    fn tiny_png() -> Vec<u8> {
        let path = std::env::temp_dir().join(format!("ds4-mimo-prep-{}.png", std::process::id()));
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64",
                "-frames:v",
                "1",
            ])
            .arg(&path)
            .status()
            .expect("ffmpeg");
        assert!(status.success());
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes
    }

    #[test]
    fn still_image_expands_the_single_pad() {
        let png = tiny_png();
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{}"}}}}]}}]}}"#,
            b64(&png)
        );
        let parsed = parse_chat_request(&ParseEnv::default(), &body).unwrap();
        let count = ds4_core::image_pad_count(&parsed.images[0].data).unwrap() as usize;
        let stub = vec![
            ds4_core::VISION_START,
            ds4_core::IMAGE_PAD,
            ds4_core::VISION_END,
        ];
        let media = prepare_media(&engine(), &parsed, stub).unwrap();
        assert_eq!(pads(&media.tokens, ds4_core::IMAGE_PAD), count);
        assert!(count > 1);
        assert_eq!(media.vision.len(), 1);
        assert!(media.videos.is_empty());
    }

    fn tiny_mp4() -> Vec<u8> {
        let path = std::env::temp_dir().join(format!("ds4-mimo-prep-{}.mp4", std::process::id()));
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:d=0.4",
                "-r",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&path)
            .status()
            .expect("ffmpeg");
        assert!(status.success());
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes
    }

    fn b64(data: &[u8]) -> String {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        let mut index = 0;
        while index < data.len() {
            let b0 = data[index];
            let b1 = if index + 1 < data.len() {
                data[index + 1]
            } else {
                0
            };
            let b2 = if index + 2 < data.len() {
                data[index + 2]
            } else {
                0
            };
            let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
            out.push(TABLE[((n >> 18) & 63) as usize] as char);
            out.push(TABLE[((n >> 12) & 63) as usize] as char);
            if index + 1 < data.len() {
                out.push(TABLE[((n >> 6) & 63) as usize] as char);
            } else {
                out.push('=');
            }
            if index + 2 < data.len() {
                out.push(TABLE[(n & 63) as usize] as char);
            } else {
                out.push('=');
            }
            index += 3;
        }
        out
    }

    fn wav_silence(samples: usize) -> Vec<u8> {
        let data_bytes = samples * 2;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data_bytes) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24_000u32.to_le_bytes());
        out.extend_from_slice(&(48_000u32).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        out.extend(std::iter::repeat(0).take(data_bytes));
        out
    }

    #[test]
    fn video_and_joint_audio_replace_the_jinja_stubs() {
        let mp4 = tiny_mp4();
        let (_duration, packed) = ds4_core::load_video(&mp4).unwrap();
        let visual: u32 = packed.iter().map(|pair| pair.tokens).sum();
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"video_url","video_url":{{"url":"data:video/mp4;base64,{}"}}}}]}}]}}"#,
            b64(&mp4)
        );
        let parsed = parse_chat_request(&ParseEnv::default(), &body).unwrap();
        let stub = vec![
            ds4_core::VISION_START,
            ds4_core::VIDEO_PAD,
            ds4_core::VISION_END,
        ];
        let media = prepare_media(&engine(), &parsed, stub).unwrap();
        assert_eq!(media.tokens.first().copied(), Some(ds4_core::VIDEO_START));
        assert_eq!(media.tokens.last().copied(), Some(ds4_core::VIDEO_END));
        assert_eq!(pads(&media.tokens, ds4_core::VIDEO_PAD), visual as usize);
        assert_eq!(pads(&media.tokens, 77), packed.len());
        assert_eq!(media.videos.len(), 1);
        assert!(!media.videos[0].frames.is_empty());
        assert!(media.audios.is_empty());

        let wav = wav_silence(24_000);
        let audio_len = ds4_core::audio_pad_count(&wav).unwrap() as usize;
        let joint_body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"video_url","video_url":{{"url":"data:video/mp4;base64,{}"}}}},{{"type":"input_audio","input_audio":{{"format":"wav","data":"{}"}}}}]}}]}}"#,
            b64(&mp4),
            b64(&wav)
        );
        let joint = parse_chat_request(&ParseEnv::default(), &joint_body).unwrap();
        let joint_stub = vec![
            ds4_core::VISION_START,
            ds4_core::VIDEO_PAD,
            ds4_core::VISION_END,
            ds4_core::AUDIO_START,
            ds4_core::AUDIO_PAD,
            ds4_core::AUDIO_END,
        ];
        let joint_media = prepare_media(&engine(), &joint, joint_stub.clone()).unwrap();
        assert_eq!(pads(&joint_media.tokens, ds4_core::AUDIO_PAD), audio_len);
        assert_eq!(joint_media.audios.len(), 1);
        assert_eq!(joint_media.videos.len(), 1);

        let doubled = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"video_url","video_url":{{"url":"data:video/mp4;base64,{}"}}}},{{"type":"input_audio","input_audio":{{"format":"wav","data":"{}"}}}},{{"type":"input_audio","input_audio":{{"format":"wav","data":"{}"}}}}]}}]}}"#,
            b64(&mp4),
            b64(&wav),
            b64(&wav)
        );
        let doubled = parse_chat_request(&ParseEnv::default(), &doubled).unwrap();
        let mut doubled_stub = joint_stub;
        doubled_stub.extend_from_slice(&[
            ds4_core::AUDIO_START,
            ds4_core::AUDIO_PAD,
            ds4_core::AUDIO_END,
        ]);
        let err = match prepare_media(&engine(), &doubled, doubled_stub) {
            Err(err) => err,
            Ok(_) => panic!("second audio was accepted"),
        };
        assert!(err.to_string().contains("second audio"), "{err}");

        let split = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"video_url","video_url":{{"url":"data:video/mp4;base64,{}"}}}}]}},{{"role":"user","content":[{{"type":"input_audio","input_audio":{{"format":"wav","data":"{}"}}}}]}}]}}"#,
            b64(&mp4),
            b64(&wav)
        );
        let split = parse_chat_request(&ParseEnv::default(), &split).unwrap();
        let split_stub = vec![
            ds4_core::VISION_START,
            ds4_core::VIDEO_PAD,
            ds4_core::VISION_END,
            198,
            ds4_core::AUDIO_START,
            ds4_core::AUDIO_PAD,
            ds4_core::AUDIO_END,
        ];
        let split_media = prepare_media(&engine(), &split, split_stub).unwrap();
        assert_eq!(split_media.videos.len(), 1);
        assert_eq!(split_media.audios.len(), 1);
        assert_eq!(pads(&split_media.tokens, ds4_core::AUDIO_PAD), audio_len);
    }
}
