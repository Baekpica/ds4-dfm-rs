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
use crate::render::{ModelSyntax, DSML_IMAGE};
use crate::route::{
    compute_needs, think_mode_from_enabled, Api, NeedInput, ReqKind, ThinkMode, WireSurface,
};
use std::sync::Arc;

pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const DEFAULT_MIN_P: f32 = 0.05;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EosPolicy {
    #[default]
    Default,
    Reasoning,
    Global,
}

impl EosPolicy {
    pub(crate) fn excluded(self, eos: i32, in_reasoning: bool) -> Option<i32> {
        (eos >= 0
            && match self {
                Self::Default => false,
                Self::Reasoning => in_reasoning,
                Self::Global => true,
            })
        .then_some(eos)
    }
}

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
    Audio(usize),
    Video(usize),
    ToolResult { id: String, content: String },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestAudio {
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVideo {
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatMsg {
    pub role: String,
    pub name: String,
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
    /// The leading non-null member of a `type` array.  A client that renders an
    /// optional field the way schemars does -- `Option<u8>` goes out as
    /// `{"type":["integer","null"]}` -- writes no scalar type at all, so
    /// `prop_type` is empty for it.  Only the GLM path reads this, because
    /// GLM's template writes a string verbatim and every other value as JSON
    /// (chat_template.jinja:50) and the declared type is the only way to tell
    /// the two apart; the tolerant paths keep reading `prop_type` alone.
    pub prop_type_first: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParseEnv {
    pub default_model: String,
    pub default_tokens: i32,
    pub default_effort: ThinkMode,
    pub default_temp: f32,
    /// LIVE call ids for this surface at parse time (C `*_live_has_call_id`).
    pub live_ids: Vec<String>,
    /// The engine's request-omitted defaults for its own model set (unit H):
    /// thinking OFF unless asked (server_msgs.c request_init), reasoning
    /// effort HIGH when a bare thinking flag asks for it, min_p 0.0
    /// (ds4.h:54-56) and the engine's effort-name collapse
    /// (`think_mode_from_enabled`, server_msgs.c:186-189: everything non-zero
    /// below max is HIGH).  True only for the V4.1 (ds41) family in this
    /// port: the engine's parse is shared across its DeepSeek models, but the
    /// other families keep the port's own wire vocabulary (their references
    /// are their own stacks, not this engine).
    pub engine_defaults: bool,
    /// The GLM-5.3 scope: the artifact's own request-omitted values
    /// (`GlmValues`) instead of the fixed DeepSeek-style sampler this port
    /// applies to every other syntax.  The bin sets it when the model it
    /// opened is GLM-5.3, either variant.
    pub glm_values: bool,
}

impl Default for ParseEnv {
    fn default() -> Self {
        Self {
            default_model: "deepseek-v4-flash".into(),
            default_tokens: 393216,
            default_effort: ThinkMode::Low,
            default_temp: default_temperature(),
            live_ids: Vec::new(),
            engine_defaults: false,
            glm_values: false,
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

/// The GLM-5.3 scope's request-omitted sampler and effort level.  The
/// artifact states its own: the template honours `low` and `high` and reads
/// every other level as max (chat_template.jinja:2-3), and its
/// generation_config pins temperature 1.0 with top_p 0.95 and no min_p or
/// top_k key at all (generation_config.json, zai-org/GLM-5.3-Flash
/// @84c6a6a).  The host's own 1.0/1.0/0.05 sampler (ds4.h:52-54) is a
/// different model's contract, so the GLM scope carries its own values, each
/// one settable per deployment through the same knob family as ds41's
/// (DS4_SERVER_DEFAULT_TEMP / _MIN_P): the server serves one family per
/// process, so one set of environment values is enough.  Temperature stays
/// with `default_temperature` (1.0 on both contracts).
pub struct GlmValues {
    pub top_p: f32,
    pub min_p: f32,
    pub effort: ThinkMode,
    /// Sampler ceiling: `None` serves the request's own temperature.  A value
    /// pins it (0.0 = greedy, the only regime the artifact's own temperature
    /// family measured stable).
    pub temp_max: Option<f32>,
    /// Penalty face (core_decode_penalty.c).  GLM-5.3-VQ's own card measures the
    /// VQ conversion as lossy (held-out prose PPL 4.8555 -> 8.0797) and records a
    /// greedy-degeneration case; on this engine the artifact answers and then
    /// runs on without an end token unless the sampling is held down.  A penalty
    /// is the reference's own cure for that (it reads history, so it also acts at
    /// temperature 0), and the engine serves such a request plain -- the device
    /// trial cannot carry a penalty.
    pub freq_penalty: f32,
    pub presence_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    /// Penalty lookback in tokens of the committed history, prompt included;
    /// 0 keeps the face over the generated segment alone.
    /// `GLM53_DEFAULT_PENALTY_WINDOW` carries the evidence.
    pub penalty_window: i32,
    /// Forced-thought budgets, in tokens, one per effort level (the same
    /// three the template knows).  0 is OFF.  A GLM chat turn reserves
    /// nothing post-thinking, so without a bound it never emits the close and
    /// runs to the token cap (the measured runaway); the machinery
    /// (`dsml.rs` sampling_override) forced the close for tool turns only.
    /// A non-zero budget replaces the tool-turn cap for that level, so one
    /// number governs both kinds of turn; 0 keeps the reference's 64/256/512
    /// and leaves a chat turn unbounded.
    pub think_cap_low: i32,
    pub think_cap_high: i32,
    pub think_cap_max: i32,
}

impl GlmValues {
    /// The forced-thought budget for one level; 0 leaves the turn to its own
    /// budget and the model's own end token.
    pub fn think_cap(&self, mode: ThinkMode) -> i32 {
        match mode {
            ThinkMode::Low => self.think_cap_low,
            ThinkMode::High => self.think_cap_high,
            ThinkMode::Max => self.think_cap_max,
            ThinkMode::None => 0,
        }
    }
}

/// generation_config.json: top_p 0.95, temperature 1.0, and nothing else --
/// an absent min_p key means the artifact does not truncate below the
/// nucleus, so the port's 0.05 (ds4.h:54) must not apply here.
const GLM53_DEFAULT_TOP_P: f32 = 0.95;
const GLM53_DEFAULT_MIN_P: f32 = 0.0;
/// The template's own default: anything unlisted, `max` included, is max.
const GLM53_DEFAULT_EFFORT: ThinkMode = ThinkMode::Max;
/// Measured on the Spark (2026-10-17, docs/glm53-vq-sampler-2026-10-17.md):
/// at the artifact's own temperature the model answers and then keeps going
/// (5/5 runs hit the token cap with no end token, at every effort level);
/// greedy stops 5/5 but can lock onto a repeated phrase on a "fill N words"
/// prompt.  DRY (the reference's own cure) removes the literal repeat and is
/// therefore available, but it does NOT stop the runaway, and because a
/// penalty reads the token history it also switches the GLM drafts off -- a
/// ~2x decode cost.  It therefore ships OFF and the deployment turns it on
/// when it wants the loop protection more than the drafting.
const GLM53_DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
/// Dry's own contract (llama.cpp): 1.75^L from the allowed length up.
const GLM53_DEFAULT_DRY_BASE: f32 = 1.75;
const GLM53_DEFAULT_DRY_ALLOWED_LENGTH: i32 = 2;
/// No ceiling: the request's temperature is served as sent (the host rule).
const GLM53_DEFAULT_TEMP_MAX: Option<f32> = None;
/// The forced-thought budgets ship OFF: the deployment opts in per level, so
/// the bound is measured rather than assumed to help.
const GLM53_DEFAULT_THINK_CAP: i32 = 0;
/// The knob window.  A budget this large is inert (the turn's own budget is
/// the binding limit), so the bound only catches a typo.
const GLM53_THINK_CAP_MAX: i32 = 1_000_000;
/// The penalty face's lookback, in tokens of the committed history, prompt
/// included.  0 -- the default -- keeps the face over the generated segment
/// alone (ds4.c family_cont_penalize), the shape every earlier measurement
/// used; the deployment widens it when it wants a loop breaker that can see
/// the repeated ACTION, which lives in the prompt rather than in the turn's
/// own few generated tokens.  The ordering is the artifact runtime's own:
/// llama.cpp feeds every prompt token into the sampler chain
/// (tools/server/server-context.cpp:433) and bounds the lookback with
/// repeat_last_n / dry_penalty_last_n (its server README: default 64, 0
/// disabled).
const GLM53_DEFAULT_PENALTY_WINDOW: i32 = 0;
/// A window this large is inert (the sequence's own length binds first), so
/// the bound only catches a typo.
const GLM53_PENALTY_WINDOW_MAX: i32 = 1_000_000;

/// The forced-thought budget one deployment knob names.  0 is OFF; a bad
/// value is reported and disables the bound rather than picking a number.
fn resolve_think_cap(name: &str, raw: Option<&str>) -> i32 {
    resolve_number(
        name,
        raw,
        GLM53_DEFAULT_THINK_CAP as f32,
        0.0..=GLM53_THINK_CAP_MAX as f32,
    ) as i32
}

/// The deployment's temperature ceiling.  Absent (or empty) is no ceiling;
/// any value in [0, 2] pins every GLM request's temperature to at most it,
/// with 0.0 meaning greedy -- the only regime the artifact measured stable.
fn resolve_temp_max(raw: Option<&str>) -> Option<f32> {
    let raw = raw.filter(|s| !s.is_empty())?;
    Some(resolve_number(
        "DS4_SERVER_GLM_TEMP_MAX",
        Some(raw),
        0.0,
        0.0..=2.0,
    ))
}

/// top_p and min_p are probabilities, so a value outside [0, 1] is a typo.
const UNIT_BOUNDS: std::ops::RangeInclusive<f32> = 0.0..=1.0;

/// Resolve one `DS4_SERVER_DEFAULT_*` probability against the artifact's
/// value.  An unparseable or out-of-range setting is reported once at
/// startup instead of being clamped (0.2 and 2.0 are equally likely to be
/// typos) or silently ignored.
fn resolve_unit(name: &str, raw: Option<&str>, fallback: f32) -> f32 {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else {
        return fallback;
    };

    let Ok(value) = raw.parse::<f32>() else {
        eprintln!("ds4-server: {name}={raw:?} is not a number; using {fallback}");
        return fallback;
    };

    if !value.is_finite() || !UNIT_BOUNDS.contains(&value) {
        eprintln!("ds4-server: {name}={raw:?} is outside [0, 1]; using {fallback}");
        return fallback;
    }

    value
}

/// Resolve one numeric `DS4_SERVER_*` setting against its fallback, with the
/// same rule as resolve_unit: a bad value is reported once, never clamped.
fn resolve_number(
    name: &str,
    raw: Option<&str>,
    fallback: f32,
    bounds: std::ops::RangeInclusive<f32>,
) -> f32 {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else {
        return fallback;
    };

    let Ok(value) = raw.parse::<f32>() else {
        eprintln!("ds4-server: {name}={raw:?} is not a number; using {fallback}");
        return fallback;
    };

    if !value.is_finite() || !bounds.contains(&value) {
        eprintln!(
            "ds4-server: {name}={raw:?} is outside [{}, {}]; using {fallback}",
            bounds.start(),
            bounds.end()
        );
        return fallback;
    }

    value
}

/// The effort level a GLM request that does not name one gets.  The knob
/// names the artifact's own three levels; any other name is a typo and is
/// reported rather than mapped onto a level the template would not read.
fn resolve_effort(raw: Option<&str>) -> ThinkMode {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else {
        return GLM53_DEFAULT_EFFORT;
    };

    match raw.trim() {
        "low" => ThinkMode::Low,
        "high" => ThinkMode::High,
        "max" => ThinkMode::Max,
        _ => {
            eprintln!(
                "ds4-server: DS4_SERVER_DEFAULT_EFFORT={raw:?} is not low, high or max; using max"
            );
            GLM53_DEFAULT_EFFORT
        }
    }
}

/// The GLM scope's values, read from the environment once per process: these
/// are deployment settings fixed at server start, not per-request inputs.
pub fn glm_values() -> &'static GlmValues {
    static RESOLVED: std::sync::OnceLock<GlmValues> = std::sync::OnceLock::new();

    RESOLVED.get_or_init(|| GlmValues {
        top_p: resolve_unit(
            "DS4_SERVER_DEFAULT_TOP_P",
            std::env::var("DS4_SERVER_DEFAULT_TOP_P").ok().as_deref(),
            GLM53_DEFAULT_TOP_P,
        ),
        min_p: resolve_unit(
            "DS4_SERVER_DEFAULT_MIN_P",
            std::env::var("DS4_SERVER_DEFAULT_MIN_P").ok().as_deref(),
            GLM53_DEFAULT_MIN_P,
        ),
        effort: resolve_effort(std::env::var("DS4_SERVER_DEFAULT_EFFORT").ok().as_deref()),
        temp_max: resolve_temp_max(std::env::var("DS4_SERVER_GLM_TEMP_MAX").ok().as_deref()),
        freq_penalty: resolve_number(
            "DS4_SERVER_DEFAULT_FREQ_PENALTY",
            std::env::var("DS4_SERVER_DEFAULT_FREQ_PENALTY")
                .ok()
                .as_deref(),
            0.0,
            -2.0..=2.0,
        ),
        presence_penalty: resolve_number(
            "DS4_SERVER_DEFAULT_PRESENCE_PENALTY",
            std::env::var("DS4_SERVER_DEFAULT_PRESENCE_PENALTY")
                .ok()
                .as_deref(),
            0.0,
            -2.0..=2.0,
        ),
        dry_multiplier: resolve_number(
            "DS4_SERVER_DEFAULT_DRY_MULTIPLIER",
            std::env::var("DS4_SERVER_DEFAULT_DRY_MULTIPLIER")
                .ok()
                .as_deref(),
            GLM53_DEFAULT_DRY_MULTIPLIER,
            0.0..=64.0,
        ),
        dry_base: resolve_number(
            "DS4_SERVER_DEFAULT_DRY_BASE",
            std::env::var("DS4_SERVER_DEFAULT_DRY_BASE").ok().as_deref(),
            GLM53_DEFAULT_DRY_BASE,
            1.0..=64.0,
        ),
        dry_allowed_length: resolve_number(
            "DS4_SERVER_DEFAULT_DRY_ALLOWED_LENGTH",
            std::env::var("DS4_SERVER_DEFAULT_DRY_ALLOWED_LENGTH")
                .ok()
                .as_deref(),
            GLM53_DEFAULT_DRY_ALLOWED_LENGTH as f32,
            0.0..=64.0,
        ) as i32,
        penalty_window: resolve_number(
            "DS4_SERVER_GLM_PENALTY_WINDOW",
            std::env::var("DS4_SERVER_GLM_PENALTY_WINDOW")
                .ok()
                .as_deref(),
            GLM53_DEFAULT_PENALTY_WINDOW as f32,
            0.0..=GLM53_PENALTY_WINDOW_MAX as f32,
        ) as i32,
        think_cap_low: resolve_think_cap(
            "DS4_SERVER_GLM_THINK_CAP_LOW",
            std::env::var("DS4_SERVER_GLM_THINK_CAP_LOW")
                .ok()
                .as_deref(),
        ),
        think_cap_high: resolve_think_cap(
            "DS4_SERVER_GLM_THINK_CAP_HIGH",
            std::env::var("DS4_SERVER_GLM_THINK_CAP_HIGH")
                .ok()
                .as_deref(),
        ),
        think_cap_max: resolve_think_cap(
            "DS4_SERVER_GLM_THINK_CAP_MAX",
            std::env::var("DS4_SERVER_GLM_THINK_CAP_MAX")
                .ok()
                .as_deref(),
        ),
    })
}

/// The startup line for the GLM scope, so the deployment can read back what
/// it actually serves.  The three budgets are low/high/max; 0 is off.
pub fn glm_values_report() -> String {
    let v = glm_values();
    format!(
        "GLM values: temperature={} top_p={} min_p={} effort={:?} temp_max={} \
         freq_penalty={} presence_penalty={} dry={}x{}^L from L={} \
         penalty_window={} think_cap={}/{}/{}",
        default_temperature(),
        v.top_p,
        v.min_p,
        v.effort,
        match v.temp_max {
            Some(max) => format!("{max}"),
            None => "as-requested".to_string(),
        },
        v.freq_penalty,
        v.presence_penalty,
        v.dry_multiplier,
        v.dry_base,
        v.dry_allowed_length,
        v.penalty_window,
        v.think_cap_low,
        v.think_cap_high,
        v.think_cap_max
    )
}

/// The request-omitted effort level for a scope.  The GLM-5.3 scope takes the
/// artifact's own default (max, or the deployment's override); every other
/// family keeps the port's low.
pub fn default_effort(glm: bool) -> ThinkMode {
    if glm {
        glm_values().effort
    } else {
        ThinkMode::Low
    }
}

/// The syntaxes that keep the port's fixed thinking sampler rather than the
/// request's own knobs.  The host applies that sampler only to values the
/// client left out (ds4_server.c:13951-13959); the port's other families do
/// not track which knobs were sent, so they keep the fixed tuple.  GLM-5.3 is
/// the exception both ways: its request-omitted values are the artifact's own
/// (GlmValues) and an explicit client value must win, as in the host.
pub fn fixed_thinking_sampling(syntax: ModelSyntax) -> bool {
    !matches!(syntax, ModelSyntax::IQuestQ1 | ModelSyntax::Glm53)
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
    /// OpenAI frequency / presence penalties (the engine parses both,
    /// server_parse_chat.c:91-104 and server_parse_responses.c:328-334); any
    /// non-zero value takes the V4.1 host penalty route.
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// DRY (llama.cpp's sequence repetition penalty; the engine parses the
    /// three keys, server_parse_chat.c:119-127): dry_set = dry_multiplier
    /// appeared; base / allowed_length fall back to 1.75 / 2 at face build
    /// (server_generate_v41.c:363-365).
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_set: bool,
    /// Penalty lookback in tokens of the committed history, prompt included.
    /// Deployment-scoped like `think_cap` (the GLM scope supplies it, no wire
    /// field does), so a request that leaves it alone reads the generated
    /// segment exactly as before.
    pub penalty_window: i32,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub return_token_ids: bool,
    pub think_mode: ThinkMode,
    pub eos_policy: EosPolicy,
    pub has_tools: bool,
    pub has_tool_results: bool,
    pub tool_choice: ToolChoice,
    pub required_tool_prefix: Vec<i32>,
    pub required_think_end_prefix: Vec<i32>,
    /// Forced-thought budget in tokens; 0 is OFF.  Set from the GLM scope by
    /// the level this request resolved to (parse_request), so a bounded
    /// thought applies to chat turns the tool-turn gate would never reach.
    pub think_cap: i32,
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
    pub audios: Vec<RequestAudio>,
    pub videos: Vec<RequestVideo>,
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
            top_p: if env.glm_values {
                glm_values().top_p
            } else {
                DEFAULT_TOP_P
            },
            min_p: if env.engine_defaults {
                0.0
            } else if env.glm_values {
                glm_values().min_p
            } else {
                DEFAULT_MIN_P
            },
            seed: 0,
            frequency_penalty: if env.glm_values {
                glm_values().freq_penalty
            } else {
                0.0
            },
            presence_penalty: if env.glm_values {
                glm_values().presence_penalty
            } else {
                0.0
            },
            dry_multiplier: if env.glm_values {
                glm_values().dry_multiplier
            } else {
                0.0
            },
            dry_base: if env.glm_values {
                glm_values().dry_base
            } else {
                0.0
            },
            dry_allowed_length: if env.glm_values {
                glm_values().dry_allowed_length
            } else {
                0
            },
            dry_set: env.glm_values,
            penalty_window: if env.glm_values {
                glm_values().penalty_window
            } else {
                0
            },
            stream: false,
            stream_include_usage: false,
            return_token_ids: false,
            think_mode: ThinkMode::Low,
            eos_policy: EosPolicy::Default,
            has_tools: false,
            has_tool_results: false,
            tool_choice: ToolChoice::Auto,
            required_tool_prefix: Vec::new(),
            required_think_end_prefix: Vec::new(),
            think_cap: 0,
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
            audios: Vec::new(),
            videos: Vec::new(),
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
        if !self.audios.is_empty() || !self.videos.is_empty() {
            self.needs |= crate::route::NEED_AUDIO;
        }
        if self.eos_policy != EosPolicy::Default {
            self.needs |= crate::route::NEED_EOS_POLICY;
        }
    }
}

const CHAT_IMAGE_MAX_COUNT: usize = 4;
const CHAT_IMAGE_MAX_BYTES: usize = 10 * 1024 * 1024;
const CHAT_IMAGE_TOTAL_MAX: usize = 20 * 1024 * 1024;

struct ImageInputs {
    items: Vec<RequestImage>,
    bytes: usize,
    count_limit: Option<usize>,
}

impl ImageInputs {
    fn new(env: &ParseEnv) -> Self {
        Self {
            items: Vec::new(),
            bytes: 0,
            // YoungAi server_parse_tools.c:289 grows the image list; ds41
            // checks expanded context and memory. Other families keep four.
            count_limit: (!env.engine_defaults).then_some(CHAT_IMAGE_MAX_COUNT),
        }
    }
}

impl std::ops::Deref for ImageInputs {
    type Target = [RequestImage];

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

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

fn decode_media_base64(src: &str, request_remaining: usize) -> Result<Vec<u8>, String> {
    let src = src.as_bytes();
    if src.is_empty() || src.len() & 3 != 0 {
        return Err("invalid media base64 length".into());
    }
    let padding = usize::from(src[src.len() - 1] == b'=') + usize::from(src[src.len() - 2] == b'=');
    let decoded = src.len() / 4 * 3 - padding;
    if decoded == 0 || decoded > CHAT_IMAGE_MAX_BYTES {
        return Err("media exceeds 10 MiB decoded limit".into());
    }
    if decoded > request_remaining {
        return Err("media exceeds 20 MiB request limit".into());
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
            return Err("invalid media base64 data".into());
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
        return Err("invalid media base64 padding".into());
    }
    Ok(out)
}

fn media_count(images: &[RequestImage], audios: &[RequestAudio], videos: &[RequestVideo]) -> usize {
    images.len() + audios.len() + videos.len()
}

fn media_bytes(images: &[RequestImage], audios: &[RequestAudio], videos: &[RequestVideo]) -> usize {
    images.iter().map(|v| v.data.len()).sum::<usize>()
        + audios.iter().map(|v| v.data.len()).sum::<usize>()
        + videos.iter().map(|v| v.data.len()).sum::<usize>()
}

fn add_image_base64(
    images: &mut ImageInputs,
    audios: &[RequestAudio],
    videos: &[RequestVideo],
    mime_type: &str,
    payload: &str,
) -> Result<usize, String> {
    let mime = match mime_type {
        "image/png" => ImageMime::Png,
        "image/jpeg" => ImageMime::Jpeg,
        _ => return Err(format!("unsupported image media type: {mime_type}")),
    };
    let limit = if audios.is_empty() && videos.is_empty() {
        images.count_limit
    } else {
        Some(CHAT_IMAGE_MAX_COUNT)
    };
    if limit.is_some_and(|n| media_count(images, audios, videos) >= n) {
        return Err("at most 4 media inputs are supported".into());
    }
    let total = images.bytes + media_bytes(&[], audios, videos);
    if total > CHAT_IMAGE_TOTAL_MAX {
        return Err("media exceeds 20 MiB request limit".into());
    }
    let data = decode_media_base64(payload, CHAT_IMAGE_TOTAL_MAX - total)?;
    let magic_matches = match mime {
        ImageMime::Png => data.starts_with(b"\x89PNG\r\n\x1a\n"),
        ImageMime::Jpeg => data.starts_with(&[0xff, 0xd8, 0xff]),
    };
    if !magic_matches {
        return Err("image bytes do not match declared media type".into());
    }
    let index = images.len();
    images.bytes += data.len();
    images.items.push(RequestImage {
        mime,
        data: data.into(),
    });
    Ok(index)
}

fn add_image_data_uri(
    images: &mut ImageInputs,
    audios: &[RequestAudio],
    videos: &[RequestVideo],
    uri: &str,
) -> Result<usize, String> {
    let Some(rest) = uri.strip_prefix("data:") else {
        return Err("image_url must be a base64 data URI".into());
    };
    let Some((mime, payload)) = rest.split_once(";base64,") else {
        return Err("image_url must be a base64 data URI".into());
    };
    if mime.is_empty() {
        return Err("image_url must be a base64 data URI".into());
    }
    add_image_base64(images, audios, videos, mime, payload)
}

fn add_video(
    videos: &mut Vec<RequestVideo>,
    images: &[RequestImage],
    audios: &[RequestAudio],
    payload: &str,
) -> Result<usize, String> {
    if media_count(images, audios, videos) >= CHAT_IMAGE_MAX_COUNT {
        return Err("at most 4 media inputs are supported".into());
    }
    let total = media_bytes(images, audios, videos);
    if total > CHAT_IMAGE_TOTAL_MAX {
        return Err("media exceeds 20 MiB request limit".into());
    }
    let data = decode_media_base64(payload, CHAT_IMAGE_TOTAL_MAX - total)?;
    let ftyp = data.len() >= 12 && data[4..8] == *b"ftyp";
    if !ftyp {
        return Err("video bytes do not match MP4".into());
    }
    let index = videos.len();
    videos.push(RequestVideo { data: data.into() });
    Ok(index)
}

fn add_audio(
    audios: &mut Vec<RequestAudio>,
    images: &[RequestImage],
    videos: &[RequestVideo],
    format: &str,
    payload: &str,
) -> Result<usize, String> {
    if format != "wav" {
        return Err("audio input currently requires WAV format".into());
    }
    if media_count(images, audios, videos) >= CHAT_IMAGE_MAX_COUNT {
        return Err("at most 4 media inputs are supported".into());
    }
    let total = media_bytes(images, audios, videos);
    if total > CHAT_IMAGE_TOTAL_MAX {
        return Err("media exceeds 20 MiB request limit".into());
    }
    let data = decode_media_base64(payload, CHAT_IMAGE_TOTAL_MAX - total)?;
    if !data.starts_with(b"RIFF") || data.get(8..12) != Some(b"WAVE") {
        return Err("audio bytes do not match WAV format".into());
    }
    let index = audios.len();
    audios.push(RequestAudio { data: data.into() });
    Ok(index)
}

fn validate_audio_refs(
    msgs: &[ChatMsg],
    images: &[RequestImage],
    audios: &[RequestAudio],
    videos: &[RequestVideo],
) -> Result<(), String> {
    if ((!audios.is_empty() || !videos.is_empty())
        && media_count(images, audios, videos) > CHAT_IMAGE_MAX_COUNT)
        || media_bytes(images, audios, videos) > CHAT_IMAGE_TOTAL_MAX
    {
        return Err("media exceeds the 4 input or 20 MiB request limit".into());
    }
    let mut seen = vec![false; audios.len()];
    for msg in msgs {
        for part in &msg.parts {
            let ChatPart::Audio(index) = part else {
                continue;
            };
            if msg.role != "user" {
                return Err("audio is allowed only in user messages".into());
            }
            let Some(slot) = seen.get_mut(*index) else {
                return Err("invalid audio reference".into());
            };
            if *slot {
                return Err("duplicate audio reference".into());
            }
            *slot = true;
        }
    }
    if seen.iter().any(|v| !v) {
        return Err("unreferenced audio payload".into());
    }
    Ok(())
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
            // The engine's render treats user, tool and function as the same
            // user-like body (server_dsml_render.c:218-221 role_is_user_like),
            // so a V4.1 tool result may carry pixels -- the MCP screenshot
            // loop.  A head that admits images but cannot place one inside a
            // tool result is refused by name in generation_blocked; a
            // system/assistant image stays a parse error.
            if !matches!(msg.role.as_str(), "user" | "tool" | "function") {
                return Err("images are allowed only in user messages or tool results".into());
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

/// A tool/function message is rendered by its content, wrapped in the
/// `<tool_result>` envelope, so an image block has to leave its placeholder in
/// that text, in client block order.  The engine writes it into the same
/// buffer at parse time (`server_parse_tools.c:289` through
/// `mm_image_source_to_text`; the Chat Completions content sink at
/// `server_buf.c:232`).  A user message gets its placeholder from the parts at
/// render time instead, so the text a non-DSML head renders never carries the
/// token.  A tool_result that already rendered its envelope (the Anthropic
/// path writes both the envelope and the placeholder into the content) is left
/// alone.
fn write_tool_image_placeholders(msg: &mut ChatMsg) {
    if !matches!(msg.role.as_str(), "tool" | "function") {
        return;
    }
    if msg
        .parts
        .iter()
        .any(|part| matches!(part, ChatPart::ToolResult { .. }))
    {
        return;
    }
    if !msg
        .parts
        .iter()
        .any(|part| matches!(part, ChatPart::Image(_)))
    {
        return;
    }
    let mut rendered = String::with_capacity(msg.content.len() + DSML_IMAGE.len());
    for part in &msg.parts {
        match part {
            ChatPart::Text(text) => rendered.push_str(text),
            ChatPart::Image(_) => rendered.push_str(DSML_IMAGE),
            _ => {}
        }
    }
    msg.content = rendered;
}

fn validate_video_refs(msgs: &[ChatMsg], videos: &[RequestVideo]) -> Result<(), String> {
    if videos.is_empty() {
        return Ok(());
    }
    let mut seen = vec![false; videos.len()];
    for msg in msgs {
        for part in &msg.parts {
            let ChatPart::Video(index) = part else {
                continue;
            };
            if msg.role != "user" {
                return Err("video is allowed only in user messages".into());
            }
            let Some(slot) = seen.get_mut(*index) else {
                return Err("invalid video reference".into());
            };
            if *slot {
                return Err("duplicate video reference".into());
            }
            *slot = true;
        }
    }
    if seen.iter().any(|seen| !seen) {
        return Err("unreferenced video payload".into());
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
    images: &mut ImageInputs,
    audios: &mut Vec<RequestAudio>,
    videos: &mut Vec<RequestVideo>,
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
            let mut input_audio = None;
            let mut video_url = None;
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
                } else if key == "input_audio" {
                    input_audio = json_raw_value(p);
                    if input_audio.is_none() {
                        return false;
                    }
                } else if key == "image_url" {
                    image_url = parse_chat_image_url(p);
                    if image_url.is_none() {
                        return false;
                    }
                } else if key == "video_url" {
                    video_url = parse_chat_image_url(p);
                    if video_url.is_none() {
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
                (Some("image_url"), _, Some(url)) => {
                    match add_image_data_uri(images, audios, videos, &url) {
                        Ok(index) => msg.parts.push(ChatPart::Image(index)),
                        Err(error) => {
                            *err = error;
                            return false;
                        }
                    }
                }
                (Some("video_url") | Some("video"), _, _) => {
                    let Some(url) = video_url.as_deref() else {
                        *err = "video_url must be a base64 data URI".into();
                        return false;
                    };
                    let Some(rest) = url.strip_prefix("data:") else {
                        *err = "video_url must be a base64 data URI".into();
                        return false;
                    };
                    let Some((mime, payload)) = rest.split_once(";base64,") else {
                        *err = "video_url must be a base64 data URI".into();
                        return false;
                    };
                    if mime != "video/mp4" {
                        *err = "video input currently requires video/mp4".into();
                        return false;
                    }
                    match add_video(videos, images, audios, payload) {
                        Ok(index) => msg.parts.push(ChatPart::Video(index)),
                        Err(error) => {
                            *err = error;
                            return false;
                        }
                    }
                }
                (Some("input_audio"), _, _) => {
                    let value = input_audio
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
                    let format = value.as_ref().and_then(|v| v["format"].as_str());
                    let data = value.as_ref().and_then(|v| v["data"].as_str());
                    let (Some(format), Some(data)) = (format, data) else {
                        *err = "input_audio requires format and base64 data".into();
                        return false;
                    };
                    match add_audio(audios, images, videos, format, data) {
                        Ok(index) => msg.parts.push(ChatPart::Audio(index)),
                        Err(error) => {
                            *err = error;
                            return false;
                        }
                    }
                }
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
    images: &mut ImageInputs,
    audios: &mut Vec<RequestAudio>,
    videos: &mut Vec<RequestVideo>,
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
            } else if key == "name" {
                msg.name = json_string(p)?;
            } else if key == "content" {
                msg.content.clear();
                msg.parts.clear();
                if !parse_openai_message_content(p, &mut msg, images, audios, videos, err) {
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
        write_tool_image_placeholders(&mut msg);
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

/// Anthropic `tool_result.content`: a bare string, null, or an array of content
/// blocks.  The engine's `json_tool_result_content` (server_parse_tools.c:330):
/// text blocks concatenate exactly like the legacy string path, so rendered
/// bytes stay stable and existing disk-KV prefixes keep matching; image blocks
/// -- browser/screenshot MCP tools -- register through the same path
/// user-message images use, with the image placeholder left where the block
/// sits in the client's order (the engine's text-level render,
/// server_parse_tools.c:290).  Only pixels fail closed: an undeliverable image
/// rejects the request, never silently vanishes; text-bearing blocks of other
/// types keep their text.
///
/// Returns (rendered, plain).  The DSML render writes `rendered`, the text with
/// the placeholders it can place; every other head reads the ToolResult part,
/// which carries `plain`: they have no way to place a V4.1 placeholder, and a
/// leaked DSML token would be worse than the image staying out of their text
/// (their tool results keep rendering exactly as before).
fn json_tool_result_content(
    p: &mut Json<'_>,
    msg: &mut ChatMsg,
    images: &mut ImageInputs,
    err: &mut String,
) -> Option<(String, String)> {
    p.ws();
    if p.peek() == Some(b'"') {
        let text = json_string(p)?;
        return Some((text.clone(), text));
    }
    if p.lit("null") {
        return Some((String::new(), String::new()));
    }
    if p.peek() != Some(b'[') {
        if !json_skip_value(p) {
            return None;
        }
        return Some((String::new(), String::new()));
    }
    p.i += 1;
    let mut rendered = String::new();
    let mut plain = String::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            let text = json_string(p)?;
            rendered.push_str(&text);
            plain.push_str(&text);
        } else if p.peek() == Some(b'{') {
            p.i += 1;
            let mut typ = None;
            let mut text = None;
            let mut source = None;
            p.ws();
            while p.peek().is_some() && p.peek() != Some(b'}') {
                let key = json_string(p)?;
                p.ws();
                if p.bump() != Some(b':') {
                    return None;
                }
                if key == "type" {
                    typ = json_string(p);
                    if typ.is_none() {
                        return None;
                    }
                } else if key == "text" {
                    text = json_string(p);
                    if text.is_none() {
                        return None;
                    }
                } else if key == "source" {
                    source = parse_anthropic_image_source(p);
                    if source.is_none() {
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
            if typ.as_deref() == Some("image") {
                let Some((source_type, media_type, data)) = source else {
                    *err = "Anthropic image source must use base64 data".into();
                    return None;
                };
                if source_type != "base64" || media_type.is_empty() || data.is_empty() {
                    *err = "Anthropic image source must use base64 data".into();
                    return None;
                }
                match add_image_base64(images, &[], &[], &media_type, &data) {
                    Ok(index) => {
                        msg.parts.push(ChatPart::Image(index));
                        rendered.push_str(DSML_IMAGE);
                    }
                    Err(error) => {
                        *err = error;
                        return None;
                    }
                }
            } else if let Some(t) = text {
                rendered.push_str(&t);
                plain.push_str(&t);
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
    Some((rendered, plain))
}

fn parse_anthropic_content_block(
    p: &mut Json<'_>,
    msg: &mut ChatMsg,
    images: &mut ImageInputs,
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
    let mut tool_result_plain = None;
    let mut result_is_error = false;
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
            match json_tool_result_content(p, msg, images, err) {
                Some((rendered, plain)) => {
                    tool_result = Some(rendered);
                    tool_result_plain = Some(plain);
                }
                None => return false,
            }
        } else if key == "is_error" {
            match json_bool(p) {
                Some(value) => result_is_error = value,
                None => return false,
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
            match add_image_base64(images, &[], &[], &media_type, &data) {
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
            let text = tool_result.unwrap_or_default();
            // Keep original bytes before the legacy encoder escapes XML.  The
            // part carries the plain text (no V4.1 placeholder): it is what the
            // non-DSML heads render.
            msg.parts.push(ChatPart::ToolResult {
                id: id.clone().unwrap_or_default(),
                content: tool_result_plain.take().unwrap_or_default(),
            });
            if let Some(ref i) = id {
                add_tool_call_id(msg, i);
            }
            let mut b = msg.content.clone();
            b.push_str("<tool_result>");
            // A failed tool's payload is often empty or short; the flag itself
            // is the signal (the engine's [tool_error] marker, log.md 2026-07-12).
            if result_is_error {
                b.push_str("[tool_error] ");
            }
            append_tool_result_text(&mut b, &text);
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
    images: &mut ImageInputs,
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
    let mut image_offsets = Vec::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            if let Some(s) = json_string(p) {
                append_text_part(msg, s);
            } else {
                return false;
            }
        } else if p.peek() == Some(b'{') {
            let offset = msg.content.len();
            let parts_before = msg.parts.len();
            if !parse_anthropic_content_block(p, msg, images, err) {
                return false;
            }
            if matches!(&msg.parts[parts_before..], [ChatPart::Image(_)]) {
                image_offsets.push(offset);
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
    if p.bump() != Some(b']') {
        return false;
    }
    // Mixed tool messages use flat DSML content. Preserve top-level images
    // there too, in block order (YoungAi server_parse_tools.c:281-290).
    if image_offsets.is_empty()
        || !msg
            .parts
            .iter()
            .any(|p| matches!(p, ChatPart::ToolResult { .. }))
    {
        return true;
    }
    let mut rendered =
        String::with_capacity(msg.content.len() + image_offsets.len() * DSML_IMAGE.len());
    let mut from = 0;
    for offset in image_offsets {
        rendered.push_str(&msg.content[from..offset]);
        rendered.push_str(DSML_IMAGE);
        from = offset;
    }
    rendered.push_str(&msg.content[from..]);
    msg.content = rendered;
    true
}

fn parse_anthropic_messages(
    p: &mut Json<'_>,
    images: &mut ImageInputs,
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
        write_tool_image_placeholders(&mut msg);
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

/// The head of a `type` array, skipping "null": schemars renders `Option<u8>`
/// as `{"type":["integer","null"]}`, and `schema_property_type` reads only the
/// scalar spelling, so the GLM path needs this to learn the declared type.
fn schema_property_type_first(json: &str) -> Option<String> {
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
            return json_type_list_first(&mut p);
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

/// Reads a `type` value written as an array and returns its first member that
/// is not "null" (the first member when every entry is "null"); a scalar type
/// or anything unreadable yields None.
fn json_type_list_first(p: &mut Json) -> Option<String> {
    p.ws();
    if p.peek() != Some(b'[') {
        return None;
    }
    p.i += 1;
    let mut head: Option<String> = None;
    loop {
        p.ws();
        if p.peek() != Some(b'"') {
            return head;
        }
        let member = json_string(p)?;
        if head.is_none() {
            head = Some(member.clone());
        }
        if member != "null" {
            return Some(member);
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
    }
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
                let typ_first = schema_property_type_first(&property).unwrap_or_default();
                order.prop.push(prop);
                order.prop_type.push(typ);
                order.prop_type_first.push(typ_first);
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
    mut images: Option<&mut ImageInputs>,
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
                match add_image_data_uri(images, &[], &[], &image_url) {
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
    images: &mut ImageInputs,
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
        let mut output_parts = Vec::new();
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
                    let (text, parts) = parse_responses_content_array_mm(p, Some(images), err)?;
                    output_parts = parts;
                    Some(text)
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
            write_tool_image_placeholders(&mut msg);
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
                parts: output_parts,
                ..Default::default()
            };
            if let Some(id) = call_id.as_deref().or(item_id.as_deref()) {
                add_tool_call_id(&mut msg, id);
            }
            write_tool_image_placeholders(&mut msg);
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

fn apply_think(
    r: &mut ParsedRequest,
    got: bool,
    mut enabled: bool,
    effort: ThinkMode,
    engine_defaults: bool,
) {
    enabled = apply_think_aliases(r, got, enabled);
    r.think_mode = if engine_defaults {
        engine_think_mode(enabled, effort)
    } else {
        think_mode_from_enabled(enabled, effort)
    };
}

/// The engine's `think_mode_from_enabled` (server_msgs.c:186-189): nothing
/// below max survives above zero -- DS4 exposes only HIGH and MAX, so low /
/// medium / minimal collapse to HIGH ("callers that need *no* reasoning must
/// use none instead", the engine's own comment) and a bare thinking flag
/// with the engine's HIGH default means HIGH.
fn engine_think_mode(enabled: bool, effort: ThinkMode) -> ThinkMode {
    if !enabled || effort == ThinkMode::None {
        ThinkMode::None
    } else if effort == ThinkMode::Max {
        ThinkMode::Max
    } else {
        ThinkMode::High
    }
}

pub fn parse_chat_request(env: &ParseEnv, body: &str) -> Result<ParsedRequest, String> {
    let mut r = ParsedRequest::init(ReqKind::Chat, env);
    let mut err = String::new();
    let mut p = Json::new(body);
    let mut got_messages = false;
    let mut got_thinking = false;
    // The engine's request-omitted defaults (server_parse_chat.c:16-17 and
    // the other parsers' same pair): thinking OFF unless a flag asks, effort
    // HIGH.  Only the engine's own model set takes them (unit H,
    // ParseEnv::engine_defaults); other families keep the port's defaults.
    let mut thinking_enabled = !env.engine_defaults;
    let mut reasoning_effort = if env.engine_defaults {
        ThinkMode::High
    } else {
        env.default_effort
    };
    let mut msgs = Vec::new();
    let mut images = ImageInputs::new(env);
    let mut audios = Vec::new();
    let mut videos = Vec::new();
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
            match parse_messages(&mut p, &mut images, &mut audios, &mut videos, &mut err) {
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
        } else if key == "frequency_penalty" {
            json_number(&mut p)
                .map(|v| r.frequency_penalty = v as f32)
                .is_some()
        } else if key == "presence_penalty" {
            json_number(&mut p)
                .map(|v| r.presence_penalty = v as f32)
                .is_some()
        } else if key == "dry_multiplier" || key == "dry_base" || key == "dry_allowed_length" {
            // DRY, llama.cpp's names (server_parse_chat.c:119-127): the
            // multiplier appearing marks dry_set; base / allowed_length fall
            // back to 1.75 / 2 at face build (server_generate_v41.c:363-365).
            json_number(&mut p)
                .map(|v| {
                    if key == "dry_multiplier" {
                        r.dry_multiplier = v as f32;
                        r.dry_set = true;
                    } else if key == "dry_base" {
                        r.dry_base = v as f32;
                    } else {
                        r.dry_allowed_length = v as i32;
                    }
                })
                .is_some()
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
                    // The engine (server_parse_chat.c:157-165): the OpenAI
                    // effort field alone is the client asking for thinking --
                    // got_thinking set, enabled = effort != none.
                    got_thinking = true;
                    thinking_enabled = e != ThinkMode::None;
                    true
                }
                Ok(None) => {
                    // "null": the effort stays at its default and the key
                    // still counts as an explicit ask (the engine's
                    // parse_reasoning_effort_value returns true unchanged).
                    got_thinking = true;
                    thinking_enabled = reasoning_effort != ThinkMode::None;
                    true
                }
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
    validate_audio_refs(&msgs, &images, &audios, &videos)?;
    validate_video_refs(&msgs, &videos)?;
    r.has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if !prepare_tool_choice(&mut r, &mut tool_schemas, &mut orders, &mut err) {
        return Err(err);
    }
    apply_think(
        &mut r,
        got_thinking,
        thinking_enabled,
        reasoning_effort,
        env.engine_defaults,
    );
    r.messages = msgs;
    r.images = images.items;
    r.audios = audios;
    r.videos = videos;
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
    // The engine's request-omitted defaults (server_parse_chat.c:16-17 and
    // the other parsers' same pair): thinking OFF unless a flag asks, effort
    // HIGH.  Only the engine's own model set takes them (unit H,
    // ParseEnv::engine_defaults); other families keep the port's defaults.
    let mut thinking_enabled = !env.engine_defaults;
    let mut reasoning_effort = if env.engine_defaults {
        ThinkMode::High
    } else {
        env.default_effort
    };

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
        } else if key == "frequency_penalty" || key == "presence_penalty" {
            // The engine's completions wire has both, same meaning as the chat
            // wire (server_parse_completion.c:108-115: "以前这里没解析, 传了被
            // 静默丢掉" -- unparsed meant silently dropped; fixed in a8e878a,
            // the KV-reuse commit whose gate sends frequency_penalty 1e-30 to
            // take the host penalty route).
            let is_freq = key == "frequency_penalty";
            json_number(&mut p)
                .map(|v| {
                    if is_freq {
                        r.frequency_penalty = v as f32;
                    } else {
                        r.presence_penalty = v as f32;
                    }
                })
                .is_some()
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
    apply_think(
        &mut r,
        got_thinking,
        thinking_enabled,
        reasoning_effort,
        env.engine_defaults,
    );
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
    // The engine's request-omitted defaults (server_parse_chat.c:16-17 and
    // the other parsers' same pair): thinking OFF unless a flag asks, effort
    // HIGH.  Only the engine's own model set takes them (unit H,
    // ParseEnv::engine_defaults); other families keep the port's defaults.
    let mut thinking_enabled = !env.engine_defaults;
    let mut reasoning_effort = if env.engine_defaults {
        ThinkMode::High
    } else {
        env.default_effort
    };
    let mut msgs = Vec::new();
    let mut images = ImageInputs::new(env);
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
                        // The engine (server_parse_anthropic_req.c:178-183):
                        // an effort-only ask turns thinking on (high/max) or
                        // off (none); an explicit thinking field still wins.
                        if !got_thinking {
                            thinking_enabled = reasoning_effort != ThinkMode::None;
                        }
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
                    if !got_thinking {
                        thinking_enabled = reasoning_effort != ThinkMode::None;
                    }
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
    apply_think(
        &mut r,
        got_thinking,
        thinking_enabled,
        reasoning_effort,
        env.engine_defaults,
    );
    match anthropic_validate_tool_results(&msgs, &env.live_ids, &mut err) {
        Ok(live) => r.anthropic_requires_live_tool_state = live,
        Err(()) => return Err(err),
    }
    r.live_call_ids = crate::cont::live_tool_result_ids(Api::Anthropic, &msgs);
    r.messages = msgs;
    r.images = images.items;
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
    // The engine's request-omitted defaults (server_parse_chat.c:16-17 and
    // the other parsers' same pair): thinking OFF unless a flag asks, effort
    // HIGH.  Only the engine's own model set takes them (unit H,
    // ParseEnv::engine_defaults); other families keep the port's defaults.
    let mut thinking_enabled = !env.engine_defaults;
    let mut reasoning_effort = if env.engine_defaults {
        ThinkMode::High
    } else {
        env.default_effort
    };
    let mut msgs = Vec::new();
    let mut images = ImageInputs::new(env);
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
        } else if key == "frequency_penalty" {
            json_number(&mut p)
                .map(|v| r.frequency_penalty = v as f32)
                .is_some()
        } else if key == "presence_penalty" {
            json_number(&mut p)
                .map(|v| r.presence_penalty = v as f32)
                .is_some()
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
                        // The engine (server_parse_responses.c:363-370): an
                        // explicit effort is the client opting into thinking
                        // control -- got_thinking set, enabled = effort != none.
                        got_thinking = true;
                        thinking_enabled = reasoning_effort != ThinkMode::None;
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
    apply_think(
        &mut r,
        got_thinking,
        thinking_enabled,
        reasoning_effort,
        env.engine_defaults,
    );
    match responses_validate_tool_outputs(&msgs, r.think_mode, &env.live_ids, &mut err) {
        Ok((t, reason)) => {
            r.responses_requires_live_tool_state = t;
            r.responses_requires_live_reasoning = reason;
        }
        Err(()) => return Err(err),
    }
    r.live_call_ids = crate::cont::live_tool_result_ids(Api::Responses, &msgs);
    r.messages = msgs;
    r.images = images.items;
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
    let r = match surface {
        WireSurface::OpenaiChat => parse_chat_request(env, body),
        WireSurface::OpenaiCompletion => parse_completion_request(env, body),
        WireSurface::Anthropic => parse_anthropic_request(env, body),
        WireSurface::Responses => parse_responses_request(env, body),
    };

    r.map(|mut r| {
        // The deployment's GLM ceiling is the ONE value the scope applies over
        // an explicit client request: the artifact's own temperature measured
        // unstable (docs/glm53-vq-sampler-2026-10-17.md), and a client cannot
        // be expected to know that.  Every other GLM value is the request's or
        // the artifact's default.
        if env.glm_values {
            if let Some(max) = glm_values().temp_max {
                if r.temperature > max {
                    r.temperature = max;
                }
            }
            // The bound is per level, and the level is only known once the
            // body has been parsed.
            r.think_cap = glm_values().think_cap(r.think_mode);
        }
        r
    })
}

#[cfg(test)]
mod glm_value_tests {
    use super::{
        default_effort, resolve_effort, resolve_unit, GLM53_DEFAULT_EFFORT, GLM53_DEFAULT_MIN_P,
        GLM53_DEFAULT_TEMP_MAX,
    };
    use crate::route::ThinkMode;

    // The deployment override is accepted only inside the probability window;
    // everything else is reported and falls back to the artifact's value.
    #[test]
    fn resolves_valid_unit_override() {
        assert_eq!(resolve_unit("T", Some("0.95"), 0.95), 0.95);
        assert_eq!(resolve_unit("T", Some("0"), 1.0), 0.0);
        assert_eq!(resolve_unit("T", Some("1"), 1.0), 1.0);
    }

    #[test]
    fn rejects_bad_unit_override() {
        for raw in ["", "abc", "-0.1", "1.5", "2", "nan", "inf"] {
            assert_eq!(resolve_unit("T", Some(raw), 0.95), 0.95, "{raw}");
        }

        assert_eq!(resolve_unit("T", None, 0.95), 0.95);
    }

    // The template's own default, and only its two named levels, are real for
    // this artifact (chat_template.jinja:2-3).
    #[test]
    fn resolves_only_the_template_levels() {
        assert_eq!(resolve_effort(None), GLM53_DEFAULT_EFFORT);
        assert_eq!(resolve_effort(Some("low")), ThinkMode::Low);
        assert_eq!(resolve_effort(Some("high")), ThinkMode::High);
        assert_eq!(resolve_effort(Some("max")), ThinkMode::Max);
        for raw in ["", "banana", "xhigh", "none", "off"] {
            assert_eq!(resolve_effort(Some(raw)), GLM53_DEFAULT_EFFORT, "{raw}");
        }
    }

    // Only the GLM scope takes the artifact default; the other families keep
    // the port's low.
    #[test]
    fn scope_selects_the_default_level() {
        assert_eq!(default_effort(false), ThinkMode::Low);
        // The GLM scope takes the resolved deployment value (the artifact's
        // max unless DS4_SERVER_DEFAULT_EFFORT says otherwise).
        assert_eq!(default_effort(true), super::glm_values().effort);
        assert_ne!(GLM53_DEFAULT_EFFORT, ThinkMode::Low);
    }

    // The penalty window is the deployment's knob, not a wire field.  The
    // artifact's runtime feeds every prompt token into its sampler chain
    // (llama.cpp tools/server/server-context.cpp:433) and bounds the penalty
    // lookback with repeat_last_n / dry_penalty_last_n (its server README:
    // default 64, 0 disabled).  The GLM scope copies that ordering; 0 -- the
    // default -- leaves the face over the generated segment alone, so every
    // lane stays byte-identical until a deployment asks for the window.
    #[test]
    fn penalty_window_is_deployment_scoped_and_off_by_default() {
        use super::{
            resolve_number, GLM53_DEFAULT_PENALTY_WINDOW, GLM53_PENALTY_WINDOW_MAX,
        };
        assert_eq!(GLM53_DEFAULT_PENALTY_WINDOW, 0);
        let resolve = |raw: Option<&str>| {
            resolve_number(
                "DS4_SERVER_GLM_PENALTY_WINDOW",
                raw,
                GLM53_DEFAULT_PENALTY_WINDOW as f32,
                0.0..=GLM53_PENALTY_WINDOW_MAX as f32,
            ) as i32
        };
        assert_eq!(resolve(None), 0);
        assert_eq!(resolve(Some("1024")), 1024);
        for raw in ["", "abc", "-1", "nan", "2000000"] {
            assert_eq!(resolve(Some(raw)), 0, "{raw}");
        }
    }

    // The window is reachable by the GLM scope alone.  A V4.1 / Qwen process
    // (env.glm_values false) leaves it 0 whatever the deployment exports, so
    // those families keep the exact face and the exact code path they had.
    #[test]
    fn only_the_glm_scope_can_carry_a_penalty_window() {
        use super::{parse_request, ParseEnv};
        use crate::route::{ThinkMode, WireSurface};
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
        let env = |glm_values: bool| ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 16,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
            engine_defaults: false,
            glm_values,
        };
        let glm = env(true);
        let glm_req = parse_request(WireSurface::OpenaiChat, &glm, body).unwrap();
        assert_eq!(
            glm_req.penalty_window,
            super::GLM53_DEFAULT_PENALTY_WINDOW,
            "the GLM scope takes the deployment's window"
        );

        let other_req = parse_request(WireSurface::OpenaiChat, &env(false), body).unwrap();
        assert_eq!(
            other_req.penalty_window, 0,
            "every other family keeps the generated-segment face"
        );
    }

    // The GLM scope's sampler guard: DRY ships on (the artifact's own measured
    // failure is answering and then running on), the floors stay at the
    // artifact's values, and a numeric override is range-checked like top_p.
    #[test]
    fn scope_carries_the_sampler_guard() {
        use super::{resolve_number, GLM53_DEFAULT_DRY_MULTIPLIER};
        let v = super::glm_values();
        assert_eq!(v.dry_multiplier, GLM53_DEFAULT_DRY_MULTIPLIER);
        assert_eq!(v.dry_base, 1.75);
        assert_eq!(v.dry_allowed_length, 2);
        assert_eq!(v.min_p, GLM53_DEFAULT_MIN_P);
        assert_eq!(v.temp_max, GLM53_DEFAULT_TEMP_MAX);

        assert_eq!(resolve_number("T", Some("0.8"), 0.0, -2.0..=2.0), 0.8);
        assert_eq!(resolve_number("T", Some("-0.5"), 0.0, -2.0..=2.0), -0.5);
        for raw in ["", "abc", "3", "nan"] {
            assert_eq!(
                resolve_number("T", Some(raw), 0.0, -2.0..=2.0),
                0.0,
                "{raw}"
            );
        }
    }

    // The forced-thought budget is per level and OFF by default: the unit is
    // measured, not assumed, so an unset deployment serves today's bytes.
    #[test]
    fn think_cap_is_per_level_and_off_by_default() {
        use super::{resolve_think_cap, GlmValues};
        let v = GlmValues {
            top_p: 0.95,
            min_p: 0.0,
            effort: ThinkMode::Max,
            temp_max: None,
            freq_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            penalty_window: 0,
            think_cap_low: 64,
            think_cap_high: 256,
            think_cap_max: 512,
        };
        assert_eq!(v.think_cap(ThinkMode::Low), 64);
        assert_eq!(v.think_cap(ThinkMode::High), 256);
        assert_eq!(v.think_cap(ThinkMode::Max), 512);
        // Nothing to bound when the turn does not think.
        assert_eq!(v.think_cap(ThinkMode::None), 0);

        // The shipped scope has no bound until the deployment sets one.
        assert_eq!(super::glm_values().think_cap(ThinkMode::Low), 0);

        assert_eq!(resolve_think_cap("T", None), 0);
        assert_eq!(resolve_think_cap("T", Some("")), 0);
        assert_eq!(resolve_think_cap("T", Some("64")), 64);
        for raw in ["-1", "abc", "nan", "2000000"] {
            assert_eq!(resolve_think_cap("T", Some(raw)), 0, "{raw}");
        }
    }

    // The deployment ceiling guards every path, including a client that sends
    // its own temperature (which the body parse has just applied).
    #[test]
    fn temp_ceiling_overrides_a_client_request() {
        use super::{parse_chat_request, ParseEnv};
        let env = ParseEnv {
            glm_values: true,
            ..ParseEnv::default()
        };
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"temperature":1.0}"#;
        let parsed = super::parse_request(crate::route::WireSurface::OpenaiChat, &env, body)
            .expect("request parses");
        // The ceiling is read from the environment once per process; this test
        // only asserts the site exists and that an inert scope serves as sent.
        let max = super::glm_values().temp_max;
        match max {
            Some(max) => assert_eq!(parsed.temperature, 1.0f32.min(max)),
            None => assert_eq!(parsed.temperature, 1.0),
        }
        assert_eq!(
            parse_chat_request(&ParseEnv::default(), body)
                .expect("request parses")
                .temperature,
            1.0
        );
    }

    // The deployment ceiling: absent is no ceiling, 0.0 means greedy, and a
    // typo is reported rather than silently disabling the pin.
    #[test]
    fn temp_ceiling_is_opt_in() {
        use super::resolve_temp_max;
        assert_eq!(resolve_temp_max(None), None);
        assert_eq!(resolve_temp_max(Some("")), None);
        assert_eq!(resolve_temp_max(Some("0")), Some(0.0));
        assert_eq!(resolve_temp_max(Some("0.5")), Some(0.5));
        assert_eq!(resolve_temp_max(Some("3")), Some(0.0));
        assert_eq!(resolve_temp_max(Some("banana")), Some(0.0));
    }
}

#[cfg(test)]
mod eos_policy_tests {
    use super::EosPolicy;
    use crate::stream::ChatFormat;
    use crate::tools::SemAccum;

    #[test]
    fn reasoning_policy_tracks_think_close_across_pieces() {
        const EOS: i32 = 248_046;
        let mut acc = SemAccum::init(true, false, true, ChatFormat::Qwen4Exp, b"<think>");

        assert_eq!(
            EosPolicy::Default.excluded(EOS, acc.thinking_inside()),
            None
        );
        assert_eq!(
            EosPolicy::Reasoning.excluded(EOS, acc.thinking_inside()),
            Some(EOS)
        );
        assert_eq!(
            EosPolicy::Global.excluded(EOS, acc.thinking_inside()),
            Some(EOS)
        );

        acc.feed(b"</thi", &[]);
        assert_eq!(
            EosPolicy::Reasoning.excluded(EOS, acc.thinking_inside()),
            Some(EOS)
        );
        acc.feed(b"nk>", &[]);
        assert_eq!(
            EosPolicy::Reasoning.excluded(EOS, acc.thinking_inside()),
            None
        );
        assert_eq!(
            EosPolicy::Global.excluded(EOS, acc.thinking_inside()),
            Some(EOS)
        );
    }
}
