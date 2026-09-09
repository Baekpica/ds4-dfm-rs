//! `route_decide` + `request_compute_needs` from `ds4_server.c` at v0.6.5-dfm.
//! Pure functions. Do not improve the table.
// allow: SIZE_OK — frozen C route_decide / compute_needs table; do not split.

mod env;
pub use env::cont_tools_from_env;

pub const NEED_STREAMING: u32 = 1 << 0;
pub const NEED_PER_ROW_SAMPLING: u32 = 1 << 1;
pub const NEED_THINKING: u32 = 1 << 2;
pub const NEED_STOP_SCAN: u32 = 1 << 3;
pub const NEED_TOOL_SCAN: u32 = 1 << 4;
pub const NEED_TOKEN_IDS: u32 = 1 << 5;
pub const NEED_LIVE_FRONTIER: u32 = 1 << 6;
pub const NEED_CONTINUATION_PUBLISH: u32 = 1 << 7;
pub const NEED_CORRECTIVE_RECOVERY: u32 = 1 << 8;
pub const NEED_DURABLE_RESPONSE: u32 = 1 << 9;
pub const NEED_PREFILL_ONLY: u32 = 1 << 10;
pub const NEED_BANK_FRONTIER: u32 = 1 << 11;
pub const NEED_IMAGE: u32 = 1 << 12;
// Audio currently uses the serial Inkling media path only.
pub const NEED_AUDIO: u32 = 1 << 13;

pub const ROUTE_CONT_MASK: u32 = NEED_STREAMING
    | NEED_PER_ROW_SAMPLING
    | NEED_THINKING
    | NEED_STOP_SCAN
    | NEED_TOOL_SCAN
    | NEED_TOKEN_IDS
    | NEED_IMAGE;
pub const ROUTE_STATIC_MASK: u32 = 0;

pub const LANE_SERIAL: u8 = 0;
pub const LANE_CONTINUOUS: u8 = 1;
pub const LANE_STATIC: u8 = 2;
pub const LANE_NONE: u8 = 0xFF;

pub const REASON_CONT: u8 = 0;
pub const REASON_STATIC_NO_CONT: u8 = 1;
pub const REASON_STATIC_PROMPT_BOUNDS: u8 = 2;
pub const REASON_COALESCE_OFF: u8 = 3;
pub const REASON_SURFACE: u8 = 4;
pub const REASON_NEED_LIVE_FRONTIER: u8 = 5;
pub const REASON_NEED_CONTINUATION_PUBLISH: u8 = 6;
pub const REASON_NEED_CORRECTIVE_RECOVERY: u8 = 7;
pub const REASON_NEED_DURABLE: u8 = 8;
pub const REASON_NEED_PREFILL_ONLY: u8 = 9;
pub const REASON_TOKEN_IDS_PROJECTION: u8 = 10;
pub const REASON_TOOLS_COMPLETION: u8 = 11;
pub const REASON_CONT_UNAVAILABLE: u8 = 12;
pub const REASON_CONT_BANK: u8 = 13;

pub const REASON_NAMES: [&str; 14] = [
    "continuous",
    "static_no_cont",
    "static_prompt_bounds",
    "coalesce_off",
    "surface",
    "need_live_frontier",
    "need_continuation_publish",
    "need_corrective_recovery",
    "need_durable_response",
    "need_prefill_only",
    "token_ids_projection",
    "tools_completion_kind",
    "cont_unavailable",
    "continuous_bank_continuation",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireSurface {
    OpenaiChat = 0,
    OpenaiCompletion = 1,
    Anthropic = 2,
    Responses = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Openai = 0,
    Anthropic = 1,
    Responses = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqKind {
    Chat = 0,
    Completion = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteEnv {
    pub coalesce: bool,
    pub have_cont: bool,
    pub cont_anthropic: bool,
    pub cont_responses: bool,
    pub cont_tools_anthropic: bool,
    pub cont_tools_responses: bool,
    pub seq_cap: i32,
    pub prompt_len: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteDecision {
    pub lane: u8,
    pub reason: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct NeedInput {
    pub api: Api,
    pub kind: ReqKind,
    pub stream: bool,
    pub temperature: f32,
    pub think: bool,
    pub stop_count: u32,
    pub has_tools: bool,
    pub return_token_ids: bool,
    pub responses_requires_live_tool_state: bool,
    pub responses_requires_live_reasoning: bool,
    pub anthropic_requires_live_tool_state: bool,
    pub live_state_bank_owned: bool,
    pub max_tokens_set: bool,
    pub max_tokens: i32,
    pub has_images: bool,
}

pub fn wire_surface_for(api: Api, kind: ReqKind) -> WireSurface {
    match api {
        Api::Anthropic => WireSurface::Anthropic,
        Api::Responses => WireSurface::Responses,
        Api::Openai => match kind {
            ReqKind::Completion => WireSurface::OpenaiCompletion,
            ReqKind::Chat => WireSurface::OpenaiChat,
        },
    }
}

pub fn compute_needs(r: &NeedInput) -> u32 {
    let mut n = 0u32;
    if r.stream {
        n |= NEED_STREAMING;
    }
    if r.temperature > 0.0 {
        n |= NEED_PER_ROW_SAMPLING;
    }
    if r.think {
        n |= NEED_THINKING;
    }
    if r.stop_count > 0 {
        n |= NEED_STOP_SCAN;
    }
    if r.has_tools {
        n |= NEED_TOOL_SCAN;
    }
    if r.return_token_ids {
        n |= NEED_TOKEN_IDS;
    }
    if r.has_images {
        n |= NEED_IMAGE;
    }
    if r.responses_requires_live_tool_state
        || r.responses_requires_live_reasoning
        || r.anthropic_requires_live_tool_state
    {
        let bank = r.live_state_bank_owned && !r.responses_requires_live_reasoning;
        n |= if bank {
            NEED_BANK_FRONTIER
        } else {
            NEED_LIVE_FRONTIER
        };
    }
    if r.has_tools && r.api != Api::Openai {
        n |= NEED_CONTINUATION_PUBLISH;
        if !r.stream {
            n |= NEED_CORRECTIVE_RECOVERY;
        }
    }
    if r.api == Api::Anthropic && r.max_tokens_set && r.max_tokens <= 0 {
        n |= NEED_PREFILL_ONLY;
    }
    n
}

pub fn route_decide(needs: u32, surf: WireSurface, env: &RouteEnv) -> RouteDecision {
    let mut d = RouteDecision {
        lane: LANE_SERIAL,
        reason: 0,
    };
    if needs & NEED_DURABLE_RESPONSE != 0 {
        d.lane = LANE_NONE;
        d.reason = REASON_NEED_DURABLE;
        return d;
    }
    let tools_promoted =
        (env.cont_anthropic && env.cont_tools_anthropic && surf == WireSurface::Anthropic)
            || (env.cont_responses && env.cont_tools_responses && surf == WireSurface::Responses);
    if needs & NEED_LIVE_FRONTIER != 0 {
        d.reason = REASON_NEED_LIVE_FRONTIER;
        return d;
    }
    if (needs & NEED_BANK_FRONTIER) != 0 && !tools_promoted {
        d.reason = REASON_NEED_LIVE_FRONTIER;
        return d;
    }
    if needs & NEED_CONTINUATION_PUBLISH != 0 && !((needs & NEED_STREAMING) != 0 && tools_promoted)
    {
        d.reason = REASON_NEED_CONTINUATION_PUBLISH;
        return d;
    }
    if needs & NEED_CORRECTIVE_RECOVERY != 0 {
        d.reason = REASON_NEED_CORRECTIVE_RECOVERY;
        return d;
    }
    if needs & NEED_PREFILL_ONLY != 0 {
        d.reason = REASON_NEED_PREFILL_ONLY;
        return d;
    }
    if (needs & NEED_TOKEN_IDS) != 0
        && !((needs & NEED_STREAMING) != 0 && surf == WireSurface::OpenaiChat)
    {
        d.reason = REASON_TOKEN_IDS_PROJECTION;
        return d;
    }
    if (needs & NEED_TOOL_SCAN) != 0 && surf == WireSurface::OpenaiCompletion {
        d.reason = REASON_TOOLS_COMPLETION;
        return d;
    }
    let mut cont_mask = ROUTE_CONT_MASK;
    if tools_promoted {
        if needs & NEED_STREAMING != 0 {
            cont_mask |= NEED_CONTINUATION_PUBLISH;
        }
        cont_mask |= NEED_BANK_FRONTIER;
    }
    if needs & !cont_mask != 0 {
        d.reason = REASON_CONT_UNAVAILABLE;
        return d;
    }
    let cont_promoted = (env.cont_anthropic && surf == WireSurface::Anthropic)
        || (env.cont_responses && surf == WireSurface::Responses);
    if surf != WireSurface::OpenaiChat && surf != WireSurface::OpenaiCompletion && !cont_promoted {
        d.reason = REASON_SURFACE;
        return d;
    }
    if !env.coalesce {
        d.reason = REASON_COALESCE_OFF;
        return d;
    }
    if env.have_cont && env.prompt_len > 0 && env.prompt_len <= env.seq_cap {
        d.lane = LANE_CONTINUOUS;
        d.reason = if needs & NEED_BANK_FRONTIER != 0 {
            REASON_CONT_BANK
        } else {
            REASON_CONT
        };
        return d;
    }
    if needs & NEED_IMAGE != 0 {
        d.lane = LANE_NONE;
        d.reason = REASON_CONT_UNAVAILABLE;
        return d;
    }
    if (needs & !ROUTE_STATIC_MASK) == 0
        && (surf == WireSurface::OpenaiChat
            || surf == WireSurface::OpenaiCompletion
            || cont_promoted)
    {
        d.lane = LANE_STATIC;
        d.reason = if env.have_cont {
            REASON_STATIC_PROMPT_BOUNDS
        } else {
            REASON_STATIC_NO_CONT
        };
        return d;
    }
    d.reason = REASON_CONT_UNAVAILABLE;
    d
}

/// Inc 0b three-state budget. Parsers preload the server default into
/// `max_tokens` when the client omitted the field.
pub fn decode_budget(max_tokens_set: bool, max_tokens: i32, server_default: i32) -> i32 {
    if max_tokens_set && max_tokens <= 0 {
        return 0;
    }
    if max_tokens > 0 {
        max_tokens
    } else {
        server_default
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    None = 0,
    Low = 1,
    High = 2,
    Max = 3,
}

pub fn think_mode_enabled(mode: ThinkMode) -> bool {
    mode != ThinkMode::None
}

pub fn think_mode_from_enabled(enabled: bool, effort: ThinkMode) -> ThinkMode {
    if !enabled || effort == ThinkMode::None {
        ThinkMode::None
    } else {
        effort
    }
}

/// Exact names from `parse_reasoning_effort_name`. Aliases round down.
pub fn parse_reasoning_effort_name(s: &str) -> Option<ThinkMode> {
    match s {
        "max" => Some(ThinkMode::Max),
        "high" | "xhigh" => Some(ThinkMode::High),
        "low" | "medium" | "minimal" => Some(ThinkMode::Low),
        "none" | "off" => Some(ThinkMode::None),
        _ => None,
    }
}
